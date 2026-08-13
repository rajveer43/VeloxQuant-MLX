"""Parity tests for the fused RaBitQ encode kernel.

The primary reference recomputes the identical float32 operation tree in
numpy (diagonal flip, sequential Sylvester-order WHT butterfly, sign
pack, L1/D) so packed bits must match exactly. A second test
cross-checks against the production rotation path
(mx.fast.hadamard_transform, as used by RaBitQQuantizer) with a tiny
bit-flip allowance for values that land within float rounding of zero.
A third feeds the encoder's outputs straight into rabitq_fused_attend
to prove the two kernels compose end-to-end.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import rabitq_encode, rabitq_fused_attend

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Reference implementation (numpy float32, same op tree as the kernel)
# ---------------------------------------------------------------------------


def _wht_butterfly(x: np.ndarray) -> np.ndarray:
    """Sequential Sylvester-order Walsh-Hadamard transform (unscaled)."""
    y = x.copy()
    d = y.shape[-1]
    stride = 1
    while stride < d:
        for i in range(0, d, stride * 2):
            for j in range(stride):
                a = y[..., i + j].copy()
                b = y[..., i + j + stride].copy()
                y[..., i + j] = a + b
                y[..., i + j + stride] = a - b
        stride <<= 1
    return y


def _reference_encode(keys_fp16: np.ndarray, diag: np.ndarray):
    D = keys_fp16.shape[-1]
    y = _wht_butterfly(keys_fp16.astype(np.float32) * diag[None, :])
    y = y / np.sqrt(np.float32(D))
    bits = np.packbits((y >= 0).astype(np.uint8), axis=1, bitorder="little")[:, : D // 8]
    mag = np.abs(y).sum(axis=1) / D
    return bits, mag.astype(np.float32)


def _make_inputs(N, D, seed=0):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((N, D)).astype(np.float16)
    diag = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
    return keys, diag


# ---------------------------------------------------------------------------
# Parity vs identical-op-tree reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [8, 32, 64, 128, 256])
@pytest.mark.parametrize("N", [1, 33, 512])
def test_rabitq_encode_parity(D, N):
    keys, diag = _make_inputs(N, D, seed=D + N)
    ref_bits, ref_mag = _reference_encode(keys, diag)

    k_bits, k_mag = rabitq_encode(mx.array(keys), mx.array(diag))
    mx.eval(k_bits, k_mag)

    assert k_bits.shape == (N, D // 8)
    assert k_bits.dtype == mx.uint8
    assert k_mag.shape == (N,)
    assert k_mag.dtype == mx.float32

    np.testing.assert_array_equal(np.array(k_bits), ref_bits)
    np.testing.assert_allclose(np.array(k_mag), ref_mag, rtol=1e-4, atol=1e-6)


def test_rabitq_encode_matches_production_rotation():
    """Cross-check vs mx.hadamard_transform (RaBitQQuantizer's path).

    Sign bits may legitimately differ for rotated values within float
    rounding of zero, so allow a <= 0.01% bit-mismatch budget instead of
    exact equality.
    """
    N, D = 256, 128
    keys, diag = _make_inputs(N, D, seed=42)

    rotated = mx.hadamard_transform(
        mx.array(keys.astype(np.float32) * diag[None, :]),
        scale=1.0 / float(D) ** 0.5,
    )
    mx.eval(rotated)
    rot_np = np.array(rotated, dtype=np.float32)
    ref_bits = np.packbits((rot_np >= 0).astype(np.uint8), axis=1, bitorder="little")
    ref_mag = np.abs(rot_np).sum(axis=1) / D

    k_bits, k_mag = rabitq_encode(mx.array(keys), mx.array(diag))
    mx.eval(k_bits, k_mag)

    got = np.unpackbits(np.array(k_bits), axis=1, bitorder="little")
    want = np.unpackbits(ref_bits[:, : D // 8], axis=1, bitorder="little")
    mismatch = (got != want).mean()
    assert mismatch <= 1e-4, f"bit mismatch fraction {mismatch:.2e} exceeds 1e-4"
    np.testing.assert_allclose(np.array(k_mag), ref_mag, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: encode output feeds the attend kernel
# ---------------------------------------------------------------------------


def test_rabitq_encode_feeds_fused_attend():
    B, H, S_q, S_kv, D = 1, 2, 1, 64, 128
    rng = np.random.default_rng(3)
    keys, diag = _make_inputs(H * S_kv, D, seed=3)

    k_bits_flat, k_mag_flat = rabitq_encode(mx.array(keys), mx.array(diag))
    mx.eval(k_bits_flat, k_mag_flat)

    k_bits = np.array(k_bits_flat).reshape(B, H, S_kv, D // 8)
    k_mag = np.array(k_mag_flat).reshape(B, H, S_kv)
    k_const = np.zeros((B, H, S_kv), dtype=np.float32)

    q = rng.standard_normal((B, H, S_q, D)).astype(np.float16)
    q_scale = (rng.uniform(0.05, 0.15, (B, H, S_q)) / np.sqrt(D)).astype(np.float32)
    v_idx = rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8)
    v_cents = np.sort(rng.standard_normal(16)).astype(np.float32)

    out = rabitq_fused_attend(
        mx.array(q),
        mx.array(q_scale),
        mx.array(k_bits),
        mx.array(k_mag),
        mx.array(k_const),
        mx.array(v_idx),
        mx.array(v_cents),
    )
    mx.eval(out)

    # Reference attend computed from the encoder's own outputs.
    D_f = float(D)
    q_bits = q >= 0
    k_unpacked = np.unpackbits(k_bits, axis=-1, count=D, bitorder="little").astype(bool)
    ham = (q_bits[:, :, :, None, :] ^ k_unpacked[:, :, None, :, :]).sum(-1)
    scores = (D_f - 2.0 * ham) * q_scale[..., None] * k_mag[:, :, None, :]
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(-1, keepdims=True)
    expected = np.einsum("bhqs,bhsd->bhqd", w, v_cents[v_idx])

    np.testing.assert_allclose(np.array(out, dtype=np.float32), expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rabitq_encode_rejects_bad_shapes():
    keys, diag = _make_inputs(4, 64)
    with pytest.raises(ValueError, match="must be 2D"):
        rabitq_encode(mx.array(keys[None]), mx.array(diag))
    with pytest.raises(ValueError, match="power of two"):
        rabitq_encode(mx.array(np.zeros((4, 24), np.float16)), mx.array(np.ones(24, np.float32)))
    with pytest.raises(ValueError, match="diag"):
        rabitq_encode(mx.array(keys), mx.array(diag[:32]))

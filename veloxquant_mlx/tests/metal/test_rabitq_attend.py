"""Parity tests for the fused RaBitQ asymmetric decode + attend kernel.

The reference recomputes the exact same math in numpy float32:
unpack the key sign bits, count Hamming distance against the binarized
query, form the affine scores, softmax, and matmul against dequantized
codebook values. The kernel's fp16 output must match within tolerance.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import rabitq_fused_attend

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Reference implementation (numpy float32)
# ---------------------------------------------------------------------------


def _reference_attend(
    q: np.ndarray,  # [B, H, S_q, D] float16
    q_scale: np.ndarray,  # [B, H, S_q] float32
    k_bits: np.ndarray,  # [B, H, S_kv, D//8] uint8
    k_mag: np.ndarray,  # [B, H, S_kv] float32
    k_const: np.ndarray,  # [B, H, S_kv] float32
    v_idx: np.ndarray,  # [B, H, S_kv, D] uint8
    v_cents: np.ndarray,  # [n_cents] float32
) -> np.ndarray:
    D = q.shape[-1]
    q_bits = q >= 0  # [B,H,Sq,D]
    k_unpacked = np.unpackbits(k_bits, axis=-1, count=D, bitorder="little").astype(
        bool
    )  # [B,H,Skv,D]
    ham = (
        (q_bits[:, :, :, None, :] ^ k_unpacked[:, :, None, :, :]).sum(axis=-1).astype(np.float32)
    )  # [B,H,Sq,Skv]

    scores = (float(D) - 2.0 * ham) * q_scale[:, :, :, None] * k_mag[:, :, None, :] + k_const[
        :, :, None, :
    ]
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)  # [B,H,Sq,Skv]

    v_hat = v_cents[v_idx]  # [B,H,Skv,D]
    return np.einsum("bhqs,bhsd->bhqd", w, v_hat).astype(np.float32)


def _make_inputs(B, H, S_q, S_kv, D, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, S_q, D)).astype(np.float16)
    # Scales sized so scores stay in a well-conditioned softmax range:
    # |D - 2*ham| <= D, q_scale * k_mag ~ 1e-2 -> |score| <~ 2.6 at D=256.
    q_scale = (rng.uniform(0.05, 0.15, (B, H, S_q)) / np.sqrt(D)).astype(np.float32)
    k_bits = rng.integers(0, 256, (B, H, S_kv, D // 8), dtype=np.uint8)
    k_mag = rng.uniform(0.5, 1.5, (B, H, S_kv)).astype(np.float32)
    k_const = rng.uniform(-0.2, 0.2, (B, H, S_kv)).astype(np.float32)
    v_idx = rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8)
    v_cents = np.sort(rng.standard_normal(16)).astype(np.float32)
    return q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("S_kv", [1, 33, 256])
@pytest.mark.parametrize("S_q", [1, 8])
@pytest.mark.parametrize("BH", [(1, 1), (2, 2)])
def test_rabitq_attend_parity(D, S_kv, S_q, BH):
    B, H = BH
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(
        B, H, S_q, S_kv, D, seed=D + S_kv + S_q + B * H
    )

    expected = _reference_attend(q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents)

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
    out_np = np.array(out, dtype=np.float32)

    assert out.shape == (B, H, S_q, D)
    assert out.dtype == mx.float16
    np.testing.assert_allclose(out_np, expected, atol=1e-2, rtol=1e-2)


def test_rabitq_attend_d256_boundary():
    """D=256 exercises the full my_out[8] accumulator and N_BYTES == TG."""
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(1, 2, 4, 64, 256, seed=99)
    expected = _reference_attend(q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents)
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
    np.testing.assert_allclose(np.array(out, dtype=np.float32), expected, atol=1e-2, rtol=1e-2)


def test_rabitq_attend_single_slot_returns_that_value():
    """With S_kv=1 the softmax weight is exactly 1 -> output == dequantized v."""
    B, H, S_q, D = 1, 1, 3, 64
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(B, H, S_q, 1, D, seed=7)
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
    v_hat = v_cents[v_idx][:, :, 0, :]  # [B,H,D]
    expected = np.broadcast_to(v_hat[:, :, None, :], (B, H, S_q, D))
    np.testing.assert_allclose(np.array(out, dtype=np.float32), expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rabitq_attend_rejects_bad_shapes():
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(1, 1, 1, 4, 64)
    with pytest.raises(ValueError, match="must be 4D"):
        rabitq_fused_attend(
            mx.array(q[0]),
            mx.array(q_scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            mx.array(v_idx),
            mx.array(v_cents),
        )
    with pytest.raises(ValueError, match="k_mag"):
        rabitq_fused_attend(
            mx.array(q),
            mx.array(q_scale),
            mx.array(k_bits),
            mx.array(k_mag[:, :, :2]),
            mx.array(k_const),
            mx.array(v_idx),
            mx.array(v_cents),
        )
    with pytest.raises(ValueError, match="divisible by 8"):
        rabitq_fused_attend(
            mx.array(q[..., :60]),
            mx.array(q_scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            mx.array(v_idx),
            mx.array(v_cents),
        )

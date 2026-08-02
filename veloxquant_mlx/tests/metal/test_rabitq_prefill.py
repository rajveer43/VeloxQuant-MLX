"""Parity tests for the simdgroup_matrix tiled prefill attention kernel.

Reference recomputes the exact score model in numpy float32: keys
decoded to +-1 * magnitude from packed bits, exact-dot scores with a
scalar softmax scale plus per-key bias, softmax, then matmul against
nibble-decoded codebook values. Tolerances cover the kernel's fp16 tile
arithmetic (float running accumulators, half 8x8 MAC fragments).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import rabitq_prefill_attend, rabitq_pack_values

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Reference implementation (numpy float32)
# ---------------------------------------------------------------------------


def _reference_prefill(q, scale, k_bits, k_mag, k_const, v_idx_unpacked, v_cents):
    D = q.shape[-1]
    signs = (
        np.unpackbits(k_bits, axis=-1, count=D, bitorder="little").astype(np.float32) * 2.0 - 1.0
    )  # [B,H,Skv,D] +-1
    k_hat = signs * k_mag[..., None]  # [B,H,Skv,D]
    scores = (
        np.einsum("bhqd,bhsd->bhqs", q.astype(np.float32), k_hat) * scale + k_const[:, :, None, :]
    )
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    v_hat = v_cents[v_idx_unpacked]  # [B,H,Skv,D]
    return np.einsum("bhqs,bhsd->bhqd", w, v_hat).astype(np.float32)


def _make_inputs(B, H, S_q, S_kv, D, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, S_q, D)).astype(np.float16)
    scale = np.array([1.0 / np.sqrt(D)], dtype=np.float32)
    k_bits = rng.integers(0, 256, (B, H, S_kv, D // 8), dtype=np.uint8)
    k_mag = rng.uniform(0.05, 0.15, (B, H, S_kv)).astype(np.float32)
    k_const = rng.uniform(-0.2, 0.2, (B, H, S_kv)).astype(np.float32)
    v_idx = rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8)
    v_cents = np.sort(rng.standard_normal(16)).astype(np.float32)
    return q, scale, k_bits, k_mag, k_const, v_idx, v_cents


def _run_kernel(q, scale, k_bits, k_mag, k_const, v_idx_unpacked, v_cents):
    packed = rabitq_pack_values(mx.array(v_idx_unpacked))
    out = rabitq_prefill_attend(
        mx.array(q),
        mx.array(scale),
        mx.array(k_bits),
        mx.array(k_mag),
        mx.array(k_const),
        packed,
        mx.array(v_cents),
    )
    mx.eval(out)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("S_q", [1, 8, 33, 256])
@pytest.mark.parametrize("S_kv", [8, 33, 2048])
@pytest.mark.parametrize("BH", [(1, 1), (2, 2)])
def test_prefill_attend_parity(D, S_q, S_kv, BH):
    B, H = BH
    args = _make_inputs(B, H, S_q, S_kv, D, seed=D + S_q + S_kv + B * H)
    expected = _reference_prefill(*args)
    got = _run_kernel(*args)
    assert got.shape == (B, H, S_q, D)
    np.testing.assert_allclose(got, expected, atol=2e-2, rtol=2e-2)


def test_prefill_matches_decode_kernel_semantics():
    """S_q=1 must behave like single-query attention over the same cache
    (sanity: the tiled path agrees with the reference at decode shape)."""
    args = _make_inputs(1, 2, 1, 500, 128, seed=77)
    expected = _reference_prefill(*args)
    got = _run_kernel(*args)
    np.testing.assert_allclose(got, expected, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_prefill_rejects_bad_inputs():
    q, scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(1, 1, 8, 16, 64)
    packed = rabitq_pack_values(mx.array(v_idx))
    with pytest.raises(ValueError, match="128 limit"):
        rabitq_prefill_attend(
            mx.array(np.zeros((1, 1, 8, 256), np.float16)),
            mx.array(scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            packed,
            mx.array(v_cents),
        )
    with pytest.raises(ValueError, match="nibble-packed"):
        rabitq_prefill_attend(
            mx.array(q),
            mx.array(scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            mx.array(v_idx),
            mx.array(v_cents),
        )

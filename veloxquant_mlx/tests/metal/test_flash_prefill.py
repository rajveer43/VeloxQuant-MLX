"""Parity tests for the plain-fp16 causal flash prefill kernel.

Reference is standard causal self-attention in numpy float32: exact
Q@K^T dot scores, causal mask aligned to the tail of the KV cache
(``q_abs = (S_kv - S_q) + q_pos``, same convention as
``rabitq_prefill_attend(causal=True)`` / ``fused_sdpa.metal``), softmax,
then matmul against V. Tolerances cover the kernel's fp16 tile
arithmetic (float running accumulators, half 8x8 MAC fragments).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import flash_prefill_attend

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Reference implementation (numpy float32)
# ---------------------------------------------------------------------------


def _reference_flash(q, k, v, scale):
    S_q, S_kv = q.shape[2], k.shape[2]
    scores = (
        np.einsum("bhqd,bhsd->bhqs", q.astype(np.float32), k.astype(np.float32)) * scale
    )
    q_abs = (S_kv - S_q) + np.arange(S_q)
    kv_pos = np.arange(S_kv)
    mask = kv_pos[None, :] > q_abs[:, None]  # [S_q, S_kv]
    scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("bhqs,bhsd->bhqd", w, v.astype(np.float32)).astype(np.float32)


def _make_inputs(B, H, S_q, S_kv, D, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, S_q, D)).astype(np.float16)
    k = rng.standard_normal((B, H, S_kv, D)).astype(np.float16)
    v = rng.standard_normal((B, H, S_kv, D)).astype(np.float16)
    scale = np.array([1.0 / np.sqrt(D)], dtype=np.float32)
    return q, k, v, scale


def _run_kernel(q, k, v, scale):
    out = flash_prefill_attend(mx.array(q), mx.array(k), mx.array(v), mx.array(scale))
    mx.eval(out)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("S_q", [1, 8, 33, 256])
@pytest.mark.parametrize("S_kv", [8, 33, 2048])
@pytest.mark.parametrize("BH", [(1, 1), (2, 2)])
def test_flash_prefill_attend_parity(D, S_q, S_kv, BH):
    if S_q > S_kv:
        pytest.skip("causal alignment requires S_q <= S_kv")
    B, H = BH
    args = _make_inputs(B, H, S_q, S_kv, D, seed=D + S_q + S_kv + B * H)
    expected = _reference_flash(*args)
    got = _run_kernel(*args)
    assert got.shape == (B, H, S_q, D)
    np.testing.assert_allclose(got, expected, atol=3e-2, rtol=3e-2)


def test_flash_prefill_from_scratch_self_attention():
    """S_q == S_kv: the actual issue #277 target — a fresh conversation
    with no pre-existing cache prefix."""
    args = _make_inputs(1, 4, 64, 64, 128, seed=99)
    expected = _reference_flash(*args)
    got = _run_kernel(*args)
    np.testing.assert_allclose(got, expected, atol=3e-2, rtol=3e-2)


def test_flash_prefill_cache_continuation():
    """S_kv > S_q: queries are new tokens appended after an existing
    (uncompressed) prefix — the KV-cache-continuation shape."""
    args = _make_inputs(1, 2, 16, 500, 128, seed=13)
    expected = _reference_flash(*args)
    got = _run_kernel(*args)
    np.testing.assert_allclose(got, expected, atol=3e-2, rtol=3e-2)


def test_flash_prefill_causal_masks_future():
    """Sanity: changing a future (masked-out) key must not change the
    output — proof the causal block-skip isn't accidentally attending
    to slots past q_abs."""
    B, H, S, D = 1, 2, 40, 64
    q, k, v, scale = _make_inputs(B, H, S, S, D, seed=21)
    out1 = _run_kernel(q, k, v, scale)

    k2, v2 = k.copy(), v.copy()
    k2[:, :, -1, :] = 1000.0  # perturb the last (future-for-every-row-but-the-last) key
    v2[:, :, -1, :] = 1000.0
    out2 = _run_kernel(q, k2, v2, scale)

    # Every row except the very last one must be unaffected.
    np.testing.assert_allclose(out1[:, :, :-1, :], out2[:, :, :-1, :], atol=1e-3)
    assert not np.allclose(out1[:, :, -1, :], out2[:, :, -1, :])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_flash_prefill_rejects_bad_inputs():
    q, k, v, scale = _make_inputs(1, 1, 8, 16, 64)
    with pytest.raises(ValueError, match="128 limit"):
        flash_prefill_attend(
            mx.array(np.zeros((1, 1, 8, 256), np.float16)),
            mx.array(k),
            mx.array(v),
            mx.array(scale),
        )
    with pytest.raises(ValueError, match="v must be"):
        flash_prefill_attend(
            mx.array(q),
            mx.array(k),
            mx.array(np.zeros((1, 1, 1, 64), np.float16)),
            mx.array(scale),
        )

"""Parity tests for the experimental row-owned streaming prefill kernel.

Reference is standard causal self-attention in numpy float32: exact
Q@K^T dot scores, causal mask aligned to the tail of the KV cache
(``q_abs = (S_kv - S_q) + q_pos``, same convention as
``flash_prefill_attend`` / ``rabitq_prefill_attend(causal=True)`` /
``fused_sdpa.metal``), softmax, then matmul against V.

Tolerance: ``atol=3e-2, rtol=3e-2`` — the same tolerance used by
``test_flash_prefill.py``. Both kernels store Q/K/V as fp16 and widen to
float for arithmetic (this kernel widens per-scalar in registers rather
than through half 8x8 MAC fragments, so empirically it is *more* accurate
than the tiled kernel — observed max abs error in exploratory runs was
~1e-3, an order of magnitude under this bound — but the bound is kept at
the existing kernel's 3e-2 for an apples-to-apples comparison and to
leave headroom for fp16 storage-precision effects at longer sequences /
larger D, rather than fitting the tolerance to a single measurement).

Every implementation variant (``streaming``, ``streaming_block2``,
``streaming_block4``, ``streaming_block8``, ``streaming_multirow``) is
checked against the same reference and shape matrix.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import streaming_prefill_attend

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)

_IMPLEMENTATIONS = [
    "streaming",
    "streaming_block2",
    "streaming_block4",
    "streaming_block8",
    "streaming_multirow",
]

_ATOL, _RTOL = 3e-2, 3e-2


# ---------------------------------------------------------------------------
# Reference implementation (numpy float32) — identical convention to
# test_flash_prefill.py's _reference_flash.
# ---------------------------------------------------------------------------


def _reference_flash(q, k, v, scale):
    S_q, S_kv = q.shape[2], k.shape[2]
    scores = np.einsum("bhqd,bhsd->bhqs", q.astype(np.float32), k.astype(np.float32)) * scale
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


def _run_kernel(q, k, v, scale, implementation):
    out = streaming_prefill_attend(
        mx.array(q), mx.array(k), mx.array(v), mx.array(scale), implementation=implementation
    )
    mx.eval(out)
    return np.array(out, dtype=np.float32)


def _assert_no_nan_inf(got: np.ndarray, name: str) -> None:
    assert not np.isnan(got).any(), f"{name}: NaN in output"
    assert not np.isinf(got).any(), f"{name}: Inf in output"


# ---------------------------------------------------------------------------
# Parity sweep across all implementations x D x representative S_q/S_kv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("S_q", [1, 8, 33, 256])
@pytest.mark.parametrize("S_kv", [8, 33, 2048])
@pytest.mark.parametrize("BH", [(1, 1), (2, 2)])
def test_streaming_prefill_attend_parity(implementation, D, S_q, S_kv, BH):
    if S_q > S_kv:
        pytest.skip("causal alignment requires S_q <= S_kv")
    B, H = BH
    args = _make_inputs(B, H, S_q, S_kv, D, seed=D + S_q + S_kv + B * H)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    assert got.shape == (B, H, S_q, D)
    _assert_no_nan_inf(got, implementation)
    np.testing.assert_allclose(got, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# First / middle / last query token behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("D", [32, 64, 128])
def test_streaming_prefill_first_middle_last_token(implementation, D):
    """S_q == S_kv self-attention: check the first row (sees only itself),
    a middle row, and the last row (sees the full KV range) individually."""
    B, H = 1, 2
    args = _make_inputs(B, H, 65, 65, D, seed=500 + D)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    _assert_no_nan_inf(got, implementation)
    for row, label in [(0, "first"), (32, "middle"), (64, "last")]:
        np.testing.assert_allclose(
            got[:, :, row, :],
            expected[:, :, row, :],
            atol=_ATOL,
            rtol=_RTOL,
            err_msg=f"{implementation} D={D} {label} token (row={row}) mismatch",
        )


# ---------------------------------------------------------------------------
# S_q == 1 (single new token — decode-shaped prefill call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("S_kv", [1, 17, 500])
def test_streaming_prefill_s_q_one(implementation, D, S_kv):
    args = _make_inputs(1, 2, 1, S_kv, D, seed=600 + D + S_kv)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    assert got.shape == (1, 2, 1, D)
    _assert_no_nan_inf(got, implementation)
    np.testing.assert_allclose(got, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# S_q < S_kv: cache-continuation shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("D", [32, 64, 128])
def test_streaming_prefill_cache_continuation(implementation, D):
    """S_kv > S_q: queries are new tokens appended after an existing
    (uncompressed) prefix — the KV-cache-continuation shape."""
    args = _make_inputs(1, 2, 16, 500, D, seed=700 + D)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    _assert_no_nan_inf(got, implementation)
    np.testing.assert_allclose(got, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# Non-multiple-of-block-size sequence lengths (S=33, S=127) — exercises the
# blocked variants' tail loop (visible % KV_BLOCK != 0) explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("S", [33, 127])
def test_streaming_prefill_non_multiple_of_block(implementation, D, S):
    args = _make_inputs(1, 3, S, S, D, seed=800 + D + S)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    _assert_no_nan_inf(got, implementation)
    np.testing.assert_allclose(got, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# Non-multiple-of-ROWS_PER_TG query-block sizes (multirow uses 4 rows/tg)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("S", [1, 5, 31, 33, 63, 65, 97])
def test_streaming_prefill_query_block_partial(implementation, S):
    """S_q not a multiple of 4 (ROWS_PER_TG for multirow): every valid row
    must still be written correctly."""
    D = 64
    args = _make_inputs(1, 1, S, S, D, seed=900 + S)
    expected = _reference_flash(*args)
    got = _run_kernel(*args, implementation=implementation)
    assert got.shape == (1, 1, S, D)
    _assert_no_nan_inf(got, implementation)
    np.testing.assert_allclose(got, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# Causal masking sanity: perturbing a future key must not change earlier
# rows' output — proof the loop-bound-as-mask isn't accidentally attending
# to slots past q_abs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_streaming_prefill_causal_masks_future(implementation):
    B, H, S, D = 1, 2, 40, 64
    q, k, v, scale = _make_inputs(B, H, S, S, D, seed=21)
    out1 = _run_kernel(q, k, v, scale, implementation)

    k2, v2 = k.copy(), v.copy()
    k2[:, :, -1, :] = 1000.0  # perturb the last (future-for-every-row-but-the-last) key
    v2[:, :, -1, :] = 1000.0
    out2 = _run_kernel(q, k2, v2, scale, implementation)

    # Every row except the very last one must be unaffected.
    np.testing.assert_allclose(out1[:, :, :-1, :], out2[:, :, :-1, :], atol=1e-3)
    assert not np.allclose(out1[:, :, -1, :], out2[:, :, -1, :])


# ---------------------------------------------------------------------------
# Cross-implementation agreement: all variants must agree with each other
# (not just with the numpy reference) to a tight tolerance, since they are
# all computing the identical algorithm at different block granularities.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64, 128])
def test_streaming_prefill_variants_agree_with_each_other(D):
    args = _make_inputs(2, 3, 77, 203, D, seed=1000 + D)
    outs = {impl: _run_kernel(*args, implementation=impl) for impl in _IMPLEMENTATIONS}
    baseline = outs["streaming"]
    for impl, got in outs.items():
        _assert_no_nan_inf(got, impl)
        np.testing.assert_allclose(
            got,
            baseline,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"{impl} disagrees with block=1 baseline beyond fp arithmetic-order noise",
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_streaming_prefill_rejects_bad_inputs():
    q, k, v, scale = _make_inputs(1, 1, 8, 16, 64)
    with pytest.raises(ValueError, match="multiple of 32"):
        streaming_prefill_attend(
            mx.array(np.zeros((1, 1, 8, 48), np.float16)),
            mx.array(k),
            mx.array(v),
            mx.array(scale),
        )
    with pytest.raises(ValueError, match="v must be"):
        streaming_prefill_attend(
            mx.array(q),
            mx.array(k),
            mx.array(np.zeros((1, 1, 1, 64), np.float16)),
            mx.array(scale),
        )
    with pytest.raises(ValueError, match="128 limit"):
        streaming_prefill_attend(
            mx.array(np.zeros((1, 1, 8, 256), np.float16)),
            mx.array(np.zeros((1, 1, 16, 256), np.float16)),
            mx.array(np.zeros((1, 1, 16, 256), np.float16)),
            mx.array(scale),
        )
    with pytest.raises(ValueError, match="unknown implementation"):
        streaming_prefill_attend(
            mx.array(q), mx.array(k), mx.array(v), mx.array(scale), implementation="bogus"
        )

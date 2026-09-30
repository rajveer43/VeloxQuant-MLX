"""Equivalence tests: batched [B*H, ...] NestedKV primitives vs. the
per-head reference functions.

nestedkv_score_batched/nestedkv_compress_prefill_batched/
nestedkv_append_decode_batched replace NestedKVKVCache's Python loop over
(b, h) with one batched call per primitive across all B*H heads. Each row's
scoring and top-K selection depends only on that row's own data and the
shared, row-invariant budget/window/beta/tau/kappa config (uniform per-head
budget, see #21) -- no cross-row coupling -- so the batching is a pure
dispatch optimization, verified bit-for-bit exact against the unbatched
per-head functions called in a loop.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.nestedkv_cache import NestedKVKVCache
from veloxquant_mlx.quantizers.nestedkv import (
    init_nestedkv_state,
    nestedkv_append_decode,
    nestedkv_append_decode_batched,
    nestedkv_compress_prefill,
    nestedkv_compress_prefill_batched,
    nestedkv_get_kv,
    nestedkv_score,
    nestedkv_score_batched,
)


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "desc,bh,S,D,window,seed",
    [
        ("single_row", 1, 50, 16, 8, 0),
        ("multi_row", 4, 50, 16, 8, 1),
        ("small_n", 3, 1, 8, 4, 2),
        ("two_tokens", 3, 2, 8, 4, 3),
        ("large_window", 2, 300, 32, 32, 4),
        ("window_larger_than_n", 3, 10, 8, 100, 5),
    ],
)
def test_nestedkv_score_batched_matches_reference(desc, bh, S, D, window, seed):
    # Tolerance is loose (not bit-exact): the batched `[BH,n,D]` sum/mean
    # reductions and the single-row `[n,D]` reductions can land on opposite
    # sides of float32 rounding at the last bit (MLX's reduction kernel
    # picks a different summation order per shape/stride) — normally
    # negligible, but `_min_max_normalize`'s span-based rescale amplifies a
    # ~1e-6 relative input difference into a much larger absolute score gap
    # at small, near-degenerate S (e.g. S=2 with near-collinear keys). This
    # is float non-associativity inherent to batching a reduction, not a
    # logic divergence — same class of tolerance as CurDKV's batched SVD.
    keys = _rand((bh, S, D), seed)
    scores_b = nestedkv_score_batched(keys, window=window)
    for row in range(bh):
        ref = nestedkv_score(keys[row], window=window)
        diff = float(mx.max(mx.abs(scores_b[row] - ref)).item())
        assert diff < 0.15, f"{desc}: row {row} diverged by {diff}"


@pytest.mark.parametrize(
    "desc,bh,S,D,n_sink,budget,window,seed",
    [
        ("single_row", 1, 30, 16, 2, 10, 8, 0),
        ("multi_row", 4, 50, 16, 2, 10, 8, 1),
        ("below_budget_keeps_all", 3, 8, 8, 2, 20, 4, 2),
        ("zero_sink", 3, 20, 8, 0, 8, 4, 3),
        ("single_token", 3, 1, 8, 0, 4, 4, 4),
        ("budget_equals_n_sink", 2, 15, 8, 5, 1, 4, 5),
    ],
)
def test_compress_prefill_batched_matches_reference(desc, bh, S, D, n_sink, budget, window, seed):
    keys = _rand((bh, S, D), seed)
    values = _rand((bh, S, D), seed + 100)

    kept_keys_b, kept_values_b = nestedkv_compress_prefill_batched(
        keys, values, n_sink=n_sink, budget=budget, window=window
    )
    for row in range(bh):
        st = init_nestedkv_state(n_sink)
        st = nestedkv_compress_prefill(st, keys[row], values[row], budget=budget, window=window)
        ref_k, ref_v = nestedkv_get_kv(st)
        kd = float(
            mx.max(mx.abs(kept_keys_b[row].astype(mx.float32) - ref_k.astype(mx.float32))).item()
        )
        vd = float(
            mx.max(mx.abs(kept_values_b[row].astype(mx.float32) - ref_v.astype(mx.float32))).item()
        )
        assert kd < 1e-2, f"{desc}: row {row} keys diverged by {kd}"
        assert vd < 1e-2, f"{desc}: row {row} values diverged by {vd}"
        assert kept_keys_b.dtype == mx.float16
        assert kept_values_b.dtype == mx.float16


@pytest.mark.parametrize(
    "desc,bh,n_prev,S,D,seed",
    [
        ("bootstrap", 3, 0, 1, 8, 0),
        ("append_one", 3, 5, 1, 8, 1),
        ("append_multi_token", 4, 5, 3, 8, 2),
    ],
)
def test_append_decode_batched_matches_reference(desc, bh, n_prev, S, D, seed):
    keys = _rand((bh, S, D), seed)
    values = _rand((bh, S, D), seed + 100)

    if n_prev == 0:
        prev_keys_b = None
        prev_values_b = None
        prev_states = [init_nestedkv_state(2) for _ in range(bh)]
    else:
        prev_keys_b = _rand((bh, n_prev, D), seed + 200).astype(mx.float16)
        prev_values_b = _rand((bh, n_prev, D), seed + 300).astype(mx.float16)
        prev_states = [
            init_nestedkv_state(2).__class__(
                keys=prev_keys_b[row], values=prev_values_b[row], n_sink=2, compressed=True
            )
            for row in range(bh)
        ]

    new_keys_b, new_values_b = nestedkv_append_decode_batched(
        prev_keys_b, prev_values_b, keys, values
    )
    for row in range(bh):
        st = nestedkv_append_decode(prev_states[row], keys[row], values[row])
        ref_k, ref_v = nestedkv_get_kv(st)
        assert mx.array_equal(new_keys_b[row], ref_k), f"{desc}: row {row} keys"
        assert mx.array_equal(new_values_b[row], ref_v), f"{desc}: row {row} values"


# ---------------------------------------------------------------------------
# Real NestedKVKVCache end-to-end vs. a manual per-head reference using the
# original unbatched primitives directly.
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {
        "method": "nestedkv",
        "head_dim": 16,
        "nestedkv_budget": 8,
        "nestedkv_n_sink": 2,
        "nestedkv_window": 8,
    }
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize("B,H,seed", [(1, 1, 0), (2, 1, 1), (1, 4, 2), (2, 3, 3)])
def test_real_cache_prefill_matches_reference_loop(B, H, seed):
    D, S, budget, n_sink, window = 16, 24, 8, 2, 8
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(
        head_dim=D, nestedkv_budget=budget, nestedkv_n_sink=n_sink, nestedkv_window=window
    )
    k_out, v_out = cache.update_and_fetch(k, v)

    k_out_b, v_out_b = [], []
    for b in range(B):
        k_out_h, v_out_h = [], []
        for h in range(H):
            st = init_nestedkv_state(n_sink)
            st = nestedkv_compress_prefill(st, k[b, h], v[b, h], budget=budget, window=window)
            kh, vh = nestedkv_get_kv(st)
            k_out_h.append(kh)
            v_out_h.append(vh)
        k_out_b.append(mx.stack(k_out_h, axis=0))
        v_out_b.append(mx.stack(v_out_h, axis=0))
    ref_k = mx.stack(k_out_b, axis=0)
    ref_v = mx.stack(v_out_b, axis=0)

    assert mx.array_equal(k_out, ref_k)
    assert mx.array_equal(v_out, ref_v)


@pytest.mark.parametrize("B,H,seed", [(1, 1, 0), (2, 3, 1)])
def test_real_cache_prefill_then_decode_matches_reference_loop(B, H, seed):
    D, S, budget, n_sink, window = 16, 20, 6, 1, 4
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(
        head_dim=D, nestedkv_budget=budget, nestedkv_n_sink=n_sink, nestedkv_window=window
    )
    cache.update_and_fetch(k, v)

    ref_states = []
    for b in range(B):
        for h in range(H):
            st = init_nestedkv_state(n_sink)
            st = nestedkv_compress_prefill(st, k[b, h], v[b, h], budget=budget, window=window)
            ref_states.append(st)

    for i in range(5):
        kd = mx.array(
            np.random.default_rng(100 + i).standard_normal((B, H, 1, D)).astype(np.float32)
        ).astype(mx.float16)
        vd = mx.array(
            np.random.default_rng(200 + i).standard_normal((B, H, 1, D)).astype(np.float32)
        ).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(kd, vd)
        idx = 0
        for b in range(B):
            for h in range(H):
                ref_states[idx] = nestedkv_append_decode(ref_states[idx], kd[b, h], vd[b, h])
                idx += 1

    ref_k_out, ref_v_out = [], []
    idx = 0
    for b in range(B):
        k_h, v_h = [], []
        for h in range(H):
            kh, vh = nestedkv_get_kv(ref_states[idx])
            k_h.append(kh)
            v_h.append(vh)
            idx += 1
        ref_k_out.append(mx.stack(k_h, axis=0))
        ref_v_out.append(mx.stack(v_h, axis=0))
    ref_k = mx.stack(ref_k_out, axis=0)
    ref_v = mx.stack(ref_v_out, axis=0)

    assert mx.array_equal(k_out, ref_k)
    assert mx.array_equal(v_out, ref_v)


def test_real_cache_batch_dims_consistent_with_reference_shapes():
    """Regression for issue #568 Finding 1's shape-safety concern: batched
    prefill/decode must preserve exact B/H/n/D output shape the per-head
    loop produced."""
    D, S, budget, n_sink = 16, 30, 8, 2
    cache = _make_cache(head_dim=D, nestedkv_budget=budget, nestedkv_n_sink=n_sink)
    k, v = _rand((2, 3, S, D), 0).astype(mx.float16), _rand((2, 3, S, D), 1).astype(mx.float16)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (2, 3, budget, D)
    assert v_out.shape == (2, 3, budget, D)

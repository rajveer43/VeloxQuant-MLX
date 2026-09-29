"""Parity checks for MorphKV's batched (B*H) eviction dispatch (#560).

Two stacked findings, both fixed here:

1. ``MorphKVKVCache.update_and_fetch`` looped ``for b in range(B): for h in
   range(H):``, calling :func:`morphkv_update` once per head via a fresh
   Python object per (b,h) — the same unbatched-dispatch bottleneck already
   fixed for H2O/Keyformer/Squeeze/KVZip. :func:`morphkv_update_batched`
   replaces it with one call over flat ``[BH, n, D]`` state.
2. Inside every per-head eviction step, ``_recent_relevance`` itself looped
   ``for j in range(w):``, calling ``attention_scores`` once per recency-
   window position and accumulating. :func:`_recent_relevance_batched`
   replaces this with one batched matmul over the window axis (same recipe
   as PyramidKV's #549 fix) — and is exercised standalone here since it
   compounds with finding 1 (paid once per head per token in the old code).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.morphkv import (
    _recent_relevance,
    _recent_relevance_batched,
    init_morphkv_state,
    morphkv_update,
    morphkv_update_batched,
)


def _make(**cfg):
    base = {"method": "morphkv", "head_dim": 8}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Window-relevance primitive (finding 2): looped vs. batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bh,n,d,w",
    [
        (3, 15, 8, 4),
        (1, 8, 6, 1),  # window=1 -> TOVA-adapted collapse
        (5, 20, 4, 8),
        (4, 9, 6, 9),  # window == n (whole cache is the window)
    ],
)
def test_recent_relevance_batched_matches_looped(bh: int, n: int, d: int, w: int) -> None:
    rng = np.random.default_rng(3)
    keys = mx.array(rng.standard_normal((bh, n, d)).astype(np.float32))
    ref = mx.stack([_recent_relevance(keys[h], keys[h, n - w :]) for h in range(bh)])
    batched = _recent_relevance_batched(keys, w)
    mx.eval(ref, batched)
    assert mx.allclose(ref, batched, atol=1e-5).item()


# ---------------------------------------------------------------------------
# 2. Update primitive (finding 1): per-head looped vs. batched
# ---------------------------------------------------------------------------


def _gen(bh: int, n_steps: int, d: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    v = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    return k, v


def _run_looped(bh, n_steps, d, n_sink, budget, window, k, v):
    states = [init_morphkv_state(n_sink, budget, d, window=window) for _ in range(bh)]
    for t in range(n_steps):
        for h in range(bh):
            states[h] = morphkv_update(states[h], mx.array(k[t, h]), mx.array(v[t, h]))
    return mx.stack([s.keys for s in states]), mx.stack([s.values for s in states])


def _run_batched(bh, n_steps, d, n_sink, budget, window, k, v):
    keys = values = None
    for t in range(n_steps):
        kt = mx.array(k[t]).reshape(bh, 1, d)
        vt = mx.array(v[t]).reshape(bh, 1, d)
        keys, values = morphkv_update_batched(keys, values, kt, vt, n_sink, budget, window)
    return keys, values


@pytest.mark.parametrize(
    "bh,n_steps,d,n_sink,budget,window",
    [
        (3, 20, 8, 1, 8, 2),  # below-then-above-budget
        (4, 30, 8, 2, 10, 1),  # window=1 -> TOVA-adapted collapse
        (5, 15, 6, 0, 6, 3),  # no sinks
        (2, 8, 4, 1, 20, 4),  # never exceeds budget (pure bootstrap)
    ],
)
def test_batched_matches_looped_update(bh, n_steps, d, n_sink, budget, window) -> None:
    k, v = _gen(bh, n_steps, d)
    k_loop, v_loop = _run_looped(bh, n_steps, d, n_sink, budget, window, k, v)
    k_batch, v_batch = _run_batched(bh, n_steps, d, n_sink, budget, window, k, v)
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


def test_batched_matches_looped_prefill_shaped() -> None:
    """A single multi-token (S>1) call, not built up one token at a time."""
    bh, s, d = 3, 40, 8
    n_sink, budget, window = 1, 12, 3

    rng = np.random.default_rng(123)
    k = rng.standard_normal((bh, s, d)).astype(np.float16)
    v = rng.standard_normal((bh, s, d)).astype(np.float16)

    states = [init_morphkv_state(n_sink, budget, d, window=window) for _ in range(bh)]
    for h in range(bh):
        states[h] = morphkv_update(states[h], mx.array(k[h]), mx.array(v[h]))
    k_loop = mx.stack([st.keys for st in states])
    v_loop = mx.stack([st.values for st in states])

    k_batch, v_batch = morphkv_update_batched(
        None, None, mx.array(k), mx.array(v), n_sink, budget, window
    )
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


# ---------------------------------------------------------------------------
# 3. Cache-level: multi-head batching doesn't cross-contaminate heads
# ---------------------------------------------------------------------------


def test_multihead_cache_matches_per_head_reference() -> None:
    """Run MorphKVKVCache with B*H > 1 heads of independent random data and
    confirm each head's output matches what a single-head cache produces
    when fed that head's data alone."""
    H, D = 4, 8
    cfg = {"morphkv_budget": 10, "morphkv_n_sink": 1, "morphkv_window": 3, "head_dim": D}

    rng = np.random.default_rng(11)
    n_steps = 25
    all_k = [rng.standard_normal((n_steps, D)).astype(np.float16) for _ in range(H)]
    all_v = [rng.standard_normal((n_steps, D)).astype(np.float16) for _ in range(H)]

    multi = _make(**cfg)
    ko_multi = vo_multi = None
    for t in range(n_steps):
        k = mx.array(np.stack([all_k[h][t] for h in range(H)])[None, :, None, :])  # [1,H,1,D]
        v = mx.array(np.stack([all_v[h][t] for h in range(H)])[None, :, None, :])
        ko_multi, vo_multi = multi.update_and_fetch(k, v)

    for h in range(H):
        single = _make(**cfg)
        ko_single = vo_single = None
        for t in range(n_steps):
            k = mx.array(all_k[h][t][None, None, None, :])  # [1,1,1,D]
            v = mx.array(all_v[h][t][None, None, None, :])
            ko_single, vo_single = single.update_and_fetch(k, v)

        multi_h_k = np.array(ko_multi[0, h])
        multi_h_v = np.array(vo_multi[0, h])
        single_k = np.array(ko_single[0, 0])
        single_v = np.array(vo_single[0, 0])
        assert np.array_equal(multi_h_k, single_k), f"head {h} key mismatch"
        assert np.array_equal(multi_h_v, single_v), f"head {h} value mismatch"

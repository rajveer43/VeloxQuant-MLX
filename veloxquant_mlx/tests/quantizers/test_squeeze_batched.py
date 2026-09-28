"""Parity checks for SqueezeAttention's batched (B*H) eviction dispatch (#558).

``SqueezeAttentionCache.update_and_fetch`` used to loop ``for b in range(B):
for h in range(H):``, calling :func:`squeeze_update` once per head via a
fresh Python object per (b,h) — the same unbatched-dispatch bottleneck
already fixed for H2O. ``squeeze_update`` is documented ("Identical
mechanics to h2o_update") as H2O's cumulative-attention-mass scorer minus
RoPE remap and the grace/decay extensions, so :func:`squeeze_update_batched`
is a direct, simplified port of :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`'s
batching approach. The post-prefill one-shot re-budget
(``SqueezeAttentionCache._apply_budget`` / the old ``_trim_state``) is
likewise batched via :func:`squeeze_trim_batched`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.squeeze_cache import SqueezeAttentionCache
from veloxquant_mlx.quantizers.squeeze import (
    init_squeeze_state,
    squeeze_trim_batched,
    squeeze_update,
    squeeze_update_batched,
)


def _make(**cfg):
    base = {"method": "squeeze", "head_dim": 8}
    base.update(cfg)
    return SqueezeAttentionCache(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Update primitive: per-head looped vs. batched
# ---------------------------------------------------------------------------


def _gen(bh: int, n_steps: int, d: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    v = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    return k, v


def _run_looped(bh, n_steps, d, n_sink, budget, k, v):
    states = [init_squeeze_state(n_sink, budget, d) for _ in range(bh)]
    for t in range(n_steps):
        for h in range(bh):
            states[h] = squeeze_update(states[h], mx.array(k[t, h]), mx.array(v[t, h]))
    return (
        mx.stack([s.keys for s in states]),
        mx.stack([s.values for s in states]),
        mx.stack([s.scores for s in states]),
    )


def _run_batched(bh, n_steps, d, n_sink, budget, k, v):
    keys = values = scores = None
    for t in range(n_steps):
        kt = mx.array(k[t]).reshape(bh, 1, d)
        vt = mx.array(v[t]).reshape(bh, 1, d)
        keys, values, scores = squeeze_update_batched(keys, values, scores, kt, vt, n_sink, budget)
    return keys, values, scores


@pytest.mark.parametrize(
    "bh,n_steps,d,n_sink,budget",
    [
        (3, 20, 8, 1, 8),  # below-then-above-budget
        (4, 30, 8, 2, 10),
        (5, 15, 6, 0, 6),  # no sinks
        (2, 8, 4, 1, 20),  # never exceeds budget (pure bootstrap)
    ],
)
def test_batched_matches_looped_update(bh, n_steps, d, n_sink, budget) -> None:
    k, v = _gen(bh, n_steps, d)
    k_loop, v_loop, s_loop = _run_looped(bh, n_steps, d, n_sink, budget, k, v)
    k_batch, v_batch, s_batch = _run_batched(bh, n_steps, d, n_sink, budget, k, v)
    mx.eval(k_loop, v_loop, s_loop, k_batch, v_batch, s_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()
    assert mx.allclose(s_loop, s_batch, atol=1e-5).item()


def test_batched_matches_looped_prefill_shaped() -> None:
    """A single multi-token (S>1) call, not built up one token at a time."""
    bh, s, d = 3, 40, 8
    n_sink, budget = 1, 12

    rng = np.random.default_rng(123)
    k = rng.standard_normal((bh, s, d)).astype(np.float16)
    v = rng.standard_normal((bh, s, d)).astype(np.float16)

    states = [init_squeeze_state(n_sink, budget, d) for _ in range(bh)]
    for h in range(bh):
        states[h] = squeeze_update(states[h], mx.array(k[h]), mx.array(v[h]))
    k_loop = mx.stack([st.keys for st in states])
    v_loop = mx.stack([st.values for st in states])

    k_batch, v_batch, _ = squeeze_update_batched(
        None, None, None, mx.array(k), mx.array(v), n_sink, budget
    )
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


# ---------------------------------------------------------------------------
# 2. Trim primitive (post-prefill one-shot re-budget): looped vs. batched
# ---------------------------------------------------------------------------


def _trim_row_ref(k, v, s, n_sink, budget):
    n = k.shape[0]
    if n <= budget:
        return k, v, s
    n_sink_eff = min(n_sink, n)
    if n_sink_eff > 0:
        inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
        protected = mx.concatenate([inf_block, s[n_sink_eff:]], axis=0)
    else:
        protected = s
    order = mx.argsort(protected)
    keep = order[n - budget :]
    keep_sorted = mx.sort(keep)
    idx = [int(x.item()) for x in keep_sorted]
    return k[idx], v[idx], s[idx]


@pytest.mark.parametrize(
    "bh,n,d,n_sink,new_budget",
    [
        (4, 20, 8, 2, 10),
        (3, 15, 6, 0, 5),
        (5, 12, 4, 3, 6),
        (2, 10, 4, 1, 20),  # new_budget >= n -> no-op
    ],
)
def test_trim_batched_matches_looped(bh, n, d, n_sink, new_budget) -> None:
    rng = np.random.default_rng(5)
    keys = mx.array(rng.standard_normal((bh, n, d)).astype(np.float16))
    values = mx.array(rng.standard_normal((bh, n, d)).astype(np.float16))
    scores = mx.array(rng.standard_normal((bh, n)).astype(np.float32))

    ref_k, ref_v, ref_s = [], [], []
    for h in range(bh):
        k, v, s = _trim_row_ref(keys[h], values[h], scores[h], n_sink, new_budget)
        ref_k.append(k)
        ref_v.append(v)
        ref_s.append(s)
    ref_k, ref_v, ref_s = mx.stack(ref_k), mx.stack(ref_v), mx.stack(ref_s)

    batch_k, batch_v, batch_s = squeeze_trim_batched(keys, values, scores, n_sink, new_budget)
    mx.eval(ref_k, ref_v, ref_s, batch_k, batch_v, batch_s)
    assert mx.array_equal(ref_k, batch_k).item()
    assert mx.array_equal(ref_v, batch_v).item()
    assert mx.array_equal(ref_s, batch_s).item()


def test_trim_batched_empty_state_is_noop() -> None:
    out = squeeze_trim_batched(None, None, None, n_sink=2, budget=10)
    assert out == (None, None, None)


# ---------------------------------------------------------------------------
# 3. Cache-level: multi-head batching doesn't cross-contaminate heads
# ---------------------------------------------------------------------------


def test_multihead_cache_matches_per_head_reference() -> None:
    """Run SqueezeAttentionCache with B*H > 1 heads of independent random
    data and confirm each head's output matches what a single-head cache
    produces when fed that head's data alone."""
    H, D = 4, 8
    cfg = {"squeeze_budget": 10, "squeeze_n_sink": 1, "head_dim": D}

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

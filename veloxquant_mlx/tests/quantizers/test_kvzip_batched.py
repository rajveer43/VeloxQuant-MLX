"""Parity checks for KVzip's batched (B*H) eviction dispatch (#559).

``KVzipKVCache.update_and_fetch`` used to loop ``for b in range(B): for h
in range(H):``, calling :func:`kvzip_update` once per head via a fresh
Python object per (b,h) — the same unbatched-dispatch bottleneck already
fixed for H2O/Keyformer/Squeeze. :func:`kvzip_update_batched` replaces the
outer dispatch loop with one call operating on flat ``[BH, n, D]`` state,
batching the per-token append/rank/evict math the same way
:func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched` batches H2O.

Unlike H2O, KVzip carries no cumulative score and no RoPE remap — each
step's reconstruction-reliance ranking is recomputed fresh from the live
keep set against the configured probe (``"context"``: full keep set;
``"latest"``: single most-recent key, the TOVA-adapted collapse). Both
probes are covered here.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.kvzip import (
    init_kvzip_state,
    kvzip_update,
    kvzip_update_batched,
)


def _make(**cfg):
    base = {"method": "kvzip", "head_dim": 8}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Update primitive: per-head looped vs. batched
# ---------------------------------------------------------------------------


def _gen(bh: int, n_steps: int, d: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    v = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    return k, v


def _run_looped(bh, n_steps, d, n_sink, budget, probe, k, v):
    states = [init_kvzip_state(n_sink, budget, d, probe=probe) for _ in range(bh)]
    for t in range(n_steps):
        for h in range(bh):
            states[h] = kvzip_update(states[h], mx.array(k[t, h]), mx.array(v[t, h]))
    return mx.stack([s.keys for s in states]), mx.stack([s.values for s in states])


def _run_batched(bh, n_steps, d, n_sink, budget, probe, k, v):
    keys = values = None
    for t in range(n_steps):
        kt = mx.array(k[t]).reshape(bh, 1, d)
        vt = mx.array(v[t]).reshape(bh, 1, d)
        keys, values = kvzip_update_batched(keys, values, kt, vt, n_sink, budget, probe)
    return keys, values


@pytest.mark.parametrize(
    "bh,n_steps,d,n_sink,budget,probe",
    [
        (3, 20, 8, 1, 8, "context"),  # below-then-above-budget
        (4, 30, 8, 2, 10, "latest"),  # TOVA-adapted collapse
        (5, 15, 6, 0, 6, "context"),  # no sinks
        (2, 8, 4, 1, 20, "latest"),  # never exceeds budget (pure bootstrap)
    ],
)
def test_batched_matches_looped_update(bh, n_steps, d, n_sink, budget, probe) -> None:
    k, v = _gen(bh, n_steps, d)
    k_loop, v_loop = _run_looped(bh, n_steps, d, n_sink, budget, probe, k, v)
    k_batch, v_batch = _run_batched(bh, n_steps, d, n_sink, budget, probe, k, v)
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


@pytest.mark.parametrize("probe", ["context", "latest"])
def test_batched_matches_looped_prefill_shaped(probe: str) -> None:
    """A single multi-token (S>1) call, not built up one token at a time."""
    bh, s, d = 3, 40, 8
    n_sink, budget = 1, 12

    rng = np.random.default_rng(123)
    k = rng.standard_normal((bh, s, d)).astype(np.float16)
    v = rng.standard_normal((bh, s, d)).astype(np.float16)

    states = [init_kvzip_state(n_sink, budget, d, probe=probe) for _ in range(bh)]
    for h in range(bh):
        states[h] = kvzip_update(states[h], mx.array(k[h]), mx.array(v[h]))
    k_loop = mx.stack([st.keys for st in states])
    v_loop = mx.stack([st.values for st in states])

    k_batch, v_batch = kvzip_update_batched(
        None, None, mx.array(k), mx.array(v), n_sink, budget, probe
    )
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


# ---------------------------------------------------------------------------
# 2. Cache-level: multi-head batching doesn't cross-contaminate heads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", ["context", "latest"])
def test_multihead_cache_matches_per_head_reference(probe: str) -> None:
    """Run KVzipKVCache with B*H > 1 heads of independent random data and
    confirm each head's output matches what a single-head cache produces
    when fed that head's data alone."""
    H, D = 4, 8
    cfg = {"kvzip_budget": 10, "kvzip_n_sink": 1, "kvzip_probe": probe, "head_dim": D}

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

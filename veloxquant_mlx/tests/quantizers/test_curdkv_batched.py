"""Equivalence tests: curdkv_update_batched vs. per-head curdkv_update.

CurDKV's per-token leverage-score recurrence (score accumulation, eviction,
RoPE remap) is a genuine sequential dependency and stays a Python loop over
S in both paths — what curdkv_update_batched changes is running that
recurrence across all BH heads in one batched MLX/numpy call per step
instead of BH separate Python-level calls into curdkv_update. The batched
SVD (numpy.linalg.svd's native stacked-leading-dims support) computes the
exact same per-row factorization as BH separate unbatched SVD calls would,
so the two paths are expected to be bit-for-bit identical: no float32
reduction-order difference exists here the way it does for CaM's merge
blend, since nothing is summed across rows/heads — each row's leverage
score is an independent per-row SVD, batched only for LAPACK dispatch
efficiency, not for the linear algebra itself.

Also covers the real CurDKVKVCache end-to-end (factory-level) against a
manual per-head reference loop, since that is the path real callers use.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.curdkv import (
    curdkv_get_kv,
    curdkv_update,
    curdkv_update_batched,
    init_curdkv_state,
)


def _rand_kv(B: int, H: int, S: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _reference_loop(B, H, D, n_sink, budget, rank_cap, rope_base, steps):
    """BH independent curdkv_update states, stepped in lockstep."""
    states = [
        init_curdkv_state(n_sink, budget, D, rank_cap, rope_base=rope_base) for _ in range(B * H)
    ]
    for k_bh, v_bh in steps:
        idx = 0
        for b in range(B):
            for h in range(H):
                states[idx] = curdkv_update(states[idx], k_bh[b, h], v_bh[b, h])
                idx += 1
    keys_out, values_out = [], []
    for st in states:
        k, v = curdkv_get_kv(st)
        keys_out.append(k)
        values_out.append(v)
    return mx.stack(keys_out, axis=0), mx.stack(values_out, axis=0)


def _batched_run(B, H, D, n_sink, budget, rank_cap, rope_base, steps):
    keys = values = leverage_scores = n_updates = positions = None
    next_pos = 0
    for k_bh, v_bh in steps:
        bh = B * H
        S = k_bh.shape[2]
        new_keys = k_bh.reshape(bh, S, D)
        new_values = v_bh.reshape(bh, S, D)
        keys, values, leverage_scores, n_updates, positions, next_pos = curdkv_update_batched(
            keys,
            values,
            leverage_scores,
            n_updates,
            positions,
            new_keys,
            new_values,
            n_sink,
            budget,
            rank_cap,
            rope_base,
            next_pos,
        )
    return keys, values


@pytest.mark.parametrize(
    "desc,B,H,D,n_sink,budget,rank_cap,steps_spec,data_seed",
    [
        ("single_head_below_budget", 1, 1, 8, 1, 16, 4, [4], 1),
        ("single_batch_multi_head", 1, 3, 8, 1, 6, 4, [4, 1, 1, 1, 1], 2),
        ("multi_batch_multi_head", 2, 3, 8, 2, 6, 4, [3, 1, 1, 1, 1, 1], 3),
        ("no_sink", 1, 2, 8, 0, 5, 4, [3, 1, 1, 1, 1], 4),
        ("rank_cap_exceeds_n", 1, 2, 8, 1, 4, 16, [2, 1, 1, 1], 5),
        ("large_prefill_then_decode", 1, 2, 8, 2, 5, 3, [10, 1, 1, 1], 6),
        ("single_bh_many_decode_steps", 1, 1, 8, 1, 4, 2, [1] * 12, 7),
        ("wide_heads", 1, 6, 8, 1, 5, 4, [3, 1, 1, 1], 8),
    ],
)
def test_batched_matches_reference_loop(
    desc, B, H, D, n_sink, budget, rank_cap, steps_spec, data_seed
):
    rope_base = 10000.0
    steps = []
    seed = data_seed * 1000
    for S in steps_spec:
        k, v = _rand_kv(B, H, S, D, seed=seed)
        steps.append((k, v))
        seed += 1

    ref_k, ref_v = _reference_loop(B, H, D, n_sink, budget, rank_cap, rope_base, steps)
    batched_k, batched_v = _batched_run(B, H, D, n_sink, budget, rank_cap, rope_base, steps)

    assert batched_k.shape == ref_k.shape, desc
    assert batched_v.shape == ref_v.shape, desc
    assert mx.array_equal(batched_k, ref_k), f"{desc}: keys diverged"
    assert mx.array_equal(batched_v, ref_v), f"{desc}: values diverged"


def test_batched_zero_step_noop():
    keys, values, leverage_scores, n_updates, positions, next_pos = curdkv_update_batched(
        None, None, None, None, None, mx.zeros((2, 0, 4)), mx.zeros((2, 0, 4)), 1, 8, 4, 10000.0, 0
    )
    assert keys is None
    assert next_pos == 0


def test_batched_raises_on_sink_ge_budget():
    with pytest.raises(ValueError, match="n_sink"):
        curdkv_update_batched(
            None,
            None,
            None,
            None,
            None,
            mx.zeros((2, 1, 4)),
            mx.zeros((2, 1, 4)),
            4,
            4,
            2,
            10000.0,
            0,
        )


# ---------------------------------------------------------------------------
# Real CurDKVKVCache end-to-end vs. a manual per-head reference loop
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {"method": "curdkv", "head_dim": 8, "curdkv_budget": 6, "curdkv_n_sink": 1}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize(
    "B,H,seed",
    [(1, 1, 0), (2, 1, 1), (1, 4, 2), (2, 3, 3)],
)
def test_real_cache_matches_reference_loop(B, H, seed):
    D = 8
    n_sink = 1
    budget = 6
    rank_cap = 4

    cache = _make_cache(
        head_dim=D, curdkv_budget=budget, curdkv_n_sink=n_sink, curdkv_rank_cap=rank_cap
    )

    step_sizes = [5, 1, 1, 1, 1, 1]
    steps = []
    s = seed * 100
    for S in step_sizes:
        k, v = _rand_kv(B, H, S, D, seed=s)
        steps.append((k, v))
        s += 1

    cache_out_k = cache_out_v = None
    for k, v in steps:
        cache_out_k, cache_out_v = cache.update_and_fetch(k, v)
        cache_out_k, cache_out_v = cache.state[:2]  # Compare eviction results stored for the next call.

    ref_k, ref_v = _reference_loop(B, H, D, n_sink, budget, rank_cap, 10000.0, steps)
    n_kept = cache_out_k.shape[2]
    ref_k = ref_k.reshape(B, H, n_kept, D)
    ref_v = ref_v.reshape(B, H, n_kept, D)

    assert mx.array_equal(cache_out_k, ref_k)
    assert mx.array_equal(cache_out_v, ref_v)

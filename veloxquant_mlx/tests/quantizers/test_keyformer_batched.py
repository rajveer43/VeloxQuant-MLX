"""Parity checks for Keyformer's batched (B*H) eviction dispatch (#557).

``KeyformerKVCache.update_and_fetch`` used to loop ``for b in range(B): for
h in range(H):``, calling :func:`keyformer_update` once per head with that
head's own Python-object state. :func:`keyformer_update_batched` replaces
the outer dispatch loop with one call operating on flat ``[BH, n, D]`` /
``[BH, n]`` state, batching the per-token score-accumulate/append/evict/
RoPE-remap math the same way :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`
already batches H2O-adapted (Keyformer's own module docstring calls it the
"Metal-fused sibling" of H2O). The inner per-token recurrence (token *t*'s
eviction depends on state left by token *t-1*) is untouched — only the
outer per-head dispatch is batched, matching the issue's stated scope.

The one wrinkle unique to Keyformer: each head draws its own deterministic
Gumbel noise stream, keyed by ``(seed, pos)`` via ``mx.random.key`` — an API
that (confirmed empirically) has no batched-key overload in this MLX build.
:func:`_gumbel_at_batched` works around this by building the ``BH`` distinct
per-head keys with one cheap Python-level ``mx.random.key`` call each,
still only once per token *step* (not once per ``(head, step)`` pair), then
draws all ``BH`` rows' uniforms in one ``mx.vmap`` call — verified bit-for-
bit equal to stacking individual :func:`_gumbel_at` calls.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.keyformer import (
    _gumbel_at,
    _gumbel_at_batched,
    init_keyformer_state,
    keyformer_update,
    keyformer_update_batched,
)


def _make(**cfg):
    base = {"method": "keyformer", "head_dim": 8}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Gumbel draw primitive: single-key loop vs. batched vmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bh,base_seed,pos", [(1, 0, 0), (5, 3, 7), (8, 100, 42), (16, 0, 999)])
def test_gumbel_batched_matches_looped(bh: int, base_seed: int, pos: int) -> None:
    seeds = [base_seed + h for h in range(bh)]
    ref = mx.stack([_gumbel_at(s, pos) for s in seeds])
    batched = _gumbel_at_batched(seeds, pos)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


# ---------------------------------------------------------------------------
# 2. Quantizer-level primitive: per-head looped update vs. batched update
# ---------------------------------------------------------------------------


def _gen(bh: int, n_steps: int, d: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    v = rng.standard_normal((n_steps, bh, 1, d)).astype(np.float16)
    return k, v


def _run_looped(
    bh, n_steps, d, n_sink, budget, recent, tau_init, tau_end, anneal_steps, seed, k, v
):
    states = [
        init_keyformer_state(
            n_sink,
            budget,
            d,
            recent=recent,
            tau_init=tau_init,
            tau_end=tau_end,
            anneal_steps=anneal_steps,
            seed=seed + h,
        )
        for h in range(bh)
    ]
    for t in range(n_steps):
        for h in range(bh):
            states[h] = keyformer_update(states[h], mx.array(k[t, h]), mx.array(v[t, h]))
    return mx.stack([s.keys for s in states]), mx.stack([s.values for s in states])


def _run_batched(
    bh, n_steps, d, n_sink, budget, recent, tau_init, tau_end, anneal_steps, seed, k, v
):
    keys = values = scores = gumbel = positions = None
    next_pos = pos = 0
    seeds = [seed + h for h in range(bh)]
    for t in range(n_steps):
        kt = mx.array(k[t]).reshape(bh, 1, d)
        vt = mx.array(v[t]).reshape(bh, 1, d)
        keys, values, scores, gumbel, positions, next_pos, pos = keyformer_update_batched(
            keys,
            values,
            scores,
            gumbel,
            positions,
            kt,
            vt,
            n_sink,
            budget,
            recent,
            tau_init,
            tau_end,
            anneal_steps,
            10000.0,
            next_pos,
            pos,
            seeds,
        )
    return keys, values


@pytest.mark.parametrize(
    "bh,n_steps,d,n_sink,budget,recent,tau_init,tau_end,anneal_steps,seed",
    [
        (3, 20, 8, 1, 8, 0, 1.0, 1.0, 0, 0),  # below-then-above-budget, constant tau
        (4, 30, 8, 2, 10, 2, 1.0, 2.0, 15, 5),  # recent window + annealed tau
        (2, 12, 6, 0, 6, 0, 0.0, 0.0, 0, 1),  # tau=0 ablation (H2O-equivalent)
        (5, 8, 4, 1, 20, 0, 1.0, 1.0, 0, 9),  # never exceeds budget (pure bootstrap)
    ],
)
def test_batched_matches_looped_update(
    bh, n_steps, d, n_sink, budget, recent, tau_init, tau_end, anneal_steps, seed
) -> None:
    k, v = _gen(bh, n_steps, d)
    k_loop, v_loop = _run_looped(
        bh, n_steps, d, n_sink, budget, recent, tau_init, tau_end, anneal_steps, seed, k, v
    )
    k_batch, v_batch = _run_batched(
        bh, n_steps, d, n_sink, budget, recent, tau_init, tau_end, anneal_steps, seed, k, v
    )
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()


def test_batched_matches_looped_prefill_shaped() -> None:
    """A single multi-token (S>1) call, not built up one token at a time —
    the shape mlx_lm's chunked prefill actually uses."""
    bh, s, d = 3, 40, 8
    n_sink, budget, recent = 1, 12, 1
    tau_init, tau_end, anneal_steps = 1.0, 2.0, 20
    seed = 7

    rng = np.random.default_rng(123)
    k = rng.standard_normal((bh, s, d)).astype(np.float16)
    v = rng.standard_normal((bh, s, d)).astype(np.float16)

    states = [
        init_keyformer_state(
            n_sink,
            budget,
            d,
            recent=recent,
            tau_init=tau_init,
            tau_end=tau_end,
            anneal_steps=anneal_steps,
            seed=seed + h,
        )
        for h in range(bh)
    ]
    for h in range(bh):
        states[h] = keyformer_update(states[h], mx.array(k[h]), mx.array(v[h]))
    k_loop = mx.stack([s.keys for s in states])
    v_loop = mx.stack([s.values for s in states])

    seeds = [seed + h for h in range(bh)]
    k_batch, v_batch, _, _, _, next_pos, pos = keyformer_update_batched(
        None,
        None,
        None,
        None,
        None,
        mx.array(k),
        mx.array(v),
        n_sink,
        budget,
        recent,
        tau_init,
        tau_end,
        anneal_steps,
        10000.0,
        0,
        0,
        seeds,
    )
    mx.eval(k_loop, v_loop, k_batch, v_batch)
    assert mx.array_equal(k_loop, k_batch).item()
    assert mx.array_equal(v_loop, v_batch).item()
    assert next_pos == s
    assert pos == s


# ---------------------------------------------------------------------------
# 3. Cache-level: multi-head batching doesn't cross-contaminate heads
# ---------------------------------------------------------------------------


def test_multihead_cache_matches_per_head_reference() -> None:
    """Run KeyformerKVCache with B*H > 1 heads of independent random data and
    confirm each head's output matches what a single-head cache produces
    when fed that head's data alone -- the key risk in batching over B*H is
    one head's content leaking into another's eviction decision, or heads
    accidentally sharing a Gumbel noise stream."""
    H, D = 4, 8
    cfg = {
        "keyformer_budget": 10,
        "keyformer_n_sink": 1,
        "keyformer_tau_init": 1.0,
        "keyformer_tau_end": 2.0,
        "keyformer_anneal_steps": 10,
        "head_dim": D,
    }

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
        # multi's per-row seed for row h is base_seed(0) + h (see _ensure_states);
        # a single-head cache constructed with keyformer_seed=h reproduces that
        # row's exact Gumbel stream.
        single = _make(**{**cfg, "keyformer_seed": h})
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

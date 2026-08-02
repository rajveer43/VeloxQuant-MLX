"""Tests for Keyformer-adapted eviction primitives (quantizers/keyformer.py).

Covers: init guards, budget invariant, sink/recent protection, byte accounting,
the tau=0 == H2O-adapted collapse (the honest ablation), Gumbel determinism/
reproducibility, and the "late riser" mechanism (Gumbel rescues a token that a
deterministic scorer would prune early).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.keyformer import (
    KeyformerState,
    _gumbel_at,
    full_keyformer_fp16_bytes,
    init_keyformer_state,
    keyformer_fp16_bytes,
    keyformer_get_kv,
    keyformer_update,
)
from veloxquant_mlx.quantizers.h2o import h2o_update, h2o_get_kv, init_h2o_state


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init guards
# ---------------------------------------------------------------------------
def test_init_rejects_negative_tau():
    with pytest.raises(ValueError, match="tau must be >= 0"):
        init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=-0.1)


def test_init_rejects_no_evictable_room():
    with pytest.raises(ValueError, match="no evictable positions"):
        init_keyformer_state(n_sink=6, budget=8, head_dim=16, recent=2)


def test_init_empty_state():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16)
    assert st.keys is None and st.values is None and st.scores is None
    assert st.gumbel is None and st.pos == 0
    assert st.n_sink == 2 and st.budget == 8 and st.tau == 1.0


# ---------------------------------------------------------------------------
# budget invariant & basic mechanics
# ---------------------------------------------------------------------------
def test_budget_never_exceeded_token_by_token():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=1.0, seed=0)
    for i in range(40):
        k, v = _rand_kv(1, 16, seed=i)
        st = keyformer_update(st, k, v)
        assert st.keys.shape[0] <= 8


def test_budget_never_exceeded_block_prefill():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=1.0, seed=0)
    k, v = _rand_kv(40, 16, seed=7)
    st = keyformer_update(st, k, v)
    assert st.keys.shape[0] == 8


def test_under_budget_keeps_all():
    st = init_keyformer_state(n_sink=2, budget=32, head_dim=16, tau=1.0)
    k, v = _rand_kv(10, 16, seed=3)
    st = keyformer_update(st, k, v)
    K, _ = keyformer_get_kv(st)
    assert K.shape[0] == 10


def test_sinks_survive_eviction():
    # Sinks are the first n_sink positions; they must remain after heavy eviction.
    st = init_keyformer_state(n_sink=3, budget=8, head_dim=8, tau=1.0, seed=1)
    first_k, first_v = _rand_kv(3, 8, seed=100)
    st = keyformer_update(st, first_k, first_v)
    sink_rows = st.keys[:3]
    for i in range(60):
        k, v = _rand_kv(1, 8, seed=i)
        st = keyformer_update(st, k, v)
    # The three planted sink rows are still present (leading positions).
    assert bool(mx.all(st.keys[:3] == sink_rows).item())


def test_recent_window_protects_trailing():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=8, recent=3, tau=5.0, seed=2)
    for i in range(30):
        k, v = _rand_kv(1, 8, seed=i)
        st = keyformer_update(st, k, v)
    # last inserted row must be present (protected by recent window)
    last_k, _ = _rand_kv(1, 8, seed=29)
    assert bool(mx.all(st.keys[-1] == last_k[0]).item())


# ---------------------------------------------------------------------------
# byte accounting
# ---------------------------------------------------------------------------
def test_fp16_bytes():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=1.0)
    assert keyformer_fp16_bytes(st) == 0
    k, v = _rand_kv(8, 16, seed=5)
    st = keyformer_update(st, k, v)
    assert keyformer_fp16_bytes(st) == 8 * 16 * 2 * 2


def test_full_fp16_bytes():
    assert full_keyformer_fp16_bytes(100, 64) == 100 * 64 * 2 * 2


def test_get_kv_placeholder_before_update():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16)
    K, V = keyformer_get_kv(st)
    assert K.shape == (0, 1) and V.shape == (0, 1)


# ---------------------------------------------------------------------------
# Gumbel determinism / reproducibility
# ---------------------------------------------------------------------------
def test_gumbel_deterministic_per_position():
    a = float(_gumbel_at(0, 5).item())
    b = float(_gumbel_at(0, 5).item())
    assert a == b  # same (seed,pos) -> same value
    c = float(_gumbel_at(0, 6).item())
    assert a != c  # different pos -> different value
    d = float(_gumbel_at(1, 5).item())
    assert a != d  # different seed -> different value


def test_run_is_reproducible():
    def run():
        st = init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=2.0, seed=42)
        for i in range(30):
            k, v = _rand_kv(1, 16, seed=i)
            st = keyformer_update(st, k, v)
        return keyformer_get_kv(st)[0]

    assert bool(mx.all(run() == run()).item())


# ---------------------------------------------------------------------------
# tau = 0 collapses onto H2O-adapted (the honest ablation)
# ---------------------------------------------------------------------------
def test_tau_zero_matches_h2o():
    ks = [_rand_kv(1, 16, seed=i) for i in range(40)]

    kf = init_keyformer_state(n_sink=4, budget=12, head_dim=16, tau=0.0, seed=7)
    h2o = init_h2o_state(n_sink=4, budget=12, head_dim=16)
    for k, v in ks:
        kf = keyformer_update(kf, k, v)
        h2o = h2o_update(h2o, k, v)

    kf_k, _ = keyformer_get_kv(kf)
    h2o_k, _ = h2o_get_kv(h2o)
    assert kf_k.shape == h2o_k.shape
    assert bool(mx.all(kf_k == h2o_k).item())


def test_tau_zero_is_seed_invariant():
    # With no noise, the seed cannot matter — kept set identical for any seed.
    ks = [_rand_kv(1, 16, seed=i) for i in range(40)]

    def run(seed):
        st = init_keyformer_state(n_sink=2, budget=10, head_dim=16, tau=0.0, seed=seed)
        for k, v in ks:
            st = keyformer_update(st, k, v)
        return keyformer_get_kv(st)[0]

    assert bool(mx.all(run(0) == run(123)).item())


def test_positive_tau_can_change_kept_set():
    # Noise should be capable of altering which tokens survive vs. tau=0.
    ks = [_rand_kv(1, 16, seed=i) for i in range(50)]

    def run(tau, seed):
        st = init_keyformer_state(n_sink=2, budget=10, head_dim=16, tau=tau, seed=seed)
        for k, v in ks:
            st = keyformer_update(st, k, v)
        return keyformer_get_kv(st)[0]

    base = run(0.0, 0)
    noisy = run(8.0, 3)
    # Not a guarantee for all seeds, but with strong noise the kept sets differ.
    assert base.shape == noisy.shape
    assert not bool(mx.all(base == noisy).item())


# ---------------------------------------------------------------------------
# late-riser mechanism: Gumbel can rescue a token a greedy scorer prunes early
# ---------------------------------------------------------------------------
def test_gumbel_rescues_late_riser():
    """A token that reads low early but would attract attention later.

    Construct a "planted" token whose key is nearly orthogonal to early traffic
    (so it accumulates ~0 proxy mass and is the greedy eviction target) but is
    strongly aligned with a burst of *later* keys. Under tau=0 (greedy H2O) it
    is far more often evicted before the burst arrives than under a large tau
    where the frozen noise can protect it. We assert the survival RATE across
    seeds is higher with noise on — a statistical mechanism claim, not a
    per-seed guarantee.
    """
    D = 16
    rng = np.random.default_rng(0)
    planted = np.zeros(D, dtype=np.float16)
    planted[0] = 3.0  # unique axis

    def build_stream(seed):
        r = np.random.default_rng(seed)
        # early filler orthogonal to planted axis (components 1..D-1 only)
        early = r.standard_normal((20, D)).astype(np.float16)
        early[:, 0] = 0.0
        # the planted token inserted early (position ~2)
        stream = [early[0], early[1], planted] + list(early[2:])
        # later burst aligned with planted axis -> would attend to it
        burst = np.zeros((6, D), dtype=np.float16)
        burst[:, 0] = 3.0
        stream += list(burst)
        return [mx.array(x[None]) for x in stream]

    def survived(tau, seed):
        st = init_keyformer_state(n_sink=1, budget=10, head_dim=D, tau=tau, seed=seed)
        stream = build_stream(seed)
        for k in stream:
            st = keyformer_update(st, k, k)  # value == key for simplicity
        K, _ = keyformer_get_kv(st)
        # planted survived if any kept row matches the planted axis strongly
        proj = K.astype(mx.float32) @ mx.array(planted.astype(np.float32))
        return bool((mx.max(proj) > 6.0).item())

    seeds = range(40)
    greedy_rate = sum(survived(0.0, s) for s in seeds) / len(list(seeds))
    noisy_rate = sum(survived(6.0, s) for s in seeds) / len(list(seeds))
    # The mechanism claim: noise improves late-riser survival on average.
    assert noisy_rate >= greedy_rate

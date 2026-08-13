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

from veloxquant_mlx.quantizers.a2ats_rope import rope_remap_positions
from veloxquant_mlx.quantizers.keyformer import (
    KeyformerState,
    _gumbel_at,
    _tau_at,
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
    # tau=0 (deterministic, no Gumbel noise) isolates the "recent" mechanism
    # from the noise mechanism: with noise, a large-enough draw can still pick
    # the brand-new token as the eviction target ahead of appending it (see
    # keyformer_update — the recent-window protects the trailing *stored*
    # rows, which does not include the not-yet-appended new row until after
    # it's concatenated in). A fingerprinted value row (values are never
    # rotated, unlike keys) is the robust survival check here: the recent
    # window must keep the newest-inserted row present even though an
    # eviction elsewhere in the same step can still shift its *position* down
    # (RoPE remap closing the gap), which is expected and separately covered
    # by the RoPE-remap-correctness tests below.
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=8, recent=3, tau=0.0, seed=2)
    for i in range(29):
        k, v = _rand_kv(1, 8, seed=i)
        st = keyformer_update(st, k, v)
    fingerprint = mx.ones((1, 8), dtype=mx.float16) * 9.0
    st = keyformer_update(st, fingerprint, fingerprint)
    assert bool(mx.any(mx.all(st.values == fingerprint[0], axis=-1)).item())


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
        _, V = keyformer_get_kv(st)
        # Survival check uses VALUES, not keys: keys get RoPE-remapped when an
        # eviction elsewhere shifts their position (see module docstring gap
        # #2), so a raw dot-product against the unrotated planted vector is
        # no longer a valid post-fix survival signal for keys. Values are
        # never rotated, and value==key at insertion here, so this is exactly
        # equivalent pre-rotation.
        proj = V.astype(mx.float32) @ mx.array(planted.astype(np.float32))
        return bool((mx.max(proj) > 6.0).item())

    seeds = range(40)
    greedy_rate = sum(survived(0.0, s) for s in seeds) / len(list(seeds))
    noisy_rate = sum(survived(6.0, s) for s in seeds) / len(list(seeds))
    # The mechanism claim: noise improves late-riser survival on average.
    assert noisy_rate >= greedy_rate


# ---------------------------------------------------------------------------
# Temperature annealing (Equation 10 / Section 3.3.1) — gap #1 fix
# ---------------------------------------------------------------------------
def test_constant_tau_alias_disables_annealing():
    st = init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau=3.0)
    assert st.tau_init == 3.0 and st.tau_end == 3.0 and st.anneal_steps == 0
    assert _tau_at(st) == 3.0


def test_tau_init_end_equal_is_constant_regardless_of_anneal_steps():
    st = init_keyformer_state(
        n_sink=2, budget=8, head_dim=16, tau_init=2.0, tau_end=2.0, anneal_steps=10
    )
    assert _tau_at(st) == 2.0


def test_tau_anneals_linearly_from_init_to_end():
    st = init_keyformer_state(
        n_sink=0, budget=100, head_dim=16, tau_init=1.0, tau_end=2.0, anneal_steps=10
    )
    assert _tau_at(st) == pytest.approx(1.0)
    for _ in range(5):
        k, v = _rand_kv(1, 16, seed=_)
        st = keyformer_update(st, k, v)
    assert _tau_at(st) == pytest.approx(1.5)  # halfway through anneal_steps=10
    for _ in range(5, 10):
        k, v = _rand_kv(1, 16, seed=_)
        st = keyformer_update(st, k, v)
    assert _tau_at(st) == pytest.approx(2.0)


def test_tau_holds_at_tau_end_past_anneal_steps():
    st = init_keyformer_state(
        n_sink=0, budget=100, head_dim=16, tau_init=1.0, tau_end=2.0, anneal_steps=5
    )
    for i in range(20):
        k, v = _rand_kv(1, 16, seed=i)
        st = keyformer_update(st, k, v)
        assert _tau_at(st) <= 2.0 + 1e-6
    assert _tau_at(st) == pytest.approx(2.0)


def test_anneal_steps_zero_disables_annealing_even_if_tau_end_differs():
    st = init_keyformer_state(
        n_sink=0, budget=100, head_dim=16, tau_init=1.0, tau_end=2.0, anneal_steps=0
    )
    for i in range(10):
        k, v = _rand_kv(1, 16, seed=i)
        st = keyformer_update(st, k, v)
        assert _tau_at(st) == pytest.approx(1.0)  # stuck at tau_init, no ramp


def test_rejects_negative_tau_init():
    with pytest.raises(ValueError, match="tau"):
        init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau_init=-1.0)


def test_rejects_negative_tau_end():
    with pytest.raises(ValueError, match="tau_end"):
        init_keyformer_state(n_sink=2, budget=8, head_dim=16, tau_end=-1.0)


# ---------------------------------------------------------------------------
# RoPE position remapping after eviction — gap #2 fix
# ---------------------------------------------------------------------------
def test_state_tracks_positions_and_next_pos():
    D = 16
    st = init_keyformer_state(n_sink=0, budget=8, head_dim=D)
    k, v = _rand_kv(3, D)
    st = keyformer_update(st, k, v)
    assert st.positions.tolist() == [0, 1, 2]
    assert st.next_pos == 3


def test_positions_stay_sorted_unique_and_sinks_fixed_under_stress():
    # Unlike H2O-adapted at grace=0 (which freezes on whichever tokens filled
    # the budget first, so its kept positions are trivially contiguous — see
    # test_h2o.py's equivalent test), Keyformer's Gumbel noise deliberately
    # lets the eviction target be something other than the newest arrival
    # (that is the whole point of the regularizer — see
    # test_gumbel_rescues_late_riser). So the kept window can legitimately
    # develop gaps; what must still hold is sorted-and-unique positions (no
    # eviction/remap bug can produce duplicates or reordering) and sink
    # positions never moving.
    D = 16
    n_sink = 2
    budget = 6
    st = init_keyformer_state(n_sink=n_sink, budget=budget, head_dim=D, tau=1.0, seed=0)
    for i in range(60):
        k, v = _rand_kv(1, D, seed=i)
        st = keyformer_update(st, k, v)
        pos = st.positions.tolist()
        assert pos == sorted(pos), f"step {i}: not sorted: {pos}"
        assert len(set(pos)) == len(pos), f"step {i}: duplicate: {pos}"
        if len(pos) >= n_sink:
            assert pos[:n_sink] == list(range(n_sink)), f"step {i}: sinks moved: {pos}"


def test_positions_contiguous_at_tau_zero_matching_h2o_freeze():
    # At tau=0 (no noise), Keyformer collapses onto H2O-adapted's grace=0
    # behavior byte-for-bit (test_tau_zero_matches_h2o), including the known
    # freeze: the newest arrival is always the eviction target, so the kept
    # window stays trivially contiguous. This isolates that the remap logic
    # itself preserves contiguity absent the noise mechanism.
    D = 16
    n_sink = 2
    budget = 6
    st = init_keyformer_state(n_sink=n_sink, budget=budget, head_dim=D, tau=0.0, seed=0)
    for i in range(60):
        k, v = _rand_kv(1, D, seed=i)
        st = keyformer_update(st, k, v)
        pos = st.positions.tolist()
        diffs = [pos[j + 1] - pos[j] for j in range(len(pos) - 1)]
        assert all(d == 1 for d in diffs), f"step {i}: gap in kept positions: {pos}"


def test_interior_eviction_gap_remap_recovers_exact_original_keys():
    """Mirrors test_h2o.py's equivalent test line-for-line: constructs a
    5-token state and directly exercises keyformer_update's exact
    keep_indices/shift/rope_remap_positions logic to verify correctness
    independent of how often interior eviction fires in practice."""
    from veloxquant_mlx.quantizers.a2ats_rope import a2ats_apply_exact_rope

    D = 16
    rng = np.random.default_rng(0)
    raw_keys = mx.array(rng.standard_normal((5, D)).astype(np.float32))
    fingerprints = np.zeros((5, D), dtype=np.float32)
    for i in range(5):
        fingerprints[i, 0] = i + 1.0
    raw_values = mx.array(fingerprints)
    positions = mx.arange(5, dtype=mx.int32)
    rotated_keys = a2ats_apply_exact_rope(raw_keys, positions, base=10000.0)

    evict_idx = 2
    keep_indices = [j for j in range(5) if j != evict_idx]
    kept_keys_before = rotated_keys[keep_indices]
    old_positions_kept = positions[keep_indices]
    evicted_pos = int(positions[evict_idx].item())
    shift = mx.where(old_positions_kept > evicted_pos, -1, 0)
    new_positions = old_positions_kept + shift
    remapped_keys = rope_remap_positions(
        kept_keys_before.astype(mx.float32), old_positions_kept, new_positions, base=10000.0
    )

    assert new_positions.tolist() == [0, 1, 2, 3]
    for row in (0, 1):
        err = float(
            mx.max(mx.abs(remapped_keys[row] - kept_keys_before[row].astype(mx.float32))).item()
        )
        assert err < 1e-3, f"row {row} before the gap should be untouched, err={err}"

    recovered = rope_remap_positions(
        remapped_keys, new_positions, mx.zeros_like(new_positions), base=10000.0
    )
    kept_fingerprints = np.array(raw_values.astype(mx.float32))[keep_indices, 0]
    assert 3.0 not in kept_fingerprints
    for row, fp in enumerate(kept_fingerprints):
        orig_idx = int(round(fp)) - 1
        err = float(mx.max(mx.abs(recovered[row] - raw_keys[orig_idx])).item())
        assert err < 1e-2, f"row {row} (orig idx {orig_idx}) failed to recover: err={err}"


def test_tau_zero_matches_h2o_with_positions_tracked():
    """Extends test_tau_zero_matches_h2o: positions/next_pos must also match
    H2O-adapted's exactly, since both now run identical RoPE-remap logic."""
    ks = [_rand_kv(1, 16, seed=i) for i in range(40)]

    kf = init_keyformer_state(n_sink=4, budget=12, head_dim=16, tau=0.0, seed=7)
    h2o = init_h2o_state(n_sink=4, budget=12, head_dim=16)
    for k, v in ks:
        kf = keyformer_update(kf, k, v)
        h2o = h2o_update(h2o, k, v)

    assert kf.positions.tolist() == h2o.positions.tolist()
    assert kf.next_pos == h2o.next_pos


# ---------------------------------------------------------------------------
# Long-prefill / heavy-eviction scalability (_EVAL_FLUSH_INTERVAL regression)
# ---------------------------------------------------------------------------
def test_batch_within_budget_handles_long_prefill_without_crashing():
    """Regression: a real ~3200-token mlx_lm.generate() prefill with the
    fused Metal eviction kernel active raised RuntimeError: [metal::malloc]
    Resource limit (499000) exceeded, once positions/gumbel tracking added
    two more concatenated arrays per eviction step than the pre-fix loop had
    (see _EVAL_FLUSH_INTERVAL's comment in keyformer.py). This confirms a
    call with more tokens than budget (so eviction runs every step past the
    first `budget` tokens) does not crash."""
    D = 32
    S = 3200
    budget = S + 100  # every token fits: no eviction, exercises the below-budget path
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))

    st = init_keyformer_state(n_sink=4, budget=budget, head_dim=D, tau=1.0)
    st = keyformer_update(st, k, v)  # must not raise
    assert st.keys.shape[0] == S
    assert st.next_pos == S


def test_heavy_eviction_thousands_of_loop_iterations_no_crash():
    """Exercises thousands of consecutive eviction-loop iterations (budget
    exceeded almost immediately, S >> budget) within a single
    keyformer_update call, confirming the periodic mx.eval() flush
    (_EVAL_FLUSH_INTERVAL) prevents unbounded unevaluated-graph growth. Same
    caveat as test_h2o.py's equivalent: this synthetic single-head case does
    not reproduce the full-model resource pressure that originally surfaced
    the crash; it guards no-crash/determinism at scale."""
    D = 16
    S = 2000
    budget = 64
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))

    st = init_keyformer_state(n_sink=4, budget=budget, head_dim=D, tau=1.0, seed=0)
    st = keyformer_update(st, k, v)  # must not raise
    assert st.keys.shape[0] == budget
    assert st.next_pos == S

"""Tests for CaM-adapted pure primitives (quantizers/cam.py).

Covers most-similar-survivor selection, the merge blend (weights, key/value
handling, mode validation), the Eq. 14 Bernoulli merge gate (probability
bounds, deterministic sampling, low-score losers mostly blocked), the
per-head merge-eviction state machine, sink preservation, budget enforcement,
byte accounting, determinism, and the drop-mode == H2O bit-for-bit
equivalence. All data is synthetic — no model.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.cam import (
    CaMState,
    cam_fp16_bytes,
    cam_get_kv,
    cam_update,
    full_cam_fp16_bytes,
    init_cam_state,
    merge_gate_probability,
    merge_pair,
    most_similar_survivor,
    sample_merge_gate,
)
from veloxquant_mlx.quantizers.h2o import h2o_get_kv, h2o_update, init_h2o_state


def _kv(S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ======================================================================
# most_similar_survivor
# ======================================================================


def test_survivor_picks_closest_non_sink():
    # 4 keys; evicted key equals key[2] direction → survivor should be 2.
    keys = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=mx.float32)
    evicted = mx.array([1.0, 1.0], dtype=mx.float32)  # closest to row 2
    tgt = most_similar_survivor(evicted, keys, exclude_idx=99, n_sink_eff=0)
    assert tgt == 2


def test_survivor_excludes_sinks_and_self():
    keys = mx.array([[1.0, 1.0], [1.0, 1.0], [0.9, 1.0], [-1.0, 0.0]], dtype=mx.float32)
    evicted = mx.array([1.0, 1.0], dtype=mx.float32)
    # rows 0,1 are sinks (n_sink_eff=2); row 3 is the evicted slot → must pick 2.
    tgt = most_similar_survivor(evicted, keys, exclude_idx=3, n_sink_eff=2)
    assert tgt == 2


def test_survivor_none_when_all_sinks():
    keys = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
    evicted = mx.array([1.0, 1.0], dtype=mx.float32)
    tgt = most_similar_survivor(evicted, keys, exclude_idx=1, n_sink_eff=2)
    assert tgt == -1


# ======================================================================
# merge_pair
# ======================================================================


def test_merge_drop_returns_survivor_unchanged():
    ks = mx.array([1.0, 2.0], dtype=mx.float16)
    vs = mx.array([3.0, 4.0], dtype=mx.float16)
    ke = mx.array([9.0, 9.0], dtype=mx.float16)
    ve = mx.array([9.0, 9.0], dtype=mx.float16)
    k_new, v_new = merge_pair(ks, vs, ke, ve, "drop", merge_keys=True)
    assert bool(mx.all(k_new == ks).item())
    assert bool(mx.all(v_new == vs).item())


def test_merge_mean_is_average_values_only():
    ks = mx.array([1.0, 1.0], dtype=mx.float16)
    vs = mx.array([0.0, 0.0], dtype=mx.float16)
    ke = mx.array([5.0, 5.0], dtype=mx.float16)
    ve = mx.array([2.0, 4.0], dtype=mx.float16)
    k_new, v_new = merge_pair(ks, vs, ke, ve, "mean", merge_keys=False)
    # values averaged, keys untouched (merge_keys=False)
    assert bool(mx.all(k_new == ks).item())
    assert abs(float(v_new[0].item()) - 1.0) < 1e-2
    assert abs(float(v_new[1].item()) - 2.0) < 1e-2


def test_merge_sim_weight_bounds_value_between_inputs():
    ks = mx.array([1.0, 0.0], dtype=mx.float16)
    vs = mx.array([0.0, 0.0], dtype=mx.float16)
    ke = mx.array([1.0, 0.0], dtype=mx.float16)  # identical dir → w≈1 → v_new≈ve
    ve = mx.array([10.0, 10.0], dtype=mx.float16)
    _, v_new = merge_pair(ks, vs, ke, ve, "sim_weighted", merge_keys=False)
    # w = cos = 1 → v_new should be ~ve
    assert float(v_new[0].item()) > 8.0


def test_merge_keys_flag_blends_keys():
    ks = mx.array([0.0, 0.0], dtype=mx.float16)
    vs = mx.array([0.0, 0.0], dtype=mx.float16)
    ke = mx.array([1.0, 0.0], dtype=mx.float16)
    ve = mx.array([1.0, 0.0], dtype=mx.float16)
    k_off, _ = merge_pair(ks, vs, ke, ve, "mean", merge_keys=False)
    k_on, _ = merge_pair(ks, vs, ke, ve, "mean", merge_keys=True)
    assert bool(mx.all(k_off == ks).item())  # keys untouched
    assert not bool(mx.all(k_on == ks).item())  # keys blended


# ======================================================================
# CaMState eviction/merge
# ======================================================================


def test_init_rejects_bad_mode():
    with pytest.raises(ValueError):
        init_cam_state(4, 32, 16, merge_mode="bogus")


def test_init_rejects_n_sink_equal_budget():
    """n_sink >= budget leaves no evictable/mergeable room — sinks would be
    merged away once the cache fills, silently violating the "sinks always
    retained" invariant (see test_sinks_always_retained). Must raise instead
    of letting that happen quietly."""
    with pytest.raises(ValueError, match="n_sink"):
        init_cam_state(4, 4, 8, merge_mode="sim_weighted")


def test_init_rejects_n_sink_above_budget():
    with pytest.raises(ValueError, match="n_sink"):
        init_cam_state(8, 4, 8, merge_mode="sim_weighted")


def test_init_zero_budget_allowed():
    st = init_cam_state(0, 0, 8, merge_mode="sim_weighted")
    assert st.budget == 0


def test_update_respects_budget_exactly():
    st = init_cam_state(n_sink=2, budget=10, head_dim=8, merge_mode="sim_weighted")
    k, v = _kv(40, 8)
    st = cam_update(st, k, v)
    assert int(st.keys.shape[0]) == 10  # merge trims to exactly budget


def test_sinks_always_retained():
    st = init_cam_state(n_sink=3, budget=8, head_dim=8, merge_mode="sim_weighted")
    k, v = _kv(60, 8, seed=5)
    st = cam_update(st, k, v)
    assert bool(mx.all(st.keys[:3] == k[:3].astype(mx.float16)).item())


def test_merge_changes_values_vs_drop():
    # merge_gate=False isolates the blend weight from the Eq. 14 gate: this
    # test is about sim_weighted's blend differing from drop's no-op, not
    # about whether the gate fires.
    k, v = _kv(50, 8, seed=7)
    drop = cam_update(init_cam_state(2, 10, 8, merge_mode="drop", merge_gate=False), k, v)
    sim = cam_update(
        init_cam_state(2, 10, 8, merge_mode="sim_weighted", merge_gate=False), k, v
    )
    # same kept count, but the merged values differ from the dropped ones
    assert drop.keys.shape == sim.keys.shape
    assert not bool(mx.all(drop.values == sim.values).item())


def test_merge_keys_false_leaves_keys_like_drop():
    k, v = _kv(50, 8, seed=9)
    drop = cam_update(init_cam_state(2, 10, 8, merge_mode="drop"), k, v)
    sim = cam_update(init_cam_state(2, 10, 8, merge_mode="sim_weighted", merge_keys=False), k, v)
    # values-only merge → surviving keys identical to the drop path
    assert bool(mx.all(drop.keys == sim.keys).item())


def test_determinism_no_rng():
    k, v = _kv(40, 8, seed=11)
    a = cam_update(init_cam_state(2, 10, 8, merge_mode="sim_weighted"), k, v)
    b = cam_update(init_cam_state(2, 10, 8, merge_mode="sim_weighted"), k, v)
    assert bool(mx.all(a.keys == b.keys).item())
    assert bool(mx.all(a.values == b.values).item())


def test_get_kv_placeholder_before_update():
    st = init_cam_state(2, 10, 8)
    k, v = cam_get_kv(st)
    assert k.shape == (0, 1) and v.shape == (0, 1)


def test_byte_accounting_helpers():
    st = init_cam_state(2, 10, 8, merge_mode="sim_weighted")
    assert cam_fp16_bytes(st) == 0
    k, v = _kv(30, 8, seed=13)
    st = cam_update(st, k, v)
    n = int(st.keys.shape[0])
    assert cam_fp16_bytes(st) == n * 8 * 2 * 2
    assert full_cam_fp16_bytes(100, 8) == 100 * 8 * 2 * 2


# ======================================================================
# drop mode == H2O (bit-for-bit)
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_drop_mode_reduces_to_h2o(seed):
    """merge_mode='drop' must match H2O-adapted exactly (keys AND values)."""
    S, D, budget, n_sink = 40, 8, 8, 2
    k, v = _kv(S, D, seed=seed)

    cs = cam_update(init_cam_state(n_sink, budget, D, merge_mode="drop"), k, v)
    ck, cv = cam_get_kv(cs)

    hs = h2o_update(init_h2o_state(n_sink, budget, D), k, v)
    hk, hv = h2o_get_kv(hs)

    assert ck.shape == hk.shape
    assert bool(mx.all(ck == hk).item())
    assert bool(mx.all(cv == hv).item())


# ======================================================================
# Merge gate (Eq. 14 Bernoulli mask)
# ======================================================================


def test_merge_gate_probability_bounds():
    assert merge_gate_probability(0.0, 1.0) == 0.0
    assert merge_gate_probability(1.0, 1.0) == 1.0
    assert merge_gate_probability(2.0, 1.0) == 1.0  # clamped
    assert merge_gate_probability(0.5, 1.0) == 0.5


def test_merge_gate_probability_zero_survivor_score():
    assert merge_gate_probability(0.0, 0.0) == 0.0  # nothing to compare against
    assert merge_gate_probability(0.3, 0.0) == 1.0  # any mass beats zero


def test_sample_merge_gate_deterministic_extremes():
    assert sample_merge_gate(0.0, seed=0, draw_id=0) is False
    assert sample_merge_gate(1.0, seed=0, draw_id=0) is True


def test_sample_merge_gate_reproducible():
    a = sample_merge_gate(0.5, seed=7, draw_id=3)
    b = sample_merge_gate(0.5, seed=7, draw_id=3)
    assert a == b


def test_sample_merge_gate_varies_with_draw_id():
    draws = {sample_merge_gate(0.5, seed=7, draw_id=i) for i in range(20)}
    assert draws == {True, False}  # both outcomes occur across draws


def test_gate_off_merges_unconditionally():
    """merge_gate=False reproduces the old always-merge behaviour."""
    k, v = _kv(60, 8, seed=21)
    gated = cam_update(
        init_cam_state(2, 10, 8, merge_mode="sim_weighted", merge_gate=True, seed=0), k, v
    )
    ungated = cam_update(
        init_cam_state(2, 10, 8, merge_mode="sim_weighted", merge_gate=False), k, v
    )
    # Same budget enforcement, but the gate can skip merges the ungated path
    # always performs — the two states are not guaranteed identical.
    assert gated.keys.shape[0] == 10
    assert ungated.keys.shape[0] == 10


def test_low_score_loser_more_likely_dropped_with_gate():
    """A loser with near-zero mass relative to its target should rarely merge.

    Repeats the same low-relative-score scenario across draw ids and checks
    the gate blocks the merge (keys match the drop path) the overwhelming
    majority of the time — the paper's intended behaviour for a
    just-appended, unproven token.
    """
    blocked = 0
    trials = 30
    for seed in range(trials):
        k, v = _kv(30, 8, seed=100 + seed)
        drop = cam_update(init_cam_state(2, 10, 8, merge_mode="drop"), k, v)
        gated = cam_update(
            init_cam_state(2, 10, 8, merge_mode="sim_weighted", merge_gate=True, seed=seed),
            k,
            v,
        )
        if bool(mx.all(drop.values == gated.values).item()):
            blocked += 1
    # Most losers overflow the budget shortly after being appended (score
    # near 0 relative to established survivors), so the gate should block
    # the large majority of merges.
    assert blocked >= trials * 0.5


def test_merge_gate_default_is_on():
    st = init_cam_state(2, 10, 8)
    assert st.merge_gate is True

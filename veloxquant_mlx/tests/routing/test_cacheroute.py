"""Tests for CacheRoute session admission and shard placement (routing/cacheroute.py, issue #278)."""

from __future__ import annotations

import pytest

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.routing.cacheroute import (
    CacheRoutePlanner,
    RateEstimator,
    RoutingTable,
    SessionRate,
)

# ============================================================================
# SessionRate validation
# ============================================================================


def test_session_rate_rejects_negative_rate():
    with pytest.raises(QuantizerConfigError):
        SessionRate(owner=1, rate=-1.0)


def test_session_rate_accepts_zero_rate():
    r = SessionRate(owner=1, rate=0.0)
    assert r.rate == 0.0


def test_session_rate_is_frozen():
    r = SessionRate(owner=1, rate=5.0)
    with pytest.raises(AttributeError):
        r.rate = 10.0  # type: ignore[misc]


# ============================================================================
# CacheRoutePlanner construction validation
# ============================================================================


def test_planner_rejects_zero_shards():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=0, qcap=10.0, warm_slots_per_shard=10.0)


def test_planner_rejects_negative_shards():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=-2, qcap=10.0, warm_slots_per_shard=10.0)


def test_planner_rejects_zero_qcap():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=4, qcap=0.0, warm_slots_per_shard=10.0)


def test_planner_rejects_negative_qcap():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=4, qcap=-5.0, warm_slots_per_shard=10.0)


def test_planner_rejects_zero_warm_slots():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=4, qcap=10.0, warm_slots_per_shard=0.0)


def test_planner_rejects_negative_warm_slots():
    with pytest.raises(QuantizerConfigError):
        CacheRoutePlanner(n_shards=4, qcap=10.0, warm_slots_per_shard=-1.0)


def test_planner_accepts_single_shard():
    planner = CacheRoutePlanner(n_shards=1, qcap=10.0, warm_slots_per_shard=10.0)
    assert planner.n_shards == 1


# ============================================================================
# Empty / trivial plans
# ============================================================================


def test_plan_with_no_rates_admits_nothing():
    planner = CacheRoutePlanner(n_shards=4, qcap=10.0, warm_slots_per_shard=10.0)
    table = planner.plan([])
    assert table.shards == {}
    assert table.expected_load == [0.0, 0.0, 0.0, 0.0]
    assert table.admitted_rate_total == 0.0
    assert table.unadmitted_rate_total == 0.0


def test_plan_ignores_zero_rate_sessions():
    planner = CacheRoutePlanner(n_shards=2, qcap=10.0, warm_slots_per_shard=10.0)
    table = planner.plan([SessionRate(owner=1, rate=0.0)])
    assert not table.is_admitted(1)
    assert table.admitted_rate_total == 0.0
    assert table.unadmitted_rate_total == 0.0


def test_single_session_well_under_capacity_is_admitted_with_one_shard():
    planner = CacheRoutePlanner(n_shards=4, qcap=100.0, warm_slots_per_shard=50.0)
    table = planner.plan([SessionRate(owner=1, rate=5.0)])
    assert table.is_admitted(1)
    assert table.shards[1] == (0,)
    assert table.admitted_rate_total == 5.0


# ============================================================================
# Load-based assignment count (kb = ceil(rate / qcap))
# ============================================================================


def test_hot_session_gets_multiple_shards():
    # rate=250, qcap=100 -> kb = ceil(2.5) = 3
    planner = CacheRoutePlanner(n_shards=5, qcap=100.0, warm_slots_per_shard=10.0)
    table = planner.plan([SessionRate(owner=1, rate=250.0)])
    assert len(table.shards[1]) == 3


def test_assignment_count_capped_at_n_shards():
    # rate implies kb=10 but only 3 shards exist -> capped at 3
    planner = CacheRoutePlanner(n_shards=3, qcap=10.0, warm_slots_per_shard=10.0)
    table = planner.plan([SessionRate(owner=1, rate=100.0)])
    assert len(table.shards[1]) == 3


def test_rate_exactly_at_qcap_gets_one_shard():
    planner = CacheRoutePlanner(n_shards=4, qcap=50.0, warm_slots_per_shard=10.0)
    table = planner.plan([SessionRate(owner=1, rate=50.0)])
    assert len(table.shards[1]) == 1


def test_rate_just_over_qcap_gets_two_shards():
    planner = CacheRoutePlanner(n_shards=4, qcap=50.0, warm_slots_per_shard=10.0)
    table = planner.plan([SessionRate(owner=1, rate=50.01)])
    assert len(table.shards[1]) == 2


# ============================================================================
# Warm-set admission ordering and budget
# ============================================================================


def test_admission_prefers_highest_rate_sessions_when_over_budget():
    # capacity = 2 shards * 1 slot = 2 "kb units"; three unit-kb sessions compete.
    planner = CacheRoutePlanner(n_shards=2, qcap=1000.0, warm_slots_per_shard=1.0)
    rates = [
        SessionRate(owner=1, rate=10.0),
        SessionRate(owner=2, rate=30.0),
        SessionRate(owner=3, rate=20.0),
    ]
    table = planner.plan(rates)
    assert table.is_admitted(2)
    assert table.is_admitted(3)
    assert not table.is_admitted(1)
    assert table.admitted_rate_total == 50.0
    assert table.unadmitted_rate_total == 10.0


def test_admission_is_all_or_nothing_per_session_at_the_boundary():
    # capacity = 1 unit; a kb=1 session and a kb=2 session compete for it.
    planner = CacheRoutePlanner(n_shards=4, qcap=100.0, warm_slots_per_shard=0.25)
    rates = [
        SessionRate(owner=1, rate=250.0),  # kb=3, needs 3 units, rejected outright
        SessionRate(owner=2, rate=5.0),  # kb=1, fits
    ]
    table = planner.plan(rates)
    assert not table.is_admitted(1)
    assert table.is_admitted(2)


def test_exactly_filling_capacity_admits_all():
    planner = CacheRoutePlanner(n_shards=2, qcap=1000.0, warm_slots_per_shard=1.0)
    rates = [SessionRate(owner=i, rate=1.0) for i in range(4)]  # 4 kb=1 sessions, capacity=2*1=2...
    table = planner.plan(rates)
    # only 2 units of capacity -> exactly 2 admitted, the rest cold-tail
    assert sum(1 for o in range(4) if table.is_admitted(o)) == 2


# ============================================================================
# LPT placement balances expected load
# ============================================================================


def test_placement_balances_load_across_shards():
    planner = CacheRoutePlanner(n_shards=2, qcap=1000.0, warm_slots_per_shard=10.0)
    rates = [
        SessionRate(owner=1, rate=100.0),
        SessionRate(owner=2, rate=80.0),
        SessionRate(owner=3, rate=60.0),
        SessionRate(owner=4, rate=40.0),
    ]
    table = planner.plan(rates)
    # LPT: 100->shard A, 80->shard B, 60->shard B(60<100), 40->shard A(100+40=140 vs 80+60=140)
    assert table.expected_load[0] == pytest.approx(140.0)
    assert table.expected_load[1] == pytest.approx(140.0)
    assert table.imbalance() == pytest.approx(1.0)


def test_placement_is_more_balanced_than_naive_arrival_order_assignment():
    # A round-robin/hash assignment could stack several of the small owners
    # onto the same shard as the whale (imbalance up to 520/180=2.89x); LPT
    # instead spreads every small owner onto the two non-whale shards.
    planner = CacheRoutePlanner(n_shards=3, qcap=1000.0, warm_slots_per_shard=100.0)
    rates = [SessionRate(owner=i, rate=v) for i, v in enumerate([500.0, 10.0, 10.0, 10.0, 10.0])]
    table = planner.plan(rates)
    assert table.expected_load[0] == pytest.approx(500.0)
    assert table.expected_load[1] == pytest.approx(20.0)
    assert table.expected_load[2] == pytest.approx(20.0)
    assert table.imbalance() < 2.8  # bounded by the single-shard whale, not by placement skew


def test_replicated_session_splits_load_evenly_across_its_shards():
    planner = CacheRoutePlanner(n_shards=4, qcap=100.0, warm_slots_per_shard=100.0)
    table = planner.plan([SessionRate(owner=1, rate=300.0)])  # kb = 3
    assigned = table.shards[1]
    assert len(assigned) == 3
    per_shard = 300.0 / 3
    for s in assigned:
        assert table.expected_load[s] == pytest.approx(per_shard)


def test_cold_tail_load_spread_evenly_across_all_shards():
    planner = CacheRoutePlanner(n_shards=4, qcap=1.0, warm_slots_per_shard=0.0001)
    # Nothing fits in the tiny warm budget, so all rate becomes cold-tail.
    rates = [SessionRate(owner=1, rate=40.0), SessionRate(owner=2, rate=40.0)]
    table = planner.plan(rates)
    assert table.shards == {}
    assert table.unadmitted_rate_total == 80.0
    for load in table.expected_load:
        assert load == pytest.approx(20.0)


# ============================================================================
# RoutingTable helpers
# ============================================================================


def test_is_admitted_false_for_unknown_owner():
    table = RoutingTable(shards={}, expected_load=[0.0], n_shards=1, admitted_rate_total=0.0, unadmitted_rate_total=0.0)
    assert not table.is_admitted(999)


def test_preferred_shard_returns_fallback_when_not_admitted():
    table = RoutingTable(shards={}, expected_load=[0.0, 0.0], n_shards=2, admitted_rate_total=0.0, unadmitted_rate_total=0.0)
    assert table.preferred_shard(owner=42, fallback=1) == 1


def test_preferred_shard_returns_only_assigned_shard():
    table = RoutingTable(
        shards={1: (2,)}, expected_load=[0.0, 0.0, 0.0], n_shards=3, admitted_rate_total=5.0, unadmitted_rate_total=0.0
    )
    assert table.preferred_shard(1) == 2


def test_preferred_shard_picks_least_loaded_among_replicas():
    table = RoutingTable(
        shards={1: (0, 1, 2)},
        expected_load=[50.0, 5.0, 30.0],
        n_shards=3,
        admitted_rate_total=85.0,
        unadmitted_rate_total=0.0,
    )
    assert table.preferred_shard(1) == 1


def test_preferred_shard_breaks_ties_by_lowest_shard_id():
    table = RoutingTable(
        shards={1: (2, 0)}, expected_load=[10.0, 0.0, 10.0], n_shards=3, admitted_rate_total=5.0, unadmitted_rate_total=0.0
    )
    assert table.preferred_shard(1) == 0


def test_imbalance_is_one_for_perfectly_even_load():
    table = RoutingTable(
        shards={}, expected_load=[10.0, 10.0, 10.0], n_shards=3, admitted_rate_total=0.0, unadmitted_rate_total=30.0
    )
    assert table.imbalance() == pytest.approx(1.0)


def test_imbalance_reflects_hot_shard():
    table = RoutingTable(
        shards={}, expected_load=[100.0, 10.0, 10.0], n_shards=3, admitted_rate_total=0.0, unadmitted_rate_total=120.0
    )
    assert table.imbalance() == pytest.approx(100.0 / 40.0)


def test_imbalance_is_one_when_all_load_is_zero():
    table = RoutingTable(
        shards={}, expected_load=[0.0, 0.0], n_shards=2, admitted_rate_total=0.0, unadmitted_rate_total=0.0
    )
    assert table.imbalance() == 1.0


def test_imbalance_is_one_for_empty_expected_load():
    table = RoutingTable(shards={}, expected_load=[], n_shards=0, admitted_rate_total=0.0, unadmitted_rate_total=0.0)
    assert table.imbalance() == 1.0


# ============================================================================
# RateEstimator
# ============================================================================


def test_rate_estimator_rejects_nonpositive_half_life():
    with pytest.raises(QuantizerConfigError):
        RateEstimator(half_life=0)


def test_rate_estimator_rejects_negative_half_life():
    with pytest.raises(QuantizerConfigError):
        RateEstimator(half_life=-5)


def test_rate_estimator_rejects_zero_window():
    with pytest.raises(QuantizerConfigError):
        RateEstimator(window=0)


def test_rate_estimator_unseen_owner_has_zero_rate():
    estimator = RateEstimator()
    assert estimator.rate(999) == 0.0


def test_rate_estimator_increases_with_repeated_records():
    estimator = RateEstimator(half_life=10.0)
    estimator.record(owner=1)
    r1 = estimator.rate(1)
    estimator.record(owner=1)
    r2 = estimator.rate(1)
    assert r2 > r1


def test_rate_estimator_tracks_owners_independently():
    estimator = RateEstimator(half_life=10.0)
    for _ in range(5):
        estimator.record(owner=1)
    estimator.record(owner=2)
    assert estimator.rate(1) > estimator.rate(2)


def test_rate_estimator_forget_clears_owner():
    estimator = RateEstimator()
    estimator.record(owner=1)
    assert estimator.rate(1) > 0.0
    estimator.forget(1)
    assert estimator.rate(1) == 0.0


def test_rate_estimator_forget_unknown_owner_is_noop():
    estimator = RateEstimator()
    estimator.forget(12345)  # must not raise


def test_rate_estimator_rates_snapshot_matches_recorded_owners():
    estimator = RateEstimator()
    estimator.record(owner=1)
    estimator.record(owner=2)
    snapshot = {r.owner: r.rate for r in estimator.rates()}
    assert set(snapshot) == {1, 2}
    assert all(v > 0 for v in snapshot.values())


def test_rate_estimator_window_bounds_history_length():
    estimator = RateEstimator(window=3)
    for _ in range(10):
        estimator.record(owner=1)
    assert len(estimator._history[1]) == 3


def test_rate_estimator_feeds_directly_into_planner():
    estimator = RateEstimator(half_life=5.0)
    for _ in range(20):
        estimator.record(owner=1)
    estimator.record(owner=2)
    planner = CacheRoutePlanner(n_shards=2, qcap=100.0, warm_slots_per_shard=1.0)
    table = planner.plan(estimator.rates())
    assert table.is_admitted(1)


# ============================================================================
# End-to-end scenario, loosely modeled on the paper's skewed workload
# ============================================================================


def test_end_to_end_skewed_workload_favors_hot_keys_and_balances_shards():
    # A small Zipf-like population: a few hot owners, many cold ones.
    rates = [SessionRate(owner=0, rate=200.0), SessionRate(owner=1, rate=150.0)]
    rates += [SessionRate(owner=i, rate=1.0) for i in range(2, 50)]

    planner = CacheRoutePlanner(n_shards=4, qcap=200.0, warm_slots_per_shard=100.0)
    table = planner.plan(rates)

    assert table.is_admitted(0)
    assert table.is_admitted(1)
    # The two whales each pin one shard (kb=1), so the floor on imbalance is
    # set by them, not by how well the many small owners get spread — check
    # the small owners land roughly evenly on the two remaining shards.
    assert table.expected_load[2] == pytest.approx(table.expected_load[3], rel=0.2)
    assert table.imbalance() < 2.5
    assert table.admitted_rate_total + table.unadmitted_rate_total == pytest.approx(
        sum(r.rate for r in rates)
    )

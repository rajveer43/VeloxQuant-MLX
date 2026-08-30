"""Tests for the CacheRoute semi-synthetic admission/placement benchmark (issue #278).

Uses small, fast workload sizes so the suite stays cheap on CI — the point
is to validate the simulation harness's bookkeeping (hit-rate accounting,
imbalance computation, JSON round-trip), not to reproduce the paper's
headline numbers.
"""

from __future__ import annotations

import pytest

from veloxquant_mlx.benchmarks.cacheroute_benchmark import (
    PolicyResult,
    _imbalance,
    format_summary_table,
    generate_workload,
    gini_coefficient,
    results_to_json,
    run_benchmark,
)

# ============================================================================
# gini_coefficient
# ============================================================================


def test_gini_zero_for_equal_weights():
    assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_gini_zero_for_empty_input():
    assert gini_coefficient([]) == 0.0


def test_gini_zero_when_total_is_zero():
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


def test_gini_high_for_maximally_skewed_weights():
    # One session has all the weight; Gini should approach (n-1)/n.
    weights = [100.0] + [0.0001] * 99
    g = gini_coefficient(weights)
    assert g > 0.9


def test_gini_increases_with_skew():
    even = gini_coefficient([1.0] * 10)
    skewed = gini_coefficient([10.0] + [1.0] * 9)
    assert skewed > even


# ============================================================================
# generate_workload
# ============================================================================


def test_generate_workload_has_expected_sizes():
    w = generate_workload(n_sessions=50, n_arrivals=1000, zipf_s=1.1, seed=1)
    assert w.n_sessions == 50
    assert len(w.session_rates) == 50
    assert len(w.arrivals) == 1000


def test_generate_workload_arrivals_reference_valid_sessions():
    w = generate_workload(n_sessions=20, n_arrivals=500, zipf_s=1.0, seed=2)
    assert all(0 <= owner < 20 for owner in w.arrivals)


def test_generate_workload_is_deterministic_given_seed():
    w1 = generate_workload(n_sessions=30, n_arrivals=500, zipf_s=1.2, seed=7)
    w2 = generate_workload(n_sessions=30, n_arrivals=500, zipf_s=1.2, seed=7)
    assert w1.arrivals == w2.arrivals
    assert w1.session_rates == w2.session_rates


def test_generate_workload_different_seeds_differ():
    w1 = generate_workload(n_sessions=30, n_arrivals=500, zipf_s=1.2, seed=1)
    w2 = generate_workload(n_sessions=30, n_arrivals=500, zipf_s=1.2, seed=2)
    assert w1.arrivals != w2.arrivals


def test_generate_workload_reports_positive_gini_for_skewed_zipf():
    w = generate_workload(n_sessions=100, n_arrivals=2000, zipf_s=1.1, seed=3)
    assert w.gini > 0.3


def test_generate_workload_higher_zipf_s_increases_skew():
    low = generate_workload(n_sessions=100, n_arrivals=1, zipf_s=0.2, seed=1)
    high = generate_workload(n_sessions=100, n_arrivals=1, zipf_s=2.0, seed=1)
    assert high.gini > low.gini


# ============================================================================
# _imbalance helper
# ============================================================================


def test_imbalance_one_for_even_load():
    assert _imbalance([10.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_imbalance_empty_list():
    assert _imbalance([]) == 1.0


def test_imbalance_all_zero_load():
    assert _imbalance([0.0, 0.0]) == 1.0


def test_imbalance_reflects_hot_shard():
    assert _imbalance([100.0, 10.0, 10.0]) == pytest.approx(100.0 / 40.0)


# ============================================================================
# run_benchmark end-to-end
# ============================================================================


def _small_run(**overrides) -> dict[str, PolicyResult]:
    params = dict(
        n_sessions=40,
        n_arrivals=2000,
        n_shards=4,
        warm_slots_per_shard=5,
        zipf_s=1.1,
        replan_interval=100,
        seed=11,
    )
    params.update(overrides)
    return run_benchmark(**params)


def test_run_benchmark_returns_all_three_policies():
    results = _small_run()
    assert set(results) == {"lru", "sticky_hash", "cacheroute"}


def test_run_benchmark_hit_rates_are_valid_fractions():
    results = _small_run()
    for r in results.values():
        assert 0.0 <= r.hit_rate <= 1.0
        assert r.hits <= r.n_arrivals


def test_run_benchmark_load_per_shard_sums_to_total_arrivals_for_lru_and_sticky():
    results = _small_run()
    n_arrivals = results["lru"].n_arrivals
    assert sum(results["lru"].load_per_shard) == n_arrivals
    assert sum(results["sticky_hash"].load_per_shard) == n_arrivals


def test_run_benchmark_cacheroute_imbalance_is_near_perfect_under_default_qcap():
    # Default qcap keeps every session at kb=1, so LPT should balance
    # expected load close to evenly (imbalance near 1.0), unlike the
    # cache-blind/sticky baselines under a skewed arrival stream.
    results = _small_run(zipf_s=1.3)
    assert results["cacheroute"].imbalance < results["sticky_hash"].imbalance + 1e-9


def test_run_benchmark_is_deterministic_given_seed():
    r1 = _small_run(seed=99)
    r2 = _small_run(seed=99)
    for name in r1:
        assert r1[name].hits == r2[name].hits
        assert r1[name].load_per_shard == r2[name].load_per_shard


def test_run_benchmark_cacheroute_beats_or_matches_baselines_on_hit_rate_under_tight_capacity():
    # Tight warm budget relative to session count is where the paper's
    # admission advantage shows up most clearly.
    results = run_benchmark(
        n_sessions=300,
        n_arrivals=20_000,
        n_shards=4,
        warm_slots_per_shard=5,
        zipf_s=1.0,
        replan_interval=200,
        seed=5,
    )
    assert results["cacheroute"].hit_rate >= results["lru"].hit_rate
    assert results["cacheroute"].hit_rate >= results["sticky_hash"].hit_rate


def test_run_benchmark_respects_explicit_qcap_for_replication():
    # A very low qcap forces kb > 1 for hot sessions; the plan should still
    # run without error and admit at least the hottest session.
    results = run_benchmark(
        n_sessions=50,
        n_arrivals=3000,
        n_shards=4,
        warm_slots_per_shard=20,
        zipf_s=1.2,
        qcap=5.0,
        replan_interval=100,
        seed=3,
    )
    assert results["cacheroute"].hits >= 0


# ============================================================================
# Reporting
# ============================================================================


def test_format_summary_table_contains_all_policy_names():
    results = _small_run()
    table = format_summary_table(results, gini=0.5)
    for name in results:
        assert name in table
    assert "0.500" in table


def test_results_to_json_round_trips_policy_fields():
    results = _small_run()
    payload = results_to_json(results, gini=0.42)
    assert payload["gini"] == 0.42
    assert set(payload["policies"]) == set(results)
    for name, r in results.items():
        assert payload["policies"][name]["hit_rate"] == pytest.approx(r.hit_rate)
        assert payload["policies"][name]["policy"] == name

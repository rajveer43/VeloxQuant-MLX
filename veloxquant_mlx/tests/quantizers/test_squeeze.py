"""Tests for SqueezeAttention-adapted quantizer primitives — 2D layer×token eviction.

SqueezeAttention-adapted (arXiv:2404.04793) measures each layer's attention
concentration (cosine-dispersion proxy) and reallocates a fixed total budget by
inverse-concentration, then evicts within each layer using H2O's cumulative-
attention-mass scorer. Tests cover: the concentration_score proxy (identical vs
orthogonal keys, edge cases), the squeeze_budgets allocator (strength=0 uniform,
strength=1 inverse-concentration split, mean preservation, sink floor, monotone
w.r.t. concentration, edge cases, strength bounds) and squeeze_update / state
mechanics (bootstrap, budget enforcement, sink protection, byte accounting,
determinism). All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.squeeze import (
    SqueezeState,
    concentration_score,
    full_squeeze_fp16_bytes,
    init_squeeze_state,
    squeeze_budgets,
    squeeze_fp16_bytes,
    squeeze_get_kv,
    squeeze_update,
)


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ======================================================================
# concentration_score
# ======================================================================


def test_concentration_identical_keys_is_one():
    """Rows all pointing the same direction → cosine 1 → max concentration."""
    keys = mx.ones((6, 8))
    assert concentration_score(keys) == pytest.approx(1.0, abs=1e-4)


def test_concentration_orthogonal_keys_is_zero():
    """Mutually orthogonal rows → cosine 0 → min concentration."""
    keys = mx.eye(8)[:6]
    assert concentration_score(keys) == pytest.approx(0.0, abs=1e-4)


def test_concentration_scale_invariant():
    """Concentration depends on direction only, not magnitude."""
    keys = mx.array(np.random.default_rng(1).standard_normal((10, 8)).astype(np.float32))
    c1 = concentration_score(keys)
    c2 = concentration_score(keys * 100.0)
    assert c1 == pytest.approx(c2, abs=1e-4)


def test_concentration_fewer_than_two_rows_is_neutral():
    """0 or 1 rows have no pairs → neutral 0.0 (yields average budget)."""
    assert concentration_score(mx.zeros((0, 8))) == 0.0
    assert concentration_score(mx.ones((1, 8))) == 0.0
    assert concentration_score(None) == 0.0


def test_concentration_in_range():
    """Score always lies in [-1, 1]."""
    for seed in range(5):
        keys = mx.array(np.random.default_rng(seed).standard_normal((12, 16)).astype(np.float32))
        c = concentration_score(keys)
        assert -1.0 <= c <= 1.0


# ======================================================================
# squeeze_budgets — the allocator
# ======================================================================


def test_budgets_length_matches_layers():
    b = squeeze_budgets([0.1, 0.5, 0.9, 0.2], avg_budget=100, n_sink=4, strength=1.0)
    assert len(b) == 4


def test_strength_zero_is_uniform():
    """strength=0 → every layer gets exactly avg_budget (reduces to H2O)."""
    b = squeeze_budgets([0.1, 0.9, 0.5, 0.0, 1.0], avg_budget=256, n_sink=4, strength=0.0)
    assert b == [256, 256, 256, 256, 256]


def test_strength_zero_uniform_regardless_of_concentration():
    """Uniform result is independent of the concentration vector at strength=0."""
    b1 = squeeze_budgets([0.0, 0.0, 0.0], avg_budget=128, n_sink=4, strength=0.0)
    b2 = squeeze_budgets([0.3, 0.7, 0.99], avg_budget=128, n_sink=4, strength=0.0)
    assert b1 == b2 == [128, 128, 128]


def test_mean_preserved_at_full_strength():
    """Reallocation holds the mean ≈ avg_budget (before floor clamping)."""
    conc = [0.1, 0.3, 0.5, 0.7, 0.9]
    b = squeeze_budgets(conc, avg_budget=200, n_sink=4, strength=1.0)
    assert abs(sum(b) / len(b) - 200) <= 200 * 0.05


def test_broad_layer_gets_more_than_concentrated():
    """Low concentration (broad) → larger budget; high concentration → smaller."""
    b = squeeze_budgets([0.1, 0.9], avg_budget=100, n_sink=4, strength=1.0)
    assert b[0] > b[1]


def test_budget_monotone_in_concentration():
    """Sorted-descending concentration → non-increasing budgets."""
    conc = [0.05, 0.25, 0.5, 0.75, 0.95]
    b = squeeze_budgets(conc, avg_budget=300, n_sink=4, strength=1.0)
    assert all(b[i] >= b[i + 1] for i in range(len(b) - 1))


def test_floor_at_n_sink_plus_one():
    """No budget drops below n_sink + 1 even for maximally concentrated layers."""
    b = squeeze_budgets([0.0, 1.0, 1.0, 1.0], avg_budget=50, n_sink=8, strength=1.0)
    assert min(b) >= 9


def test_all_concentrated_falls_back_uniform():
    """If every layer is maximally concentrated, weights degenerate to uniform."""
    b = squeeze_budgets([1.0, 1.0, 1.0], avg_budget=64, n_sink=4, strength=1.0)
    assert b == [64, 64, 64]


def test_negative_concentration_clamped():
    """Negative cosine (very broad) clamps to 0 → max-budget end, no crash."""
    b = squeeze_budgets([-0.5, 0.9], avg_budget=100, n_sink=4, strength=1.0)
    assert b[0] > b[1]


def test_intermediate_strength_between_uniform_and_full():
    """strength=0.5 lands between uniform and the full reallocation."""
    conc = [0.1, 0.9]
    uni = squeeze_budgets(conc, avg_budget=100, n_sink=4, strength=0.0)
    half = squeeze_budgets(conc, avg_budget=100, n_sink=4, strength=0.5)
    full = squeeze_budgets(conc, avg_budget=100, n_sink=4, strength=1.0)
    assert uni[0] <= half[0] <= full[0]
    assert uni[1] >= half[1] >= full[1]


def test_single_layer():
    assert squeeze_budgets([0.5], avg_budget=100, n_sink=4, strength=1.0) == [100]


def test_single_layer_floors():
    assert squeeze_budgets([0.9], avg_budget=2, n_sink=8, strength=1.0) == [9]


def test_empty_layers():
    assert squeeze_budgets([], avg_budget=100, n_sink=4, strength=1.0) == []


def test_strength_out_of_range_raises():
    with pytest.raises(ValueError):
        squeeze_budgets([0.5, 0.5], avg_budget=100, n_sink=4, strength=1.5)
    with pytest.raises(ValueError):
        squeeze_budgets([0.5, 0.5], avg_budget=100, n_sink=4, strength=-0.1)


# ======================================================================
# init_squeeze_state
# ======================================================================


def test_init_state_rejects_n_sink_equal_budget() -> None:
    """n_sink >= budget leaves no evictable room — sinks would be evicted
    once the cache fills, defeating the sink guarantee."""
    with pytest.raises(ValueError, match="n_sink"):
        init_squeeze_state(n_sink=4, budget=4, head_dim=32)


def test_init_state_rejects_n_sink_above_budget() -> None:
    with pytest.raises(ValueError, match="n_sink"):
        init_squeeze_state(n_sink=8, budget=4, head_dim=32)


def test_init_state_allows_disabled_cache() -> None:
    """n_sink=0, budget=0 is a valid 'disabled cache' configuration."""
    st = init_squeeze_state(n_sink=0, budget=0, head_dim=32)
    assert st.n_sink == 0
    assert st.budget == 0


# ======================================================================
# squeeze_update — eviction mechanics (reuses H2O scorer)
# ======================================================================


def test_bootstrap_single_token():
    st = init_squeeze_state(n_sink=2, budget=8, head_dim=32)
    k, v = _rand_kv(1)
    st = squeeze_update(st, k, v)
    assert st.keys.shape[0] == 1
    assert st.scores.shape[0] == 1


def test_budget_never_exceeded():
    """Budget is enforced across a long stream of single-token updates."""
    budget = 10
    st = init_squeeze_state(n_sink=4, budget=budget, head_dim=32)
    for step in range(40):
        k, v = _rand_kv(1, seed=step)
        st = squeeze_update(st, k, v)
        assert st.keys.shape[0] <= budget


def test_budget_plus_one_trims_to_budget():
    budget = 6
    st = init_squeeze_state(n_sink=2, budget=budget, head_dim=16)
    k, v = _rand_kv(budget + 1, D=16)
    st = squeeze_update(st, k, v)
    assert st.keys.shape[0] == budget


def test_sinks_always_present():
    """The first n_sink positions survive all evictions."""
    n_sink = 3
    st = init_squeeze_state(n_sink=n_sink, budget=6, head_dim=8)
    # distinctive sink keys we can identify after eviction
    sink_k = mx.arange(n_sink * 8).reshape(n_sink, 8).astype(mx.float16) + 1000.0
    sink_v = mx.zeros((n_sink, 8), dtype=mx.float16)
    st = squeeze_update(st, sink_k, sink_v)
    for step in range(20):
        k, v = _rand_kv(1, D=8, seed=step)
        st = squeeze_update(st, k, v)
    # first n_sink rows unchanged
    assert bool(mx.all(st.keys[:n_sink] == sink_k.astype(mx.float16)).item())


def test_n_sink_zero_edge_case():
    st = init_squeeze_state(n_sink=0, budget=5, head_dim=8)
    for step in range(15):
        k, v = _rand_kv(1, D=8, seed=step)
        st = squeeze_update(st, k, v)
    assert st.keys.shape[0] <= 5


def test_scores_non_negative():
    st = init_squeeze_state(n_sink=2, budget=12, head_dim=16)
    for step in range(20):
        k, v = _rand_kv(1, D=16, seed=step)
        st = squeeze_update(st, k, v)
    assert bool(mx.all(st.scores >= 0).item())


def test_byte_accounting():
    st = init_squeeze_state(n_sink=2, budget=8, head_dim=32)
    assert squeeze_fp16_bytes(st) == 0
    k, v = _rand_kv(5, D=32)
    st = squeeze_update(st, k, v)
    assert squeeze_fp16_bytes(st) == 5 * 32 * 2 * 2
    assert full_squeeze_fp16_bytes(100, 32) == 100 * 32 * 2 * 2


def test_get_kv_placeholder_before_update():
    st = init_squeeze_state(n_sink=2, budget=8, head_dim=32)
    k, v = squeeze_get_kv(st)
    assert k.shape == (0, 1) and v.shape == (0, 1)


def test_determinism():
    """Identical inputs → identical retained state."""

    def run():
        st = init_squeeze_state(n_sink=3, budget=9, head_dim=16)
        for step in range(25):
            k, v = _rand_kv(1, D=16, seed=step)
            st = squeeze_update(st, k, v)
        return st.keys

    a, b = run(), run()
    assert bool(mx.all(a == b).item())

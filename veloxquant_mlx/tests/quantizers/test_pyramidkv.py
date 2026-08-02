"""Tests for PyramidKV-adapted quantizer primitives — layer-adaptive budget eviction.

PyramidKV-adapted (arXiv:2406.02069) allocates a pyramid of per-layer budgets
(large early, small deep, fixed mean) and evicts within each layer using H2O's
cumulative-attention-mass scorer. Tests cover: the pyramid_budgets allocator
(shape, monotonicity, mean preservation, flat==uniform, sink floor, edge cases)
and pyramid_update / state mechanics (bootstrap, budget enforcement, sink
protection, byte accounting, determinism). All data is synthetic — no model
loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.pyramidkv import (
    PyramidState,
    full_pyramid_fp16_bytes,
    init_pyramid_state,
    pyramid_budgets,
    pyramid_fp16_bytes,
    pyramid_get_kv,
    pyramid_update,
)


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# pyramid_budgets — the allocator (the distinguishing feature)
# ---------------------------------------------------------------------------


def test_budgets_length_matches_layers() -> None:
    b = pyramid_budgets(n_layers=12, avg_budget=512, n_sink=4, beta=2.0)
    assert len(b) == 12


def test_budgets_monotonically_decreasing() -> None:
    """Early layers get more budget than deep layers."""
    b = pyramid_budgets(n_layers=16, avg_budget=256, n_sink=4, beta=2.0)
    for i in range(len(b) - 1):
        assert b[i] >= b[i + 1], f"layer {i}={b[i]} < layer {i + 1}={b[i + 1]}"
    assert b[0] > b[-1]  # strictly decreasing overall


def test_budgets_mean_approx_avg() -> None:
    """The schedule's mean stays close to the requested average budget."""
    avg = 512
    b = pyramid_budgets(n_layers=32, avg_budget=avg, n_sink=4, beta=2.0)
    mean = sum(b) / len(b)
    # Rounding + floor may nudge it slightly; within 5% is the contract.
    assert abs(mean - avg) / avg < 0.05, f"mean {mean} vs avg {avg}"


def test_budgets_flat_when_beta_1() -> None:
    """beta=1.0 → every layer gets the average (reduces to uniform H2O)."""
    b = pyramid_budgets(n_layers=10, avg_budget=300, n_sink=4, beta=1.0)
    assert all(x == 300 for x in b)


def test_budgets_steeper_beta_wider_spread() -> None:
    """Larger beta widens the gap between first and last layer."""
    b2 = pyramid_budgets(n_layers=20, avg_budget=512, n_sink=4, beta=2.0)
    b3 = pyramid_budgets(n_layers=20, avg_budget=512, n_sink=4, beta=3.0)
    spread2 = b2[0] - b2[-1]
    spread3 = b3[0] - b3[-1]
    assert spread3 > spread2


def test_budgets_floor_at_n_sink_plus_1() -> None:
    """No layer's budget drops below n_sink + 1, even with a steep pyramid."""
    n_sink = 8
    b = pyramid_budgets(n_layers=24, avg_budget=64, n_sink=n_sink, beta=4.0)
    assert all(x >= n_sink + 1 for x in b), b


def test_budgets_single_layer() -> None:
    b = pyramid_budgets(n_layers=1, avg_budget=512, n_sink=4, beta=2.0)
    assert b == [512]


def test_budgets_empty() -> None:
    assert pyramid_budgets(n_layers=0, avg_budget=512, n_sink=4) == []


def test_budgets_beta_below_1_raises() -> None:
    with pytest.raises(ValueError):
        pyramid_budgets(n_layers=8, avg_budget=512, n_sink=4, beta=0.5)


# ---------------------------------------------------------------------------
# init_pyramid_state
# ---------------------------------------------------------------------------


def test_init_state_fields() -> None:
    st = init_pyramid_state(n_sink=4, budget=128, head_dim=64)
    assert st.n_sink == 4
    assert st.budget == 128
    assert st.keys is None
    assert st.values is None
    assert st.scores is None


def test_init_state_rejects_n_sink_equal_budget() -> None:
    """n_sink >= budget leaves no evictable room — sinks would be evicted
    once the cache fills, defeating the sink guarantee."""
    with pytest.raises(ValueError, match="n_sink"):
        init_pyramid_state(n_sink=4, budget=4, head_dim=32)


def test_init_state_rejects_n_sink_above_budget() -> None:
    with pytest.raises(ValueError, match="n_sink"):
        init_pyramid_state(n_sink=8, budget=4, head_dim=32)


def test_init_state_allows_disabled_cache() -> None:
    """n_sink=0, budget=0 is a valid 'disabled cache' configuration."""
    st = init_pyramid_state(n_sink=0, budget=0, head_dim=32)
    assert st.n_sink == 0
    assert st.budget == 0


# ---------------------------------------------------------------------------
# pyramid_get_kv — empty state
# ---------------------------------------------------------------------------


def test_get_kv_empty_returns_zero_rows() -> None:
    st = init_pyramid_state(n_sink=4, budget=16, head_dim=32)
    k, v = pyramid_get_kv(st)
    assert k.shape[0] == 0
    assert v.shape[0] == 0


# ---------------------------------------------------------------------------
# pyramid_update — eviction bounded by the per-layer budget
# ---------------------------------------------------------------------------


def test_single_token_absorbed() -> None:
    D = 32
    st = init_pyramid_state(n_sink=4, budget=16, head_dim=D)
    k, v = _rand_kv(S=1, D=D)
    st = pyramid_update(st, k, v)
    assert st.keys.shape == (1, D)


def test_multi_token_below_budget_all_kept() -> None:
    D = 32
    st = init_pyramid_state(n_sink=4, budget=16, head_dim=D)
    k, v = _rand_kv(S=8, D=D)
    st = pyramid_update(st, k, v)
    assert st.keys.shape[0] == 8


def test_budget_never_exceeded() -> None:
    D = 32
    budget = 8
    st = init_pyramid_state(n_sink=2, budget=budget, head_dim=D)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=i)
        st = pyramid_update(st, ki, vi)
    assert st.keys.shape[0] <= budget


def test_budget_exactly_enforced_at_boundary() -> None:
    D = 16
    budget = 5
    st = init_pyramid_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=budget + 1, D=D)
    st = pyramid_update(st, k, v)
    assert st.keys.shape[0] == budget


def test_different_budgets_give_different_kept_counts() -> None:
    """Two layers with different pyramid budgets retain different token counts."""
    D = 16
    k, v = _rand_kv(S=40, D=D, seed=7)
    st_big = init_pyramid_state(n_sink=2, budget=20, head_dim=D)
    st_small = init_pyramid_state(n_sink=2, budget=6, head_dim=D)
    st_big = pyramid_update(st_big, k, v)
    st_small = pyramid_update(st_small, k, v)
    assert st_big.keys.shape[0] == 20
    assert st_small.keys.shape[0] == 6


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_sinks_never_evicted() -> None:
    D = 8
    n_sink = 3
    budget = 4
    k_sink = mx.ones((n_sink, D), dtype=mx.float16)
    v_sink = mx.ones((n_sink, D), dtype=mx.float16) * 2.0
    st = init_pyramid_state(n_sink=n_sink, budget=budget, head_dim=D)
    st = pyramid_update(st, k_sink, v_sink)
    for i in range(20):
        ki, vi = _rand_kv(S=1, D=D, seed=100 + i)
        st = pyramid_update(st, ki, vi)
    ko, _ = pyramid_get_kv(st)
    assert ko.shape[0] <= budget
    for r in range(min(n_sink, ko.shape[0])):
        assert float(ko[r, 0].item()) == pytest.approx(1.0, abs=1e-2)


def test_n_sink_zero_allows_all_evictions() -> None:
    D = 16
    budget = 4
    st = init_pyramid_state(n_sink=0, budget=budget, head_dim=D)
    k, v = _rand_kv(S=20, D=D)
    st = pyramid_update(st, k, v)
    assert st.keys.shape[0] <= budget


# ---------------------------------------------------------------------------
# Score accumulation (inherited from H2O scorer)
# ---------------------------------------------------------------------------


def test_scores_non_negative() -> None:
    D = 32
    st = init_pyramid_state(n_sink=2, budget=8, head_dim=D)
    k, v = _rand_kv(S=8, D=D)
    st = pyramid_update(st, k, v)
    scores_np = np.array(st.scores.tolist())
    assert np.all(scores_np >= 0.0)


def test_score_array_length_matches_keys() -> None:
    D = 16
    st = init_pyramid_state(n_sink=2, budget=6, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = pyramid_update(st, k, v)
    assert st.scores.shape[0] == st.keys.shape[0]


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_pyramid_fp16_bytes_formula() -> None:
    D = 64
    st = init_pyramid_state(n_sink=4, budget=16, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = pyramid_update(st, k, v)
    n_kept = st.keys.shape[0]
    assert pyramid_fp16_bytes(st) == n_kept * D * 2 * 2


def test_pyramid_fp16_bytes_empty_state() -> None:
    st = init_pyramid_state(n_sink=4, budget=16, head_dim=64)
    assert pyramid_fp16_bytes(st) == 0


def test_full_pyramid_fp16_bytes_formula() -> None:
    assert full_pyramid_fp16_bytes(100, 128) == 100 * 128 * 2 * 2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_identical_inputs() -> None:
    D = 32
    budget = 8
    k, v = _rand_kv(S=12, D=D, seed=42)
    st_a = pyramid_update(init_pyramid_state(2, budget, D), k, v)
    st_b = pyramid_update(init_pyramid_state(2, budget, D), k, v)
    ka, _ = pyramid_get_kv(st_a)
    kb, _ = pyramid_get_kv(st_b)
    mse = float(mx.mean((ka.astype(mx.float32) - kb.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)

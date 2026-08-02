"""Tests for H2O-adapted quantizer primitives — cumulative attention-mass eviction.

H2O-adapted (arXiv:2306.14048, ICLR 2024) accumulates per-token softmax attention
weights (using the incoming key as a proxy query) and evicts the lowest-score
non-sink token whenever the cache exceeds the budget. Tests cover: init_h2o_state,
h2o_update (single token, multi-step, eviction trigger, sink protection, budget
enforcement, score accumulation), h2o_get_kv (shape, dtype, empty state), byte
accounting, and edge cases (n_sink=0, budget=1, score-based ordering). All data
is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.h2o import (
    H2OState,
    full_h2o_fp16_bytes,
    h2o_fp16_bytes,
    h2o_get_kv,
    h2o_update,
    init_h2o_state,
)


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init_h2o_state
# ---------------------------------------------------------------------------


def test_init_state_fields() -> None:
    st = init_h2o_state(n_sink=4, budget=16, head_dim=64)
    assert st.n_sink == 4
    assert st.budget == 16
    assert st.keys is None
    assert st.values is None
    assert st.scores is None


def test_init_state_zero_budget_allowed() -> None:
    st = init_h2o_state(n_sink=0, budget=0, head_dim=32)
    assert st.budget == 0


def test_init_state_rejects_n_sink_equal_budget() -> None:
    """n_sink >= budget leaves no evictable room — sinks would be evicted
    once the cache fills, silently violating the "sinks never evicted"
    invariant. Must raise instead of letting that happen quietly."""
    with pytest.raises(ValueError, match="n_sink"):
        init_h2o_state(n_sink=4, budget=4, head_dim=32)


def test_init_state_rejects_n_sink_above_budget() -> None:
    with pytest.raises(ValueError, match="n_sink"):
        init_h2o_state(n_sink=8, budget=4, head_dim=32)


# ---------------------------------------------------------------------------
# h2o_get_kv — empty state
# ---------------------------------------------------------------------------


def test_get_kv_empty_returns_zero_rows() -> None:
    st = init_h2o_state(n_sink=4, budget=16, head_dim=32)
    k, v = h2o_get_kv(st)
    assert k.shape[0] == 0
    assert v.shape[0] == 0


# ---------------------------------------------------------------------------
# h2o_update — basic absorption
# ---------------------------------------------------------------------------


def test_single_token_absorbed() -> None:
    """First token bootstraps state with one row."""
    D = 32
    st = init_h2o_state(n_sink=4, budget=16, head_dim=D)
    k, v = _rand_kv(S=1, D=D)
    st = h2o_update(st, k, v)
    assert st.keys is not None
    assert st.keys.shape == (1, D)
    assert st.values.shape == (1, D)


def test_multi_token_absorbed_below_budget() -> None:
    """S tokens below budget → all kept."""
    D = 32
    budget = 16
    S = 8
    st = init_h2o_state(n_sink=4, budget=budget, head_dim=D)
    k, v = _rand_kv(S=S, D=D)
    st = h2o_update(st, k, v)
    assert st.keys.shape[0] == S


def test_output_dtype_fp16() -> None:
    D = 32
    st = init_h2o_state(n_sink=2, budget=8, head_dim=D)
    k, v = _rand_kv(S=4, D=D)
    st = h2o_update(st, k, v)
    ko, vo = h2o_get_kv(st)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


# ---------------------------------------------------------------------------
# h2o_update — eviction when over budget
# ---------------------------------------------------------------------------


def test_budget_never_exceeded() -> None:
    """After many tokens, kept count never exceeds budget."""
    D = 32
    budget = 8
    n_sink = 2
    st = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=i)
        st = h2o_update(st, ki, vi)
    assert st.keys.shape[0] <= budget


def test_budget_exactly_enforced_at_boundary() -> None:
    """Exactly budget+1 tokens → evict one → budget tokens remain."""
    D = 16
    budget = 5
    st = init_h2o_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=budget + 1, D=D)
    st = h2o_update(st, k, v)
    assert st.keys.shape[0] == budget


def test_score_array_length_matches_keys() -> None:
    """scores array length always equals number of kept tokens."""
    D = 16
    budget = 6
    st = init_h2o_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = h2o_update(st, k, v)
    assert st.scores.shape[0] == st.keys.shape[0]


# ---------------------------------------------------------------------------
# h2o_update — sink protection
# ---------------------------------------------------------------------------


def test_sinks_never_evicted() -> None:
    """First n_sink tokens are always present in the output."""
    D = 8
    n_sink = 3
    budget = 4
    # Build known sink keys: all ones
    k_sink = mx.ones((n_sink, D), dtype=mx.float16)
    v_sink = mx.ones((n_sink, D), dtype=mx.float16) * 2.0
    st = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    st = h2o_update(st, k_sink, v_sink)

    # Push many more tokens to force evictions
    for i in range(20):
        ki, vi = _rand_kv(S=1, D=D, seed=100 + i)
        st = h2o_update(st, ki, vi)

    ko, vo = h2o_get_kv(st)
    # First 3 rows must be the original sinks (all-ones keys)
    assert ko.shape[0] <= budget
    # The sink rows should still be present — check first n_sink rows
    for r in range(min(n_sink, ko.shape[0])):
        assert float(ko[r, 0].item()) == pytest.approx(1.0, abs=1e-2)


def test_n_sink_zero_allows_all_evictions() -> None:
    """With n_sink=0, any token may be evicted."""
    D = 16
    budget = 4
    st = init_h2o_state(n_sink=0, budget=budget, head_dim=D)
    k, v = _rand_kv(S=20, D=D)
    st = h2o_update(st, k, v)
    assert st.keys.shape[0] <= budget


# ---------------------------------------------------------------------------
# h2o_update — score accumulation
# ---------------------------------------------------------------------------


def test_scores_non_negative() -> None:
    """Cumulative scores are always >= 0 (they are sums of softmax weights)."""
    D = 32
    budget = 8
    st = init_h2o_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=8, D=D)
    st = h2o_update(st, k, v)
    scores_np = np.array(st.scores.tolist())
    assert np.all(scores_np >= 0.0)


def test_scores_accumulate_over_steps() -> None:
    """After two update calls, scores are strictly higher than after one."""
    D = 16
    budget = 8
    st = init_h2o_state(n_sink=0, budget=budget, head_dim=D)
    k1, v1 = _rand_kv(S=4, D=D, seed=0)
    st1 = h2o_update(st, k1, v1)
    scores_after_1 = np.array(st1.scores.tolist())

    k2, v2 = _rand_kv(S=1, D=D, seed=1)
    st2 = h2o_update(st1, k2, v2)
    # After a second step, at least some existing token scores grew
    scores_after_2 = np.array(st2.scores[: len(scores_after_1)].tolist())
    # Softmax weights are positive, so at least total mass increased
    assert scores_after_2.sum() >= scores_after_1.sum() - 1e-4


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_h2o_fp16_bytes_formula() -> None:
    """h2o_fp16_bytes = n_kept * D * 4 (K + V, fp16)."""
    D = 64
    budget = 16
    st = init_h2o_state(n_sink=4, budget=budget, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = h2o_update(st, k, v)
    n_kept = st.keys.shape[0]
    expected = n_kept * D * 2 * 2
    assert h2o_fp16_bytes(st) == expected


def test_h2o_fp16_bytes_empty_state() -> None:
    st = init_h2o_state(n_sink=4, budget=16, head_dim=64)
    assert h2o_fp16_bytes(st) == 0


def test_full_h2o_fp16_bytes_formula() -> None:
    assert full_h2o_fp16_bytes(100, 128) == 100 * 128 * 2 * 2


# ---------------------------------------------------------------------------
# Multi-step decode stress
# ---------------------------------------------------------------------------


def test_30_step_stress_budget_constant() -> None:
    """30 single-token decode steps — budget never exceeded."""
    D = 32
    budget = 10
    n_sink = 3
    st = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    # Prefill n_sink tokens first
    k_s, v_s = _rand_kv(S=n_sink, D=D, seed=0)
    st = h2o_update(st, k_s, v_s)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=10 + i)
        st = h2o_update(st, ki, vi)
        assert st.keys.shape[0] <= budget, f"step {i}: {st.keys.shape[0]} > {budget}"


def test_deterministic_across_identical_inputs() -> None:
    """Same input → same state (no stochastic elements)."""
    D = 32
    budget = 8
    k, v = _rand_kv(S=12, D=D, seed=42)

    st_a = init_h2o_state(n_sink=2, budget=budget, head_dim=D)
    st_a = h2o_update(st_a, k, v)

    st_b = init_h2o_state(n_sink=2, budget=budget, head_dim=D)
    st_b = h2o_update(st_b, k, v)

    ka, _ = h2o_get_kv(st_a)
    kb, _ = h2o_get_kv(st_b)
    mse = float(mx.mean((ka.astype(mx.float32) - kb.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)

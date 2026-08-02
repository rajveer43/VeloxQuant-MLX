"""Unit tests for Q-Filters-adapted query-agnostic projection eviction.

Covers:
  - under-budget passthrough
  - pre-calibration passthrough (no eviction before the filter freezes)
  - estimate_filter_dir recovers a planted dominant direction
  - over-budget selection = the budget highest-projection positions, order kept
  - sink and recent-window protection; guard + sign validation
  - sign=-1 inversion
  - frozen-filter determinism (a stored score never changes once frozen)
  - given-same-filter order invariance (NOT path independence — Q-Filters
    is path-DEPENDENT by construction)
  - byte accounting incl. the float32 filter_dir term, and placeholders
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.qfilters import (
    estimate_filter_dir,
    full_qfilters_fp16_bytes,
    init_qfilters_state,
    qfilters_fp16_bytes,
    qfilters_get_kv,
    qfilters_update,
)


def _kv(S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((S, D)).astype(np.float16)
    v = rng.standard_normal((S, D)).astype(np.float16)
    return mx.array(k), mx.array(v)


def _proj(k: mx.array, d: mx.array, sign: int = 1) -> np.ndarray:
    return sign * (np.array(k, dtype=np.float32) @ np.array(d, dtype=np.float32))


# ------------------------------------------------------------------
# Basic behavior
# ------------------------------------------------------------------


def test_under_budget_passthrough_in_order() -> None:
    st = init_qfilters_state(n_sink=2, budget=64, head_dim=8, calib_tokens=8)
    k, v = _kv(10, 8, seed=1)
    st = qfilters_update(st, k, v)
    ko, vo = qfilters_get_kv(st)
    assert np.array_equal(np.array(ko), np.array(k.astype(mx.float16)))
    assert np.array_equal(np.array(vo), np.array(v.astype(mx.float16)))


def test_pre_calibration_passthrough_even_over_budget() -> None:
    """Below calib_tokens the filter is None — nothing is evicted even when
    the token count exceeds the budget."""
    st = init_qfilters_state(n_sink=0, budget=8, head_dim=8, calib_tokens=100)
    k, v = _kv(50, 8, seed=2)  # 50 > budget 8, but 50 < calib_tokens 100
    st = qfilters_update(st, k, v)
    assert st.filter_dir is None
    assert st.keys.shape[0] == 50  # nothing evicted yet


def test_estimate_filter_dir_recovers_planted_axis() -> None:
    rng = np.random.default_rng(3)
    D, N, axis = 8, 300, 5
    base = rng.standard_normal((N, D)).astype(np.float32) * 0.1
    planted = np.zeros(D, dtype=np.float32)
    planted[axis] = 1.0
    keys = base + rng.standard_normal((N, 1)).astype(np.float32) * 4.0 * planted
    d = np.array(estimate_filter_dir(mx.array(keys)))
    cos = abs(float(d @ planted) / (np.linalg.norm(d) * np.linalg.norm(planted)))
    assert cos > 0.99


def test_over_budget_keeps_highest_projection_in_order() -> None:
    S, D, budget = 64, 8, 20
    st = init_qfilters_state(n_sink=0, budget=budget, head_dim=D, calib_tokens=16)
    k, v = _kv(S, D, seed=4)
    st = qfilters_update(st, k, v)
    assert st.keys.shape[0] == budget

    scores = _proj(k, st.filter_dir, sign=1)
    expected_idx = np.sort(np.argsort(scores)[-budget:])
    assert np.array_equal(np.array(st.keys), np.array(k)[expected_idx])
    assert np.array_equal(np.array(st.values), np.array(v)[expected_idx])


def test_sinks_protected_even_with_low_projection() -> None:
    S, D, n_sink, budget = 64, 8, 3, 20
    st = init_qfilters_state(n_sink=n_sink, budget=budget, head_dim=D, calib_tokens=16)
    k, v = _kv(S, D, seed=5)
    st = qfilters_update(st, k, v)
    assert st.keys.shape[0] == budget
    assert np.array_equal(np.array(st.keys[:n_sink]), np.array(k[:n_sink]))


def test_recent_window_protected() -> None:
    S, D, budget, recent = 64, 8, 20, 3
    st = init_qfilters_state(n_sink=0, budget=budget, head_dim=D, recent=recent, calib_tokens=16)
    k, v = _kv(S, D, seed=6)
    st = qfilters_update(st, k, v)
    assert st.keys.shape[0] == budget
    assert np.array_equal(np.array(st.keys[-recent:]), np.array(k[-recent:]))


def test_guard_and_sign_validation() -> None:
    with pytest.raises(ValueError, match="evictable"):
        init_qfilters_state(n_sink=4, budget=8, head_dim=8, recent=4)
    with pytest.raises(ValueError, match="sign"):
        init_qfilters_state(n_sink=0, budget=8, head_dim=8, sign=2)


def test_sign_inverts_selection() -> None:
    S, D, budget = 64, 8, 20
    k, v = _kv(S, D, seed=7)
    pos = qfilters_update(init_qfilters_state(0, budget, D, calib_tokens=16, sign=1), k, v)
    neg = qfilters_update(init_qfilters_state(0, budget, D, calib_tokens=16, sign=-1), k, v)
    assert not np.array_equal(np.array(pos.keys), np.array(neg.keys))


# ------------------------------------------------------------------
# Frozen-filter properties
# ------------------------------------------------------------------


def test_filter_frozen_and_scores_stable() -> None:
    D, budget = 8, 128
    st = init_qfilters_state(n_sink=0, budget=budget, head_dim=D, calib_tokens=16)
    k1, v1 = _kv(20, D, seed=8)
    st = qfilters_update(st, k1, v1)
    frozen = np.array(st.filter_dir).copy()
    scores_before = np.array(st.scores[:20]).copy()

    k2, v2 = _kv(20, D, seed=9)
    st = qfilters_update(st, k2, v2)
    # Filter unchanged after freezing; earlier tokens' scores unchanged.
    assert np.array_equal(np.array(st.filter_dir), frozen)
    np.testing.assert_allclose(np.array(st.scores[:20]), scores_before, rtol=1e-5)


def test_given_same_filter_order_invariance() -> None:
    """NOT path independence (the filter itself can differ per path, and
    positional sink protection is order-sensitive). But with a fixed injected
    filter and no positional protection (n_sink=recent=0), the kept set is the
    top-budget-by-score regardless of arrival grouping — pure top-k."""
    S, D, budget = 40, 8, 12
    k, v = _kv(S, D, seed=10)
    rng = np.random.default_rng(11)
    fixed = rng.standard_normal(D).astype(np.float32)
    fixed = mx.array(fixed / np.linalg.norm(fixed))

    def run(order):
        st = init_qfilters_state(0, budget, D, calib_tokens=1)
        st.filter_dir = fixed  # inject, bypass estimation
        for t in order:
            st = qfilters_update(st, k[t : t + 1], v[t : t + 1])
        return st

    block = run(list(range(S)))
    perm = list(np.random.default_rng(12).permutation(S))
    # Same *set* of tokens regardless of arrival order (compare as sorted rows).
    ka = np.sort(np.array(block.keys), axis=0)
    kb = np.sort(np.array(run(perm).keys), axis=0)
    assert np.array_equal(ka, kb)


# ------------------------------------------------------------------
# Accounting / placeholders
# ------------------------------------------------------------------


def test_bytes_accounting_includes_filter() -> None:
    D, budget = 16, 8
    st = init_qfilters_state(n_sink=0, budget=budget, head_dim=D, calib_tokens=4)
    assert qfilters_fp16_bytes(st) == 0
    k, v = _kv(40, D, seed=13)
    st = qfilters_update(st, k, v)
    # K+V fp16 for `budget` tokens, plus D float32 for the frozen filter.
    assert qfilters_fp16_bytes(st) == budget * D * 2 * 2 + D * 4
    assert full_qfilters_fp16_bytes(40, D) == 40 * D * 2 * 2


def test_empty_state_placeholder() -> None:
    st = init_qfilters_state(n_sink=0, budget=8, head_dim=8)
    ko, vo = qfilters_get_kv(st)
    assert ko.shape == (0, 1) and vo.shape == (0, 1)

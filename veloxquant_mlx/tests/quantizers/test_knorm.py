"""Unit tests for L2Norm-adapted intrinsic key-norm eviction primitives.

Covers:
  - under-budget passthrough with correct norms and order
  - over-budget selection = the budget lowest-norm positions, order preserved
  - sink and recent-window protection; guard validation
  - keep="high" inversion
  - norm immutability (intrinsic scores never update)
  - path independence at recent=0 (block vs token-by-token, bit-for-bit)
  - byte accounting and empty-state placeholders
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.knorm import (
    full_knorm_fp16_bytes,
    init_knorm_state,
    knorm_fp16_bytes,
    knorm_get_kv,
    knorm_update,
)


def _kv(S, D, seed=0, scale=None):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((S, D)).astype(np.float16)
    if scale is not None:
        k = (k * scale[:, None]).astype(np.float16)
    v = rng.standard_normal((S, D)).astype(np.float16)
    return mx.array(k), mx.array(v)


def _norms(k: mx.array) -> np.ndarray:
    return np.linalg.norm(np.array(k, dtype=np.float32), axis=-1)


# ------------------------------------------------------------------
# Basic behavior
# ------------------------------------------------------------------


def test_under_budget_passthrough_in_order() -> None:
    st = init_knorm_state(n_sink=2, budget=16, head_dim=8)
    k, v = _kv(10, 8, seed=1)
    st = knorm_update(st, k, v)
    ko, vo = knorm_get_kv(st)
    assert np.array_equal(np.array(ko), np.array(k.astype(mx.float16)))
    assert np.array_equal(np.array(vo), np.array(v.astype(mx.float16)))
    np.testing.assert_allclose(np.array(st.norms), _norms(k), rtol=2e-3)


def test_over_budget_keeps_lowest_norms_in_order() -> None:
    S, D, budget = 32, 8, 12
    st = init_knorm_state(n_sink=0, budget=budget, head_dim=D)
    k, v = _kv(S, D, seed=2)
    st = knorm_update(st, k, v)
    assert st.keys.shape[0] == budget

    expected_idx = np.sort(np.argsort(_norms(k))[:budget])
    assert np.array_equal(np.array(st.keys), np.array(k)[expected_idx])
    assert np.array_equal(np.array(st.values), np.array(v)[expected_idx])


def test_sinks_protected_even_with_highest_norms() -> None:
    S, D, n_sink, budget = 24, 8, 3, 8
    scale = np.ones(S, dtype=np.float32)
    scale[:n_sink] = 50.0  # sinks get enormous norms
    st = init_knorm_state(n_sink=n_sink, budget=budget, head_dim=D)
    k, v = _kv(S, D, seed=3, scale=scale)
    st = knorm_update(st, k, v)
    assert st.keys.shape[0] == budget
    # First n_sink kept rows are exactly the original sink rows.
    assert np.array_equal(np.array(st.keys[:n_sink]), np.array(k[:n_sink]))


def test_recent_window_protected() -> None:
    S, D, budget, recent = 24, 8, 8, 3
    scale = np.ones(S, dtype=np.float32)
    scale[-recent:] = 50.0  # newest tokens get enormous norms
    st = init_knorm_state(n_sink=0, budget=budget, head_dim=D, recent=recent)
    k, v = _kv(S, D, seed=4, scale=scale)
    st = knorm_update(st, k, v)
    assert st.keys.shape[0] == budget
    assert np.array_equal(np.array(st.keys[-recent:]), np.array(k[-recent:]))


def test_guard_sink_plus_recent_vs_budget() -> None:
    with pytest.raises(ValueError, match="evictable"):
        init_knorm_state(n_sink=4, budget=8, head_dim=8, recent=4)
    with pytest.raises(ValueError, match="keep"):
        init_knorm_state(n_sink=0, budget=8, head_dim=8, keep="middle")


def test_keep_high_inverts_selection() -> None:
    S, D, budget = 32, 8, 12
    k, v = _kv(S, D, seed=5)
    lo = knorm_update(init_knorm_state(0, budget, D, keep="low"), k, v)
    hi = knorm_update(init_knorm_state(0, budget, D, keep="high"), k, v)
    norms = _norms(k)
    expected_hi = np.sort(np.argsort(norms)[-budget:])
    assert np.array_equal(np.array(hi.keys), np.array(k)[expected_hi])
    # Disjoint apart from possible middle overlap — at least not identical.
    assert not np.array_equal(np.array(lo.keys), np.array(hi.keys))


# ------------------------------------------------------------------
# Intrinsic-score properties
# ------------------------------------------------------------------


def test_norms_immutable_across_updates() -> None:
    D, budget = 8, 64
    st = init_knorm_state(n_sink=0, budget=budget, head_dim=D)
    k1, v1 = _kv(8, D, seed=6)
    st = knorm_update(st, k1, v1)
    before = np.array(st.norms[:8]).copy()
    k2, v2 = _kv(8, D, seed=7)
    st = knorm_update(st, k2, v2)
    assert np.array_equal(np.array(st.norms[:8]), before)


def test_path_independence_block_vs_tokenwise() -> None:
    """With recent=0, the kept set is the global budget-lowest regardless of
    arrival grouping — 'keep k best with a heap'. Bit-for-bit check."""
    S, D, budget, n_sink = 40, 8, 10, 2
    k, v = _kv(S, D, seed=8)

    block = knorm_update(init_knorm_state(n_sink, budget, D), k, v)

    stream = init_knorm_state(n_sink, budget, D)
    for t in range(S):
        stream = knorm_update(stream, k[t : t + 1], v[t : t + 1])

    assert np.array_equal(np.array(block.keys), np.array(stream.keys))
    assert np.array_equal(np.array(block.values), np.array(stream.values))


# ------------------------------------------------------------------
# Accounting / placeholders
# ------------------------------------------------------------------


def test_bytes_accounting() -> None:
    D, budget = 16, 8
    st = init_knorm_state(n_sink=0, budget=budget, head_dim=D)
    assert knorm_fp16_bytes(st) == 0
    k, v = _kv(20, D, seed=9)
    st = knorm_update(st, k, v)
    assert knorm_fp16_bytes(st) == budget * D * 2 * 2
    assert full_knorm_fp16_bytes(20, D) == 20 * D * 2 * 2


def test_empty_state_placeholder() -> None:
    st = init_knorm_state(n_sink=0, budget=8, head_dim=8)
    ko, vo = knorm_get_kv(st)
    assert ko.shape == (0, 1) and vo.shape == (0, 1)

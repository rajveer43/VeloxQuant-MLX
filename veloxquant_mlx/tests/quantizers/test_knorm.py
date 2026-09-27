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
    KnormState,
    full_knorm_fp16_bytes,
    init_knorm_state,
    knorm_fp16_bytes,
    knorm_get_kv,
    knorm_update,
    knorm_update_batched,
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


# ------------------------------------------------------------------
# Batched (N = B*H) equivalence against the per-row scalar loop
# ------------------------------------------------------------------


def _rand_kv_batch(N, S, D, seed):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((N, S, D)).astype(np.float16)
    v = rng.standard_normal((N, S, D)).astype(np.float16)
    return mx.array(k), mx.array(v)


@pytest.mark.parametrize("budget", [4, 8])
@pytest.mark.parametrize("n_sink,recent", [(0, 0), (1, 0), (0, 2), (1, 1)])
@pytest.mark.parametrize("keep", ["low", "high"])
@pytest.mark.parametrize("n_prior,s_new", [(0, 5), (3, 4), (7, 1), (0, 12)])
def test_batched_matches_loop_N_gt1(budget, n_sink, recent, keep, n_prior, s_new) -> None:
    if n_sink + recent >= budget:
        pytest.skip("invalid guard combination")
    N, D = 3, 6
    seed = hash((budget, n_sink, recent, keep, n_prior, s_new)) % (2**31)

    prior_k, prior_v = _rand_kv_batch(N, n_prior, D, seed)
    new_k, new_v = _rand_kv_batch(N, s_new, D, seed + 1)
    prior_norms = mx.sqrt(mx.sum(prior_k.astype(mx.float32) ** 2, axis=-1))

    out_k, out_v, out_n = knorm_update_batched(
        prior_k, prior_v, prior_norms, new_k, new_v, budget, n_sink, recent, keep
    )

    for i in range(N):
        st = KnormState(
            keys=prior_k[i] if n_prior else None,
            values=prior_v[i] if n_prior else None,
            norms=prior_norms[i] if n_prior else None,
            n_sink=n_sink,
            budget=budget,
            recent=recent,
            keep=keep,
        )
        st = knorm_update(st, new_k[i], new_v[i])
        np.testing.assert_array_equal(np.array(out_k[i]), np.array(st.keys))
        np.testing.assert_array_equal(np.array(out_v[i]), np.array(st.values))
        np.testing.assert_array_equal(np.array(out_n[i]), np.array(st.norms))


def test_batched_multi_step_matches_loop_N_gt1() -> None:
    N, D, budget, n_sink, recent, keep = 2, 5, 6, 1, 1, "low"
    prior_k = mx.zeros((N, 0, D), dtype=mx.float16)
    prior_v = mx.zeros((N, 0, D), dtype=mx.float16)
    prior_n = mx.zeros((N, 0), dtype=mx.float32)
    states = [init_knorm_state(n_sink, budget, D, recent=recent, keep=keep) for _ in range(N)]

    for step, s in enumerate([2, 3, 1, 4, 2]):
        new_k, new_v = _rand_kv_batch(N, s, D, seed=100 + step)
        prior_k, prior_v, prior_n = knorm_update_batched(
            prior_k, prior_v, prior_n, new_k, new_v, budget, n_sink, recent, keep
        )
        for i in range(N):
            states[i] = knorm_update(states[i], new_k[i], new_v[i])
            np.testing.assert_array_equal(np.array(prior_k[i]), np.array(states[i].keys))
            np.testing.assert_array_equal(np.array(prior_v[i]), np.array(states[i].values))
            np.testing.assert_array_equal(np.array(prior_n[i]), np.array(states[i].norms))

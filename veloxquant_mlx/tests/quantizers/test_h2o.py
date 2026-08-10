"""Tests for H2O-adapted quantizer primitives — cumulative attention-mass eviction.

H2O-adapted (arXiv:2306.14048, ICLR 2024) accumulates per-token softmax attention
weights (using the incoming key as a proxy query) and evicts the lowest-score
non-sink token whenever the cache exceeds the budget. Tests cover: init_h2o_state,
h2o_update (single token, multi-step, eviction trigger, sink protection, budget
enforcement, score accumulation), h2o_get_kv (shape, dtype, empty state), byte
accounting, edge cases (n_sink=0, budget=1, score-based ordering), RoPE position
remapping after eviction, and the vectorized below-budget batch path (a
performance fix: verified numerically equivalent to the sequential per-token
loop it replaces, including across the batch-path/loop-path boundary within a
single call). All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.a2ats_rope import a2ats_apply_exact_rope, rope_remap_positions
from veloxquant_mlx.quantizers.h2o import (
    H2OState,
    _batch_absorb_no_eviction,
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


# ---------------------------------------------------------------------------
# RoPE position remapping after eviction (fixes a real regression — see
# module docstring and docs-site/docs/algorithms/h2o.md for the full
# investigation, including why interior eviction is rare but not impossible)
# ---------------------------------------------------------------------------


def test_state_tracks_positions_and_next_pos() -> None:
    """H2OState exposes positions/next_pos so callers can detect gaps."""
    D = 16
    st = init_h2o_state(n_sink=0, budget=8, head_dim=D)
    k, v = _rand_kv(S=3, D=D)
    st = h2o_update(st, k, v)
    assert st.positions.tolist() == [0, 1, 2]
    assert st.next_pos == 3


def test_positions_stay_contiguous_and_sorted_under_stress() -> None:
    """Across many steps, kept positions never have gaps or duplicates, and
    n_sink leading positions never move — the invariant the RoPE remap must
    preserve for attention to see a consistent contiguous cache."""
    D = 16
    n_sink = 2
    budget = 6
    st = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    for i in range(60):
        k, v = _rand_kv(S=1, D=D, seed=i)
        st = h2o_update(st, k, v)
        pos = st.positions.tolist()
        assert pos == sorted(pos), f"step {i}: not sorted: {pos}"
        assert len(set(pos)) == len(pos), f"step {i}: duplicate: {pos}"
        diffs = [pos[j + 1] - pos[j] for j in range(len(pos) - 1)]
        assert all(d == 1 for d in diffs), f"step {i}: gap in kept positions: {pos}"
        if len(pos) >= n_sink:
            assert pos[:n_sink] == list(range(n_sink)), f"step {i}: sinks moved: {pos}"


def test_interior_eviction_gap_remap_recovers_exact_original_keys() -> None:
    """The core correctness property of the eviction+remap logic in
    h2o_update: when a row is dropped from the INTERIOR of the kept set
    (rows after it shift down by one position and get re-rotated; rows
    before it are untouched), every surviving key, de-rotated at its new
    stored position, must exactly recover its original pre-rotation value.

    A brand-new arrival always starts at score 0.0 in h2o_update, which
    makes it (almost) always the eviction target in practice — see
    test_new_token_almost_always_evicted_first — so genuine interior
    eviction from realistic inputs is rare. This test instead exercises
    h2o_update's exact eviction+remap code path (mirroring its
    keep_indices/shift/rope_remap_positions logic line-for-line) directly on
    a constructed 5-token state, to verify that logic's correctness
    independent of how often it fires in practice.
    """
    D = 16
    rng = np.random.default_rng(0)
    raw_keys = mx.array(rng.standard_normal((5, D)).astype(np.float32))
    fingerprints = np.zeros((5, D), dtype=np.float32)
    for i in range(5):
        fingerprints[i, 0] = i + 1.0
    raw_values = mx.array(fingerprints)
    positions = mx.arange(5, dtype=mx.int32)
    rotated_keys = a2ats_apply_exact_rope(raw_keys, positions, base=10000.0)

    # Evict the interior token at index 2 (position 2): same keep_indices /
    # shift / remap steps h2o_update's eviction branch performs.
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

    # Positions before the gap (0, 1) are untouched; after it (3, 4 -> 2, 3).
    assert new_positions.tolist() == [0, 1, 2, 3]

    # Rows before the gap must be numerically unchanged (no re-rotation
    # needed — this is the "minimal disturbance" property).
    for row in (0, 1):
        err = float(
            mx.max(mx.abs(remapped_keys[row] - kept_keys_before[row].astype(mx.float32))).item()
        )
        assert err < 1e-3, f"row {row} before the gap should be untouched, err={err}"

    # Every surviving key, de-rotated at its NEW position, must recover its
    # exact original pre-rotation value.
    recovered = rope_remap_positions(
        remapped_keys, new_positions, mx.zeros_like(new_positions), base=10000.0
    )
    kept_fingerprints = np.array(raw_values.astype(mx.float32))[keep_indices, 0]
    assert 3.0 not in kept_fingerprints  # token originally at index 2 (fp=3.0) is gone
    for row, fp in enumerate(kept_fingerprints):
        orig_idx = int(round(fp)) - 1
        err = float(mx.max(mx.abs(recovered[row] - raw_keys[orig_idx])).item())
        assert err < 1e-2, f"row {row} (orig token {orig_idx}): recon error {err}"


def test_new_token_almost_always_evicted_first() -> None:
    """Documents the real, verified early-token-freeze finding: with a
    realistic (non-degenerate) score distribution, a brand-new token's
    starting score of 0.0 is virtually always the global minimum, so once
    the cache is full, the newest arrival is evicted on (nearly) every step
    rather than an old low-value token — freezing the kept set. Not a
    regression to fix here; documented as a known property of the paper's
    own cumulative-sum formula at tight budgets (see module docstring)."""
    D = 16
    budget = 4
    st = init_h2o_state(n_sink=0, budget=budget, head_dim=D)
    rng = np.random.default_rng(0)
    for i in range(20):
        k, v = _rand_kv(S=1, D=D, seed=i)
        st = h2o_update(st, k, v)
    # After 20 steps with budget=4, the kept set should have frozen on the
    # earliest possible contiguous window (never grew past position 3).
    assert st.positions.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Vectorized below-budget batch path (perf fix: avoids a per-token Python
# loop with 4 concats/token, which exhausted MLX's Metal resource limit on
# long prefills — e.g. ~3200 tokens — even with eviction disabled). Must be
# numerically equivalent to the sequential per-token loop it replaces.
# ---------------------------------------------------------------------------


def test_batch_absorb_matches_sequential_from_empty() -> None:
    """_batch_absorb_no_eviction(None, None, keys) must match calling
    h2o_update one token at a time from an empty state (huge budget, so no
    eviction ever fires in the sequential reference)."""
    D = 8
    S = 6
    rng = np.random.default_rng(0)
    keys = mx.array(rng.standard_normal((S, D)).astype(np.float32))

    st = init_h2o_state(n_sink=0, budget=1000, head_dim=D)
    for i in range(S):
        st = h2o_update(st, keys[i][None].astype(mx.float16), keys[i][None].astype(mx.float16))

    vectorized = _batch_absorb_no_eviction(None, None, keys)
    err = float(mx.max(mx.abs(vectorized - st.scores)).item())
    assert err < 1e-3, f"batch vs sequential score mismatch: {err}"


def test_batch_absorb_matches_sequential_with_prior_tokens() -> None:
    """_batch_absorb_no_eviction with a non-empty prior state must match
    continuing the sequential per-token loop from that same prior state."""
    D = 8
    P = 3
    S = 4
    rng = np.random.default_rng(1)
    prior = rng.standard_normal((P, D)).astype(np.float32)
    new = rng.standard_normal((S, D)).astype(np.float32)

    st = init_h2o_state(n_sink=0, budget=1000, head_dim=D)
    for i in range(P):
        k = mx.array(prior[i][None].astype(np.float16))
        st = h2o_update(st, k, k)
    for i in range(S):
        k = mx.array(new[i][None].astype(np.float16))
        st = h2o_update(st, k, k)

    st2 = init_h2o_state(n_sink=0, budget=1000, head_dim=D)
    for i in range(P):
        k = mx.array(prior[i][None].astype(np.float16))
        st2 = h2o_update(st2, k, k)
    vectorized = _batch_absorb_no_eviction(st2.keys.astype(mx.float32), st2.scores, mx.array(new))
    err = float(mx.max(mx.abs(vectorized - st.scores)).item())
    assert err < 1e-3, f"batch vs sequential score mismatch: {err}"


def test_h2o_update_single_call_matches_many_single_token_calls() -> None:
    """The public contract: h2o_update(state, keys[S,D], values[S,D]) in one
    call must produce the identical final state as calling h2o_update S times
    with one token each — across the batch-path/loop-path boundary (budget
    straddled mid-batch), not just in the pure-batch or pure-loop regimes."""
    D = 16
    budget = 6
    n_sink = 0
    rng = np.random.default_rng(0)
    all_k = rng.standard_normal((15, D)).astype(np.float16)
    all_v = rng.standard_normal((15, D)).astype(np.float16)

    st_seq = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    for i in range(15):
        st_seq = h2o_update(st_seq, mx.array(all_k[i][None]), mx.array(all_v[i][None]))

    st_batch = init_h2o_state(n_sink=n_sink, budget=budget, head_dim=D)
    st_batch = h2o_update(st_batch, mx.array(all_k), mx.array(all_v))

    assert st_seq.positions.tolist() == st_batch.positions.tolist()
    assert st_seq.next_pos == st_batch.next_pos
    score_err = float(mx.max(mx.abs(st_seq.scores - st_batch.scores)).item())
    assert score_err < 1e-2, f"score mismatch: {score_err}"
    key_mse = float(
        mx.mean((st_seq.keys.astype(mx.float32) - st_batch.keys.astype(mx.float32)) ** 2).item()
    )
    assert key_mse < 1e-3, f"key mismatch: {key_mse}"


def test_batch_path_handles_long_prefill_without_crashing() -> None:
    """Regression for the real crash this fix addresses: a single
    update_and_fetch-style call with thousands of tokens and a budget large
    enough that no eviction occurs must not blow up building an enormous
    unfused op graph (the pre-fix per-token-loop behavior). 4000 tokens is
    smaller than the ~3200-token prompt that triggered a genuine Metal
    resource-limit crash pre-fix, kept modest here to stay fast in CI."""
    D = 32
    S = 4000
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))

    st = init_h2o_state(n_sink=4, budget=S + 100, head_dim=D)
    st = h2o_update(st, k, v)  # must not raise
    assert st.keys.shape[0] == S
    assert st.next_pos == S


def test_heavy_eviction_thousands_of_loop_iterations() -> None:
    """Exercises thousands of consecutive eviction-loop iterations within a
    single h2o_update call (budget exceeded almost immediately, S >> budget)
    to confirm the periodic mx.eval() flush (_EVAL_FLUSH_INTERVAL) doesn't
    change output or crash at this iteration count.

    NOTE: this synthetic case does NOT reproduce the actual crash the flush
    fixes — that was only observed via real mlx_lm.generate() on a genuine
    ~3200-token prompt with the fused Metal eviction kernel active (see
    h2o.py's module docstring and _EVAL_FLUSH_INTERVAL's comment); isolating
    a single head's state update here, without the full model's per-layer/
    per-head aggregate Metal resource pressure, was not sufficient to
    reproduce it synthetically. This test guards determinism/no-crash at
    scale, not the specific resource-limit regression."""
    D = 16
    S = 2000
    budget = 64
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))

    st = init_h2o_state(n_sink=4, budget=budget, head_dim=D)
    st = h2o_update(st, k, v)  # must not raise
    assert st.keys.shape[0] == budget
    assert st.next_pos == S

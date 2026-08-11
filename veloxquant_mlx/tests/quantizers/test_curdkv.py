"""Tests for CurDKV-adapted quantizer primitives — value-aware leverage-score eviction.

CurDKV-adapted (arXiv:2509.15038, NeurIPS 2025) estimates per-token leverage
scores from the proxy attention-weighted value block (using the incoming key
as a proxy query) and evicts the lowest-score non-sink token whenever the
cache exceeds the budget. Unlike H2O-adapted's key-only attention-mass score,
CurDKV's leverage score is a joint (key, value) signal — see
test_identical_keys_different_values_diverge and
test_planted_geometry_curdkv_prefers_value_relevant_tokens, which pin the
new mechanism against H2O's key-only blind spot. Tests cover:
init_curdkv_state, curdkv_update (single token, multi-step, eviction trigger,
sink protection, budget enforcement, score accumulation), curdkv_get_kv
(shape, dtype, empty state), byte accounting, and edge cases (n_sink=0,
budget=1, degenerate all-zero values). All data is synthetic — no model
loading.
"""

from __future__ import annotations

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.curdkv import (
    CurDKVState,
    full_curdkv_fp16_bytes,
    curdkv_fp16_bytes,
    curdkv_get_kv,
    curdkv_update,
    init_curdkv_state,
)
from veloxquant_mlx.quantizers.h2o import h2o_update, init_h2o_state, h2o_get_kv


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init_curdkv_state
# ---------------------------------------------------------------------------


def test_init_state_fields() -> None:
    st = init_curdkv_state(n_sink=4, budget=16, head_dim=64, rank_cap=8)
    assert st.n_sink == 4
    assert st.budget == 16
    assert st.rank_cap == 8
    assert st.keys is None
    assert st.values is None
    assert st.leverage_scores is None


def test_init_state_zero_budget_allowed() -> None:
    st = init_curdkv_state(n_sink=0, budget=0, head_dim=32)
    assert st.budget == 0


def test_init_state_default_rank_cap() -> None:
    st = init_curdkv_state(n_sink=0, budget=8, head_dim=32)
    assert st.rank_cap == 16


def test_init_state_rejects_n_sink_equal_budget() -> None:
    """n_sink >= budget leaves no evictable room — sinks would be evicted
    once the cache fills, defeating the sink guarantee."""
    with pytest.raises(ValueError, match="n_sink"):
        init_curdkv_state(n_sink=4, budget=4, head_dim=32)


def test_init_state_rejects_n_sink_above_budget() -> None:
    with pytest.raises(ValueError, match="n_sink"):
        init_curdkv_state(n_sink=8, budget=4, head_dim=32)


# ---------------------------------------------------------------------------
# curdkv_get_kv — empty state
# ---------------------------------------------------------------------------


def test_get_kv_empty_returns_zero_rows() -> None:
    st = init_curdkv_state(n_sink=4, budget=16, head_dim=32)
    k, v = curdkv_get_kv(st)
    assert k.shape[0] == 0
    assert v.shape[0] == 0


# ---------------------------------------------------------------------------
# curdkv_update — basic absorption
# ---------------------------------------------------------------------------


def test_single_token_absorbed() -> None:
    """First token bootstraps state with one row."""
    D = 32
    st = init_curdkv_state(n_sink=4, budget=16, head_dim=D)
    k, v = _rand_kv(S=1, D=D)
    st = curdkv_update(st, k, v)
    assert st.keys is not None
    assert st.keys.shape == (1, D)
    assert st.values.shape == (1, D)


def test_multi_token_absorbed_below_budget() -> None:
    """S tokens below budget → all kept."""
    D = 32
    budget = 16
    S = 8
    st = init_curdkv_state(n_sink=4, budget=budget, head_dim=D)
    k, v = _rand_kv(S=S, D=D)
    st = curdkv_update(st, k, v)
    assert st.keys.shape[0] == S


def test_output_dtype_fp16() -> None:
    D = 32
    st = init_curdkv_state(n_sink=2, budget=8, head_dim=D)
    k, v = _rand_kv(S=4, D=D)
    st = curdkv_update(st, k, v)
    ko, vo = curdkv_get_kv(st)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


# ---------------------------------------------------------------------------
# curdkv_update — eviction when over budget
# ---------------------------------------------------------------------------


def test_budget_never_exceeded() -> None:
    """After many tokens, kept count never exceeds budget."""
    D = 32
    budget = 8
    n_sink = 2
    st = init_curdkv_state(n_sink=n_sink, budget=budget, head_dim=D)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=i)
        st = curdkv_update(st, ki, vi)
    assert st.keys.shape[0] <= budget


def test_budget_exactly_enforced_at_boundary() -> None:
    """Exactly budget+1 tokens → evict one → budget tokens remain."""
    D = 16
    budget = 5
    st = init_curdkv_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=budget + 1, D=D)
    st = curdkv_update(st, k, v)
    assert st.keys.shape[0] == budget


def test_score_array_length_matches_keys() -> None:
    """leverage_scores array length always equals number of kept tokens."""
    D = 16
    budget = 6
    st = init_curdkv_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = curdkv_update(st, k, v)
    assert st.leverage_scores.shape[0] == st.keys.shape[0]


# ---------------------------------------------------------------------------
# curdkv_update — sink protection
# ---------------------------------------------------------------------------


def test_sinks_never_evicted() -> None:
    """First n_sink tokens are always present in the output, even when their
    value contribution is deliberately made negligible (low leverage)."""
    D = 8
    n_sink = 3
    budget = 4
    # Sink keys look "important" but values are near-zero — low leverage —
    # yet sink protection must keep them regardless of score.
    k_sink = mx.ones((n_sink, D), dtype=mx.float16)
    v_sink = mx.zeros((n_sink, D), dtype=mx.float16)
    st = init_curdkv_state(n_sink=n_sink, budget=budget, head_dim=D)
    st = curdkv_update(st, k_sink, v_sink)

    # Push many more tokens (with real values) to force evictions.
    for i in range(20):
        ki, vi = _rand_kv(S=1, D=D, seed=100 + i)
        st = curdkv_update(st, ki, vi)

    ko, vo = curdkv_get_kv(st)
    assert ko.shape[0] <= budget
    for r in range(min(n_sink, ko.shape[0])):
        assert float(ko[r, 0].item()) == pytest.approx(1.0, abs=1e-2)


def test_n_sink_zero_allows_all_evictions() -> None:
    """With n_sink=0, any token may be evicted."""
    D = 16
    budget = 4
    st = init_curdkv_state(n_sink=0, budget=budget, head_dim=D)
    k, v = _rand_kv(S=20, D=D)
    st = curdkv_update(st, k, v)
    assert st.keys.shape[0] <= budget


# ---------------------------------------------------------------------------
# curdkv_update — leverage-score accumulation
# ---------------------------------------------------------------------------


def test_scores_non_negative() -> None:
    """Cumulative leverage scores are always >= 0."""
    D = 32
    budget = 8
    st = init_curdkv_state(n_sink=2, budget=budget, head_dim=D)
    k, v = _rand_kv(S=8, D=D)
    st = curdkv_update(st, k, v)
    scores_np = np.array(st.leverage_scores.tolist())
    assert np.all(scores_np >= 0.0)


def test_degenerate_all_zero_values_no_nan() -> None:
    """All-zero value block must not produce NaN/inf leverage scores."""
    D = 16
    budget = 6
    st = init_curdkv_state(n_sink=1, budget=budget, head_dim=D)
    k = mx.array(np.random.default_rng(0).standard_normal((8, D)).astype(np.float16))
    v = mx.zeros((8, D), dtype=mx.float16)
    st = curdkv_update(st, k, v)
    scores_np = np.array(st.leverage_scores.tolist())
    assert np.all(np.isfinite(scores_np))


# ---------------------------------------------------------------------------
# The core new-mechanism tests: value-awareness
# ---------------------------------------------------------------------------


def test_identical_keys_different_values_diverge() -> None:
    """Two rows with IDENTICAL keys but DIFFERENT values must receive
    different leverage scores — direct proof the mechanism is value-aware,
    unlike H2O's key-only attention-mass score (which would tie them)."""
    from veloxquant_mlx.quantizers.curdkv import _leverage_scores

    D = 16
    rng = np.random.default_rng(0)
    shared_key = rng.standard_normal(D).astype(np.float32)

    two_keys = mx.array(np.stack([shared_key, shared_key]))
    two_values_divergent = mx.array(
        np.stack(
            [
                5.0 * rng.standard_normal(D).astype(np.float32),
                np.zeros(D, dtype=np.float32),
            ]
        )
    )
    query = mx.array(shared_key)
    scores_divergent = _leverage_scores(query, two_keys, two_values_divergent, rank_cap=16)
    s_np = np.array(scores_divergent.tolist())
    assert s_np[0] != pytest.approx(s_np[1], abs=1e-9), (
        "identical keys with divergent values must not produce identical leverage scores"
    )


def test_planted_geometry_curdkv_prefers_value_relevant_tokens() -> None:
    """Plant two token classes with equal key-similarity to the query proxy:
    class 1 (key-similar, value-relevant/large) and class 2 (key-similar,
    value-irrelevant/near-zero). At a tight budget, CurDKV must retain
    class-1 tokens preferentially. H2O (key-only attention-mass), given the
    SAME keys, cannot distinguish the two classes and evicts near-uniformly.
    Run over multiple seeds — a rate, not one lucky run.
    """
    D = 24
    budget = 6
    n_classes_each = 6
    wins_curdkv = 0
    trials = 8

    for seed in range(trials):
        rng = np.random.default_rng(seed)
        base_direction = rng.standard_normal(D).astype(np.float32)
        base_direction /= np.linalg.norm(base_direction)

        # Both classes share near-identical keys (aligned with base_direction
        # plus small noise) so any key-only scorer treats them alike.
        class1_keys = base_direction[None, :] + 0.01 * rng.standard_normal((n_classes_each, D))
        class2_keys = base_direction[None, :] + 0.01 * rng.standard_normal((n_classes_each, D))

        # Class 1 gets large, output-relevant values; class 2 gets near-zero
        # values (negligible contribution to the attention output).
        class1_values = 5.0 * rng.standard_normal((n_classes_each, D))
        class2_values = 0.001 * rng.standard_normal((n_classes_each, D))

        # Interleave arrival order (class1, class2, class1, class2, ...) so
        # neither class systematically arrives earlier/later — H2O-style
        # eviction is recency-sensitive (new tokens start at score 0), so a
        # block-concatenated arrival order would confound "value" with
        # "recency." Interleaving isolates the value effect.
        keys = np.empty((2 * n_classes_each, D), dtype=np.float16)
        values = np.empty((2 * n_classes_each, D), dtype=np.float16)
        keys[0::2] = class1_keys.astype(np.float16)
        keys[1::2] = class2_keys.astype(np.float16)
        values[0::2] = class1_values.astype(np.float16)
        values[1::2] = class2_values.astype(np.float16)

        st = init_curdkv_state(n_sink=0, budget=budget, head_dim=D)
        st = curdkv_update(st, mx.array(keys), mx.array(values))
        ko, _ = curdkv_get_kv(st)
        kept = np.array(ko.tolist())

        # Count how many kept rows are "close" to a class-1 key (heuristic:
        # nearest by cosine similarity to class1 vs class2 centroid values is
        # not observable post-eviction from keys alone since keys are nearly
        # identical across classes; instead compare against the ORIGINAL
        # value magnitude of survivors by re-matching keys to source rows).
        n_class1_kept = 0
        n_class2_kept = 0
        for row in kept:
            d1 = np.min(np.linalg.norm(class1_keys.astype(np.float16) - row, axis=1))
            d2 = np.min(np.linalg.norm(class2_keys.astype(np.float16) - row, axis=1))
            if d1 < d2:
                n_class1_kept += 1
            else:
                n_class2_kept += 1

        if n_class1_kept > n_class2_kept:
            wins_curdkv += 1

    assert wins_curdkv >= trials * 0.75, (
        f"CurDKV should preferentially retain value-relevant (class-1) tokens "
        f"in at least 75% of trials; got {wins_curdkv}/{trials}"
    )


def test_h2o_blind_spot_on_same_planted_geometry() -> None:
    """On the SAME planted geometry (identical keys across classes, divergent
    values), H2O's key-only attention-mass score cannot distinguish the two
    classes — it evicts near-uniformly regardless of value relevance. This
    demonstrates the baseline actually has the blind spot CurDKV fixes."""
    D = 24
    budget = 6
    n_classes_each = 6
    rng = np.random.default_rng(0)
    base_direction = rng.standard_normal(D).astype(np.float32)
    base_direction /= np.linalg.norm(base_direction)

    class1_keys = base_direction[None, :] + 0.01 * rng.standard_normal((n_classes_each, D))
    class2_keys = base_direction[None, :] + 0.01 * rng.standard_normal((n_classes_each, D))
    class1_values = 5.0 * rng.standard_normal((n_classes_each, D))
    class2_values = 0.001 * rng.standard_normal((n_classes_each, D))

    # Interleaved arrival order — see comment in
    # test_planted_geometry_curdkv_prefers_value_relevant_tokens: a
    # block-concatenated order confounds "value" with "recency" under H2O's
    # recency-sensitive scoring (new tokens start at score 0).
    keys = np.empty((2 * n_classes_each, D), dtype=np.float16)
    values = np.empty((2 * n_classes_each, D), dtype=np.float16)
    keys[0::2] = class1_keys.astype(np.float16)
    keys[1::2] = class2_keys.astype(np.float16)
    values[0::2] = class1_values.astype(np.float16)
    values[1::2] = class2_values.astype(np.float16)

    st = init_h2o_state(n_sink=0, budget=budget, head_dim=D)
    st = h2o_update(st, mx.array(keys), mx.array(values))
    ko, _ = h2o_get_kv(st)
    kept = np.array(ko.tolist())

    n_class1_kept = 0
    n_class2_kept = 0
    for row in kept:
        d1 = np.min(np.linalg.norm(class1_keys.astype(np.float16) - row, axis=1))
        d2 = np.min(np.linalg.norm(class2_keys.astype(np.float16) - row, axis=1))
        if d1 < d2:
            n_class1_kept += 1
        else:
            n_class2_kept += 1

    # H2O's key-only scoring should NOT show a strong preference for class-1
    # (value-relevant) tokens — near-uniform split expected since keys are
    # statistically identical between classes.
    assert abs(n_class1_kept - n_class2_kept) <= budget // 2 + 1, (
        "H2O should not show a strong value-aware preference on identical-key "
        "geometry — its blind spot is exactly the point of this test"
    )


# ---------------------------------------------------------------------------
# SVD non-convergence fallback (real-model regression)
# ---------------------------------------------------------------------------


def test_leverage_scores_falls_back_when_svd_does_not_converge() -> None:
    """np.linalg.svd's ``gesdd`` LAPACK driver has documented, data-dependent
    non-convergence flakiness (observed on Apple's Accelerate backend during
    real-model generation — see
    https://github.com/rajveer43/VeloxQuant-MLX/issues/147 for the crash
    trace: a full mlx_lm.generate() run through CurDKVKVCache died with
    ``numpy.linalg.LinAlgError: SVD did not converge`` on ordinary, non-
    adversarial real K/V activations). This is not reliably reproducible
    with synthetic inputs, so this test forces the failure via a mock and
    checks that ``_leverage_scores`` falls back to scipy's more robust
    ``gesvd`` driver instead of propagating the crash.
    """
    from veloxquant_mlx.quantizers.curdkv import _leverage_scores

    D = 16
    n = 10
    rng = np.random.default_rng(0)
    keys = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    values = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    query = mx.array(rng.standard_normal(D).astype(np.float32))

    with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("SVD did not converge")):
        scores = _leverage_scores(query, keys, values, rank_cap=8)

    scores_np = np.array(scores.tolist())
    assert np.all(np.isfinite(scores_np))
    assert scores_np.sum() == pytest.approx(1.0, abs=1e-4)


def test_leverage_scores_does_not_hang_when_gesvd_fallback_stalls() -> None:
    """The scipy ``gesvd`` fallback (see previous test) is itself not
    guaranteed to return promptly on every input — a real-model validation
    run observed an 11+ minute stall on a single call that normally
    completes in under a second (issue #147). ``_robust_svd`` bounds the
    fallback with a watchdog timeout (``_GESVD_TIMEOUT_S``) so a stalled
    ``gesvd`` call can never hang the eviction loop: on timeout it returns
    an all-zero result, which ``_leverage_scores`` already treats the same
    as its existing degenerate-all-zero-value-block case.

    This test forces BOTH ``np.linalg.svd`` to fail AND the ``gesvd``
    fallback to hang (mocked as an indefinite sleep), and asserts the call
    returns well within the timeout budget instead of hanging, with a
    finite, valid (all-zero) result.
    """
    import time

    from veloxquant_mlx.quantizers.curdkv import _GESVD_TIMEOUT_S, _leverage_scores

    D = 16
    n = 10
    rng = np.random.default_rng(0)
    keys = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    values = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    query = mx.array(rng.standard_normal(D).astype(np.float32))

    def _hang_forever(*_a, **_k):
        time.sleep(_GESVD_TIMEOUT_S * 100)
        raise AssertionError("should never actually finish sleeping in this test")

    t0 = time.time()
    with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("SVD did not converge")):
        with patch("scipy.linalg.svd", side_effect=_hang_forever):
            scores = _leverage_scores(query, keys, values, rank_cap=8)
    elapsed = time.time() - t0

    assert elapsed < _GESVD_TIMEOUT_S * 2, (
        f"_leverage_scores took {elapsed:.2f}s — should return near the "
        f"{_GESVD_TIMEOUT_S}s watchdog budget, not hang indefinitely"
    )
    scores_np = np.array(scores.tolist())
    assert np.all(np.isfinite(scores_np))
    assert np.all(scores_np == 0.0)


# ---------------------------------------------------------------------------
# Age-bias regression: newcomer vs. long-lived survivor eviction fairness
# ---------------------------------------------------------------------------


def test_high_value_newcomer_not_evicted_on_arrival() -> None:
    """A brand-new token with a value vector far larger (more
    output-relevant) than anything already cached must not be evicted on the
    very step it arrives, just because existing survivors have had many
    prior steps to accumulate leverage score.

    Regression test: comparing a newcomer's single-step self-leverage score
    directly against survivors' many-step CUMULATIVE scores is a scale
    mismatch — the cumulative sum grows with age regardless of value, so an
    old-but-mild survivor could out-score a new-but-critical token purely by
    having been in the cache longer. Eviction must compare MEAN per-step
    leverage (cumulative score / steps survived) instead.

    Identifies the newcomer by ``n_updates == 1`` (it has been through
    exactly one accumulation step) rather than matching raw key values: once
    RoPE-remapping was added (see the FIXED, CONFIRMED-ON-REAL-MODELS
    PROBLEM in the module docstring), a survivor whose storage index shifts
    due to an unrelated eviction gets its key rotated to a new position, so
    exact fp16 key-value matching is no longer a reliable way to track "the
    same logical token" across an update call.
    """
    D = 16
    budget = 4
    st = init_curdkv_state(n_sink=0, budget=budget, head_dim=D)
    rng = np.random.default_rng(3)

    # Fill the cache and let mild-value survivors accumulate score over many
    # steps, so their raw cumulative sums grow large purely from age.
    for _ in range(24):
        ki = mx.array(rng.standard_normal((1, D)).astype(np.float16)) * 0.1
        vi = mx.array(rng.standard_normal((1, D)).astype(np.float16)) * 0.1
        st = curdkv_update(st, ki, vi)

    # A single newcomer with a value vector 1000x larger than anything seen
    # so far — should clearly deserve retention over the mild survivors.
    k_important = mx.array(rng.standard_normal((1, D)).astype(np.float16)) * 0.1
    v_important = mx.array(rng.standard_normal((1, D)).astype(np.float16)) * 100.0
    st = curdkv_update(st, k_important, v_important)

    n_updates = np.array(st.n_updates.tolist())
    newcomer_rows = np.where(n_updates == 1.0)[0]
    assert len(newcomer_rows) == 1, (
        "high-value newcomer was evicted purely due to accumulation-age bias"
    )


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_curdkv_fp16_bytes_formula() -> None:
    """curdkv_fp16_bytes = n_kept * D * 4 (K + V, fp16)."""
    D = 64
    budget = 16
    st = init_curdkv_state(n_sink=4, budget=budget, head_dim=D)
    k, v = _rand_kv(S=10, D=D)
    st = curdkv_update(st, k, v)
    n_kept = st.keys.shape[0]
    expected = n_kept * D * 2 * 2
    assert curdkv_fp16_bytes(st) == expected


def test_curdkv_fp16_bytes_empty_state() -> None:
    st = init_curdkv_state(n_sink=4, budget=16, head_dim=64)
    assert curdkv_fp16_bytes(st) == 0


def test_full_curdkv_fp16_bytes_formula() -> None:
    assert full_curdkv_fp16_bytes(100, 128) == 100 * 128 * 2 * 2


# ---------------------------------------------------------------------------
# Multi-step decode stress + determinism
# ---------------------------------------------------------------------------


def test_30_step_stress_budget_constant() -> None:
    """30 single-token decode steps — budget never exceeded."""
    D = 32
    budget = 10
    n_sink = 3
    st = init_curdkv_state(n_sink=n_sink, budget=budget, head_dim=D)
    k_s, v_s = _rand_kv(S=n_sink, D=D, seed=0)
    st = curdkv_update(st, k_s, v_s)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=10 + i)
        st = curdkv_update(st, ki, vi)
        assert st.keys.shape[0] <= budget, f"step {i}: {st.keys.shape[0]} > {budget}"


def test_deterministic_across_identical_inputs() -> None:
    """Same input → same state (no stochastic elements)."""
    D = 32
    budget = 8
    k, v = _rand_kv(S=12, D=D, seed=42)

    st_a = init_curdkv_state(n_sink=2, budget=budget, head_dim=D)
    st_a = curdkv_update(st_a, k, v)

    st_b = init_curdkv_state(n_sink=2, budget=budget, head_dim=D)
    st_b = curdkv_update(st_b, k, v)

    ka, _ = curdkv_get_kv(st_a)
    kb, _ = curdkv_get_kv(st_b)
    mse = float(mx.mean((ka.astype(mx.float32) - kb.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)

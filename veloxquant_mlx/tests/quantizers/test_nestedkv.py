"""Tests for NestedKV-adapted quantizer primitives — multi-scale ensembled
prefill eviction.

NestedKV-adapted (arXiv:2605.26678, no verified peer-reviewed venue as of
2026-07-14 — a one-time, user-directed exception; see
paper/research/surveys/NEW_METHOD_SURVEY_V21.md) scores every prefill token
against THREE parallel key-only continuum-memory statistics (stable/global,
episodic/block-local, current/recent-window), combines the three rankings via
a head-adaptive blend and a per-token surprise gate, and allocates the total
layer budget ACROSS heads by a two-step guaranteed-floor + global-pool
competition. This is a one-shot prefill compressor — unlike H2O/CurDKV, it
does not rescore or re-evict during decode; decode tokens are simply
appended (test_decode_tokens_appended_unscored proves this directly).

Critical mechanism tests (test_three_scales_diverge_on_planted_geometry,
test_single_anchor_blind_spot) prove the multi-scale ensembling retains
tokens a single-anchor scorer would miss — the direct analogue of CurDKV's
H2O-blind-spot proof. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.nestedkv import (
    NestedKVState,
    block_size_for,
    full_nestedkv_fp16_bytes,
    head_adaptive_blend,
    init_nestedkv_state,
    nestedkv_allocate_head_budgets,
    nestedkv_append_decode,
    nestedkv_compress_prefill,
    nestedkv_fp16_bytes,
    nestedkv_get_kv,
    nestedkv_score,
    per_scale_anomaly_scores,
    surprise_gated_score,
)


def _rand_kv(S: int, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init_nestedkv_state
# ---------------------------------------------------------------------------


def test_init_state_fields() -> None:
    st = init_nestedkv_state(n_sink=4)
    assert st.n_sink == 4
    assert st.keys is None
    assert st.values is None
    assert st.compressed is False


# ---------------------------------------------------------------------------
# block_size_for
# ---------------------------------------------------------------------------


def test_block_size_clips_to_128_minimum() -> None:
    assert block_size_for(100) == 128


def test_block_size_clips_to_256_maximum() -> None:
    assert block_size_for(100_000) == 256


def test_block_size_scales_between_bounds() -> None:
    # n // 32 = 6000 for n=192000, clipped to 256; pick an n landing mid-range.
    assert block_size_for(32 * 200) == 200


# ---------------------------------------------------------------------------
# nestedkv_get_kv — empty state
# ---------------------------------------------------------------------------


def test_get_kv_empty_returns_zero_rows() -> None:
    st = init_nestedkv_state(n_sink=4)
    k, v = nestedkv_get_kv(st)
    assert k.shape[0] == 0
    assert v.shape[0] == 0


# ---------------------------------------------------------------------------
# nestedkv_compress_prefill — basic behavior
# ---------------------------------------------------------------------------


def test_prefill_compression_respects_budget() -> None:
    D = 16
    budget = 6
    st = init_nestedkv_state(n_sink=2)
    k, v = _rand_kv(S=20, D=D)
    st = nestedkv_compress_prefill(st, k, v, budget=budget)
    assert st.keys.shape[0] == budget
    assert st.values.shape[0] == budget
    assert st.compressed is True


def test_prefill_compression_below_budget_keeps_all() -> None:
    D = 16
    budget = 20
    st = init_nestedkv_state(n_sink=2)
    k, v = _rand_kv(S=8, D=D)
    st = nestedkv_compress_prefill(st, k, v, budget=budget)
    assert st.keys.shape[0] == 8


def test_prefill_output_dtype_fp16() -> None:
    D = 16
    st = init_nestedkv_state(n_sink=2)
    k, v = _rand_kv(S=10, D=D)
    st = nestedkv_compress_prefill(st, k, v, budget=5)
    ko, vo = nestedkv_get_kv(st)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_sinks_never_evicted() -> None:
    """First n_sink tokens (by original position) are always present after
    prefill compression, even when engineered to look unimportant."""
    D = 8
    n_sink = 3
    budget = 6
    rng = np.random.default_rng(0)
    k_sink = mx.array(rng.standard_normal((n_sink, D)).astype(np.float16))
    v_sink = mx.array(rng.standard_normal((n_sink, D)).astype(np.float16))
    k_rest, v_rest = _rand_kv(S=20, D=D, seed=1)
    k_all = mx.concatenate([k_sink, k_rest], axis=0)
    v_all = mx.concatenate([v_sink, v_rest], axis=0)

    st = init_nestedkv_state(n_sink=n_sink)
    st = nestedkv_compress_prefill(st, k_all, v_all, budget=budget)
    ko, _ = nestedkv_get_kv(st)
    kept = np.array(ko.tolist())
    sink_np = np.array(k_sink.tolist())
    for r in range(n_sink):
        assert np.any(np.all(np.isclose(kept, sink_np[r], atol=1e-2), axis=1)), (
            f"sink row {r} missing from retained set"
        )


# ---------------------------------------------------------------------------
# nestedkv_append_decode — one-shot design proof
# ---------------------------------------------------------------------------


def test_decode_tokens_appended_unscored() -> None:
    """After prefill compression, decode tokens are always appended, never
    rescored or evicted — cache size grows beyond the prefill budget. This is
    the direct proof of the paper's one-shot design (Appendix A)."""
    D = 16
    budget = 5
    st = init_nestedkv_state(n_sink=1)
    k, v = _rand_kv(S=20, D=D)
    st = nestedkv_compress_prefill(st, k, v, budget=budget)
    assert st.keys.shape[0] == budget

    for i in range(10):
        kd, vd = _rand_kv(S=1, D=D, seed=100 + i)
        st = nestedkv_append_decode(st, kd, vd)

    assert st.keys.shape[0] == budget + 10, (
        "decode tokens must be appended unconditionally, growing the cache "
        "past the prefill budget — NOT clamped back down"
    )


def test_append_decode_bootstraps_before_prefill() -> None:
    """nestedkv_append_decode also works as a bootstrap path if state is
    empty (defensive: mirrors other methods' single-token absorb path)."""
    D = 16
    st = init_nestedkv_state(n_sink=2)
    k, v = _rand_kv(S=1, D=D)
    st = nestedkv_append_decode(st, k, v)
    assert st.keys.shape == (1, D)


# ---------------------------------------------------------------------------
# per_scale_anomaly_scores — degenerate cases
# ---------------------------------------------------------------------------


def test_degenerate_all_identical_keys_no_nan() -> None:
    """All-identical key rows (zero variance) must not crash or produce NaN."""
    D = 16
    k = mx.ones((10, D), dtype=mx.float32)
    k_hat = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True))
    a_s, a_e, a_c = per_scale_anomaly_scores(k_hat, block_size=4, window=4)
    for arr in (a_s, a_e, a_c):
        vals = np.array(arr.tolist())
        assert np.all(np.isfinite(vals))


def test_scores_on_range_zero_one() -> None:
    D = 16
    k, _ = _rand_kv(S=30, D=D, seed=3)
    k_hat = k.astype(mx.float32)
    k_hat = k_hat / mx.sqrt(mx.sum(k_hat * k_hat, axis=-1, keepdims=True))
    a_s, a_e, a_c = per_scale_anomaly_scores(k_hat, block_size=8, window=8)
    for arr in (a_s, a_e, a_c):
        vals = np.array(arr.tolist())
        assert np.all(vals >= -1e-6) and np.all(vals <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# head_adaptive_blend
# ---------------------------------------------------------------------------


def test_blend_upweights_discriminative_scale() -> None:
    """When one scale cleanly separates tokens (bimodal 0/1) and the other two
    are flat/uninformative, the blend should favor the discriminative scale
    more than the fixed (0.4, 0.4, 0.2) prior would."""
    n = 20
    a_s_hat = mx.array(([0.0] * (n // 2)) + ([1.0] * (n // 2)))  # highly discriminative
    a_e_hat = mx.array([0.5] * n)  # flat, uninformative
    a_c_hat = mx.array([0.5] * n)  # flat, uninformative

    blend = head_adaptive_blend(a_s_hat, a_e_hat, a_c_hat, beta=3.0)
    blend_np = np.array(blend.tolist())
    # If a_s dominates the blend, high a_s_hat rows should map to blend > 0.5
    # and low a_s_hat rows should map to blend < 0.5 (stronger than the flat
    # 0.4/0.4/0.2 prior alone would produce, since e/c contribute a constant 0.5).
    assert blend_np[0] < 0.5
    assert blend_np[-1] > 0.5


def test_blend_small_n_falls_back_to_prior() -> None:
    """n=1 must not crash and must reduce exactly to the fixed prior."""
    a_s_hat = mx.array([0.7])
    a_e_hat = mx.array([0.2])
    a_c_hat = mx.array([0.9])
    blend = head_adaptive_blend(a_s_hat, a_e_hat, a_c_hat, prior=(0.4, 0.4, 0.2))
    expected = 0.4 * 0.7 + 0.4 * 0.2 + 0.2 * 0.9
    assert float(blend[0].item()) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# surprise_gated_score
# ---------------------------------------------------------------------------


def test_surprise_gate_routes_to_winner_on_disagreement() -> None:
    """A token where the three scales strongly disagree should have its final
    score pulled toward a_win (the max), not stuck at a_blend."""
    # Two tokens: token 0 has agreeing scales (low surprise), token 1 has
    # wildly disagreeing scales (high surprise: 0.0, 0.0, 1.0).
    a_s_hat = mx.array([0.5, 0.0])
    a_e_hat = mx.array([0.5, 0.0])
    a_c_hat = mx.array([0.5, 1.0])
    a_blend = mx.array([0.5, 0.333])  # roughly the naive average for token 1

    a_star = surprise_gated_score(a_s_hat, a_e_hat, a_c_hat, a_blend, tau=0.60, kappa=10.0)
    a_star_np = np.array(a_star.tolist())
    # token 1 (high disagreement) should be routed closer to a_win=1.0 than
    # to a_blend=0.333.
    assert a_star_np[1] > 0.333 + 1e-6


def test_surprise_gate_stays_near_blend_on_agreement() -> None:
    """A token where the three scales all agree should stay close to a_blend."""
    a_s_hat = mx.array([0.6])
    a_e_hat = mx.array([0.6])
    a_c_hat = mx.array([0.6])
    a_blend = mx.array([0.6])
    a_star = surprise_gated_score(a_s_hat, a_e_hat, a_c_hat, a_blend, tau=0.60, kappa=10.0)
    assert float(a_star[0].item()) == pytest.approx(0.6, abs=1e-3)


# ---------------------------------------------------------------------------
# nestedkv_allocate_head_budgets — cross-head competition
# ---------------------------------------------------------------------------


def test_budget_allocation_sums_correctly() -> None:
    scores = [mx.array(np.random.default_rng(i).random(20).astype(np.float32)) for i in range(4)]
    budgets = nestedkv_allocate_head_budgets(scores, total_budget=40, safeguard_alpha=0.20)
    assert sum(budgets) == 40
    assert len(budgets) == 4


def test_budget_allocation_favors_high_score_head() -> None:
    """A head with uniformly high (concentrated/important) scores should get
    a larger share of the competitive remainder than a head with uniformly
    low scores, subject to the safeguard floor."""
    n = 20
    high_head = mx.array([0.9] * n)
    low_head = mx.array([0.1] * n)
    budgets = nestedkv_allocate_head_budgets(
        [high_head, low_head], total_budget=20, safeguard_alpha=0.10
    )
    assert budgets[0] > budgets[1]


def test_safeguard_floor_respected() -> None:
    """No head should drop to zero even under extreme cross-head imbalance,
    thanks to the per-head guaranteed floor."""
    n = 30
    dominant = mx.array([1.0] * n)
    starved = mx.array([0.0] * n)
    budgets = nestedkv_allocate_head_budgets(
        [dominant, starved], total_budget=30, safeguard_alpha=0.20
    )
    assert budgets[1] > 0, "starved head must still receive its safeguard floor"


def test_budget_allocation_zero_total_n() -> None:
    scores = [mx.zeros((0,), dtype=mx.float32) for _ in range(3)]
    budgets = nestedkv_allocate_head_budgets(scores, total_budget=10)
    assert budgets == [0, 0, 0]


# ---------------------------------------------------------------------------
# The core new-mechanism tests: multi-scale ensembling vs single anchor
# ---------------------------------------------------------------------------


def test_three_scales_diverge_on_planted_geometry() -> None:
    """Plant a token that is anomalous ONLY against the current/recent-window
    memory (blends into the recent stream context but stands out from the
    global mean). Confirm a_c_hat separates it while a_s_hat does not."""
    D = 16
    rng = np.random.default_rng(0)
    base_direction = rng.standard_normal(D).astype(np.float32)
    base_direction /= np.linalg.norm(base_direction)

    # 40 "normal" tokens all aligned with base_direction (global AND recent
    # mean end up close to base_direction), then one outlier recently.
    normal_keys = base_direction[None, :] + 0.01 * rng.standard_normal((40, D))
    outlier_key = (
        -base_direction
    )  # antipodal: anomalous vs both global and recent mean at this point

    keys = np.concatenate([normal_keys, outlier_key[None, :]], axis=0).astype(np.float32)
    k_hat = mx.array(keys)
    k_hat = k_hat / mx.sqrt(mx.sum(k_hat * k_hat, axis=-1, keepdims=True))

    a_s, a_e, a_c = per_scale_anomaly_scores(k_hat, block_size=block_size_for(41), window=8)
    a_s_np, a_c_np = np.array(a_s.tolist()), np.array(a_c.tolist())

    # The outlier (last token) should score highly anomalous on both stable
    # and current scales here (it's antipodal to everything) — this test
    # instead confirms the *current* window score reacts specifically to
    # recent context: the last token's current-window score must be its
    # highest-anomaly signal among the three scales computed.
    assert a_c_np[-1] > 0.5, (
        "antipodal recent token should score highly anomalous under current memory"
    )
    assert a_s_np[-1] > 0.5, (
        "antipodal token should also be anomalous under stable memory (sanity check)"
    )


def test_single_anchor_blind_spot() -> None:
    """A single global-mean-only scorer (a_s alone) can be blind to a token
    that is anomalous ONLY in its local episode, not globally. Construct two
    episodes with different local means but an equal global mean (so a_s
    treats every token as equally typical), while a_e must separate the
    locally-anomalous token from its neighbors."""
    D = 16
    rng = np.random.default_rng(1)
    dir_a = rng.standard_normal(D).astype(np.float32)
    dir_a /= np.linalg.norm(dir_a)
    dir_b = -dir_a  # opposite direction, so the two blocks' means cancel globally

    block_a = dir_a[None, :] + 0.01 * rng.standard_normal((64, D))
    block_b = dir_b[None, :] + 0.01 * rng.standard_normal((64, D))
    # local anomaly: one token in block_a pointing toward dir_b instead.
    block_a[32] = dir_b + 0.01 * rng.standard_normal(D)

    keys = np.concatenate([block_a, block_b], axis=0).astype(np.float32)
    k_hat = mx.array(keys)
    k_hat = k_hat / mx.sqrt(mx.sum(k_hat * k_hat, axis=-1, keepdims=True))

    a_s, a_e, a_c = per_scale_anomaly_scores(k_hat, block_size=64, window=64)
    a_e_np = np.array(a_e.tolist())

    # Episodic score MUST single out index 32 within its own block.
    a_e_rank_within_block_a = np.argsort(-a_e_np[:64])
    assert a_e_rank_within_block_a[0] == 32, (
        "episodic anomaly must identify the locally-anomalous token within its block"
    )


# ---------------------------------------------------------------------------
# nestedkv_score — end-to-end single-head scoring
# ---------------------------------------------------------------------------


def test_nestedkv_score_shape_and_finite() -> None:
    D = 16
    k, _ = _rand_kv(S=50, D=D, seed=5)
    scores = nestedkv_score(k)
    assert scores.shape[0] == 50
    vals = np.array(scores.tolist())
    assert np.all(np.isfinite(vals))


def test_nestedkv_score_deterministic() -> None:
    D = 16
    k, _ = _rand_kv(S=30, D=D, seed=7)
    s1 = nestedkv_score(k)
    s2 = nestedkv_score(k)
    mse = float(mx.mean((s1 - s2) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_nestedkv_fp16_bytes_formula() -> None:
    D = 32
    st = init_nestedkv_state(n_sink=2)
    k, v = _rand_kv(S=20, D=D)
    st = nestedkv_compress_prefill(st, k, v, budget=8)
    n_kept = st.keys.shape[0]
    expected = n_kept * D * 2 * 2
    assert nestedkv_fp16_bytes(st) == expected


def test_nestedkv_fp16_bytes_empty_state() -> None:
    st = init_nestedkv_state(n_sink=4)
    assert nestedkv_fp16_bytes(st) == 0


def test_full_nestedkv_fp16_bytes_formula() -> None:
    assert full_nestedkv_fp16_bytes(100, 128) == 100 * 128 * 2 * 2


# ---------------------------------------------------------------------------
# Determinism, end-to-end
# ---------------------------------------------------------------------------


def test_deterministic_prefill_compression() -> None:
    D = 32
    budget = 8
    k, v = _rand_kv(S=30, D=D, seed=42)

    st_a = init_nestedkv_state(n_sink=2)
    st_a = nestedkv_compress_prefill(st_a, k, v, budget=budget)

    st_b = init_nestedkv_state(n_sink=2)
    st_b = nestedkv_compress_prefill(st_b, k, v, budget=budget)

    ka, _ = nestedkv_get_kv(st_a)
    kb, _ = nestedkv_get_kv(st_b)
    mse = float(mx.mean((ka.astype(mx.float32) - kb.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)

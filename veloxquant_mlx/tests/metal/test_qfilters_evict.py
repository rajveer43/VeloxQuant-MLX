"""Metal-kernel tests for Q-Filters fused eviction.

The kernels must be a drop-in replacement for the pure-MLX selection in
``qfilters_update``, so nearly every test here is a parity check against a
numpy reference rather than a property check — a kernel that is merely
"reasonable" but disagrees with the reference path is a silent correctness
bug.

Covers:
  - projection scores match ``sign * (keys @ filter_dir)`` in fp32
  - sink / recent rows are forced to +inf and always survive
  - evicted set matches an argsort reference, keys and values together
  - kept rows stay in original temporal order
  - budget is respected exactly even when scores tie (the overflow trap)
  - sign=-1 inverts the selection
  - guards: over-budget precondition, budget ceiling, shape mismatches
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="requires Metal GPU")

from veloxquant_mlx.metal._qfilters_evict import (  # noqa: E402
    QFILTERS_MAX_BUDGET,
    qfilters_fused_evict,
    qfilters_score,
)


def _inputs(bh: int, n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((bh, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((bh, n, d)).astype(np.float16))
    f = rng.standard_normal((bh, d))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return k, v, mx.array(f.astype(np.float32))


def _reference_keep(scores: np.ndarray, budget: int) -> np.ndarray:
    """Indices the pure-MLX path keeps: top-`budget`, restored to temporal order."""
    order = np.argsort(scores, kind="stable")
    return np.sort(order[len(scores) - budget :])


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def test_scores_match_matmul_reference() -> None:
    k, _, f = _inputs(3, 40, 16, seed=1)
    got = np.array(qfilters_score(k, f))

    ref = np.array(k, dtype=np.float32) @ np.array(f, dtype=np.float32)[..., None]
    ref = ref.squeeze(-1)
    assert np.allclose(got, ref, atol=1e-3)


def test_score_sign_inverts() -> None:
    k, _, f = _inputs(2, 32, 8, seed=2)
    pos = np.array(qfilters_score(k, f, sign=1))
    neg = np.array(qfilters_score(k, f, sign=-1))
    assert np.allclose(pos, -neg, atol=1e-4)


def test_protected_rows_score_infinite() -> None:
    k, _, f = _inputs(2, 32, 8, seed=3)
    s = np.array(qfilters_score(k, f, n_sink=3, recent=4))

    assert np.all(np.isinf(s[:, :3])), "sinks must be +inf"
    assert np.all(np.isinf(s[:, -4:])), "recent window must be +inf"
    assert np.all(np.isfinite(s[:, 3:-4]))


def test_score_rejects_filter_shape_mismatch() -> None:
    k, _, _ = _inputs(2, 16, 8, seed=4)
    with pytest.raises(ValueError, match="does not match"):
        qfilters_score(k, mx.zeros((2, 4), dtype=mx.float32))


def test_score_rejects_bad_sign() -> None:
    k, _, f = _inputs(2, 16, 8, seed=5)
    with pytest.raises(ValueError, match=r"sign must be \+1 or -1"):
        qfilters_score(k, f, sign=0)


# ------------------------------------------------------------------
# Eviction parity
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bh", "n", "d", "budget", "n_sink", "recent"),
    [
        (1, 32, 8, 16, 0, 0),
        (3, 64, 16, 20, 2, 3),
        (2, 128, 32, 64, 4, 0),
        (4, 50, 8, 49, 1, 1),  # evict exactly one row
    ],
)
def test_eviction_matches_reference(bh, n, d, budget, n_sink, recent) -> None:
    k, v, f = _inputs(bh, n, d, seed=10)
    ko, vo, so = qfilters_fused_evict(k, v, f, budget=budget, n_sink=n_sink, recent=recent)

    assert ko.shape == (bh, budget, d)
    assert vo.shape == (bh, budget, d)
    assert so.shape == (bh, budget)

    scores = np.array(qfilters_score(k, f, n_sink=n_sink, recent=recent))
    for g in range(bh):
        keep = _reference_keep(scores[g], budget)
        assert np.array_equal(np.array(k)[g][keep], np.array(ko)[g]), f"keys differ (group {g})"
        assert np.array_equal(np.array(v)[g][keep], np.array(vo)[g]), f"values differ (group {g})"


def test_protected_rows_always_survive() -> None:
    bh, n, d, budget = 3, 64, 16, 24
    k, v, f = _inputs(bh, n, d, seed=11)
    n_sink, recent = 3, 5

    ko, _, _ = qfilters_fused_evict(k, v, f, budget=budget, n_sink=n_sink, recent=recent)
    kept = np.array(ko)
    src = np.array(k)

    for g in range(bh):
        # Sinks land at the front, the recent window at the back, because kept
        # rows preserve temporal order.
        assert np.array_equal(kept[g][:n_sink], src[g][:n_sink])
        assert np.array_equal(kept[g][-recent:], src[g][-recent:])


def test_kept_rows_stay_in_temporal_order() -> None:
    """Scores of kept rows need not be sorted — arrival order is preserved."""
    bh, n, d, budget = 2, 48, 8, 20
    k, v, f = _inputs(bh, n, d, seed=12)
    ko, _, _ = qfilters_fused_evict(k, v, f, budget=budget)

    scores = np.array(qfilters_score(k, f))
    src, kept = np.array(k), np.array(ko)
    for g in range(bh):
        keep = _reference_keep(scores[g], budget)
        assert list(keep) == sorted(keep)
        assert np.array_equal(src[g][keep], kept[g])


def test_sign_negative_selects_opposite_rows() -> None:
    bh, n, d, budget = 2, 40, 8, 12
    k, v, f = _inputs(bh, n, d, seed=13)

    ko_pos, _, _ = qfilters_fused_evict(k, v, f, budget=budget, sign=1)
    ko_neg, _, _ = qfilters_fused_evict(k, v, f, budget=budget, sign=-1)

    scores = np.array(qfilters_score(k, f, sign=-1))
    for g in range(bh):
        keep = _reference_keep(scores[g], budget)
        assert np.array_equal(np.array(k)[g][keep], np.array(ko_neg)[g])
    # The two arms disagree — otherwise the sign knob would be inert.
    assert not np.array_equal(np.array(ko_pos), np.array(ko_neg))


def test_tied_scores_do_not_overflow_budget() -> None:
    """The tie trap: `score >= thresh` would keep more than `budget` rows.

    All-identical keys make every score equal, so a naive threshold admits
    every row. The kept set must still be exactly `budget` rows.
    """
    bh, n, d, budget = 2, 64, 8, 16
    k = mx.ones((bh, n, d), dtype=mx.float16)
    v = mx.ones((bh, n, d), dtype=mx.float16)
    f = mx.zeros((bh, d), dtype=mx.float32) + (1.0 / np.sqrt(d)).astype(np.float32)

    ko, _, _ = qfilters_fused_evict(k, v, f, budget=budget)
    assert ko.shape == (bh, budget, d)
    assert np.all(np.array(ko) == 1.0), "no uninitialized padding may leak in"


def test_all_protected_rows_tie_at_infinity() -> None:
    """Protected rows all score +inf; the budget must still be respected."""
    bh, n, d, budget = 2, 40, 8, 20
    k, v, f = _inputs(bh, n, d, seed=14)
    # 10 sinks + 10 recent == budget: the kept set is entirely protected rows.
    ko, _, _ = qfilters_fused_evict(k, v, f, budget=budget, n_sink=10, recent=10)

    assert ko.shape == (bh, budget, d)
    src, kept = np.array(k), np.array(ko)
    for g in range(bh):
        assert np.array_equal(kept[g][:10], src[g][:10])
        assert np.array_equal(kept[g][10:], src[g][-10:])


# ------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------


def test_rejects_group_not_over_budget() -> None:
    k, v, f = _inputs(2, 16, 8, seed=15)
    with pytest.raises(ValueError, match="only invoke this when the group is over budget"):
        qfilters_fused_evict(k, v, f, budget=16)


def test_rejects_budget_above_ceiling() -> None:
    k, v, f = _inputs(1, 8, 4, seed=16)
    with pytest.raises(ValueError, match="QFILTERS_MAX_BUDGET"):
        qfilters_fused_evict(k, v, f, budget=QFILTERS_MAX_BUDGET + 1)


def test_rejects_nonpositive_budget() -> None:
    k, v, f = _inputs(1, 8, 4, seed=17)
    with pytest.raises(ValueError, match="budget must be > 0"):
        qfilters_fused_evict(k, v, f, budget=0)


def test_rejects_mismatched_values() -> None:
    k, _, f = _inputs(2, 16, 8, seed=18)
    with pytest.raises(ValueError, match="must match"):
        qfilters_fused_evict(k, mx.zeros((2, 16, 4), dtype=mx.float16), f, budget=8)


def test_rejects_protection_exceeding_budget() -> None:
    k, v, f = _inputs(2, 32, 8, seed=19)
    with pytest.raises(ValueError, match="exceeds"):
        qfilters_fused_evict(k, v, f, budget=8, n_sink=6, recent=6)


def test_rejects_non_3d_keys() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        qfilters_fused_evict(
            mx.zeros((16, 8), dtype=mx.float16),
            mx.zeros((16, 8), dtype=mx.float16),
            mx.zeros((1, 8), dtype=mx.float32),
            budget=4,
        )

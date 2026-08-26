"""Unit tests for Q-Filters offline query-SVD calibration (paper §3.2, Eq. 1).

The point of this module is the sign. The key-SVD fallback in
``quantizers/qfilters.py`` recovers the dominant axis but not its orientation;
the paper's query-SVD recovers both, because Theorem 3.3's ``kappa^h > 0`` is
a property of the query distribution. These tests pin that difference down
rather than merely checking shapes.

Covers:
  - compute_qfilters recovers a planted query-drift direction WITH its sign
  - the sign survives when the planted direction is negated (the case the
    key-SVD path cannot resolve)
  - no mean-centering: drift, not variance, determines the filter
  - deterministic subsampling — same activations give the same filters
  - GQA group-averaging shape, unit norm, and divisibility guard
  - artifact round-trip, model-id guard, and version/corruption failures
  - input validation
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.qfilters import estimate_filter_dir
from veloxquant_mlx.quantizers.qfilters_calibration import (
    QFILTERS_ARTIFACT_VERSION,
    QFiltersCalibration,
    average_gqa_filters,
    compute_qfilters,
    load_qfilters,
    save_qfilters,
)


def _planted_queries(h: int, n: int, d: int, direction: np.ndarray, drift: float, seed: int = 0):
    """Query cloud drifting along ``direction`` (paper Observation 3.1)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((h, n, d)).astype(np.float32) * 0.5
    return mx.array((q + drift * direction).astype(np.float32))


def _unit(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d)
    return (u / np.linalg.norm(u)).astype(np.float32)


# ------------------------------------------------------------------
# The sign — the whole reason this module exists
# ------------------------------------------------------------------


def test_recovers_planted_direction_with_sign() -> None:
    u = _unit(16, seed=1)
    filters = compute_qfilters(_planted_queries(3, 2000, 16, u, drift=3.0))

    assert filters.shape == (3, 16)
    for h in range(3):
        cos = float(np.dot(np.array(filters[h]), u))
        # Signed, not |cos| — recovering +u rather than -u is the point.
        assert cos > 0.99, f"head {h}: cos={cos:.4f} (wrong sign or axis)"


def test_sign_follows_a_negated_drift_direction() -> None:
    """Filters must track the drift's orientation, not just its axis."""
    u = _unit(16, seed=2)
    pos = compute_qfilters(_planted_queries(1, 2000, 16, u, drift=3.0))
    neg = compute_qfilters(_planted_queries(1, 2000, 16, -u, drift=3.0))

    assert float(np.dot(np.array(pos[0]), u)) > 0.99
    assert float(np.dot(np.array(neg[0]), -u)) > 0.99
    # The two filters point in opposite directions — the orientation is real
    # information, not an artifact of the SVD's arbitrary sign convention.
    assert float(np.dot(np.array(pos[0]), np.array(neg[0]))) < -0.99


def test_query_svd_resolves_sign_where_key_svd_cannot() -> None:
    """The documented contrast with the key-SVD fallback estimator.

    Keys commonly drift along ``-u`` (the paper's ``epsilon = -1``, observed
    for "a vast majority of heads"). The query-SVD filter lands on ``+u``
    sign and all. The key-SVD fallback mean-centers, which subtracts that
    drift, so it can at best recover the *axis* from the keys' spread — never
    the orientation. Here the keys are given genuine anisotropy along ``u`` so
    the fallback has an axis to find at all; even then its sign is arbitrary.
    """
    u = _unit(16, seed=3)
    q_filter = np.array(compute_qfilters(_planted_queries(1, 2000, 16, u, drift=3.0))[0])
    assert float(np.dot(q_filter, u)) > 0.99, "query-SVD must recover +u with its sign"

    rng = np.random.default_rng(4)
    # Spread (not offset) along u: centering preserves this, so the key-SVD
    # has a dominant axis to recover.
    spread = (rng.standard_normal((2000, 1)) * 4.0) * u
    keys = mx.array((rng.standard_normal((2000, 16)) * 0.5 + spread).astype(np.float32))
    k_filter = np.array(estimate_filter_dir(keys))

    # Axis recovered...
    assert abs(float(np.dot(k_filter, u))) > 0.9
    # ...but the orientation is fixed by a deterministic pivot convention, not
    # by any property of attention — which is exactly why `qfilters_sign` is a
    # real ablation knob on the fallback path and not on the calibrated one.


def test_key_svd_loses_a_pure_drift_direction() -> None:
    """Why the fallback is a fallback: centering destroys drift-only geometry.

    With keys that carry the paper's drift but isotropic spread, the
    mean-centering key estimator has no signal left to lock onto and returns
    an essentially unrelated direction.
    """
    u = _unit(16, seed=12)
    rng = np.random.default_rng(13)
    keys = mx.array((rng.standard_normal((2000, 16)) * 0.5 - 3.0 * u).astype(np.float32))

    k_filter = np.array(estimate_filter_dir(keys))
    assert abs(float(np.dot(k_filter, u))) < 0.3

    # The same drift geometry is recovered perfectly from the query side.
    q_filter = np.array(compute_qfilters(_planted_queries(1, 2000, 16, u, drift=3.0))[0])
    assert float(np.dot(q_filter, u)) > 0.99


def test_does_not_mean_center() -> None:
    """Drift, not centered variance, selects the filter (paper Observation 3.1).

    Queries drift along ``u`` while a competing orthogonal axis ``w`` carries
    all the *centered* variance. The uncentered SVD the paper specifies keys
    on the second moment, so it returns ``u``; a mean-centering estimator
    subtracts the drift away and returns ``w`` instead.

    Note the drift must dominate the second moment (``drift^2 > spread^2``)
    for ``u`` to be the correct answer at all — uncentered SVD maximizes
    ``E[<q,v>^2]``, not the drift in isolation.
    """
    d = 12
    u = np.zeros(d, dtype=np.float32)
    u[0] = 1.0
    w = np.zeros(d, dtype=np.float32)
    w[1] = 1.0

    rng = np.random.default_rng(5)
    noise = rng.standard_normal((1, 4000, d)).astype(np.float32) * 0.1
    spread = (rng.standard_normal((1, 4000, 1)).astype(np.float32) * 2.0) * w
    q = mx.array(noise + spread + 6.0 * u)

    f = np.array(compute_qfilters(q)[0])
    assert float(np.dot(f, u)) > 0.9, "filter followed centered variance, not drift"
    assert abs(float(np.dot(f, w))) < 0.3

    # Pin the contrast: centering the same data yields the variance axis.
    flat = np.array(q[0], dtype=np.float64)
    _, _, vt_centered = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    assert abs(float(np.dot(vt_centered[0].astype(np.float32), w))) > 0.9


def test_subsampling_is_deterministic() -> None:
    q = _planted_queries(2, 5000, 16, _unit(16, seed=6), drift=2.0)
    a = compute_qfilters(q, max_svd_samples=500)
    b = compute_qfilters(q, max_svd_samples=500)
    assert np.array_equal(np.array(a), np.array(b))


def test_filters_are_unit_norm() -> None:
    f = compute_qfilters(_planted_queries(4, 800, 16, _unit(16, seed=7), drift=2.0))
    norms = np.linalg.norm(np.array(f), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ------------------------------------------------------------------
# GQA group averaging (paper §3.2)
# ------------------------------------------------------------------


def test_gqa_averaging_shape_and_norm() -> None:
    f = compute_qfilters(_planted_queries(8, 500, 16, _unit(16, seed=8), drift=2.0))
    grouped = average_gqa_filters(f, n_kv_heads=2)

    assert grouped.shape == (2, 16)
    assert np.allclose(np.linalg.norm(np.array(grouped), axis=1), 1.0, atol=1e-5)


def test_gqa_averaging_is_identity_for_mha() -> None:
    """H_q == n_kv_heads means one head per group — filters pass through."""
    f = compute_qfilters(_planted_queries(3, 500, 16, _unit(16, seed=9), drift=2.0))
    out = average_gqa_filters(f, n_kv_heads=3)
    assert np.allclose(np.array(out), np.array(f), atol=1e-5)


def test_gqa_averaging_merges_aligned_heads() -> None:
    """Heads sharing a drift direction average to that same direction."""
    u = _unit(16, seed=10)
    f = compute_qfilters(_planted_queries(4, 800, 16, u, drift=3.0))
    grouped = average_gqa_filters(f, n_kv_heads=1)
    assert float(np.dot(np.array(grouped[0]), u)) > 0.99


def test_gqa_rejects_non_divisible_head_count() -> None:
    f = compute_qfilters(_planted_queries(6, 300, 8, _unit(8, seed=11), drift=2.0))
    with pytest.raises(ValueError, match="must divide"):
        average_gqa_filters(f, n_kv_heads=4)


# ------------------------------------------------------------------
# Artifact persistence
# ------------------------------------------------------------------


def _calibration(n_layers: int = 3, h: int = 2, d: int = 16) -> QFiltersCalibration:
    filters = [
        compute_qfilters(_planted_queries(h, 300, d, _unit(d, seed=20 + i), drift=2.0))
        for i in range(n_layers)
    ]
    return QFiltersCalibration(
        filters=filters, model_id="test/model", n_samples=300, dataset="synthetic"
    )


def test_artifact_round_trip(tmp_path) -> None:
    cal = _calibration()
    path = save_qfilters(cal, tmp_path / "qf.npz")
    loaded = load_qfilters(path)

    assert loaded.n_layers == cal.n_layers
    assert loaded.model_id == "test/model"
    assert loaded.n_samples == 300
    assert loaded.dataset == "synthetic"
    assert loaded.head_dim() == 16
    for a, b in zip(cal.filters, loaded.filters, strict=True):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6)


def test_load_enforces_model_id(tmp_path) -> None:
    """Filters are model-specific — a mismatch must fail loudly, not degrade."""
    path = save_qfilters(_calibration(), tmp_path / "qf.npz")

    assert load_qfilters(path, expect_model_id="test/model").n_layers == 3
    with pytest.raises(ValueError, match="calibrated on"):
        load_qfilters(path, expect_model_id="other/model")


def test_load_rejects_version_mismatch(tmp_path) -> None:
    import json

    path = tmp_path / "stale.npz"
    meta = json.dumps(
        {
            "version": QFILTERS_ARTIFACT_VERSION + 1,
            "model_id": "m",
            "n_layers": 1,
            "n_samples": 1,
            "dataset": "d",
        }
    )
    np.savez(path, __meta__=np.array(meta), layer_0=np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="artifact version"):
        load_qfilters(path)


def test_load_rejects_missing_layer(tmp_path) -> None:
    import json

    path = tmp_path / "truncated.npz"
    meta = json.dumps(
        {
            "version": QFILTERS_ARTIFACT_VERSION,
            "model_id": "m",
            "n_layers": 2,  # claims 2 layers...
            "n_samples": 1,
            "dataset": "d",
        }
    )
    # ...but only layer_0 is present.
    np.savez(path, __meta__=np.array(meta), layer_0=np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="missing layer_1"):
        load_qfilters(path)


def test_load_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_qfilters(tmp_path / "nope.npz")


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_compute_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[H, N, D\]"):
        compute_qfilters(mx.zeros((4, 8)))


def test_compute_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        compute_qfilters(mx.zeros((2, 1, 8)))


def test_average_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[H_q, D\]"):
        average_gqa_filters(mx.zeros((2, 3, 4)), n_kv_heads=1)


def test_average_rejects_nonpositive_kv_heads() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        average_gqa_filters(mx.zeros((4, 8)), n_kv_heads=0)

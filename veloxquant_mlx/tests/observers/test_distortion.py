"""Tests for DistortionObserver, including the sample-weighting regression for #70."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.observers.base import QuantizationEvent
from veloxquant_mlx.observers.distortion import DistortionObserver


def _event(x_orig: np.ndarray, x_recon: np.ndarray) -> QuantizationEvent:
    return QuantizationEvent(
        stage="test",
        input_shape=x_orig.shape,
        metadata={"x_original": x_orig, "x_reconstructed": x_recon},
    )


class TestReportEmptyState:
    def test_report_with_no_events_is_zero(self) -> None:
        obs = DistortionObserver(b=2, d=4)
        report = obs.report()
        assert report.empirical_mse == 0.0
        assert report.empirical_ip_distortion == 0.0
        assert report.n_samples == 0


class TestSampleWeightedMSE:
    """Regression for #70: report() must be a sample-weighted mean, not a
    mean of per-event batch means."""

    def test_single_batch_matches_plain_mse(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 8))
        x_hat = x + rng.standard_normal((50, 8)) * 0.1
        obs = DistortionObserver(b=2, d=8)
        obs.on_event(_event(x, x_hat))
        report = obs.report()
        expected = float(np.mean(np.sum((x - x_hat) ** 2, axis=-1)))
        assert report.empirical_mse == pytest.approx(expected)
        assert report.n_samples == 50

    def test_varying_batch_sizes_are_sample_weighted_not_event_averaged(self) -> None:
        """The issue's exact reproduction: a large low-error batch and a
        tiny high-error batch must combine via sample-weighted mean, not
        an unweighted mean-of-batch-means."""
        obs = DistortionObserver(b=2, d=4)

        x1 = np.zeros((1000, 4))
        x1_hat = x1 + 0.01
        obs.on_event(_event(x1, x1_hat))

        x2 = np.zeros((2, 4))
        x2_hat = x2 + 100.0
        obs.on_event(_event(x2, x2_hat))

        report = obs.report()

        true_mse = (np.sum((x1 - x1_hat) ** 2) + np.sum((x2 - x2_hat) ** 2)) / (1000 + 2)
        assert report.empirical_mse == pytest.approx(true_mse)
        assert report.n_samples == 1002

        # The old (buggy) unweighted mean-of-means would have been ~20000,
        # ~250x the correct value -- assert we're nowhere near that.
        assert report.empirical_mse < 100.0

    def test_equal_batch_sizes_unaffected(self) -> None:
        """When all batches are the same size, sample-weighted and
        event-averaged means coincide -- this must not regress."""
        rng = np.random.default_rng(1)
        obs = DistortionObserver(b=2, d=4)
        batches = [rng.standard_normal((10, 4)) for _ in range(5)]
        recons = [b + rng.standard_normal((10, 4)) * 0.2 for b in batches]
        for b, r in zip(batches, recons, strict=True):
            obs.on_event(_event(b, r))

        report = obs.report()
        all_orig = np.concatenate(batches)
        all_recon = np.concatenate(recons)
        expected = float(np.mean(np.sum((all_orig - all_recon) ** 2, axis=-1)))
        assert report.empirical_mse == pytest.approx(expected)
        assert report.n_samples == 50

    def test_single_unbatched_vector_counts_as_one_sample(self) -> None:
        """A 1D (d,) event (no batch axis) must not crash and must count
        as exactly one sample."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        x_hat = np.array([1.1, 2.1, 3.1, 4.1])
        obs = DistortionObserver(b=2, d=4)
        obs.on_event(_event(x, x_hat))
        report = obs.report()
        assert report.n_samples == 1
        assert report.empirical_mse == pytest.approx(float(np.sum((x - x_hat) ** 2)))


class TestSampleWeightedIPDistortion:
    def test_ip_distortion_is_sample_weighted(self) -> None:
        query = np.array([1.0, 0.0, 0.0, 0.0])
        obs = DistortionObserver(b=2, d=4, query=query)

        x1 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (100, 1))
        x1_hat = x1 + np.array([0.01, 0, 0, 0])
        obs.on_event(_event(x1, x1_hat))

        x2 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 1))
        x2_hat = x2 + np.array([10.0, 0, 0, 0])
        obs.on_event(_event(x2, x2_hat))

        report = obs.report()
        true_ip = x1 @ query
        approx_ip = x1_hat @ query
        true_ip2 = x2 @ query
        approx_ip2 = x2_hat @ query
        expected = (np.sum((true_ip - approx_ip) ** 2) + np.sum((true_ip2 - approx_ip2) ** 2)) / (
            100 + 1
        )
        assert report.empirical_ip_distortion == pytest.approx(expected)

    def test_no_query_gives_zero_ip_distortion(self) -> None:
        obs = DistortionObserver(b=2, d=4)
        obs.on_event(_event(np.zeros((5, 4)), np.ones((5, 4))))
        report = obs.report()
        assert report.empirical_ip_distortion == 0.0


class TestMissingMetadata:
    def test_event_missing_metadata_is_ignored(self) -> None:
        obs = DistortionObserver(b=2, d=4)
        obs.on_event(QuantizationEvent(stage="s", input_shape=(1, 4), metadata={}))
        report = obs.report()
        assert report.n_samples == 0


class TestReset:
    def test_reset_clears_accumulated_state(self) -> None:
        obs = DistortionObserver(b=2, d=4)
        obs.on_event(_event(np.zeros((10, 4)), np.ones((10, 4))))
        assert obs.report().n_samples == 10
        obs.reset()
        report = obs.report()
        assert report.n_samples == 0
        assert report.empirical_mse == 0.0

"""Tests for OutlierDetector."""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.outlier.detector import OutlierDetector


class TestOutlierDetector:
    def test_get_outlier_channels_are_distinct(self) -> None:
        """Regression test for #66: repeated identical observations must not
        cause SortedChannelIndex to return duplicate channel indices, which
        would silently drop a genuine outlier channel from the selection."""
        det = OutlierDetector(n_outliers=3, n_calib=1)
        k = np.array([1.0, 5.0, 5.0, 2.0, 0.5], dtype=np.float32)
        det.observe(k)
        det.observe(k)
        det.observe(k)

        channels = det.get_outlier_channels()
        assert len(channels) == len(set(channels.tolist())) == 3
        # Magnitudes are [1.0, 5.0, 5.0, 2.0, 0.5] -> top 3 are channels 1, 2, 3.
        assert set(channels.tolist()) == {1, 2, 3}

    def test_calibration_and_reset(self) -> None:
        det = OutlierDetector(n_outliers=2, n_calib=3)
        k = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        assert not det.is_calibrated
        det.observe(k)
        det.observe(k)
        assert not det.is_calibrated
        det.observe(k)
        assert det.is_calibrated
        assert det.n_observed == 3

        det.reset()
        assert not det.is_calibrated
        assert det.n_observed == 0

    def test_repeated_observations_bound_heap_growth(self) -> None:
        """Regression test for #66: the underlying heap must not grow
        unboundedly when the same (or unchanged-mean) vectors are observed
        repeatedly."""
        det = OutlierDetector(n_outliers=4, n_calib=1)
        d = 16
        k = np.arange(d, dtype=np.float32)
        for _ in range(20):
            det.observe(k)
        # Running mean of an identical vector is constant after the first
        # observation, so re-inserts should be no-ops and the heap should
        # stay at d entries, not grow to 20 * d.
        assert len(det._index._heap) == d

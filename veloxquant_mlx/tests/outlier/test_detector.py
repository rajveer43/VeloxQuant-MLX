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

    def test_repeated_observations_bound_state_growth(self) -> None:
        """Regression test for #66, updated for the vectorized top-k rewrite.

        #66 was about an underlying heap growing unboundedly (one entry per
        observation instead of per distinct channel) when the same vector
        was observed repeatedly. The detector no longer keeps a heap at all
        — per-token state is a single running ``_sum_abs`` array — so the
        actual invariant to pin is that its size tracks the channel count
        ``d``, never the number of observations."""
        det = OutlierDetector(n_outliers=4, n_calib=1)
        d = 16
        k = np.arange(d, dtype=np.float32)
        for _ in range(20):
            det.observe(k)
        assert det._sum_abs is not None
        assert det._sum_abs.shape == (d,)
        # Correctness carries over too: an identical vector observed
        # repeatedly must still yield the same top-k as a single observation.
        assert set(det.get_outlier_channels().tolist()) == {12, 13, 14, 15}

    def test_get_outlier_channels_before_any_observation_is_empty(self) -> None:
        det = OutlierDetector(n_outliers=4, n_calib=1)
        channels = det.get_outlier_channels()
        assert channels.shape == (0,)
        assert channels.dtype == np.int32

    def test_get_outlier_channels_clamps_to_available_channels(self) -> None:
        """n_outliers larger than the vector's own channel count must not
        raise (np.argpartition requires kth < len(arr)) — it should return
        every channel instead of over-asking for more than exist."""
        det = OutlierDetector(n_outliers=10, n_calib=1)
        k = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        det.observe(k)
        channels = det.get_outlier_channels()
        assert set(channels.tolist()) == {0, 1, 2}

    def test_get_outlier_channels_returns_sorted_ascending(self) -> None:
        det = OutlierDetector(n_outliers=3, n_calib=1)
        k = np.array([9.0, 1.0, 8.0, 2.0, 7.0], dtype=np.float32)
        det.observe(k)
        channels = det.get_outlier_channels().tolist()
        assert channels == sorted(channels)
        assert channels == [0, 2, 4]

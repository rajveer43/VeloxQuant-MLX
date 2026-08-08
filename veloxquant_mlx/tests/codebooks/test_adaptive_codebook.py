"""Tests for AdaptiveScalarCodebook, including the calibration-truncation
regression for #68."""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.codebooks.adaptive_codebook import AdaptiveScalarCodebook


class TestObserveDoesNotTruncateWithinABatch:
    """Regression for #68: a single observe() call larger than n_calib must
    not silently drop the rows beyond n_calib -- all rows must be counted
    and eligible to contribute to calibration."""

    def test_issue_exact_repro_counts_all_observed_rows(self) -> None:
        rng = np.random.default_rng(0)
        codebook = AdaptiveScalarCodebook(b=2, d=8, n_calib=10, n_hist_bins=8)
        y = rng.standard_normal((50, 8)).astype(np.float32)
        codebook.observe(y)

        assert codebook.is_calibrated
        # Before the fix, this was 10 (only the first `remaining` rows of
        # the single large batch were ever counted/buffered).
        assert codebook._n_observed == 50

    def test_calibration_is_not_biased_to_first_n_calib_rows(self) -> None:
        """Fitting must be able to draw from the whole batch, not just a
        `y_np[:remaining]` prefix -- construct data where the first
        n_calib rows have different statistics than the rest, and confirm
        the fitted centroids reflect the full batch, not just the prefix."""
        d = 4
        n_calib = 20
        n_total = 400

        # First n_calib rows: tightly clustered near 0.
        head = np.full((n_calib, d), 0.0, dtype=np.float32)
        # Remaining rows: a much wider spread, centered far from 0.
        rng = np.random.default_rng(1)
        tail = (rng.standard_normal((n_total - n_calib, d)) * 5.0 + 20.0).astype(np.float32)
        y = np.concatenate([head, tail], axis=0)

        codebook = AdaptiveScalarCodebook(b=2, d=d, n_calib=n_calib, n_hist_bins=32, seed=0)
        codebook.observe(y)

        centroids = codebook.centroids_numpy()
        # If calibration were still biased to the all-zero prefix, fitted
        # centroids would cluster near 0. With the full batch eligible for
        # (random) subsampling, centroids should reflect the tail's scale.
        assert np.abs(centroids).mean() > 5.0, centroids

    def test_multi_call_accumulation_still_works(self) -> None:
        """Calibration spread across several smaller observe() calls (the
        non-truncating case) must be unaffected."""
        rng = np.random.default_rng(2)
        codebook = AdaptiveScalarCodebook(b=2, d=4, n_calib=30, n_hist_bins=16)
        y = rng.standard_normal((30, 4)).astype(np.float32)
        for i in range(3):
            codebook.observe(y[i * 10 : (i + 1) * 10])
        assert codebook.is_calibrated
        assert codebook._n_observed == 30

    def test_observe_is_noop_after_calibration(self) -> None:
        rng = np.random.default_rng(3)
        codebook = AdaptiveScalarCodebook(b=2, d=4, n_calib=10, n_hist_bins=8)
        codebook.observe(rng.standard_normal((10, 4)).astype(np.float32))
        assert codebook.is_calibrated
        n_observed_at_calibration = codebook._n_observed

        codebook.observe(rng.standard_normal((100, 4)).astype(np.float32))
        assert codebook._n_observed == n_observed_at_calibration

    def test_seed_makes_calibration_reproducible(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.standard_normal((100, 8)).astype(np.float32)

        cb1 = AdaptiveScalarCodebook(b=2, d=8, n_calib=20, n_hist_bins=16, seed=7)
        cb1.observe(y)
        cb2 = AdaptiveScalarCodebook(b=2, d=8, n_calib=20, n_hist_bins=16, seed=7)
        cb2.observe(y)

        assert np.allclose(cb1.centroids_numpy(), cb2.centroids_numpy())

    def test_quantize_works_immediately_after_calibration_from_oversized_batch(self) -> None:
        import mlx.core as mx

        rng = np.random.default_rng(5)
        codebook = AdaptiveScalarCodebook(b=3, d=16, n_calib=32, n_hist_bins=32)
        y = rng.standard_normal((500, 16)).astype(np.float32)
        codebook.observe(y)

        idx = codebook.quantize(mx.array(y[:5]))
        dequant = codebook.dequantize(idx)
        assert np.asarray(dequant).shape == (5, 16)

"""Tests for PDF functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from veloxquant_mlx.math.distributions import beta_pdf, gaussian_pdf, polar_angle_pdf


class TestGaussianPDF:
    def test_normalization(self) -> None:
        sigma = 1.0
        x = np.linspace(-10, 10, 100_000)
        integral = np.trapezoid(gaussian_pdf(x, sigma=sigma), x)
        assert abs(integral - 1.0) < 1e-3

    def test_peak_at_zero(self) -> None:
        sigma = 1.0
        x = np.linspace(-5, 5, 1001)  # odd count so 0 is exactly included
        vals = gaussian_pdf(x, sigma=sigma)
        peak_x = x[np.argmax(vals)]
        assert abs(peak_x) < 0.01  # peak must be within 0.01 of zero

    def test_larger_sigma_wider(self) -> None:
        x = np.array([1.0])
        p1 = gaussian_pdf(x, sigma=1.0)[0]
        p2 = gaussian_pdf(x, sigma=2.0)[0]
        assert p1 > p2  # taller at x=1 for sigma=1


class TestBetaPDF:
    def test_support(self) -> None:
        d = 64
        x_outside = np.array([-1.5, 1.5])
        vals = beta_pdf(x_outside, d=d)
        np.testing.assert_array_equal(vals, 0.0)

    def test_normalization(self) -> None:
        for d in [10, 64, 128]:
            x = np.linspace(-0.999, 0.999, 100_000)
            integral = np.trapezoid(beta_pdf(x, d=d), x)
            assert abs(integral - 1.0) < 1e-2, f"d={d}: normalization failed"

    def test_symmetric(self) -> None:
        d = 64
        x = np.linspace(-0.9, 0.9, 1000)
        vals = beta_pdf(x, d=d)
        vals_neg = beta_pdf(-x, d=d)
        np.testing.assert_allclose(vals, vals_neg[::-1], rtol=1e-5)


class TestPolarAnglePDF:
    def test_level1_uniform(self) -> None:
        x = np.linspace(0, 2 * math.pi - 0.01, 1000)
        vals = polar_angle_pdf(x, level=1)
        np.testing.assert_allclose(vals, 1.0 / (2 * math.pi), rtol=1e-5)

    def test_level1_normalization(self) -> None:
        x = np.linspace(0, 2 * math.pi - 1e-6, 10_000)
        integral = np.trapezoid(polar_angle_pdf(x, level=1), x)
        assert abs(integral - 1.0) < 0.01

    @pytest.mark.parametrize("level", [2, 3, 4])
    def test_higher_level_normalization(self, level: int) -> None:
        x = np.linspace(0, math.pi / 2, 10_000)
        integral = np.trapezoid(polar_angle_pdf(x, level=level), x)
        assert abs(integral - 1.0) < 0.05, f"level={level}: normalization {integral:.4f}"

    def test_level2_peaks_at_pi4(self) -> None:
        x = np.linspace(0, math.pi / 2, 1000)
        vals = polar_angle_pdf(x, level=2)
        peak_idx = np.argmax(vals)
        # f(ψ) ∝ sin(2ψ)^0 = 1 at level 2 (exponent = 0), but level 3 peaks at pi/4
        # Level 2: exponent = 2^(2-1)-1 = 1, f ∝ sin(2ψ)^1, peaks at ψ=pi/4
        assert abs(x[peak_idx] - math.pi / 4) < 0.1

    def test_invalid_level(self) -> None:
        with pytest.raises(ValueError):
            polar_angle_pdf(np.array([0.5]), level=0)

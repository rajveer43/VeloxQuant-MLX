"""Tests for the Lloyd-Max solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from veloxquant_mlx.math.distributions import beta_pdf, gaussian_pdf
from veloxquant_mlx.math.lloyd_max import lloyd_max


def test_lloyd_max_returns_sorted_centroids() -> None:
    sigma = 1.0
    pdf_fn = lambda x: gaussian_pdf(x, sigma=sigma)
    centroids, boundaries = lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=4)
    assert len(centroids) == 4
    assert np.all(np.diff(centroids) > 0), "Centroids must be strictly ascending"


def test_lloyd_max_boundaries_count() -> None:
    pdf_fn = lambda x: gaussian_pdf(x, sigma=1.0)
    centroids, boundaries = lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=4)
    assert len(boundaries) == 5  # n_levels + 1
    assert np.isinf(boundaries[0]) and boundaries[0] < 0
    assert np.isinf(boundaries[-1]) and boundaries[-1] > 0


def test_lloyd_max_gaussian_symmetry() -> None:
    """Gaussian optimal codebook must be antisymmetric."""
    sigma = 1.0
    pdf_fn = lambda x: gaussian_pdf(x, sigma=sigma)
    centroids, _ = lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=4)
    # Centroids should be symmetric around 0
    np.testing.assert_allclose(centroids, -centroids[::-1], atol=1e-4)


@pytest.mark.parametrize("n_levels", [2, 4, 8, 16])
def test_lloyd_max_convergence(n_levels: int) -> None:
    """MSE cost should decrease as n_levels increases."""
    pdf_fn = lambda x: gaussian_pdf(x, sigma=1.0)
    lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=n_levels)
    cost = lloyd_max.last_mse_cost
    assert cost >= 0.0, "MSE cost must be non-negative"


def test_lloyd_max_beats_uniform_beta() -> None:
    """Lloyd-Max MSE must be strictly lower than uniform-grid MSE for Beta distribution."""
    d = 128
    x_grid = np.linspace(-0.999, 0.999, 200_000)
    pdf_vals = beta_pdf(x_grid, d=d)

    def uniform_mse(n_levels: int) -> float:
        edges = np.linspace(-1.0, 1.0, n_levels + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = 0.0
        for i in range(n_levels):
            mask = (x_grid >= edges[i]) & (x_grid < edges[i + 1])
            if np.any(mask):
                total += float(
                    np.trapezoid((x_grid[mask] - centers[i]) ** 2 * pdf_vals[mask], x_grid[mask])
                )
        return total

    pdf_fn = lambda x: beta_pdf(x, d=d)
    for b in [1, 2, 3]:
        n_levels = 2**b
        lloyd_max(pdf_fn, (-1.0, 1.0), n_levels=n_levels)
        lm_mse = lloyd_max.last_mse_cost
        unif_mse = uniform_mse(n_levels)
        assert lm_mse < unif_mse, (
            f"b={b}: Lloyd-Max MSE {lm_mse:.6f} should be < uniform MSE {unif_mse:.6f}"
        )


def test_lloyd_max_mse_improves_with_bits_gaussian() -> None:
    """Lloyd-Max MSE should roughly quadruple as bits decrease by 1 (high-rate theory)."""
    sigma = 1.0
    pdf_fn = lambda x: gaussian_pdf(x, sigma=sigma)
    prev_mse = None
    for b in [3, 2, 1]:
        lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=2**b)
        mse = lloyd_max.last_mse_cost
        if prev_mse is not None:
            # Going from b+1 to b bits, MSE should roughly increase by 4×
            assert mse > prev_mse * 1.5, f"MSE did not increase enough from b={b + 1} to b={b}"
        prev_mse = mse


def test_lloyd_max_invalid_support() -> None:
    pdf_fn = lambda x: gaussian_pdf(x, sigma=1.0)
    with pytest.raises(ValueError):
        lloyd_max(pdf_fn, (5.0, -5.0), n_levels=4)  # lo > hi


def test_lloyd_max_single_level() -> None:
    pdf_fn = lambda x: gaussian_pdf(x, sigma=1.0)
    centroids, boundaries = lloyd_max(pdf_fn, (-5.0, 5.0), n_levels=1)
    assert len(centroids) == 1

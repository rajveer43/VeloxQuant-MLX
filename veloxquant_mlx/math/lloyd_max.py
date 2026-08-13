from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from veloxquant_mlx.core.constants import LLOYD_MAX_N_ITER, LLOYD_MAX_N_QUAD, LLOYD_MAX_TOL


def lloyd_max(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    support: Tuple[float, float],
    n_levels: int,
    n_iter: int = LLOYD_MAX_N_ITER,
    tol: float = LLOYD_MAX_TOL,
    n_quad_points: int = LLOYD_MAX_N_QUAD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Lloyd-Max 1-D scalar quantisation problem.

    Iterates the Lloyd-Max optimality conditions until convergence:

    1. Initialise centroids uniformly over support.
    2. Loop:
       a. Boundaries = midpoints between adjacent centroids, extended to ±∞.
       b. Update centroids via numerical integration:
          c_i = ∫_{b_{i-1}}^{b_i} x · f(x) dx / ∫_{b_{i-1}}^{b_i} f(x) dx
       c. Stop when max|c_new - c_old| < tol.
    3. Compute and cache the final MSE cost.

    The resulting ``last_mse_cost`` attribute stores::

        C = Σ_i ∫_{b_{i-1}}^{b_i} (x - c_i)² · f(x) dx

    Args:
        pdf_fn: Vectorised PDF callable: pdf_fn(x) -> non-negative values.
        support: (lo, hi) defining the distribution's effective support.
        n_levels: Number of quantisation levels k = 2^b.
        n_iter: Maximum number of Lloyd-Max iterations.
        tol: Convergence tolerance on centroid shift.
        n_quad_points: Number of quadrature points for numerical integration.

    Returns:
        Tuple (centroids, boundaries) both sorted ascending.
        - centroids: shape (n_levels,).
        - boundaries: shape (n_levels + 1,) including ±∞ at the edges.

    Raises:
        ValueError: If n_levels < 1 or support is invalid.
    """
    lo, hi = float(support[0]), float(support[1])
    if lo >= hi:
        raise ValueError(f"lloyd_max: support must have lo < hi, got ({lo}, {hi})")
    if n_levels < 1:
        raise ValueError(f"lloyd_max: n_levels must be >= 1, got {n_levels}")

    # Dense quadrature grid
    x_grid = np.linspace(lo, hi, n_quad_points)
    p_grid = pdf_fn(x_grid)
    dx = x_grid[1] - x_grid[0]

    # Initialise centroids uniformly
    centroids = np.linspace(lo, hi, n_levels)

    for _ in range(n_iter):
        # a. Compute Voronoi boundaries as midpoints
        boundaries = np.concatenate(
            [
                [-np.inf],
                (centroids[:-1] + centroids[1:]) / 2.0,
                [np.inf],
            ]
        )

        # b. Update centroids via trapezoid integration on grid
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            b_lo = boundaries[i]
            b_hi = boundaries[i + 1]
            mask = (x_grid >= b_lo) & (x_grid <= b_hi)
            if not np.any(mask):
                new_centroids[i] = centroids[i]
                continue
            xm = x_grid[mask]
            pm = p_grid[mask]
            mass = np.trapezoid(pm, xm)
            if mass < 1e-12:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = np.trapezoid(xm * pm, xm) / mass

        # c. Check convergence
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    # Final boundaries
    boundaries = np.concatenate(
        [
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2.0,
            [np.inf],
        ]
    )

    # Compute final MSE cost
    mse_cost = 0.0
    for i in range(n_levels):
        b_lo = boundaries[i]
        b_hi = boundaries[i + 1]
        mask = (x_grid >= b_lo) & (x_grid <= b_hi)
        if not np.any(mask):
            continue
        xm = x_grid[mask]
        pm = p_grid[mask]
        mse_cost += np.trapezoid((xm - centroids[i]) ** 2 * pm, xm)

    # Store as module-level last_mse_cost for external inspection
    lloyd_max.last_mse_cost = float(mse_cost)  # type: ignore[attr-defined]

    return centroids.astype(np.float64), boundaries.astype(np.float64)


# Initialise attribute so it always exists
lloyd_max.last_mse_cost = 0.0  # type: ignore[attr-defined]

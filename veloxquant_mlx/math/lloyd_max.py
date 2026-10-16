"""Lloyd-Max algorithm for optimal (MSE-minimizing) non-uniform scalar quantization.

Given a source PDF (see :mod:`veloxquant_mlx.math.distributions`) and a
target number of levels, iteratively solves for the centroids and Voronoi
boundaries that minimize expected squared quantization error — the
non-uniform-codebook counterpart to uniform quantization used by
codebook-based methods in :mod:`veloxquant_mlx.codebooks`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veloxquant_mlx.core.constants import LLOYD_MAX_N_ITER, LLOYD_MAX_N_QUAD, LLOYD_MAX_TOL


def lloyd_max(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    support: tuple[float, float],
    n_levels: int,
    n_iter: int = LLOYD_MAX_N_ITER,
    tol: float = LLOYD_MAX_TOL,
    n_quad_points: int = LLOYD_MAX_N_QUAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Lloyd-Max 1-D scalar quantisation problem.

    Iterates the Lloyd-Max optimality conditions until convergence:

    1. Initialise centroids at equal-mass quantiles of the PDF over the support.
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

    # Initialise centroids at the equal-probability-mass quantiles. A uniform
    # grid over a support much wider than the PDF (beta_pdf on (-1, 1) at
    # d=128) leaves the outermost cells massless; they then never move and the
    # codebook keeps dead centroids at the support edges.
    seg = (p_grid[1:] + p_grid[:-1]) / 2.0 * np.diff(x_grid)
    cdf = np.concatenate([[0.0], np.cumsum(seg)])
    if cdf[-1] > 0.0:
        centroids = np.interp((np.arange(n_levels) + 0.5) / n_levels * cdf[-1], cdf, x_grid)
    else:
        centroids = np.linspace(lo, hi, n_levels)

    # Cumulative trapezoid integrals of f, x f and x^2 f on the grid. A cell's
    # integral is the difference of these evaluated (by linear interpolation)
    # at its Voronoi boundaries, so the partial grid interval at each cell edge
    # is counted instead of dropped -- masking whole grid points under-counts
    # every cell by up to one interval per edge, which is a large relative error
    # once cells are only a few dozen grid points wide (b = 8).
    dx = np.diff(x_grid)
    cum_m = np.concatenate([[0.0], np.cumsum((p_grid[1:] + p_grid[:-1]) / 2.0 * dx)])
    xp = x_grid * p_grid
    cum_f = np.concatenate([[0.0], np.cumsum((xp[1:] + xp[:-1]) / 2.0 * dx)])
    xxp = x_grid * xp
    cum_s = np.concatenate([[0.0], np.cumsum((xxp[1:] + xxp[:-1]) / 2.0 * dx)])

    def _cell_integrals(cents: np.ndarray):
        inner = (cents[:-1] + cents[1:]) / 2.0
        edges = np.concatenate([[lo], np.clip(inner, lo, hi), [hi]])
        m = np.diff(np.interp(edges, x_grid, cum_m))
        f = np.diff(np.interp(edges, x_grid, cum_f))
        sq = np.diff(np.interp(edges, x_grid, cum_s))
        return m, f, sq

    for _ in range(n_iter):
        mass, first, _ = _cell_integrals(centroids)
        safe = mass >= 1e-12
        new_centroids = np.where(safe, first / np.where(safe, mass, 1.0), centroids)

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

    # Final MSE cost: sum_i (E[x^2] - 2 c_i E[x] + c_i^2 E[1]) over each cell
    mass, first, second = _cell_integrals(centroids)
    mse_cost = float(np.sum(second - 2.0 * centroids * first + centroids**2 * mass))

    # Store as module-level last_mse_cost for external inspection
    lloyd_max.last_mse_cost = float(mse_cost)  # type: ignore[attr-defined]

    return centroids.astype(np.float64), boundaries.astype(np.float64)


# Initialise attribute so it always exists
lloyd_max.last_mse_cost = 0.0  # type: ignore[attr-defined]

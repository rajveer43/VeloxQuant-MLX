from __future__ import annotations

import math

import numpy as np

from veloxquant_mlx.core.abstractions import CodebookStrategy
from veloxquant_mlx.core.registry import CodebookRegistry
from veloxquant_mlx.math.distributions import beta_pdf, gaussian_pdf, polar_angle_pdf
from veloxquant_mlx.math.lloyd_max import lloyd_max


@CodebookRegistry.register("lloyd_max_gaussian")
class LloydMaxGaussianStrategy(CodebookStrategy):
    """Optimal codebook for N(0, 1/d) — high-d TurboQuant coordinates.

    Uses Lloyd-Max algorithm with a Gaussian PDF scaled to sigma = 1/sqrt(d).

    Args:
        support_sigma_factor: Support range as multiples of sigma (default 5).
    """

    def __init__(self, support_sigma_factor: float = 6.0) -> None:
        self._support_sigma_factor = support_sigma_factor

    def compute_centroids(self, b: int, d: int) -> np.ndarray:
        """Compute 2^b centroids for N(0, 1/d) distribution.

        Args:
            b: Bit-width.
            d: Vector dimension.

        Returns:
            Sorted centroid array of shape (2^b,), float64.
        """
        sigma = 1.0 / math.sqrt(d)
        support = (-self._support_sigma_factor * sigma, self._support_sigma_factor * sigma)
        pdf_fn = lambda x: gaussian_pdf(x, sigma=sigma)
        centroids, _ = lloyd_max(pdf_fn, support, n_levels=2**b)
        return centroids.astype(np.float64)

    def __repr__(self) -> str:
        return f"LloydMaxGaussianStrategy(support_sigma_factor={self._support_sigma_factor})"


@CodebookRegistry.register("lloyd_max_beta")
class LloydMaxBetaStrategy(CodebookStrategy):
    """Optimal codebook for the exact Beta-distributed coordinates of TurboQuant.

    Uses the analytically correct Beta distribution for the given dimension d.
    Preferred for d < 64 where the Gaussian approximation is inaccurate.

    Args:
        None.
    """

    def compute_centroids(self, b: int, d: int) -> np.ndarray:
        """Compute 2^b centroids for Beta(d/2, d/2) coordinate distribution.

        Args:
            b: Bit-width.
            d: Vector dimension.

        Returns:
            Sorted centroid array of shape (2^b,), float64.
        """
        support = (-1.0 + 1e-6, 1.0 - 1e-6)
        pdf_fn = lambda x: beta_pdf(x, d)
        centroids, _ = lloyd_max(pdf_fn, support, n_levels=2**b)
        return centroids.astype(np.float64)

    def __repr__(self) -> str:
        return "LloydMaxBetaStrategy()"


@CodebookRegistry.register("polar_angle_sampling")
class PolarAngleSamplingStrategy(CodebookStrategy):
    """Optimal codebook for PolarQuant angle distribution at each level.

    Uses Lloyd-Max on the analytically derived polar angle PDF. The level
    must be set via the ``level`` attribute before calling compute_centroids().

    Args:
        level: Polar recursion level (1-indexed, must be >= 1).
    """

    def __init__(self, level: int = 1) -> None:
        self.level = level

    def compute_centroids(self, b: int, d: int) -> np.ndarray:
        """Compute 2^b centroids for polar angle PDF at the configured level.

        Args:
            b: Bit-width.
            d: Vector dimension (unused; distribution is dimension-independent).

        Returns:
            Sorted centroid array of shape (2^b,), float64.
        """
        if self.level == 1:
            support = (0.0, 2 * math.pi - 1e-6)
        else:
            support = (1e-6, math.pi / 2 - 1e-6)
        pdf_fn = lambda x: polar_angle_pdf(x, self.level)
        centroids, _ = lloyd_max(pdf_fn, support, n_levels=2**b)
        return centroids.astype(np.float64)

    def __repr__(self) -> str:
        return f"PolarAngleSamplingStrategy(level={self.level})"


@CodebookRegistry.register("uniform")
class UniformStrategy(CodebookStrategy):
    """Uniform (midpoint) codebook — baseline for comparison.

    Divides the support range into 2^b equal-width cells and places
    centroids at cell midpoints.

    Args:
        lo: Lower bound of the uniform support (default -1.0).
        hi: Upper bound of the uniform support (default 1.0).
    """

    def __init__(self, lo: float = -1.0, hi: float = 1.0) -> None:
        self._lo = lo
        self._hi = hi

    def compute_centroids(self, b: int, d: int) -> np.ndarray:
        """Compute 2^b uniformly spaced centroids.

        Args:
            b: Bit-width.
            d: Unused.

        Returns:
            Sorted centroid array of shape (2^b,), float64.
        """
        k = 2**b
        step = (self._hi - self._lo) / k
        return np.array([self._lo + (i + 0.5) * step for i in range(k)], dtype=np.float64)

    def __repr__(self) -> str:
        return f"UniformStrategy(lo={self._lo}, hi={self._hi})"

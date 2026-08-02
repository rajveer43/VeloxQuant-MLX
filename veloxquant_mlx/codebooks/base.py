from __future__ import annotations

import math
from typing import Literal

import numpy as np

from veloxquant_mlx.core.abstractions import Codebook, CodebookStrategy
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.core.registry import CodebookRegistry


class CodebookFactory:
    """Factory for creating ScalarCodebook instances via registered strategies.

    All instantiation of codebooks should go through this factory.

    Example::

        cb = CodebookFactory.create("gaussian", b=2, d=128)
    """

    @staticmethod
    def create(
        distribution: Literal["gaussian", "beta", "polar_level", "uniform"],
        b: int,
        d: int,
        polar_level: int = 1,
    ) -> Codebook:
        """Create a ScalarCodebook for the given distribution and bit-width.

        Args:
            distribution: Centroid distribution type.
                - ``"gaussian"``: N(0, 1/d) — high-d TurboQuant.
                - ``"beta"``: exact Beta distribution — low-d TurboQuant.
                - ``"polar_level"``: polar angle distribution (requires polar_level).
                - ``"uniform"``: uniformly spaced centroids.
            b: Bit-width (1 <= b <= 8).
            d: Vector dimension.
            polar_level: Polar recursion level (only used when distribution="polar_level").

        Returns:
            Configured ScalarCodebook instance.

        Raises:
            QuantizerConfigError: If parameters are invalid.
        """
        from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook

        if b < 1 or b > 8:
            raise QuantizerConfigError(f"CodebookFactory: b must be in [1, 8], got {b}")
        if d < 1:
            raise QuantizerConfigError(f"CodebookFactory: d must be >= 1, got {d}")

        # Map distribution to registry key + constructor args
        if distribution == "gaussian":
            strategy_key = "lloyd_max_gaussian"
            strategy = CodebookRegistry.get(strategy_key)()
        elif distribution == "beta":
            strategy_key = "lloyd_max_beta"
            strategy = CodebookRegistry.get(strategy_key)()
        elif distribution == "polar_level":
            from veloxquant_mlx.codebooks.strategies import PolarAngleSamplingStrategy

            strategy = PolarAngleSamplingStrategy(level=polar_level)
        elif distribution == "uniform":
            strategy_key = "uniform"
            strategy = CodebookRegistry.get(strategy_key)()
        else:
            raise QuantizerConfigError(
                f"CodebookFactory: unknown distribution '{distribution}'. "
                f"Choices: gaussian, beta, polar_level, uniform."
            )

        centroids = strategy.compute_centroids(b=b, d=d)
        return ScalarCodebook(centroids.astype(np.float32))

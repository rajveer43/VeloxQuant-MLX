"""Scalar codebooks and their centroid-fitting strategies for quantizers.

Groups :class:`ScalarCodebook` (nearest-centroid quantize/dequantize),
:class:`CodebookFactory` (builds a codebook for a given distribution and
bit-width), and the ``CodebookStrategy`` implementations that fit centroids
via Lloyd-Max on a Gaussian, Beta, or polar-angle PDF, or space them
uniformly — the pieces every scalar-quantization method in ``quantizers/``
and ``cache/`` composes to turn continuous coordinates into ``b``-bit codes.
"""

from __future__ import annotations

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook
from veloxquant_mlx.codebooks.strategies import (
    LloydMaxBetaStrategy,
    LloydMaxGaussianStrategy,
    PolarAngleSamplingStrategy,
    UniformStrategy,
)

__all__ = [
    "CodebookFactory",
    "ScalarCodebook",
    "LloydMaxGaussianStrategy",
    "LloydMaxBetaStrategy",
    "PolarAngleSamplingStrategy",
    "UniformStrategy",
]

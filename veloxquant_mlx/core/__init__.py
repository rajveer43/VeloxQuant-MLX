"""Framework-level primitives shared across the whole quantization pipeline.

Re-exports the abstract base classes every quantizer/codebook/cache/store
implements (:class:`Quantizer`, :class:`Codebook`, :class:`KVCache`,
:class:`ArtifactStore`, etc. from :mod:`~veloxquant_mlx.core.abstractions`),
the data-carrying types passed between pipeline stages
(:class:`EncodedVector`, :class:`QuantizationContext`, :class:`TransformResult`),
the package's exception hierarchy, its registries
(:class:`QuantizerRegistry`, :class:`CodebookRegistry`,
:class:`PreconditionerRegistry`), and shared numeric constants — the
common vocabulary the rest of ``veloxquant_mlx`` is built on.
"""

from __future__ import annotations

from veloxquant_mlx.core.abstractions import (
    ArtifactStore,
    Codebook,
    CodebookStrategy,
    InnerProductStrategy,
    KVCache,
    Preconditioner,
    QuantizationHandler,
    QuantizationObserver,
    Quantizer,
    Transform,
)
from veloxquant_mlx.core.constants import (
    DEFAULT_JL_DIM,
    DEFAULT_N_CALIB_TOKENS,
    DEFAULT_N_OUTLIER_CHANNELS,
    DEFAULT_POLAR_LEVELS,
    DEFAULT_SEED,
    LLOYD_MAX_N_ITER,
    LLOYD_MAX_N_QUAD,
    LLOYD_MAX_TOL,
    LOWER_MSE_FACTOR,
    SQRT_PI_OVER_2,
    UPPER_MSE_FACTOR,
    VORONOI_LINEAR_THRESHOLD,
)
from veloxquant_mlx.core.context import EncodedVector, QuantizationContext, TransformResult
from veloxquant_mlx.core.exceptions import (
    ArtifactNotFoundError,
    CodebookDimensionMismatch,
    CyclicPipelineError,
    QuantizerConfigError,
)
from veloxquant_mlx.core.registry import CodebookRegistry, PreconditionerRegistry, QuantizerRegistry

__all__ = [
    "ArtifactStore",
    "Codebook",
    "CodebookStrategy",
    "InnerProductStrategy",
    "KVCache",
    "Preconditioner",
    "QuantizationHandler",
    "QuantizationObserver",
    "Quantizer",
    "Transform",
    "EncodedVector",
    "QuantizationContext",
    "TransformResult",
    "ArtifactNotFoundError",
    "CodebookDimensionMismatch",
    "CyclicPipelineError",
    "QuantizerConfigError",
    "QuantizerRegistry",
    "CodebookRegistry",
    "PreconditionerRegistry",
    "DEFAULT_JL_DIM",
    "DEFAULT_N_CALIB_TOKENS",
    "DEFAULT_N_OUTLIER_CHANNELS",
    "DEFAULT_POLAR_LEVELS",
    "DEFAULT_SEED",
    "LLOYD_MAX_N_ITER",
    "LLOYD_MAX_N_QUAD",
    "LLOYD_MAX_TOL",
    "LOWER_MSE_FACTOR",
    "SQRT_PI_OVER_2",
    "UPPER_MSE_FACTOR",
    "VORONOI_LINEAR_THRESHOLD",
]

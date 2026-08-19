"""veloxquant_mlx — KV cache quantization for Apple Silicon MLX.

Implements TurboQuant, TurboQuantRVQ, PolarQuant, and QJL plus the
RateQuant per-layer bit allocator for production LLM inference.
"""

from __future__ import annotations

from veloxquant_mlx.allocators import (
    allocate_bits_ratequant,
    apply_dual_transform_keys,
    apply_dual_transform_queries,
    calibrate_layer_sensitivities,
    calibrate_smooth_factors,
    fit_distortion_curve,
    train_codebook,
    walsh_hadamard_matrix,
)
from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
from veloxquant_mlx.core.abstractions import (
    ArtifactStore,
    KVCache,
    Quantizer,
    QuantizationObserver,
)
from veloxquant_mlx.core.context import EncodedVector, QuantizationContext, TransformResult
from veloxquant_mlx.core.exceptions import (
    ArtifactNotFoundError,
    CodebookDimensionMismatch,
    CyclicPipelineError,
    QuantizerConfigError,
)
from veloxquant_mlx.observers import KeyNormObserver, KeyNormReport
from veloxquant_mlx.quantizers.base import QuantizerFactory

__all__ = [
    # Configuration & builders
    "KVCacheBuilder",
    "KVCacheConfig",
    "KVCacheFactory",
    # Abstractions
    "ArtifactStore",
    "KVCache",
    "Quantizer",
    "QuantizationObserver",
    # Data types
    "EncodedVector",
    "QuantizationContext",
    "TransformResult",
    # Exceptions
    "ArtifactNotFoundError",
    "CodebookDimensionMismatch",
    "CyclicPipelineError",
    "QuantizerConfigError",
    # Quantizer registry
    "QuantizerFactory",
    # RateQuant allocators (per-layer mixed-precision)
    "allocate_bits_ratequant",
    "calibrate_layer_sensitivities",
    "fit_distortion_curve",
    # VecInfer
    "VecInferKVCache",
    "apply_dual_transform_keys",
    "apply_dual_transform_queries",
    "calibrate_smooth_factors",
    "train_codebook",
    "walsh_hadamard_matrix",
    # Observers
    "KeyNormObserver",
    "KeyNormReport",
]

__version__ = "0.51.1"

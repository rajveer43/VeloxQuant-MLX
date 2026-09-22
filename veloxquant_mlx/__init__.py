"""veloxquant_mlx — KV cache quantization for Apple Silicon MLX.

Implements TurboQuant, TurboQuantRVQ, PolarQuant, and QJL plus the
RateQuant per-layer bit allocator for production LLM inference.

Preview APIs: :class:`~veloxquant_mlx.planning.AutoOptimizer` (automatic
hardware-aware strategy selection, RFC #469) and
:class:`~veloxquant_mlx.routing.CacheRoutePlanner` (rate-aware session
routing) are exported here for convenience but are subject to change
without a major version bump — see their module docstrings
(:mod:`veloxquant_mlx.planning`, :mod:`veloxquant_mlx.routing.cacheroute`)
for what "preview" means for each.
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
from veloxquant_mlx.config import (
    AutoConfigResult,
    HardwareInfo,
    WorkloadSpec,
    detect_hardware_info,
    select_kv_cache_config,
)
from veloxquant_mlx.core.abstractions import (
    ArtifactStore,
    KVCache,
    QuantizationObserver,
    Quantizer,
)
from veloxquant_mlx.core.context import EncodedVector, QuantizationContext, TransformResult
from veloxquant_mlx.core.exceptions import (
    ArtifactNotFoundError,
    BlockPoolExhaustedError,
    CodebookDimensionMismatch,
    CyclicPipelineError,
    QuantizerConfigError,
)
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, PooledKVCache
from veloxquant_mlx.observers import KeyNormObserver, KeyNormReport
from veloxquant_mlx.planning import (
    AutoOptimizer,
    AutoOptimizerOptions,
    HardwareProfile,
    MemoryEstimate,
    ModelProfile,
    RecommendationResult,
    WorkloadObjective,
    WorkloadProfile,
)
from veloxquant_mlx.profiling import (
    KVCacheProfiler,
    LayerProfile,
    MLXCacheProfiler,
    ProfileReport,
    format_profile_table,
    profile_layers,
)
from veloxquant_mlx.quantizers.base import QuantizerFactory
from veloxquant_mlx.routing import CacheRoutePlanner, RateEstimator, RoutingTable, SessionRate

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
    "BlockPoolExhaustedError",
    "CodebookDimensionMismatch",
    "CyclicPipelineError",
    "QuantizerConfigError",
    # KV-cache block pool allocator (issue #249)
    "BlockPoolAllocator",
    "PoolConfig",
    "PooledKVCache",
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
    # Profiling (issue #252)
    "KVCacheProfiler",
    "MLXCacheProfiler",
    "LayerProfile",
    "ProfileReport",
    "format_profile_table",
    "profile_layers",
    # Hardware-aware automatic configuration (issue #253)
    "AutoConfigResult",
    "HardwareInfo",
    "WorkloadSpec",
    "detect_hardware_info",
    "select_kv_cache_config",
    # Automatic strategy selection (RFC method="auto") -- PREVIEW API, see
    # veloxquant_mlx.planning's module docstring.
    "AutoOptimizer",
    "AutoOptimizerOptions",
    "HardwareProfile",
    "ModelProfile",
    "WorkloadProfile",
    "WorkloadObjective",
    "RecommendationResult",
    "MemoryEstimate",
    # CacheRoute: rate-aware session admission and shard placement (issue #278)
    # -- PREVIEW API, see veloxquant_mlx.routing.cacheroute's module docstring.
    "CacheRoutePlanner",
    "RateEstimator",
    "RoutingTable",
    "SessionRate",
]

__version__ = "0.91.2"

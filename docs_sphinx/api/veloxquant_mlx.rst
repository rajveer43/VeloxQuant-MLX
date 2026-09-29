veloxquant\_mlx package
=======================

.. automodule:: veloxquant_mlx
   :noindex:

Public API
----------

The names below are re-exported at the top level for convenience; each is
documented once, at its defining module (linked from the subpackage pages
below), to avoid duplicate entries.

.. currentmodule:: veloxquant_mlx

.. autosummary::

   KVCacheBuilder
   KVCacheConfig
   KVCacheFactory
   ArtifactStore
   KVCache
   Quantizer
   QuantizationObserver
   EncodedVector
   QuantizationContext
   TransformResult
   ArtifactNotFoundError
   BlockPoolExhaustedError
   CodebookDimensionMismatch
   CyclicPipelineError
   QuantizerConfigError
   BlockPoolAllocator
   PoolConfig
   PooledKVCache
   QuantizerFactory
   allocate_bits_ratequant
   calibrate_layer_sensitivities
   fit_distortion_curve
   VecInferKVCache
   apply_dual_transform_keys
   apply_dual_transform_queries
   calibrate_smooth_factors
   train_codebook
   walsh_hadamard_matrix
   KeyNormObserver
   KeyNormReport
   KVCacheProfiler
   MLXCacheProfiler
   LayerProfile
   ProfileReport
   format_profile_table
   profile_layers
   AutoConfigResult
   HardwareInfo
   WorkloadSpec
   detect_hardware_info
   select_kv_cache_config
   AutoOptimizer
   AutoOptimizerOptions
   HardwareProfile
   ModelProfile
   WorkloadProfile
   WorkloadObjective
   RecommendationResult
   MemoryEstimate
   CacheRoutePlanner
   RateEstimator
   RoutingTable
   SessionRate

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   veloxquant_mlx.allocators
   veloxquant_mlx.artifacts
   veloxquant_mlx.benchmarks
   veloxquant_mlx.cache
   veloxquant_mlx.cli
   veloxquant_mlx.codebooks
   veloxquant_mlx.config
   veloxquant_mlx.core
   veloxquant_mlx.dsa
   veloxquant_mlx.handlers
   veloxquant_mlx.integration
   veloxquant_mlx.math
   veloxquant_mlx.memory
   veloxquant_mlx.metal
   veloxquant_mlx.observers
   veloxquant_mlx.outlier
   veloxquant_mlx.planning
   veloxquant_mlx.preconditioners
   veloxquant_mlx.profiling
   veloxquant_mlx.quantizers
   veloxquant_mlx.routing
   veloxquant_mlx.spectral
   veloxquant_mlx.tools
   veloxquant_mlx.transfer
   veloxquant_mlx.transforms
   veloxquant_mlx.ui
   veloxquant_mlx.weight

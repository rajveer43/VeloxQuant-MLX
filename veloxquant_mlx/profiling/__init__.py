"""Kernel-level timing and memory profiling for VeloxQuant KV caches.

See :class:`KVCacheProfiler` for the primary entry point.
"""

from __future__ import annotations

from veloxquant_mlx.profiling.hardware_profiler import (
    HardwareProfile,
    chip_generation,
    detect_hardware_profile,
    measure_bandwidth_gbps,
)
from veloxquant_mlx.profiling.kv_profiler import (
    KVCacheProfiler,
    LayerProfile,
    MLXCacheProfiler,
    ProfileReport,
    format_profile_table,
    profile_layers,
)
from veloxquant_mlx.profiling.model_profiler import (
    ModelProfile,
    architecture_alias,
    attention_type_from_heads,
    profile_model_from_config,
    profile_model_from_model,
)

__all__ = [
    "KVCacheProfiler",
    "LayerProfile",
    "MLXCacheProfiler",
    "ProfileReport",
    "format_profile_table",
    "profile_layers",
    "HardwareProfile",
    "detect_hardware_profile",
    "chip_generation",
    "measure_bandwidth_gbps",
    "ModelProfile",
    "profile_model_from_config",
    "profile_model_from_model",
    "architecture_alias",
    "attention_type_from_heads",
]

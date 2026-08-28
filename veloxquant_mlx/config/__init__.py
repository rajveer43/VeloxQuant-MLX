"""Hardware-aware automatic KV-cache configuration (issue #253)."""

from __future__ import annotations

from veloxquant_mlx.config.auto_config import (
    AutoConfigResult,
    HardwareInfo,
    WorkloadSpec,
    detect_hardware_info,
    select_kv_cache_config,
)

__all__ = [
    "WorkloadSpec",
    "HardwareInfo",
    "AutoConfigResult",
    "detect_hardware_info",
    "select_kv_cache_config",
]

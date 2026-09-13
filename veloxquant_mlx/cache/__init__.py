"""KV-cache implementations for every quantization/eviction method in VeloxQuant-MLX.

Each sibling module implements one method's ``KVCache`` (TurboQuant, PolarQuant,
QJL, KIVI-family, eviction methods like SnapKV/H2O/PyramidKV/ChunkKV, cross-layer
methods like XQuant/MiniCache/xKV, etc.); this package re-exports the most
commonly used cache classes plus the construction entry points —
:class:`KVCacheConfig` (the shared configuration dataclass),
:class:`KVCacheFactory` (builds a single cache from a config), and
:class:`KVCacheBuilder` (fluent config builder / per-model cache-list
construction via ``for_model``). See ``base.py`` for the full method
registry and the standalone-vs-mlx_lm-protocol distinction.
"""

from __future__ import annotations

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.cachegen_cache import CacheGenKVCache
from veloxquant_mlx.cache.cam_cache import CaMKVCache
from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
from veloxquant_mlx.cache.gear_cache import GEARKVCache
from veloxquant_mlx.cache.h2o_cache import H2OKVCache
from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
from veloxquant_mlx.cache.palu_cache import PALUKVCache
from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
from veloxquant_mlx.cache.pyramidkv_cache import PyramidKVCache
from veloxquant_mlx.cache.qjl_cache import QJLKVCache
from veloxquant_mlx.cache.rocketkv_cache import RocketKVKVCache
from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
from veloxquant_mlx.cache.squeeze_cache import SqueezeAttentionCache
from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache
from veloxquant_mlx.cache.tova_cache import TOVAKVCache
from veloxquant_mlx.cache.turboquant_cache import TurboQuantKVCache
from veloxquant_mlx.cache.zipcache_cache import ZipCacheKVCache

__all__ = [
    "KVCacheBuilder",
    "KVCacheConfig",
    "KVCacheFactory",
    "CacheGenKVCache",
    "GEARKVCache",
    "MiniCacheKVCache",
    "PALUKVCache",
    "PolarQuantKVCache",
    "QJLKVCache",
    "SlidingWindowKVCache",
    "TurboQuantKVCache",
    "ZipCacheKVCache",
    "SnapKVKVCache",
    "StreamingLLMKVCache",
    "H2OKVCache",
    "TOVAKVCache",
    "PyramidKVCache",
    "SqueezeAttentionCache",
    "ChunkKVCache",
    "CaMKVCache",
    "RocketKVKVCache",
]

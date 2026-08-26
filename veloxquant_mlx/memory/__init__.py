"""KV-cache-aware block pool allocator (issue #249).

Fixed-size block allocation and reuse for KV-cache storage, addressing
fragmentation and repeated malloc/free overhead during long-running or
multi-request inference — the memory-allocation analogue of the bit-level
compression the rest of VeloxQuant-MLX provides.

Core pieces:

- :class:`BlockPoolAllocator` / :class:`PoolConfig` — pure-Python block
  bookkeeping (allocate, free, reuse, per-owner tracking, fragmentation and
  allocation stats). No MLX dependency; usable and testable on any platform.
- :class:`MLXBlockStorage` — pairs with a pool to provide the actual
  `mx.array` buffers, one per block id, reused in place across requests and
  supporting mixed compression formats (different blocks may hold fp16,
  int8, int4, ... data) within the same pool.
- :class:`PooledKVCache` — wraps any existing VeloxQuant ``KVCache`` so its
  token appends check out/return blocks from a shared pool, without
  changing the wrapped cache's own compression logic.

Typical usage::

    from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, PooledKVCache
    from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

    pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=512))
    inner = KVCacheBuilder().with_method("kivi").with_head_dim(128).build()
    cache = PooledKVCache(inner, pool, owner=request_id, format="int2")
    ...
    cache.release()  # return blocks to the pool for reuse by the next request
"""

from __future__ import annotations

from veloxquant_mlx.memory.block_pool import (
    AllocationStats,
    Block,
    BlockPoolAllocator,
    PoolConfig,
)
from veloxquant_mlx.memory.pooled_cache import PooledKVCache

__all__ = [
    "AllocationStats",
    "Block",
    "BlockPoolAllocator",
    "PoolConfig",
    "PooledKVCache",
]


def __getattr__(name: str):
    if name == "MLXBlockStorage":
        from veloxquant_mlx.memory.mlx_storage import MLXBlockStorage

        return MLXBlockStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

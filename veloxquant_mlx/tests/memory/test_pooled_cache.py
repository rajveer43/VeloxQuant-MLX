"""Tests for PooledKVCache, the block-pool-backed KVCache wrapper."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.pooled_cache import PooledKVCache


class _FakeKVCache:
    """Minimal VeloxQuant KVCache stub that just counts appended tokens."""

    def __init__(self) -> None:
        self.keys: list[Any] = []
        self.values: list[Any] = []

    def append_key(self, k: Any) -> None:
        self.keys.append(k)

    def append_value(self, v: Any) -> None:
        self.values.append(v)

    def attend(self, q: Any) -> Any:
        return mx.zeros((1,))

    def memory_bytes(self) -> int:
        return len(self.keys) * 2

    def __len__(self) -> int:
        return len(self.values)


BLOCK_SIZE = 4


def _pool(n_blocks: int = 16) -> BlockPoolAllocator:
    return BlockPoolAllocator(PoolConfig(block_size=BLOCK_SIZE, n_blocks=n_blocks))


def test_append_delegates_to_inner_cache():
    pool = _pool()
    inner = _FakeKVCache()
    cache = PooledKVCache(inner, pool, owner=1)
    cache.append(mx.zeros((8,)), mx.zeros((8,)))
    assert len(inner.keys) == 1
    assert len(inner.values) == 1
    assert len(cache) == 1


def test_checks_out_one_block_per_block_size_tokens():
    pool = _pool()
    inner = _FakeKVCache()
    cache = PooledKVCache(inner, pool, owner=1)
    for _ in range(BLOCK_SIZE):
        cache.append(mx.zeros((8,)), mx.zeros((8,)))
    # Exactly one block's worth of tokens -> 1 K block + 1 V block.
    assert cache.n_blocks_held() == 2

    cache.append(mx.zeros((8,)), mx.zeros((8,)))
    # One token past the block boundary -> a second K and V block checked out.
    assert cache.n_blocks_held() == 4


def test_release_returns_blocks_to_pool():
    pool = _pool(n_blocks=8)
    inner = _FakeKVCache()
    cache = PooledKVCache(inner, pool, owner=1)
    for _ in range(BLOCK_SIZE + 1):
        cache.append(mx.zeros((8,)), mx.zeros((8,)))
    assert pool.stats.blocks_in_use() > 0
    cache.release()
    assert pool.stats.blocks_in_use() == 0
    assert cache.n_blocks_held() == 0


def test_two_caches_do_not_interfere_with_different_owners():
    pool = _pool(n_blocks=8)
    cache1 = PooledKVCache(_FakeKVCache(), pool, owner=1)
    cache2 = PooledKVCache(_FakeKVCache(), pool, owner=2)
    cache1.append(mx.zeros((8,)), mx.zeros((8,)))
    cache2.append(mx.zeros((8,)), mx.zeros((8,)))
    cache1.release()
    assert cache1.n_blocks_held() == 0
    assert cache2.n_blocks_held() == 2  # unaffected by cache1's release


def test_attend_and_memory_bytes_delegate_to_inner():
    pool = _pool()
    inner = _FakeKVCache()
    cache = PooledKVCache(inner, pool, owner=1)
    cache.append(mx.zeros((8,)), mx.zeros((8,)))
    assert cache.memory_bytes() == inner.memory_bytes()
    out = cache.attend(mx.zeros((8,)))
    assert out.shape == (1,)


def test_pool_exhaustion_propagates_from_inner_allocate():
    from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError

    pool = BlockPoolAllocator(
        PoolConfig(block_size=4, n_blocks=1)
    )  # 0 free per stream when separate_kv
    cache = PooledKVCache(_FakeKVCache(), pool, owner=1)
    with pytest.raises(BlockPoolExhaustedError):
        cache.append(mx.zeros((8,)), mx.zeros((8,)))

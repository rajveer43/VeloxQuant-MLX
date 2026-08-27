"""Tests for PoolBackedKVCache: the mlx_lm-protocol cache backed by BlockPoolAllocator.

Compares against mlx_lm.models.cache.KVCache (the class this replaces) to
confirm the growth/state/trim contract is bit-for-bit compatible, and
checks that growth is actually reflected in the pool's own AllocationStats.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _StockKVCache

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.pool_backed_cache import PoolBackedKVCache, build_pooled_caches

B, H, D = 1, 2, 8


def _pool(block_size: int = 4, n_blocks: int = 64) -> BlockPoolAllocator:
    return BlockPoolAllocator(PoolConfig(block_size=block_size, n_blocks=n_blocks))


def _step(n_tokens: int, seed: int = 0):
    key = mx.random.key(seed)
    k = mx.random.normal((B, H, n_tokens, D), key=key, dtype=mx.float32).astype(mx.float16)
    v = mx.random.normal((B, H, n_tokens, D), key=key + 1, dtype=mx.float32).astype(mx.float16)
    return k, v


def test_matches_stock_kvcache_output_single_step():
    stock = _StockKVCache()
    pooled = PoolBackedKVCache(_pool(block_size=4, n_blocks=256), owner=1, step=stock.step)
    k, v = _step(3)
    sk, sv = stock.update_and_fetch(k, v)
    pk, pv = pooled.update_and_fetch(k, v)
    assert bool(mx.array_equal(sk, pk))
    assert bool(mx.array_equal(sv, pv))
    assert stock.offset == pooled.offset


def test_matches_stock_kvcache_across_many_steps_including_growth():
    stock = _StockKVCache()
    pool = _pool(block_size=4, n_blocks=256)
    pooled = PoolBackedKVCache(pool, owner=1, step=stock.step)
    for i in range(20):
        k, v = _step(3, seed=i * 2)
        sk, sv = stock.update_and_fetch(k, v)
        pk, pv = pooled.update_and_fetch(k, v)
        assert bool(mx.array_equal(sk, pk)), f"keys diverged at step {i}"
        assert bool(mx.array_equal(sv, pv)), f"values diverged at step {i}"
    assert stock.offset == pooled.offset == 60


def test_growth_is_recorded_on_the_pool():
    pool = _pool(block_size=4, n_blocks=64)
    cache = PoolBackedKVCache(pool, owner=1, step=4)
    assert pool.stats.n_allocations == 0
    k, v = _step(3)
    cache.update_and_fetch(k, v)
    # step=4 tokens grown, block_size=4 -> exactly one K block + one V block
    assert pool.stats.n_allocations == 2
    assert pool.stats.blocks_in_use() == 2


def test_no_reallocation_within_a_single_growth_chunk():
    pool = _pool(block_size=4, n_blocks=64)
    cache = PoolBackedKVCache(pool, owner=1, step=8)
    k1, v1 = _step(3, seed=0)
    cache.update_and_fetch(k1, v1)
    n_after_first = pool.stats.n_allocations
    k2, v2 = _step(2, seed=10)  # 3 + 2 = 5, still <= 8-token chunk
    cache.update_and_fetch(k2, v2)
    assert pool.stats.n_allocations == n_after_first  # no new growth needed


def test_release_frees_every_block_this_cache_grew():
    pool = _pool(block_size=4, n_blocks=64)
    cache = PoolBackedKVCache(pool, owner=1, step=4)
    for i in range(5):
        k, v = _step(3, seed=i)
        cache.update_and_fetch(k, v)
    assert pool.stats.blocks_in_use() > 0
    cache.release()
    assert pool.stats.blocks_in_use() == 0


def test_two_caches_share_pool_without_interference():
    pool = _pool(block_size=4, n_blocks=64)
    cache1 = PoolBackedKVCache(pool, owner=1, step=4)
    cache2 = PoolBackedKVCache(pool, owner=2, step=4)
    k, v = _step(3)
    cache1.update_and_fetch(k, v)
    cache2.update_and_fetch(k, v)
    cache1.release()
    assert pool.stats.blocks_in_use() == 2  # only cache2's blocks remain
    cache2.release()
    assert pool.stats.blocks_in_use() == 0


def test_released_blocks_are_reused_by_a_later_cache():
    pool = _pool(block_size=4, n_blocks=8)
    cache1 = PoolBackedKVCache(pool, owner=1, step=4)
    k, v = _step(4)
    cache1.update_and_fetch(k, v)
    cache1.release()

    cache2 = PoolBackedKVCache(pool, owner=2, step=4)
    cache2.update_and_fetch(k, v)
    assert pool.stats.n_reused >= 1


def test_trim_matches_stock_semantics():
    stock = _StockKVCache()
    pooled = PoolBackedKVCache(_pool(block_size=4, n_blocks=256), owner=1, step=stock.step)
    k, v = _step(10)
    stock.update_and_fetch(k, v)
    pooled.update_and_fetch(k, v)
    n_trimmed_stock = stock.trim(4)
    n_trimmed_pooled = pooled.trim(4)
    assert n_trimmed_stock == n_trimmed_pooled == 4
    assert stock.offset == pooled.offset == 6


def test_state_roundtrip():
    pool = _pool()
    cache = PoolBackedKVCache(pool, owner=1, step=8)
    k, v = _step(5)
    cache.update_and_fetch(k, v)
    keys, values = cache.state
    assert keys.shape == (B, H, 5, D)
    assert values.shape == (B, H, 5, D)


def test_empty_and_nbytes_before_any_update():
    cache = PoolBackedKVCache(_pool(), owner=1)
    assert cache.empty()
    assert cache.nbytes == 0


def test_is_trimmable():
    cache = PoolBackedKVCache(_pool(), owner=1)
    assert cache.is_trimmable() is True


def test_default_step_comes_from_pool_block_size():
    pool = _pool(block_size=32)
    cache = PoolBackedKVCache(pool, owner=1)
    assert cache.step == 32


def test_explicit_step_overrides_pool_block_size():
    pool = _pool(block_size=32)
    cache = PoolBackedKVCache(pool, owner=1, step=256)
    assert cache.step == 256


def test_build_pooled_caches_one_per_layer():
    class _FakeAttn:
        pass

    class _FakeLayer:
        def __init__(self):
            self.self_attn = _FakeAttn()

    class _FakeModel:
        def __init__(self, n_layers):
            self.layers = [_FakeLayer() for _ in range(n_layers)]

    pool = _pool()
    model = _FakeModel(n_layers=4)
    caches = build_pooled_caches(model, pool, owner=1)
    assert len(caches) == 4
    assert all(isinstance(c, PoolBackedKVCache) for c in caches)
    assert all(c.owner == 1 for c in caches)


def test_build_pooled_caches_share_one_owner_release():
    class _FakeLayer:
        self_attn = object()

    class _FakeModel:
        layers = [_FakeLayer(), _FakeLayer(), _FakeLayer()]

    pool = _pool(block_size=4, n_blocks=64)
    caches = build_pooled_caches(_FakeModel(), pool, owner=7, step=4)
    k, v = _step(3)
    for c in caches:
        c.update_and_fetch(k, v)
    assert pool.stats.blocks_in_use() > 0
    caches[0].release()  # any one cache releases every layer's blocks (shared owner)
    assert pool.stats.blocks_in_use() == 0

"""Tests for BlockPoolAllocator.free_all (VeloxQuant-MLX#510)."""

from __future__ import annotations

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig


def _pool(n_blocks: int = 64, block_size: int = 16) -> BlockPoolAllocator:
    # separate_kv=True (default) splits n_blocks in half per stream, so
    # double it here: callers pass the number of "k"-stream blocks they want.
    return BlockPoolAllocator(PoolConfig(block_size=block_size, n_blocks=n_blocks * 2))


def test_free_all_returns_every_block_to_the_free_list():
    pool = _pool(n_blocks=8)
    pool.allocate(stream="k", n_tokens=16 * 4, owner=1)
    assert pool.n_free("k") == 4

    pool.free_all(owner=1)
    assert pool.n_free("k") == 8
    assert pool.blocks_for(owner=1) == []


def test_free_all_only_affects_its_own_owner():
    pool = _pool(n_blocks=8)
    pool.allocate(stream="k", n_tokens=16 * 2, owner=1)
    pool.allocate(stream="k", n_tokens=16 * 2, owner=2)

    pool.free_all(owner=1)
    assert len(pool.blocks_for(owner=1)) == 0
    assert len(pool.blocks_for(owner=2)) == 2


def test_free_all_is_idempotent_for_unknown_or_already_freed_owner():
    pool = _pool(n_blocks=4)
    pool.free_all(owner=999)  # never allocated; must not raise

    pool.allocate(stream="k", n_tokens=16, owner=1)
    pool.free_all(owner=1)
    pool.free_all(owner=1)  # second call: owner already popped, no-op
    assert pool.n_free("k") == 4


def test_free_all_clears_owner_bookkeeping_not_just_blocks():
    pool = _pool(n_blocks=4)
    pool.allocate(stream="k", n_tokens=16, owner=1)
    pool.free_all(owner=1)
    assert 1 not in pool._owner_blocks
    assert 1 not in pool._active_owners


def test_free_all_updates_fragmentation_and_free_stats():
    pool = _pool(n_blocks=4, block_size=16)
    blocks = pool.allocate(stream="k", n_tokens=1, owner=1)  # 1 block, n_used=1 < 16 -> fragmented
    assert pool.stats._fragmented_count == 1

    n_frees_before = pool.stats.n_frees
    pool.free_all(owner=1)
    assert pool.stats._fragmented_count == 0
    assert pool.stats.n_frees == n_frees_before + len(blocks)


def test_free_all_scales_linearly_not_superlinearly():
    """VeloxQuant-MLX#510: free_all previously did an O(n) membership
    check + O(n) list.remove() per block even though it already has the
    owner's complete block-id list; confirmed 2x blocks -> 2.0-3.4x time
    before the fix. This is a coarse smoke check, not a strict benchmark.
    """
    import time

    def _time_free_all(n_blocks: int) -> float:
        pool = _pool(n_blocks=n_blocks, block_size=1)
        pool.allocate(stream="k", n_tokens=n_blocks, owner=1)
        start = time.perf_counter()
        pool.free_all(owner=1)
        return time.perf_counter() - start

    small = _time_free_all(2_000)
    large = _time_free_all(16_000)  # 8x blocks
    # A truly O(n) free_all should land near 8x; a superlinear one would
    # be dramatically worse. Generous upper bound to avoid CI flakiness.
    assert large < small * 20 + 0.05

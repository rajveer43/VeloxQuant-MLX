"""Unit tests for the KV-cache block pool allocator (no MLX required).

Lives outside veloxquant_mlx/tests/ on purpose — see
docs/CI_AND_TESTING.md#two-test-directories-and-why. BlockPoolAllocator
itself (veloxquant_mlx/memory/block_pool.py) has no MLX dependency, so it
gets fast, cheap coverage here instead of requiring the Apple Silicon
runner just to be collected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(mod_name: str, rel_path: str):
    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# core.exceptions has no third-party deps; load it first so block_pool's
# `from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError`
# resolves against this already-imported stand-in module.
_exceptions_mod = _load("veloxquant_mlx.core.exceptions", "veloxquant_mlx/core/exceptions.py")
sys.modules.setdefault("veloxquant_mlx", type(sys)("veloxquant_mlx"))
sys.modules.setdefault("veloxquant_mlx.core", type(sys)("veloxquant_mlx.core"))
sys.modules["veloxquant_mlx.core.exceptions"] = _exceptions_mod

_block_pool = _load("block_pool", "veloxquant_mlx/memory/block_pool.py")

AllocationStats = _block_pool.AllocationStats
Block = _block_pool.Block
BlockPoolAllocator = _block_pool.BlockPoolAllocator
PoolConfig = _block_pool.PoolConfig
BlockPoolExhaustedError = _exceptions_mod.BlockPoolExhaustedError


# --- PoolConfig validation --------------------------------------------------


def test_pool_config_rejects_invalid_block_size():
    with pytest.raises(ValueError):
        PoolConfig(block_size=0)


def test_pool_config_rejects_invalid_n_blocks():
    with pytest.raises(ValueError):
        PoolConfig(n_blocks=0)


# --- Basic allocate/free -----------------------------------------------------


def test_allocate_returns_enough_blocks_for_tokens():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=16, separate_kv=False))
    blocks = pool.allocate(stream="kv", n_tokens=10, owner=1)
    # ceil(10 / 4) = 3 blocks
    assert len(blocks) == 3
    assert sum(b.n_used for b in blocks) == 10


def test_allocate_exact_multiple_of_block_size():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=16, separate_kv=False))
    blocks = pool.allocate(stream="kv", n_tokens=8, owner=1)
    assert len(blocks) == 2
    assert all(b.n_used == 4 for b in blocks)


def test_free_returns_blocks_to_pool():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    blocks = pool.allocate(stream="kv", n_tokens=16, owner=1)
    assert pool.n_free(stream="kv") == 0
    pool.free(blocks)
    assert pool.n_free(stream="kv") == 4
    for b in blocks:
        assert b.owner is None
        assert b.n_used == 0


def test_free_all_returns_only_owners_blocks():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    pool.allocate(stream="kv", n_tokens=4, owner=2)
    pool.free_all(owner=1)
    # 8 blocks total, 1 checked out to owner=2, 1 freed back from owner=1 -> 7 free
    assert pool.n_free(stream="kv") == 7
    assert pool.blocks_for(owner=1) == []
    assert len(pool.blocks_for(owner=2)) == 1


def test_free_is_idempotent():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    blocks = pool.allocate(stream="kv", n_tokens=4, owner=1)
    pool.free(blocks)
    pool.free(blocks)  # should not raise or double-count
    assert pool.n_free(stream="kv") == 4


# --- Exhaustion --------------------------------------------------------------


def test_allocate_raises_when_pool_exhausted():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=2, separate_kv=False))
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=100, owner=1)
    assert pool.stats.n_exhausted == 1


def test_failed_allocation_is_all_or_nothing():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=2, separate_kv=False))
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=100, owner=1)
    # No blocks should have been handed out despite requesting more than fit.
    assert pool.n_free(stream="kv") == 2
    assert pool.blocks_for(owner=1) == []


def test_allocate_rejects_non_positive_n_tokens():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    with pytest.raises(ValueError):
        pool.allocate(stream="kv", n_tokens=0, owner=1)


# --- K/V separation ----------------------------------------------------------


def test_separate_kv_streams_do_not_share_free_lists():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=True))
    assert pool.n_free("k") == 4
    assert pool.n_free("v") == 4
    pool.allocate(stream="k", n_tokens=16, owner=1)
    assert pool.n_free("k") == 0
    assert pool.n_free("v") == 4  # v stream untouched


def test_invalid_stream_raises_when_separate_kv():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=True))
    with pytest.raises(ValueError):
        pool.allocate(stream="kv", n_tokens=1, owner=1)


def test_stream_argument_ignored_when_not_separate_kv():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    # Any stream label resolves to the single shared "kv" free list.
    blocks_k = pool.allocate(stream="k", n_tokens=4, owner=1)
    blocks_v = pool.allocate(stream="v", n_tokens=4, owner=1)
    assert blocks_k[0].stream == "kv"
    assert blocks_v[0].stream == "kv"


# --- Reuse across requests (the point of the pool) ---------------------------


def test_freed_blocks_are_reused_not_left_idle():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    first = pool.allocate(stream="kv", n_tokens=16, owner=1)
    first_ids = {b.block_id for b in first}
    pool.free_all(owner=1)

    second = pool.allocate(stream="kv", n_tokens=16, owner=2)
    second_ids = {b.block_id for b in second}

    assert first_ids == second_ids  # exact same physical blocks reused
    assert pool.stats.n_reused == len(second)


def test_never_used_blocks_are_not_counted_as_reused():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    assert pool.stats.n_reused == 0


# --- Stats: allocations, frees, peak, fragmentation --------------------------


def test_stats_track_allocations_and_frees():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=8, owner=1)
    assert pool.stats.n_allocations == 2
    pool.free_all(owner=1)
    assert pool.stats.n_frees == 2
    assert pool.stats.blocks_in_use() == 0


def test_peak_blocks_in_use_tracks_high_water_mark():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=16, owner=1)  # 4 blocks in use
    pool.free_all(owner=1)
    pool.allocate(stream="kv", n_tokens=4, owner=2)  # only 1 block in use now
    assert pool.stats.peak_blocks_in_use == 4
    assert pool.stats.blocks_in_use() == 1


def test_fragmentation_zero_when_all_blocks_full_or_free():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=8, owner=1)  # two full blocks, none partial
    assert pool.stats.fragmentation() == 0.0


def test_fragmentation_counts_partially_used_allocated_blocks():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=1, owner=1)  # 1 block, only 1/4 used -> fragmented
    assert pool.stats.fragmentation() == pytest.approx(1 / 4)


def test_allocation_stats_repr_does_not_raise():
    stats = AllocationStats(n_blocks=4)
    assert "AllocationStats" in repr(stats)


# --- Mixed compression formats ------------------------------------------------


def test_blocks_can_carry_different_format_tags_in_same_pool():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    fp16_blocks = pool.allocate(stream="kv", n_tokens=4, owner=1, format="fp16")
    int2_blocks = pool.allocate(stream="kv", n_tokens=4, owner=2, format="int2")
    assert fp16_blocks[0].format == "fp16"
    assert int2_blocks[0].format == "int2"


def test_pool_allocator_repr_does_not_raise():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    assert "BlockPoolAllocator" in repr(pool)


def test_block_repr_and_defaults():
    block = Block(block_id=0)
    assert block.owner is None
    assert block.n_used == 0
    assert block.format == "fp16"

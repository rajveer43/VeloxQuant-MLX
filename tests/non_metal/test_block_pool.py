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
import threading
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
OwnerAlreadyActiveError = _exceptions_mod.OwnerAlreadyActiveError


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


# --- Growth on exhaustion (PoolConfig.grow_on_exhaustion / max_blocks) -------


def test_grow_on_exhaustion_disabled_by_default_still_raises():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=2, separate_kv=False))
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=100, owner=1)


def test_grow_on_exhaustion_grows_pool_instead_of_raising():
    pool = BlockPoolAllocator(
        PoolConfig(block_size=4, n_blocks=2, separate_kv=False, grow_on_exhaustion=True)
    )
    blocks = pool.allocate(stream="kv", n_tokens=16, owner=1)  # needs 4 blocks, only 2 exist
    assert len(blocks) == 4
    assert pool.stats.n_grown == 2
    assert pool.stats.n_blocks == 4
    assert pool.stats.n_exhausted == 0


def test_grow_on_exhaustion_respects_max_blocks_cap():
    pool = BlockPoolAllocator(
        PoolConfig(
            block_size=4,
            n_blocks=2,
            separate_kv=False,
            grow_on_exhaustion=True,
            max_blocks=3,
        )
    )
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=16, owner=1)  # needs 4, cap allows only 1 more
    assert pool.stats.n_grown == 1  # grew as far as the cap allowed before still failing
    assert pool.stats.n_exhausted == 1


def test_grow_on_exhaustion_growth_is_all_or_nothing_under_cap():
    pool = BlockPoolAllocator(
        PoolConfig(
            block_size=4,
            n_blocks=2,
            separate_kv=False,
            grow_on_exhaustion=True,
            max_blocks=3,
        )
    )
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=16, owner=1)
    # Failed allocation still hands out nothing, even though growth happened.
    assert pool.blocks_for(owner=1) == []


def test_pool_config_rejects_max_blocks_below_n_blocks():
    with pytest.raises(ValueError):
        PoolConfig(n_blocks=8, max_blocks=4)


def test_grow_on_exhaustion_only_grows_the_exhausted_stream():
    pool = BlockPoolAllocator(
        PoolConfig(block_size=4, n_blocks=8, separate_kv=True, grow_on_exhaustion=True)
    )
    pool.allocate(stream="k", n_tokens=100, owner=1)  # exhausts and grows "k"
    assert pool.n_free("v") == 4  # v stream untouched


# --- Concurrency safety -------------------------------------------------------


def test_concurrent_allocations_from_many_threads_do_not_corrupt_stats():
    pool = BlockPoolAllocator(PoolConfig(block_size=1, n_blocks=500, separate_kv=False))
    errors: list[Exception] = []

    def worker(owner: int) -> None:
        try:
            pool.allocate(stream="kv", n_tokens=1, owner=owner)
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(500)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert pool.stats.n_allocations == 500
    assert pool.n_free("kv") == 0
    # Every block handed out exactly once: no two threads got the same block id.
    all_ids = [b.block_id for owner in range(500) for b in pool.blocks_for(owner)]
    assert len(all_ids) == len(set(all_ids)) == 500


def test_concurrent_allocate_and_free_do_not_corrupt_free_list():
    pool = BlockPoolAllocator(PoolConfig(block_size=1, n_blocks=64, separate_kv=False))

    def worker(owner: int) -> None:
        for _ in range(20):
            pool.allocate(stream="kv", n_tokens=1, owner=owner)
            pool.free_all(owner=owner)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pool.n_free("kv") == 64  # everything returned; no block leaked or double-freed
    assert pool.stats.n_allocations == pool.stats.n_frees == 320


# --- Free-list shrink / defragmentation --------------------------------------


def test_shrink_retires_only_free_blocks():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)  # 1 block in use, 7 free
    retired = pool.shrink(stream="kv", target_free=2)
    assert retired == 5
    assert pool.n_free("kv") == 2
    assert pool.stats.n_blocks == 3
    assert pool.stats.n_retired == 5


def test_shrink_never_touches_in_use_blocks():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    blocks = pool.allocate(stream="kv", n_tokens=16, owner=1)  # all 4 blocks in use
    retired = pool.shrink(stream="kv", target_free=0)
    assert retired == 0
    assert pool.stats.blocks_in_use() == 4
    pool.free(blocks)
    assert pool.n_free("kv") == 4


def test_shrink_is_noop_when_already_at_or_below_target():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    retired = pool.shrink(stream="kv", target_free=10)
    assert retired == 0
    assert pool.n_free("kv") == 4


def test_shrink_rejects_negative_target():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=4, separate_kv=False))
    with pytest.raises(ValueError):
        pool.shrink(stream="kv", target_free=-1)


# --- Owner collision protection -----------------------------------------------


def test_allocate_does_not_raise_for_repeated_allocate_by_same_active_owner():
    # A single request legitimately calls allocate() multiple times as it
    # grows (e.g. PoolBackedKVCache appending more blocks) -- this must not
    # be treated as a collision.
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    blocks = pool.allocate(stream="kv", n_tokens=4, owner=1)
    assert len(blocks) == 1


def test_registered_owner_still_active_after_being_used_by_allocate():
    # A caller that wants collision detection registers the owner id up
    # front (e.g. when handing it out from a request-id counter). allocate()
    # sees it's already active and proceeds (continuation, not collision),
    # but the id stays reserved for a *second* register_owner() call --
    # exactly the case that catches a real accidental id reuse.
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.register_owner(1)
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    with pytest.raises(OwnerAlreadyActiveError):
        pool.register_owner(1)


def test_owner_id_reusable_after_free_all():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    pool.free_all(owner=1)
    # Same owner id, new "request" -- should not raise now that it's released.
    blocks = pool.allocate(stream="kv", n_tokens=4, owner=1)
    assert len(blocks) == 1


def test_register_owner_raises_on_collision_before_any_allocation():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.register_owner(1)
    with pytest.raises(OwnerAlreadyActiveError):
        pool.register_owner(1)


def test_release_owner_clears_registration_without_blocks():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.register_owner(1)
    pool.release_owner(1)
    pool.register_owner(1)  # no longer active, should not raise


def test_failed_allocation_does_not_leave_owner_stuck_active():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=1, separate_kv=False))
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate(stream="kv", n_tokens=100, owner=1)
    # Registration happens before the exhaustion check fires; the owner
    # never got any blocks, so free_all is a no-op, and re-registering
    # cleanly should still be possible via release_owner.
    pool.release_owner(1)
    pool.register_owner(1)


# --- Observability: history / Prometheus export -------------------------------


def test_history_records_a_snapshot_per_mutation():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    assert len(pool.history) == 0
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    assert len(pool.history) == 1
    pool.free_all(owner=1)
    assert len(pool.history) == 2


def test_history_snapshots_are_independent_of_live_stats():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    snapshot = pool.history[-1]
    pool.allocate(stream="kv", n_tokens=4, owner=2)
    assert snapshot.n_allocations == 1  # frozen at the time it was recorded
    assert pool.stats.n_allocations == 2  # live stats kept moving


def test_history_is_bounded():
    pool = BlockPoolAllocator(PoolConfig(block_size=1, n_blocks=4, separate_kv=False))
    for i in range(_block_pool.DEFAULT_HISTORY_SIZE + 50):
        pool.allocate(stream="kv", n_tokens=1, owner=i)
        pool.free_all(owner=i)
    assert len(pool.history) == _block_pool.DEFAULT_HISTORY_SIZE


def test_stats_to_prometheus_contains_expected_metrics():
    pool = BlockPoolAllocator(PoolConfig(block_size=4, n_blocks=8, separate_kv=False))
    pool.allocate(stream="kv", n_tokens=4, owner=1)
    text = pool.stats.to_prometheus()
    assert "veloxquant_block_pool_blocks_in_use 1" in text
    assert "veloxquant_block_pool_blocks_total 8" in text
    assert "veloxquant_block_pool_allocations_total 1" in text
    assert text.endswith("\n")


def test_stats_to_prometheus_custom_prefix():
    stats = AllocationStats(n_blocks=1)
    text = stats.to_prometheus(prefix="my_pool")
    assert "my_pool_blocks_in_use 0" in text
    assert "veloxquant_block_pool" not in text


def test_stats_snapshot_is_a_detached_copy():
    stats = AllocationStats(n_blocks=4, n_allocations=2)
    snap = stats.snapshot()
    stats.n_allocations = 99
    assert snap.n_allocations == 2

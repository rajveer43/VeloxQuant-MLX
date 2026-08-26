"""Tests for MLXBlockStorage, the mx.array-backed storage layer for BlockPoolAllocator."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.mlx_storage import MLXBlockStorage

HEAD_DIM = 8
BLOCK_SIZE = 4


def _pool(n_blocks: int = 8, separate_kv: bool = False) -> BlockPoolAllocator:
    return BlockPoolAllocator(
        PoolConfig(block_size=BLOCK_SIZE, n_blocks=n_blocks, separate_kv=separate_kv)
    )


def test_buffer_for_allocates_zeros_of_expected_shape():
    pool = _pool()
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block,) = pool.allocate(stream="kv", n_tokens=1, owner=1)
    buf = storage.buffer_for(block)
    assert buf.shape == (BLOCK_SIZE, HEAD_DIM)
    assert bool(mx.all(buf == 0))


def test_write_then_read_round_trips_values():
    pool = _pool()
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block,) = pool.allocate(stream="kv", n_tokens=2, owner=1)
    values = mx.arange(2 * HEAD_DIM, dtype=mx.float16).reshape(2, HEAD_DIM)
    storage.write(block, offset=0, values=values)
    out = storage.read(block)
    assert out.shape == (2, HEAD_DIM)
    assert bool(mx.all(out == values))


def test_write_overflow_raises():
    pool = _pool()
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block,) = pool.allocate(stream="kv", n_tokens=1, owner=1)
    values = mx.zeros((BLOCK_SIZE + 1, HEAD_DIM), dtype=mx.float16)
    with pytest.raises(ValueError):
        storage.write(block, offset=0, values=values)


def test_reused_block_buffer_is_recycled_not_reallocated():
    pool = _pool(n_blocks=4)
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block1,) = pool.allocate(stream="kv", n_tokens=1, owner=1)
    buf1 = storage.buffer_for(block1)
    pool.free_all(owner=1)
    (block2,) = pool.allocate(stream="kv", n_tokens=1, owner=2, format="fp16")
    assert block2.block_id == block1.block_id  # LIFO reuse: same physical block
    buf2 = storage.buffer_for(block2)
    assert buf1 is buf2  # buffer object reused in place, no new allocation


def test_format_change_reallocates_buffer_with_new_dtype():
    pool = _pool(n_blocks=4)
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block,) = pool.allocate(stream="kv", n_tokens=1, owner=1, format="fp16")
    buf_fp16 = storage.buffer_for(block)
    assert buf_fp16.dtype == mx.float16

    pool.free_all(owner=1)
    (block2,) = pool.allocate(stream="kv", n_tokens=1, owner=2, format="int2")
    buf_int2 = storage.buffer_for(block2)
    assert buf_int2.dtype == mx.uint8


def test_resident_bytes_reflects_materialized_buffers_only():
    pool = _pool(n_blocks=4)
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    assert storage.resident_bytes() == 0
    (block,) = pool.allocate(stream="kv", n_tokens=1, owner=1, format="fp16")
    storage.buffer_for(block)
    expected = BLOCK_SIZE * HEAD_DIM * 2  # fp16 = 2 bytes/element
    assert storage.resident_bytes() == expected


def test_release_drops_buffer():
    pool = _pool(n_blocks=4)
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (block,) = pool.allocate(stream="kv", n_tokens=1, owner=1)
    storage.buffer_for(block)
    assert storage.resident_bytes() > 0
    pool.free([block])
    storage.release(block)
    assert storage.resident_bytes() == 0


def test_mixed_formats_coexist_in_same_pool():
    pool = _pool(n_blocks=8)
    storage = MLXBlockStorage(pool, head_dim=HEAD_DIM)
    (fp16_block,) = pool.allocate(stream="kv", n_tokens=1, owner=1, format="fp16")
    (int8_block,) = pool.allocate(stream="kv", n_tokens=1, owner=2, format="int8")
    fp16_buf = storage.buffer_for(fp16_block)
    int8_buf = storage.buffer_for(int8_block)
    assert fp16_buf.dtype == mx.float16
    assert int8_buf.dtype == mx.uint8

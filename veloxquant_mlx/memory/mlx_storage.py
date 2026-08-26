from __future__ import annotations

from typing import Any

from veloxquant_mlx.memory.block_pool import Block, BlockPoolAllocator

# Bytes per element for the compression formats a pool is expected to host
# side by side. "fp16" blocks store raw vectors; the int* formats store
# packed codes plus the fixed per-block scale/zero-point overhead is
# accounted for by the caller (the packed byte count already reflects the
# code width; this table only disambiguates same-block-size formats for
# stats/reporting).
_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp16": 2.0,
    "fp32": 4.0,
    "int8": 1.0,
    "int4": 0.5,
    "int2": 0.25,
    "int1": 0.125,
}


class MLXBlockStorage:
    """Backing array storage for a :class:`BlockPoolAllocator`.

    Allocates one `mx.array` buffer per block id, shaped
    ``(block_size, head_dim)``, lazily on first write. Blocks recycled by
    the pool reuse their existing buffer in place — writing into a reused
    block overwrites its previous contents without a new `mx.array`
    allocation, which is the point of pairing this with the pool (no
    malloc/free churn during steady-state generation).

    Because different blocks may hold different compression formats
    (``Block.format``), each block's buffer dtype is chosen from its
    format at write time; a block's buffer is only reallocated when the
    format assigned to that block id actually changes (rare — format is
    normally stable for the life of a request).

    Args:
        pool: The allocator this storage is paired with.
        head_dim: Width of each stored vector (last axis of every buffer).
    """

    def __init__(self, pool: BlockPoolAllocator, head_dim: int) -> None:
        self.pool = pool
        self.head_dim = head_dim
        self._buffers: dict[int, Any] = {}
        self._dtypes: dict[int, str] = {}

    @staticmethod
    def _mx_dtype(format: str):
        import mlx.core as mx  # lazy import — keeps this module importable without MLX installed

        if format in ("fp16",):
            return mx.float16
        if format == "fp32":
            return mx.float32
        if format in ("int8", "int4", "int2", "int1"):
            # Sub-byte formats are packed by the caller's codec; the pool
            # stores packed codes as uint8 regardless of bit-width.
            return mx.uint8
        raise ValueError(f"MLXBlockStorage: unsupported format {format!r}")

    def buffer_for(self, block: Block):
        """Return the `mx.array` buffer backing ``block``, allocating it if needed.

        Args:
            block: A block returned by ``pool.allocate()``.

        Returns:
            mx.array of shape (pool.config.block_size, head_dim).
        """
        import mlx.core as mx

        existing_dtype = self._dtypes.get(block.block_id)
        if block.block_id not in self._buffers or existing_dtype != block.format:
            dtype = self._mx_dtype(block.format)
            self._buffers[block.block_id] = mx.zeros(
                (self.pool.config.block_size, self.head_dim), dtype=dtype
            )
            self._dtypes[block.block_id] = block.format
        return self._buffers[block.block_id]

    def write(self, block: Block, offset: int, values) -> None:
        """Write ``values`` into ``block`` starting at token-slot ``offset``.

        Args:
            block: Destination block.
            offset: Token-slot offset within the block (0-indexed).
            values: mx.array of shape (n, head_dim) to write, n <= block_size - offset.
        """
        buf = self.buffer_for(block)
        n = values.shape[0]
        if offset + n > self.pool.config.block_size:
            raise ValueError(
                f"MLXBlockStorage: write of {n} rows at offset {offset} overflows "
                f"block_size={self.pool.config.block_size}"
            )
        buf[offset : offset + n] = values
        self._buffers[block.block_id] = buf

    def read(self, block: Block):
        """Return the live (used) slice of ``block``'s buffer.

        Args:
            block: Source block.

        Returns:
            mx.array of shape (block.n_used, head_dim).
        """
        buf = self.buffer_for(block)
        return buf[: block.n_used]

    def resident_bytes(self) -> int:
        """Total bytes currently materialized across all allocated buffers.

        Only counts blocks that have had a buffer allocated (lazily, on
        first write/read) — this is the actual resident footprint, not the
        pool's full pre-allocated capacity.
        """
        total = 0.0
        for block_id, buf in self._buffers.items():
            fmt = self._dtypes[block_id]
            per_elem = _BYTES_PER_ELEMENT.get(fmt, buf.dtype.size)
            total += buf.shape[0] * buf.shape[1] * per_elem
        return int(total)

    def release(self, block: Block) -> None:
        """Drop the buffer for a freed block, letting MLX reclaim its memory.

        Call this after ``pool.free([block])`` if you want resident memory
        to shrink immediately; otherwise the buffer is kept around (still
        zero new-allocation cost) and silently reused/overwritten the next
        time this block id is handed out.
        """
        self._buffers.pop(block.block_id, None)
        self._dtypes.pop(block.block_id, None)

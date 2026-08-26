from __future__ import annotations

from dataclasses import dataclass, field

from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError

# A block holds this many token slots' worth of storage. Chosen to amortize
# per-allocation overhead (mirrors vLLM/PagedAttention-style block sizes)
# while keeping internal fragmentation (up to block_size - 1 wasted slots
# per sequence) small relative to typical generation lengths.
DEFAULT_BLOCK_SIZE: int = 16


@dataclass
class PoolConfig:
    """Configuration for a :class:`BlockPoolAllocator`.

    Attributes:
        block_size: Number of token slots stored per block.
        n_blocks: Total number of blocks the pool manages.
        separate_kv: If True, keys and values are allocated from
            independent free lists (K and V blocks are never interchanged).
            If False, a single free list backs both keys and values, which
            packs tighter when K and V pressure is uneven across requests.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    n_blocks: int = 1024
    separate_kv: bool = True

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"PoolConfig: block_size must be >= 1, got {self.block_size}")
        if self.n_blocks < 1:
            raise ValueError(f"PoolConfig: n_blocks must be >= 1, got {self.n_blocks}")


@dataclass
class Block:
    """A single fixed-size block of KV-cache storage.

    Attributes:
        block_id: Index into the pool's backing storage; stable for the
            lifetime of the pool (blocks are reused in place, never moved).
        stream: Which logical stream ("k" or "v") this block belongs to.
            Always "kv" when the owning pool was built with
            ``separate_kv=False``.
        format: Compression-format tag of the data currently stored in this
            block (e.g. "fp16", "int2", "int4"). Lets a single pool host
            blocks written by different quantization methods concurrently.
        n_used: Number of token slots currently occupied (<= block_size).
        owner: Opaque request id the block is currently checked out to,
            or None if the block is free.
        ever_allocated: True once this block id has been handed out by
            :meth:`BlockPoolAllocator.allocate` at least once. Used to
            distinguish a fresh allocation from a reuse for stats purposes.
    """

    block_id: int
    stream: str = "kv"
    format: str = "fp16"
    n_used: int = 0
    owner: int | None = None
    ever_allocated: bool = False


@dataclass
class AllocationStats:
    """Running counters for a :class:`BlockPoolAllocator`.

    All counts are cumulative since pool construction unless noted.

    Attributes:
        n_blocks: Total blocks managed by the pool.
        n_allocations: Number of successful allocate() calls.
        n_frees: Number of free() calls.
        n_reused: Allocations satisfied by a block that had previously been
            freed and returned to the pool (as opposed to a block that had
            never been handed out before).
        n_exhausted: Allocation attempts that failed because no free block
            was available.
        peak_blocks_in_use: High-water mark of concurrently allocated blocks.
    """

    n_blocks: int
    n_allocations: int = 0
    n_frees: int = 0
    n_reused: int = 0
    n_exhausted: int = 0
    peak_blocks_in_use: int = 0

    def blocks_in_use(self) -> int:
        """Current number of allocated (non-free) blocks."""
        return self.n_allocations - self.n_frees

    def fragmentation(self) -> float:
        """Fraction of managed blocks that are neither free nor fully used.

        A block counts as fragmented once it has been allocated (n_used > 0)
        but is not full. Returns 0.0 for an allocator with no blocks.
        """
        if self.n_blocks == 0:
            return 0.0
        return self._fragmented_count / self.n_blocks

    _fragmented_count: int = field(default=0, repr=False)

    def __repr__(self) -> str:
        return (
            f"AllocationStats(in_use={self.blocks_in_use()}/{self.n_blocks}, "
            f"allocations={self.n_allocations}, frees={self.n_frees}, "
            f"reused={self.n_reused}, exhausted={self.n_exhausted}, "
            f"peak={self.peak_blocks_in_use}, fragmentation={self.fragmentation():.3f})"
        )


class BlockPoolAllocator:
    """Fixed-size block allocator for KV-cache storage.

    Pre-allocates ``n_blocks`` fixed-size blocks up front and hands them out
    on request, avoiding per-token malloc/free during generation. Freed
    blocks return to a free list and are reused by later allocations
    (LIFO — a just-freed block tends to be cache-hot), so steady-state
    generation performs zero new-memory allocation once the pool is warm.

    Keys and values can be tracked on independent free lists
    (``PoolConfig.separate_kv=True``, the default) so K-heavy and V-heavy
    workloads (e.g. asymmetric bit-widths) don't compete for the same
    blocks, or share a single free list for tighter packing when K/V
    pressure is uneven across requests.

    This class only manages *block bookkeeping* (which block ids are free,
    who owns which, how full each is, fragmentation/latency stats). It does
    not itself own array storage — pair it with a backing store (e.g.
    :class:`veloxquant_mlx.memory.mlx_storage.MLXBlockStorage`) that
    allocates the actual `mx.array` buffers indexed by block id.

    Args:
        config: Pool sizing/behavior configuration.

    Example::

        pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=256))
        blocks = pool.allocate(stream="k", n_tokens=40, owner=request_id, format="int2")
        ...
        pool.free_all(owner=request_id)
    """

    def __init__(self, config: PoolConfig | None = None) -> None:
        self.config = config or PoolConfig()
        self._streams = ("k", "v") if self.config.separate_kv else ("kv",)

        self._free: dict[str, list[int]] = {s: [] for s in self._streams}
        self._blocks: dict[int, Block] = {}
        self._owner_blocks: dict[int, list[int]] = {}

        block_id = 0
        for stream in self._streams:
            n_per_stream = (
                self.config.n_blocks
                if len(self._streams) == 1
                else (self.config.n_blocks // len(self._streams))
            )
            for _ in range(n_per_stream):
                self._blocks[block_id] = Block(block_id=block_id, stream=stream)
                self._free[stream].append(block_id)
                block_id += 1

        self.stats = AllocationStats(n_blocks=len(self._blocks))

    def _resolve_stream(self, stream: str) -> str:
        if not self.config.separate_kv:
            return "kv"
        if stream not in ("k", "v"):
            raise ValueError(f"BlockPoolAllocator: stream must be 'k' or 'v', got {stream!r}")
        return stream

    def allocate(
        self,
        stream: str,
        n_tokens: int,
        owner: int,
        format: str = "fp16",
    ) -> list[Block]:
        """Allocate enough blocks to hold ``n_tokens`` slots.

        Args:
            stream: "k" or "v" (ignored, treated as "kv", when the pool was
                built with ``separate_kv=False``).
            n_tokens: Number of token slots needed.
            owner: Opaque id (e.g. request id) the blocks are checked out to.
            format: Compression-format tag to record on each block.

        Returns:
            List of newly allocated :class:`Block` instances, in order.

        Raises:
            BlockPoolExhaustedError: If fewer than the required number of
                free blocks remain. No blocks are allocated on failure
                (all-or-nothing).
            ValueError: If ``n_tokens`` <= 0 or ``stream`` is invalid.
        """
        if n_tokens <= 0:
            raise ValueError(f"BlockPoolAllocator: n_tokens must be > 0, got {n_tokens}")
        s = self._resolve_stream(stream)
        n_needed = -(-n_tokens // self.config.block_size)  # ceil div

        free_list = self._free[s]
        if len(free_list) < n_needed:
            self.stats.n_exhausted += 1
            raise BlockPoolExhaustedError(
                f"BlockPoolAllocator: requested {n_needed} block(s) for stream "
                f"{s!r} but only {len(free_list)} free (pool size "
                f"{len(self._blocks)})."
            )

        allocated: list[Block] = []
        remaining = n_tokens
        for _ in range(n_needed):
            block_id = free_list.pop()  # LIFO: reuse the most recently freed block first
            block = self._blocks[block_id]
            is_reuse = block.ever_allocated
            block.owner = owner
            block.format = format
            block.n_used = min(remaining, self.config.block_size)
            block.ever_allocated = True
            remaining -= block.n_used
            allocated.append(block)
            self.stats.n_allocations += 1
            if is_reuse:
                self.stats.n_reused += 1

        self._owner_blocks.setdefault(owner, []).extend(b.block_id for b in allocated)
        self._recompute_fragmentation()
        self.stats.peak_blocks_in_use = max(
            self.stats.peak_blocks_in_use, self.stats.blocks_in_use()
        )
        return allocated

    def free(self, blocks: list[Block]) -> None:
        """Return blocks to the free list for their stream.

        Args:
            blocks: Blocks previously returned by :meth:`allocate`.
        """
        for block in blocks:
            if block.owner is None:
                continue  # already free; idempotent
            owner_list = self._owner_blocks.get(block.owner)
            if owner_list is not None and block.block_id in owner_list:
                owner_list.remove(block.block_id)
                if not owner_list:
                    del self._owner_blocks[block.owner]
            block.owner = None
            block.n_used = 0
            self._free[block.stream].append(block.block_id)
            self.stats.n_frees += 1
        self._recompute_fragmentation()

    def free_all(self, owner: int) -> None:
        """Free every block currently checked out to ``owner``.

        Args:
            owner: Opaque id passed to a prior :meth:`allocate` call.
        """
        block_ids = self._owner_blocks.get(owner, [])
        blocks = [self._blocks[bid] for bid in list(block_ids)]
        self.free(blocks)

    def blocks_for(self, owner: int) -> list[Block]:
        """Return the blocks currently checked out to ``owner``, in allocation order."""
        return [self._blocks[bid] for bid in self._owner_blocks.get(owner, [])]

    def n_free(self, stream: str = "k") -> int:
        """Number of free blocks remaining for ``stream``."""
        s = self._resolve_stream(stream)
        return len(self._free[s])

    def _recompute_fragmentation(self) -> None:
        full = self.config.block_size
        self.stats._fragmented_count = sum(
            1 for b in self._blocks.values() if b.owner is not None and b.n_used < full
        )

    def __repr__(self) -> str:
        return f"BlockPoolAllocator({self.stats!r})"

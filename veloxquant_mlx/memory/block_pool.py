from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError, OwnerAlreadyActiveError

# A block holds this many token slots' worth of storage. Chosen to amortize
# per-allocation overhead (mirrors vLLM/PagedAttention-style block sizes)
# while keeping internal fragmentation (up to block_size - 1 wasted slots
# per sequence) small relative to typical generation lengths.
DEFAULT_BLOCK_SIZE: int = 16

# How many AllocationStats snapshots to retain in BlockPoolAllocator.history.
# Bounded so long-running servers don't leak memory on the history buffer
# itself; recent pressure trends are what matter for catching exhaustion
# before it happens, not the full lifetime record.
DEFAULT_HISTORY_SIZE: int = 256


@dataclass
class PoolConfig:
    """Configuration for a :class:`BlockPoolAllocator`.

    Attributes:
        block_size: Number of token slots stored per block.
        n_blocks: Total number of blocks the pool manages initially.
        separate_kv: If True, keys and values are allocated from
            independent free lists (K and V blocks are never interchanged).
            If False, a single free list backs both keys and values, which
            packs tighter when K and V pressure is uneven across requests.
        grow_on_exhaustion: If True, an allocate() call that would otherwise
            raise BlockPoolExhaustedError instead grows the exhausted
            stream's block count just enough to satisfy the request (see
            max_blocks). If False (default), exhaustion always raises.
        max_blocks: Upper bound on total blocks across all streams once
            grown. None means unbounded growth. Ignored when
            grow_on_exhaustion is False. Growth that would exceed this cap
            still raises BlockPoolExhaustedError.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    n_blocks: int = 1024
    separate_kv: bool = True
    grow_on_exhaustion: bool = False
    max_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"PoolConfig: block_size must be >= 1, got {self.block_size}")
        if self.n_blocks < 1:
            raise ValueError(f"PoolConfig: n_blocks must be >= 1, got {self.n_blocks}")
        if self.max_blocks is not None and self.max_blocks < self.n_blocks:
            raise ValueError(
                f"PoolConfig: max_blocks ({self.max_blocks}) must be >= n_blocks ({self.n_blocks})"
            )


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
        n_grown: Number of blocks added to the pool by exhaustion-triggered
            growth (see PoolConfig.grow_on_exhaustion).
        n_retired: Number of blocks permanently removed from the pool by
            shrink() (see BlockPoolAllocator.shrink).
        peak_blocks_in_use: High-water mark of concurrently allocated blocks.
    """

    n_blocks: int
    n_allocations: int = 0
    n_frees: int = 0
    n_reused: int = 0
    n_exhausted: int = 0
    n_grown: int = 0
    n_retired: int = 0
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

    def snapshot(self) -> AllocationStats:
        """Return an independent copy of this stats object for history/export."""
        copy = AllocationStats(
            n_blocks=self.n_blocks,
            n_allocations=self.n_allocations,
            n_frees=self.n_frees,
            n_reused=self.n_reused,
            n_exhausted=self.n_exhausted,
            n_grown=self.n_grown,
            n_retired=self.n_retired,
            peak_blocks_in_use=self.peak_blocks_in_use,
        )
        copy._fragmented_count = self._fragmented_count
        return copy

    def to_prometheus(self, prefix: str = "veloxquant_block_pool") -> str:
        """Render these stats as Prometheus text-exposition-format gauges/counters.

        Args:
            prefix: Metric name prefix.

        Returns:
            Newline-terminated text in the Prometheus exposition format,
            suitable for serving directly from a ``/metrics`` endpoint or
            appending to a scrape response.
        """
        lines = [
            f"# TYPE {prefix}_blocks_in_use gauge",
            f"{prefix}_blocks_in_use {self.blocks_in_use()}",
            f"# TYPE {prefix}_blocks_total gauge",
            f"{prefix}_blocks_total {self.n_blocks}",
            f"# TYPE {prefix}_fragmentation gauge",
            f"{prefix}_fragmentation {self.fragmentation():.6f}",
            f"# TYPE {prefix}_peak_blocks_in_use gauge",
            f"{prefix}_peak_blocks_in_use {self.peak_blocks_in_use}",
            f"# TYPE {prefix}_allocations_total counter",
            f"{prefix}_allocations_total {self.n_allocations}",
            f"# TYPE {prefix}_frees_total counter",
            f"{prefix}_frees_total {self.n_frees}",
            f"# TYPE {prefix}_reused_total counter",
            f"{prefix}_reused_total {self.n_reused}",
            f"# TYPE {prefix}_exhausted_total counter",
            f"{prefix}_exhausted_total {self.n_exhausted}",
            f"# TYPE {prefix}_grown_total counter",
            f"{prefix}_grown_total {self.n_grown}",
            f"# TYPE {prefix}_retired_total counter",
            f"{prefix}_retired_total {self.n_retired}",
        ]
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return (
            f"AllocationStats(in_use={self.blocks_in_use()}/{self.n_blocks}, "
            f"allocations={self.n_allocations}, frees={self.n_frees}, "
            f"reused={self.n_reused}, exhausted={self.n_exhausted}, "
            f"grown={self.n_grown}, retired={self.n_retired}, "
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

    Thread-safe: every mutating and read method acquires an internal lock,
    so one pool can be shared across threads/concurrent request handlers
    without external synchronization. The lock is a plain (non-reentrant)
    ``threading.Lock`` — do not call back into the pool from inside a
    callback invoked while holding it.

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
        self._lock = threading.Lock()

        self._free: dict[str, list[int]] = {s: [] for s in self._streams}
        self._blocks: dict[int, Block] = {}
        self._owner_blocks: dict[int, list[int]] = {}
        self._active_owners: set[int] = set()
        self._next_block_id = 0

        for stream in self._streams:
            n_per_stream = (
                self.config.n_blocks
                if len(self._streams) == 1
                else (self.config.n_blocks // len(self._streams))
            )
            self._add_blocks(stream, n_per_stream)

        self.stats = AllocationStats(n_blocks=len(self._blocks))
        self.history: deque[AllocationStats] = deque(maxlen=DEFAULT_HISTORY_SIZE)

    def _add_blocks(self, stream: str, count: int) -> None:
        for _ in range(count):
            block_id = self._next_block_id
            self._next_block_id += 1
            self._blocks[block_id] = Block(block_id=block_id, stream=stream)
            self._free[stream].append(block_id)

    def _resolve_stream(self, stream: str) -> str:
        if not self.config.separate_kv:
            return "kv"
        if stream not in ("k", "v"):
            raise ValueError(f"BlockPoolAllocator: stream must be 'k' or 'v', got {stream!r}")
        return stream

    def _record_history(self) -> None:
        self.history.append(self.stats.snapshot())

    def register_owner(self, owner: int) -> None:
        """Mark ``owner`` as active, guarding against accidental id reuse.

        :meth:`allocate` calls this automatically for any owner not yet
        seen, but a second allocate() for an owner that is *already*
        active is treated as that same request continuing to grow, not a
        collision (see allocate()'s docstring) — it can't otherwise tell
        the two cases apart. Call register_owner() explicitly yourself
        (once, before any allocate() calls for this owner) if you want a
        genuine collision — e.g. a request-id counter that wrapped around,
        or two callers racing on the same id — to raise immediately,
        including on a second register_owner() call for a still-active id.

        Raises:
            OwnerAlreadyActiveError: If ``owner`` already has blocks checked
                out or was registered and not yet released.
        """
        with self._lock:
            self._register_owner_locked(owner)

    def _register_owner_locked(self, owner: int) -> None:
        if owner in self._active_owners:
            raise OwnerAlreadyActiveError(
                f"BlockPoolAllocator: owner {owner!r} is already active "
                "(has checked-out blocks or was registered without a "
                "matching free_all/release_owner). Use a different owner "
                "id, or free_all(owner) first if this is intentional reuse."
            )
        self._active_owners.add(owner)

    def release_owner(self, owner: int) -> None:
        """Clear ``owner`` from the active-owner set without freeing blocks.

        Rarely needed directly — :meth:`free_all` calls this once an
        owner's last block is freed. Useful if you registered an owner via
        :meth:`register_owner` but never actually allocated anything for it.
        """
        with self._lock:
            self._active_owners.discard(owner)

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
                Repeated allocate() calls for an owner that is already
                active on this pool are treated as that same request
                growing (the normal case — see PoolBackedKVCache) and never
                raise. To catch a *different* caller accidentally reusing
                an id that is still checked out, call
                :meth:`register_owner` yourself before the first
                allocate() — a second register_owner() for an id that is
                still active raises OwnerAlreadyActiveError.
            format: Compression-format tag to record on each block.

        Returns:
            List of newly allocated :class:`Block` instances, in order.

        Raises:
            BlockPoolExhaustedError: If fewer than the required number of
                free blocks remain and the pool cannot grow enough to cover
                the gap (see ``PoolConfig.grow_on_exhaustion``/``max_blocks``).
                No blocks are allocated on failure (all-or-nothing).
            ValueError: If ``n_tokens`` <= 0 or ``stream`` is invalid.
        """
        if n_tokens <= 0:
            raise ValueError(f"BlockPoolAllocator: n_tokens must be > 0, got {n_tokens}")
        s = self._resolve_stream(stream)
        n_needed = -(-n_tokens // self.config.block_size)  # ceil div

        with self._lock:
            if owner not in self._active_owners:
                self._register_owner_locked(owner)

            free_list = self._free[s]
            if len(free_list) < n_needed:
                self._try_grow_locked(s, n_needed - len(free_list))

            if len(free_list) < n_needed:
                self.stats.n_exhausted += 1
                self._record_history()
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
            self._record_history()
            return allocated

    def _try_grow_locked(self, stream: str, n_short: int) -> None:
        """Add blocks to ``stream`` to cover an ``n_short``-block shortfall, if allowed."""
        if not self.config.grow_on_exhaustion:
            return
        if self.config.max_blocks is not None:
            headroom = self.config.max_blocks - len(self._blocks)
            if headroom <= 0:
                return
            n_short = min(n_short, headroom)
        if n_short <= 0:
            return
        self._add_blocks(stream, n_short)
        self.stats.n_blocks = len(self._blocks)
        self.stats.n_grown += n_short

    def free(self, blocks: list[Block]) -> None:
        """Return blocks to the free list for their stream.

        Args:
            blocks: Blocks previously returned by :meth:`allocate`.
        """
        with self._lock:
            self._free_locked(blocks)
            self._record_history()

    def _free_locked(self, blocks: list[Block]) -> None:
        for block in blocks:
            if block.owner is None:
                continue  # already free; idempotent
            owner = block.owner
            owner_list = self._owner_blocks.get(owner)
            if owner_list is not None and block.block_id in owner_list:
                owner_list.remove(block.block_id)
                if not owner_list:
                    del self._owner_blocks[owner]
                    self._active_owners.discard(owner)
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
        with self._lock:
            block_ids = self._owner_blocks.get(owner, [])
            blocks = [self._blocks[bid] for bid in list(block_ids)]
            self._free_locked(blocks)
            self._active_owners.discard(owner)
            self._record_history()

    def blocks_for(self, owner: int) -> list[Block]:
        """Return the blocks currently checked out to ``owner``, in allocation order."""
        with self._lock:
            return [self._blocks[bid] for bid in self._owner_blocks.get(owner, [])]

    def n_free(self, stream: str = "k") -> int:
        """Number of free blocks remaining for ``stream``."""
        s = self._resolve_stream(stream)
        with self._lock:
            return len(self._free[s])

    def shrink(self, stream: str, target_free: int) -> int:
        """Permanently retire fully-free blocks from ``stream`` down to ``target_free``.

        Use this to give memory back when a pool was sized for peak load
        that turned out much higher than steady-state usage. Only blocks
        currently on the free list are eligible — in-use blocks are never
        touched, so this never breaks a live request. Retired blocks are
        removed from the pool's bookkeeping entirely (not just marked
        free/unfree), so any external backing storage indexed by block id
        (e.g. :class:`~veloxquant_mlx.memory.mlx_storage.MLXBlockStorage`)
        may drop the corresponding buffer.

        Args:
            stream: "k" or "v" (ignored, treated as "kv", when the pool was
                built with ``separate_kv=False``).
            target_free: Desired number of free blocks to retain for this
                stream after shrinking. If the stream already has this many
                or fewer free blocks, this is a no-op.

        Returns:
            Number of blocks actually retired.

        Raises:
            ValueError: If ``target_free`` < 0.
        """
        if target_free < 0:
            raise ValueError(f"BlockPoolAllocator: target_free must be >= 0, got {target_free}")
        s = self._resolve_stream(stream)
        with self._lock:
            free_list = self._free[s]
            n_to_retire = max(0, len(free_list) - target_free)
            retired = 0
            for _ in range(n_to_retire):
                block_id = free_list.pop()
                del self._blocks[block_id]
                retired += 1
            if retired:
                self.stats.n_blocks = len(self._blocks)
                self.stats.n_retired += retired
                self._recompute_fragmentation()
                self._record_history()
            return retired

    def _recompute_fragmentation(self) -> None:
        full = self.config.block_size
        self.stats._fragmented_count = sum(
            1 for b in self._blocks.values() if b.owner is not None and b.n_used < full
        )

    def __repr__(self) -> str:
        return f"BlockPoolAllocator({self.stats!r})"

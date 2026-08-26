from __future__ import annotations

from typing import Any

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.memory.block_pool import Block, BlockPoolAllocator

# Reserved for requests that have not been given an explicit id.
_DEFAULT_OWNER: int = -1


class PooledKVCache(KVCache):
    """Wraps any VeloxQuant KVCache with block-pool-backed memory accounting.

    The wrapped cache still owns and drives its own storage arrays (its
    compression format may be anything registered with
    :class:`~veloxquant_mlx.cache.base.KVCacheFactory`); this wrapper checks
    out and returns fixed-size blocks from a shared
    :class:`BlockPoolAllocator` in lock-step with token appends, so a
    multi-request server can track allocation counts, reuse, and
    fragmentation across every active cache from one pool instead of each
    cache doing its own ad hoc growth.

    Block granularity is ``pool.config.block_size`` tokens: a new block is
    requested from the pool only when the previous one fills up, and every
    block owned by this cache is returned to the pool in one call when the
    request completes (:meth:`release`). No block is ever allocated or
    freed on a per-token basis, matching the "avoid malloc/free during
    generation" goal in issue #249.

    Args:
        inner: The underlying KVCache that performs actual compression.
        pool: Shared BlockPoolAllocator to draw block accounting from.
        owner: Opaque id identifying this cache's request/sequence. Two
            PooledKVCache instances must use different owners to avoid
            each other's :meth:`release` reclaiming the wrong blocks.
        format: Compression-format tag recorded on each block this cache
            checks out (purely informational — passed through to
            ``pool.allocate(format=...)``).
    """

    def __init__(
        self,
        inner: KVCache,
        pool: BlockPoolAllocator,
        owner: int = _DEFAULT_OWNER,
        format: str = "fp16",
    ) -> None:
        self._inner = inner
        self._pool = pool
        self._owner = owner
        self._format = format
        self._k_blocks: list[Block] = []
        self._v_blocks: list[Block] = []
        self._k_used_in_last: int = 0
        self._v_used_in_last: int = 0
        self._n_tokens: int = 0

    def _ensure_capacity(self, blocks: list[Block], used_in_last: int, stream: str) -> int:
        block_size = self._pool.config.block_size
        if blocks and used_in_last < block_size:
            return used_in_last  # current tail block still has room
        new_blocks = self._pool.allocate(
            stream=stream, n_tokens=1, owner=self._owner, format=self._format
        )
        blocks.extend(new_blocks)
        return 0

    def append_key(self, k: Any) -> None:
        """Check out a K block from the pool if needed, then append to the inner cache.

        Args:
            k: Key vector, shape (d,), fp16.
        """
        self._k_used_in_last = self._ensure_capacity(self._k_blocks, self._k_used_in_last, "k")
        self._inner.append_key(k)
        self._k_used_in_last += 1
        self._k_blocks[-1].n_used = self._k_used_in_last

    def append_value(self, v: Any) -> None:
        """Check out a V block from the pool if needed, then append to the inner cache.

        Args:
            v: Value vector, shape (d,), fp16.
        """
        self._v_used_in_last = self._ensure_capacity(self._v_blocks, self._v_used_in_last, "v")
        self._inner.append_value(v)
        self._v_used_in_last += 1
        self._v_blocks[-1].n_used = self._v_used_in_last
        self._n_tokens += 1

    def attend(self, q: Any) -> Any:
        """Delegate attention computation to the inner cache.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """
        return self._inner.attend(q)

    def memory_bytes(self) -> int:
        """Return memory usage of the inner (compressed) cache."""
        return self._inner.memory_bytes()

    def release(self) -> None:
        """Return every block this cache holds to the pool.

        Call this once the request finishes (or is evicted) so its blocks
        become available for reuse by later requests.
        """
        self._pool.free_all(owner=self._owner)
        self._k_blocks = []
        self._v_blocks = []
        self._k_used_in_last = 0
        self._v_used_in_last = 0

    def n_blocks_held(self) -> int:
        """Total K + V blocks currently checked out to this cache."""
        return len(self._k_blocks) + len(self._v_blocks)

    def __len__(self) -> int:
        return self._n_tokens

    def __repr__(self) -> str:
        return (
            f"PooledKVCache(inner={self._inner!r}, owner={self._owner}, "
            f"blocks={self.n_blocks_held()})"
        )

from __future__ import annotations

from typing import Any

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.dsa.ring_buffer import RingBuffer


class SlidingWindowKVCache(KVCache):
    """Wraps any KVCache with sliding-window token eviction.

    Maintains a RingBuffer of the most recent ``window_size`` token slots.
    When the window is full, the oldest token is evicted.

    Because the inner cache does not support random deletion, we maintain
    a fresh inner cache per sliding window: a new one is created each time
    a token is evicted. This simple strategy is correct but rebuilds the
    cache every ``window_size`` tokens; suitable for inference not training.

    For efficiency, we cache all key-value vectors in the ring buffer and
    re-feed them to a fresh inner cache on window advance.

    Args:
        inner: The underlying KVCache to wrap.
        window_size: Number of tokens to keep.
    """

    def __init__(self, inner: KVCache, window_size: int) -> None:
        if window_size < 1:
            raise ValueError(f"SlidingWindowKVCache: window_size must be >= 1, got {window_size}")
        self._inner = inner
        self._window_size = window_size
        # Ring buffers for raw key and value vectors
        self._raw_keys: RingBuffer = RingBuffer(window_size)
        self._raw_values: RingBuffer = RingBuffer(window_size)
        self._pending_key: Any = None
        self._n_tokens: int = 0

    def append_key(self, k: Any) -> None:
        """Buffer a key vector for paired insertion with the next value.

        Args:
            k: Key vector, shape (d,), fp16.
        """
        self._pending_key = k

    def append_value(self, v: Any) -> None:
        """Insert a key-value pair and evict the oldest token if full.

        Args:
            v: Value vector, shape (d,), fp16.
        """
        if self._pending_key is None:
            raise RuntimeError(
                "SlidingWindowKVCache: append_value() called without a preceding append_key()."
            )
        evicted_k = self._raw_keys.append(self._pending_key)
        self._raw_values.append(v)
        self._pending_key = None
        self._n_tokens += 1

        # If there was an eviction, rebuild the inner cache
        if evicted_k is not None:
            self._rebuild_inner()

    def _rebuild_inner(self) -> None:
        """Rebuild the inner cache from the current window of raw vectors."""
        # Create a fresh inner cache of the same type
        from copy import deepcopy

        fresh = deepcopy(self._inner)
        # Reset its state
        if hasattr(fresh, "_n_tokens"):
            fresh._n_tokens = 0
        if hasattr(fresh, "_k_indices"):
            from veloxquant_mlx.dsa.ring_buffer import RingBuffer

            cap = fresh._k_indices._capacity
            fresh._k_indices = RingBuffer(cap)
            fresh._k_signs = RingBuffer(cap)
            fresh._k_norms = RingBuffer(cap)
            fresh._v_cache = RingBuffer(cap)
            fresh._v_scales = RingBuffer(cap)
            if hasattr(fresh, "_k_residual_norms"):
                fresh._k_residual_norms = RingBuffer(cap)
        # Re-append window tokens
        for i in range(len(self._raw_keys)):
            fresh.append_key(self._raw_keys[i])
            fresh.append_value(self._raw_values[i])
        self._inner = fresh

    def attend(self, q: Any) -> Any:
        """Delegate to the inner cache for attention computation.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """
        return self._inner.attend(q)

    def memory_bytes(self) -> int:
        """Return memory usage of the inner (windowed) cache."""
        return self._inner.memory_bytes()

    def __len__(self) -> int:
        return min(self._n_tokens, self._window_size)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowKVCache(window={self._window_size}, "
            f"n_stored={len(self)}, "
            f"total_seen={self._n_tokens})"
        )

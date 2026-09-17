"""Sliding-window eviction decorator wrapping any standalone KVCache.

:class:`SlidingWindowKVCache` composes over another
:class:`~veloxquant_mlx.core.abstractions.KVCache` instance to cap it at the
most recent ``window_size`` tokens. Because the wrapped caches (TurboQuant,
PolarQuant, QJL, etc.) don't support random deletion, eviction is
implemented by keeping raw key/value vectors in ring buffers and, each time
the window advances, resetting the inner cache to empty (via its own
``reset()``) and re-feeding it the current window — correct but O(window)
per eviction, so it targets inference rather than training. Only compatible
with :data:`~veloxquant_mlx.cache.base.STANDALONE_METHODS` (see
``KVCacheFactory.create``'s ``sliding_window`` handling).
"""

from __future__ import annotations

from typing import Any

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.dsa.ring_buffer import RingBuffer


class SlidingWindowKVCache(KVCache):
    """Wraps any KVCache with sliding-window token eviction.

    Maintains a RingBuffer of the most recent ``window_size`` token slots.
    When the window is full, the oldest token is evicted.

    Because the inner cache does not support random deletion, we maintain
    the window by resetting the inner cache to empty and re-feeding it the
    current window's tokens each time one is evicted. This simple strategy
    is correct but rebuilds the cache every ``window_size`` tokens; suitable
    for inference not training.

    We cache all key-value vectors in our own ring buffer so they're
    available to re-feed on each window advance.

    Args:
        inner: The underlying KVCache to wrap.
        window_size: Number of tokens to keep.
    """

    def __init__(self, inner: KVCache, window_size: int) -> None:
        if window_size < 1:
            raise QuantizerConfigError(
                f"SlidingWindowKVCache: window_size must be >= 1, got {window_size}"
            )
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
        """Rebuild the inner cache from the current window of raw vectors.

        Uses the wrapped cache's own ``reset()`` (part of the ``KVCache``
        ABC) rather than guessing internal attribute names — every concrete
        subclass knows how to clear its own token storage while preserving
        its quantizer/calibration state, which a generic reflective reset
        cannot do correctly across differently-shaped implementations. See
        VeloxQuant-MLX#274: the previous attribute-guessing reset matched no
        registered cache class, so eviction never actually happened and the
        inner cache grew unbounded across the life of the request.
        """
        self._inner.reset()
        for i in range(len(self._raw_keys)):
            self._inner.append_key(self._raw_keys[i])
            self._inner.append_value(self._raw_values[i])

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

    def reset(self) -> None:
        """Clear the window and the inner cache, returning both to empty."""
        self._raw_keys = RingBuffer(self._window_size)
        self._raw_values = RingBuffer(self._window_size)
        self._pending_key = None
        self._n_tokens = 0
        self._inner.reset()

    def __len__(self) -> int:
        return min(self._n_tokens, self._window_size)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowKVCache(window={self._window_size}, "
            f"n_stored={len(self)}, "
            f"total_seen={self._n_tokens})"
        )

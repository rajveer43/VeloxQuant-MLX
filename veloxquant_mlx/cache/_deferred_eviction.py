"""Shared attention inputs for caches that evict after fetching (#610)."""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask


class DeferredEvictionMixin:
    """Keep current attention separate from the state retained for the next call.

    Subclasses gather ``_positions`` with the same indices as their stored
    K/V. Positions have shape [B, H, kept]; masks use head zero, following
    the existing eviction mask contract and its documented limitations.
    """

    def __init__(self) -> None:
        super().__init__()
        self._positions: mx.array | None = None

    def _prepare_attention(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        B, H, S, _ = keys.shape
        positions = mx.broadcast_to(
            mx.arange(self.offset, self.offset + S, dtype=mx.int32)[None, None], (B, H, S)
        )
        keys, values = keys.astype(mx.float16), values.astype(mx.float16)
        if self._positions is not None:
            n = self._positions.shape[-1]
            keys = mx.concatenate([self.keys[:, :, :n], keys], axis=2)
            values = mx.concatenate([self.values[:, :, :n], values], axis=2)
            positions = mx.concatenate([self._positions, positions], axis=2)
        return keys, values, positions

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        if self._positions is None:
            return super().make_mask(N, return_array=return_array, window_size=window_size)
        B = self._positions.shape[0]
        queries = mx.broadcast_to(
            mx.arange(self.offset, self.offset + N, dtype=mx.int32)[None], (B, N)
        )
        positions = mx.concatenate([self._positions[:, 0], queries], axis=1)
        return eviction_make_mask(queries, positions, N, return_array, window_size)

"""StreamingLLM-adapted KV cache — sink + recency-window structural eviction.

Inspired by "Efficient Streaming Language Models with Attention Sinks"
(Xiao et al., ICLR 2024, arXiv:2309.17453). Documented as "StreamingLLM-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

Structural eviction: keep the first ``stream_n_sink`` token positions unconditionally
(attention sinks) plus the most recent ``stream_window_size`` token positions. All other
positions are permanently evicted. The cache never grows beyond
``stream_n_sink + stream_window_size`` positions, making decode constant-memory.

This is **positional** eviction — orthogonal to SnapKV-adapted (score-based eviction)
and to all quantization methods (which compress all tokens to fewer bits).

Adaptation limitations (stated plainly):
  - No attention mask adjustment: the model attends to all returned K/V positions; we
    only bound what K/V rows exist.
  - No RoPE position-ID remapping: original token positions are preserved in the
    returned rows.
  - Fixed sink count (``stream_n_sink``), not adaptive.

Byte accounting:
    stream_kept_bytes   — fp16 bytes stored (sink + recent positions, K + V)
    full_seq_bytes      — hypothetical cost if all tokens were kept as fp16
    streaming_ratio     — full_seq_bytes / stream_kept_bytes (> 1 once window fills)
    tokens_seen         — total positions ever passed to update_and_fetch (all heads avg)
    tokens_in_window    — current sink + recent positions in cache (first head)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.streaming_llm import (
    StreamingWindow,
    full_stream_fp16_bytes,
    init_streaming_window,
    stream_fp16_bytes,
    stream_get_kv,
    stream_update,
)


class StreamingLLMKVCache(_MLXKVCache):
    """KV cache implementing StreamingLLM-adapted sink + recency-window eviction.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``stream_n_sink``      (int, default 4)    — initial positions always kept,
            ``stream_window_size`` (int, default 512)  — FIFO recent-token capacity.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        The cache never grows beyond ``stream_n_sink + stream_window_size`` positions.
        Both prefill (S > 1) and decode (S == 1) tokens are processed identically —
        all go through the sink/window logic. This is the key difference from SnapKV-
        adapted, which evicts only at prefill. StreamingLLM operates continuously.
        Single-layer (no coordinator); ``for_model`` propagates all ``stream_*``
        fields automatically via ``dataclasses.replace``.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._n_sink = int(getattr(config, "stream_n_sink", 4))
        self._window_size = int(getattr(config, "stream_window_size", 512))

        self._windows: list[StreamingWindow] = []  # one per (B, H)
        self._B: int = 0
        self._H: int = 0
        self._D: int = 0

        self._stream_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0  # sum over all (B, H) heads

    # ------------------------------------------------------------------
    def _ensure_windows(self, B: int, H: int, D: int) -> None:
        """Initialise per-head window list on first call."""
        if len(self._windows) == 0:
            self._B = B
            self._H = H
            self._D = D
            self._windows = [init_streaming_window(self._n_sink, D) for _ in range(B * H)]

    def _window_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply sink+window eviction, return full window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens.
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_keep, D]`` fp16, where
            ``n_keep = min(n_sink + n_recent, stream_n_sink + stream_window_size)``.
        """
        B, H, S, D = keys.shape
        self._ensure_windows(B, H, D)

        # Byte accounting for this batch
        fp16_new = B * H * S * D * 2 * 2  # K + V, fp16
        self._full_seq_bytes += fp16_new
        self._tokens_seen_total += B * H * S

        # Update each head's window
        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._window_idx(b, h)
                w = self._windows[idx]
                w = stream_update(
                    w,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                    n_sink=self._n_sink,
                    window_size=self._window_size,
                )
                self._windows[idx] = w
                k_h, v_h = stream_get_kv(w)
                k_out_h.append(k_h)  # [n_keep, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_keep, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_keep, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Recount kept bytes from first (B=0, H=0) head as representative
        kept_bytes = stream_fp16_bytes(self._windows[0]) * B * H
        self._stream_kept_bytes = kept_bytes  # snapshot (not cumulative; current state)

        # K_out/V_out is the full sink+recent window every call, not a
        # delta — reset so the base class's append-only buffer starts fresh
        # instead of stacking on top of the previous call's rows. Without
        # this, self.keys/self.values/self.offset stay at __init__ defaults
        # forever, and mlx_lm's generate() crashes on `cache.state` during
        # chunked prefill (see #83).
        self.keys = None
        self.values = None
        self.offset = 0
        return super().update_and_fetch(K_out, V_out)

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def stream_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, sink + recent)."""
        return self._stream_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def streaming_ratio(self) -> float:
        """full_seq_bytes / stream_kept_bytes; > 1 once window fills."""
        if self._stream_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._stream_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_in_window(self) -> int:
        """Current sink + recent count for head (0, 0)."""
        if not self._windows:
            return 0
        w = self._windows[0]
        return w.n_sink + w.n_recent


__all__ = ["StreamingLLMKVCache"]

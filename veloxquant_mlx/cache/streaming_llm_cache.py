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

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.quantizers.streaming_llm import (
    StreamingWindow,
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

        # True absolute token position, independent of how many rows survive
        # eviction. Reported as ``self.offset`` so mlx_lm's RoPE stays correct
        # after tokens are dropped (see #171 and update_and_fetch).
        self._true_offset: int = 0

        # [B, n_kept] int32 true absolute position of each currently-stored
        # (head 0) row — see make_mask() and update_and_fetch()'s #370
        # deferred-eviction docstrings. None before the first update. Unlike
        # score-based eviction methods, StreamingLLM's kept positions are
        # purely structural (sink prefix + FIFO recency window) and need no
        # quantizer-side bookkeeping — derived directly from
        # self._windows[0]'s n_sink/n_recent/tokens_seen each call.
        self._kept_positions: mx.array | None = None

        # (K_out, V_out) actually returned by the last call — since #370's
        # deferred eviction, generally NOT the same as this call's full
        # (capped) window. A following S==0 no-op call must return this
        # unchanged (mirrors TOVAKVCache/H2OKVCache's equivalent handling).
        self._last_returned: tuple[mx.array, mx.array] | None = None

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
            ``(K_out, V_out)`` for THIS call's own attention — the full,
            un-evicted concatenation of whatever the sink+recent window held
            before this call plus the ``S`` new tokens (see #370 below), NOT
            capped at ``stream_n_sink + stream_window_size``. What gets
            *stored* afterward (the window state, visible to the next call)
            is capped as before.

        mlx_lm builds the attention mask for this call from hidden states —
        before q/k/v projections exist, let alone this cache's own
        ``update_and_fetch`` — so it is fixed (as either the "causal" string
        or an explicit array from ``make_mask``, called with only this
        call's query count ``N``) before eviction can possibly run. If this
        method shrank what it returns to fewer than ``N`` keys via window
        trimming, that already-fixed mask would silently desync from the
        shape it was built for (VeloxQuant-MLX#370). So trimming is
        deferred: this call returns the full pre-trim concatenation
        (matching the mask ``make_mask`` already built from the previous
        call's true kept positions — see that method), and only the stored
        window shrinks, for the *next* call's ``make_mask`` to reflect
        correctly.
        """
        B, H, S, D = keys.shape
        self._ensure_windows(B, H, D)

        if S == 0:
            if self._last_returned is not None:
                return self._last_returned
            return keys.astype(mx.float16), values.astype(mx.float16)

        # Byte accounting for this batch
        fp16_new = B * H * S * D * 2 * 2  # K + V, fp16
        self._full_seq_bytes += fp16_new
        self._tokens_seen_total += B * H * S

        # Update each head's window, capturing the OLD (pre-this-call)
        # window's K/V before stream_update overwrites self._windows[idx] —
        # this call's own RETURN is old_window ++ this call's raw incoming
        # tokens (#370), never the new (capped) window.
        k_out_b, v_out_b = [], []
        k_full_b, v_full_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            k_full_h, v_full_h = [], []
            for h in range(H):
                idx = self._window_idx(b, h)
                w_old = self._windows[idx]
                new_k_bh = keys[b, h].astype(mx.float16)
                new_v_bh = values[b, h].astype(mx.float16)
                if w_old.n_sink == 0 and w_old.n_recent == 0:
                    # Nothing stored yet for this head — stream_get_kv would
                    # return a degenerate (0,1)-shaped placeholder, not
                    # (0,D); skip the concat entirely (mirrors H2OKVCache/
                    # TOVAKVCache's `previous_k is None` bootstrap case).
                    k_full_h.append(new_k_bh)
                    v_full_h.append(new_v_bh)
                else:
                    k_old, v_old = stream_get_kv(w_old)
                    k_full_h.append(mx.concatenate([k_old, new_k_bh], axis=0))
                    v_full_h.append(mx.concatenate([v_old, new_v_bh], axis=0))

                w = stream_update(
                    w_old,
                    new_k_bh,
                    new_v_bh,
                    n_sink=self._n_sink,
                    window_size=self._window_size,
                )
                self._windows[idx] = w
                k_h, v_h = stream_get_kv(w)
                k_out_h.append(k_h)  # [n_keep, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_keep, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))
            k_full_b.append(mx.stack(k_full_h, axis=0))  # [H, n_full, D]
            v_full_b.append(mx.stack(v_full_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_keep, D] — for STORAGE
        V_out = mx.stack(v_out_b, axis=0)
        K_full = mx.stack(k_full_b, axis=0)  # [B, H, n_full, D] — this call's RETURN
        V_full = mx.stack(v_full_b, axis=0)

        # Recount kept bytes from first (B=0, H=0) head as representative
        kept_bytes = stream_fp16_bytes(self._windows[0]) * B * H
        self._stream_kept_bytes = kept_bytes  # snapshot (not cumulative; current state)

        # head-0 true kept positions per batch element, for the NEXT call's
        # make_mask (see that method) — purely structural, derived directly
        # from the new (post-trim) window's sink/recent boundaries, not
        # this call's own mask (already fixed by the time we get here).
        w0 = self._windows[self._window_idx(0, 0)]
        sink_pos = mx.arange(0, w0.n_sink, dtype=mx.int32)
        recent_pos = mx.arange(w0.tokens_seen - w0.n_recent, w0.tokens_seen, dtype=mx.int32)
        kept_pos_1d = mx.concatenate([sink_pos, recent_pos], axis=0)
        self._kept_positions = mx.broadcast_to(kept_pos_1d[None, :], (B, kept_pos_1d.shape[0]))

        # K_out/V_out is the full sink+recent window every call, not a
        # delta — reset so the base class's append-only buffer starts fresh
        # instead of stacking on top of the previous call's rows. Without
        # this, self.keys/self.values/self.offset stay at __init__ defaults
        # forever, and mlx_lm's generate() crashes on `cache.state` during
        # chunked prefill (see #83).
        self.keys = None
        self.values = None
        self.offset = 0
        super().update_and_fetch(K_out, V_out)
        out = (K_full, V_full)
        self._last_returned = out

        # RoPE position correctness (see #171).
        #
        # mlx_lm rotates BOTH the query and the incoming key at
        # ``offset=cache.offset`` *before* calling update_and_fetch. The base
        # class above just set ``self.offset`` to the number of RETAINED rows,
        # so once the window saturates (n_keep pinned at n_sink + window_size)
        # the offset stops advancing and every subsequent token is rotated at
        # ~n_keep while its true position keeps climbing — an unbounded drift
        # that scrambles attention.
        #
        # StreamingLLM PRESERVES the original position of every surviving
        # token (the window drops rows but never renumbers them), so all
        # stored keys already carry rotations for their true absolute
        # positions. RoPE is relative — <rope(q,i), rope(k,j)> depends only on
        # i-j — so reporting the true position here puts queries, new keys,
        # and survivors back on one consistent absolute axis, and no
        # re-rotation of survivors is needed.
        #
        # NOTE: this is the *faithful-to-this-implementation* choice, not the
        # StreamingLLM paper's. The paper assigns positions by position
        # *within the cache* rather than in the original sequence; doing that
        # here would require re-rotating every survivor each step. See #171.
        self._true_offset += S
        self.offset = self._true_offset
        return out

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Explicit position-based causal mask — see VeloxQuant-MLX#370.

        Called BEFORE this step's own ``update_and_fetch`` (and thus before
        this step's own window trim, which ``update_and_fetch`` defers past
        this step's return anyway — see its docstring). ``self._kept_positions``
        holds the true positions of whatever the sink+recent window already
        holds from the *previous* call, which is exactly what
        ``update_and_fetch`` will concatenate its ``N`` new tokens onto — so
        a mask sized ``[B, 1, N, len(kept) + N]`` covers this call's actual
        returned key count precisely.
        """
        if self._kept_positions is None:
            return super().make_mask(N, return_array=return_array, window_size=window_size)
        B = self._kept_positions.shape[0]
        prev_positions = self._kept_positions
        new_positions = mx.arange(self.offset, self.offset + N, dtype=mx.int32)
        new_positions = mx.broadcast_to(new_positions[None, :], (B, N))
        key_positions = mx.concatenate([prev_positions, new_positions], axis=1)
        query_positions = new_positions
        return eviction_make_mask(
            query_positions, key_positions, N, return_array=return_array, window_size=window_size
        )

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

    @property
    def tokens_kept(self) -> int:
        """Alias for :attr:`tokens_in_window`, matching the ``tokens_kept``
        name every other eviction cache uses (h2o, tova, pyramidkv, snapkv,
        squeeze, ...) for "tokens currently in the (B=0, H=0) head's cache".
        Telemetry code (e.g. the ``/v1/kv/stats`` token-count aggregator)
        probes for ``tokens_kept`` specifically via ``hasattr``/``getattr``;
        without this alias it silently reports 0 retained tokens for
        streaming_llm regardless of actual eviction state, since
        ``tokens_seen`` alone is already enough to pass its "has telemetry"
        check. Found verifying VeloxQuant-Studio issue #29.
        """
        return self.tokens_in_window

    # ------------------------------------------------------------------
    # Batching guard (see VeloxQuant-MLX#358)
    # ------------------------------------------------------------------
    # ``mlx_lm.server``'s ``BatchGenerator`` calls ``_merge_caches`` on every
    # ``PromptProcessingBatch`` it builds -- including the very first, single-
    # sequence one -- whenever ``hasattr(cache, "merge")`` is ``True`` on a
    # fresh per-layer probe. The inherited ``KVCache.merge()`` classmethod
    # delegates to ``BatchKVCache.merge()``, which for a batch of brand-new
    # (empty) caches takes the "no cache has content" fast path and silently
    # returns a plain empty ``BatchKVCache`` in place of ``StreamingLLMKVCache``
    # -- no sink, no sliding window, no per-token eviction, plain fp16 growth,
    # while the server still believes it is running ``streaming_llm`` (the
    # same silent-substitution pattern as the other #358 occurrences). A bare
    # method override is insufficient since ``hasattr()`` would still report
    # ``True`` for a classmethod defined on the class; the property must raise
    # on access instead so ``hasattr`` sees it as absent.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError("StreamingLLMKVCache does not support merge() — see VeloxQuant-MLX#358")
        )
    )


__all__ = ["StreamingLLMKVCache"]

"""ChunkKV-adapted KV cache — chunk-level (semantic-block) eviction.

Inspired by "ChunkKV: Semantic-Preserving KV Cache Compression for Efficient
Long-Context LLM Inference" (Liu et al., 2025, arXiv:2502.00299). Documented as
"ChunkKV-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

ChunkKV evicts KV at **chunk granularity**: the sequence is partitioned into
contiguous chunks of ``chunk_size`` tokens, and a chunk is kept or dropped as a
whole. Where H2O/TOVA/SnapKV keep the top *tokens* by importance — which shreds
contiguous spans whose meaning is collective — ChunkKV keeps the top *chunks*, so
surviving context stays locally coherent. Chunk importance is a pooled proxy over
an existing per-token signal: cumulative attention mass (``score="attn_mass"``,
the H2O scorer) or mean key L2 norm (``score="key_norm"``).

By default every layer/head resolves its own chunks independently, so the
standard ``KVCacheBuilder.for_model`` path (one ``ChunkKVCache`` per layer) is
all it needs. When ``chunk_size == 1`` the method reduces bit-for-bit to
H2O-adapted (each chunk is one token).

Optionally, setting ``chunkkv_reuse_layers > 1`` enables the paper's Algorithm 2
(layer-wise index reuse): layers are grouped into consecutive blocks of that
size, only the first ("leader") layer per block runs eviction, and the rest
("follower" layers) reuse the leader's exact kept-token positions via a shared
``ChunkKVIndexReuseCoordinator`` (injected at ``for_model`` build time, mirroring
``SqueezeCoordinator``'s pattern but publishing every step rather than once at
prefill, since eviction can recur at any step).

This is the seventh distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted     : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted        : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted       : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted  : H2O scoring with a fixed per-layer pyramid budget.
  - SqueezeAttention-adapted : H2O scoring with a data-driven per-layer budget.
  - ChunkKV-adapted    : H2O/key-norm scoring, evicted at CHUNK granularity
    (whole contiguous blocks) instead of per token.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted / SnapKV-adapted).
  - Pooled per-token score as a proxy for the paper's attention-over-chunk
    importance (mean-pooled, same chunk-granular decision).
  - Streaming eviction (drop a chunk once the cache exceeds budget by a chunk)
    rather than a single one-shot prefill compression.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Byte accounting:
    chunkkv_kept_bytes — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / chunkkv_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
    chunk_size         — this cache's eviction granularity (diagnostic)
    is_index_reuse_leader — whether this layer evicts (leader) or reuses (follower)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.cache.chunkkv_coordinator import ChunkKVIndexReuseCoordinator
from veloxquant_mlx.quantizers.chunkkv import (
    chunkkv_apply_reuse_indices_batched,
    chunkkv_trim_batched,
    chunkkv_update_batched,
)


class ChunkKVCache(_MLXKVCache):
    """KV cache implementing ChunkKV-adapted chunk-level eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``chunkkv_budget`` (int, default 512)     — max tokens kept (sinks incl.).
            ``chunkkv_chunk_size`` (int, default 8)   — eviction granularity ``C``;
                ``1`` reduces to H2O-adapted exactly.
            ``chunkkv_n_sink`` (int, default 4)       — leading positions never evicted.
            ``chunkkv_score`` (str, default "attn_mass") — "attn_mass" | "key_norm".
        layer_id: This layer's index (used to resolve leader/follower role and
            report/query the coordinator). ``None`` for single-cache
            construction — the layer then always runs its own eviction (as if
            it were its own leader).
        coordinator: Shared :class:`ChunkKVIndexReuseCoordinator`, or ``None``.
            When present and ``layer_id`` is not the block's leader, this layer
            reuses the leader's kept-token positions instead of evicting itself.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        eviction loop. Per-head state is lazily initialised on the first call.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(
        self,
        config: Any,
        layer_id: int | None = None,
        coordinator: ChunkKVIndexReuseCoordinator | None = None,
    ) -> None:
        super().__init__()
        self._budget = int(getattr(config, "chunkkv_budget", 512))
        self._chunk_size = int(getattr(config, "chunkkv_chunk_size", 8))
        self._n_sink = int(getattr(config, "chunkkv_n_sink", 4))
        self._score_mode = str(getattr(config, "chunkkv_score", "attn_mass"))

        self._layer_id = layer_id
        self._coordinator = coordinator
        self._is_leader = coordinator is None or layer_id is None or coordinator.is_leader(layer_id)

        self._head_dim: int = 0
        self._B: int = 0
        self._H: int = 0
        self._initialised: bool = False

        # Flat, padded [BH, n_max, D] / [BH, n_max] state -- replaces the old
        # per-(b,h) ChunkKVState list. Padding columns at/beyond
        # self._lengths[i] are unused garbage. Batching every head into one
        # call (instead of a Python loop calling chunkkv_update /
        # chunkkv_apply_reuse_indices once per (b,h) pair) removes the
        # O(B*H) Python-dispatch bottleneck (issue #564), following the same
        # template as H2OKVCache/#504 -- generalized to a padded/ragged-
        # lengths representation since chunk eviction can remove more (or
        # fewer) than one row per step, unlike H2O's always-exactly-one.
        self._bh_keys: mx.array | None = None
        self._bh_values: mx.array | None = None
        self._bh_scores: mx.array | None = None  # None for follower layers
        self._bh_positions: mx.array | None = None
        self._bh_lengths: mx.array | None = None

        self._chunkkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # True absolute token position, independent of how many rows survive
        # eviction. Reported as ``self.offset`` so mlx_lm's RoPE stays correct
        # after tokens are dropped (see #171-style handling elsewhere).
        self._true_offset: int = 0

        # [B, n_kept] int32 true absolute position of each currently-stored
        # (head 0) row — see make_mask() and update_and_fetch()'s #370
        # deferred-eviction docstrings. None before the first update.
        self._kept_positions: mx.array | None = None

        # (K_out, V_out) actually returned by the last call — since #370's
        # deferred eviction, generally NOT the same as this call's full
        # (capped, chunk-aligned) retained state. A following S==0 no-op
        # call must return this unchanged.
        self._last_returned: tuple[mx.array, mx.array] | None = None

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily record shape on first call."""
        if not self._initialised:
            self._B = B
            self._H = H
            self._head_dim = D
            self._initialised = True

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply ChunkKV eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` for THIS call's own attention — the full,
            un-evicted concatenation of whatever was stored before this call
            plus the ``S`` new tokens (see #370 below), NOT capped at
            ``chunkkv_budget`` or chunk-aligned. What gets *stored*
            afterward (visible to the next call) is capped/chunk-aligned as
            before.

        mlx_lm builds the attention mask for this call from hidden states —
        before q/k/v projections exist, let alone this cache's own
        ``update_and_fetch`` — so it is fixed (as either the "causal" string
        or an explicit array from ``make_mask``, called with only this
        call's query count ``N``) before eviction can possibly run. If this
        method shrank what it returns to fewer than ``N`` keys via chunk
        eviction, that already-fixed mask would silently desync from the
        shape it was built for (VeloxQuant-MLX#370). So eviction is
        deferred: this call returns the full pre-eviction concatenation
        (matching the mask ``make_mask`` already built from the previous
        call's true kept positions — see that method), and only the stored
        per-head states shrink, for the *next* call's ``make_mask`` to
        reflect correctly.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        # Cast the whole [B, H, S, D] tensor once up front (a no-op when
        # already fp16, which is the common case on this MLX/Metal target)
        # instead of re-issuing `.astype(mx.float16)` per (b, h) slice below
        # — avoids B*H redundant cast ops per call for input that's already
        # the target dtype.
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)

        if S == 0:
            if self._last_returned is not None:
                return self._last_returned
            return keys, values

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        new_positions_1d = mx.arange(self._true_offset, self._true_offset + S, dtype=mx.int32)
        new_positions = mx.broadcast_to(new_positions_1d[None, :], (B * H, S))

        # Capture the pre-update stored K/V (for THIS call's own deferred
        # return) BEFORE running eviction/reuse — see #370. Uniform n_prev
        # across every (b,h) row is guaranteed: the previous call's trim
        # (below) always re-aligns every row to the same common length
        # before returning, so this stays a plain rectangular concat.
        prev_keys_flat = (
            None if self._bh_keys is None else self._bh_keys[:, : int(self._bh_lengths[0].item())]
        )
        prev_values_flat = (
            None
            if self._bh_values is None
            else self._bh_values[:, : int(self._bh_lengths[0].item())]
        )

        new_keys_flat = keys.reshape(B * H, S, D)
        new_values_flat = values.reshape(B * H, S, D)

        # 1) Update every head's state — leaders evict and publish their kept
        #    positions; followers reuse their leader's positions (Algorithm 2).
        needs_publish = self._coordinator is not None and self._layer_id is not None
        if self._is_leader:
            update_result = chunkkv_update_batched(
                self._bh_keys,
                self._bh_values,
                self._bh_scores,
                self._bh_positions,
                self._bh_lengths,
                new_keys_flat,
                new_values_flat,
                new_positions,
                self._n_sink,
                self._budget,
                self._chunk_size,
                self._score_mode,
                record_kept_positions=needs_publish,
            )
            if needs_publish:
                (
                    self._bh_keys,
                    self._bh_values,
                    self._bh_scores,
                    self._bh_positions,
                    self._bh_lengths,
                    kept_positions_per_row,
                ) = update_result
                # Coordinator's publish/fetch contract is per-(layer, head),
                # not per-(layer, batch, head) -- publish row b=0's decisions
                # per head. Every batch row of a given head makes the
                # identical eviction decision (shared config/budget/chunking,
                # and every row started this call at the identical trimmed
                # length -- see __init__'s uniform-length invariant note),
                # so row 0 is representative, matching what a per-head-only
                # (no batch axis) coordinator API already assumes.
                for h in range(H):
                    self._coordinator.publish(self._layer_id, h, kept_positions_per_row[h])
            else:
                (
                    self._bh_keys,
                    self._bh_values,
                    self._bh_scores,
                    self._bh_positions,
                    self._bh_lengths,
                ) = update_result
        else:
            # _is_leader is False only when coordinator/layer_id are both
            # non-None (see the `_is_leader =` assignment in __init__).
            assert self._coordinator is not None and self._layer_id is not None
            head_lists: list[list[list[int]]] = []
            for h in range(H):
                fetched = self._coordinator.fetch(self._layer_id, h)
                if fetched is None:
                    raise RuntimeError(
                        f"ChunkKVCache: follower layer {self._layer_id} fetched no "
                        f"published indices for head {h} — its leader layer must be "
                        f"updated first within the same step (see "
                        f"ChunkKVIndexReuseCoordinator.fetch's docstring)."
                    )
                head_lists.append(fetched)
            # Broadcast each head's shared list across every batch row, in
            # true (b,h) flat order (b*H+h) -- same head list reused for
            # every batch element, since the coordinator publishes per-head
            # only (see the leader branch's publish loop above).
            kept_positions_per_row = [head_lists[h] for _b in range(B) for h in range(H)]

            (
                self._bh_keys,
                self._bh_values,
                self._bh_positions,
                self._bh_lengths,
            ) = chunkkv_apply_reuse_indices_batched(
                self._bh_keys,
                self._bh_values,
                self._bh_positions,
                self._bh_lengths,
                new_keys_flat,
                new_values_flat,
                new_positions,
                kept_positions_per_row,
            )
            self._bh_scores = None

        # 2) Whole-chunk retention lets heads keep slightly different token counts;
        #    the MLX attention path needs a rectangular [B, H, n_kept, D] output,
        #    so align every head to the common minimum kept-length by dropping each
        #    head's oldest non-sink tokens down to that length. When chunk_size=1
        #    all heads already hold exactly `budget`, so no trimming occurs and the
        #    H2O equivalence is preserved. Follower heads already mirror their
        #    leader's kept-length exactly, so this is a no-op for them in practice.
        min_kept = int(mx.min(self._bh_lengths).item())
        (
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_positions,
            self._bh_lengths,
        ) = chunkkv_trim_batched(
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_positions,
            self._bh_lengths,
            self._n_sink,
            min_kept,
        )

        K_out = self._bh_keys[:, :min_kept].reshape(
            B, H, min_kept, D
        )  # [B, H, min_kept, D] — STORAGE
        V_out = self._bh_values[:, :min_kept].reshape(B, H, min_kept, D)

        if prev_keys_flat is None:
            K_full = new_keys_flat.reshape(B, H, S, D)
            V_full = new_values_flat.reshape(B, H, S, D)
        else:
            n_full = prev_keys_flat.shape[1] + S
            K_full = mx.concatenate([prev_keys_flat, new_keys_flat], axis=1).reshape(
                B, H, n_full, D
            )
            V_full = mx.concatenate([prev_values_flat, new_values_flat], axis=1).reshape(
                B, H, n_full, D
            )

        # Byte accounting: bytes currently retained across all (b,h) rows.
        self._chunkkv_kept_bytes = B * H * min_kept * D * 2 * 2

        # head-0 true kept positions per batch element, for the NEXT call's
        # make_mask (see that method) — not this call's own mask, already
        # fixed by the time we get here.
        self._kept_positions = self._bh_positions[:, :min_kept].reshape(B, H, min_kept)[:, 0, :]

        # K_out/V_out is the full retained state every call, not a delta —
        # reset so the base class's append-only buffer starts fresh instead
        # of stacking on top of the previous call's rows. Without this,
        # self.keys/self.values/self.offset stay at __init__ defaults
        # forever, and mlx_lm's generate() crashes on `cache.state` during
        # chunked prefill (see #83).
        self.keys = None
        self.values = None
        self.offset = 0
        super().update_and_fetch(K_out, V_out)
        self._true_offset += S
        self.offset = self._true_offset
        out = (K_full, V_full)
        self._last_returned = out
        return out

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Explicit position-based causal mask — see VeloxQuant-MLX#370.

        Called BEFORE this step's own ``update_and_fetch`` (and thus before
        this step's own eviction, which ``update_and_fetch`` defers past
        this step's return anyway — see its docstring). ``self._kept_positions``
        holds the true positions of whatever every head's state already
        stores from the *previous* call, which is exactly what
        ``update_and_fetch`` will concatenate its ``N`` new tokens onto —
        so a mask sized ``[B, 1, N, len(kept) + N]`` covers this call's
        actual returned key count precisely.
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
    def layer_budget(self) -> int:
        """This cache's per-layer token budget."""
        return self._budget

    @property
    def chunk_size(self) -> int:
        """This cache's eviction granularity ``C`` (diagnostic)."""
        return self._chunk_size

    @property
    def is_index_reuse_leader(self) -> bool:
        """True if this layer runs its own eviction (leader, or reuse disabled)."""
        return self._is_leader

    @property
    def chunkkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._chunkkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / chunkkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._chunkkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._chunkkv_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if self._bh_lengths is None:
            return 0
        return int(self._bh_lengths[0].item())


__all__ = ["ChunkKVCache"]

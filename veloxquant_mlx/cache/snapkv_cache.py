"""SnapKV-adapted KV cache — prefill observation-window token eviction.

Inspired by "SnapKV: LLM Knows What You are Looking for Before Generation"
(Yuan et al., ICLR 2025, arXiv:2404.14469). Documented as "SnapKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

Token eviction: during prefill (S > 1), the last ``snap_obs_window`` key rows
act as proxy queries; their softmax attention over all prefix tokens produces
a per-token importance score. Only the top-``snap_budget`` tokens (plus the
first ``snap_n_sink`` sink positions) are retained as fp16. All subsequent
decode tokens (S == 1) are always appended — never evicted.

mlx_lm's chunked prefill calls ``update_and_fetch`` once per
``prefill_step_size`` chunk for prompts longer than one chunk. Every S > 1
call re-enforces the budget against the *accumulated* kept set (prior kept
tokens concatenated with the new chunk), not just that chunk in isolation —
otherwise the retained count would grow by up to ``snap_budget`` per chunk
instead of staying capped (see #84).

This is the repo's first **eviction** method. Every other method compresses
all tokens to fewer bits; SnapKV-adapted stores fewer tokens at full fp16
precision. The two axes compose: wrap a quantizer cache around the selected
subset for combined eviction + compression.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: obs-window uses key vectors, not true prompt query
    vectors (not visible at ``update_and_fetch``).
  - No max-pool smoothing (paper's ``kernel_size > 1`` not implemented).
  - Uniform ``snap_budget`` across all heads.

Byte accounting:
    evicted_key_bytes / evicted_value_bytes  — fp16 bytes for the kept subset
    full_key_bytes    / full_value_bytes     — hypothetical cost without eviction
    eviction_ratio                           — full_fp16 / kept_fp16 (> 1 = savings)
    tokens_kept / tokens_total              — diagnostic token counters
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.snapkv import (
    _snapkv_compress_batched,
    snapkv_compress,
)


class SnapKVKVCache(_MLXKVCache):
    """KV cache implementing SnapKV-adapted prefill eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``snap_budget``     (int, default 512)  — max tokens retained after prefill,
            ``snap_obs_window`` (int, default 32)   — trailing keys used as proxy queries,
            ``snap_n_sink``     (int, default 4)    — initial positions always kept.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Single-layer (no coordinator); ``for_model`` propagates all ``snap_*``
        fields automatically via ``dataclasses.replace``.
        Eviction happens at every prefill chunk (S > 1) — mlx_lm's chunked
        prefill (``prefill_step_size``) calls ``update_and_fetch`` once per
        chunk for prompts longer than one chunk, so budget enforcement must
        re-run over the accumulated kept set, not just each chunk in
        isolation (see #84). Decode tokens (S == 1) are always kept.
        Accumulated decode tokens grow the cache beyond the initial budget in
        the decode phase — consistent with the paper's design (the budget
        constrains prefill history, not the decode stream).
    """

    # Class-level defaults so the ``offset`` property below is safe to read
    # and write during ``super().__init__()``, before instance attributes
    # have been created.
    _in_base: bool = False
    _row_offset: int = 0
    _true_offset: int = 0

    @property
    def _storage_dtype(self) -> mx.Dtype | None:
        if self._storage_dtype_name is None:
            return None
        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._backend = getattr(config, "snap_backend", "auto")
        if self._backend not in ("auto", "mlx", "metal", "reference"):
            raise QuantizerConfigError(f"Unsupported SnapKV backend: {self._backend}")
        self._dtype_policy = getattr(config, "snap_dtype", "auto")
        if self._dtype_policy not in ("auto", "float16"):
            raise QuantizerConfigError(f"Unsupported SnapKV dtype policy: {self._dtype_policy}")
        # Stored as a name, not the mlx.core.Dtype itself: mlx_lm.server
        # deepcopies cache entries per request, and mx.core.Dtype objects
        # (mx.float16, mx.bfloat16, ...) raise TypeError from copy.deepcopy
        # ("cannot pickle 'mlx.core.Dtype' object").
        self._storage_dtype_name: str | None = None
        self._batched_scoring = getattr(config, "snap_batched_scoring", False)
        self._budget = int(getattr(config, "snap_budget", 512))
        self._obs_window = int(getattr(config, "snap_obs_window", 32))
        self._n_sink = int(getattr(config, "snap_n_sink", 4))

        self._evicted_key_bytes = 0
        self._evicted_value_bytes = 0
        self._full_key_bytes = 0
        self._full_value_bytes = 0
        self._tokens_kept = 0
        self._tokens_total = 0

        # True once the first S>1 (prefill) call has been processed. Any
        # later S>1 call is a subsequent chunk of the *same* prompt (mlx_lm
        # chunked prefill), not a fresh prefill, and must re-enforce the
        # budget against the accumulated kept set rather than compressing
        # the new chunk in isolation and appending.
        self._prefill_done = False

        # True absolute token position, independent of how many rows survive
        # eviction. Surfaced through the ``offset`` property so mlx_lm's RoPE
        # stays correct after tokens are dropped (see #171).
        self._true_offset: int = 0

        # [B, n_kept] int32 true absolute position of each currently-stored
        # row, head 0 only (mlx_lm's mask contract can't vary by attention
        # head — see _eviction_mask.py's module docstring). Used by
        # make_mask() to build an explicit causal mask instead of relying on
        # the inherited "causal" string shortcut, which silently mis-attends
        # once eviction has made the kept set non-contiguous (#370).
        self._kept_positions: mx.array | None = None

        # Set by _process_prefill on this cache's first eviction-triggering
        # call to (evicted_k, evicted_v) — applied to self.keys/self.values
        # AFTER super().update_and_fetch() has already returned the full,
        # un-evicted set for this call's own (already-mask-fixed) attention.
        # See _process_prefill's docstring for why eviction must be deferred
        # past this call's own attention rather than applied before it.
        self._post_step_evicted: tuple[mx.array, mx.array] | None = None

    # ------------------------------------------------------------------
    # ``offset`` carries TWO meanings that diverge as soon as eviction drops a
    # row, and #171 was caused by conflating them:
    #
    #   1. mlx_lm's attention layer reads ``cache.offset`` as the POSITION to
    #      rotate the query and incoming key at, before update_and_fetch runs.
    #   2. The base ``KVCache.update_and_fetch`` uses it as the write cursor
    #      and return slice (``self.keys[..., :self.offset, :]``).
    #
    # SnapKV's prefill compression drops tokens, so the retained row count
    # falls permanently behind the true position — under (2)'s value every
    # later token gets rotated at a position short by exactly the number
    # evicted. Returning the true position from (2) instead would make the
    # base class slice past the end of what it stored.
    #
    # So the two are separated: ``_row_offset`` backs the base class's
    # bookkeeping, while reads from outside get the true position. This is
    # sound because SnapKV PRESERVES original positions — snap_select_indices
    # returns kept indices sorted ascending and never renumbers them — so
    # stored keys already carry rotations for their true positions, and RoPE
    # is relative (<rope(q,i), rope(k,j)> depends only on i-j), so no
    # re-rotation of survivors is needed.
    # The base class does ``prev = self.offset`` ... ``self.offset += S``, so
    # during that call ``offset`` must read and write ROW counts. Outside it,
    # mlx_lm must see the true position. ``_in_base`` flips between the two.
    @property
    def offset(self) -> int:
        """True absolute token position (NOT the retained row count).

        While the base class's ``update_and_fetch`` is on the stack this
        yields the retained row count instead, so its cursor arithmetic and
        return slice stay correct.
        """
        return self._row_offset if self._in_base else self._true_offset

    @offset.setter
    def offset(self, value: int) -> None:
        """Restore the retained row count (base-class bookkeeping only; see the getter for the true/retained distinction)."""
        self._row_offset = value

    # ------------------------------------------------------------------
    def _evict_head(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array, int]:
        """Evict ``[S, D]`` K/V for one head → ``([n_kept, D], [n_kept, D], n_kept)``."""
        state = snapkv_compress(
            keys,
            values,
            budget=self._budget,
            obs_window=self._obs_window,
            n_sink=self._n_sink,
            backend=self._backend,
        )
        return state.kept_keys, state.kept_values, state.n_kept

    def _process_prefill(self, keys: mx.array, values: mx.array):
        """Compute this call's eviction, but do NOT apply it to what's
        returned for THIS step's own attention.

        mlx_lm builds the attention mask from hidden states before q/k/v
        projections run, so it is fixed (``mask="causal"``, sized for this
        call's own N queries) before ``update_and_fetch`` is ever invoked —
        this cache cannot make that mask reflect an eviction it hasn't
        performed yet. Shrinking the returned keys here would leave that
        already-fixed mask silently wrong for this call specifically (see
        VeloxQuant-MLX#370: the mask assumes a contiguous trailing window,
        false once eviction drops non-trailing rows).

        So the eviction decision is computed here (for byte accounting and
        to seed ``self._true_kept_k/v`` — what gets *stored* for future
        calls) but the K/V actually returned for this step's attention are
        the full, un-evicted set, for which the pre-built "causal" mask
        (this call's query count == this call's key count) is exactly
        correct. ``update_and_fetch`` compresses ``self.keys``/``self.values``
        down to the evicted subset immediately afterward, so every
        subsequent call sees the compact stored cache and a correct
        ``make_mask`` (see ``_kept_positions``).
        """
        B, H, S, D = keys.shape
        k, v, indices = _snapkv_compress_batched(
            keys,
            values,
            self._budget,
            self._obs_window,
            self._n_sink,
            backend=self._backend,
            output_dtype=self._storage_dtype,
            batched_scoring=self._batched_scoring,
            return_indices=True,
        )
        kept = k.shape[2]
        self._evicted_key_bytes += B * H * kept * D * 2
        self._evicted_value_bytes += B * H * kept * D * 2
        self._full_key_bytes += B * H * S * D * 2
        self._full_value_bytes += B * H * S * D * 2
        self._tokens_kept += B * H * kept
        self._tokens_total += B * H * S
        # indices are positions within this call's own [S] frame, which
        # starts at the true absolute offset seen so far (0, for the first
        # prefill call — later chunks recompute from scratch in
        # _process_prefill_chunk, not here).
        self._kept_positions = indices[:, 0, :] + self._true_offset
        self._post_step_evicted = (k, v)
        return keys.astype(self._storage_dtype), values.astype(self._storage_dtype)

    def _process_decode(self, keys: mx.array, values: mx.array):
        """Pass through decode tokens (S == 1) — never evicted."""
        B, H, S, D = keys.shape
        fp16_cost = B * H * S * D * 2
        self._evicted_key_bytes += fp16_cost
        self._evicted_value_bytes += fp16_cost
        self._full_key_bytes += fp16_cost
        self._full_value_bytes += fp16_cost
        self._tokens_kept += B * H * S
        self._tokens_total += B * H * S
        new_pos = mx.arange(self._true_offset, self._true_offset + S, dtype=mx.int32)
        new_pos = mx.broadcast_to(new_pos[None, :], (B, S))
        self._kept_positions = (
            new_pos
            if self._kept_positions is None
            else mx.concatenate([self._kept_positions, new_pos], axis=1)
        )
        return keys.astype(self._storage_dtype), values.astype(self._storage_dtype)

    def _process_prefill_chunk(self, keys: mx.array, values: mx.array):
        """Re-enforce the budget for a later chunk of the same prefill.

        Concatenates each head's already-kept tokens (``self.keys`` /
        ``self.values`` from prior chunks, sink-anchored at true position 0)
        with the new chunk, then re-runs ``snapkv_compress`` over that
        concatenation so the total retained count stays capped at
        ``self._budget`` instead of growing by up to ``budget`` per chunk.

        Byte/token accounting: this recomputed kept set *replaces* what was
        previously counted as kept for this (b, h) (the prior chunk's kept
        rows are being re-selected from, not kept in addition to), so the
        prior per-(b, h) kept contribution is subtracted before adding the
        new one. ``tokens_total`` only grows by this chunk's ``S`` (prior
        chunks' totals were already counted when first seen).
        """
        B, H, S, D = keys.shape
        # Retained row count — NOT self.offset, which since #171 reports the
        # true absolute token position for RoPE and diverges from the row
        # count as soon as eviction drops anything. Not self.keys.shape[2]
        # either: that is the base class's over-allocated buffer size.
        prev_kept = self._row_offset
        cat_k = mx.concatenate([self.keys[:, :, :prev_kept], keys], axis=2)
        cat_v = mx.concatenate([self.values[:, :, :prev_kept], values], axis=2)
        k_out, v_out, indices = _snapkv_compress_batched(
            cat_k,
            cat_v,
            self._budget,
            self._obs_window,
            self._n_sink,
            backend=self._backend,
            output_dtype=self._storage_dtype,
            batched_scoring=self._batched_scoring,
            return_indices=True,
        )
        # indices select from the concatenation [prior kept rows (true
        # positions in self._kept_positions) ++ this chunk's S new rows
        # (true positions true_offset..true_offset+S)] — build that same
        # concatenated position frame (head 0, per batch) and gather.
        assert self._kept_positions is not None  # prefill_done implies this
        new_pos = mx.arange(self._true_offset, self._true_offset + S, dtype=mx.int32)
        new_pos = mx.broadcast_to(new_pos[None, :], (B, S))
        cat_pos = mx.concatenate([self._kept_positions, new_pos], axis=1)
        self._kept_positions = mx.take_along_axis(cat_pos, indices[:, 0, :], axis=1)
        delta = B * H * (k_out.shape[2] - prev_kept)
        self._evicted_key_bytes += delta * D * 2
        self._evicted_value_bytes += delta * D * 2
        self._tokens_kept += delta

        self._full_key_bytes += B * H * S * D * 2
        self._full_value_bytes += B * H * S * D * 2
        self._tokens_total += B * H * S

        # The recomputed kept set replaces (not appends to) what's stored:
        # reset so the base class's append-only update_and_fetch starts a
        # fresh buffer instead of stacking this chunk's output on top of
        # the pre-recompression rows still sitting in self.keys/values.
        self.offset = 0
        self.keys = None
        self.values = None
        return k_out, v_out

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Prefill: score and evict down to the retention budget (SnapKV window-attention proxy), deferred to after this step's own attention. Decode: pass through unevicted."""
        if self._storage_dtype_name is None:
            self._storage_dtype_name = (
                "bfloat16"
                if self._dtype_policy == "auto" and keys.dtype == values.dtype == mx.bfloat16
                else "float16"
            )
        is_prefill = keys.shape[2] > 1
        if is_prefill:
            if not self._prefill_done:
                k_out, v_out = self._process_prefill(keys, values)
                self._prefill_done = True
            else:
                k_out, v_out = self._process_prefill_chunk(keys, values)
        else:
            k_out, v_out = self._process_decode(keys, values)
        # Track the true absolute position separately from the retained row
        # count the base class maintains in ``_row_offset``. See the ``offset``
        # property for why the two must not be conflated.
        self._true_offset += keys.shape[2]
        self._in_base = True
        try:
            result = super().update_and_fetch(k_out, v_out)
        finally:
            self._in_base = False
        # Deferred eviction (see _process_prefill): this call's own attention
        # already got the correct (full, un-evicted) `result` above, matching
        # the mask mlx_lm had already fixed before this call ran. Now shrink
        # what's *stored* so future calls' make_mask sees the compact kept
        # set and future update_and_fetch calls append onto it, not the full
        # un-evicted history.
        if self._post_step_evicted is not None:
            k_evicted, v_evicted = self._post_step_evicted
            self._post_step_evicted = None
            self.keys = k_evicted
            self.values = v_evicted
            self._row_offset = k_evicted.shape[2]
        return result

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Explicit position-based causal mask — see VeloxQuant-MLX#370.

        The inherited ``KVCache.make_mask`` returns the string ``"causal"``
        whenever ``offset == 0`` (the very first prefill call, before any
        eviction has run) or otherwise builds a mask keyed off row index
        rather than true position — both wrong once eviction has made the
        kept keys a non-contiguous subset. Before the first prefill call
        (``_kept_positions`` is still ``None``, nothing evicted yet) this
        falls back to the base class's behavior exactly, since a plain
        trailing-window mask is correct there.
        """
        if self._kept_positions is None:
            return super().make_mask(N, return_array=return_array, window_size=window_size)
        query_positions = mx.arange(self._true_offset, self._true_offset + N, dtype=mx.int32)
        query_positions = mx.broadcast_to(
            query_positions[None, :], (self._kept_positions.shape[0], N)
        )
        return eviction_make_mask(
            query_positions,
            self._kept_positions,
            N,
            return_array=return_array,
            window_size=window_size,
        )

    # ------------------------------------------------------------------
    @property
    def evicted_key_bytes(self) -> int:
        """Bytes stored for kept key rows (fp16)."""
        return self._evicted_key_bytes

    @property
    def evicted_value_bytes(self) -> int:
        """Bytes stored for kept value rows (fp16)."""
        return self._evicted_value_bytes

    @property
    def full_key_bytes(self) -> int:
        """Hypothetical fp16 key cost without any eviction."""
        return self._full_key_bytes

    @property
    def full_value_bytes(self) -> int:
        """Hypothetical fp16 value cost without any eviction."""
        return self._full_value_bytes

    @property
    def tokens_kept(self) -> int:
        """Total token positions retained across all heads and steps."""
        return self._tokens_kept

    @property
    def tokens_total(self) -> int:
        """Total token positions seen (before eviction) across all heads and steps."""
        return self._tokens_total

    @property
    def eviction_ratio(self) -> float:
        """full_fp16_bytes / kept_fp16_bytes; > 1 means storage savings."""
        total_kept = self._evicted_key_bytes + self._evicted_value_bytes
        total_full = self._full_key_bytes + self._full_value_bytes
        if total_kept == 0:
            return 1.0
        return total_full / total_kept

    @property
    def keep_rate(self) -> float:
        """Fraction of tokens retained (tokens_kept / tokens_total)."""
        if self._tokens_total == 0:
            return 1.0
        return self._tokens_kept / self._tokens_total

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() reads a corrupted ``self.offset`` and both writes and
        returns the wrong thing, worse than a mere bookkeeping rollback.

        The base ``KVCache.trim(n)`` does ``n = min(self.offset, n); self.offset
        -= n``. Outside ``update_and_fetch`` (``_in_base`` is False), this
        class's ``offset`` property returns ``_true_offset`` — the absolute
        token position used for RoPE — not the retained row count
        (``_row_offset``) the base class's buffer actually holds. So ``n`` is
        clamped against the wrong (larger) number, and the setter then writes
        ``_row_offset -= n``, which can leave ``_row_offset`` **larger than
        the number of rows ever written** whenever eviction has dropped any
        tokens (the normal case once prefill has run). The next
        ``update_and_fetch`` then returns a slice reaching past the real
        data into the pre-allocated buffer's stale/uninitialized rows,
        silently feeding garbage into attention rather than crashing.
        Reproduced directly: budget=4, 20 real prefill tokens (so
        ``_row_offset`` == 4, ``_true_offset`` == 20); ``trim(3)`` leaves
        ``_row_offset`` == 17 though only 4 rows are real, and the following
        decode step returns an 18-row slice (see VeloxQuant-Studio issue #27).
        """
        return False

    # ------------------------------------------------------------------
    # Batching guard (see VeloxQuant-MLX#358)
    # ------------------------------------------------------------------
    # mlx_lm.server's batching path (BatchGenerator._make_batch ->
    # PromptProcessingBatch.__init__ -> _merge_caches) decides batchability
    # via hasattr(cache, "merge") and calls it even on a brand-new, empty
    # cache for the very first batch. Without this guard, SnapKVKVCache
    # inherits the base KVCache.merge classmethod, which hasattr() reports as
    # present; its "no cache has content" fast path then silently returns a
    # plain BatchKVCache instead of raising -- before the first token
    # generates. Substituting a BatchKVCache here would silently drop all
    # SnapKV eviction, byte accounting, and the true-offset RoPE correction
    # (see #171): every downstream token position and stored row would be
    # wrong with no error raised.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError("SnapKVKVCache does not support merge() — see VeloxQuant-MLX#358")
        )
    )


__all__ = ["SnapKVKVCache"]

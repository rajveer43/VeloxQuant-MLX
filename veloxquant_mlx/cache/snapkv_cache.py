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

from veloxquant_mlx.quantizers.snapkv import (
    full_fp16_bytes,
    snapkv_compress,
    snapkv_fp16_bytes,
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

    def __init__(self, config: Any) -> None:
        super().__init__()
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
        )
        return state.kept_keys, state.kept_values, state.n_kept

    def _process_prefill(self, keys: mx.array, values: mx.array):
        """Evict ``[B, H, S, D]`` prefill K/V per head; accumulate byte accounting."""
        B, H, S, D = keys.shape
        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                k_h, v_h, n_kept = self._evict_head(keys[b, h], values[b, h])
                k_out_h.append(k_h)
                v_out_h.append(v_h)
                # byte accounting: kept fp16 K and V rows
                self._evicted_key_bytes += n_kept * D * 2
                self._evicted_value_bytes += n_kept * D * 2
                self._full_key_bytes += S * D * 2
                self._full_value_bytes += S * D * 2
                self._tokens_kept += n_kept
                self._tokens_total += S
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))
        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

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
        return keys.astype(mx.float16), values.astype(mx.float16)

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
        prev_kept_bytes = prev_kept * D * 2  # per (b, h), K or V alone
        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                prior_k = self.keys[b, h, :prev_kept, :]
                prior_v = self.values[b, h, :prev_kept, :]
                cat_k = mx.concatenate([prior_k, keys[b, h]], axis=0)
                cat_v = mx.concatenate([prior_v, values[b, h]], axis=0)
                k_h, v_h, n_kept = self._evict_head(cat_k, cat_v)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
                kept_bytes = n_kept * D * 2
                self._evicted_key_bytes += kept_bytes - prev_kept_bytes
                self._evicted_value_bytes += kept_bytes - prev_kept_bytes
                self._tokens_kept += n_kept - prev_kept
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))
        k_out = mx.stack(k_out_b, axis=0)
        v_out = mx.stack(v_out_b, axis=0)

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
            return super().update_and_fetch(k_out, v_out)
        finally:
            self._in_base = False

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


__all__ = ["SnapKVKVCache"]

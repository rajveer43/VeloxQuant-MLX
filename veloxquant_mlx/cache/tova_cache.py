"""TOVA-adapted KV cache — current-step attention-weight token eviction.

Inspired by "Transformers are Multi-State RNNs" (Oren et al., 2024,
arXiv:2401.06104), whose TOVA (Token Omission Via Attention) policy keeps a
fixed-size cache by dropping, at each step, the single token receiving the
lowest attention weight in the *current* step. Documented as "TOVA-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

Memoryless eviction: on each incoming token the approximate current-step
attention distribution over the (post-append) cache is computed using the new
key vector as a proxy query (true query not visible at the cache wrapper level).
Whenever the cache exceeds ``tova_budget`` tokens, the lowest current-step-weight
non-sink token is permanently dropped. The cache never exceeds ``tova_budget``
positions.

This is the fourth distinct eviction axis in VeloxQuant-MLX:
  - SnapKV-adapted : score-based, fires once at prefill end only.
  - StreamingLLM-adapted : positional (recency + sink), fires every step.
  - H2O-adapted    : cumulative attention mass (inertial), fires every step.
  - TOVA-adapted   : current-step attention weight (memoryless), fires every step.

TOVA vs H2O — the key distinction:
  H2O carries a running sum of attention weights, so a token that was a heavy
  hitter in the past resists eviction (inertial). TOVA discards all history and
  scores by the present step only, so a token that stops being attended to is
  evicted even if it dominated earlier. TOVA is the more reactive policy.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: current-step attention weights are computed using the
    new key vector in place of the true query. Same approximation as
    SnapKV-adapted and H2O-adapted.
  - No RoPE position-ID *renumbering* after eviction. Surviving tokens keep
    their original absolute positions (``tova_update`` drops the evicted row
    and keeps the rest in temporal order), so ``self.offset`` reports the
    true token position and RoPE stays correct without re-rotating survivors
    (see ``update_and_fetch`` and :issue:`171`, :issue:`175`). Positions do
    become non-contiguous where tokens were dropped.
  - Uniform budget and n_sink across all heads.

Byte accounting:
    tova_kept_bytes   — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / tova_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.quantizers.tova import (
    TovaState,
    _resolve_backend,
    _tova_update_batched,
    init_tova_state,
)


class TOVAKVCache(_MLXKVCache):
    """KV cache implementing TOVA-adapted current-step attention eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``tova_budget`` (int, default 512) — maximum tokens retained at any time,
            ``tova_n_sink`` (int, default 4)   — leading positions never evicted.
            ``tova_backend`` (str, default "auto") — auto, mlx, metal, or reference.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        TOVA update loop — unlike SnapKV-adapted, there is no prefill-only phase.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()`` propagates
        all ``tova_*`` fields automatically via ``dataclasses.replace``.
        The per-head state is lazily initialised on the first call to
        ``update_and_fetch`` when shapes are known.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "tova_budget", 512))
        self._n_sink = int(getattr(config, "tova_n_sink", 4))
        self._backend = getattr(config, "tova_backend", "auto")
        _resolve_backend(self._backend)

        self._head_dim: int = 0
        self._states: list[TovaState] = []
        self._B: int = 0
        self._H: int = 0

        self._tova_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # True absolute token position, independent of how many rows survive
        # eviction. Reported as ``self.offset`` so mlx_lm's RoPE stays correct
        # after tokens are dropped (see #171 and update_and_fetch).
        self._true_offset: int = 0

        # [B, n_kept] int32 true absolute position of each currently-stored
        # (head 0) row — see make_mask() and update_and_fetch()'s #370
        # deferred-eviction docstrings. None before the first update.
        self._kept_positions: mx.array | None = None
        # Per-(b,h) flat [BH, n] true absolute positions, parallel to
        # self.keys/self.values — the full per-head bookkeeping backing
        # _kept_positions (which only surfaces head 0).
        self._bh_positions: mx.array | None = None
        # (K_out, V_out) actually returned by the last non-empty (S>0) call
        # — since #370's deferred eviction, that is generally NOT the same
        # object as self.keys/self.values (which may already be capped down
        # to budget by the time a later S==0 no-op call is made). A
        # following S==0 call must return exactly this, unchanged, to stay a
        # true no-op for mlx_lm callers that re-invoke update_and_fetch with
        # an empty chunk mid-generation.
        self._last_returned: tuple[mx.array, mx.array] | None = None

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head TovaState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [init_tova_state(self._n_sink, self._budget, D) for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply TOVA eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` for THIS call's own attention — the full,
            un-evicted concatenation of whatever was stored before this call
            plus the ``S`` new tokens (see #370 below), NOT capped at
            ``tova_budget``. What gets *stored* afterward (``self.keys`` /
            ``self.values``, visible to the next call) is capped at
            ``tova_budget`` as before.

        mlx_lm builds the attention mask for this call from hidden states —
        before q/k/v projections exist, let alone this cache's own
        ``update_and_fetch`` — so it is fixed (as either the "causal" string
        or an explicit array from ``make_mask``, called with only this
        call's query count ``N``) before eviction can possibly run. If this
        method shrank what it returns to fewer than ``N`` keys via eviction,
        that already-fixed mask would silently desync from the shape it was
        built for (VeloxQuant-MLX#370). So eviction is deferred: this call
        returns the full pre-eviction concatenation (matching the mask
        ``make_mask`` already built from the previous call's true kept
        positions — see that method), and only ``self.keys``/``self.values``
        (and ``self._bh_positions``/``self._kept_positions``) shrink, for
        the *next* call's ``make_mask`` to reflect correctly.
        """
        if keys.ndim != 4 or keys.shape != values.shape:
            raise ValueError("tova cache: K/V must have matching [B,H,S,D] shapes")
        B, H, S, D = keys.shape
        if min(B, H, D) < 1:
            raise ValueError("tova cache: batch/head/dimension must be positive")
        if self._states and (self._B, self._H, self._head_dim) != (B, H, D):
            raise ValueError("tova cache: batch/head/dimension cannot change after initialization")
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        if S == 0:
            if self._last_returned is not None:
                return self._last_returned
            if self.keys is None:
                return keys.astype(mx.float16), values.astype(mx.float16)
            return self.keys, self.values

        previous_k = None if self.keys is None else self.keys.reshape(B * H, -1, D)
        previous_v = None if self.values is None else self.values.reshape(B * H, -1, D)
        keys_fixed = keys.astype(mx.float16).reshape(B * H, S, D)
        values_fixed = values.astype(mx.float16).reshape(B * H, S, D)

        new_positions = mx.arange(self._true_offset, self._true_offset + S, dtype=mx.int32)
        new_positions = mx.broadcast_to(new_positions[None, :], (B * H, S))

        k, v, positions = _tova_update_batched(
            previous_k,
            previous_v,
            keys_fixed,
            values_fixed,
            self._n_sink,
            self._budget,
            backend=self._backend,
            positions=self._bh_positions,
            new_positions=new_positions,
        )
        self._states = [TovaState(k[h], v[h], self._n_sink, self._budget) for h in range(B * H)]
        self._tova_kept_bytes = k.size * 4
        # Store the complete retained state directly, avoiding the base class's
        # padded append buffer and a redundant full-cache copy on every decode.
        self.keys = k.reshape(B, H, -1, D)
        self.values = v.reshape(B, H, -1, D)
        self._bh_positions = positions
        n_kept = self.keys.shape[2]
        # head-0 true kept positions per batch element, for the NEXT call's
        # make_mask (see that method) — not this call's own mask, already
        # fixed by the time we get here.
        self._kept_positions = positions.reshape(B, H, n_kept)[:, 0, :]
        self._true_offset += S
        self.offset = self._true_offset

        if previous_k is None:
            # First call ever — nothing to concatenate; the mask mlx_lm
            # already built for this call was "causal" over N==S queries
            # against S keys (correct: no prior state to misalign with).
            result = keys_fixed.reshape(B, H, S, D), values_fixed.reshape(B, H, S, D)
        else:
            full_k = mx.concatenate([previous_k, keys_fixed], axis=1)
            full_v = mx.concatenate([previous_v, values_fixed], axis=1)
            n_full = full_k.shape[1]
            result = full_k.reshape(B, H, n_full, D), full_v.reshape(B, H, n_full, D)
        self._last_returned = result
        return result

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Explicit position-based causal mask — see VeloxQuant-MLX#370.

        Called BEFORE this step's own ``update_and_fetch`` (and thus before
        this step's own eviction, which ``update_and_fetch`` defers past
        this step's return anyway — see its docstring). ``self._kept_positions``
        holds the true positions of whatever ``self.keys`` already stores
        from the *previous* call, which is exactly ``update_and_fetch``'s
        ``previous_k`` this call will concatenate its ``N`` new tokens onto
        — so a mask sized ``[B, 1, N, len(kept) + N]`` covers this call's
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

    def size(self) -> int:
        """Stored rows; offset separately tracks the absolute RoPE position."""
        return 0 if self.keys is None else self.keys.shape[2]

    @property
    def state(self):
        """Stored (keys, values); evicted positions are not separately encoded."""
        return self.keys, self.values

    @state.setter
    def state(self, value):
        """Restore ``(keys, values)``; treats the restored row count as the true offset (exact mid-history restore of evicted positions is not supported)."""
        # K/V alone do not encode evicted positions. Preserve the base class's
        # row-count estimate on restore; exact mid-history restoration is not
        # supported without separate absolute-position metadata.
        self.keys, self.values = value
        self._states = []
        self._true_offset = self.size()
        self.offset = self._true_offset
        if self.keys is not None:
            b, h, n, d = self.keys.shape
            self._ensure_states(b, h, d)
            k, v = self.keys.reshape(b * h, n, d), self.values.reshape(b * h, n, d)
            self._states = [TovaState(k[i], v[i], self._n_sink, self._budget) for i in range(b * h)]
            # Restored rows' true original positions are not recoverable
            # from K/V alone — treat them as a contiguous trailing window
            # ending at the restored row count, same least-wrong estimate
            # `_true_offset` above already makes (see H2OKVCache.state's
            # setter docstring for the identical limitation/rationale).
            self._bh_positions = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None], (b * h, n))
            self._kept_positions = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None], (b, n))
        else:
            self._bh_positions = None
            self._kept_positions = None
        self._tova_kept_bytes = 0 if self.keys is None else self.keys.size * 4
        self._last_returned = None

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def tova_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._tova_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / tova_kept_bytes; > 1 means memory savings over fp16."""
        if self._tova_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._tova_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if not self._states or self._states[0].keys is None:
            return 0
        return int(self._states[0].keys.shape[0])

    # Without this, TOVAKVCache inherits the base mlx_lm KVCache.merge()
    # classmethod unchanged, so hasattr(cache, "merge") is True and
    # mlx_lm.server's batching machinery treats this cache as batchable.
    # Calling the inherited merge() on a fresh per-request cache silently
    # substitutes a plain BatchKVCache with no eviction and no error --
    # the request then runs with an unbounded fp16 cache while the server
    # still reports method="tova" and a NOT_TRIMMABLE tier, giving no
    # indication eviction never happened. See VeloxQuant-MLX#358; found
    # verifying VeloxQuant-Studio issue #31 (15th occurrence).
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError("TOVAKVCache does not support merge() — see VeloxQuant-MLX#358")
        )
    )


__all__ = ["TOVAKVCache"]

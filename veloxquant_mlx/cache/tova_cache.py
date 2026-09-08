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
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= tova_budget`` for all heads.
        """
        if keys.ndim != 4 or keys.shape != values.shape:
            raise ValueError("tova cache: K/V must have matching [B,H,S,D] shapes")
        B, H, S, D = keys.shape
        if min(B, H, D) < 1:
            raise ValueError("tova cache: batch/head/dimension must be positive")
        if self._states and (B, H, D) != (self._B, self._H, self._head_dim):
            raise ValueError("tova cache: batch/head/dimension cannot change after initialization")
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        if S == 0:
            if self.keys is None:
                return keys.astype(mx.float16), values.astype(mx.float16)
            return self.keys, self.values
        previous_k = None if self.keys is None else self.keys.reshape(B * H, -1, D)
        previous_v = None if self.values is None else self.values.reshape(B * H, -1, D)
        k, v = _tova_update_batched(
            previous_k,
            previous_v,
            keys.astype(mx.float16).reshape(B * H, S, D),
            values.astype(mx.float16).reshape(B * H, S, D),
            self._n_sink,
            self._budget,
            backend=self._backend,
        )
        self._states = [TovaState(k[h], v[h], self._n_sink, self._budget) for h in range(B * H)]
        self._tova_kept_bytes = k.size * 4
        # Store the complete retained state directly, avoiding the base class's
        # padded append buffer and a redundant full-cache copy on every decode.
        self.keys = k.reshape(B, H, -1, D)
        self.values = v.reshape(B, H, -1, D)
        self._true_offset += S
        self.offset = self._true_offset
        return self.keys, self.values

    def size(self) -> int:
        """Stored rows; offset separately tracks the absolute RoPE position."""
        return 0 if self.keys is None else self.keys.shape[2]

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, value):
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
        self._tova_kept_bytes = 0 if self.keys is None else self.keys.size * 4

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


__all__ = ["TOVAKVCache"]

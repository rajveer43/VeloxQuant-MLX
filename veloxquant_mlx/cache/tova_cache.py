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
  - No RoPE position-ID remapping after eviction; original positions are
    preserved in returned rows.
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
    full_tova_fp16_bytes,
    init_tova_state,
    tova_fp16_bytes,
    tova_get_kv,
    tova_update,
)


class TOVAKVCache(_MLXKVCache):
    """KV cache implementing TOVA-adapted current-step attention eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``tova_budget`` (int, default 512) — maximum tokens retained at any time,
            ``tova_n_sink`` (int, default 4)   — leading positions never evicted.

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

        self._head_dim: int = 0
        self._states: list[TovaState] = []
        self._B: int = 0
        self._H: int = 0

        self._tova_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

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
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = tova_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = tova_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._tova_kept_bytes = sum(tova_fp16_bytes(st) for st in self._states)

        # K_out/V_out is the full retained state every call, not a delta —
        # reset so the base class's append-only buffer starts fresh instead
        # of stacking on top of the previous call's rows. Without this,
        # self.keys/self.values/self.offset stay at __init__ defaults
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

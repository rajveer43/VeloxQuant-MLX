"""H2O-adapted KV cache — cumulative attention-mass heavy-hitter oracle eviction.

Inspired by "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048).
Documented as "H2O-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Heavy-hitter eviction: each incoming token's approximate attention distribution
over the existing cache is computed using the new key vector as a proxy query
(true query not visible at the cache wrapper level). The resulting softmax
weights are accumulated into a per-token cumulative importance score. Whenever
the cache exceeds ``h2o_budget`` tokens, the lowest-score non-sink token is
permanently dropped. The cache never exceeds ``h2o_budget`` positions.

This is the third distinct eviction axis in VeloxQuant-MLX:
  - SnapKV-adapted : score-based, fires once at prefill end only.
  - StreamingLLM-adapted : positional (recency + sink), fires every step.
  - H2O-adapted    : cumulative attention mass, fires every step when over budget.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: attention weights are computed using the new key vector
    in place of the true query (not visible at cache level). Same approximation
    as SnapKV-adapted.
  - No RoPE position-ID remapping after eviction; original positions are
    preserved in returned rows.
  - Uniform budget and n_sink across all heads.
  - Scores accumulate as a running sum of softmax weights; the paper accumulates
    unnormalised attention logits in some variants — this may diverge at very
    low budgets.

Byte accounting:
    h2o_kept_bytes    — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / h2o_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.h2o import (
    H2OState,
    full_h2o_fp16_bytes,
    h2o_fp16_bytes,
    h2o_get_kv,
    h2o_update,
    init_h2o_state,
)


class H2OKVCache(_MLXKVCache):
    """KV cache implementing H2O-adapted heavy-hitter oracle eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``h2o_budget`` (int, default 512) — maximum tokens retained at any time,
            ``h2o_n_sink`` (int, default 4)   — leading positions never evicted.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        H2O update loop — unlike SnapKV-adapted, there is no prefill-only phase.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()`` propagates
        all ``h2o_*`` fields automatically via ``dataclasses.replace``.
        The per-head state is lazily initialised on the first call to
        ``update_and_fetch`` when shapes are known.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "h2o_budget", 512))
        self._n_sink = int(getattr(config, "h2o_n_sink", 4))

        self._head_dim: int = 0
        self._states: list[H2OState] = []
        self._B: int = 0
        self._H: int = 0

        self._h2o_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head H2OState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [init_h2o_state(self._n_sink, self._budget, D) for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply H2O eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= h2o_budget`` for all heads.
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
                st = h2o_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = h2o_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._h2o_kept_bytes = sum(h2o_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def h2o_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._h2o_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / h2o_kept_bytes; > 1 means memory savings over fp16."""
        if self._h2o_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._h2o_kept_bytes

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


__all__ = ["H2OKVCache"]

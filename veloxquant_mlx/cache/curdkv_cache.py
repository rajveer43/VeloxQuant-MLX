"""CurDKV-adapted KV cache — value-aware leverage-score heavy-hitter eviction.

Inspired by "Value-Guided KV Compression for LLMs via Approximated CUR
Decomposition" (Sengupta, Chaudhary, Chakraborty; NeurIPS 2025,
arXiv:2509.15038). Documented as "CurDKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

Value-aware eviction: each incoming token's approximate leverage score over
the existing cache is estimated from the joint (key, value) structure — the
proxy attention-weighted value block's dominant singular directions — using
the new key vector as a proxy query (true query not visible at the cache
wrapper level). The resulting scores are accumulated into a per-token
cumulative importance score. Whenever the cache exceeds ``curdkv_budget``
tokens, the lowest-score non-sink token is permanently dropped. The cache
never exceeds ``curdkv_budget`` positions.

This is the fourteenth eviction-family method in VeloxQuant-MLX, and the
first whose retention score is value-aware rather than key-only:
  - H2O-adapted    : cumulative attention mass over keys only.
  - KNorm-adapted  : intrinsic key-vector norm only.
  - Q-Filters      : frozen per-head key-SVD projection direction.
  - CurDKV-adapted : leverage scores over the joint (key, value) block — a
                     token with a "important-looking" key but a
                     near-zero/orthogonal value contribution is correctly
                     deprioritized here, unlike the key-only methods above.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: leverage scores are computed using the new key vector
    in place of the true query (not visible at cache level). Same
    approximation as H2O-adapted/SnapKV-adapted.
  - Approximated leverage scores via a small-rank SVD of the proxy
    attention-weighted value block, not the paper's own CUR sampling
    algorithm — a standard leverage-score estimator, not a reproduction of
    the paper's specific sketching routine.
  - No RoPE position-ID remapping after eviction; original positions are
    preserved in returned rows.
  - Uniform budget and n_sink across all heads.

Byte accounting:
    curdkv_kept_bytes  — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / curdkv_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.curdkv import (
    CurDKVState,
    curdkv_fp16_bytes,
    curdkv_get_kv,
    curdkv_update,
    full_curdkv_fp16_bytes,
    init_curdkv_state,
)


class CurDKVKVCache(_MLXKVCache):
    """KV cache implementing CurDKV-adapted value-aware leverage-score eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``curdkv_budget`` (int, default 512)   — maximum tokens retained at any time,
            ``curdkv_n_sink`` (int, default 4)      — leading positions never evicted,
            ``curdkv_rank_cap`` (int, default 16)   — SVD rank cap for leverage-score estimation.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        CurDKV update loop — unlike SnapKV-adapted, there is no prefill-only
        phase.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()`` propagates
        all ``curdkv_*`` fields automatically via ``dataclasses.replace``.
        The per-head state is lazily initialised on the first call to
        ``update_and_fetch`` when shapes are known.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "curdkv_budget", 512))
        self._n_sink = int(getattr(config, "curdkv_n_sink", 4))
        self._rank_cap = int(getattr(config, "curdkv_rank_cap", 16))

        self._head_dim: int = 0
        self._states: list[CurDKVState] = []
        self._B: int = 0
        self._H: int = 0

        self._curdkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head CurDKVState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_curdkv_state(self._n_sink, self._budget, D, self._rank_cap)
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply CurDKV eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= curdkv_budget`` for all heads.
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
                st = curdkv_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = curdkv_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._curdkv_kept_bytes = sum(curdkv_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def curdkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._curdkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / curdkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._curdkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._curdkv_kept_bytes

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


__all__ = ["CurDKVKVCache"]

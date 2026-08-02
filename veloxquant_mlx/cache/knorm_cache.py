"""L2Norm-adapted KV cache — intrinsic key-norm eviction.

Inspired by "A Simple and Effective L2 Norm-Based Strategy for KV Cache
Compression" (Devoto, Zhao, Scardapane, Minervini — EMNLP 2024;
arXiv:2406.11430). Documented as "L2Norm-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

The repo's first **intrinsic-signal** eviction cache: token importance is
read directly off the stored key vector's L2 norm — the paper's finding is
that in trained decoder LMs a *low* key norm predicts *high* future
attention, so the cache keeps the lowest-norm tokens and evicts the
highest-norm ones. No attention scores, no key-as-query proxy (the
approximation H2O/SnapKV/TOVA need), no structure-only recency rule
(StreamingLLM): the paper's actual signal is fully observable at the cache
level, making this the cleanest adaptation in the eviction family.

Because the score is intrinsic (computed once at insertion, never updated):
  - eviction vectorizes as one protected top-k per incoming block — no
    per-token softmax-over-cache loop like H2O;
  - with ``knorm_recent=0`` the kept set is **path-independent**: prefill in
    one block and token-by-token decode yield bit-for-bit identical caches
    (the "keep k best with a heap" invariant — see quantizers/knorm.py).

Adaptation limitations (stated plainly):
  - The low-norm ⇒ high-attention correlation is the paper's empirical claim
    about trained models — not validated here on synthetic data (the
    benchmark's isotropic control shows no advantage, honestly reported).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget and n_sink across all heads.
  - ``knorm_recent`` (trailing protected window) is an extension, off by
    default; enabling it breaks the path-independence property.

Byte accounting (same names as H2OKVCache):
    knorm_kept_bytes  — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / knorm_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.knorm import (
    KnormState,
    full_knorm_fp16_bytes,
    init_knorm_state,
    knorm_fp16_bytes,
    knorm_get_kv,
    knorm_update,
)


class L2NormKVCache(_MLXKVCache):
    """KV cache implementing L2Norm-adapted intrinsic key-norm eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``knorm_budget`` (int, default 512) — max tokens retained (incl. sinks),
            ``knorm_n_sink`` (int, default 4)   — leading positions never evicted,
            ``knorm_recent`` (int, default 0)   — trailing protected window (extension),
            ``knorm_keep``  (str, default "low") — "low" = paper finding; "high" = inverted.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Single-layer (no coordinator); the default ``KVCacheBuilder.for_model()``
        path returns one ``L2NormKVCache`` per attention layer. Per-head state
        is lazily initialised on the first ``update_and_fetch``. Validation
        (keep mode, sink/recent-vs-budget guard) happens at construction.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "knorm_budget", 512))
        self._n_sink = int(getattr(config, "knorm_n_sink", 4))
        self._recent = int(getattr(config, "knorm_recent", 0))
        self._keep = str(getattr(config, "knorm_keep", "low"))

        # Fail at build time with clear messages (delegates the guards).
        init_knorm_state(self._n_sink, self._budget, 1, recent=self._recent, keep=self._keep)

        self._head_dim: int = 0
        self._states: list[KnormState] = []
        self._B: int = 0
        self._H: int = 0

        self._knorm_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_knorm_state(
                    self._n_sink, self._budget, D, recent=self._recent, keep=self._keep
                )
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply key-norm eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= knorm_budget`` for all heads.
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
                st = knorm_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = knorm_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._knorm_kept_bytes = sum(knorm_fp16_bytes(st) for st in self._states)
        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def knorm_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._knorm_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / knorm_kept_bytes; > 1 means memory savings over fp16."""
        if self._knorm_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._knorm_kept_bytes

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


__all__ = ["L2NormKVCache"]

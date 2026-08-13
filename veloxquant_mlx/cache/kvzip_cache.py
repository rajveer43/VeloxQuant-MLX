"""KVzip-adapted KV cache — context-reconstruction reliance retention.

Inspired by "KVzip: Query-Agnostic KV Cache Compression with Context
Reconstruction" (Kim, Kim, Kwon, Lee, Yun, Song, NeurIPS 2025 (Oral),
arXiv:2505.23416, github.com/snu-mllab/KVzip). Documented as "KVzip-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

Keeps a constant-size cache by ranking stored tokens according to their
**context-reconstruction reliance** (paper's contribution: score a KV pair by
how much the model relies on it to reconstruct its own context, query-agnostic,
then evict the least-relied-upon pairs). Setting ``kvzip_probe = "latest"``
reduces this to a latest-token (TOVA-adapted-style) eviction — the honest
reference behavior, checked by a dedicated test.

Where it sits: the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM / Keyformer / MorphKV). The
distinguishing axis is reconstruction reliance (attention from a fixed
reconstruction probe), versus cumulative (H2O), latest-only (TOVA), or
recent-window (MorphKV) query attention.

THE HONESTY CRUX:
  1. Proxy reconstruction — the stored/incoming KEYS stand in for the unseen
     reconstruction queries (as H2O / TOVA / MorphKV-adapted).
  2. Query-agnostic, recomputed each step from the live keep set against the
     probe — NOT a cumulative accumulator. Only ``probe = "latest"`` collapse is
     pinned exactly (to the latest-token ranking); no H2O collapse is claimed.
  3. Not validated on a trained model; the paper's accuracy/memory numbers are
     the paper's, on trained models — never reproduced or claimed here. The
     mechanism's benefit is measured only under a constructed
     reconstruction-shift geometry, with a null control.

Adaptation limitations (stated plainly):
  - Key-as-reconstruction-probe proxy (crux 1).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / probe across all heads.
  - Leading ``kvzip_n_sink`` tokens protected as sinks; no trailing window is
    force-protected (a token survives only if the reconstruction probe relies
    on it).

Byte accounting (same names as H2OKVCache / MorphKVKVCache):
    kvzip_kept_bytes — fp16 bytes for retained K + V tokens
    full_seq_bytes   — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / kvzip_kept_bytes (> 1 = savings)
    tokens_seen      — total token positions ever passed to update_and_fetch
    tokens_kept      — tokens currently in the (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.kvzip import (
    KVzipState,
    full_kvzip_fp16_bytes,
    init_kvzip_state,
    kvzip_fp16_bytes,
    kvzip_get_kv,
    kvzip_update,
)


class KVzipKVCache(_MLXKVCache):
    """KV cache implementing KVzip-adapted reconstruction-reliance retention for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``kvzip_budget`` (int, default 512) — max tokens kept (incl. sinks),
            ``kvzip_n_sink`` (int, default 4)   — leading positions never evicted,
            ``kvzip_probe``  (str, default "context") — reconstruction probe;
                "latest" collapses onto the TOVA-adapted latest-token ranking.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) go through the same update
        loop. Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``kvzip_*`` fields via ``dataclasses.replace``. Per-head
        state is lazily initialised on the first ``update_and_fetch``. KVzip is
        deterministic (no RNG). Validation (budget/sink bounds, probe value)
        happens at construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "kvzip_budget", 512))
        self._n_sink = int(getattr(config, "kvzip_n_sink", 4))
        self._probe = str(getattr(config, "kvzip_probe", "context"))

        # Fail at build time with clear messages (delegates the guards).
        init_kvzip_state(self._n_sink, self._budget, 1, probe=self._probe)

        self._head_dim: int = 0
        self._states: list[KVzipState] = []
        self._B: int = 0
        self._H: int = 0

        self._kvzip_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_kvzip_state(self._n_sink, self._budget, D, probe=self._probe)
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply reconstruction-reliance eviction, return window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= kvzip_budget`` for all heads.
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
                st = kvzip_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = kvzip_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._kvzip_kept_bytes = sum(kvzip_fp16_bytes(st) for st in self._states)

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
    def kvzip_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._kvzip_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / kvzip_kept_bytes; > 1 means memory savings over fp16."""
        if self._kvzip_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._kvzip_kept_bytes

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


__all__ = ["KVzipKVCache"]

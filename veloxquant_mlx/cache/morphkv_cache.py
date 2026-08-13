"""MorphKV-adapted KV cache — recent-window correlation retention.

Inspired by "Dialogue Without Limits: Constant-Sized KV Caches for Extended
Responses in LLMs" (Ghadia, Kumar, Jain, Nair, Das, ICML 2025,
arXiv:2503.00979). Documented as "MorphKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

Keeps a constant-size cache by ranking stored tokens according to their
correlation with the aggregate proxy-attention of a **sliding window of recent
tokens** (paper's contribution: eliminate the "early-token bias" of cumulative
H2O-style scoring by tracking what the recent context is actually attending to).
Setting ``morphkv_window = 1`` reduces this to a latest-token (TOVA-adapted-
style) eviction — the honest reference behavior, checked by a dedicated test.

Where it sits: the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM / Keyformer). The distinguishing
axis is recent-*window* correlation, versus cumulative (H2O) or latest-only
(TOVA) scoring.

THE HONESTY CRUX:
  1. Proxy query — the incoming KEY stands in for the unseen query (as H2O /
     TOVA / SnapKV / Keyformer-adapted).
  2. Constant-size, recomputed each step from the live keep set + recent window
     — NOT a cumulative accumulator. Only ``window = 1`` collapse is pinned
     exactly (to the latest-token ranking); no H2O collapse is claimed.
  3. Not validated on a trained model; the paper's accuracy/memory numbers are
     the paper's, on trained models — never reproduced or claimed here. The
     mechanism's benefit is measured only under a constructed topic-shift
     geometry, with a null control.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / window across all heads.
  - Trailing ``morphkv_window`` tokens protected (recency context that drives
    the ranking); leading ``morphkv_n_sink`` tokens protected as sinks.

Byte accounting (same names as H2OKVCache / KeyformerKVCache):
    morphkv_kept_bytes — fp16 bytes for retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / morphkv_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.morphkv import (
    MorphKVState,
    full_morphkv_fp16_bytes,
    init_morphkv_state,
    morphkv_fp16_bytes,
    morphkv_get_kv,
    morphkv_update,
)


class MorphKVKVCache(_MLXKVCache):
    """KV cache implementing MorphKV-adapted recent-window retention for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``morphkv_budget`` (int, default 512) — max tokens kept (incl. sinks),
            ``morphkv_n_sink`` (int, default 4)   — leading positions never evicted,
            ``morphkv_window`` (int, default 8)   — trailing recent-attention window.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) go through the same update
        loop. Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``morphkv_*`` fields via ``dataclasses.replace``. Per-head
        state is lazily initialised on the first ``update_and_fetch``. MorphKV is
        deterministic (no RNG). Validation (budget/window/sink bounds) happens at
        construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "morphkv_budget", 512))
        self._n_sink = int(getattr(config, "morphkv_n_sink", 4))
        self._window = int(getattr(config, "morphkv_window", 8))

        # Fail at build time with clear messages (delegates the guards).
        init_morphkv_state(self._n_sink, self._budget, 1, window=self._window)

        self._head_dim: int = 0
        self._states: list[MorphKVState] = []
        self._B: int = 0
        self._H: int = 0

        self._morphkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_morphkv_state(self._n_sink, self._budget, D, window=self._window)
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply recent-window eviction, return window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= morphkv_budget`` for all heads.
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
                st = morphkv_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = morphkv_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._morphkv_kept_bytes = sum(morphkv_fp16_bytes(st) for st in self._states)

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
    def morphkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._morphkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / morphkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._morphkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._morphkv_kept_bytes

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


__all__ = ["MorphKVKVCache"]

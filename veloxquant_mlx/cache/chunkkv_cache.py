"""ChunkKV-adapted KV cache — chunk-level (semantic-block) eviction.

Inspired by "ChunkKV: Semantic-Preserving KV Cache Compression for Efficient
Long-Context LLM Inference" (Liu et al., 2025, arXiv:2502.00299). Documented as
"ChunkKV-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

ChunkKV evicts KV at **chunk granularity**: the sequence is partitioned into
contiguous chunks of ``chunk_size`` tokens, and a chunk is kept or dropped as a
whole. Where H2O/TOVA/SnapKV keep the top *tokens* by importance — which shreds
contiguous spans whose meaning is collective — ChunkKV keeps the top *chunks*, so
surviving context stays locally coherent. Chunk importance is a pooled proxy over
an existing per-token signal: cumulative attention mass (``score="attn_mass"``,
the H2O scorer) or mean key L2 norm (``score="key_norm"``).

Like H2O/TOVA and unlike XQuant/MiniCache/SqueezeAttention, ChunkKV needs **no
runtime coordinator** — every layer/head resolves its own chunks independently,
so the default ``KVCacheBuilder.for_model`` path (one ``ChunkKVCache`` per layer)
is all it needs. When ``chunk_size == 1`` the method reduces bit-for-bit to
H2O-adapted (each chunk is one token).

This is the seventh distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted     : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted        : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted       : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted  : H2O scoring with a fixed per-layer pyramid budget.
  - SqueezeAttention-adapted : H2O scoring with a data-driven per-layer budget.
  - ChunkKV-adapted    : H2O/key-norm scoring, evicted at CHUNK granularity
    (whole contiguous blocks) instead of per token.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted / SnapKV-adapted).
  - Pooled per-token score as a proxy for the paper's attention-over-chunk
    importance (mean-pooled, same chunk-granular decision).
  - No layer-wise kept-index reuse (a decode-speed trick in the paper); each
    layer resolves chunks independently.
  - Streaming eviction (drop a chunk once the cache exceeds budget by a chunk)
    rather than a single one-shot prefill compression.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Byte accounting:
    chunkkv_kept_bytes — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / chunkkv_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
    chunk_size         — this cache's eviction granularity (diagnostic)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.chunkkv import (
    ChunkKVState,
    chunkkv_fp16_bytes,
    chunkkv_get_kv,
    chunkkv_trim_to,
    chunkkv_update,
    init_chunkkv_state,
)


class ChunkKVCache(_MLXKVCache):
    """KV cache implementing ChunkKV-adapted chunk-level eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``chunkkv_budget`` (int, default 512)     — max tokens kept (sinks incl.).
            ``chunkkv_chunk_size`` (int, default 8)   — eviction granularity ``C``;
                ``1`` reduces to H2O-adapted exactly.
            ``chunkkv_n_sink`` (int, default 4)       — leading positions never evicted.
            ``chunkkv_score`` (str, default "attn_mass") — "attn_mass" | "key_norm".

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        eviction loop. Per-head state is lazily initialised on the first call.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "chunkkv_budget", 512))
        self._chunk_size = int(getattr(config, "chunkkv_chunk_size", 8))
        self._n_sink = int(getattr(config, "chunkkv_n_sink", 4))
        self._score_mode = str(getattr(config, "chunkkv_score", "attn_mass"))

        self._head_dim: int = 0
        self._states: list[ChunkKVState] = []
        self._B: int = 0
        self._H: int = 0

        self._chunkkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head ChunkKVState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_chunkkv_state(
                    self._n_sink,
                    self._budget,
                    D,
                    chunk_size=self._chunk_size,
                    score_mode=self._score_mode,
                )
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply ChunkKV eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= budget`` for all heads and ``n_kept`` is chunk-aligned.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        # 1) Update every head's state independently.
        for b in range(B):
            for h in range(H):
                idx = self._head_idx(b, h)
                self._states[idx] = chunkkv_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )

        # 2) Whole-chunk retention lets heads keep slightly different token counts;
        #    the MLX attention path needs a rectangular [B, H, n_kept, D] output,
        #    so align every head to the common minimum kept-length by dropping each
        #    head's oldest non-sink tokens down to that length. When chunk_size=1
        #    all heads already hold exactly `budget`, so no trimming occurs and the
        #    H2O equivalence is preserved.
        min_kept = min(
            (chunkkv_get_kv(st)[0].shape[0] for st in self._states),
            default=0,
        )
        for idx, st in enumerate(self._states):
            self._states[idx] = chunkkv_trim_to(st, min_kept)

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                k_h, v_h = chunkkv_get_kv(self._states[self._head_idx(b, h)])
                k_out_h.append(k_h)  # [min_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, min_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, min_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states.
        self._chunkkv_kept_bytes = sum(chunkkv_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def layer_budget(self) -> int:
        """This cache's per-layer token budget."""
        return self._budget

    @property
    def chunk_size(self) -> int:
        """This cache's eviction granularity ``C`` (diagnostic)."""
        return self._chunk_size

    @property
    def chunkkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._chunkkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / chunkkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._chunkkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._chunkkv_kept_bytes

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


__all__ = ["ChunkKVCache"]

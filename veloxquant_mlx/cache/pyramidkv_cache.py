"""PyramidKV-adapted KV cache — layer-adaptive budget attention-mass eviction.

Inspired by "PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funneling" (Cai et al., 2024, arXiv:2406.02069). Documented as
"PyramidKV-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

PyramidKV is H2O's cumulative-attention-mass eviction with a **per-layer budget**
instead of a single global one. The observation is *pyramidal information
funneling*: early layers attend broadly (need a big cache), deep layers attend
narrowly (need a small one). Allocating a pyramid of budgets — large early,
small deep, same average — beats a uniform budget at equal total memory.

Each layer's budget is resolved once at ``KVCacheBuilder.for_model()`` build time
by ``pyramid_budgets(n_layers, avg, n_sink, beta)`` and baked into that layer's
config as ``pyramid_resolved_budget``. Unlike XQuant / MiniCache, PyramidKV needs
**no runtime coordinator** — layers never exchange data during generation; the
only cross-layer information is each layer's index, consumed at build time.

Single-cache construction (``KVCacheFactory.create``) has no layer context, so it
falls back to ``pyramid_budget`` (the average) — behaving as one uniform-budget
H2O layer. The pyramid only takes effect via ``for_model``.

This is the fifth distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted    : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted   : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted : H2O scoring with a per-layer pyramid budget.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted / SnapKV-adapted).
  - Fixed monotone (linear) budget schedule rather than the paper's
    prefill-entropy-derived allocation — funneling shape preserved, exact
    per-layer values not data-driven.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Byte accounting:
    pyramid_kept_bytes — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / pyramid_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
    layer_budget       — this layer's resolved budget (diagnostic)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.pyramidkv import (
    PyramidState,
    init_pyramid_state,
    pyramid_fp16_bytes,
    pyramid_get_kv,
    pyramid_update,
)


class PyramidKVCache(_MLXKVCache):
    """KV cache implementing PyramidKV-adapted layer-adaptive eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``pyramid_resolved_budget`` (int or None) — this layer's budget from
                the pyramid schedule; set by ``for_model``. If None, falls back to
                ``pyramid_budget`` (single-layer / uniform behaviour).
            ``pyramid_budget`` (int, default 512) — average/fallback budget.
            ``pyramid_n_sink`` (int, default 4)   — leading positions never evicted.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        eviction loop. Per-head state is lazily initialised on the first call.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        resolved = getattr(config, "pyramid_resolved_budget", None)
        if resolved is None:
            resolved = int(getattr(config, "pyramid_budget", 512))
        self._budget = int(resolved)
        self._n_sink = int(getattr(config, "pyramid_n_sink", 4))

        self._head_dim: int = 0
        self._states: list[PyramidState] = []
        self._B: int = 0
        self._H: int = 0

        self._pyramid_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head PyramidState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [init_pyramid_state(self._n_sink, self._budget, D) for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply PyramidKV eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= layer_budget`` for all heads.
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
                st = pyramid_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = pyramid_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._pyramid_kept_bytes = sum(pyramid_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def layer_budget(self) -> int:
        """This layer's resolved per-layer budget (from the pyramid schedule)."""
        return self._budget

    @property
    def pyramid_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._pyramid_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / pyramid_kept_bytes; > 1 means memory savings over fp16."""
        if self._pyramid_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._pyramid_kept_bytes

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


__all__ = ["PyramidKVCache"]

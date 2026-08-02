"""CaM-adapted KV cache — Cache Merging (merge evicted tokens instead of dropping).

Inspired by "CaM: Cache Merging for Memory-efficient LLMs Inference" (Zhang, Du,
Luo, Zhong, Zhang, Liu & Ji, ICML 2024, PMLR 235:58840-58850). Documented as
"CaM-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Every other eviction cache in the repo (SnapKV, StreamingLLM, H2O, TOVA,
PyramidKV, SqueezeAttention, ChunkKV) permanently **drops** the tokens it evicts.
CaM instead **merges** each evicted token into the surviving token it most
resembles (a cosine-similarity-weighted blend of the value rows, and optionally
the keys), then removes only the now-redundant slot — so the information is
folded into a neighbour rather than discarded. This is the first method on the
**merge-vs-drop** axis; the eviction *choice* is H2O's, only the disposition
differs.

Like H2O/TOVA/ChunkKV and unlike XQuant/MiniCache/SqueezeAttention, CaM needs
**no runtime coordinator** — every layer/head merges independently — so the
default ``KVCacheBuilder.for_model`` path (one ``CaMKVCache`` per layer) is all it
needs. With ``cam_merge="drop"`` the blend weight is zero and CaM reduces
**bit-for-bit** to H2O-adapted.

This is the eighth distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted     : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted        : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted       : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted  : H2O scoring with a fixed per-layer pyramid budget.
  - SqueezeAttention-adapted : H2O scoring with a data-driven per-layer budget.
  - ChunkKV-adapted    : H2O/key-norm scoring, evicted at CHUNK granularity.
  - CaM-adapted        : H2O scoring + eviction, but the loser is MERGED into a
    survivor (cosine-weighted) rather than dropped.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted) for both the importance score and
    the merge-target similarity.
  - Cosine-similarity merge weight rather than the paper's attention-prominence
    weight (which is ~0 for a just-appended token that overflows before it
    accumulates mass — the common case at the streaming eviction boundary).
  - Single nearest-survivor merge (no multi-target soft assignment / sampling).
  - No RoPE position-ID remapping after merge.
  - Uniform budget across heads within a layer.

Byte accounting:
    cam_kept_bytes    — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / cam_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
    merge_mode        — this cache's merge disposition (diagnostic)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.cam import (
    CaMState,
    cam_fp16_bytes,
    cam_get_kv,
    cam_update,
    init_cam_state,
)


class CaMKVCache(_MLXKVCache):
    """KV cache implementing CaM-adapted cache-merging eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``cam_budget`` (int, default 512)   — maximum tokens retained.
            ``cam_n_sink`` (int, default 4)     — leading positions never evicted.
            ``cam_merge`` (str, default "sim_weighted") — "sim_weighted" |
                "mean" | "drop"; "drop" reduces bit-for-bit to H2O-adapted.
            ``cam_merge_keys`` (bool, default False) — merge keys too (values are
                always merged).

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        merge loop. Per-head state is lazily initialised on the first call.
        Because CaM merges (not drops) it always trims to exactly ``budget`` — the
        output is rectangular ``[B, H, budget, D]`` once past budget, so no
        cross-head alignment is needed.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "cam_budget", 512))
        self._n_sink = int(getattr(config, "cam_n_sink", 4))
        self._merge_mode = str(getattr(config, "cam_merge", "sim_weighted"))
        self._merge_keys = bool(getattr(config, "cam_merge_keys", False))

        self._head_dim: int = 0
        self._states: list[CaMState] = []
        self._B: int = 0
        self._H: int = 0

        self._cam_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head CaMState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_cam_state(
                    self._n_sink,
                    self._budget,
                    D,
                    merge_mode=self._merge_mode,
                    merge_keys=self._merge_keys,
                )
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply CaM merge-eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= cam_budget`` for all heads.
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
                st = cam_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = cam_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states.
        self._cam_kept_bytes = sum(cam_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def layer_budget(self) -> int:
        """This cache's per-layer token budget."""
        return self._budget

    @property
    def merge_mode(self) -> str:
        """This cache's merge disposition (diagnostic)."""
        return self._merge_mode

    @property
    def cam_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._cam_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / cam_kept_bytes; > 1 means memory savings over fp16."""
        if self._cam_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._cam_kept_bytes

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


__all__ = ["CaMKVCache"]

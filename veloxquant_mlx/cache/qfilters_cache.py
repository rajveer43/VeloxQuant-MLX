"""Q-Filters-adapted KV cache — query-agnostic projection eviction.

Inspired by "Q-Filters: Leveraging QK Geometry for Efficient KV Cache
Compression" (arXiv:2503.02812, **preprint**). Documented as
"Q-Filters-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Scores each cached key by its projection onto a frozen per-head direction
(the Q-Filter) and evicts the lowest-scoring tokens — attention-quality
importance without computing attention and without a query at eviction time.
The repo's fourth eviction scorer class: not attention/proxy (SnapKV, H2O,
TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM), not structural
(StreamingLLM, sink), not intrinsic-norm (L2Norm).

THE HONESTY CRUX: the paper derives the filter from query-distribution SVD
offline; a cache never sees queries, so we derive it from the SVD of the
first ``qfilters_calib_tokens`` observed KEYS and freeze it. A documented
deviation — a different estimator of the same head-geometry direction, never
claimed equivalent, validated only under constructed geometry (see the
benchmark's ``filter_cosine`` field and isotropic control).

Path-DEPENDENCE (contrast with L2NormKVCache): the filter is estimated from
whichever chunk first crosses the calibration threshold, so prefill-in-one-
block and token-by-token decode can freeze *different* directions and
diverge. There is deliberately no prefill/decode bit-for-bit equivalence
guarantee here.

Adaptation limitations (stated plainly):
  - Filter is key-SVD-derived, not query-SVD-derived (the crux above).
  - The anisotropy/attention-prediction claim is the paper's, about trained
    models; nothing here validates it on synthetic data.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget and n_sink across all heads.
  - ``qfilters_recent`` (trailing protected window) is an extension, off by
    default.

Byte accounting (same names as L2NormKVCache):
    qfilters_kept_bytes — fp16 bytes for retained K + V (plus the frozen
                          float32 filter direction, once it exists)
    full_seq_bytes      — hypothetical fp16 cost if all tokens were kept
    compression_ratio   — full_seq_bytes / qfilters_kept_bytes (> 1 = savings)
    tokens_seen         — total token positions ever passed to update_and_fetch
    tokens_kept         — tokens currently in the (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.qfilters import (
    QFiltersState,
    full_qfilters_fp16_bytes,
    init_qfilters_state,
    qfilters_fp16_bytes,
    qfilters_get_kv,
    qfilters_update,
)


class QFiltersKVCache(_MLXKVCache):
    """KV cache implementing Q-Filters-adapted projection eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``qfilters_budget`` (int, default 512) — max tokens kept (incl. sinks),
            ``qfilters_n_sink`` (int, default 4)   — leading positions never evicted,
            ``qfilters_recent`` (int, default 0)   — trailing protected window (extension),
            ``qfilters_calib_tokens`` (int, default 128) — tokens observed before
                the filter direction is estimated and frozen,
            ``qfilters_sign`` (int, default 1)     — +1 = paper direction; -1 = inverted.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Single-layer (no coordinator); the default ``KVCacheBuilder.for_model()``
        path returns one ``QFiltersKVCache`` per attention layer. Per-head
        state is lazily initialised on the first ``update_and_fetch``.
        Validation (sign, sink/recent-vs-budget) happens at construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "qfilters_budget", 512))
        self._n_sink = int(getattr(config, "qfilters_n_sink", 4))
        self._recent = int(getattr(config, "qfilters_recent", 0))
        self._calib = int(getattr(config, "qfilters_calib_tokens", 128))
        self._sign = int(getattr(config, "qfilters_sign", 1))

        # Fail at build time with clear messages (delegates the guards).
        init_qfilters_state(
            self._n_sink,
            self._budget,
            1,
            recent=self._recent,
            calib_tokens=self._calib,
            sign=self._sign,
        )

        self._head_dim: int = 0
        self._states: list[QFiltersState] = []
        self._B: int = 0
        self._H: int = 0

        self._qfilters_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_qfilters_state(
                    self._n_sink,
                    self._budget,
                    D,
                    recent=self._recent,
                    calib_tokens=self._calib,
                    sign=self._sign,
                )
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply projection eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= qfilters_budget`` once the filter has been frozen.
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
                st = qfilters_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = qfilters_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._qfilters_kept_bytes = sum(qfilters_fp16_bytes(st) for st in self._states)

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
    def qfilters_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V + filter dirs)."""
        return self._qfilters_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / qfilters_kept_bytes; > 1 means memory savings over fp16."""
        if self._qfilters_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._qfilters_kept_bytes

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


__all__ = ["QFiltersKVCache"]

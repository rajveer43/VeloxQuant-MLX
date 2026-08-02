"""KVQuant-NUQ KV cache wrapper — non-uniform quantization + outlier isolation.

Inspired by "KVQuant: Towards 10 Million Context Length LLM Inference with KV
Cache Quantization" (arXiv:2401.18079, NeurIPS 2024). Documented as
"KVQuant-adapted (VeloxQuant-MLX implementation)" — implements the two
cache-observable pillars (NUQ datatype + dense/sparse outlier isolation) and
documents the third (pre-RoPE keys) as out of scope. See
:mod:`veloxquant_mlx.quantizers.kvquant` for the numerics.

Quantization axes (matching KVQuant's asymmetry, the same axes KIVI uses):
    Keys   — per-channel: each head-dim channel gets its own non-uniform levels
             (sample axis = tokens). Channels have stable, distinct distributions.
    Values — per-token: each token gets its own levels (sample axis = channels).

Level lifecycle:
    Prefill (S > 1) fits the NUQ levels from the incoming batch and freezes them
    (``kvquant_refit_interval == 0``, the default — mirrors SVDq's frozen-V).
    Decode tokens quantize against the frozen levels. With a positive refit
    interval, levels are re-fit every N decode steps from the most recent token.

Byte accounting:
    compressed_*  — NUQ codes + per-(channel/token) level table + fp16 outlier
                    side-channel (value + position index).
    fp16_*        — uncompressed cost for the ratio.

What is NOT implemented (documented):
    - Pre-RoPE key quantization (needs a model-forward hook — outside the cache
      contract; we see post-RoPE keys only).
    - Offline calibration-set level fitting (we fit online; zero calibration).
    - Attention-aware sensitivity weighting (needs attention scores).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.kvquant import (
    dequant_nuq,
    fit_nuq_levels,
    quantize_nuq,
    split_dense_sparse,
)


class KVQuantKVCache(_MLXKVCache):
    """KV cache implementing KVQuant-NUQ non-uniform quantization.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``kvquant_bits``             (int, default 3),
            ``kvquant_outlier_fraction`` (float, default 0.01),
            ``kvquant_group_size``       (int, default 32; reserved for grouped fits),
            ``kvquant_lloyd_iters``      (int, default 8),
            ``kvquant_refit_interval``   (int, default 0 = freeze prefill levels).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._bits: int = int(getattr(config, "kvquant_bits", 3))
        self._outlier_fraction: float = float(getattr(config, "kvquant_outlier_fraction", 0.01))
        self._group_size: int = int(getattr(config, "kvquant_group_size", 32))
        self._lloyd_iters: int = int(getattr(config, "kvquant_lloyd_iters", 8))
        self._refit_interval: int = int(getattr(config, "kvquant_refit_interval", 0))

        # Frozen levels fit at prefill: keys per-channel [H, L, D],
        # values per-token use levels [H, L, D] in transposed space.
        self._key_levels: Optional[list] = None  # list over heads of [L, D]
        self._value_levels: Optional[list] = None  # list over heads of [L, D] (channel-as-sample)
        self._n_tokens: int = 0
        self._outlier_count: int = 0

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._compressed_value_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._fp16_value_bytes: int = 0

    # ------------------------------------------------------------------
    # Per-head NUQ application
    # ------------------------------------------------------------------
    def _quant_keys_head(self, k_sd: mx.array, levels: Optional[mx.array]):
        """Keys: per-channel NUQ on [S, D]. Returns (recon_fp16, levels_used)."""
        ds = split_dense_sparse(k_sd, self._outlier_fraction)
        if levels is None:
            levels = fit_nuq_levels(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq(ds.inliers, levels)
        recon = dequant_nuq(codes, levels).astype(mx.float32)
        recon = mx.where(ds.outlier_mask, ds.outlier_vals, recon)
        self._outlier_count += int(mx.sum(ds.outlier_mask).item())
        return recon.astype(mx.float16), levels

    def _quant_values_head(self, v_sd: mx.array, levels: Optional[mx.array]):
        """Values: per-token NUQ on [S, D] → transpose so tokens are columns."""
        v_ds = v_sd.T  # [D, S]: now each column is one token
        ds = split_dense_sparse(v_ds, self._outlier_fraction)
        if levels is None:
            levels = fit_nuq_levels(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq(ds.inliers, levels)
        recon = dequant_nuq(codes, levels).astype(mx.float32)
        recon = mx.where(ds.outlier_mask, ds.outlier_vals, recon)
        self._outlier_count += int(mx.sum(ds.outlier_mask).item())
        return recon.astype(mx.float16).T, levels  # back to [S, D]

    def _apply(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        # Keys use per-channel levels (stable across tokens) → fit at prefill and
        # freeze. Values use per-token levels (one set per token) → inherently
        # re-fit every call; they are never frozen across steps.
        is_prefill = self._key_levels is None
        refit_keys = is_prefill or (
            self._refit_interval > 0
            and self._n_tokens > 0
            and (self._n_tokens % self._refit_interval == 0)
        )
        key_levels = None if refit_keys else self._key_levels  # list[H] of [L, D] or None

        k_out_b, v_out_b = [], []
        new_klev = [None] * H
        for b in range(B):
            k_h, v_h = [], []
            for h in range(H):
                # Share frozen key levels across batch; fit once on (b==0) when refitting.
                kl = key_levels[h] if key_levels is not None else (new_klev[h] if b > 0 else None)
                kq, klev = self._quant_keys_head(keys[b, h], kl)
                vq, vlev = self._quant_values_head(values[b, h], None)  # values: always fresh
                new_klev[h] = klev
                last_vlev = vlev
                k_h.append(kq)
                v_h.append(vq)
            k_out_b.append(mx.stack(k_h, axis=0))
            v_out_b.append(mx.stack(v_h, axis=0))

        if refit_keys:
            self._key_levels = new_klev
        self._value_levels = [last_vlev]  # most-recent per-token levels (introspection)

        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        k_out, v_out = self._apply(keys, values)
        self._n_tokens += S
        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, v_out)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        L = 1 << self._bits
        # Codes: bits per element. Level table: L fp16 per channel (keys) / per
        # token (values). Outlier side-channel: fp16 value + ~index bits.
        code_bytes = math.ceil(S * D * self._bits / 8)
        key_table_bytes = L * D * 2  # per-channel table
        val_table_bytes = L * S * 2  # per-token table
        idx_bits = max(1, math.ceil(math.log2(max(2, S * D))))
        n_out = max(0, int(round(S * D * self._outlier_fraction)))
        outlier_bytes = n_out * (2 + math.ceil(idx_bits / 8))

        self._compressed_key_bytes += (code_bytes + key_table_bytes + outlier_bytes) * B * H
        self._compressed_value_bytes += (code_bytes + val_table_bytes + outlier_bytes) * B * H
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bits(self) -> int:
        return self._bits

    @property
    def outlier_fraction(self) -> float:
        return self._outlier_fraction

    @property
    def outlier_count(self) -> int:
        return self._outlier_count

    @property
    def key_levels(self):
        return self._key_levels

    @property
    def value_levels(self):
        return self._value_levels

    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def effective_bits(self) -> float:
        """Effective per-element key bits (codes + table + outliers vs fp16)."""
        if self._fp16_key_bytes == 0:
            return float(self._bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["KVQuantKVCache"]

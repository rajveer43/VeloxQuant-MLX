"""Kitty KV cache wrapper — dynamic channel-wise mixed-precision key quantization.

Inspired by "Kitty: Plug-and-Play Continuous Batching with Dynamic Token
Selection" (arXiv:2511.18643, Nov 2025, unreviewed preprint). Documented as
"Kitty-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Design:
    Prefill (first call, S > 1):
        1. Compute per-channel variance of the incoming key batch K ∈ R^{S×D}.
        2. Rank channels by variance; route top-hi_fraction to hi_bit,
           remainder to lo_bit asymmetric group quantization.
        3. Initialise running accumulators: key_sum [H, D] and key_sq_sum [H, D]
           from the prefill batch so decode steps can update them incrementally.
        4. Reconstruct fp16 keys and pass to the underlying mlx_lm KVCache.

    Decode (subsequent calls, S == 1 per step):
        1. Update running accumulators with the new key token.
        2. Re-derive channel ranking from updated accumulators.
        3. Quantize the new key at the mixed-precision assignment.
        4. Reconstruct fp16 and forward to underlying KVCache.

    Values are left at fp16 throughout (Kitty is a key-only method).

Byte accounting:
    compressed_key_bytes — at effective avg_bits rate (no group-param overhead)
    fp16_key_bytes       — cost if stored as fp16 (for ratio)
    value_fp16_bytes     — values always fp16

Effective bit-width (default):
    avg_bits = 0.25×4 + 0.75×2 = 2.5 bits/element → 6.4× key bandwidth reduction
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.kitty import (
    compute_running_variance,
    quantize_mixed_channels,
    rank_channels_by_sensitivity,
)


class KittyKVCache(_MLXKVCache):
    """KV cache implementing Kitty dynamic channel-wise mixed-precision compression.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``head_dim`` (D),
            ``kitty_hi_fraction`` (float, default 0.25),
            ``kitty_hi_bit``      (int, default 4),
            ``kitty_lo_bit``      (int, default 2),
            ``kitty_group_size``  (int, default 32).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D: int = int(config.head_dim)
        self._hi_fraction: float = float(getattr(config, "kitty_hi_fraction", 0.25))
        if not 0.0 <= self._hi_fraction <= 1.0:
            raise ValueError(f"kitty: kitty_hi_fraction must be in [0, 1], got {self._hi_fraction}")
        self._hi_bit: int = int(getattr(config, "kitty_hi_bit", 4))
        self._lo_bit: int = int(getattr(config, "kitty_lo_bit", 2))
        self._group_size: int = int(getattr(config, "kitty_group_size", 32))

        # Running accumulators per head — shape [H, D], initialised at prefill
        self._key_sum: Optional[mx.array] = None  # [H, D] fp32
        self._key_sq_sum: Optional[mx.array] = None  # [H, D] fp32
        self._n_keys: int = 0  # total tokens accumulated

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------
    def _quantize_keys(self, keys: mx.array) -> mx.array:
        """Quantize keys [B, H, S, D] using current channel rankings.

        Uses per-head variance accumulated in self._key_sum / self._key_sq_sum.
        On first call (prefill), ranking is derived directly from the batch.
        On decode calls, uses running statistics updated before this call.
        """
        B, H, S, D = keys.shape
        out_batches = []
        for b in range(B):
            out_heads = []
            for h in range(H):
                k_bh = keys[b, h]  # [S, D]

                if self._n_keys == 0:
                    # Prefill: rank from the batch itself
                    hi_idx, lo_idx = rank_channels_by_sensitivity(k_bh, self._hi_fraction)
                else:
                    # Decode: rank from running variance per this head
                    var_h = compute_running_variance(
                        self._key_sum[h], self._key_sq_sum[h], self._n_keys
                    )
                    var_list = var_h.tolist()
                    sorted_idx = sorted(range(D), key=lambda i: -var_list[i])
                    n_hi = max(1, int(D * self._hi_fraction))
                    hi_idx = sorted(sorted_idx[:n_hi])
                    lo_idx = sorted(sorted_idx[n_hi:])

                k_q = quantize_mixed_channels(
                    k_bh,
                    hi_idx,
                    lo_idx,
                    hi_bit=self._hi_bit,
                    lo_bit=self._lo_bit,
                    group_size=self._group_size,
                )
                out_heads.append(k_q)
            out_batches.append(mx.stack(out_heads, axis=0))  # [H, S, D]
        return mx.stack(out_batches, axis=0)  # [B, H, S, D]

    def _update_accumulators(self, keys: mx.array) -> None:
        """Update running key_sum and key_sq_sum from incoming keys [B, H, S, D]."""
        B, H, S, D = keys.shape
        k32 = keys.astype(mx.float32)
        # Average over batch dimension; sum over sequence
        k_mean_b = mx.mean(k32, axis=0)  # [H, S, D]
        new_sum = mx.sum(k_mean_b, axis=1)  # [H, D]
        new_sq_sum = mx.sum(k_mean_b * k_mean_b, axis=1)  # [H, D]
        mx.eval(new_sum, new_sq_sum)

        if self._key_sum is None:
            self._key_sum = new_sum
            self._key_sq_sum = new_sq_sum
        else:
            self._key_sum = self._key_sum + new_sum
            self._key_sq_sum = self._key_sq_sum + new_sq_sum
            mx.eval(self._key_sum, self._key_sq_sum)

        self._n_keys += S

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape

        # Quantize using current statistics (prefill: _n_keys == 0 → batch variance)
        k_out = self._quantize_keys(keys)

        # Update running accumulators with the (fp16) quantized keys so that
        # decode steps see the same distribution as what was stored.
        self._update_accumulators(k_out)

        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, values)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        n_hi = max(1, int(D * self._hi_fraction))
        n_lo = D - n_hi

        def _channel_bytes(n_tokens: int, n_ch: int, b: int) -> int:
            code_bytes = math.ceil(n_tokens * n_ch * b / 8)
            n_groups = math.ceil(n_tokens / self._group_size)
            param_bytes = n_groups * n_ch * 2 * 2  # scale + zero, fp16
            return (code_bytes + param_bytes) * H * B

        self._compressed_key_bytes += _channel_bytes(S, n_hi, self._hi_bit) + _channel_bytes(
            S, n_lo, self._lo_bit
        )
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def value_fp16_bytes(self) -> int:
        return self._value_fp16_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Effective average bit-width per key element."""
        n_hi = max(1, int(self._D * self._hi_fraction))
        n_lo = self._D - n_hi
        return (n_hi * self._hi_bit + n_lo * self._lo_bit) / self._D

    @property
    def hi_fraction(self) -> float:
        return self._hi_fraction

    @property
    def hi_bit(self) -> int:
        return self._hi_bit

    @property
    def lo_bit(self) -> int:
        return self._lo_bit

    @property
    def group_size(self) -> int:
        return self._group_size


__all__ = ["KittyKVCache"]

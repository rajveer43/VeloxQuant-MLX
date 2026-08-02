"""AdaKV-proxy KV cache wrapper — per-head adaptive bit allocation over KIVI.

Inspired by "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference" (arXiv:2407.11550, 2024). Documented
as "AdaKV-proxy (VeloxQuant-MLX implementation)" — a proxy adaptation, not a
faithful port. See :mod:`veloxquant_mlx.quantizers.adakv` for the algorithm.

Design:
    Prefill (first call, S > 1):
        1. Update running per-head norm accumulators from the incoming batch.
        2. Recompute the per-head bit assignment from current norm-variance
           estimates under the global average-bits budget.
        3. Quantize each head's keys at its assigned bit-width and forward the
           reconstructed fp16 keys to the underlying mlx_lm KVCache.

    Decode (subsequent calls, S == 1 per step):
        1. Update accumulators with the new key token.
        2. Recompute the per-head bit assignment (every step by default).
        3. Quantize the new key at the per-head assignment, forward fp16.

    Values are left at fp16 throughout (AdaKV-proxy is a key-only method).

Byte accounting:
    compressed_key_bytes — weighted by each head's assigned bit-width
    fp16_key_bytes       — cost if stored as fp16 (for ratio)
    value_fp16_bytes     — values always fp16

What is NOT implemented (documented):
    - True Ada-KV head-adaptive *eviction* budget (needs softmax attention).
    - Cross-layer budget sharing.
    - Caching the bit assignment across update_interval > 1 steps. The
      ``adakv_update_interval`` config field is wired but the assignment is
      recomputed every step regardless (future optimisation).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.adakv import (
    allocate_head_bits,
    compute_head_norm_variance,
    quantize_head,
)


class AdaKVCache(_MLXKVCache):
    """KV cache implementing AdaKV-proxy per-head adaptive bit allocation.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``adakv_target_avg_bits`` (float, default 2.0),
            ``adakv_lo_bit``          (int, default 2),
            ``adakv_mid_bit``         (int, default 3),
            ``adakv_hi_bit``          (int, default 4),
            ``adakv_group_size``      (int, default 32),
            ``adakv_update_interval`` (int, default 1).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._target_avg_bits: float = float(getattr(config, "adakv_target_avg_bits", 2.0))
        self._lo_bit: int = int(getattr(config, "adakv_lo_bit", 2))
        self._mid_bit: int = int(getattr(config, "adakv_mid_bit", 3))
        self._hi_bit: int = int(getattr(config, "adakv_hi_bit", 4))
        self._group_size: int = int(getattr(config, "adakv_group_size", 32))
        self._update_interval: int = max(1, int(getattr(config, "adakv_update_interval", 1)))

        # Allowed bit set (dedup + sort). mid==hi collapses to a 2-tier set.
        self._allowed_bits: list[int] = sorted({self._lo_bit, self._mid_bit, self._hi_bit})

        # Running per-head accumulators of the per-token key L2 norm.
        self._norm_sum: Optional[mx.array] = None  # [H] fp32
        self._norm_sq_sum: Optional[mx.array] = None  # [H] fp32
        self._n_tokens: int = 0  # total tokens seen

        # Current per-head bit assignment ([H] ints). Set on first update.
        self._head_bits: Optional[list[int]] = None

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------
    def _update_norm_accumulators(self, keys: mx.array) -> None:
        """Update running per-head sum/sum-of-squares of per-token ‖k_t‖₂.

        Args:
            keys: [B, H, S, D]. Per-token norms are averaged over the batch.
        """
        B, H, S, D = keys.shape
        k32 = keys.astype(mx.float32)
        norms = mx.sqrt(mx.sum(k32 * k32, axis=-1))  # [B, H, S]
        norms_b = mx.mean(norms, axis=0)  # [H, S] (avg over batch)
        new_sum = mx.sum(norms_b, axis=-1)  # [H]
        new_sq_sum = mx.sum(norms_b * norms_b, axis=-1)  # [H]
        mx.eval(new_sum, new_sq_sum)

        if self._norm_sum is None:
            self._norm_sum = new_sum
            self._norm_sq_sum = new_sq_sum
        else:
            self._norm_sum = self._norm_sum + new_sum
            self._norm_sq_sum = self._norm_sq_sum + new_sq_sum
            mx.eval(self._norm_sum, self._norm_sq_sum)

        self._n_tokens += S

    def _running_head_importance(self) -> mx.array:
        """Per-head norm variance from running accumulators ([H] fp32)."""
        if self._norm_sum is None or self._n_tokens < 2:
            H = 0 if self._norm_sum is None else self._norm_sum.shape[0]
            return mx.zeros((H,), dtype=mx.float32)
        n = self._n_tokens
        mean = self._norm_sum / n
        return mx.maximum(self._norm_sq_sum / n - mean * mean, 0.0)

    def _recompute_head_bits(self, n_heads: int) -> None:
        """Recompute the per-head bit assignment from current statistics."""
        importance = self._running_head_importance()
        self._head_bits = allocate_head_bits(
            importance,
            target_avg_bits=self._target_avg_bits,
            allowed_bits=self._allowed_bits,
            n_heads=n_heads,
        )

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------
    def _quantize_per_head(self, keys: mx.array) -> mx.array:
        """Quantize keys [B, H, S, D] with each head at its assigned bit-width."""
        B, H, S, D = keys.shape
        assert self._head_bits is not None and len(self._head_bits) == H
        out_batches = []
        for b in range(B):
            out_heads = []
            for h in range(H):
                k_q = quantize_head(keys[b, h], self._head_bits[h], self._group_size)
                out_heads.append(k_q)
            out_batches.append(mx.stack(out_heads, axis=0))  # [H, S, D]
        return mx.stack(out_batches, axis=0)  # [B, H, S, D]

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape

        self._update_norm_accumulators(keys)
        self._recompute_head_bits(H)
        k_out = self._quantize_per_head(keys)

        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, values)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        n_groups = math.ceil(S / self._group_size)
        assert self._head_bits is not None
        for h in range(H):
            b = self._head_bits[h]
            code_bytes = math.ceil(S * D * b / 8)
            param_bytes = n_groups * D * 2 * 2  # scale + zero, fp16
            self._compressed_key_bytes += (code_bytes + param_bytes) * B
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def head_bits(self) -> list[int]:
        """Current per-head bit assignment ([H] ints), or [] before first update."""
        return list(self._head_bits) if self._head_bits is not None else []

    @property
    def assigned_avg_bits(self) -> float:
        """Actual average bits/element across heads (0.0 before first update)."""
        if not self._head_bits:
            return 0.0
        return sum(self._head_bits) / len(self._head_bits)

    @property
    def head_importance(self) -> list[float]:
        """Current per-head norm-variance importance ([H] floats)."""
        return self._running_head_importance().tolist()

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
    def target_avg_bits(self) -> float:
        return self._target_avg_bits

    @property
    def allowed_bits(self) -> list[int]:
        return list(self._allowed_bits)

    @property
    def group_size(self) -> int:
        return self._group_size


__all__ = ["AdaKVCache"]

"""SVDq KV cache wrapper — sub-2-bit key compression via offline SVD.

Inspired by "SVDq: Singular Value Decomposition-based KV Cache Quantization"
(arXiv:2502.15304, Feb 2025, unreviewed preprint).  Documented as
"SVDq-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Design:

  Prefill (first call where S > 1):
    1. Compute truncated SVD of the incoming key batch K ∈ R^{S×D}.
    2. Store the right singular vectors V [D, r] and mean key K̄ [D] as layer
       state.  These are O(D²) and negligible for long sequences.
    3. Project keys into latent space L = (K - K̄) @ V → [S, r].
    4. Apply mixed-precision group quantization to L (top-25% channels at
       4-bit, rest at 2-bit, ordered by singular value magnitude).
    5. Reconstruct fp16 keys for the downstream SDPA call.
    6. Accumulate latents in a growing list (quantized-then-dequantized, so
       the downstream cache sees fp16, consistent with all other wrappers).

  Decode (S == 1 or S < prefill threshold):
    1. Project new key into the already-stored V space.
    2. Quantize and reconstruct fp16.
    3. Pass through to the underlying mlx_lm KVCache.

  Values are left at fp16 throughout (the paper notes values have weak
  low-rank structure; compressing values is left to stacked wrappers).

Byte accounting:
  compressed_key_bytes  — latent storage at mixed-bit effective rate
  fp16_key_bytes        — what full fp16 would cost (for ratio computation)
  value_fp16_bytes      — values are always fp16 (reported separately)

  Effective key bit-width ≈ (r/D) * weighted_avg(hi_bit, lo_bit) where the
  weight is hi_fraction.  For r = 0.5D, hi_fraction = 0.25:
    effective ≈ 0.5 * (0.25*4 + 0.75*2) = 0.5 * 2.5 = 1.25 bits/element.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.svdq import (
    quantize_latents_mixed,
    reconstruct_keys,
    svd_compress_keys,
)


class SVDqKVCache(_MLXKVCache):
    """KV cache implementing SVDq sub-2-bit key compression.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``head_dim`` (D),
            ``svdq_rank`` (int | None — explicit rank; None → energy threshold),
            ``svdq_energy_threshold`` (float, default 0.95),
            ``svdq_hi_bit`` (int, default 4 — bits for top channels),
            ``svdq_lo_bit`` (int, default 2 — bits for remaining channels),
            ``svdq_hi_fraction`` (float, default 0.25),
            ``svdq_group_size`` (int, default 32).

    Notes:
        Does not expose ``.bits`` — mlx_lm's SDPA checks ``hasattr(cache, "bits")``
        to route to a quantized kernel; we keep that path clean.
        Exposes ``.assigned_avg_bits`` with the effective key bit-width.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._rank: Optional[int] = getattr(config, "svdq_rank", None)
        self._energy_threshold: float = float(getattr(config, "svdq_energy_threshold", 0.95))
        self._hi_bit: int = int(getattr(config, "svdq_hi_bit", 4))
        self._lo_bit: int = int(getattr(config, "svdq_lo_bit", 2))
        self._hi_fraction: float = float(getattr(config, "svdq_hi_fraction", 0.25))
        if not 0.0 <= self._hi_fraction <= 1.0:
            raise ValueError(f"svdq: svdq_hi_fraction must be in [0, 1], got {self._hi_fraction}")
        self._group_size: int = int(getattr(config, "svdq_group_size", 32))

        # SVD state — set on first prefill call
        self._V: Optional[mx.array] = None  # [D, r] fp32
        self._K_mean: Optional[mx.array] = None  # [D] fp32
        self._singular_values: Optional[mx.array] = None  # [r] fp32
        self._r: int = 0  # actual rank used

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0
        self._tokens_seen: int = 0

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------
    def _run_prefill_svd(self, keys: mx.array) -> mx.array:
        """Compute SVD on keys [B, H, S, D], store projection, return reconstructed keys."""
        B, H, S, D = keys.shape
        # Process head 0 of batch 0 for the SVD; apply the same V to all heads.
        # Keys across heads share the same D-dimensional space.
        k0 = keys[0, 0].astype(mx.float32)  # [S, D]
        L, V, K_mean, s_vals = svd_compress_keys(
            k0, rank=self._rank, energy_threshold=self._energy_threshold
        )
        self._V = V  # [D, r]
        self._K_mean = K_mean  # [D]
        self._singular_values = s_vals  # [r]
        self._r = int(V.shape[1])
        mx.eval(self._V, self._K_mean, self._singular_values)

        # Project and quantize all heads
        return self._project_quantize_reconstruct(keys)

    def _project_quantize_reconstruct(self, keys: mx.array) -> mx.array:
        """Project keys → latent → quantize → reconstruct for all [B, H, S, D]."""
        B, H, S, D = keys.shape
        V = self._V
        K_mean = self._K_mean
        sv = self._singular_values

        out_heads = []
        for b in range(B):
            out_batch = []
            for h in range(H):
                k_bh = keys[b, h].astype(mx.float32)  # [S, D]
                k_centered = k_bh - K_mean[None, :]
                L = k_centered @ V  # [S, r]
                L_q = quantize_latents_mixed(
                    L,
                    sv,
                    hi_bit=self._hi_bit,
                    lo_bit=self._lo_bit,
                    hi_fraction=self._hi_fraction,
                    group_size=self._group_size,
                )
                k_hat = reconstruct_keys(L_q, V, K_mean)  # [S, D] fp16
                out_batch.append(k_hat)
            out_heads.append(mx.stack(out_batch, axis=0))  # [H, S, D]
        return mx.stack(out_heads, axis=0)  # [B, H, S, D]

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape

        if self._V is None:
            # First call — run SVD on the incoming batch (prefill)
            k_out = self._run_prefill_svd(keys)
        else:
            # Subsequent calls — project into existing V
            k_out = self._project_quantize_reconstruct(keys)

        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, values)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        r = self._r if self._r > 0 else D
        n_hi = max(1, int(r * self._hi_fraction))
        n_lo = r - n_hi

        # Latent storage: n_hi channels at hi_bit + n_lo at lo_bit per token,
        # plus group-quant overhead (scale + zero per group, fp16).
        def _latent_bytes(n_tokens: int, n_ch: int, b: int) -> int:
            code_bytes = math.ceil(n_tokens * n_ch * b / 8)
            n_groups = math.ceil(n_tokens / self._group_size)
            param_bytes = n_groups * n_ch * 2 * 2  # scale + zero, fp16
            return (code_bytes + param_bytes) * H * B

        key_bytes = _latent_bytes(S, n_hi, self._hi_bit) + _latent_bytes(S, n_lo, self._lo_bit)
        # V [D, r] + K_mean [D] stored once — amortized over tokens seen
        projection_bytes = (D * r + D) * 4 * H * B  # fp32

        self._compressed_key_bytes += key_bytes + projection_bytes
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2
        self._tokens_seen += S

    # ------------------------------------------------------------------
    # Reporting
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
        """Effective key bit-width in the latent space."""
        if self._r == 0:
            return float(self._hi_bit)
        n_hi = max(1, int(self._r * self._hi_fraction))
        n_lo = self._r - n_hi
        weighted = (n_hi * self._hi_bit + n_lo * self._lo_bit) / self._r
        # Scale by r/D — latent dim is smaller than original
        return weighted * self._r / self._D

    @property
    def rank(self) -> int:
        """Actual SVD rank used after energy-threshold selection."""
        return self._r


__all__ = ["SVDqKVCache"]

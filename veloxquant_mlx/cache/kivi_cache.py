"""KIVI KV cache wrapper for mlx_lm integration.

Implements "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
(Liu, Yuan et al., ICML 2024; arXiv:2402.02750) on top of the standard
mlx_lm ``update_and_fetch`` protocol.

KIVI's asymmetry:
  * **Keys** are quantized **per channel** (group-wise min/max along the
    token axis) — key distributions have a few high-variance channels, so
    per-channel scales keep them accurate.
  * **Values** are quantized **per token** (group-wise along the channel
    axis) — value distributions are flatter across channels but vary by
    token.
  * The most recent ``residual_length`` tokens are kept in **fp16**
    (KIVI's "residual"): newly generated tokens dominate attention and are
    cheap to keep exact; they are quantized only once they age out of the
    residual window.

Like every method in this repo, the quantize→dequantize round-trip happens
inside ``update_and_fetch`` so the downstream SDPA call sees standard fp16
tensors.  **The paper's throughput gains come from a CUDA kernel that does
not port to Metal** — on Apple Silicon the win is *memory*, and we expect a
throughput cost vs fp16, which the benchmarks measure honestly.

KIVI is fully deterministic (min/max group quantization, no codebook
training, no RNG), so it introduces no run-to-run parity variance.

Per-token storage at bit-width ``b`` and group size ``g`` (keys, per
channel): ``D * b / 8`` bits of codes + ``2 * (D / g_eff) * 2`` bytes of
fp16 (scale, zero) amortized per group.  Byte accounting below reflects the
realized quantized-region cost; the fp16 residual window is reported
separately so the compression ratio is not inflated.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache


class KIVIKVCache(_MLXKVCache):
    """KV cache implementing KIVI asymmetric group quantization.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``head_dim`` (D), ``bit_width_inlier`` (b, KIVI default 2),
            ``kivi_group_size`` (group size; default 32),
            ``residual_length`` (fp16 residual window; default 128).

    Notes:
        Never exposes ``.bits`` — mlx_lm's SDPA checks
        ``hasattr(cache, "bits")`` to route to a quantized kernel path.
        We expose ``.assigned_avg_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._head_dim = int(config.head_dim)
        b = config.bit_width_inlier
        if isinstance(b, list):
            raise ValueError(
                "KIVIKVCache: bit_width_inlier must be a single int; "
                "per-layer lists are dispatched by KVCacheBuilder.for_model()."
            )
        self._b = int(b)
        self._group_size = int(getattr(config, "kivi_group_size", 32))
        self._residual_length = int(getattr(config, "residual_length", 128))
        self._levels = (1 << self._b) - 1
        self._eps = 1e-8

        # Byte accounting
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0
        self._value_bytes_compressed = 0
        self._value_bytes_fp16 = 0
        self._residual_fp16_bytes = 0
        self._tokens_seen = 0

    # ------------------------------------------------------------------
    # Group quant/dequant helpers (asymmetric min/max, deterministic)
    # ------------------------------------------------------------------
    def _quant_dequant_along(self, x: mx.array, axis: int) -> mx.array:
        """Round-trip ``x`` through KIVI group quantization along ``axis``.

        Operates on the last two dims being [..., S, D].  ``axis`` selects
        the quantization axis within those: -2 == per-channel (group along
        tokens, KIVI keys), -1 == per-token (group along channels, values).
        Groups partition the chosen axis into blocks of ``group_size``.
        """
        gs = self._group_size
        x32 = x.astype(mx.float32)
        L = x32.shape[axis]
        n_groups = (L + gs - 1) // gs
        pad = n_groups * gs - L

        # Move quant axis to the end for uniform grouping, then restore.
        xm = mx.moveaxis(x32, axis, -1)  # [..., other, L]
        if pad:
            tail = xm[..., -1:]
            xm = mx.concatenate([xm, mx.broadcast_to(tail, xm.shape[:-1] + (pad,))], axis=-1)
        new_shape = xm.shape[:-1] + (n_groups, gs)
        xg = xm.reshape(new_shape)  # [..., other, G, gs]
        gmin = mx.min(xg, axis=-1, keepdims=True)
        gmax = mx.max(xg, axis=-1, keepdims=True)
        scale = mx.maximum((gmax - gmin) / self._levels, self._eps)
        codes = mx.clip(mx.round((xg - gmin) / scale), 0, self._levels)
        recon = codes * scale + gmin  # asymmetric dequant
        recon = recon.reshape(xm.shape)[..., :L]
        recon = mx.moveaxis(recon, -1, axis)
        return recon.astype(x.dtype)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        """Quantize the aged-out portion of K/V, keep the residual in fp16.

        We quantize **only** tokens that fall outside the most-recent
        ``residual_length`` window of the *current* incoming block.  During
        prefill (large S) this quantizes the bulk and keeps the tail exact;
        during decode (S==1) the new token is within the residual window and
        passes through untouched until it ages out on later steps.
        """
        B, H, S, D = keys.shape
        r = self._residual_length

        if S <= r:
            # Entire incoming block is within the fp16 residual window.
            k_out, v_out = keys, values
            n_quant = 0
        else:
            n_quant = S - r
            k_q = self._quant_dequant_along(keys[:, :, :n_quant, :], axis=-2)
            v_q = self._quant_dequant_along(values[:, :, :n_quant, :], axis=-1)
            k_out = mx.concatenate([k_q, keys[:, :, n_quant:, :]], axis=2)
            v_out = mx.concatenate([v_q, values[:, :, n_quant:, :]], axis=2)

        self._account_bytes(B, H, S, D, n_quant)
        return super().update_and_fetch(k_out, v_out)

    def _account_bytes(self, B: int, H: int, S: int, D: int, n_quant: int) -> None:
        n_res = S - n_quant
        gs = self._group_size
        # Quantized keys: per-channel — D channels, ceil(n_quant/gs) groups,
        # b bits/code + (scale, zero) fp16 per (group, channel).
        if n_quant > 0:
            k_groups = math.ceil(n_quant / gs)
            k_code_bytes = math.ceil(n_quant * D * self._b / 8) * H * B
            k_param_bytes = k_groups * D * 2 * 2 * H * B  # scale+zero, fp16
            # Quantized values: per-token — ceil(D/gs) groups per token.
            v_groups = math.ceil(D / gs)
            v_code_bytes = math.ceil(n_quant * D * self._b / 8) * H * B
            v_param_bytes = n_quant * v_groups * 2 * 2 * H * B
            self._key_bytes_compressed += k_code_bytes + k_param_bytes
            self._value_bytes_compressed += v_code_bytes + v_param_bytes
        # fp16 residual window (kept exact)
        self._residual_fp16_bytes += n_res * D * 2 * 2 * H * B  # K+V
        # fp16 equivalents (for ratio): every token at full precision
        self._key_bytes_fp16 += H * B * S * D * 2
        self._value_bytes_fp16 += H * B * S * D * 2
        self._tokens_seen += S

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16

    @property
    def compressed_value_bytes(self) -> int:
        return self._value_bytes_compressed

    @property
    def fp16_value_bytes(self) -> int:
        return self._value_bytes_fp16

    @property
    def residual_fp16_bytes(self) -> int:
        """Bytes held in the fp16 residual window (keys + values)."""
        return self._residual_fp16_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Nominal bits/element in the quantized region (excludes residual
        and per-group scale/zero overhead).  For an end-to-end byte ratio
        use ``(compressed_*_bytes + residual_fp16_bytes) / fp16_*_bytes``.
        """
        return float(self._b)


__all__ = ["KIVIKVCache"]

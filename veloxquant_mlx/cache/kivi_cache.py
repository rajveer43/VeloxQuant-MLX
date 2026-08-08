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

        # Cumulative count of leading tokens (absolute sequence positions
        # [0, _n_quantized)) that have already been quantized in-place in
        # the parent cache's storage. Tokens age out of the fp16 residual
        # window based on true cumulative position, not this call's S.
        self._n_quantized = 0

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
        """Quantize whatever has aged out of the fp16 residual window.

        Tokens age out based on their **true cumulative position** in the
        sequence, not this call's own ``S``: after every call, the most
        recent ``residual_length`` tokens (of the *entire* history) stay
        fp16 and everything older is quantized. During decode (S==1) that
        means each new token starts in the residual window and gets
        quantized ``residual_length`` steps later, once it actually ages
        out — not never, as a per-call ``S <= residual_length`` check would
        imply.

        Implementation: store the incoming block via the parent cache
        first (so ``self.offset``/``self.keys``/``self.values`` reflect the
        full accumulated history), then quantize-in-place any newly-aged
        leading slice of that stored buffer that hasn't been quantized yet.
        """
        B, H, S, D = keys.shape
        r = self._residual_length
        k_all, v_all = super().update_and_fetch(keys, values)

        new_boundary = max(0, self.offset - r)
        n_quant_now = new_boundary - self._n_quantized
        if n_quant_now > 0:
            lo, hi = self._n_quantized, new_boundary
            k_q = self._quant_dequant_along(self.keys[:, :, lo:hi, :], axis=-2)
            v_q = self._quant_dequant_along(self.values[:, :, lo:hi, :], axis=-1)
            self.keys[:, :, lo:hi, :] = k_q
            self.values[:, :, lo:hi, :] = v_q
            self._n_quantized = new_boundary
            k_all, v_all = self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

        self._account_bytes(B, H, S, D, n_quant_now)
        return k_all, v_all

    def _account_bytes(self, B: int, H: int, S: int, D: int, n_quant_now: int) -> None:
        gs = self._group_size
        # Quantized keys: per-channel — D channels, ceil(n_quant/gs) groups,
        # b bits/code + (scale, zero) fp16 per (group, channel).
        if n_quant_now > 0:
            k_groups = math.ceil(n_quant_now / gs)
            k_code_bytes = math.ceil(n_quant_now * D * self._b / 8) * H * B
            k_param_bytes = k_groups * D * 2 * 2 * H * B  # scale+zero, fp16
            # Quantized values: per-token — ceil(D/gs) groups per token.
            v_groups = math.ceil(D / gs)
            v_code_bytes = math.ceil(n_quant_now * D * self._b / 8) * H * B
            v_param_bytes = n_quant_now * v_groups * 2 * 2 * H * B
            self._key_bytes_compressed += k_code_bytes + k_param_bytes
            self._value_bytes_compressed += v_code_bytes + v_param_bytes
        # fp16 residual window: current size (not cumulative) — this is
        # the live, still-fp16 tail, so it must plateau at residual_length
        # once the window fills, not grow with every call.
        n_res = self.offset - self._n_quantized
        self._residual_fp16_bytes = n_res * D * 2 * 2 * H * B  # K+V
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

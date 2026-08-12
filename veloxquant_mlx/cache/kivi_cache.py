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

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import kivi_group_quant_dequant as _kivi_group_quant_dequant


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

        # Resolve Metal acceleration flag (#164).  Three-state, matching
        # VecInferKVCache:
        #   None  → auto-detect (silent fallback)
        #   True  → require (raise if unavailable)
        #   False → forced pure-MLX path
        # The kernel is bit-identical to the MLX path, so this is a pure
        # performance knob — it can never change a numeric result.  Measured
        # 1.4x-2.1x on Apple M4 across both KIVI schemes; see the benchmark in
        # ``tests/metal/test_kivi_quant.py::test_kivi_quant_benchmark``.
        requested = getattr(config, "use_metal_kernels", None)
        available = metal_available()
        if requested is True and not available:
            raise RuntimeError(
                "KIVIKVCache: use_metal_kernels=True but Metal kernels "
                "are not available on this build of mlx."
            )
        self._use_metal: bool = bool(available if requested is None else requested) and available

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

        Dispatches to a fused Metal kernel when one is available (#164),
        falling back to the pure-MLX path below otherwise.  The two are
        **bit-identical**, not merely close — see
        ``tests/metal/test_kivi_quant.py`` — so enabling or disabling the
        kernel can never change a benchmark result or a cached value.
        """
        if self._use_metal:
            try:
                return _kivi_group_quant_dequant(
                    x,
                    axis=axis,
                    group_size=self._group_size,
                    levels=self._levels,
                    eps=self._eps,
                )
            except Exception:
                # Any kernel-side failure (unsupported dtype, compile error,
                # driver issue) degrades to the MLX path rather than taking
                # down generation.  Latch it off so we don't re-pay the
                # failure on every subsequent flush.
                self._use_metal = False

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
    # Residual-buffer boundary (Algorithm 1)
    # ------------------------------------------------------------------
    def _quantization_boundary(self) -> int:
        """Absolute position up to which tokens may be quantized *now*.

        Paper Algorithm 1 buffers arriving keys in ``X_Kr`` and flushes only
        once that buffer holds ``R`` tokens, quantizing the whole ``R x D``
        block at once.  The effect that matters numerically: because keys
        are quantized **per channel** (the group runs along the *token*
        axis), a flush must carry at least ``group_size`` tokens or every
        group degenerates to a single element — ``min == max``, ``scale``
        collapses to ``eps``, and the "quantized" round-trip is bit-exact
        (#162).

        So the boundary is snapped **down to a multiple of ``group_size``**:
        tokens wait in the fp16 residual until they can form whole groups.
        This also makes the result independent of how the sequence was
        chunked — one-shot prefill and token-by-token decode quantize
        exactly the same positions with exactly the same group boundaries.
        """
        aged = max(0, self.offset - self._residual_length)
        return (aged // self._group_size) * self._group_size

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        """Quantize whatever has aged out of the fp16 residual window.

        Tokens age out based on their **true cumulative position** in the
        sequence, not this call's own ``S``: after every call, the most
        recent ``residual_length`` tokens (of the *entire* history) stay
        fp16 and everything older becomes eligible.  Eligible tokens are
        then flushed in ``group_size``-aligned blocks per
        :meth:`_quantization_boundary`, so per-channel key groups always
        span a full ``group_size`` tokens.

        Implementation: store the incoming block via the parent cache
        first (so ``self.offset``/``self.keys``/``self.values`` reflect the
        full accumulated history), then quantize-in-place any newly-aged
        leading slice of that stored buffer that hasn't been quantized yet.
        """
        B, H, S, D = keys.shape
        k_all, v_all = super().update_and_fetch(keys, values)

        new_boundary = self._quantization_boundary()
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
        # Quantized keys: per-channel — D channels, n_quant/gs groups, b
        # bits/code + (scale, zero) fp16 per (group, channel).  Flushes are
        # group_size-aligned (see _quantization_boundary), so this divides
        # evenly and no group is charged for tokens it doesn't hold.
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
        """Bytes held in fp16 and not yet quantized (keys + values).

        This is the ``residual_length`` window **plus** any partial group
        still waiting for enough tokens to form a whole ``group_size``
        block, so it is bounded by ``residual_length + group_size - 1``
        tokens rather than exactly ``residual_length``.
        """
        return self._residual_fp16_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Nominal bits/element in the quantized region (excludes residual
        and per-group scale/zero overhead).  For an end-to-end byte ratio
        use ``(compressed_*_bytes + residual_fp16_bytes) / fp16_*_bytes``.
        """
        return float(self._b)

    @property
    def effective_compression_ratio(self) -> float:
        """End-to-end KV byte ratio vs fp16, residual window included.

        This is the honest number: quantized codes + per-group (scale, zero)
        + the still-fp16 residual tail, against the fp16 cost of every token
        seen.  Reporting ``compressed_*`` alone overstates the win because
        it silently omits the fp16 residual (#162).
        """
        total_fp16 = self._key_bytes_fp16 + self._value_bytes_fp16
        if total_fp16 == 0:
            return 1.0
        total = (
            self._key_bytes_compressed + self._value_bytes_compressed + self._residual_fp16_bytes
        )
        return total_fp16 / total if total else 1.0


__all__ = ["KIVIKVCache"]

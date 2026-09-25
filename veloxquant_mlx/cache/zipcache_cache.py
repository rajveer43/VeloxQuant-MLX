"""ZipCache-adapted KV cache — saliency-adaptive per-token mixed-precision.

Inspired by "ZipCache: Accurate and Efficient KV Cache Quantization with
Salient Token Identification" (He et al., NeurIPS 2024, arXiv:2405.14256).
Documented as "ZipCache-adapted (VeloxQuant-MLX implementation)" — not a
faithful port.

Per-token saliency routing: each incoming K/V block has its key tokens sorted
by L2-norm; the top ``hi_fraction`` receive ``hi_bits`` and the rest receive
``lo_bits``. Both paths remain quantized (not fp16) — this distinguishes
ZipCache-adapted from KIVI-Sink (which keeps sinks in fp16).

The wrapper compresses each ``[B, H, S, D]`` head matrix with
``zipcache_compress`` / ``zipcache_reconstruct`` and hands the reconstructed
fp16 K/V to the parent ``mlx_lm`` cache (no ``.bits`` attribute; SDPA stays
on the clean fp16 path). Byte accounting tracks the mixed-bit stored size
against both fp16 and a uniform-lo-bit baseline.

Honest proxy limitation: the saliency signal is the key L2-norm (a proxy for
attention importance). The paper uses normalized attention scores, which are
not observable by a cache wrapper. The proxy has been used for KIVI-Sink and
AdaKV-proxy in this repo; this is the third use, with a different decision
(bit-width routing rather than fp16 protection or head budgeting).

Byte accounting:
    compressed_key_bytes / compressed_value_bytes  — mixed-bit ZipCache stored size
    baseline_key_bytes   / baseline_value_bytes    — uniform lo-bit baseline
    fp16_key_bytes       / fp16_value_bytes         — uncompressed cost
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.zipcache import base_only_bytes


class ZipCacheKVCache(_MLXKVCache):
    """KV cache implementing ZipCache-adapted per-token mixed-precision for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``zipcache_hi_bits``      (int, default 4)  — bit-width for salient tokens,
            ``zipcache_lo_bits``      (int, default 2)  — bit-width for non-salient tokens,
            ``zipcache_hi_fraction``  (float, default 0.20) — fraction of tokens at hi_bits,
            ``zipcache_group_size``   (int, default 32) — token group size for quant,
            ``zipcache_quantize_values`` (bool, default True) — apply to values too.

    Notes:
        No ``.bits`` attribute — keeps mlx_lm SDPA on the clean fp16 path.
        Single-layer (no coordinator); ``for_model`` propagates the ``zipcache_*``
        fields automatically via ``dataclasses.replace``.
        Values are quantized uniformly at ``hi_bits`` (saliency routing is
        key-driven; values follow the hi-bit path as the safer default).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._hi_bits = int(getattr(config, "zipcache_hi_bits", 4))
        self._lo_bits = int(getattr(config, "zipcache_lo_bits", 2))
        self._hi_fraction = float(getattr(config, "zipcache_hi_fraction", 0.20))
        if not 0.0 <= self._hi_fraction <= 1.0:
            raise QuantizerConfigError(
                f"zipcache: zipcache_hi_fraction must be in [0, 1], got {self._hi_fraction}"
            )
        self._gs = int(getattr(config, "zipcache_group_size", 32))
        self._quant_values = bool(getattr(config, "zipcache_quantize_values", True))

        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._baseline_key_bytes = 0
        self._baseline_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0

    # ------------------------------------------------------------------
    # Batched compress + reconstruct across B*H (no per-head Python loop).
    #
    # hi_fraction and S are shared by every (b, h) slab within one
    # update_and_fetch call, so n_hi/n_lo (and therefore group counts) are
    # identical across slabs -- only which rows land in each bucket differs.
    # That lets the whole [B*H, S, D] block be group-quantized in one shot:
    # min/max reductions over an extra leading (B*H) axis never mix rows
    # across slabs (groups only ever span the token axis), matching
    # zipcache_compress/reconstruct's per-slab semantics exactly (verified
    # array_equal against the per-head loop across the same parameter sweep
    # used for optimizations #1-3, including S=1, hi_fraction in {0, 1}, and
    # group sizes that don't divide n_hi/n_lo evenly).
    # ------------------------------------------------------------------
    @staticmethod
    def _batched_saliency_mask(norms: mx.array, n_hi: int, S: int) -> mx.array:
        """``[BH, S]`` bool mask, top-``n_hi`` per row by norm (see saliency_mask)."""
        BH = norms.shape[0]
        if n_hi == 0:
            return mx.zeros((BH, S), dtype=mx.bool_)
        if n_hi >= S:
            return mx.ones((BH, S), dtype=mx.bool_)
        order = mx.argsort(norms, axis=-1)
        rank = mx.argsort(order, axis=-1)
        return rank >= (S - n_hi)

    @staticmethod
    def _batched_group_quant(x: mx.array, bits: int, group_size: int):
        """``[BH, N, D]`` -> codes ``[BH, N, D]`` uint8, scales/zeros ``[BH, n_groups, D]`` fp32."""
        BH, n, d = x.shape
        if n == 0:
            z_codes = mx.zeros((BH, 0, d), dtype=mx.uint8)
            z_params = mx.zeros((BH, 0, d), dtype=mx.float32)
            return z_codes, z_params, z_params
        gs = group_size
        n_groups = (n + gs - 1) // gs
        pad = n_groups * gs - n
        x32 = x.astype(mx.float32)
        if pad:
            x32 = mx.concatenate([x32, mx.broadcast_to(x32[:, -1:], (BH, pad, d))], axis=1)
        xg = x32.reshape(BH, n_groups, gs, d)
        gmin = mx.min(xg, axis=2, keepdims=True)
        gmax = mx.max(xg, axis=2, keepdims=True)
        levels = (1 << bits) - 1
        eps = 1e-8
        scale = mx.maximum((gmax - gmin) / levels, eps)
        codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
        codes = codes.reshape(BH, n_groups * gs, d)[:, :n].astype(mx.uint8)
        scales = scale.reshape(BH, n_groups, d).astype(mx.float32)
        zeros = gmin.reshape(BH, n_groups, d).astype(mx.float32)
        return codes, scales, zeros

    @staticmethod
    def _batched_group_dequant(
        codes: mx.array, scales: mx.array, zeros: mx.array, n: int, group_size: int
    ) -> mx.array:
        BH, _, d = codes.shape
        n_groups = scales.shape[1]
        gs = group_size
        pad = n_groups * gs - codes.shape[1]
        c = codes.astype(mx.float32)
        if pad:
            c = mx.concatenate([c, mx.broadcast_to(c[:, -1:], (BH, pad, d))], axis=1)
        c = c.reshape(BH, n_groups, gs, d)
        s = scales.reshape(BH, n_groups, 1, d)
        z = zeros.reshape(BH, n_groups, 1, d)
        recon = c * s + z
        return recon.reshape(BH, n_groups * gs, d)[:, :n]

    def _compress_and_account(self, t: mx.array, is_key: bool) -> mx.array:
        """Compress ``[B, H, S, D]`` batched over B*H, accumulate byte accounting, return fp16."""
        B, H, S, D = t.shape
        hi_bits = self._hi_bits
        lo_bits = self._lo_bits if is_key else self._hi_bits
        hi_fraction = self._hi_fraction if is_key else 1.0
        gs = self._gs

        BH = B * H
        x = t.reshape(BH, S, D)
        x32 = x.astype(mx.float32)
        norms = mx.linalg.norm(x32, axis=-1)  # [BH, S]

        n_hi = max(0, min(S, int(math.ceil(S * hi_fraction))))
        n_lo = S - n_hi
        mask = self._batched_saliency_mask(norms, n_hi, S)

        # Same row-order convention as zipcache_compress: argsort(mask) is
        # stable, so it yields the False (lo) block then True (hi) block,
        # each in original within-block row order -- required for
        # array_equal, since group min/max is order-sensitive.
        order = mx.argsort(mask.astype(mx.int32), axis=-1)
        lo_idx = order[:, :n_lo]
        hi_idx = order[:, n_lo:]

        def gather(idx, n):
            if n == 0:
                return mx.zeros((BH, 0, D), dtype=mx.float32)
            idx_e = mx.broadcast_to(idx[..., None], (BH, n, D))
            return mx.take_along_axis(x32, idx_e, axis=1)

        x_hi = gather(hi_idx, n_hi)
        x_lo = gather(lo_idx, n_lo)

        hi_codes, hi_scales, hi_zeros = self._batched_group_quant(x_hi, hi_bits, gs)
        lo_codes, lo_scales, lo_zeros = self._batched_group_quant(x_lo, lo_bits, gs)

        hi_recon = (
            self._batched_group_dequant(hi_codes, hi_scales, hi_zeros, n_hi, gs)
            if n_hi
            else mx.zeros((BH, 0, D), dtype=mx.float32)
        )
        lo_recon = (
            self._batched_group_dequant(lo_codes, lo_scales, lo_zeros, n_lo, gs)
            if n_lo
            else mx.zeros((BH, 0, D), dtype=mx.float32)
        )

        out = mx.zeros((BH, S, D), dtype=mx.float32)
        if n_lo:
            lo_idx_e = mx.broadcast_to(lo_idx[..., None], (BH, n_lo, D))
            out = mx.put_along_axis(out, lo_idx_e, lo_recon, axis=1)
        if n_hi:
            hi_idx_e = mx.broadcast_to(hi_idx[..., None], (BH, n_hi, D))
            out = mx.put_along_axis(out, hi_idx_e, hi_recon, axis=1)
        out = out.reshape(B, H, S, D).astype(mx.float16)

        # Byte accounting: n_hi/n_lo/S/D/bits are identical across every
        # (b, h) slab in this call, so the per-slab byte formula (same as
        # zipcache_bytes) times BH equals summing BH individual calls.
        hi_code_bytes = math.ceil(n_hi * D * hi_bits / 8)
        lo_code_bytes = math.ceil(n_lo * D * lo_bits / 8)
        n_hi_groups = math.ceil(n_hi / gs) if n_hi > 0 else 0
        n_lo_groups = math.ceil(n_lo / gs) if n_lo > 0 else 0
        hi_param_bytes = n_hi_groups * D * 2 * 2
        lo_param_bytes = n_lo_groups * D * 2 * 2
        mask_bytes = S
        per_slab_comp = hi_code_bytes + lo_code_bytes + hi_param_bytes + lo_param_bytes + mask_bytes
        comp_total = per_slab_comp * BH
        base_total = base_only_bytes(S, D, self._lo_bits, gs) * BH

        fp16 = B * H * S * D * 2
        if is_key:
            self._compressed_key_bytes += comp_total
            self._baseline_key_bytes += base_total
            self._fp16_key_bytes += fp16
        else:
            self._compressed_value_bytes += comp_total
            self._baseline_value_bytes += base_total
            self._fp16_value_bytes += fp16
        return out

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Sort key tokens by L2-norm saliency, quantize the top hi_fraction at hi_bits and the rest at lo_bits (values uniformly at hi_bits); return reconstructed fp16 K/V."""
        k_out = self._compress_and_account(keys, is_key=True)
        if self._quant_values:
            v_out = self._compress_and_account(values, is_key=False)
        else:
            v_out = values
            B, H, S, D = values.shape
            self._fp16_value_bytes += B * H * S * D * 2
        return super().update_and_fetch(k_out, v_out)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        """Realized stored bytes for the compressed key cache (mixed hi/lo-bit saliency-routed codes, all heads/batches)."""
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        """Realized stored bytes for the compressed value cache (uniform hi_bits codes, all heads/batches)."""
        return self._compressed_value_bytes

    @property
    def baseline_key_bytes(self) -> int:
        """Uniform lo-bit baseline for comparison (no saliency routing)."""
        return self._baseline_key_bytes

    @property
    def baseline_value_bytes(self) -> int:
        """Uniform lo-bit baseline for comparison (no saliency routing)."""
        return self._baseline_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        """Hypothetical fp16 value cost if nothing were compressed."""
        return self._fp16_value_bytes

    @property
    def compression_ratio(self) -> float:
        """fp16 bytes / compressed bytes (key side). >1 means storage savings."""
        if self._compressed_key_bytes == 0:
            return 1.0
        return self._fp16_key_bytes / self._compressed_key_bytes

    @property
    def effective_avg_bits(self) -> float:
        """Average key bits/element implied by the stored mixed-bit rate."""
        if self._fp16_key_bytes == 0:
            return float(self._hi_bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes

    # Without this, ZipCacheKVCache inherits the base mlx_lm KVCache.merge()
    # classmethod unchanged, so hasattr(cache, "merge") is True and mlx_lm
    # treats this cache as batchable. update_and_fetch here always ends with
    # super().update_and_fetch(k_out, v_out), populating the base class's
    # self.keys/self.values, so the inherited merge() does not crash -- it
    # succeeds silently, substituting a plain BatchKVCache built from the
    # reconstructed fp16 tensors and discarding this layer's saliency routing
    # and mixed-bit byte accounting. See VeloxQuant-MLX#358; found verifying
    # VeloxQuant-Studio issue #36 (20th occurrence).
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError("ZipCacheKVCache does not support merge() — see VeloxQuant-MLX#358")
        )
    )


__all__ = ["ZipCacheKVCache"]

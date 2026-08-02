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

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.zipcache import (
    base_only_bytes,
    zipcache_bytes,
    zipcache_compress,
    zipcache_reconstruct,
)


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
            raise ValueError(
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
    def _compress_head(self, mat: mx.array, is_key: bool) -> tuple[mx.array, int, int]:
        """Compress ``[S, D]`` per head. Returns (fp16 recon, comp_bytes, base_bytes)."""
        S, D = mat.shape
        if is_key:
            state = zipcache_compress(
                mat,
                hi_bits=self._hi_bits,
                lo_bits=self._lo_bits,
                hi_fraction=self._hi_fraction,
                group_size=self._gs,
            )
            recon = zipcache_reconstruct(state)
            comp = zipcache_bytes(state, self._gs)
        else:
            # Values: uniform hi_bits (no saliency routing — value norms are
            # less correlated with attention importance than key norms)
            state = zipcache_compress(
                mat,
                hi_bits=self._hi_bits,
                lo_bits=self._hi_bits,  # uniform at hi_bits for values
                hi_fraction=1.0,
                group_size=self._gs,
            )
            recon = zipcache_reconstruct(state)
            comp = zipcache_bytes(state, self._gs)
        base = base_only_bytes(S, D, self._lo_bits, self._gs)
        return recon, comp, base

    def _compress_and_account(self, t: mx.array, is_key: bool) -> mx.array:
        """Compress ``[B, H, S, D]`` per head, accumulate byte accounting, return fp16."""
        B, H, S, D = t.shape
        recon_b = []
        comp_total = 0
        base_total = 0
        for b in range(B):
            recon_h = []
            for h in range(H):
                mat = t[b, h]  # [S, D]
                recon, comp, base = self._compress_head(mat, is_key)
                recon_h.append(recon)
                comp_total += comp
                base_total += base
            recon_b.append(mx.stack(recon_h, axis=0))
        out = mx.stack(recon_b, axis=0)
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
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def baseline_key_bytes(self) -> int:
        """Uniform lo-bit baseline for comparison (no saliency routing)."""
        return self._baseline_key_bytes

    @property
    def baseline_value_bytes(self) -> int:
        return self._baseline_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
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


__all__ = ["ZipCacheKVCache"]

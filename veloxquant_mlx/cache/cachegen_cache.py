"""CacheGen KV cache wrapper — entropy-coded byte accounting over group quant.

Inspired by "CacheGen: KV Cache Compression and Streaming for Fast LLM Serving"
(Liu et al., SIGCOMM 2024, arXiv:2310.07240). Documented as "CacheGen-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

The reconstructed K/V the model sees are identical to plain asymmetric min/max
group quant (KIVI-style). CacheGen's contribution here is the **byte
accounting**: it measures the Shannon entropy of the token-delta-transformed
code stream and reports ``compressed_*_bytes`` from that, modelling an ideal
arithmetic coder. This captures the token-wise-locality storage win without
shipping a serial range codec that would bottleneck MLX decode.

Per-layer bit-width is configurable (``cachegen_bits``); ``KVCacheBuilder.for_model``
applies a real layer-wise schedule (deeper layers get fewer bits — CacheGen's
layer-sensitivity observation, §5.1.2) via ``layer_group_bits``, injected per layer
as ``cachegen_resolved_bits`` (see ``KVCacheBuilder._build_cachegen``).

Byte accounting:
    compressed_key_bytes / compressed_value_bytes — entropy-coded estimate
    fixed_width_key_bytes / fixed_width_value_bytes — naive packed baseline
    fp16_key_bytes / fp16_value_bytes — uncompressed cost for the ratio
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.cachegen import (
    dequant_codes,
    entropy_coded_bytes,
    fixed_width_bytes,
    quantize_to_codes,
)


class CacheGenKVCache(_MLXKVCache):
    """KV cache implementing CacheGen entropy-coded compression for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``cachegen_bits``            (int, default 4) — base/uniform bit-width.
            ``cachegen_resolved_bits``   (int or None) — per-layer bit-width injected
                by ``KVCacheBuilder.for_model`` (§5.1.2 layer-wise schedule); falls
                back to ``cachegen_bits`` when None (uniform behaviour).
            ``cachegen_group_size``      (int, default 32),
            ``cachegen_use_delta``       (bool, default True — token-delta transform),
            ``cachegen_per_channel``     (bool, default True — group entropy estimate
                by channel per §5.1.3 instead of pooling all channels).

    Notes:
        No ``.bits`` attribute — keeps mlx_lm SDPA on the clean fp16 path.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        resolved = getattr(config, "cachegen_resolved_bits", None)
        self._bits = (
            int(resolved) if resolved is not None else int(getattr(config, "cachegen_bits", 4))
        )
        self._gs = int(getattr(config, "cachegen_group_size", 32))
        self._use_delta = bool(getattr(config, "cachegen_use_delta", True))
        self._per_channel = bool(getattr(config, "cachegen_per_channel", True))

        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._fixed_width_key_bytes = 0
        self._fixed_width_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0

    # ------------------------------------------------------------------
    def _quant_and_account(self, t: mx.array, is_key: bool) -> mx.array:
        """Quantize [B, H, S, D] per head, accumulate byte accounting, return fp16."""
        B, H, S, D = t.shape
        recon_b = []
        comp = 0
        fixed = 0
        for b in range(B):
            recon_h = []
            for h in range(H):
                stream = quantize_to_codes(t[b, h], self._bits, self._gs)
                recon_h.append(dequant_codes(stream))
                comp += entropy_coded_bytes(
                    stream, use_delta=self._use_delta, per_channel=self._per_channel
                )
                fixed += fixed_width_bytes(stream)
            recon_b.append(mx.stack(recon_h, axis=0))
        out = mx.stack(recon_b, axis=0)

        fp16 = B * H * S * D * 2
        if is_key:
            self._compressed_key_bytes += comp
            self._fixed_width_key_bytes += fixed
            self._fp16_key_bytes += fp16
        else:
            self._compressed_value_bytes += comp
            self._fixed_width_value_bytes += fixed
            self._fp16_value_bytes += fp16
        return out

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        k_out = self._quant_and_account(keys, is_key=True)
        v_out = self._quant_and_account(values, is_key=False)
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
    def fixed_width_key_bytes(self) -> int:
        return self._fixed_width_key_bytes

    @property
    def fixed_width_value_bytes(self) -> int:
        return self._fixed_width_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def entropy_savings(self) -> float:
        """Fraction of the fixed-width code size saved by entropy coding (key side)."""
        if self._fixed_width_key_bytes == 0:
            return 0.0
        return 1.0 - self._compressed_key_bytes / self._fixed_width_key_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Effective key bit-width after entropy coding (vs fp16=16)."""
        if self._fp16_key_bytes == 0:
            return float(self._bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["CacheGenKVCache"]

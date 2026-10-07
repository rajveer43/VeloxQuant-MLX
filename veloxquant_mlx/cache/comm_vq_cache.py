"""CommVQ KV cache wrapper: product-VQ key compression over an ``mlx_lm`` cache.

Wraps :class:`~veloxquant_mlx.quantizers.comm_vq.CommVQQuantizer` so it is
reachable through ``KVCacheFactory`` and usable under ``mlx_lm``. Honest scope
(see #756):

* The quantizer is plain *product* VQ: each of ``n_codebooks`` sub-codebooks
  owns a disjoint slice of the head dimension. It is not the paper's additive
  scheme and enforces no RoPE commutativity.
* ``mlx_lm`` applies RoPE *before* the cache sees keys, so this wrapper encodes
  and decodes the **post-RoPE** keys directly (position-agnostic product VQ)
  and never re-applies RoPE.
* The codebooks are trained on the first ``update_and_fetch`` call, so that call
  should be the prefill. A tiny first call trains a poor codebook.
* Only keys are compressed; values stay fp16. The reconstructed fp16 keys are
  handed to the parent cache, so SDPA stays on the clean fp16 path and
  attend-time memory is unchanged: only the *reported* stored bytes shrink.

Byte accounting:
    compressed_key_bytes -- index bytes (``n_codebooks`` per key vector)
    fp16_key_bytes       -- uncompressed key cost, for the ratio
Codebook bytes are excluded from both.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer


class CommVQKVCache(_MLXKVCache):
    """KV cache that stores keys as product-VQ codes (reconstructed to fp16 on fetch).

    Args:
        config: :class:`KVCacheConfig`. Fields consumed: ``head_dim``,
            ``comm_vq_bits`` (default 8), ``comm_vq_n_codebooks`` (default 4),
            ``comm_vq_n_em_iters`` (default 20), ``seed``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        d = int(config.head_dim)
        self._quantizer = CommVQQuantizer(
            d=d,
            b=int(getattr(config, "comm_vq_bits", 8)),
            n_codebooks=int(getattr(config, "comm_vq_n_codebooks", 4)),
            seed=int(getattr(config, "seed", 42)),
            n_em_iters=int(getattr(config, "comm_vq_n_em_iters", 20)),
        )
        self._d = d
        self._compressed_key_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize keys with product VQ, hand the reconstruction to the parent cache."""
        B, H, S, D = keys.shape
        flat = keys.reshape(B * H * S, D)
        if not self._quantizer.trained:
            self._quantizer.fit(flat)
        idx = self._quantizer._encode_batch(flat)
        recon = self._quantizer._decode_batch(idx).reshape(B, H, S, D).astype(keys.dtype)

        n = B * H * S
        self._compressed_key_bytes += n * self._quantizer._n_cb
        self._fp16_key_bytes += n * D * 2
        self._fp16_value_bytes += int(values.size) * 2
        return super().update_and_fetch(recon, values)

    def trim(self, n: int) -> int:
        """Drop the ``n`` most-recent tokens and scale the byte counters to match."""
        old_offset = self.offset
        trimmed = super().trim(n)
        if trimmed and old_offset:
            keep = self.offset / old_offset
            self._compressed_key_bytes = int(self._compressed_key_bytes * keep)
            self._fp16_key_bytes = int(self._fp16_key_bytes * keep)
            self._fp16_value_bytes = int(self._fp16_value_bytes * keep)
        return trimmed

    @property
    def compressed_key_bytes(self) -> int:
        """Index bytes for all cached keys (codebook bytes excluded)."""
        return self._compressed_key_bytes

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        """Value bytes (kept uncompressed)."""
        return self._fp16_value_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Effective key bit-width (index bits per element vs fp16=16)."""
        if self._fp16_key_bytes == 0:
            return float(self._quantizer._b * self._quantizer._n_cb) / self._d
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["CommVQKVCache"]

"""TurboQuantRVQ KV cache wrapper for production mlx_lm integration.

This wraps the existing :class:`TurboQuantRVQ` quantizer in the same
``update_and_fetch`` cache protocol that mlx_lm expects. Unlike
:class:`TurboQuantKVCache` which uses its own bit-packed append/attend
storage, this class delegates storage to the underlying mlx_lm ``KVCache``
after dequantizing — matching the pattern that benchmark wrappers have
used since v0.3.0.

The advantage of this design: it slots into ``mlx_lm.generate(cache=...)``
with no monkey-patching, so users get RVQ compression by simply selecting
``method="turboquant_rvq"`` in their :class:`KVCacheConfig`.

Per-vector storage at bit-width ``b`` and head-dim ``d``:
    ``ceil(d * 2 * b / 8) + 2`` bytes
    (two index sets at b bits each, plus an fp16 per-vector norm)

For ``d=128, b=1`` this is 34 bytes/vector vs 256 bytes fp16 → 7.5× compression.
For ``d=128, b=2`` this is 66 bytes/vector vs 256 bytes fp16 → 3.9× compression.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ


class TurboQuantRVQKVCache(_MLXKVCache):
    """KV cache backed by TurboQuantRVQ two-pass residual quantization.

    Compresses keys via :class:`TurboQuantRVQ`; values pass through unchanged.

    Args:
        config: :class:`KVCacheConfig` (consumes ``head_dim``,
            ``bit_width_inlier``, ``seed``).

    Notes:
        Reports byte accounting via :pyattr:`compressed_key_bytes` and
        :pyattr:`fp16_key_bytes` so users can compute the realized compression
        ratio after a benchmark run.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        b = config.bit_width_inlier
        if isinstance(b, list):
            raise TypeError(
                "TurboQuantRVQKVCache cannot accept list-form bit_width_inlier; "
                "this list-form is consumed by KVCacheBuilder.for_model() which "
                "dispatches to the per-layer factory."
            )
        self._head_dim = int(config.head_dim)
        self._bits = int(b)
        # NOTE: must be a private attribute. mlx_lm.scaled_dot_product_attention
        # checks `hasattr(cache, "bits")` to route to its quantized SDPA kernel
        # (which expects a different cache layout). Exposing our `b` as a
        # public `.bits` property would silently break attention on some models.
        self._quantizer = TurboQuantRVQ(
            d=self._head_dim,
            b=self._bits,
            seed=int(config.seed),
            use_hadamard=True,
        )
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        kdtype = keys.dtype

        k_flat = keys.reshape(-1, D)
        # fp32 norm computation preserves bfloat16 dynamic range
        norms = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True).astype(kdtype)
        safe = mx.maximum(norms, mx.array(1e-4, dtype=kdtype))
        k_unit = (k_flat / safe).astype(mx.float16)

        ev = self._quantizer.encode(k_unit)
        k_hat_u = self._quantizer.decode(ev)
        k_dequant = (k_hat_u.astype(kdtype) * safe).reshape(B, H, S, D)

        # Byte accounting: two b-bit index sets per dim + fp16 norm
        per_tok = (math.ceil(self._head_dim * 2 * self._bits / 8) + 2) * H * B
        self._key_bytes_compressed += per_tok * S
        self._key_bytes_fp16 += H * B * S * self._head_dim * 2

        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        """Cumulative compressed key bytes since construction."""
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        """Cumulative fp16-equivalent key bytes (for ratio computation)."""
        return self._key_bytes_fp16

    @property
    def assigned_bits(self) -> int:
        """Bit-width used by this layer's quantizer.

        Named ``assigned_bits`` (not ``bits``) to avoid colliding with
        mlx_lm's quantized-SDPA check, which uses ``hasattr(cache, "bits")``
        to switch attention kernel paths.
        """
        return self._bits

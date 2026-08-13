from __future__ import annotations

from typing import Any

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.core.constants import INT8_MAX
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.dsa.ring_buffer import RingBuffer
from veloxquant_mlx.quantizers.qjl import QJLQuantizer


class QJLKVCache(KVCache):
    """Minimal KV cache using pure 1-bit QJL for key compression.

    Args:
        config: KVCacheConfig instance.
    """

    def __init__(self, config: Any) -> None:
        import mlx.core as mx

        d = config.head_dim
        m = config.jl_dim if config.jl_dim is not None else d
        seed = config.seed
        store = config.store

        self._key_quantizer = QJLQuantizer(d=d, m=m, seed=seed, store=store)

        capacity = config.capacity or 1_000_000
        self._k_signs: RingBuffer = RingBuffer(capacity)
        self._k_norms: RingBuffer = RingBuffer(capacity)
        self._v_cache: RingBuffer = RingBuffer(capacity)
        self._v_scales: RingBuffer = RingBuffer(capacity)

        self._d = d
        self._m = m
        self._n_tokens: int = 0

    def append_key(self, k: Any) -> None:
        """Encode and store a key vector.

        Args:
            k: Key vector, shape (d,), fp16.
        """
        if k.ndim == 1:
            k = k[None]
        ev = self._key_quantizer.encode(k)
        self._k_signs.append(ev.signs[0])
        self._k_norms.append(ev.norm[0])
        self._n_tokens += 1

    def append_value(self, v: Any) -> None:
        """Quantize and store a value vector.

        Args:
            v: Value vector, shape (d,), fp16.
        """
        import mlx.core as mx

        if v.ndim > 1:
            v = v.reshape(-1)
        abs_max = float(mx.max(mx.abs(v)))
        scale = max(abs_max / INT8_MAX, 1e-8)
        v_int8 = mx.clip(mx.round(v / scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
        self._v_cache.append(v_int8)
        self._v_scales.append(mx.array(scale, dtype=mx.float16))

    def attend(self, q: Any) -> Any:
        """Compute attention output.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """
        import mlx.core as mx

        n = len(self._k_signs)
        if n == 0:
            return mx.zeros((self._d,), dtype=mx.float16)

        k_signs = mx.stack([self._k_signs[i] for i in range(n)])  # (n, m)
        k_norms = mx.stack([self._k_norms[i] for i in range(n)])  # (n,)

        ev = EncodedVector(
            quantizer_type="qjl",
            batch_size=n,
            dim=self._d,
            signs=k_signs,
            norm=k_norms,
        )

        scores_raw = self._key_quantizer.estimate_inner_product(q, ev)
        scale = float(mx.sqrt(mx.array(float(self._d))))
        scores = mx.softmax(scores_raw / scale, axis=0)

        v_scales = mx.stack([self._v_scales[i] for i in range(n)])
        v_int8 = mx.stack([self._v_cache[i] for i in range(n)])
        v_hat = v_int8.astype(mx.float16) * v_scales[:, None]

        return (scores[:, None] * v_hat).sum(axis=0)

    def memory_bytes(self) -> int:
        """Estimate memory footprint."""
        n = len(self._k_signs)
        if n == 0:
            return 0
        sign_bytes = n * self._m
        norm_bytes = n * 2
        v_bytes = n * (self._d + 2)
        return sign_bytes + norm_bytes + v_bytes

    def __len__(self) -> int:
        return len(self._k_signs)

    def __repr__(self) -> str:
        return f"QJLKVCache(d={self._d}, m={self._m}, n_tokens={self._n_tokens})"

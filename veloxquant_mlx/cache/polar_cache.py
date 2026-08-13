from __future__ import annotations

from typing import Any, List

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.core.constants import INT8_MAX
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.dsa.ring_buffer import RingBuffer
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer


class PolarQuantKVCache(KVCache):
    """KV cache backed by PolarQuantizer for key compression.

    Args:
        config: KVCacheConfig instance.
    """

    def __init__(self, config: Any) -> None:
        import mlx.core as mx

        self._config = config
        d = config.head_dim
        b = config.bit_width_inlier
        seed = config.seed
        store = config.store

        self._key_quantizer = PolarQuantizer(d=d, b=b, seed=seed, store=store)

        capacity = config.capacity or 1_000_000
        self._k_angles: RingBuffer = RingBuffer(capacity)  # each item = list of angle arrays
        self._k_radii: RingBuffer = RingBuffer(capacity)
        self._v_cache: RingBuffer = RingBuffer(capacity)
        self._v_scales: RingBuffer = RingBuffer(capacity)

        self._d = d
        self._n_tokens: int = 0

    def append_key(self, k: Any) -> None:
        """Encode and cache a single key vector.

        Args:
            k: Key vector, shape (d,), fp16.
        """
        if k.ndim == 1:
            k = k[None]
        ev = self._key_quantizer.encode(k)
        # Store per-token: list of 1-element angle arrays + scalar radius
        angles_per_level = [a[0] for a in ev.angles] if ev.angles else []
        self._k_angles.append(angles_per_level)
        self._k_radii.append(ev.final_radius[0] if ev.final_radius.ndim > 0 else ev.final_radius)
        self._n_tokens += 1

    def append_value(self, v: Any) -> None:
        """Quantize and cache a value vector.

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
        """Compute attention output for a query.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """
        import mlx.core as mx

        n = len(self._k_angles)
        if n == 0:
            return mx.zeros((self._d,), dtype=mx.float16)

        # Reconstruct batch EncodedVector from stored per-token encodings
        n_levels = len(self._k_angles[0]) if n > 0 else 0
        angles_batched = [
            mx.stack([self._k_angles[i][ell] for i in range(n)]) for ell in range(n_levels)
        ]
        radii_batched = mx.stack([self._k_radii[i] for i in range(n)])
        if radii_batched.ndim == 1:
            radii_batched = radii_batched

        ev = EncodedVector(
            quantizer_type="polar",
            batch_size=n,
            dim=self._d,
            angles=angles_batched,
            final_radius=radii_batched,
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
        n = len(self._k_angles)
        if n == 0:
            return 0
        d = self._d
        n_levels = 4
        # Angle indices: uint8 per level
        angle_bytes = n * n_levels * (d // 2) * 1
        # Final radius: fp16
        radius_bytes = n * 2
        # Value cache
        v_bytes = n * (d + 2)
        return angle_bytes + radius_bytes + v_bytes

    def __len__(self) -> int:
        return len(self._k_angles)

    def __repr__(self) -> str:
        return f"PolarQuantKVCache(d={self._d}, n_tokens={self._n_tokens})"

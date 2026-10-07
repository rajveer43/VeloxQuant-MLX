"""PolarQuant-backed KV cache: polar-coordinate key compression, int8 values.

Wraps :class:`~veloxquant_mlx.quantizers.polarquant.PolarQuantizer` in the
VeloxQuant :class:`~veloxquant_mlx.core.abstractions.KVCache` ABC
(append_key/append_value/attend), storing per-token angle indices and a
scalar radius per recursion level for keys, and per-token int8 + fp16 scale
for values. This is a "standalone" method (see
:data:`~veloxquant_mlx.cache.base.STANDALONE_METHODS`): it does not
implement the ``mlx_lm`` serving protocol.
"""

from __future__ import annotations

from typing import Any

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
        # Stored as a name, not an mx.Dtype, because mlx_lm.server deepcopies
        # cache entries per request and mx.Dtype objects raise TypeError from
        # copy.deepcopy (same convention as SnapKVCache._storage_dtype_name).
        self._storage_dtype_name: str | None = None

    @property
    def _storage_dtype(self):
        import mlx.core as mx

        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

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

        if self._storage_dtype_name is None:
            self._storage_dtype_name = "bfloat16" if v.dtype == mx.bfloat16 else "float16"
        if v.ndim > 1:
            v = v.reshape(-1)
        abs_max = float(mx.max(mx.abs(v)))
        scale = max(abs_max / INT8_MAX, 1e-8)
        v_int8 = mx.clip(mx.round(v / scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
        self._v_cache.append(v_int8)
        self._v_scales.append(mx.array(scale, dtype=self._storage_dtype))

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
            return mx.zeros((self._d,), dtype=self._storage_dtype)

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
        v_hat = v_int8.astype(self._storage_dtype) * v_scales[:, None]

        return (scores[:, None] * v_hat).sum(axis=0).astype(self._storage_dtype)

    def memory_bytes(self) -> int:
        """Bytes held for all cached tokens (angle codes, radii, int8 values and scales).

        Angle codes are stored one uint8 per angle, not packed to ``b`` bits.
        """
        n = len(self._k_angles)
        if n == 0:
            return 0
        # Per-token size read from what is actually stored: every token has the same
        # layout (one uint8 index per angle at each level, d / 2^n_levels fp16 radii).
        angle_bytes = sum(int(a.nbytes) for a in self._k_angles[0])
        radius_bytes = int(self._k_radii[0].nbytes)
        v_bytes = int(self._v_cache[0].nbytes) + int(self._v_scales[0].nbytes)
        return n * (angle_bytes + radius_bytes + v_bytes)

    def reset(self) -> None:
        """Clear all stored tokens; the key quantizer (seeded, deterministic
        polar rotation) is untouched."""
        capacity = self._k_angles._capacity
        self._k_angles = RingBuffer(capacity)
        self._k_radii = RingBuffer(capacity)
        self._v_cache = RingBuffer(capacity)
        self._v_scales = RingBuffer(capacity)
        self._n_tokens = 0

    def __len__(self) -> int:
        return len(self._k_angles)

    def __repr__(self) -> str:
        return f"PolarQuantKVCache(d={self._d}, n_tokens={self._n_tokens})"

"""QJL-backed KV cache: pure 1-bit Johnson-Lindenstrauss key compression.

Wraps :class:`~veloxquant_mlx.quantizers.qjl.QJLQuantizer` in the
VeloxQuant :class:`~veloxquant_mlx.core.abstractions.KVCache` ABC
(append_key/append_value/attend), storing per-token JL sign bits and a
scalar norm for keys, and per-token int8 + fp16 scale for values. This is
a "standalone" method (see
:data:`~veloxquant_mlx.cache.base.STANDALONE_METHODS`): it does not
implement the ``mlx_lm`` serving protocol.
"""

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
        config: KVCacheConfig instance. ``jl_dim`` sets the sketch size ``m``; when unset
            it defaults to ``head_dim``. The estimator is unbiased but its variance falls
            as ``1/m``, so ``m = head_dim`` gives noisy scores (correlation with exact
            ``q.k`` ~0.64 at d=128, ~0.88 at ``m = 4 * head_dim``). Set ``jl_dim``
            higher when attention quality matters.
    """

    def __init__(self, config: Any) -> None:

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
        # Stored as a name, not an mx.Dtype, because mlx_lm.server deepcopies
        # cache entries per request and mx.Dtype objects raise TypeError from
        # copy.deepcopy (same convention as SnapKVCache._storage_dtype_name).
        self._storage_dtype_name: str | None = None

    @property
    def _storage_dtype(self):
        import mlx.core as mx

        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

    def append_key(self, k: Any) -> None:
        """Encode and store a key vector.

        Args:
            k: Key vector, shape (d,), fp16.
        """
        if k.size != self._d:
            raise ValueError(
                f"append_key expects one key vector of size {self._d}, got shape {tuple(k.shape)}."
            )
        k = k.reshape(1, self._d)
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

        if self._storage_dtype_name is None:
            self._storage_dtype_name = "bfloat16" if v.dtype == mx.bfloat16 else "float16"
        if v.size != self._d:
            raise ValueError(
                f"append_value expects one value vector of size {self._d}, got shape {tuple(v.shape)}."
            )
        v = v.reshape(-1)
        abs_max = float(mx.max(mx.abs(v)))
        scale = max(abs_max / INT8_MAX, 1e-8)
        v_int8 = mx.clip(mx.round(v / scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
        self._v_cache.append(v_int8)
        self._v_scales.append(mx.array(scale, dtype=self._storage_dtype))

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
            return mx.zeros((self._d,), dtype=self._storage_dtype)

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
        v_hat = v_int8.astype(self._storage_dtype) * v_scales[:, None]

        return (scores[:, None] * v_hat).sum(axis=0).astype(self._storage_dtype)

    def memory_bytes(self) -> int:
        """Bytes held for all cached tokens.

        Signs are stored one per int8 (``m`` bytes per key), not bit-packed, so the
        key side is 8x what a true 1-bit store would need.
        """
        n = len(self._k_signs)
        if n == 0:
            return 0
        sign_bytes = n * self._m
        norm_bytes = n * 2
        v_bytes = n * (self._d + 2)
        return sign_bytes + norm_bytes + v_bytes

    def reset(self) -> None:
        """Clear all stored tokens; the key quantizer (seeded, deterministic
        JL projection) is untouched."""
        capacity = self._k_signs._capacity
        self._k_signs = RingBuffer(capacity)
        self._k_norms = RingBuffer(capacity)
        self._v_cache = RingBuffer(capacity)
        self._v_scales = RingBuffer(capacity)
        self._n_tokens = 0

    def __len__(self) -> int:
        return len(self._k_signs)

    def __repr__(self) -> str:
        return f"QJLKVCache(d={self._d}, m={self._m}, n_tokens={len(self)})"

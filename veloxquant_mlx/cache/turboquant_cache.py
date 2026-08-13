from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.core.constants import DEFAULT_N_CALIB_TOKENS, INT8_MAX
from veloxquant_mlx.dsa.bit_pack import BitPackBuffer
from veloxquant_mlx.outlier.detector import OutlierDetector
from veloxquant_mlx.quantizers.turboquant_mse import TurboQuantMSE
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd


class TurboQuantKVCache(KVCache):
    """KV cache backed by TurboQuantProd (or TurboQuantMSE) for key compression.

    Keys are stored bit-packed: indices use b_mse bits per coordinate and
    QJL signs use 1 bit each — matching the paper's memory model.
    Values use per-token int8 with fp16 scale.

    Args:
        config: KVCacheConfig instance.
    """

    def __init__(self, config: Any) -> None:
        import mlx.core as mx

        self._config = config
        d = config.head_dim
        b = config.bit_width_inlier
        if config.jl_dim is not None:
            m = config.jl_dim
        else:
            m = TurboQuantProd.m_default(d, b)
        seed = config.seed
        store = config.store
        use_prod = config.method == "turboquant_prod"

        if use_prod:
            self._key_quantizer: Any = TurboQuantProd(
                d=d,
                b=b,
                m=m,
                seed=seed,
                store=store,
                enable_fused_query_dot=bool(getattr(config, "enable_fused_query_dot", False)),
            )
            b_mse = max(b - 1, 1)
        else:
            self._key_quantizer = TurboQuantMSE(d=d, b=b, seed=seed, store=store)
            b_mse = b

        self._d = d
        self._m_eff = getattr(self._key_quantizer, "_m_eff", m)
        self._use_prod = use_prod
        capacity = config.capacity or 1_000_000

        # Bit-packers: indices at b_mse bits; signs at 1 bit
        self._idx_packer = BitPackBuffer(b_mse)
        self._sign_packer = BitPackBuffer(1) if use_prod else None
        self._b_mse = b_mse

        self._idx_packed_len = int(math.ceil(self._d * self._b_mse / 8))
        self._sign_packed_len = int(math.ceil(self._m_eff / 8)) if use_prod else 0
        self._k_indices_packed = np.zeros((capacity, self._idx_packed_len), dtype=np.uint8)
        self._k_signs_packed = (
            np.zeros((capacity, self._sign_packed_len), dtype=np.uint8) if use_prod else None
        )
        self._k_residual_norms = np.zeros((capacity,), dtype=np.float16)
        self._v_cache = np.zeros((capacity, self._d), dtype=np.int8)
        self._v_scales = np.zeros((capacity,), dtype=np.float16)

        self._capacity = capacity
        self._size = 0
        self._head = 0

        self._n_tokens: int = 0
        self._enable_vectorized_attend = bool(getattr(config, "enable_vectorized_attend", False))
        self._enable_outlier_two_stream = bool(getattr(config, "enable_outlier_two_stream", False))
        self._n_outliers = int(getattr(config, "n_outlier_channels", 0) or 0)
        self._n_calib = int(getattr(config, "n_calib_tokens", None) or DEFAULT_N_CALIB_TOKENS)
        self._outlier_detector = (
            OutlierDetector(n_outliers=self._n_outliers, n_calib=self._n_calib)
            if self._enable_outlier_two_stream and self._n_outliers > 0
            else None
        )
        self._outlier_idx: Optional[np.ndarray] = None
        self._inlier_idx: Optional[np.ndarray] = None
        self._outlier_cache: Optional[np.ndarray] = None
        self._outlier_scales: Optional[np.ndarray] = None
        if self._outlier_detector is not None:
            self._outlier_cache = np.zeros((capacity, self._n_outliers), dtype=np.int8)
            self._outlier_scales = np.zeros((capacity,), dtype=np.float16)

    def _physical_slot_for_append(self) -> int:
        if self._size < self._capacity:
            return (self._head + self._size) % self._capacity
        return self._head

    def _commit_append(self) -> None:
        if self._size < self._capacity:
            self._size += 1
        else:
            self._head = (self._head + 1) % self._capacity
        self._n_tokens += 1

    def _physical_indices(self, n: int) -> np.ndarray:
        return (self._head + np.arange(n, dtype=np.int64)) % self._capacity

    def _unpack_indices_block(self, packed_block: np.ndarray) -> np.ndarray:
        n = packed_block.shape[0]
        if not self._enable_vectorized_attend:
            out = np.empty((n, self._d), dtype=np.uint8)
            for i in range(n):
                out[i] = self._idx_packer.unpack(packed_block[i], self._d)
            return out
        # Vectorized path for b in {1,2,4}; fallback loops only for b=3.
        b = self._b_mse
        if b == 1:
            bits = np.unpackbits(packed_block, axis=1, bitorder="little")
            return bits[:, : self._d].astype(np.uint8)
        if b == 2:
            shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
            vals = ((packed_block[:, :, None] >> shifts[None, None, :]) & 0x3).reshape(n, -1)
            return vals[:, : self._d].astype(np.uint8)
        if b == 4:
            shifts = np.array([0, 4], dtype=np.uint8)
            vals = ((packed_block[:, :, None] >> shifts[None, None, :]) & 0xF).reshape(n, -1)
            return vals[:, : self._d].astype(np.uint8)
        out = np.empty((n, self._d), dtype=np.uint8)
        for i in range(n):
            out[i] = self._idx_packer.unpack(packed_block[i], self._d)
        return out

    def _unpack_signs_block(self, packed_block: np.ndarray) -> np.ndarray:
        n = packed_block.shape[0]
        if not self._enable_vectorized_attend:
            out = np.empty((n, self._m_eff), dtype=np.int8)
            for i in range(n):
                bits = self._sign_packer.unpack(packed_block[i], self._m_eff)  # type: ignore[union-attr]
                out[i] = bits.astype(np.int8) * 2 - 1
            return out
        bits = np.unpackbits(packed_block, axis=1, bitorder="little")[:, : self._m_eff]
        return (bits.astype(np.int8) * 2 - 1).astype(np.int8)

    def append_key(self, k: Any) -> None:
        """Encode and cache a single key vector (bit-packed storage).

        Args:
            k: Key vector, shape (d,), fp16.
        """
        import mlx.core as mx

        if k.ndim == 1:
            k = k[None]
        slot = self._physical_slot_for_append()
        k_np = np.array(k, dtype=np.float16).reshape(1, -1)

        if self._outlier_detector is not None:
            self._outlier_detector.observe(k_np)
            if self._outlier_idx is None and self._outlier_detector.is_calibrated:
                outlier_idx = self._outlier_detector.get_outlier_channels()
                self._outlier_idx = outlier_idx
                all_idx = np.arange(self._d, dtype=np.int32)
                self._inlier_idx = np.setdiff1d(all_idx, outlier_idx)
            if (
                self._outlier_idx is not None
                and self._outlier_cache is not None
                and self._outlier_scales is not None
            ):
                out = k_np[:, self._outlier_idx].reshape(-1)
                abs_max = float(np.max(np.abs(out)))
                o_scale = max(abs_max / INT8_MAX, 1e-8)
                q_out = mx.clip(mx.round(out / o_scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
                self._outlier_cache[slot, :] = np.array(q_out, dtype=np.int8)
                self._outlier_scales[slot] = np.float16(o_scale)
                k_np[:, self._outlier_idx] = 0
                k_for_inlier = mx.array(k_np, dtype=k.dtype)
            else:
                k_for_inlier = k
        else:
            k_for_inlier = k

        ev = self._key_quantizer.encode(k_for_inlier)
        idx_np = np.array(ev.indices[0], dtype=np.uint8)
        self._k_indices_packed[slot, :] = self._idx_packer.pack(idx_np)

        if (
            ev.signs is not None
            and self._sign_packer is not None
            and self._k_signs_packed is not None
        ):
            sign_np = np.array(ev.signs[0], dtype=np.int8)
            sign_bits = np.where(sign_np > 0, np.uint8(1), np.uint8(0))
            self._k_signs_packed[slot, :] = self._sign_packer.pack(sign_bits)
            self._k_residual_norms[slot] = np.float16(float(ev.residual_norm[0]))
        self._commit_append()

    def append_value(self, v: Any) -> None:
        """Quantize and cache a single value vector (int8 per-token).

        Args:
            v: Value vector, shape (d,), fp16.
        """
        import mlx.core as mx

        if v.ndim > 1:
            v = v.reshape(-1)
        slot = (self._head + self._size - 1) % self._capacity
        abs_max = float(mx.max(mx.abs(v)))
        scale = max(abs_max / INT8_MAX, 1e-8)
        v_int8 = mx.clip(mx.round(v / scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
        self._v_cache[slot, :] = np.array(v_int8, dtype=np.int8)
        self._v_scales[slot] = np.float16(scale)

    def attend(self, q: Any) -> Any:
        """Compute attention output for a query vector.

        Unpacks bit-packed keys, estimates attention scores, and returns
        the weighted sum of decoded values.

        Args:
            q: Query vector, shape (d,), fp16.

        Returns:
            Attention output, shape (d,), fp16.
        """
        import mlx.core as mx

        n = self._size
        if n == 0:
            return mx.zeros((self._d,), dtype=mx.float16)

        phys = self._physical_indices(n)
        k_indices_np = self._unpack_indices_block(self._k_indices_packed[phys])
        k_indices = mx.array(k_indices_np, dtype=mx.uint8)

        from veloxquant_mlx.core.context import EncodedVector

        if self._use_prod and self._k_signs_packed is not None:
            k_signs_np = self._unpack_signs_block(self._k_signs_packed[phys])
            k_signs = mx.array(k_signs_np, dtype=mx.int8)
            k_r_norms = mx.array(self._k_residual_norms[phys], dtype=mx.float16)
            ev = EncodedVector(
                quantizer_type="turboquant_prod",
                batch_size=n,
                dim=self._d,
                indices=k_indices,
                signs=k_signs,
                residual_norm=k_r_norms,
            )
        else:
            ev = EncodedVector(
                quantizer_type="turboquant_mse",
                batch_size=n,
                dim=self._d,
                indices=k_indices,
            )

        # Estimate inner products
        scores_raw = self._key_quantizer.estimate_inner_product(q, ev)  # (n,)
        if (
            self._outlier_idx is not None
            and self._outlier_cache is not None
            and self._outlier_scales is not None
        ):
            # MLX requires an mx.array index for fancy indexing on mx arrays.
            out_q = q[mx.array(self._outlier_idx)].astype(mx.float32)
            out_v = mx.array(self._outlier_cache[phys], dtype=mx.float32)
            out_scales = mx.array(self._outlier_scales[phys], dtype=mx.float32)
            out_mask = (out_scales > 0).astype(mx.float32)
            outlier_ip = mx.sum(out_v * out_q[None, :], axis=1) * out_scales * out_mask
            scores_raw = scores_raw + outlier_ip.astype(scores_raw.dtype)
        scale = float(mx.sqrt(mx.array(float(self._d))))
        scores = mx.softmax(scores_raw / scale, axis=0)  # (n,)

        # Decode values
        v_scales = mx.array(self._v_scales[phys], dtype=mx.float16)  # (n,)
        v_int8 = mx.array(self._v_cache[phys], dtype=mx.int8)  # (n, d)
        v_hat = v_int8.astype(mx.float16) * v_scales[:, None]

        return (scores[:, None] * v_hat).sum(axis=0)

    def memory_bytes(self) -> int:
        """Return actual memory of bit-packed key-value storage.

        Returns:
            Total bytes occupied by all cached data.
        """
        n = self._size
        if n == 0:
            return 0
        d = self._d
        # Indices: b_mse bits per coordinate
        idx_bytes = n * math.ceil(d * self._b_mse / 8)
        # Signs: 1 bit per sketch dimension
        sign_bytes = n * math.ceil(self._m_eff / 8) if self._use_prod else 0
        # Residual norms: fp16
        rnorm_bytes = n * 2 if self._use_prod else 0
        # Values: int8 + fp16 scale
        v_bytes = n * (d + 2)
        outlier_bytes = 0
        if self._outlier_detector is not None and self._outlier_idx is not None:
            outlier_bytes = n * (self._n_outliers + 2)
        return idx_bytes + sign_bytes + rnorm_bytes + v_bytes + outlier_bytes

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"TurboQuantKVCache(d={self._d}, n_tokens={self._size}, "
            f"method={'prod' if self._use_prod else 'mse'}, "
            f"b_mse={self._b_mse})"
        )

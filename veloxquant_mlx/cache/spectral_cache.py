"""SpectralQuantKVCache: KV cache backed by SpectralQuant.

Matches the KVCache interface used by TurboQuantKVCache.
Keys are compressed with SpectralQuant (spectral rotation + selective QJL
+ per-group codebooks). Values are compressed identically.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from veloxquant_mlx.core.abstractions import KVCache
from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer


class SpectralQuantKVCache(KVCache):
    """KV cache backed by SpectralQuant for both key and value compression.

    Keys are compressed with a SpectralQuantizer using the key eigenvectors.
    Values are compressed with a separate SpectralQuantizer using the value
    eigenvectors (which have d_s ≈ 50 rather than ≈ 4).

    Call `calibrate(rotations)` after construction to inject the
    per-layer rotation matrices from `calibrate_spectral_rotation()`.
    Without calibration the quantizers fall back to random orthogonal
    rotation (equivalent to TurboQuant without QJL).

    Args:
        config: KVCacheConfig with method='spectral'.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        d = config.head_dim
        b = config.bit_width_inlier
        seed = config.seed
        capacity = config.capacity or 1_000_000

        self._d = d
        self._capacity = capacity
        self._size = 0
        self._head = 0
        self._n_tokens = 0

        # Paper config flags from KVCacheConfig
        self._key_d_s: int = getattr(config, "spectral_key_d_eff", 4)
        self._val_d_s: int = getattr(config, "spectral_val_d_eff", 50)
        self._apply_qjl: bool = getattr(config, "spectral_apply_qjl", False)

        # Build quantizers — will be rebuilt after calibrate()
        self._key_q = SpectralQuantizer(
            d=d,
            b_signal=b,
            b_noise=b,
            rotation=None,
            d_s=self._key_d_s,
            apply_qjl=self._apply_qjl,
            seed=seed,
        )
        self._val_q = SpectralQuantizer(
            d=d,
            b_signal=b,
            b_noise=b,
            rotation=None,
            d_s=self._val_d_s,
            apply_qjl=False,
            seed=seed + 1,  # values: never QJL
        )

        # Storage: list of EncodedVectors (supports variable-length entries)
        self._k_encoded: list[Any] = []
        self._v_encoded: list[Any] = []

    # ------------------------------------------------------------------
    # Calibration injection
    # ------------------------------------------------------------------

    def calibrate(self, rotation_entry: tuple) -> None:
        """Inject calibrated rotation matrices for this layer.

        Args:
            rotation_entry: Tuple (key_U, val_U, key_ev, val_ev, key_ds,
                val_ds) as returned by calibrate_spectral_rotation() for
                this layer.
        """
        key_U, val_U, key_ev, val_ev, key_ds, val_ds = rotation_entry
        b = self._config.bit_width_inlier
        seed = self._config.seed

        self._key_d_s = int(key_ds)
        self._val_d_s = int(val_ds)

        self._key_q = SpectralQuantizer(
            d=self._d,
            b_signal=b,
            b_noise=b,
            rotation=key_U,
            d_s=self._key_d_s,
            apply_qjl=self._apply_qjl,
            seed=seed,
        )
        self._val_q = SpectralQuantizer(
            d=self._d,
            b_signal=b,
            b_noise=b,
            rotation=val_U,
            d_s=self._val_d_s,
            apply_qjl=False,
            seed=seed + 1,
        )

    # ------------------------------------------------------------------
    # Ring-buffer helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # KVCache interface
    # ------------------------------------------------------------------

    def append_key(self, k: Any) -> None:
        """Encode and store a key vector.

        Args:
            k: Key vector, shape (d,) or (1, d), fp16.
        """
        if k.ndim == 1:
            k = k[None]
        ev = self._key_q.encode(k)
        if len(self._k_encoded) >= self._capacity:
            self._k_encoded.pop(0)
        self._k_encoded.append(ev)
        self._commit_append()

    def append_value(self, v: Any) -> None:
        """Encode and store a value vector.

        Args:
            v: Value vector, shape (d,) or (1, d), fp16.
        """
        if v.ndim > 1:
            v = v.reshape(1, -1)
        else:
            v = v[None]
        ev = self._val_q.encode(v)
        if len(self._v_encoded) >= self._capacity:
            self._v_encoded.pop(0)
        self._v_encoded.append(ev)

    def attend(self, q: Any) -> Any:
        """Compute softmax attention output for query q.

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

        # Batch encoded key vectors into a single EncodedVector for efficiency
        from veloxquant_mlx.core.context import EncodedVector

        evs_k = [self._k_encoded[int(p)] for p in phys]
        evs_v = [self._v_encoded[int(p)] for p in phys]

        batched_k = _batch_encoded(evs_k, n)
        scores_raw = self._key_q.estimate_inner_product(q, batched_k)  # (n,)

        scale = float(mx.sqrt(mx.array(float(self._d))))
        scores = mx.softmax(scores_raw / scale, axis=0)  # (n,)

        # Decode values and compute weighted sum
        batched_v = _batch_encoded(evs_v, n)
        v_hat = self._val_q.decode(batched_v)  # (n, d) fp16
        return (scores[:, None] * v_hat).sum(axis=0)

    def memory_bytes(self) -> int:
        """Return compressed memory footprint in bytes."""
        n = self._size
        if n == 0:
            return 0
        key_bits = self._d_key_bits() * n
        val_bits = self._d_val_bits() * n
        return math.ceil((key_bits + val_bits) / 8)

    def _d_key_bits(self) -> int:
        d_s = self._key_d_s
        b = self._config.bit_width_inlier
        bits = d_s * b + (self._d - d_s) * b + 32  # +32 for 2 fp16 scales
        if self._apply_qjl and self._key_q._qjl is not None:
            bits += self._key_q._jl_dim + 16
        return bits

    def _d_val_bits(self) -> int:
        b = self._config.bit_width_inlier
        return self._d * b + 32  # uniform bits + 2 fp16 scales

    def compression_ratio(self) -> float:
        """Key compression ratio vs FP16."""
        return self._key_q.compression_ratio()

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"SpectralQuantKVCache(d={self._d}, n={self._size}, "
            f"key_ds={self._key_d_s}, qjl={self._apply_qjl})"
        )


def _batch_encoded(evs: list[Any], n: int) -> Any:
    """Concatenate a list of single-token EncodedVectors into a batch."""
    import mlx.core as mx
    import numpy as np
    from veloxquant_mlx.core.context import EncodedVector

    idx_list = [np.array(ev.indices, dtype=np.int32) for ev in evs]
    indices_mx = mx.array(np.concatenate(idx_list, axis=0), dtype=mx.uint8)

    norm_list = [np.array(ev.norm, dtype=np.float16) for ev in evs]
    norm_mx = mx.array(np.concatenate(norm_list, axis=0), dtype=mx.float16)

    rad_list = [np.array(ev.final_radius, dtype=np.float16) for ev in evs]
    rad_mx = mx.array(np.concatenate(rad_list, axis=0), dtype=mx.float16)

    signs_mx = None
    r_norms_mx = None
    if evs[0].signs is not None:
        signs_mx = mx.array(
            np.concatenate([np.array(ev.signs, dtype=np.int8) for ev in evs], axis=0),
            dtype=mx.int8,
        )
        r_norms_mx = mx.array(
            np.concatenate([np.array(ev.residual_norm, dtype=np.float16) for ev in evs]),
            dtype=mx.float16,
        )

    return EncodedVector(
        quantizer_type="spectral_quant",
        batch_size=n,
        dim=evs[0].dim,
        indices=indices_mx,
        norm=norm_mx,
        final_radius=rad_mx,
        signs=signs_mx,
        residual_norm=r_norms_mx,
    )

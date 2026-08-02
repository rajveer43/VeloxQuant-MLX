from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.constants import DEFAULT_POLAR_LEVELS
from veloxquant_mlx.core.context import EncodedVector, TransformResult
from veloxquant_mlx.core.registry import QuantizerRegistry
from veloxquant_mlx.math.rotation import (
    is_hadamard_compatible,
    make_hadamard_diagonal,
    make_rotation_matrix,
)
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner
from veloxquant_mlx.transforms.polar import RecursivePolarTransform


@QuantizerRegistry.register("polar")
class PolarQuantizer(Quantizer):
    """PolarQuant: recursive polar-coordinate quantization.

    Pipeline (encode):
        y = Π·x          (random rotation)
        angles, r_final = polar_transform(y)
        For each level ℓ: angle_idx[ℓ] = codebook[ℓ].quantize(angles[ℓ])
        Store: angle_idx (list), r_final (fp16 scalar per token)

    Pipeline (decode):
        For each level ℓ: angles[ℓ] = codebook[ℓ].dequantize(angle_idx[ℓ])
        y_hat = polar_inverse(angles, r_final)
        x_hat = Π^T · y_hat

    Args:
        d: Input dimension. Must be divisible by 2^n_levels.
        b: Bit-width for each level.
        n_levels: Number of polar recursion levels (default 4).
        seed: Random seed.
        m: Unused (for API consistency).
        store: Optional ArtifactStore.
        use_hadamard: If True, use randomized Hadamard instead of QR rotation.
    """

    def __init__(
        self,
        d: int,
        b: int = 2,
        m: int = 128,
        seed: int = 42,
        n_levels: int = DEFAULT_POLAR_LEVELS,
        store: Optional[ArtifactStore] = None,
        use_hadamard: bool = False,
        **kwargs: Any,
    ) -> None:
        self._d = d
        self._b = b
        self._n_levels = n_levels
        self._seed = seed

        import mlx.core as mx

        if use_hadamard and is_hadamard_compatible(d):
            D_np = make_hadamard_diagonal(d, seed=seed)
            D = mx.array(D_np)
            self._rotation = HadamardPreconditioner(D)
        else:
            if store is not None and store.exists("rotation", d=d, seed=seed):
                Pi = store.load_rotation_matrix(d, seed)
            else:
                Pi_np = make_rotation_matrix(d, seed=seed)
                Pi = mx.array(Pi_np.astype(np.float16))
                if store is not None:
                    store.save_rotation_matrix(Pi_np, d=d, seed=seed)
            self._rotation = RotationPreconditioner(Pi)
        self._transform = RecursivePolarTransform(n_levels=n_levels)

        # Per-level codebooks
        self._codebooks = []
        for level in range(1, n_levels + 1):
            dist_key = f"polar_level{level}"
            if store is not None and store.exists("codebook", distribution=dist_key, b=b, d=d):
                cb_np = np.array(store.load_codebook(dist_key, b=b, d=d), dtype=np.float32)
                from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook

                cb = ScalarCodebook(cb_np)
            else:
                cb = CodebookFactory.create("polar_level", b=b, d=d, polar_level=level)
                if store is not None:
                    store.save_codebook(
                        cb.centroids_numpy(),  # type: ignore[attr-defined]
                        distribution=dist_key,
                        b=b,
                        d=d,
                    )
            self._codebooks.append(cb)

    def encode(self, x: Any) -> EncodedVector:
        """Encode via rotation, polar transform, and per-level codebook lookup.

        Args:
            x: Array of shape (batch, d), fp16.

        Returns:
            EncodedVector with angles (list of index arrays) and final_radius.
        """
        if x.ndim == 1:
            x = x[None]

        y = self._rotation.apply(x)
        result = self._transform.forward(y)

        # Quantize each level's angles
        angle_indices: List[Any] = []
        for ell, (angles, cb) in enumerate(zip(result.angles, self._codebooks)):
            idx = cb.quantize(angles)
            angle_indices.append(idx)

        return EncodedVector(
            quantizer_type="polar",
            batch_size=x.shape[0],
            dim=self._d,
            angles=angle_indices,
            final_radius=result.final_radius,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Reconstruct approximate vector from polar encoding.

        Args:
            ev: EncodedVector with angles (indices) and final_radius.

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """
        # Dequantize angles
        dequant_angles = [cb.dequantize(idx) for cb, idx in zip(self._codebooks, ev.angles)]

        result = TransformResult(
            angles=dequant_angles,
            final_radius=ev.final_radius,
            n_levels=self._n_levels,
        )
        y_hat = self._transform.inverse(result)
        return self._rotation.apply_inverse(y_hat)

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ by decoding then computing dot product.

        Args:
            q: Query vector, shape (d,) or (1, d).
            ev: Encoded keys.

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        q_flat = q.reshape(-1)
        k_hat = self.decode(ev)  # (batch, d)
        return k_hat @ q_flat  # (batch,)

    def __repr__(self) -> str:
        return (
            f"PolarQuantizer(d={self._d}, b={self._b}, "
            f"n_levels={self._n_levels}, seed={self._seed})"
        )

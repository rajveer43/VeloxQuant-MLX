from __future__ import annotations

from typing import Any, Optional

import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.context import EncodedVector, QuantizationContext
from veloxquant_mlx.core.registry import QuantizerRegistry
from veloxquant_mlx.math.rotation import (
    is_hadamard_compatible,
    make_hadamard_diagonal,
    make_rotation_matrix,
)
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner


@QuantizerRegistry.register("turboquant_mse")
class TurboQuantMSE(Quantizer):
    """TurboQuant MSE-optimal scalar quantizer.

    Applies a random rotation then per-coordinate Lloyd-Max quantisation.

    Pipeline (encode):
        y = rotate(x)        (QR rotation or randomized Hadamard)
        idx = argmin_k |y_j - c_k|  (scalar codebook per coordinate)
        x̂ = unrotate(codebook[idx])

    MSE bound: D_mse ≤ √(3π)/2 · 4^(-b)

    Codebook uses N(0, 1/d) distribution (valid for d ≥ 64; Beta for d < 64).

    Args:
        d: Input dimension.
        b: Bit-width.
        seed: Random seed.
        m: Unused (for API consistency with factory).
        store: Optional ArtifactStore.
        use_beta: If True, use Beta distribution codebook; else Gaussian.
        use_hadamard: If True, use randomized Hadamard (O(d log d), Metal-accelerated)
            instead of QR rotation (O(d²), CPU matmul). Requires d = m*2^k
            where m in {1, 12, 20, 28} — all powers of 2 satisfy this.
    """

    def __init__(
        self,
        d: int,
        b: int = 2,
        seed: int = 42,
        m: int = 128,
        store: Optional[ArtifactStore] = None,
        use_beta: bool = False,
        use_hadamard: bool = False,
        **kwargs: Any,
    ) -> None:
        self._d = d
        self._b = b
        self._seed = seed

        import mlx.core as mx

        if use_hadamard and is_hadamard_compatible(d):
            D_np = make_hadamard_diagonal(d, seed=seed)
            D = mx.array(D_np)
            self._rotation = HadamardPreconditioner(D)
        else:
            # QR rotation matrix
            if store is not None and store.exists("rotation", d=d, seed=seed):
                Pi = store.load_rotation_matrix(d, seed)
            else:
                Pi_np = make_rotation_matrix(d, seed=seed)
                Pi = mx.array(Pi_np.astype(np.float16))
                if store is not None:
                    store.save_rotation_matrix(Pi_np, d=d, seed=seed)
            self._rotation = RotationPreconditioner(Pi)

        # Codebook
        distribution = "beta" if (use_beta or d < 64) else "gaussian"
        dist_key = distribution
        if store is not None and store.exists("codebook", distribution=dist_key, b=b, d=d):
            cb_centroids = np.array(store.load_codebook(dist_key, b=b, d=d), dtype=np.float32)
            from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook

            self._codebook = ScalarCodebook(cb_centroids)
        else:
            self._codebook = CodebookFactory.create(distribution, b=b, d=d)
            if store is not None:
                store.save_codebook(
                    self._codebook.centroids_numpy(),  # type: ignore[attr-defined]
                    distribution=dist_key,
                    b=b,
                    d=d,
                )

    def encode(self, x: Any) -> EncodedVector:
        """Encode via rotation + codebook quantisation.

        Args:
            x: Array of shape (batch, d), fp16.

        Returns:
            EncodedVector with indices populated.
        """
        if x.ndim == 1:
            x = x[None]
        y = self._rotation.apply(x)
        indices = self._codebook.quantize(y)
        return EncodedVector(
            quantizer_type="turboquant_mse",
            batch_size=x.shape[0],
            dim=self._d,
            indices=indices,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Dequantise: retrieve centroids then unrotate.

        Args:
            ev: EncodedVector with indices.

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """
        y_hat = self._codebook.dequantize(ev.indices)
        return self._rotation.apply_inverse(y_hat)

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ by decoding k then computing the dot product.

        This is the ExactReconstructStrategy — biased by MSE error.

        Args:
            q: Query vector, shape (d,) or (1, d).
            ev: Encoded keys.

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        import mlx.core as mx

        q_flat = q.reshape(-1)
        k_hat = self.decode(ev)  # (batch, d)
        return k_hat @ q_flat  # (batch,)

    def __repr__(self) -> str:
        return f"TurboQuantMSE(d={self._d}, b={self._b}, seed={self._seed})"

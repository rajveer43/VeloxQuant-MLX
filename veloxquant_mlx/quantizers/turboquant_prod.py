from __future__ import annotations

from typing import Any, Optional

import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.constants import SQRT_PI_OVER_2
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.core.registry import QuantizerRegistry
from veloxquant_mlx.math.rotation import (
    is_hadamard_compatible,
    make_hadamard_diagonal,
    make_jl_matrix,
    make_rotation_matrix,
)
from veloxquant_mlx.preconditioners.jl_sketch import QJLEncoder
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner


@QuantizerRegistry.register("turboquant_prod")
class TurboQuantProd(Quantizer):
    """TurboQuant two-stage unbiased inner-product quantizer.

    Stage 1: MSE quantize at bit-width (b-1) using rotation + codebook.
    Stage 2: QJL on the residual r = x - x̂_mse.

    Stored representation:
        indices:       (batch, d) uint8  — MSE codebook indices
        signs:         (batch, m) int8   — QJL signs of residual
        residual_norm: (batch,) fp16     — ‖x - x̂_mse‖₂

    Inner product estimation (unbiased):
        IP(q, k) ≈ ⟨q, x̂_mse⟩ + √(π/2)/m · ‖r‖ · ⟨Sq, sign(Sr)⟩

    Args:
        d: Input dimension.
        b: Total effective bit-width. MSE stage uses (b-1) bits.
        m: JL projection dimension.
        seed: Random seed.
        store: Optional ArtifactStore.
        use_hadamard: If True, use randomized Hadamard (O(d log d), Metal-accelerated)
            instead of QR rotation (O(d²), CPU matmul).
    """

    @staticmethod
    def m_default(d: int, b: int) -> int:
        """Default JL sketch dimension as a function of (d, b).

        At b<=2 the MSE residual is large, so allocate the full d sketch
        dimensions for the QJL correction. At b>=3 the residual is small and
        m=min(d, 64) is enough.
        """
        return d if b <= 2 else min(d, 64)

    def __init__(
        self,
        d: int,
        b: int = 3,
        m: Optional[int] = None,
        seed: int = 42,
        store: Optional[ArtifactStore] = None,
        use_hadamard: bool = False,
        use_adaptive_codebook: bool = False,
        n_calib: int = 64,
        **kwargs: Any,
    ) -> None:
        self._d = d
        self._b = b
        if m is None:
            m = TurboQuantProd.m_default(d, b)
        self._m = m
        self._seed = seed
        self._b_mse = max(b - 1, 1)
        self._enable_fused_query_dot = bool(kwargs.get("enable_fused_query_dot", False))

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

        # MSE codebook at (b-1) bits
        distribution = "gaussian" if d >= 64 else "beta"
        dist_key = distribution
        if store is not None and store.exists(
            "codebook", distribution=dist_key, b=self._b_mse, d=d
        ):
            cb_np = np.array(store.load_codebook(dist_key, b=self._b_mse, d=d), dtype=np.float32)
            from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook

            self._codebook = ScalarCodebook(cb_np)
        else:
            self._codebook = CodebookFactory.create(distribution, b=self._b_mse, d=d)
            if store is not None:
                store.save_codebook(
                    self._codebook.centroids_numpy(),  # type: ignore[attr-defined]
                    distribution=dist_key,
                    b=self._b_mse,
                    d=d,
                )

        # Optionally wrap the MSE codebook in an AdaptiveScalarCodebook that
        # refits its centroids from observed (post-rotation) vectors.
        self._use_adaptive_codebook = bool(use_adaptive_codebook)
        if self._use_adaptive_codebook:
            from veloxquant_mlx.codebooks.adaptive_codebook import AdaptiveScalarCodebook

            self._codebook = AdaptiveScalarCodebook(  # type: ignore[assignment]
                b=self._b_mse,
                d=d,
                n_calib=n_calib,
                default_codebook=self._codebook,  # type: ignore[arg-type]
            )

        # JL matrix for residual QJL — Gaussian JL allows m > d
        m_eff = m
        if store is not None and store.exists("jl", d=d, m=m_eff, seed=seed):
            S = store.load_jl_matrix(d, m_eff, seed)
        else:
            S_np = make_jl_matrix(d, m_eff, seed=seed)
            S = mx.array(S_np.astype(np.float16))
            if store is not None:
                store.save_jl_matrix(S_np, d=d, m=m_eff, seed=seed)

        self._qjl = QJLEncoder(S)
        self._m_eff = m_eff

    def _mse_ip_fused(self, q_rot: Any, indices: Any) -> Any:
        """Semi-fused MSE IP path: chunked gather + reduction.

        Reduces peak memory versus materializing full y_hat.
        """
        import mlx.core as mx

        batch, d = indices.shape
        chunk = 256
        centroids = self._codebook.centroids_mx()  # type: ignore[attr-defined]
        acc = mx.zeros((batch,), dtype=mx.float32)
        for start in range(0, d, chunk):
            end = min(start + chunk, d)
            idx_chunk = indices[:, start:end]
            y_chunk = mx.take(centroids, idx_chunk, axis=0).astype(mx.float32)
            q_chunk = q_rot[start:end].astype(mx.float32)
            acc = acc + mx.sum(y_chunk * q_chunk[None, :], axis=1)
        return acc.astype(mx.float16)

    def encode(self, x: Any) -> EncodedVector:
        """Two-stage encode: MSE + QJL residual.

        Args:
            x: Array of shape (batch, d), fp16.

        Returns:
            EncodedVector with indices, signs, and residual_norm.
        """
        if x.ndim == 1:
            x = x[None]

        # Stage 1: MSE quantize
        y = self._rotation.apply(x)
        if self._use_adaptive_codebook:
            self._codebook.observe(y)  # type: ignore[attr-defined]
        indices = self._codebook.quantize(y)
        y_hat = self._codebook.dequantize(indices)
        x_hat_mse = self._rotation.apply_inverse(y_hat)

        # Stage 2: QJL on residual r = x - x̂_mse
        r = x - x_hat_mse
        signs, r_norm = self._qjl.encode_key(r)

        return EncodedVector(
            quantizer_type="turboquant_prod",
            batch_size=x.shape[0],
            dim=self._d,
            indices=indices,
            signs=signs,
            residual_norm=r_norm,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Reconstruct approximate vector from two-stage encoding.

        x̃ = x̂_mse + ‖r‖ · (√(π/2)/m) · S^T · sign(S·r)

        Args:
            ev: EncodedVector with indices, signs, residual_norm.

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """
        import mlx.core as mx

        y_hat = self._codebook.dequantize(ev.indices)
        x_hat_mse = self._rotation.apply_inverse(y_hat)

        scale = SQRT_PI_OVER_2 / self._m_eff
        x_hat_qjl = ev.residual_norm[:, None] * scale * (ev.signs.astype(mx.float16) @ self._qjl._S)
        return x_hat_mse + x_hat_qjl

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ unbiasedly using both MSE and QJL stages.

        IP(q,k) ≈ ⟨q, x̂_mse⟩ + √(π/2)/m · ‖r‖ · ⟨Sq, sign(Sr)⟩

        MSE term is evaluated as ŷ @ q_rot where q_rot = Π·q, so the
        costly per-key unrotation (x_hat = ŷ @ Π) is replaced by a single
        query rotation: O(d²) once + O(n·d) dots vs O(n·d²) previously.

        Args:
            q: Query vector, shape (d,) or (1, d).
            ev: Encoded keys.

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        import mlx.core as mx

        q_flat = q.reshape(-1)

        # Rotate query once: q_rot = Π·q  (apply does x @ Π^T = Π·x for 1-D)
        q_rot = self._rotation.apply(q_flat.reshape(1, -1)).reshape(-1)  # (d,)

        # MSE contribution: ⟨x̂_mse, q⟩ = ŷ @ (Π·q) = ŷ @ q_rot
        if self._enable_fused_query_dot:
            ip_mse = self._mse_ip_fused(q_rot, ev.indices)
        else:
            y_hat = self._codebook.dequantize(ev.indices)  # (batch, d)
            ip_mse = y_hat @ q_rot  # (batch,)

        # QJL residual contribution
        ip_qjl = self._qjl.estimate_ip(q, ev.signs, ev.residual_norm)  # (batch,)

        return ip_mse + ip_qjl

    def __repr__(self) -> str:
        return (
            f"TurboQuantProd(d={self._d}, b={self._b}, "
            f"b_mse={self._b_mse}, m={self._m_eff}, seed={self._seed})"
        )


@QuantizerRegistry.register("turboquant_prod_adaptive")
class TurboQuantProdAdaptive(TurboQuantProd):
    """TurboQuantProd with use_adaptive_codebook=True forced on by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_adaptive_codebook", True)
        super().__init__(*args, **kwargs)

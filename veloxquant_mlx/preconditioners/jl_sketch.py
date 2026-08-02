from __future__ import annotations

import math
from typing import Any, Tuple

from veloxquant_mlx.core.abstractions import Preconditioner
from veloxquant_mlx.core.constants import SQRT_PI_OVER_2
from veloxquant_mlx.core.registry import PreconditionerRegistry


@PreconditionerRegistry.register("jl")
class JLSketchPreconditioner(Preconditioner):
    """Johnson-Lindenstrauss sketch preconditioner S ∈ ℝ^(m×d).

    Forward:  y = x @ S^T   — projects from d to m dimensions.
    Inverse:  x ≈ y @ S     — low-rank reconstruction (biased).

    Args:
        S: JL projection matrix of shape (m, d), fp16 MLX array.
    """

    def __init__(self, S: Any) -> None:
        self._S = S

    def apply(self, x: Any) -> Any:
        """Project x to m dimensions: y = x @ S^T.

        Args:
            x: Array of shape (batch, d).

        Returns:
            Projected array of shape (batch, m).
        """
        return x @ self._S.T

    def apply_inverse(self, y: Any) -> Any:
        """Reconstruct approximate x from projected y: x ≈ y @ S.

        Note: this reconstruction is biased for m < d.

        Args:
            y: Projected array of shape (batch, m).

        Returns:
            Approximate array of shape (batch, d).
        """
        return y @ self._S

    @property
    def d(self) -> int:
        """Input dimension."""
        return int(self._S.shape[1])

    @property
    def m(self) -> int:
        """Output (sketch) dimension."""
        return int(self._S.shape[0])

    def __repr__(self) -> str:
        return f"JLSketchPreconditioner(d={self.d}, m={self.m})"


class QJLEncoder:
    """Encodes key vectors using the 1-bit Quantized JL transform (QJL).

    Given a shared matrix S ∈ ℝ^(m×d) with i.i.d. N(0,1) rows, encodes each
    key k as:
        signs = sign(S·k) ∈ {-1, +1}^m     stored as int8
        norm  = ‖k‖₂                        stored as fp16

    Inner product estimation (unbiased for Gaussian S rows):
        ProdQJL(q, k) = √(π/2) / m · ‖k‖₂ · ⟨S·q, sign(S·k)⟩

    The estimator is exactly unbiased: E[ProdQJL(q,k)] = ⟨q, k⟩.

    Args:
        S: JL projection matrix of shape (m, d), fp16 MLX array.
    """

    _SCALE: float = SQRT_PI_OVER_2

    def __init__(self, S: Any) -> None:
        self._S = S
        self._m = int(S.shape[0])

    def encode_key(self, k: Any) -> Tuple[Any, Any]:
        """Encode a batch of key vectors.

        Args:
            k: Key array of shape (batch, d), fp16.

        Returns:
            Tuple (signs, norm):
                - signs: int8 array of shape (batch, m).
                - norm:  fp16 array of shape (batch,).
        """
        import mlx.core as mx

        # Accumulate S·k in float32 to avoid fp16 rounding in the projection
        Sk = k.astype(mx.float32) @ self._S.astype(mx.float32).T  # (batch, m)
        signs = mx.sign(Sk).astype(mx.int8)
        norm = mx.sqrt(mx.sum(k.astype(mx.float32) * k.astype(mx.float32), axis=-1)).astype(
            mx.float16
        )
        return signs, norm

    def estimate_ip(self, q: Any, signs: Any, norm: Any) -> Any:
        """Estimate inner products ⟨q, k⟩ for all cached keys.

        ProdQJL(q, k) = √(π/2)/m · ‖k‖ · ⟨S·q, sign(S·k)⟩

        Args:
            q: Query vector of shape (d,) or (1, d), fp16.
            signs: Cached sign arrays, shape (n, m) int8.
            norm: Cached norms, shape (n,) fp16.

        Returns:
            Estimated inner products, shape (n,), fp16.
        """
        import mlx.core as mx

        q_flat = q.reshape(1, -1) if q.ndim == 1 else q  # (1, d)
        # Accumulate S·q and the final dot in float32 to reduce rounding bias
        Sq = q_flat.astype(mx.float32) @ self._S.astype(mx.float32).T  # (1, m)
        ip = (signs.astype(mx.float32) @ Sq.T).squeeze(-1)  # (n,)
        scale = self._SCALE / self._m
        return (scale * norm.astype(mx.float32) * ip).astype(mx.float16)

    @property
    def m(self) -> int:
        """Sketch dimension."""
        return self._m

    def __repr__(self) -> str:
        return f"QJLEncoder(m={self._m}, d={self._S.shape[1]})"

from __future__ import annotations

from typing import Any

import numpy as np

from veloxquant_mlx.core.abstractions import Preconditioner
from veloxquant_mlx.core.registry import PreconditionerRegistry


@PreconditionerRegistry.register("hadamard")
class HadamardPreconditioner(Preconditioner):
    """Randomized Hadamard preconditioner: H @ diag(D).

    Forward:  y = hadamard_transform(D * x) / sqrt(d)
    Inverse:  x = D * hadamard_transform(y * sqrt(d)) / d
              (H is self-inverse up to factor d; D is its own inverse since D²=I)

    Stores only D (d floats) instead of the full d×d rotation matrix.
    Uses mx.hadamard_transform which is O(d log d) and Metal-accelerated.

    Requires d = m * 2^k where m in {1, 12, 20, 28}. All powers of 2 work.

    Args:
        D: Random ±1 diagonal vector of shape (d,), float32 MLX array.
    """

    def __init__(self, D: Any) -> None:
        self._D = D
        self._d = int(D.shape[0])

    def apply(self, x: Any) -> Any:
        """Forward: y = hadamard_transform(D * x).

        mx.hadamard_transform is already normalized (H(H(x)) = x), so no
        manual sqrt(d) scaling is needed.

        Args:
            x: Array of shape (batch, d), fp16 or fp32.

        Returns:
            Rotated array of shape (batch, d), same dtype.
        """
        import mlx.core as mx

        dtype = x.dtype
        out = mx.hadamard_transform(x.astype(mx.float32) * self._D.astype(mx.float32))
        return out.astype(dtype)

    def apply_inverse(self, y: Any) -> Any:
        """Inverse: x = D * hadamard_transform(y).

        Since H is its own inverse (normalized) and D² = I, this recovers x exactly.

        Args:
            y: Rotated array of shape (batch, d).

        Returns:
            Reconstructed array of shape (batch, d).
        """
        import mlx.core as mx

        dtype = y.dtype
        out = mx.hadamard_transform(y.astype(mx.float32)) * self._D.astype(mx.float32)
        return out.astype(dtype)

    @property
    def dim(self) -> int:
        return self._d

    def __repr__(self) -> str:
        return f"HadamardPreconditioner(d={self._d})"


@PreconditionerRegistry.register("rotation")
class RotationPreconditioner(Preconditioner):
    """Orthogonal rotation preconditioner Π ∈ ℝ^(d×d).

    Forward:  y = x @ Π^T
    Inverse:  x = y @ Π       (since Π^T Π = I)

    Args:
        Pi: Orthogonal rotation matrix of shape (d, d), fp16 MLX array.
    """

    def __init__(self, Pi: Any) -> None:
        self._Pi = Pi

    def apply(self, x: Any) -> Any:
        """Rotate x: y = x @ Π^T.

        Args:
            x: Array of shape (batch, d).

        Returns:
            Rotated array of shape (batch, d).
        """
        import mlx.core as mx

        return x @ self._Pi.T

    def apply_inverse(self, y: Any) -> Any:
        """Rotate back: x = y @ Π.

        Args:
            y: Rotated array of shape (batch, d).

        Returns:
            Reconstructed array of shape (batch, d).
        """
        import mlx.core as mx

        return y @ self._Pi

    @property
    def dim(self) -> int:
        """Matrix dimension d."""
        return int(self._Pi.shape[0])

    def __repr__(self) -> str:
        return f"RotationPreconditioner(d={self.dim})"

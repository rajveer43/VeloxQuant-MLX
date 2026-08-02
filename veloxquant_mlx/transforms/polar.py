from __future__ import annotations

import math
from typing import Any, List

from veloxquant_mlx.core.abstractions import Transform
from veloxquant_mlx.core.context import TransformResult
from veloxquant_mlx.core.constants import DEFAULT_POLAR_LEVELS


class RecursivePolarTransform(Transform):
    """Recursive polar-coordinate decomposition for PolarQuant.

    Algorithm (forward, n_levels levels):
        For ℓ = 1, …, n_levels:
            1. Pair adjacent coordinates: r_pairs = r.reshape(batch, -1, 2)
            2. Compute angle: ψ_j = atan2(r[2j+1], r[2j])
               Level 1 → ψ ∈ [0, 2π); Levels ≥ 2 → ψ ∈ [0, π/2].
            3. Compute new radius: r_j = sqrt(r[2j]^2 + r[2j+1]^2)

    After n_levels, we have:
        angles:       list of n_levels arrays; angles[ℓ] has shape (batch, d/2^(ℓ+1))
        final_radius: shape (batch,) — the remaining scalar radius

    Algorithm (inverse, reverse order):
        For ℓ = n_levels-1, …, 0:
            r_j^{prev} = [r_j · cos(ψ_j), r_j · sin(ψ_j)]
        Interleave pairs back into the original order.

    Args:
        n_levels: Number of polar recursion levels (default 4).

    Raises:
        ValueError: If n_levels < 1.
    """

    def __init__(self, n_levels: int = DEFAULT_POLAR_LEVELS) -> None:
        if n_levels < 1:
            raise ValueError(f"RecursivePolarTransform: n_levels must be >= 1, got {n_levels}")
        self._n_levels = n_levels

    def forward(self, x: Any) -> TransformResult:
        """Apply recursive polar decomposition.

        The arctan2 computation is performed in float32 for precision,
        then the angles are cast back to the input dtype.

        Args:
            x: Input array of shape (batch, d), fp16. d must be divisible
               by 2^n_levels.

        Returns:
            TransformResult with angles and final_radius.

        Raises:
            ValueError: If d is not divisible by 2^n_levels.
        """
        import mlx.core as mx

        batch, d = x.shape[0], x.shape[1]
        if d % (2**self._n_levels) != 0:
            raise ValueError(
                f"RecursivePolarTransform: d={d} must be divisible by "
                f"2^n_levels=2^{self._n_levels}={2**self._n_levels}"
            )

        r = x.astype(mx.float32)
        angles: List[Any] = []

        for ell in range(self._n_levels):
            n_pairs = r.shape[-1] // 2
            r_pairs = r.reshape(batch, n_pairs, 2)  # (batch, n_pairs, 2)

            # Compute angle: atan2(y, x)
            a = mx.arctan2(r_pairs[:, :, 1], r_pairs[:, :, 0])  # (batch, n_pairs)

            # For level >= 2, fold angles into [0, pi/2]
            if ell >= 1:
                a = mx.abs(a % (math.pi / 2))

            angles.append(a.astype(x.dtype))

            # New radii: sqrt(x^2 + y^2)
            r = mx.sqrt(r_pairs[:, :, 0] ** 2 + r_pairs[:, :, 1] ** 2)

        # r now has shape (batch, d/2^n_levels)
        final_radius = r.reshape(batch, -1).astype(x.dtype)
        # Squeeze to (batch,) if d/2^n_levels == 1
        if final_radius.shape[-1] == 1:
            final_radius = final_radius[:, 0]

        return TransformResult(angles=angles, final_radius=final_radius, n_levels=self._n_levels)

    def inverse(self, result: TransformResult) -> Any:
        """Reconstruct approximate vector from polar coordinates.

        Applies inverse polar transform in reverse level order.

        Args:
            result: TransformResult produced by forward().

        Returns:
            Reconstructed array of shape (batch,d), same dtype as stored angles.
        """
        import mlx.core as mx

        angles = result.angles
        r = result.final_radius

        # Ensure r has shape (batch, 1) for broadcasting
        if r.ndim == 1:
            r = r[:, None]  # (batch, 1)

        dtype = angles[0].dtype

        for ell in range(self._n_levels - 1, -1, -1):
            theta = angles[ell].astype(mx.float32)  # (batch, n_pairs)
            r_f32 = r.astype(mx.float32)

            # Expand r to match theta dimensions
            if r_f32.shape[-1] == 1 and theta.shape[-1] > 1:
                r_f32 = mx.broadcast_to(r_f32, theta.shape)
            elif r_f32.shape[-1] != theta.shape[-1]:
                # r has more dims: it was not fully squeezed
                pass

            x_comp = r_f32 * mx.cos(theta)  # (batch, n_pairs)
            y_comp = r_f32 * mx.sin(theta)  # (batch, n_pairs)

            # Interleave: [x0, y0, x1, y1, ...]
            batch = x_comp.shape[0]
            n_pairs = x_comp.shape[1]
            interleaved = mx.stack([x_comp, y_comp], axis=2)  # (batch, n_pairs, 2)
            r = interleaved.reshape(batch, n_pairs * 2)  # (batch, 2*n_pairs)

        return r.astype(dtype)

    @property
    def n_levels(self) -> int:
        """Number of polar recursion levels."""
        return self._n_levels

    def __repr__(self) -> str:
        return f"RecursivePolarTransform(n_levels={self._n_levels})"

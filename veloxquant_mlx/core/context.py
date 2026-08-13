from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np


# Lazy MLX import to allow math/ subpackage to be MLX-free
def _mx():
    import mlx.core as mx

    return mx


@dataclass
class QuantizationContext:
    """Mutable payload flowing through a QuantizationHandler chain.

    Attributes:
        x_original: Original input vectors, shape (batch, d).
        mode: Whether the chain is encoding or decoding.
        x_current: Working copy mutated by each handler.
        norm: L2 norm stored by NormalizationHandler, shape (batch,).
        rotated: Vector after RotationHandler, shape (batch, d).
        indices: Codebook indices, shape (batch, d) uint8.
        signs: QJL sign bits, shape (batch, m) int8.
        residual_norm: Residual L2 norm for QJL stage, shape (batch,).
        angles: Per-level polar angles from PolarTransformHandler.
        final_radius: Scalar radius after all polar levels, shape (batch,).
        outlier_idx: Channel positions of outlier channels.
        packed_bits: Bit-packed index array from BitPackingHandler.
        metadata: Arbitrary stage-specific metadata.
    """

    x_original: Any  # mx.array (batch, d)
    mode: Literal["encode", "decode"]
    x_current: Any  # mx.array (batch, d)
    norm: Optional[Any] = None  # mx.array (batch,)
    rotated: Optional[Any] = None  # mx.array (batch, d)
    indices: Optional[Any] = None  # mx.array (batch, d) uint8
    signs: Optional[Any] = None  # mx.array (batch, m) int8
    residual_norm: Optional[Any] = None  # mx.array (batch,)
    angles: Optional[List[Any]] = None  # list of mx.array per level
    final_radius: Optional[Any] = None  # mx.array (batch,)
    outlier_idx: Optional[np.ndarray] = None
    packed_bits: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        shape = getattr(self.x_original, "shape", None)
        return (
            f"QuantizationContext(mode={self.mode!r}, shape={shape}, "
            f"has_norm={self.norm is not None}, "
            f"has_indices={self.indices is not None}, "
            f"has_signs={self.signs is not None})"
        )


@dataclass
class EncodedVector:
    """Typed output of Quantizer.encode().

    Different quantizer types populate different subsets of fields.

    Attributes:
        quantizer_type: Registry key of the producing quantizer.
        batch_size: Number of vectors encoded.
        dim: Original vector dimensionality.
        indices: Scalar codebook indices, shape (batch, d) uint8.
        norm: Per-vector L2 norm, shape (batch,) fp16.
        signs: QJL sign bits, shape (batch, m) int8.
        residual_norm: QJL residual norm, shape (batch,) fp16.
        angles: PolarQuant level angles, list of (batch, d/2^ℓ) arrays.
        final_radius: PolarQuant scalar radius, shape (batch,) fp16.
        outlier_idx: Outlier channel indices for CompositeQuantizer.
        outlier_encoded: Nested EncodedVector for outlier channels.
        inlier_encoded: Nested EncodedVector for inlier channels.
    """

    quantizer_type: str
    batch_size: int
    dim: int
    indices: Optional[Any] = None
    norm: Optional[Any] = None
    signs: Optional[Any] = None
    residual_norm: Optional[Any] = None
    angles: Optional[List[Any]] = None
    final_radius: Optional[Any] = None
    outlier_idx: Optional[np.ndarray] = None
    outlier_encoded: Optional["EncodedVector"] = None
    inlier_encoded: Optional["EncodedVector"] = None

    def memory_bytes(self) -> int:
        """Compute exact memory footprint of this encoded representation.

        Returns:
            Total bytes occupied by all stored arrays.
        """
        total = 0

        def _arr_bytes(arr: Any) -> int:
            if arr is None:
                return 0
            if isinstance(arr, np.ndarray):
                return arr.nbytes
            # mx.array
            try:
                import mlx.core as mx

                size = 1
                for s in arr.shape:
                    size *= s
                itemsize = {
                    mx.float16: 2,
                    mx.float32: 4,
                    mx.int8: 1,
                    mx.uint8: 1,
                    mx.bfloat16: 2,
                }.get(arr.dtype, 4)
                return size * itemsize
            except Exception:
                return 0

        total += _arr_bytes(self.indices)
        total += _arr_bytes(self.norm)
        total += _arr_bytes(self.signs)
        total += _arr_bytes(self.residual_norm)
        total += _arr_bytes(self.final_radius)
        total += _arr_bytes(self.outlier_idx)
        if self.angles:
            for a in self.angles:
                total += _arr_bytes(a)
        if self.outlier_encoded is not None:
            total += self.outlier_encoded.memory_bytes()
        if self.inlier_encoded is not None:
            total += self.inlier_encoded.memory_bytes()
        return total

    def __repr__(self) -> str:
        return (
            f"EncodedVector(type={self.quantizer_type!r}, "
            f"batch={self.batch_size}, dim={self.dim}, "
            f"bytes={self.memory_bytes()})"
        )


@dataclass
class TransformResult:
    """Output of RecursivePolarTransform.forward().

    Attributes:
        angles: List of angle arrays, one per polar level.
            angles[0] has shape (batch, d/2), angles[ℓ] has shape (batch, d/2^(ℓ+1)).
        final_radius: Scalar radius at the end of recursion, shape (batch,).
        n_levels: Number of polar recursion levels applied.
    """

    angles: List[Any]  # list of mx.array
    final_radius: Any  # mx.array (batch,)
    n_levels: int

    def __repr__(self) -> str:
        angle_shapes = [getattr(a, "shape", None) for a in self.angles]
        return (
            f"TransformResult(n_levels={self.n_levels}, "
            f"angle_shapes={angle_shapes}, "
            f"radius_shape={getattr(self.final_radius, 'shape', None)})"
        )

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.math.rotation import (
    is_hadamard_compatible,
    make_hadamard_diagonal,
    make_rotation_matrix,
)
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with TurboQuant weight compression.

    Weight matrix W (out, in) is quantized row-by-row offline:
        1. Normalize each row: w_norm = w / ||w||_2  (store norm)
        2. Rotate: w_rot = preconditioner.apply(w_norm)
        3. Quantize to b-bit Lloyd-Max indices (codebook fits N(0, 1/sqrt(in)))
        4. Store indices + per-row norms

    Forward pass:
        w_hat = norms * unrotate(centroids[indices])
        out   = x @ w_hat.T + bias

    Per-row normalization is critical: weight rows have varying magnitudes
    but the Lloyd-Max codebook is calibrated for unit-norm post-rotation
    vectors. Without normalization the codebook range is completely wrong.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Bit-width for weight quantization (2, 3, or 4).
        use_hadamard: Use randomized Hadamard (O(d log d)) instead of QR (O(d²)).
            Auto-falls back to QR if in_features is not Hadamard-compatible.
        bias: Whether the layer has a bias term.
        seed: Random seed for the rotation matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        use_hadamard: bool = True,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._in = in_features
        self._out = out_features
        self._bits = bits
        self._seed = seed

        # Rotation — auto-fallback to QR if Hadamard dimension not supported
        use_hadamard_actual = use_hadamard and is_hadamard_compatible(in_features)
        if use_hadamard_actual:
            D_np = make_hadamard_diagonal(in_features, seed=seed)
            self._preconditioner = HadamardPreconditioner(mx.array(D_np))
        else:
            Pi_np = make_rotation_matrix(in_features, seed=seed)
            self._preconditioner = RotationPreconditioner(mx.array(Pi_np.astype(np.float32)))

        # Lloyd-Max codebook for N(0, 1/sqrt(in)) — valid for unit-norm rotated rows
        distribution = "gaussian" if in_features >= 64 else "beta"
        self._codebook = CodebookFactory.create(distribution, b=bits, d=in_features)
        self._centroids: mx.array = self._codebook.centroids_mx()  # (2^bits,) fp16

        # Filled by quantize_weights()
        self._w_indices: mx.array = mx.zeros((out_features, in_features), dtype=mx.uint8)
        self._w_norms: mx.array = mx.ones((out_features, 1), dtype=mx.float32)

        self._bias: Optional[mx.array] = None
        self._has_bias = bias

    def quantize_weights(self, weight: mx.array, bias: Optional[mx.array] = None) -> None:
        """Compress a weight matrix into this layer.

        Args:
            weight: Shape (out_features, in_features), fp16 or fp32.
            bias: Optional shape (out_features,).
        """
        w = weight.astype(mx.float32)  # (out, in)

        # 1. Per-row L2 normalization — makes all rows unit-norm
        norms = mx.linalg.norm(w, axis=-1, keepdims=True)  # (out, 1)
        safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
        w_norm = w / safe_norms  # (out, in), unit-norm rows

        # 2. Rotate
        w_rot = self._preconditioner.apply(w_norm)  # (out, in)

        # 3. Quantize via argmin over Lloyd-Max codebook
        c = self._centroids.astype(mx.float32)  # (k,)
        dists = mx.abs(w_rot[:, :, None] - c[None, None, :])  # (out, in, k)
        self._w_indices = mx.argmin(dists, axis=-1).astype(mx.uint8)
        self._w_norms = safe_norms  # (out, 1)

        if bias is not None:
            self._bias = bias.astype(mx.float16)

        mx.eval(self._w_indices, self._w_norms)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: dequantize weights, rescale by norms, linear projection.

        Args:
            x: Input of shape (..., in_features).

        Returns:
            Output of shape (..., out_features).
        """
        # 1. Dequantize + unrotate → unit-norm rows
        w_rot_hat = self._centroids[self._w_indices]  # (out, in) fp16
        w_unit = self._preconditioner.apply_inverse(w_rot_hat.astype(mx.float32))  # (out, in) fp32

        # 2. Rescale rows by their original norms
        w_hat = (w_unit * self._w_norms).astype(mx.float16)  # (out, in) fp16

        # 3. Linear projection
        out = x.astype(mx.float16) @ w_hat.T
        if self._bias is not None:
            out = out + self._bias
        return out

    @property
    def memory_bytes(self) -> int:
        """Compressed storage: indices + norms (excludes codebook, shared)."""
        idx_bytes = math.ceil(self._out * self._in * self._bits / 8)
        norm_bytes = self._out * 4  # float32 per row
        return idx_bytes + norm_bytes

    @property
    def fp16_bytes(self) -> int:
        return self._out * self._in * 2

    def compression_ratio(self) -> float:
        return self.fp16_bytes / self.memory_bytes

    def __repr__(self) -> str:
        return (
            f"QuantizedLinear(in={self._in}, out={self._out}, "
            f"bits={self._bits}, compression={self.compression_ratio():.1f}x)"
        )

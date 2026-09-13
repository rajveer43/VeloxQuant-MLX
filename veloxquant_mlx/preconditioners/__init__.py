"""Preconditioning transforms applied to key/value vectors before quantization.

Preconditioners reshape a vector's distribution (via rotation, JL
projection, or Hadamard transform) so downstream scalar/vector
quantization sees near-isotropic, outlier-free inputs. Re-exports
``PreconditionerFactory`` (construction dispatch), ``JLSketchPreconditioner``
and ``QJLEncoder`` (Johnson-Lindenstrauss sketch and its 1-bit QJL
encoder), and ``RotationPreconditioner`` (orthogonal rotation).
"""

from __future__ import annotations

from veloxquant_mlx.preconditioners.base import PreconditionerFactory
from veloxquant_mlx.preconditioners.jl_sketch import JLSketchPreconditioner, QJLEncoder
from veloxquant_mlx.preconditioners.rotation import RotationPreconditioner

__all__ = [
    "PreconditionerFactory",
    "JLSketchPreconditioner",
    "QJLEncoder",
    "RotationPreconditioner",
]

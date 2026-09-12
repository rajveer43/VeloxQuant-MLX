"""Pure-numpy math helpers shared by the codebook, quantizer, and preconditioner layers.

Groups three independent pieces: closed-form/numerically-integrated PDFs
(:mod:`~veloxquant_mlx.math.distributions`) used as Lloyd-Max inputs, the
Lloyd-Max scalar-quantizer solver itself
(:mod:`~veloxquant_mlx.math.lloyd_max`), and random orthogonal/JL matrix
generators (:mod:`~veloxquant_mlx.math.rotation`) used to build
preconditioners. Re-exports ``beta_pdf``, ``gaussian_pdf``,
``polar_angle_pdf``, ``lloyd_max``, ``make_jl_matrix``, and
``make_rotation_matrix``.
"""

from __future__ import annotations

from veloxquant_mlx.math.distributions import beta_pdf, gaussian_pdf, polar_angle_pdf
from veloxquant_mlx.math.lloyd_max import lloyd_max
from veloxquant_mlx.math.rotation import make_jl_matrix, make_rotation_matrix

__all__ = [
    "beta_pdf",
    "gaussian_pdf",
    "polar_angle_pdf",
    "lloyd_max",
    "make_jl_matrix",
    "make_rotation_matrix",
]

from __future__ import annotations

from veloxquant_mlx.spectral.bit_allocator import water_fill_bits
from veloxquant_mlx.spectral.calibrate import (
    calibrate_from_vectors,
    calibrate_spectral_rotation,
    load_cached_rotations,
    save_rotations,
)
from veloxquant_mlx.spectral.participation_ratio import (
    compute_participation_ratio,
    compute_spectral_gap,
)
from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

__all__ = [
    "SpectralQuantizer",
    "calibrate_spectral_rotation",
    "calibrate_from_vectors",
    "load_cached_rotations",
    "save_rotations",
    "compute_participation_ratio",
    "compute_spectral_gap",
    "water_fill_bits",
]

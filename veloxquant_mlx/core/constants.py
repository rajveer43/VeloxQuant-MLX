from __future__ import annotations

import math

# QJL inner product estimator scale factor: sqrt(pi/2) / m applied at runtime
SQRT_PI_OVER_2: float = math.sqrt(math.pi / 2)

# TurboQuant theoretical MSE bounds: D_mse <= UPPER_MSE_FACTOR * 4^(-b)
UPPER_MSE_FACTOR: float = math.sqrt(3 * math.pi) / 2  # ≈ 2.7207

# Theoretical lower bound factor: D_mse >= 4^(-b)
LOWER_MSE_FACTOR: float = 1.0

# Lloyd-Max solver defaults
LLOYD_MAX_N_ITER: int = 500
LLOYD_MAX_TOL: float = 1e-9
LLOYD_MAX_N_QUAD: int = 10_000

# Minimum probability mass for a Voronoi cell to avoid degenerate centroids
MIN_CELL_MASS: float = 1e-12

# VoronoiTree linear-scan threshold
VORONOI_LINEAR_THRESHOLD: int = 16

# Default random seed used across the library
DEFAULT_SEED: int = 42

# Default JL projection dimension (matches head dim for quality)
DEFAULT_JL_DIM: int = 128

# Default number of polar levels
DEFAULT_POLAR_LEVELS: int = 4

# Default outlier channel count
DEFAULT_N_OUTLIER_CHANNELS: int = 4

# Minimum number of calibration tokens before OutlierDetector is considered calibrated
DEFAULT_N_CALIB_TOKENS: int = 200

# Value cache int8 clamp range
INT8_MAX: int = 127
INT8_MIN: int = -127

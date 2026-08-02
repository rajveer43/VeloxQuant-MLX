from __future__ import annotations

import math

import numpy as np


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a d-dimensional unit-sphere uniform sample.

    After a random orthogonal rotation, each coordinate of a d-dimensional
    vector drawn uniformly from the unit sphere follows::

        f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)

    on [-1, 1]. In high dimensions (d ≥ 64), this converges to N(0, 1/d).

    Args:
        x: Query points in (-1, 1).
        d: Vector dimension (must be >= 2).

    Returns:
        PDF values at each point in x. Points outside (-1, 1) return 0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = (x > -1.0) & (x < 1.0)
    if not np.any(mask):
        return out

    log_norm = math.lgamma(d / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2.0)
    alpha = (d - 3) / 2.0
    xm = x[mask]
    log_vals = log_norm + alpha * np.log1p(-(xm**2))
    out[mask] = np.exp(log_vals)
    return out


def polar_angle_pdf(psi: np.ndarray, level: int) -> np.ndarray:
    """PDF of polar angle at a given recursion level after random preconditioning.

    From PolarQuant (Theorem 1 in 2502.02617):

    * Level 1: uniform on [0, 2π).
      f(ψ) = 1/(2π)
    * Level ℓ ≥ 2: f(ψ) ∝ sin^(2^(ℓ-1) - 1)(2ψ) on [0, π/2].

    Args:
        psi: Query angles (radians).
        level: Polar recursion level (1-indexed, must be >= 1).

    Returns:
        Normalised PDF values at each psi. Points outside the support return 0.

    Raises:
        ValueError: If level < 1.
    """
    if level < 1:
        raise ValueError(f"polar_angle_pdf: level must be >= 1, got {level}")

    psi = np.asarray(psi, dtype=np.float64)
    out = np.zeros_like(psi)

    if level == 1:
        mask = (psi >= 0.0) & (psi < 2 * math.pi)
        out[mask] = 1.0 / (2 * math.pi)
        return out

    # Level >= 2
    k = 2 ** (level - 1) - 1  # exponent
    mask = (psi >= 0.0) & (psi <= math.pi / 2)
    psi_m = psi[mask]

    sin_vals = np.sin(2.0 * psi_m)
    # Avoid log(0) for endpoints
    with np.errstate(divide="ignore", invalid="ignore"):
        log_unnorm = np.where(sin_vals > 0, k * np.log(sin_vals), -np.inf)

    # Normalising constant via numerical integration
    Z = _polar_angle_normalizer(level)
    out[mask] = np.exp(log_unnorm) / Z
    return out


def _polar_angle_normalizer(level: int) -> float:
    """Compute the normalising constant for polar_angle_pdf at level >= 2."""
    k = 2 ** (level - 1) - 1
    psi = np.linspace(0.0, math.pi / 2, 100_000)
    vals = np.sin(2.0 * psi) ** k
    return float(np.trapezoid(vals, psi))


def gaussian_pdf(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """PDF of a zero-mean Gaussian with standard deviation sigma.

    f(x) = 1 / (σ√(2π)) · exp(-x²/(2σ²))

    Args:
        x: Query points.
        sigma: Standard deviation (must be > 0).

    Returns:
        PDF values at each point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coeff * np.exp(-0.5 * (x / sigma) ** 2)

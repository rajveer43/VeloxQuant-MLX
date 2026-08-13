from __future__ import annotations

import numpy as np


def compute_participation_ratio(vectors: np.ndarray) -> float:
    """Compute the effective dimensionality d_eff = (Σλ_i)² / Σλ_i².

    Args:
        vectors: Array of shape (n_samples, d), fp32 or fp16.

    Returns:
        Participation ratio as a float. Equals d when all eigenvalues are
        equal (fully spread), equals 1 when a single dimension dominates.
    """
    X = np.array(vectors, dtype=np.float32)
    X -= X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0, None)
    sum_sq = float(np.sum(eigenvalues) ** 2)
    sq_sum = float(np.sum(eigenvalues**2))
    if sq_sum < 1e-12:
        return 1.0
    return sum_sq / sq_sum


def compute_spectral_gap(vectors: np.ndarray) -> tuple[int, np.ndarray]:
    """Find the integer d_eff cutoff and return the full eigenvalue spectrum.

    Uses the participation ratio rounded to the nearest integer as the
    signal dimension count.

    Args:
        vectors: Array of shape (n_samples, d), fp32 or fp16.

    Returns:
        Tuple of (d_eff: int, eigenvalues: np.ndarray sorted descending).
    """
    X = np.array(vectors, dtype=np.float32)
    X -= X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1].copy()  # descending
    eigenvalues = np.clip(eigenvalues, 0, None)

    pr = compute_participation_ratio(vectors)
    d_eff = max(1, int(round(pr)))

    return d_eff, eigenvalues

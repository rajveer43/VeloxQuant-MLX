from __future__ import annotations

import math

import numpy as np

# mx.hadamard_transform requires d = m * 2^k where m is in this set
_HADAMARD_VALID_M = {1, 12, 20, 28}


def is_hadamard_compatible(d: int) -> bool:
    """Return True if d is supported by mx.hadamard_transform.

    MLX requires d = m * 2^k where m in {1, 12, 20, 28} and k >= 0 -- except
    for the bare non-trivial m values (d == 12, 20, 28 exactly, i.e. k=0),
    which crash mx.hadamard_transform during Metal shader compilation on the
    installed MLX version (0.32.0) rather than raising a catchable, sensible
    error (#67). k=0 with m=1 (d=1) is unaffected and works fine, since it's
    just the identity transform. All powers of 2 work (m=1, k>=0). Examples
    that do NOT work: 576=9*64, 192=3*64.
    """
    if d < 1:
        return False
    for m in _HADAMARD_VALID_M:
        if d % m == 0:
            remainder = d // m
            if remainder & (remainder - 1) == 0:  # power of 2
                if m != 1 and remainder == 1:
                    continue
                return True
    return False


def make_hadamard_diagonal(d: int, seed: int = 42) -> np.ndarray:
    """Generate a random ±1 diagonal vector for randomized Hadamard transform.

    The randomized Hadamard preconditioner applies H @ diag(D) where H is the
    Walsh-Hadamard matrix and D is this ±1 diagonal. Only D needs to be stored
    (d scalars vs d² for QR rotation).

    d must satisfy mx.hadamard_transform's constraint: d = m * 2^k where
    m in {1, 12, 20, 28}. All powers of 2 (64, 128, 256, ...) satisfy this.

    Args:
        d: Vector dimension.
        seed: NumPy random seed for reproducibility.

    Returns:
        Float32 array of shape (d,) with entries in {-1, +1}.
    """
    if d < 1:
        raise ValueError(f"make_hadamard_diagonal: d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)


def make_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """Generate a d×d random orthogonal rotation matrix via QR decomposition.

    Draws G ~ N(0, 1)^{d×d} and returns Q from G = QR (economy QR).
    The resulting Q is a Haar-distributed orthogonal matrix.

    Args:
        d: Matrix dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (d, d) with orthonormal rows (Q @ Q.T ≈ I).

    Raises:
        ValueError: If d < 1.
    """
    if d < 1:
        raise ValueError(f"make_rotation_matrix: d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float64)
    Q, _ = np.linalg.qr(G)
    return Q.astype(np.float64)


def make_jl_matrix(d: int, m: int, seed: int = 42) -> np.ndarray:
    """Generate an m×d Gaussian JL projection matrix.

    Each row is drawn i.i.d. from N(0, I_d). This is the correct
    construction for the QJL sign-based inner product estimator to be
    unbiased:

        E[sqrt(pi/2)/m * ||k|| * sum_i sign(s_i·k)(s_i·q)] = <q, k>

    Unlike orthogonal JL, Gaussian JL allows m > d.

    Args:
        d: Input dimension (must be >= 1).
        m: Sketch dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (m, d) with i.i.d. N(0,1) entries.

    Raises:
        ValueError: If d < 1 or m < 1.
    """
    if d < 1:
        raise ValueError(f"make_jl_matrix: d must be >= 1, got {d}")
    if m < 1:
        raise ValueError(f"make_jl_matrix: m must be >= 1, got {m}")

    rng = np.random.default_rng(seed + 1)
    S = rng.standard_normal((m, d)).astype(np.float64)
    return S

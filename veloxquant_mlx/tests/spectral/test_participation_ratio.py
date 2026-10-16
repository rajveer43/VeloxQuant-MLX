from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.spectral.participation_ratio import (
    compute_participation_ratio,
    compute_spectral_gap,
)


def _low_rank_data(d: int = 128, rank: int = 4, n: int = 512, seed: int = 0) -> np.ndarray:
    """Generate synthetic data that lies in a rank-dimensional subspace."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((d, rank)))
    coords = rng.standard_normal((n, rank)).astype(np.float32)
    noise = rng.standard_normal((n, d)).astype(np.float32) * 0.01
    return (coords @ basis.T).astype(np.float32) + noise


def test_participation_ratio_rank4():
    X = _low_rank_data(d=128, rank=4, n=512)
    pr = compute_participation_ratio(X)
    # Should be close to 4 (allowing noise slack)
    assert 2.0 <= pr <= 12.0, f"Expected d_eff ≈ 4 for rank-4 data, got {pr:.2f}"


def test_participation_ratio_uniform():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((512, 128)).astype(np.float32)
    pr = compute_participation_ratio(X)
    # Uniform data should have d_eff close to 128
    assert pr > 50.0, f"Uniform data should have high d_eff, got {pr:.2f}"


def test_participation_ratio_single_dim():
    rng = np.random.default_rng(2)
    n = 256
    d = 128
    # All variance in dim 0
    X = np.zeros((n, d), dtype=np.float32)
    X[:, 0] = rng.standard_normal(n).astype(np.float32)
    pr = compute_participation_ratio(X)
    assert pr < 3.0, f"Single-dim data should have d_eff ≈ 1, got {pr:.2f}"


def test_compute_spectral_gap_returns_descending_eigenvalues():
    X = _low_rank_data(d=128, rank=4, n=512)
    d_eff, eigenvalues = compute_spectral_gap(X)
    assert isinstance(d_eff, int)
    assert d_eff >= 1
    assert len(eigenvalues) == 128
    # Eigenvalues should be in descending order
    assert np.all(np.diff(eigenvalues) <= 1e-6), "Eigenvalues should be non-increasing"
    # First eigenvalue dominates
    assert eigenvalues[0] > eigenvalues[10], "First eigenvalue should be largest"


def test_compute_spectral_gap_d_eff_range():
    X = _low_rank_data(d=128, rank=4, n=512)
    d_eff, _ = compute_spectral_gap(X)
    assert 1 <= d_eff <= 128


@pytest.mark.parametrize("rank,n", [(4, 512), (50, 512), (1, 256)])
def test_participation_ratio_tracks_rank(rank: int, n: int):
    X = _low_rank_data(d=128, rank=rank, n=n, seed=rank)
    pr = compute_participation_ratio(X)
    # Allow 5x slack but should scale with rank
    assert pr <= rank * 5 + 5, f"rank={rank}: d_eff={pr:.2f} too high"


def test_participation_ratio_is_scale_invariant_for_small_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 64)) * np.linspace(3, 0.1, 64)
    expected = compute_participation_ratio(X)
    assert expected > 10
    assert compute_participation_ratio(X * 1e-4) == pytest.approx(expected, rel=1e-3)

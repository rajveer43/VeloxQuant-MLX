"""Unit tests for the PALU low-rank projection primitives.

Covers the pure maths in ``veloxquant_mlx.quantizers.palu``: head grouping,
group-head SVD, projection, and reconstruction — independent of the cache.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.palu import (
    group_head_svd,
    head_group_bounds,
    project_to_latent,
    quantize_latent,
    reconstruct_from_latent,
)


# ------------------------------------------------------------------
# head_group_bounds
# ------------------------------------------------------------------


def test_head_group_bounds_even() -> None:
    assert head_group_bounds(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_head_group_bounds_uneven() -> None:
    # 7 heads into 3 groups → sizes 3, 2, 2
    assert head_group_bounds(7, 3) == [(0, 3), (3, 5), (5, 7)]


def test_head_group_bounds_clamped() -> None:
    # More groups than heads → one group per head, no empties.
    b = head_group_bounds(2, 8)
    assert b == [(0, 1), (1, 2)]
    # n_groups < 1 clamps to 1.
    assert head_group_bounds(4, 0) == [(0, 4)]


def test_head_group_bounds_cover_all_heads() -> None:
    for H, G in [(4, 1), (5, 2), (12, 4), (3, 3)]:
        bounds = head_group_bounds(H, G)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == H
        # contiguous, no gaps/overlaps
        for (lo, hi), (nlo, _) in zip(bounds, bounds[1:]):
            assert hi == nlo


# ------------------------------------------------------------------
# group_head_svd
# ------------------------------------------------------------------


def test_group_head_svd_shapes_explicit_rank() -> None:
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((3, 40, 64)).astype(np.float32))  # [G, S, D]
    V, mu, sv = group_head_svd(x, rank=12)
    mx.eval(V, mu, sv)
    assert V.shape == (64, 12)
    assert mu.shape == (64,)
    assert sv.shape == (12,)
    # singular values descending
    sv_l = sv.tolist()
    assert all(sv_l[i] >= sv_l[i + 1] - 1e-4 for i in range(len(sv_l) - 1))


def test_group_head_svd_energy_threshold() -> None:
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((2, 64, 48)).astype(np.float32))
    V, mu, sv = group_head_svd(x, rank=None, energy_threshold=0.9)
    mx.eval(V)
    assert 1 <= V.shape[1] <= 48


def test_group_head_svd_recovers_low_rank_subspace() -> None:
    """On rank-r data the projection should reconstruct it near-perfectly."""
    rng = np.random.default_rng(2)
    S, D, r = 80, 48, 6
    U = rng.standard_normal((S, r)).astype(np.float32)
    W = rng.standard_normal((r, D)).astype(np.float32)
    X = (U @ W).astype(np.float32)
    x_group = mx.array(X[None])  # [1, S, D]
    V, mu, sv = group_head_svd(x_group, rank=r)
    L = project_to_latent(mx.array(X), V, mu)
    recon = reconstruct_from_latent(L, V, mu)
    mx.eval(recon)
    mse = float(mx.mean((recon.astype(mx.float32) - mx.array(X)) ** 2).item())
    assert mse < 1e-3, f"rank-{r} reconstruction MSE too high: {mse}"


# ------------------------------------------------------------------
# project / reconstruct round-trip
# ------------------------------------------------------------------


def test_project_reconstruct_roundtrip_shapes() -> None:
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((50, 64)).astype(np.float32))
    V, mu, _ = group_head_svd(x[None], rank=20)
    L = project_to_latent(x, V, mu)
    assert L.shape == (50, 20)
    recon = reconstruct_from_latent(L, V, mu)
    assert recon.shape == (50, 64)
    assert recon.dtype == mx.float16


# ------------------------------------------------------------------
# quantize_latent
# ------------------------------------------------------------------


def test_quantize_latent_runs_and_shapes() -> None:
    rng = np.random.default_rng(4)
    L = mx.array(rng.standard_normal((64, 16)).astype(np.float32))
    sv = mx.array(np.linspace(10, 1, 16).astype(np.float32))
    Lq = quantize_latent(L, sv, hi_bit=4, lo_bit=2, hi_fraction=0.25, group_size=16)
    mx.eval(Lq)
    assert Lq.shape == (64, 16)
    assert Lq.dtype == mx.float16
    # Quantized latents differ from the original (lossy) but stay finite.
    assert not mx.any(mx.isnan(Lq)).item()

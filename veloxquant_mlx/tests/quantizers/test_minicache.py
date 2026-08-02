"""Unit tests for MiniCache SLERP-merge primitives."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.minicache import (
    merge_pair,
    pair_layers_depth,
    reconstruct_layer,
    slerp,
    to_mag_dir,
)


# ------------------------------------------------------------------
# pair_layers_depth
# ------------------------------------------------------------------


def test_pair_layers_depth_early_all_primary() -> None:
    roles = pair_layers_depth(8, start_frac=0.5, group_size=2)
    assert all(r[0] == "primary" for r in roles[:4])


def test_pair_layers_depth_deep_has_merge() -> None:
    roles = pair_layers_depth(8, start_frac=0.5, group_size=2)
    assert any(r[0] == "merge" for r in roles[4:])


def test_pair_layers_depth_group_pairs() -> None:
    roles = pair_layers_depth(4, start_frac=0.0, group_size=2)
    # group 0 = (primary, merge), group 1 = (primary, merge)
    assert roles == [("primary", 0), ("merge", 0), ("primary", 1), ("merge", 1)]


def test_pair_layers_depth_rejects_small_group() -> None:
    with pytest.raises(ValueError):
        pair_layers_depth(8, group_size=1)


# ------------------------------------------------------------------
# slerp
# ------------------------------------------------------------------


def test_slerp_endpoints() -> None:
    d0 = mx.array([[1.0, 0.0, 0.0]])
    d1 = mx.array([[0.0, 1.0, 0.0]])
    s0 = slerp(d0, d1, 0.0)
    s1 = slerp(d0, d1, 1.0)
    mx.eval(s0, s1)
    assert np.allclose(np.array(s0), np.array(d0), atol=1e-5)
    assert np.allclose(np.array(s1), np.array(d1), atol=1e-5)


def test_slerp_output_unit_norm() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 8)).astype(np.float32)
    b = rng.standard_normal((10, 8)).astype(np.float32)
    _, da = to_mag_dir(mx.array(a))
    _, db = to_mag_dir(mx.array(b))
    s = slerp(da, db, 0.5)
    norms = mx.sqrt(mx.sum(s * s, axis=-1))
    mx.eval(norms)
    assert np.allclose(np.array(norms), 1.0, atol=1e-4)


def test_slerp_collinear_fallback() -> None:
    d0 = mx.array([[1.0, 0.0]])
    d1 = mx.array([[1.0, 0.0]])  # identical → sin(omega)=0 fallback
    s = slerp(d0, d1, 0.5)
    mx.eval(s)
    assert np.allclose(np.array(s), np.array(d0), atol=1e-5)


# ------------------------------------------------------------------
# merge / reconstruct
# ------------------------------------------------------------------


def test_to_mag_dir_recovers_vector() -> None:
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((5, 8)).astype(np.float32))
    mag, d = to_mag_dir(x)
    recon = mag * d
    mx.eval(recon)
    assert np.allclose(np.array(recon), np.array(x), atol=1e-4)


def test_merge_similar_low_error() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal((16, 32)).astype(np.float32)
    xp = mx.array(base)
    xm = mx.array(base + rng.standard_normal((16, 32)).astype(np.float32) * 0.02)
    res = merge_pair(xp, xm, retention_threshold=0.9, t=0.5)
    rp = reconstruct_layer(res, "primary")
    rm = reconstruct_layer(res, "merge")
    mx.eval(rp, rm)
    assert float(mx.mean((rp.astype(mx.float32) - xp) ** 2).item()) < 0.05
    assert float(mx.mean((rm.astype(mx.float32) - xm) ** 2).item()) < 0.05


def test_merge_preserves_magnitude() -> None:
    """Merged reconstruction keeps each layer's own per-token magnitude."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((8, 16)).astype(np.float32)
    xp = mx.array(base * 1.0)
    xm = mx.array(base * 3.0)  # same direction, 3x magnitude
    res = merge_pair(xp, xm, retention_threshold=0.99, t=0.5)
    rp = reconstruct_layer(res, "primary")
    rm = reconstruct_layer(res, "merge")
    mx.eval(rp, rm)
    # magnitudes preserved even though direction is shared
    mag_rp = mx.sqrt(mx.sum(rp.astype(mx.float32) ** 2, axis=-1))
    mag_rm = mx.sqrt(mx.sum(rm.astype(mx.float32) ** 2, axis=-1))
    ratio = float(mx.mean(mag_rm / mx.maximum(mag_rp, 1e-6)).item())
    assert ratio == pytest.approx(3.0, abs=0.1)


def test_opposite_directions_retained() -> None:
    rng = np.random.default_rng(4)
    xp = mx.array(rng.standard_normal((6, 8)).astype(np.float32))
    xm = mx.array(-np.array(xp))
    res = merge_pair(xp, xm, retention_threshold=0.9)
    assert bool(mx.all(res.retained).item())
    rp = reconstruct_layer(res, "primary")
    mx.eval(rp)
    assert float(mx.mean((rp.astype(mx.float32) - xp) ** 2).item()) < 1e-4

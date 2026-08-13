"""Tests for the VecInfer algorithmic primitives."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_keys,
    apply_dual_transform_queries,
    calibrate_smooth_factors,
    compute_query_lut,
    dequantize_vq,
    quantize_vq,
    train_codebook,
    walsh_hadamard_matrix,
)


def test_walsh_hadamard_orthogonality() -> None:
    """H @ H.T must equal identity (Walsh-Hadamard is orthonormal)."""
    for d in (2, 4, 8, 16, 32, 128):
        H = walsh_hadamard_matrix(d)
        prod = np.asarray(H @ H.T)
        eye = np.eye(d, dtype=np.float32)
        assert np.allclose(prod, eye, atol=1e-4), f"d={d} fails orthogonality"


def test_walsh_hadamard_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        walsh_hadamard_matrix(7)


def test_dual_transform_preserves_inner_product() -> None:
    """q_tilde @ K_tilde.T must equal q @ K.T (computational invariance)."""
    d, n_heads, n_tokens = 64, 4, 16
    rng = np.random.default_rng(0)
    K = mx.array(rng.standard_normal((n_tokens, n_heads, d)).astype(np.float32))
    q = mx.array(rng.standard_normal((1, n_heads, d)).astype(np.float32))

    smooth = calibrate_smooth_factors(K)
    H = walsh_hadamard_matrix(d)

    K_tilde = apply_dual_transform_keys(K, smooth, H)
    q_tilde = apply_dual_transform_queries(q, smooth, H)

    # Per-head inner products
    for h in range(n_heads):
        orig = np.asarray(q[0, h] @ K[:, h].T)
        transformed = np.asarray(q_tilde[0, h] @ K_tilde[:, h].T)
        assert np.allclose(orig, transformed, atol=1e-3), (
            f"head {h}: max diff {np.max(np.abs(orig - transformed))}"
        )


def test_calibrate_smooth_factors_shape() -> None:
    K3 = mx.random.normal((100, 4, 64))
    sm3 = calibrate_smooth_factors(K3)
    assert sm3.shape == (4, 64)

    K2 = mx.random.normal((100, 64))
    sm2 = calibrate_smooth_factors(K2)
    assert sm2.shape == (64,)


def test_smooth_factor_reduces_channel_spread() -> None:
    """After dividing by lambda, per-channel max-abs should be ~1."""
    rng = np.random.default_rng(0)
    K_np = rng.standard_normal((1000, 64)).astype(np.float32)
    K_np[:, 5] *= 20.0  # inject outlier channel
    K = mx.array(K_np)
    smooth = calibrate_smooth_factors(K)
    K_smooth = K / smooth
    max_per_channel = np.asarray(mx.max(mx.abs(K_smooth), axis=0))
    # max-abs / sqrt(max-abs) = sqrt(max-abs), bounded for unit-scale inputs
    assert max_per_channel.max() < max_per_channel.min() * 50.0


def test_train_codebook_converges() -> None:
    """Final clustering inertia should be lower than initial."""
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2000, 4)).astype(np.float32)
    x = mx.array(x_np)

    cb = train_codebook(x, n_centroids=16, max_iter=30, seed=0)
    cb_np = np.asarray(cb)
    # Inertia using trained codebook
    diff = x_np[:, None, :] - cb_np[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    trained_inertia = np.min(d2, axis=1).sum()

    # Inertia using a random codebook
    rand_cb = rng.standard_normal((16, 4)).astype(np.float32)
    diff = x_np[:, None, :] - rand_cb[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    rand_inertia = np.min(d2, axis=1).sum()

    assert trained_inertia < rand_inertia, (
        f"trained {trained_inertia:.2f} should be < random {rand_inertia:.2f}"
    )


def test_quantize_dequantize_roundtrip_error_decreases_with_b() -> None:
    """Reconstruction MSE must shrink as codebook bit-width grows."""
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((500, 64)).astype(np.float32)
    x = mx.array(x_np)
    sub_dim = 4
    train_x = mx.array(x_np.reshape(-1, sub_dim))

    errors = []
    for b in (4, 6, 8):
        cb = train_codebook(train_x, n_centroids=2**b, max_iter=20, seed=0)
        idx = quantize_vq(x, cb, sub_dim)
        recon = dequantize_vq(idx, cb)
        mse = float(mx.mean((recon - x) ** 2))
        errors.append(mse)
    # Strictly decreasing
    assert errors[0] > errors[1] > errors[2], f"errors not decreasing: {errors}"


def test_compute_query_lut_matches_explicit_dot() -> None:
    """LUT lookup must equal q_tilde @ K_hat.T for the dequantized keys."""
    rng = np.random.default_rng(0)
    sub_dim = 4
    D = 16
    cb = mx.array(rng.standard_normal((8, sub_dim)).astype(np.float32))
    q = mx.array(rng.standard_normal((D,)).astype(np.float32))

    K = mx.array(rng.standard_normal((5, D)).astype(np.float32))
    idx = quantize_vq(K, cb, sub_dim)
    K_hat = dequantize_vq(idx, cb)
    explicit = np.asarray(q @ K_hat.T)

    lut = compute_query_lut(q, cb, sub_dim)  # [n_sub, n_centroids]
    lut_np = np.asarray(lut)
    idx_np = np.asarray(idx)  # [5, n_sub]
    via_lut = np.array(
        [sum(lut_np[s, idx_np[t, s]] for s in range(D // sub_dim)) for t in range(5)],
        dtype=np.float32,
    )
    assert np.allclose(explicit, via_lut, atol=1e-4), f"explicit {explicit} vs lut {via_lut}"


def test_dual_transform_with_1d_smooth() -> None:
    """1D smooth factors (no head axis) must also preserve invariance."""
    d = 32
    rng = np.random.default_rng(1)
    K = mx.array(rng.standard_normal((20, d)).astype(np.float32))
    q = mx.array(rng.standard_normal((d,)).astype(np.float32))
    smooth = calibrate_smooth_factors(K)
    H = walsh_hadamard_matrix(d)

    K_t = apply_dual_transform_keys(K, smooth, H)
    q_t = apply_dual_transform_queries(q, smooth, H)

    orig = np.asarray(q @ K.T)
    new = np.asarray(q_t @ K_t.T)
    assert np.allclose(orig, new, atol=1e-3)


# ---------------------------------------------------------------------------
# Regression for #74: a genuine 3D per-head layout [n_heads, seq, head_dim]
# (a natural single-batch-item shape, matching the head axis at -3 the same
# as 4D [B, H, S, D]) used to require K.ndim >= 4 to reach the per-head
# broadcast branch, so it silently fell into the GQA head-averaging branch
# instead -- producing a plausible-looking but numerically wrong result with
# no error.
# ---------------------------------------------------------------------------
def test_dual_transform_keys_3d_per_head_matches_per_head_math() -> None:
    d, n_heads, seq = 8, 3, 5
    rng = np.random.default_rng(1)
    K3d = mx.array(rng.standard_normal((n_heads, seq, d)).astype(np.float32))
    smooth2d = mx.array(rng.uniform(0.5, 2.0, size=(n_heads, d)).astype(np.float32))
    H = walsh_hadamard_matrix(d)

    out = apply_dual_transform_keys(K3d, smooth2d, H)

    correct = (np.asarray(K3d) / np.asarray(smooth2d)[:, None, :]) @ np.asarray(H)
    assert np.allclose(correct, np.asarray(out), atol=1e-5)

    # Must NOT match the (incorrect) averaged-across-heads result.
    averaged = (np.asarray(K3d) / np.asarray(smooth2d).mean(axis=0)) @ np.asarray(H)
    assert not np.allclose(averaged, np.asarray(out), atol=1e-5)


def test_dual_transform_queries_3d_per_head_matches_per_head_math() -> None:
    d, n_heads, seq = 8, 3, 5
    rng = np.random.default_rng(2)
    q3d = mx.array(rng.standard_normal((n_heads, seq, d)).astype(np.float32))
    smooth2d = mx.array(rng.uniform(0.5, 2.0, size=(n_heads, d)).astype(np.float32))
    H = walsh_hadamard_matrix(d)

    out = apply_dual_transform_queries(q3d, smooth2d, H)

    correct = (np.asarray(q3d) * np.asarray(smooth2d)[:, None, :]) @ np.asarray(H)
    assert np.allclose(correct, np.asarray(out), atol=1e-5)

    averaged = (np.asarray(q3d) * np.asarray(smooth2d).mean(axis=0)) @ np.asarray(H)
    assert not np.allclose(averaged, np.asarray(out), atol=1e-5)


def test_dual_transform_keys_3d_per_head_preserves_inner_product() -> None:
    """End-to-end: 3D per-head K/q with the dual transform still preserves
    q @ K.T per head, the actual invariant the transform exists for."""
    d, n_heads, seq = 16, 4, 6
    rng = np.random.default_rng(3)
    K3d = mx.array(rng.standard_normal((n_heads, seq, d)).astype(np.float32))
    q3d = mx.array(rng.standard_normal((n_heads, 1, d)).astype(np.float32))
    smooth2d = mx.array(rng.uniform(0.5, 2.0, size=(n_heads, d)).astype(np.float32))
    H = walsh_hadamard_matrix(d)

    K_t = apply_dual_transform_keys(K3d, smooth2d, H)
    q_t = apply_dual_transform_queries(q3d, smooth2d, H)

    for h in range(n_heads):
        orig = np.asarray(q3d[h] @ K3d[h].T)
        transformed = np.asarray(q_t[h] @ K_t[h].T)
        assert np.allclose(orig, transformed, atol=1e-3), f"head {h} mismatch"


def test_dual_transform_keys_genuine_gqa_head_mismatch_still_averages() -> None:
    """A real head-count mismatch (e.g. GQA: smooth has more heads than K)
    must still fall through to the averaging branch -- this is the case the
    branch is actually for, and must not regress."""
    d, n_kv_heads, n_q_heads, seq = 8, 2, 6, 5
    rng = np.random.default_rng(4)
    K3d = mx.array(rng.standard_normal((n_kv_heads, seq, d)).astype(np.float32))
    smooth2d = mx.array(rng.uniform(0.5, 2.0, size=(n_q_heads, d)).astype(np.float32))
    H = walsh_hadamard_matrix(d)

    out = apply_dual_transform_keys(K3d, smooth2d, H)

    averaged = (np.asarray(K3d) / np.asarray(smooth2d).mean(axis=0)) @ np.asarray(H)
    assert np.allclose(averaged, np.asarray(out), atol=1e-5)

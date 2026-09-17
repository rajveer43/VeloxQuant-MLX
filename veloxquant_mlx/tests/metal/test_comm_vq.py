"""Correctness tests for the CommVQ decode + RoPE fused Metal kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal._comm_vq import comm_vq_decode_metal

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(not metal_available(), reason="requires Metal GPU"),
]


def _reference_decode(indices, codebook, positions, inv_freq, n_cb, sub_dim):
    """NumPy reference: gather + RoPE, mirroring comm_vq_decode.metal."""
    N = indices.shape[0]
    D = n_cb * sub_dim
    half = D // 2
    gathered = np.empty((N, D), dtype=np.float32)
    for b in range(N):
        for cb_i in range(n_cb):
            idx = int(indices[b, cb_i])
            gathered[b, cb_i * sub_dim : (cb_i + 1) * sub_dim] = codebook[cb_i, idx]
    x1 = gathered[:, :half]
    x2 = gathered[:, half:]
    angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
    cos_v, sin_v = np.cos(angles), np.sin(angles)
    out = np.concatenate([x1 * cos_v - x2 * sin_v, x1 * sin_v + x2 * cos_v], axis=1)
    return out.astype(np.float16)


@pytest.mark.parametrize(
    "n_cb,sub_dim,cb_size,N",
    [
        (4, 16, 16, 8),  # D=64, divides the 256-wide threadgroup evenly
        (4, 75, 16, 5),  # D=300 — not a multiple of 256 (the reported failure case)
        (3, 20, 8, 4),  # D=60, threadgroup=min(D,256)=60 also not a divisor of N*D generally
        (1, 2, 4, 1),  # smallest possible: single pair, single token
    ],
)
def test_matches_reference(n_cb, sub_dim, cb_size, N):
    rng = np.random.default_rng(0)
    D = n_cb * sub_dim
    indices_np = rng.integers(0, cb_size, size=(N, n_cb)).astype(np.uint8)
    codebook_np = rng.standard_normal((n_cb, cb_size, sub_dim)).astype(np.float32)
    positions_np = np.arange(N, dtype=np.int32)
    inv_freq_np = 1.0 / (10000.0 ** (np.arange(D // 2, dtype=np.float32) / (D // 2)))

    expected = _reference_decode(indices_np, codebook_np, positions_np, inv_freq_np, n_cb, sub_dim)

    out = comm_vq_decode_metal(
        mx.array(indices_np),
        mx.array(codebook_np).astype(mx.float16),
        mx.array(positions_np),
        mx.array(inv_freq_np),
        n_cb,
        sub_dim,
        cb_size,
    )
    mx.eval(out)

    assert out.shape == (N, D)
    err = np.max(np.abs(np.array(out).astype(np.float32) - expected.astype(np.float32)))
    assert err < 5e-2, f"max abs error {err} for D={D}, N={N}"


def test_no_out_of_bounds_on_non_multiple_grid():
    """D=300 forces grid padding past N*D; padding threads must be no-ops.

    Regression for #401: the grid used to be dispatched unrounded, so N*D not
    dividing the threadgroup size risked either an MLX-side assertion or a
    silent shortfall. This runs a shape where N*D is not itself a multiple of
    the threadgroup size and checks every output element is finite and
    populated (no leftover NaN/garbage from unwritten tail elements).
    """
    n_cb, sub_dim, cb_size, N = 4, 75, 16, 7  # D=300; N*D=2100, tg=min(300,256)=256
    rng = np.random.default_rng(1)
    D = n_cb * sub_dim
    indices_np = rng.integers(0, cb_size, size=(N, n_cb)).astype(np.uint8)
    codebook_np = rng.standard_normal((n_cb, cb_size, sub_dim)).astype(np.float32)
    positions_np = np.arange(N, dtype=np.int32)
    inv_freq_np = 1.0 / (10000.0 ** (np.arange(D // 2, dtype=np.float32) / (D // 2)))

    out = comm_vq_decode_metal(
        mx.array(indices_np),
        mx.array(codebook_np).astype(mx.float16),
        mx.array(positions_np),
        mx.array(inv_freq_np),
        n_cb,
        sub_dim,
        cb_size,
    )
    mx.eval(out)
    out_np = np.array(out)
    assert out_np.shape == (N, D)
    assert np.all(np.isfinite(out_np.astype(np.float32)))

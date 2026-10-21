"""palu's prefill fit must not SVD the tall token block (#846).

``group_head_svd`` factorised the whole ``[G*S, D]`` matrix to use only ``V``
and the singular values (1.2 s for 8 heads at 2,048 tokens). It now decomposes
the ``D x D`` Gram matrix in float64. The subspace and spectrum must still
match the exact SVD, including on steeply decaying (key-like) spectra, where a
float32 Gram would lose the weaker retained directions.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers import palu
from veloxquant_mlx.quantizers.palu import group_head_svd

G, S, D = 4, 512, 64


def _keys(decay: float, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    scale = decay ** np.arange(D)
    x = rng.standard_normal((G, S, D)) * scale @ basis.T
    return mx.array(x.astype(np.float32))


def test_fit_does_not_call_svd(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("group_head_svd must not SVD the tall block")

    monkeypatch.setattr(palu.mx.linalg, "svd", boom)
    group_head_svd(_keys(0.9), rank=8)


@pytest.mark.parametrize("decay", [1.0, 0.9, 0.8])
@pytest.mark.parametrize("rank", [4, 16, 40])
def test_subspace_and_spectrum_match_exact_svd(decay, rank):
    x = _keys(decay)
    V, mu, s = group_head_svd(x, rank=rank)

    flat = np.array(x).astype(np.float64).reshape(G * S, D)
    centered = flat - flat.mean(axis=0)
    _, s_ref, vt_ref = np.linalg.svd(centered, full_matrices=False)
    v_ref = vt_ref[:rank].T

    np.testing.assert_allclose(np.array(mu), flat.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(np.array(s), s_ref[:rank], rtol=1e-4)
    # Compare the retained subspace (signs/rotations inside it are arbitrary).
    proj = np.array(V).astype(np.float64)
    err = np.linalg.norm(proj @ proj.T - v_ref @ v_ref.T)
    assert err < 1e-3, f"retained subspace differs from exact SVD by {err:.2e}"


def test_energy_threshold_rank_matches_exact_svd():
    x = _keys(0.85)
    V, _, s = group_head_svd(x, rank=None, energy_threshold=0.9)
    flat = np.array(x).astype(np.float64).reshape(G * S, D)
    s_ref = np.linalg.svd(flat - flat.mean(axis=0), compute_uv=False)
    expected = int(np.searchsorted(np.cumsum(s_ref) / s_ref.sum(), 0.9) + 1)
    assert V.shape[1] == expected

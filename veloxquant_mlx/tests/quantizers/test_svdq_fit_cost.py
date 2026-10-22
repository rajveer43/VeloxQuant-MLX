"""svdq's prefill fit must not SVD the tall key block (#875).

``svd_compress_keys_batched`` factorised the whole ``[H, S, D]`` block, building
an ``[S, S]`` U it never used (1.2 s for 8 heads at 4,096 tokens, 4.9 s at
8,192). It now decomposes the ``D x D`` Gram matrix in float64. The retained
subspace must still be optimal, including on steeply decaying spectra, where a
float32 Gram would lose the weaker retained directions.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers import svdq
from veloxquant_mlx.quantizers.svdq import svd_compress_keys_batched

H, S, D = 3, 256, 32


def _keys(decay: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(H):
        q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        out.append(rng.standard_normal((S, D)) * decay ** np.arange(D) @ q.T)
    return np.stack(out).astype(np.float32)


def test_fit_does_not_call_svd(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("svd_compress_keys_batched must not SVD the tall block")

    monkeypatch.setattr(svdq.mx.linalg, "svd", boom)
    svd_compress_keys_batched(mx.array(_keys(0.9)), rank=8)
    svd_compress_keys_batched(mx.array(_keys(0.9)), rank=None)


@pytest.mark.parametrize("decay", [1.0, 0.9, 0.75])
@pytest.mark.parametrize("rank", [2, 8, 20])
def test_retained_subspace_is_optimal(decay, rank):
    x = _keys(decay)
    L_list, V_list, mean_list, s_list = svd_compress_keys_batched(mx.array(x), rank=rank)
    for h in range(H):
        centered = x[h].astype(np.float64) - np.array(mean_list[h], dtype=np.float64)
        s = np.linalg.svd(centered, compute_uv=False)
        np.testing.assert_allclose(np.array(s_list[h]), s[:rank], rtol=1e-4)
        v = np.array(V_list[h], dtype=np.float64)  # [D, r]
        err = np.linalg.norm(centered - centered @ v @ v.T)
        best = np.sqrt((s[rank:] ** 2).sum())  # Eckart-Young optimum
        assert err <= best * (1 + 1e-4) + 1e-5, f"rank {rank}: {err} vs optimal {best}"


def test_energy_threshold_ranks_match_exact_svd():
    x = _keys(0.8)
    _, V_list, mean_list, _ = svd_compress_keys_batched(
        mx.array(x), rank=None, energy_threshold=0.95
    )
    for h in range(H):
        centered = x[h].astype(np.float64) - np.array(mean_list[h], dtype=np.float64)
        s = np.linalg.svd(centered, compute_uv=False)
        expected = int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1)
        assert V_list[h].shape[1] == expected

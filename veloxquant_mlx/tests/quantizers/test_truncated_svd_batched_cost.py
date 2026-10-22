"""``_truncated_svd_batched`` must not SVD the tall block (#873).

GEAR's prefill spent 2.48 of 2.50 s (4,096 tokens, 8 heads) in
``mx.linalg.svd`` building a full ``[N, N]`` U it never used, so prefill grew
~quadratically with the prompt. The factors now come from the ``D x D`` Gram
matrix in float64. The low-rank approximation must still be optimal, including
on steeply decaying (key/residual-like) spectra where a float32 Gram would lose
the weaker retained directions.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers import _quant_utils
from veloxquant_mlx.quantizers._quant_utils import _truncated_svd_batched

BH, N, D = 3, 256, 32


def _batch(decay: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(BH):
        q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        out.append(rng.standard_normal((N, D)) * decay ** np.arange(D) @ q.T)
    return np.stack(out).astype(np.float32)


def test_batched_path_does_not_call_svd(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("_truncated_svd_batched must not SVD the tall block")

    monkeypatch.setattr(_quant_utils.mx.linalg, "svd", boom)
    _truncated_svd_batched(mx.array(_batch(0.9)), rank=4)
    _truncated_svd_batched(mx.array(_batch(0.9)), rank=None)


@pytest.mark.parametrize("decay", [1.0, 0.9, 0.75])
@pytest.mark.parametrize("rank", [2, 8, 20])
def test_low_rank_approximation_is_optimal(decay, rank):
    x = _batch(decay)
    L, R, ranks = _truncated_svd_batched(mx.array(x), rank=rank)
    assert ranks == [rank] * BH
    for i in range(BH):
        _, s, _ = np.linalg.svd(x[i].astype(np.float64), full_matrices=False)
        best = np.sqrt((s[rank:] ** 2).sum())  # Eckart-Young optimum
        approx = np.array(L[i, :, :rank]) @ np.array(R[i, :rank, :])
        err = np.linalg.norm(x[i] - approx)
        assert err <= best * (1 + 1e-4) + 1e-5, f"rank {rank}: {err} vs optimal {best}"


def test_energy_threshold_ranks_match_exact_svd():
    x = _batch(0.8)
    _, _, ranks = _truncated_svd_batched(mx.array(x), rank=None, energy_threshold=0.9)
    for i in range(BH):
        s = np.linalg.svd(x[i].astype(np.float64), compute_uv=False)
        expected = int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1)
        assert ranks[i] == expected

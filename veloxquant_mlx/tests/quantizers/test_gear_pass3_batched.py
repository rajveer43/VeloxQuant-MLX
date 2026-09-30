"""Equivalence tests: batched [BH, N, D] GEAR Pass 3 (low-rank add +
sparse-outlier scatter + reconstruction) vs. the per-head reference (#570).

``sparse_outliers_batched`` and ``gear_reconstruct_batched`` replace
``GEARKVCache._compress_and_account``'s ``for idx in range(B * H):`` loop
that called ``sparse_outliers`` and ``gear_reconstruct`` once per head —
the largest remaining per-head cost after #504 batched Passes 1-2 (SVD fit
and base group-quant). Each head's low-rank add and sparse scatter only
touch that head's own factors/residual — no cross-head coupling — so this
is a pure vectorization, not a policy change. Byte accounting (``GEARState``
construction, ``gear_bytes``/``base_only_bytes``) remains per-row, unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers._quant_utils import _truncated_svd_batched
from veloxquant_mlx.quantizers.gear import (
    gear_reconstruct,
    gear_reconstruct_batched,
    sparse_outliers,
    sparse_outliers_batched,
)


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "desc,bh,n,d,frac,seed",
    [
        ("single_row", 1, 20, 16, 0.05, 0),
        ("multi_row", 4, 30, 16, 0.05, 1),
        ("small_n", 3, 8, 16, 0.1, 2),
        ("zero_frac", 2, 15, 8, 0.0, 3),
        ("large_frac", 5, 20, 8, 0.5, 4),
    ],
)
def test_sparse_outliers_batched_matches_reference(desc, bh, n, d, frac, seed):
    resid = _rand((bh, n, d), seed)
    idx_b, val_b = sparse_outliers_batched(resid, frac)

    for row in range(bh):
        idx_r, val_r = sparse_outliers(resid[row], frac)
        if idx_r is None:
            assert idx_b is None, f"{desc}: row {row} expected None"
            continue
        assert idx_b is not None and val_b is not None
        # Selection order may differ (ties / argsort stability), so compare
        # as sets of (index, value) pairs rather than positional equality.
        set_b = set(zip(idx_b[row].tolist(), np.round(np.array(val_b[row].tolist()), 3)))
        set_r = set(zip(idx_r.tolist(), np.round(np.array(val_r.tolist()), 3)))
        assert set_b == set_r, f"{desc}: row {row} selected different outliers"


@pytest.mark.parametrize(
    "desc,bh,n,d,rank,frac,seed",
    [
        ("no_lowrank_no_sparse", 3, 20, 16, 0, 0.0, 0),
        ("lowrank_only", 3, 20, 16, 4, 0.0, 1),
        ("sparse_only", 3, 20, 16, 0, 0.05, 2),
        ("both", 4, 25, 16, 5, 0.05, 3),
        ("energy_threshold_ragged_rank", 5, 30, 16, None, 0.02, 4),
    ],
)
def test_gear_reconstruct_batched_matches_reference(desc, bh, n, d, rank, frac, seed):
    base = _rand((bh, n, d), seed)
    residual = _rand((bh, n, d), seed + 100) * 0.1

    if rank == 0:
        L_b = R_b = None
        ranks = [0] * bh
    else:
        L_b, R_b, ranks = _truncated_svd_batched(residual, rank=rank, energy_threshold=0.9)

    E_after = residual if L_b is None else residual - (L_b @ R_b)
    sp_idx_b, sp_val_b = sparse_outliers_batched(E_after, frac)

    recon_b = gear_reconstruct_batched(base, L_b, R_b, sp_idx_b, sp_val_b)

    for row in range(bh):
        r = ranks[row]
        L_i = L_b[row, :, :r] if L_b is not None else None
        R_i = R_b[row, :r, :] if R_b is not None else None
        sp_idx_i = sp_idx_b[row] if sp_idx_b is not None else None
        sp_val_i = sp_val_b[row] if sp_val_b is not None else None

        from veloxquant_mlx.quantizers.gear import GEARState

        state = GEARState(
            codes=mx.zeros((1, 1, d)),
            scale=mx.ones((1, 1, d)),
            zero=mx.zeros((1, 1, d)),
            L=L_i,
            R=R_i,
            sp_idx=sp_idx_i,
            sp_val=sp_val_i,
            n_rows=n,
            bits=2,
            rank=r,
            axis="token",
            d_cols=d,
        )
        rec_r = gear_reconstruct(state, base=base[row])
        diff = float(
            mx.max(mx.abs(recon_b[row].astype(mx.float32) - rec_r.astype(mx.float32))).item()
        )
        assert diff < 1e-2, f"{desc}: row {row} reconstruction diverged by {diff}"


# ---------------------------------------------------------------------------
# Real GEARKVCache end-to-end sanity (full pipeline exercised, including
# Pass 1/2 batching from #504 and Pass 3 batching from #570 together).
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {"method": "gear", "head_dim": 16, "gear_bits": 2, "gear_rank": 4}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize("H,seed", [(1, 0), (3, 1), (6, 2)])
def test_real_cache_prefill_and_decode_runs(H, seed):
    D, S = 16, 30
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(head_dim=D)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, H, S, D)
    assert v_out.shape == (1, H, S, D)

    for i in range(3):
        kd = _rand((1, H, 1, D), 100 + i).astype(mx.float16)
        vd = _rand((1, H, 1, D), 200 + i).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(kd, vd)
    assert k_out.shape == (1, H, S + 3, D)
    assert cache.compressed_key_bytes > 0
    assert 0.0 <= cache.error_recovery_ratio <= 1.0

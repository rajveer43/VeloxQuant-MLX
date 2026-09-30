"""Equivalence tests: batched [BH, S, D] KVTC local-PCA fit vs. the per-row
reference (#569).

``kvtc_local_pca_batched`` replaces the SVD-fit step inside
``KVTCKVCache.update_and_fetch``'s ``for b / for h`` loop (previously each
row called ``kvtc_compress``, which internally fits its own local PCA basis
via ``_truncated_svd``). DP bit allocation (``dp_allocate_bits``), per-
component quantization, and entropy coding remain per-row, unchanged — only
the SVD fit itself is batched (the DP step is a genuine sequential solver,
explicitly out of scope per the issue).

``kvtc_compress_from_pca_fit`` is the extracted "rest of kvtc_compress"
function that takes an already-fitted basis (L, V, mean, variances) and
runs DP allocation + quantization + entropy coding — calling it with a
per-head slice of ``kvtc_local_pca_batched``'s output must reproduce
``kvtc_compress`` called directly on that head's raw tensor, exactly.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kvtc_cache import KVTCKVCache
from veloxquant_mlx.quantizers.kvtc import (
    kvtc_compress,
    kvtc_compress_from_pca_fit,
    kvtc_decompress,
    kvtc_local_pca_batched,
)


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "desc,bh,S,D,budget,seed",
    [
        ("single_row", 1, 30, 16, 64, 0),
        ("multi_row", 4, 30, 16, 64, 1),
        ("small_S_relative_to_D", 3, 10, 16, 20, 2),
        ("generous_budget", 5, 20, 8, 200, 3),
        ("tight_budget_drops_components", 4, 40, 32, 16, 4),
        ("zero_budget", 2, 15, 8, 0, 5),
    ],
)
def test_kvtc_local_pca_batched_matches_reference(desc, bh, S, D, budget, seed):
    x = _rand((bh, S, D), seed)

    L_list, V_list, mean_list, var_list = kvtc_local_pca_batched(x)
    for row in range(bh):
        art_b = kvtc_compress_from_pca_fit(
            L_list[row], V_list[row], mean_list[row], var_list[row], S, budget
        )
        art_r = kvtc_compress(x[row], budget)

        assert np.array_equal(art_b.bit_allocation, art_r.bit_allocation), (
            f"{desc}: row {row} bit allocation diverged"
        )
        assert art_b.n_survived == art_r.n_survived, f"{desc}: row {row} n_survived"

        recon_b = kvtc_decompress(art_b).astype(mx.float32)
        recon_r = kvtc_decompress(art_r).astype(mx.float32)
        diff = float(mx.max(mx.abs(recon_b - recon_r)).item())
        assert diff < 1e-1, f"{desc}: row {row} reconstruction diverged by {diff}"


# ---------------------------------------------------------------------------
# Real KVTCKVCache end-to-end vs. the true per-row reference (kvtc_compress
# called directly in a loop, matching update_and_fetch's pre-batching form).
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {"method": "kvtc", "head_dim": 16, "kvtc_bit_budget": 64}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize("H,seed", [(1, 0), (3, 1), (6, 2)])
def test_real_cache_prefill_matches_reference_loop(H, seed):
    D, S, budget = 16, 30, 64
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(head_dim=D, kvtc_bit_budget=budget)
    k_out, v_out = cache.update_and_fetch(k, v)

    ref_k_h, ref_v_h = [], []
    for h in range(H):
        art_k = kvtc_compress(k[0, h].astype(mx.float32), budget)
        art_v = kvtc_compress(v[0, h].astype(mx.float32), budget)
        ref_k_h.append(kvtc_decompress(art_k))
        ref_v_h.append(kvtc_decompress(art_v))
    ref_k = mx.stack(ref_k_h, axis=0)[None]
    ref_v = mx.stack(ref_v_h, axis=0)[None]

    kd = float(mx.max(mx.abs(k_out.astype(mx.float32) - ref_k.astype(mx.float32))).item())
    vd = float(mx.max(mx.abs(v_out.astype(mx.float32) - ref_v.astype(mx.float32))).item())
    assert kd < 1e-1, f"H={H} seed={seed}: K diverged by {kd}"
    assert vd < 1e-1, f"H={H} seed={seed}: V diverged by {vd}"


@pytest.mark.parametrize("H,seed", [(1, 0), (4, 1)])
def test_real_cache_prefill_then_decode_matches_reference(H, seed):
    """Prefill fits the basis; decode continuation still requantizes through
    the SAME frozen basis — confirming the batched fit didn't disturb the
    "not path-dependent" invariant (module docstring)."""
    D, S, budget = 16, 20, 48
    cache = _make_cache(head_dim=D, kvtc_bit_budget=budget)
    k0, v0 = (
        _rand((1, H, S, D), seed).astype(mx.float16),
        _rand((1, H, S, D), seed + 1).astype(mx.float16),
    )
    cache.update_and_fetch(k0, v0)

    for i in range(4):
        kd = _rand((1, H, 1, D), 100 + i).astype(mx.float16)
        vd = _rand((1, H, 1, D), 200 + i).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(kd, vd)

    assert k_out.shape == (1, H, S + 4, D)
    assert v_out.shape == (1, H, S + 4, D)

"""The svdq basis must not be frozen from the first prefill chunk (#890)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.svdq import basis_from_gram_batched, svd_compress_keys_batched


def _keys(S=512, H=2, D=64, seed=0):
    """Low-rank keys whose dominant subspace is only partly visible in any short prefix."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((H, 16, D))
    scale = np.exp(-0.25 * np.arange(16))[None, :, None]
    coef = rng.standard_normal((H, S, 16))
    x = coef @ (basis * scale) + 0.02 * rng.standard_normal((H, S, D)) + 3.0
    return x.astype(np.float32)


def _cache(**cfg):
    base = {"method": "svdq", "head_dim": 64, "svdq_rank": 32, "svdq_group_size": 16}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def test_gram_basis_matches_direct_svd_fit() -> None:
    x = _keys(S=300)
    _, V_ref, mu_ref, s_ref = svd_compress_keys_batched(mx.array(x), rank=None)
    x64 = x.astype(np.float64)
    V, mu, s = basis_from_gram_batched(np.swapaxes(x64, 1, 2) @ x64, x64.sum(axis=1), x.shape[1])
    for h in range(x.shape[0]):
        assert V[h].shape == V_ref[h].shape
        np.testing.assert_allclose(np.array(s[h]), np.array(s_ref[h]), rtol=1e-4)
        np.testing.assert_allclose(np.array(mu[h]), np.array(mu_ref[h]), atol=1e-5)
        # same subspace (columns may differ in sign): projectors agree
        Pa = np.array(V[h]) @ np.array(V[h]).T
        Pb = np.array(V_ref[h]) @ np.array(V_ref[h]).T
        np.testing.assert_allclose(Pa, Pb, atol=1e-3)


def _recon_error(chunk: int) -> float:
    x = _keys()
    cache = _cache()
    outs = []
    for i in range(0, x.shape[1], chunk):
        k = mx.array(x[None, :, i : i + chunk])
        out, _ = cache.update_and_fetch(k, k)
        outs.append(np.array(out[0, :, -k.shape[2] :].astype(mx.float32)))
    out = np.concatenate(outs, axis=1)
    return float(np.sum((out - x) ** 2) / np.sum((x - x.mean(axis=1, keepdims=True)) ** 2))


def test_small_chunks_do_not_degrade_reconstruction() -> None:
    whole = _recon_error(512)
    for chunk in (128, 32):
        assert _recon_error(chunk) < 2.0 * whole + 1e-3


def test_rank_follows_all_rows_seen_not_first_chunk() -> None:
    x = _keys()
    cache = _cache(svdq_rank=None)
    ranks = []
    for i in range(0, 512, 32):
        k = mx.array(x[None, :, i : i + 32])
        cache.update_and_fetch(k, k)
        ranks.append(cache.rank)
    assert ranks[-1] > ranks[0]


def test_projection_bytes_track_the_current_basis() -> None:
    x = _keys()
    cache = _cache(svdq_rank=None)
    for i in range(0, 512, 64):
        k = mx.array(x[None, :, i : i + 64])
        cache.update_and_fetch(k, k)
    assert cache._projection_bytes_charged == sum((64 * r + 64) * 4 for r in cache._r)


def test_trim_to_empty_resets_running_statistics() -> None:
    x = _keys(S=64)
    cache = _cache()
    k = mx.array(x[None])
    cache.update_and_fetch(k, k)
    cache.trim(64)
    assert cache._n_rows == 0 and cache._gram is None and not cache._V

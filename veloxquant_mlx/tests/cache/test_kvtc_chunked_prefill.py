"""The kvtc basis and bit allocation must not be frozen from the first prefill chunk (#893)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.kvtc import kvtc_local_pca_batched, kvtc_pca_from_gram


def _keys(S=256, H=2, D=32, seed=0):
    """Low-rank keys whose dominant subspace is only partly visible in a short prefix."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((H, 12, D))
    scale = np.exp(-0.3 * np.arange(12))[None, :, None]
    x = rng.standard_normal((H, S, 12)) @ (basis * scale) + 0.02 * rng.standard_normal((H, S, D))
    return (x + 2.0).astype(np.float32)


def _cache():
    return KVCacheFactory.create(KVCacheConfig(method="kvtc", head_dim=32, kvtc_bit_budget=64))


def test_gram_pca_matches_local_pca() -> None:
    x = _keys(S=200)[0]
    _, V_ref, mu_ref, var_ref = kvtc_local_pca_batched(mx.array(x)[None])
    x64 = x.astype(np.float64)
    V, mu, var = kvtc_pca_from_gram(x64.T @ x64, x64.sum(axis=0), x.shape[0])
    assert V.shape == V_ref[0].shape
    np.testing.assert_allclose(var, var_ref[0], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(np.array(mu), np.array(mu_ref[0]), atol=1e-5)
    top = 8  # leading components are well separated; the noise tail is not unique
    Pa = np.array(V)[:, :top] @ np.array(V)[:, :top].T
    Pb = np.array(V_ref[0])[:, :top] @ np.array(V_ref[0])[:, :top].T
    np.testing.assert_allclose(Pa, Pb, atol=1e-3)


def _recon_error(chunk: int) -> float:
    x = _keys()
    cache = _cache()
    out = None
    for i in range(0, x.shape[1], chunk):
        k = mx.array(x[None, :, i : i + chunk])
        out, _ = cache.update_and_fetch(k, k)
    out = np.array(out[0].astype(mx.float32))
    return float(np.sum((out - x) ** 2) / np.sum((x - x.mean(axis=1, keepdims=True)) ** 2))


def test_small_chunks_do_not_degrade_reconstruction() -> None:
    whole = _recon_error(256)
    for chunk in (64, 16):
        assert _recon_error(chunk) < 2.0 * whole + 1e-3


def test_output_covers_every_row_and_offset_counts_new_rows() -> None:
    x = _keys()
    cache = _cache()
    for i in range(0, 256, 64):
        k = mx.array(x[None, :, i : i + 64])
        out, _ = cache.update_and_fetch(k, k)
        assert out.shape[2] == i + 64
        assert cache.offset == i + 64


def test_basis_follows_all_rows_seen() -> None:
    x = _keys()
    cache = _cache()
    n_rows = []
    for i in range(0, 256, 64):
        k = mx.array(x[None, :, i : i + 64])
        cache.update_and_fetch(k, k)
        n_rows.append(cache._keys_states[0]._n_rows)
    assert n_rows == [64, 128, 192, 256]


def test_decode_after_chunked_prefill_keeps_basis_frozen() -> None:
    x = _keys(S=130)
    cache = _cache()
    for i in range(0, 128, 64):
        k = mx.array(x[None, :, i : i + 64])
        cache.update_and_fetch(k, k)
    V_before = cache._keys_states[0]._artifact.V
    k = mx.array(x[None, :, 128:129])
    cache.update_and_fetch(k, k)
    assert bool(mx.all(V_before == cache._keys_states[0]._artifact.V).item())


def test_trim_to_empty_resets_running_statistics() -> None:
    x = _keys(S=64)
    cache = _cache()
    k = mx.array(x[None])
    cache.update_and_fetch(k, k)
    cache.trim(64)
    st = cache._keys_states[0]
    assert st._n_rows == 0 and st._gram is None and not st._fitted

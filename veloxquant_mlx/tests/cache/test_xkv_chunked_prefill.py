"""The xkv shared basis must not be frozen from the first prefill chunk (#892)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.xkv_cache import XKVCache
from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator
from veloxquant_mlx.quantizers.xkv import JointGramStats, joint_svd_compress


def _keys(S=256, H=2, D=32, seed=0):
    """Low-rank keys whose dominant subspace is only partly visible in a short prefix."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((12, D))
    scale = np.exp(-0.3 * np.arange(12))[:, None]
    x = rng.standard_normal((H, S, 12)) @ (basis * scale) + 0.02 * rng.standard_normal((H, S, D))
    return (x + 2.0).astype(np.float32)


def _cfg(**kw):
    return KVCacheConfig(method="xkv", head_dim=32, xkv_rank=None, **kw)


def test_gram_basis_matches_joint_svd_over_several_adds() -> None:
    x = _keys(S=300)[0]
    stats = JointGramStats()
    for i in range(0, 300, 100):
        stats.add(mx.array(x[i : i + 100]))
    V, mu, s = stats.basis()
    V_ref, mu_ref, s_ref = joint_svd_compress([mx.array(x)])
    assert V.shape == V_ref.shape
    np.testing.assert_allclose(np.array(mu), np.array(mu_ref), atol=1e-5)
    np.testing.assert_allclose(np.array(s), np.array(s_ref), rtol=1e-3)
    top = 8  # leading components are well separated; the noise tail is not unique
    Pa = np.array(V)[:, :top] @ np.array(V)[:, :top].T
    Pb = np.array(V_ref)[:, :top] @ np.array(V_ref)[:, :top].T
    np.testing.assert_allclose(Pa, Pb, atol=1e-3)


def _recon_error(chunk: int) -> float:
    x = _keys()
    cache = XKVCache(_cfg(), member_idx=0, group_id=0, n_members=1, coordinator=None)
    outs = []
    for i in range(0, x.shape[1], chunk):
        k = mx.array(x[None, :, i : i + chunk])
        out, _ = cache.update_and_fetch(k, k)
        outs.append(np.array(out[0, :, -k.shape[2] :].astype(mx.float32)))
    out = np.concatenate(outs, axis=1)
    return float(np.sum((out - x) ** 2) / np.sum((x - x.mean(axis=1, keepdims=True)) ** 2))


def test_small_chunks_do_not_degrade_reconstruction() -> None:
    whole = _recon_error(256)
    for chunk in (64, 16):
        assert _recon_error(chunk) < 2.0 * whole + 1e-3


def test_group_basis_pools_every_member_and_chunk() -> None:
    coord = XKVCoordinator()
    members = [
        XKVCache(_cfg(), member_idx=i, group_id=0, n_members=2, coordinator=coord) for i in range(2)
    ]
    x = _keys()
    for i in range(0, 256, 64):
        for m in members:
            k = mx.array(x[None, :, i : i + 64])
            m.update_and_fetch(k, k)
    assert coord._stats[0].n_rows == 2 * 256
    # the member iterated last saw both members' latest chunk; the first lags by one chunk
    assert members[1]._basis_version == coord.version(0)
    assert members[0]._basis_version < coord.version(0)


def test_lagging_member_adopts_latest_basis_on_decode() -> None:
    coord = XKVCoordinator()
    members = [
        XKVCache(_cfg(), member_idx=i, group_id=0, n_members=2, coordinator=coord) for i in range(2)
    ]
    x = _keys(S=130)
    for m in members:
        k = mx.array(x[None, :, :128])
        m.update_and_fetch(k, k)
    d = mx.array(x[None, :, 128:129])
    for m in members:
        m.update_and_fetch(d, d)
    np.testing.assert_allclose(
        np.array(members[0]._V_g.tolist()), np.array(members[1]._V_g.tolist()), atol=1e-6
    )
    before = np.array(members[0]._V_g.tolist())
    for _ in range(3):
        for m in members:
            m.update_and_fetch(d, d)
    np.testing.assert_array_equal(before, np.array(members[0]._V_g.tolist()))


def test_leader_basis_bytes_track_the_current_rank() -> None:
    cache = XKVCache(_cfg(), member_idx=0, group_id=0, n_members=1, coordinator=None)
    x = _keys()
    for i in range(0, 256, 64):
        k = mx.array(x[None, :, i : i + 64])
        cache.update_and_fetch(k, k)
    assert cache.shared_basis_bytes == (32 * cache.rank + 32) * 4


def test_max_ctx_bounds_a_single_call() -> None:
    coord = XKVCoordinator(max_ctx=8)
    m = XKVCache(_cfg(), member_idx=0, group_id=0, n_members=2, coordinator=coord)
    with pytest.raises(RuntimeError, match="max_ctx"):
        k = mx.array(_keys(S=16)[None])
        m.update_and_fetch(k, k)


def test_trim_to_empty_resets_pooled_statistics() -> None:
    cache = XKVCache(_cfg(), member_idx=0, group_id=0, n_members=1, coordinator=None)
    k = mx.array(_keys(S=64)[None])
    cache.update_and_fetch(k, k)
    cache.trim(64)
    assert cache._local_stats.n_rows == 0 and cache._V_g is None

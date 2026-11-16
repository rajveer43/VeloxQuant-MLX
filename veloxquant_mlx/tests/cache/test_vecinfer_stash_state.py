"""#968: stash-mode (fused_sdpa=True, memory_bound=False) ``state`` round trip.

Stash mode keeps the parent's fp16 K_hat/V_hat as the source of truth for SDPA
and stashes the codebook indices on the side. ``state`` used to return only the
index buffers, so a restored cache had ``keys is None`` and an ``offset`` equal
to the buffer capacity: the next ``update_and_fetch`` returned a zero prefix.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import metal_available

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(not metal_available(), reason="Metal kernels not available."),
]

D = 16


def _cache():
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=D,
        key_sub_dim=4,
        value_sub_dim=4,
        key_codebook_bits=6,
        value_codebook_bits=6,
        fused_sdpa=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KVCacheFactory.create(cfg)


def _kv(s: int, seed: int):
    mx.random.seed(seed)
    return mx.random.normal((1, 2, s, D)), mx.random.normal((1, 2, s, D))


def test_restored_prefix_matches_original() -> None:
    c = _cache()
    k1, v1 = c.update_and_fetch(*_kv(8, 0))
    mx.eval(k1, v1)

    c2 = _cache()
    c2.state = c.state
    assert c2.offset == 8
    assert not c2.empty()

    k2, v2 = c2.update_and_fetch(*_kv(1, 1))
    mx.eval(k2, v2)
    assert k2.shape[2] == 9
    assert mx.array_equal(k2[:, :, :8], k1)
    assert mx.array_equal(v2[:, :, :8], v1)


def test_restored_cache_matches_uninterrupted_cache() -> None:
    a = _cache()
    a.update_and_fetch(*_kv(8, 0))
    ka, va = a.update_and_fetch(*_kv(1, 1))

    b = _cache()
    b.update_and_fetch(*_kv(8, 0))
    c = _cache()
    c.state = b.state
    kc, vc = c.update_and_fetch(*_kv(1, 1))
    assert mx.array_equal(ka, kc) and mx.array_equal(va, vc)


def test_index_stash_survives_round_trip() -> None:
    c = _cache()
    c.update_and_fetch(*_kv(8, 0))
    c2 = _cache()
    c2.state = c.state
    assert c2._stored_S_kv == 8
    # Live index rows must be identical, fused_sdpa() reads exactly these.
    assert mx.array_equal(c2._stored_k_indices[:, :, :8], c._stored_k_indices[:, :, :8])
    assert mx.array_equal(c2._stored_v_indices[:, :, :8], c._stored_v_indices[:, :, :8])


def test_empty_cache_round_trip() -> None:
    c = _cache()
    c2 = _cache()
    c2.state = c.state
    assert c2.empty()
    assert c2.offset == 0


def test_fp16_only_state_does_not_leave_stale_indices() -> None:
    """A state without index buffers restores fp16 and drops the stash."""
    c = _cache()
    c.update_and_fetch(*_kv(8, 0))
    fp16_only = c.state[:-2]
    c2 = _cache()
    c2.state = fp16_only
    assert c2.offset == 8
    assert not c2.empty()
    assert c2._stored_k_indices is None and c2._stored_S_kv == 0
    k, _ = c2.update_and_fetch(*_kv(1, 1))
    assert k.shape[2] == 9

"""PALU stores latents in the fitted basis, so the basis cannot be refit per chunk.

``palu_min_fit_rows`` holds rows raw until enough have arrived for one fit (#891).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _make(min_rows: int):
    return KVCacheFactory.create(
        KVCacheConfig(
            method="palu",
            head_dim=64,
            palu_rank=16,
            palu_n_head_groups=2,
            palu_group_size=16,
            palu_min_fit_rows=min_rows,
        )
    )


def _kv(S, seed, H=4, D=64):
    rng = np.random.default_rng(seed)
    return (
        mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16)),
        mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16)),
    )


def test_default_fits_on_first_call() -> None:
    c = _make(0)
    c.update_and_fetch(*_kv(32, 0))
    assert c._keys_lr._fitted


def test_chunks_are_held_raw_until_threshold() -> None:
    c = _make(96)
    k1, v1 = _kv(32, 1)
    k2, v2 = _kv(32, 2)
    ko, _ = c.update_and_fetch(k1, v1)
    assert not c._keys_lr._fitted and bool(mx.array_equal(ko, k1))
    ko, _ = c.update_and_fetch(k2, v2)
    assert not c._keys_lr._fitted and ko.shape[2] == 64
    k3, v3 = _kv(32, 3)
    ko, vo = c.update_and_fetch(k3, v3)
    assert c._keys_lr._fitted and ko.shape == (1, 4, 96, 64) and c.offset == 96


def test_chunked_matches_single_call_once_fitted() -> None:
    k, v = _kv(96, 4)
    one = _make(96)
    ko1, _ = one.update_and_fetch(k, v)
    chunked = _make(96)
    for i in range(3):
        ko2, _ = chunked.update_and_fetch(
            k[:, :, i * 32 : (i + 1) * 32], v[:, :, i * 32 : (i + 1) * 32]
        )
    np.testing.assert_allclose(np.asarray(ko1, np.float32), np.asarray(ko2, np.float32), atol=1e-3)


def test_decode_after_held_prefill_appends() -> None:
    c = _make(64)
    c.update_and_fetch(*_kv(32, 5))
    ko, _ = c.update_and_fetch(*_kv(1, 6))
    assert ko.shape[2] == 33 and not c._keys_lr._fitted
    ko, _ = c.update_and_fetch(*_kv(32, 7))
    assert c._keys_lr._fitted and ko.shape[2] == 65


def test_trim_to_empty_resets_hold() -> None:
    c = _make(64)
    c.update_and_fetch(*_kv(48, 8))
    c.trim(48)
    c.update_and_fetch(*_kv(48, 9))
    assert c.offset == 48 and not c._keys_lr._fitted

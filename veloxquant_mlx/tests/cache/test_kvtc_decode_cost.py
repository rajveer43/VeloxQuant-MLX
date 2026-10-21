"""kvtc decode must not re-encode the whole history every step (#841)."""

from __future__ import annotations

import time
import warnings

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H = 32, 4


def _kv(S, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))


def _make():
    return KVCacheFactory.create(KVCacheConfig(method="kvtc", head_dim=D))


def _decode_ms(prefill: int) -> float:
    cache = _make()
    k = _kv(prefill, 0)
    mx.eval(*cache.update_and_fetch(k, k))
    ts = []
    for t in range(6):
        k = _kv(1, 100 + t)
        t0 = time.perf_counter()
        mx.eval(*cache.update_and_fetch(k, k))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[1:])) * 1e3


def test_decode_step_cost_does_not_scale_with_context() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short, long_ = _decode_ms(128), _decode_ms(1024)
    # Before the fix this was ~8x (every step re-encoded all rows in Python).
    assert long_ < 4 * max(short, 2.0), f"decode {short:.1f} ms @128 vs {long_:.1f} ms @1024"


def test_nbytes_reports_stored_bytes() -> None:
    cache = _make()
    k = _kv(64, 1)
    cache.update_and_fetch(k, k)
    assert cache.nbytes > 0
    assert cache.nbytes == cache.kvtc_bytes


def test_bytes_follow_decode_growth() -> None:
    cache = _make()
    k = _kv(64, 2)
    cache.update_and_fetch(k, k)
    before = cache.kvtc_bytes
    for t in range(32):
        r = _kv(1, 50 + t)
        cache.update_and_fetch(r, r)
    assert cache.kvtc_bytes > before


def test_decode_output_covers_every_row_and_trim_is_consistent() -> None:
    cache = _make()
    k = _kv(48, 3)
    cache.update_and_fetch(k, k)
    for t in range(5):
        r = _kv(1, 70 + t)
        out, _ = cache.update_and_fetch(r, r)
        assert out.shape[2] == 48 + t + 1
    assert cache.trim(3) == 3
    out, _ = cache.update_and_fetch(_kv(1, 99), _kv(1, 99))
    assert out.shape[2] == 48 + 5 - 3 + 1

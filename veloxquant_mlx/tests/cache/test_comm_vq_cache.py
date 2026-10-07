"""Tests for the CommVQ KV cache wrapper (#756)."""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.comm_vq_cache import CommVQKVCache


def _make(**kw):
    cfg = {"method": "comm_vq", "head_dim": 64, "comm_vq_bits": 6, **kw}
    return KVCacheFactory.create(KVCacheConfig(**cfg))


def _kv(S=128, H=4, D=64, seed=0):
    mx.random.seed(seed)
    return (
        mx.random.normal((1, H, S, D)).astype(mx.float16),
        mx.random.normal((1, H, S, D)).astype(mx.float16),
    )


def test_factory_builds_wrapper_and_returns_fp16_shapes():
    c = _make()
    assert isinstance(c, CommVQKVCache)
    k, v = _kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape and ko.dtype == k.dtype
    assert mx.array_equal(vo, v)  # values are untouched
    assert c.offset == 128


def test_reconstruction_beats_zero_baseline_and_counts_index_bytes():
    c = _make()
    k, v = _kv()
    ko, _ = c.update_and_fetch(k, v)
    err = float(mx.mean((ko.astype(mx.float32) - k.astype(mx.float32)) ** 2))
    energy = float(mx.mean(k.astype(mx.float32) ** 2))
    assert err < energy  # better than predicting zeros
    n = 1 * 4 * 128
    assert c.compressed_key_bytes == n * 4  # 4 codebooks, one uint8 each
    assert c.fp16_key_bytes == n * 64 * 2


def test_decode_step_after_prefill_appends_one_token():
    c = _make()
    k, v = _kv()
    c.update_and_fetch(k, v)
    k1, v1 = _kv(S=1, seed=1)
    ko, _ = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 129 and c.offset == 129


def test_trim_scales_byte_counters():
    c = _make()
    k, v = _kv()
    c.update_and_fetch(k, v)
    before = (c.compressed_key_bytes, c.fp16_key_bytes)
    assert c.trim(64) == 64
    assert c.compressed_key_bytes == before[0] // 2
    assert c.fp16_key_bytes == before[1] // 2


def test_codebooks_trained_once_on_first_call():
    c = _make()
    k, v = _kv()
    c.update_and_fetch(k, v)
    cb = c._quantizer._codebooks.copy()
    k2, v2 = _kv(seed=5)
    c.update_and_fetch(k2, v2)
    assert (c._quantizer._codebooks == cb).all()

"""Tests for KVCacheProfiler (profiling/kv_profiler.py)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder
from veloxquant_mlx.profiling import (
    KVCacheProfiler,
    ProfileReport,
    format_profile_table,
    profile_layers,
)

HEAD_DIM = 64


def _raw_cache():
    return (
        KVCacheBuilder()
        .with_method("turboquant_prod")
        .with_head_dim(HEAD_DIM)
        .with_bit_width(inlier=2)
        .with_jl_dim(HEAD_DIM)
        .with_seed(42)
        .build()
    )


def _kv(seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal(HEAD_DIM).astype(np.float16))
    v = mx.array(rng.standard_normal(HEAD_DIM).astype(np.float16))
    return k, v


def test_wraps_transparently():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    for i in range(5):
        k, v = _kv(i)
        profiled.append(k, v)
    q, _ = _kv(99)
    out = profiled.attend(q)
    mx.eval(out)
    assert out.shape == (HEAD_DIM,)
    assert len(profiled) == 5
    assert profiled.memory_bytes() == cache.memory_bytes()


def test_records_quantize_and_dequantize_calls():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    for i in range(3):
        k, v = _kv(i)
        profiled.append(k, v)
    q, _ = _kv(99)
    profiled.attend(q)
    profiled.attend(q)

    p = profiled.profile()
    assert p.n_quantize_calls == 3
    assert p.n_dequantize_calls == 2
    assert p.quantize_ms_total >= 0.0
    assert p.dequantize_ms_total >= 0.0
    assert p.tokens_written == 3


def test_peak_memory_tracks_wrapped_cache():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    for i in range(4):
        k, v = _kv(i)
        profiled.append(k, v)
    assert profiled.profile().peak_memory_bytes == cache.memory_bytes()
    assert profiled.profile().peak_memory_bytes > 0


def test_compression_ratio_positive():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    for i in range(8):
        k, v = _kv(i)
        profiled.append(k, v)
    ratio = profiled.profile().compression_ratio
    assert ratio > 1.0  # 2-bit inlier quantization should compress vs fp16


def test_reset_clears_stats():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    k, v = _kv(0)
    profiled.append(k, v)
    profiled.reset()
    p = profiled.profile()
    assert p.n_quantize_calls == 0
    assert p.tokens_written == 0
    # Wrapped cache is untouched by reset — it still has the appended token.
    assert len(cache) == 1


def test_getattr_forwards_to_wrapped_cache():
    cache = _raw_cache()
    profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=0)
    assert profiled._d == cache._d


def test_profile_layers_and_format_table():
    profilers = []
    for layer_id in range(2):
        cache = _raw_cache()
        profiled = KVCacheProfiler(cache, head_dim=HEAD_DIM, layer_id=layer_id)
        for i in range(6):
            k, v = _kv(i + layer_id * 10)
            profiled.append(k, v)
        q, _ = _kv(99)
        profiled.attend(q)
        profilers.append(profiled)

    report = profile_layers(profilers, elapsed_s=0.5)
    assert isinstance(report, ProfileReport)
    assert len(report.layers) == 2
    assert report.total_tokens == 12
    assert report.tokens_per_sec == pytest.approx(12 / 0.5)

    table = format_profile_table(report)
    assert "Layer 0" in table
    assert "Layer 1" in table
    assert "Quantize" in table
    assert "Dequantize" in table
    assert "Memory" in table
    assert "Compression ratio" in table


def test_empty_report_table_has_no_totals():
    report = ProfileReport(layers=[])
    table = format_profile_table(report)
    assert "Compression ratio" not in table

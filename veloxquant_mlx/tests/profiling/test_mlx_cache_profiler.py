"""Tests for MLXCacheProfiler (profiling/kv_profiler.py).

Uses a small fake cache implementing the mlx_lm update_and_fetch/nbytes
contract rather than KVCacheBuilder.for_model + a real model, since the
class under test only ever calls those two members.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.profiling import (
    MLXCacheProfiler,
    ProfileReport,
    format_profile_table,
    profile_layers,
)

HEAD_DIM = 64


class _FakeMLXCache:
    """Minimal stand-in for an mlx_lm-serving KVCache subclass."""

    def __init__(self, head_dim: int = HEAD_DIM):
        self._head_dim = head_dim
        self._nbytes = 0
        self.calls = 0

    def update_and_fetch(self, keys, values):
        self.calls += 1
        n_tokens = keys.shape[-2]
        self._nbytes += n_tokens * self._head_dim  # arbitrary but monotonic
        return keys, values

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def make_mask(self, *args, **kwargs):
        return None


def _kv(n_tokens: int, head_dim: int = HEAD_DIM, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape = (1, 1, n_tokens, head_dim)
    k = mx.array(rng.standard_normal(shape).astype(np.float16))
    v = mx.array(rng.standard_normal(shape).astype(np.float16))
    return k, v


def test_wraps_update_and_fetch():
    cache = _FakeMLXCache()
    profiled = MLXCacheProfiler(cache, layer_id=0)
    k, v = _kv(4)
    out_k, out_v = profiled.update_and_fetch(k, v)
    assert out_k.shape == k.shape
    assert cache.calls == 1

    p = profiled.profile()
    assert p.n_quantize_calls == 1
    assert p.n_dequantize_calls == 0
    assert p.quantize_ms_total >= 0.0
    assert p.dequantize_ms_total == 0.0
    assert p.write_ms_total == 0.0
    assert p.tokens_written == 4
    assert p.is_fused is True


def test_peak_memory_tracks_wrapped_cache_nbytes():
    cache = _FakeMLXCache()
    profiled = MLXCacheProfiler(cache, layer_id=0)
    k, v = _kv(4)
    profiled.update_and_fetch(k, v)
    assert profiled.profile().peak_memory_bytes == cache.nbytes
    assert profiled.profile().peak_memory_bytes > 0


def test_bool_is_always_true():
    # mlx_lm's create_attention_mask does `if cache and hasattr(...)`, and
    # the wrapped cache defines no __len__/__bool__ of its own — the
    # profiler must not turn that into a crash (see kv_profiler.py's
    # KVCacheProfiler.__bool__ docstring for the same issue on that class).
    cache = _FakeMLXCache()
    profiled = MLXCacheProfiler(cache, layer_id=0)
    assert bool(profiled) is True


def test_getattr_forwards_to_wrapped_cache():
    cache = _FakeMLXCache()
    profiled = MLXCacheProfiler(cache, layer_id=0)
    assert profiled.make_mask() is None


def test_profile_layers_and_format_table():
    profilers = []
    for layer_id in range(2):
        cache = _FakeMLXCache()
        profiled = MLXCacheProfiler(cache, layer_id=layer_id)
        k, v = _kv(6, seed=layer_id)
        profiled.update_and_fetch(k, v)
        profilers.append(profiled)

    report = profile_layers(profilers, elapsed_s=0.5)
    assert isinstance(report, ProfileReport)
    assert len(report.layers) == 2
    assert report.total_tokens == 12
    assert report.tokens_per_sec == pytest.approx(12 / 0.5)
    assert all(layer.is_fused for layer in report.layers)

    table = format_profile_table(report)
    assert "Layer 0" in table
    assert "Layer 1" in table

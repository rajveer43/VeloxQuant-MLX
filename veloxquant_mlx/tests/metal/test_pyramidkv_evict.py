"""GPU correctness and cache dispatch regressions for PyramidKV."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.metal._pyramidkv_evict import pyramidkv_fused_evict
from veloxquant_mlx.quantizers.pyramidkv import init_pyramid_state, pyramid_update


@pytest.mark.parametrize("shape", [(1, 2, 1), (3, 35, 31), (8, 513, 128)])
@pytest.mark.parametrize("mode", ["tie", "infinity", "nan", "middle"])
def test_compaction(shape, mode):
    h, n, d = shape
    sink = min(4, n - 1)
    k = mx.arange(h * n * d).reshape(shape).astype(mx.float16)
    v = -k
    score = float("inf") if mode == "infinity" else float("nan") if mode == "nan" else 1.0
    s = mx.full((h, n), score, dtype=mx.float32)
    loser = sink
    if mode == "middle":
        loser = (sink + n) // 2
        s = mx.where(mx.arange(n)[None] == loser, 0.0, s)
    out = pyramidkv_fused_evict(k, v, s, sink)
    keep = mx.array([i for i in range(n) if i != loser])
    expected = (k[:, keep], v[:, keep], s[:, keep])
    mx.eval(*out, *expected)
    for a, b in zip(out, expected, strict=True):
        assert mx.all((a == b) | (mx.isnan(a) & mx.isnan(b))).item()


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_update_history(backend):
    mx.random.seed(4)
    k = mx.random.normal((37, 31)).astype(mx.float16)
    v = mx.random.normal((37, 31)).astype(mx.float16)
    ref = init_pyramid_state(2, 9, 31)
    out = init_pyramid_state(2, 9, 31)
    for start, end in [(0, 4), (4, 20), (20, 37)]:
        ref = pyramid_update(ref, k[start:end], v[start:end])
        out = pyramid_update(out, k[start:end], v[start:end], backend=backend)
        for a, b in [(ref.keys, out.keys), (ref.values, out.values), (ref.scores, out.scores)]:
            assert mx.array_equal(a, b).item()


def test_cache_backend_dispatch(monkeypatch):
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.pyramidkv_cache import PyramidKVCache
    from veloxquant_mlx.metal import _pyramidkv_evict

    calls = []
    original = _pyramidkv_evict.pyramidkv_fused_evict

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_pyramidkv_evict, "pyramidkv_fused_evict", counted)
    caches = [
        PyramidKVCache(KVCacheConfig(pyramid_budget=7, pyramid_n_sink=2, pyramid_backend=b))
        for b in ("reference", "mlx", "metal")
    ]
    for length in (4, 8, 1):
        k = mx.random.normal((1, 3, length, 31)).astype(mx.float16)
        outputs = [c.update_and_fetch(k, -k) for c in caches]
        for output in outputs[1:]:
            for a, b in zip(outputs[0], output, strict=True):
                assert mx.array_equal(a, b).item()
        assert len({c.offset for c in caches}) == 1
    assert calls


def test_noncontiguous_and_stream():
    k = mx.arange(3 * 17 * 62).reshape(3, 17, 62).astype(mx.float16)[:, :, ::2]
    s = mx.ones((3, 17), dtype=mx.float32)
    out = pyramidkv_fused_evict(k, -k, s, 4, stream=mx.new_stream(mx.gpu))
    keep = mx.array([i for i in range(17) if i != 4])
    assert mx.array_equal(out[0], k[:, keep]).item()


@pytest.mark.parametrize("shape", [(0, 3, 4), (1, 3, 0)])
def test_invalid_dimensions(shape):
    k = mx.zeros(shape, dtype=mx.float16)
    with pytest.raises(ValueError):
        pyramidkv_fused_evict(k, k, mx.zeros(shape[:2], dtype=mx.float32), 0)


@pytest.mark.parametrize("backend", ["mlx", "metal"])
@pytest.mark.parametrize("budget,sink", [(1, 0), (17, 4), (64, 0)])
def test_batched_history(backend, budget, sink):
    from veloxquant_mlx.quantizers.pyramidkv import pyramid_update_heads

    mx.random.seed(18)
    k = mx.random.normal((6, 83, 31)).astype(mx.float16)
    v = mx.random.normal(k.shape).astype(mx.float16)
    reference = [init_pyramid_state(sink, budget, 31) for _ in range(6)]
    actual = [init_pyramid_state(sink, budget, 31) for _ in range(6)]
    for start, end in [(0, 3), (3, 71), (71, 83)]:
        reference = [
            pyramid_update(st, k[g, start:end], v[g, start:end]) for g, st in enumerate(reference)
        ]
        actual = pyramid_update_heads(actual, k[:, start:end], v[:, start:end], backend=backend)
        for a, b in zip(reference, actual, strict=True):
            for x, y in [(a.keys, b.keys), (a.values, b.values), (a.scores, b.scores)]:
                assert mx.array_equal(x, y).item()

"""Compare accelerated updates against the unchanged sequential reference."""

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.tova import (
    _tova_update_batched,
    init_tova_state,
    tova_update,
)


@pytest.mark.parametrize("backend", ["mlx", "metal"])
@pytest.mark.parametrize("bh,d,budget,sink", [(1, 7, 1, 0), (3, 33, 9, 8), (8, 128, 32, 4)])
@pytest.mark.parametrize("kind", ["random", "ties", "underflow"])
def test_batched_steps_match_original(backend, bh, d, budget, sink, kind):
    rng = np.random.default_rng(19)
    k = rng.normal(size=(bh, 79, d)).astype(np.float32)
    if kind == "ties":
        k[:] = 0
    if kind == "underflow":
        k *= 100
    v = rng.normal(size=k.shape).astype(np.float16)
    nk, nv = mx.array(k), mx.array(v)
    reference = [init_tova_state(sink, budget, d) for _ in range(bh)]
    keys = values = None
    start = 0
    for count in [3, 11, 1, 64]:
        keys, values = _tova_update_batched(
            keys,
            values,
            nk[:, start : start + count],
            nv[:, start : start + count],
            sink,
            budget,
            backend=backend,
        )
        for h in range(bh):
            reference[h] = tova_update(
                reference[h],
                nk[h, start : start + count],
                nv[h, start : start + count],
                backend="reference",
            )
            np.testing.assert_array_equal(np.array(keys[h]), np.array(reference[h].keys))
            np.testing.assert_array_equal(np.array(values[h]), np.array(reference[h].values))
        start += count


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_deferred_values_preserve_independent_fingerprints(backend):
    rng = np.random.default_rng(123)
    bh, d, budget = 4, 33, 11
    k = mx.array(rng.normal(size=(bh, 137, d)).astype(np.float16))
    v = mx.zeros((bh, 137, d), mx.float16)
    # Value fingerprints are independent of K and uniquely identify source rows.
    v[:, :, 0] = mx.arange(137, dtype=mx.float16)[None]
    actual_k, actual_v = _tova_update_batched(None, None, k, v, 3, budget, backend=backend)
    reference = [init_tova_state(3, budget, d) for _ in range(bh)]
    for h in range(bh):
        reference[h] = tova_update(reference[h], k[h], v[h], backend="reference")
        np.testing.assert_array_equal(np.array(actual_k[h]), np.array(reference[h].keys))
        np.testing.assert_array_equal(np.array(actual_v[h]), np.array(reference[h].values))


@pytest.mark.parametrize("backend", ["mlx", "metal", "auto"])
def test_zero_budget_legacy_behavior_and_empty_update(backend):
    st = init_tova_state(0, 0, 3)
    empty = mx.zeros((0, 3))
    assert tova_update(st, empty, empty, backend=backend) is st
    k = mx.ones((4, 3))
    expected = tova_update(st, k, k, backend="reference")
    actual = tova_update(st, k, k, backend=backend)
    np.testing.assert_array_equal(np.array(actual.keys), np.array(expected.keys))


def test_cpu_auto_and_forced_metal_rejection():
    with mx.stream(mx.cpu):
        k = mx.ones((6, 3))
        out = tova_update(init_tova_state(0, 2, 3), k, k)
        mx.eval(out.keys)
        assert out.keys.shape == (2, 3)
        with pytest.raises(ValueError, match="GPU"):
            tova_update(init_tova_state(0, 2, 3), k, k, backend="metal")


def test_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        tova_update(init_tova_state(0, 2, 3), mx.ones((1, 3)), mx.ones((1, 3)), backend="typo")


def test_long_prefill_materializes_bounded_graph(monkeypatch):
    from veloxquant_mlx.quantizers import tova

    flushes = []
    original_eval = mx.eval

    def track(*arrays):
        flushes.append(arrays[0].shape)
        return original_eval(*arrays)

    monkeypatch.setattr(tova.mx, "eval", track)
    k = mx.zeros((3, 2048, 33), mx.float16)
    keys, values = _tova_update_batched(None, None, k, k, 4, 16, backend="metal")
    original_eval(keys, values)
    assert keys.shape == (3, 16, 33)
    assert len(flushes) == (2048 - 16) // 32
    assert set(flushes) == {(3, 16, 33)}


def test_compiled_selection_matches_eager():
    from veloxquant_mlx.metal import tova_fused_evict

    k = mx.arange(3 * 19 * 7).reshape(3, 19, 7).astype(mx.float16)
    w = mx.zeros((3, 19), mx.float32)
    compiled = mx.compile(lambda a, b, c: tova_fused_evict(a, b, c, 2))
    actual = compiled(k, k, w)
    expected = tova_fused_evict(k, k, w, 2)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(e))


def test_auto_routing_and_missing_metal(monkeypatch):
    from veloxquant_mlx import metal
    from veloxquant_mlx.quantizers.tova import _resolve_backend

    assert _resolve_backend("auto", n_tokens=1) == "mlx"
    assert _resolve_backend("auto", n_tokens=16) == "metal"
    monkeypatch.setattr(metal, "metal_available", lambda: False)
    assert _resolve_backend("auto", n_tokens=16) == "mlx"


def test_cache_batches_all_heads_in_one_selection(monkeypatch):
    from veloxquant_mlx import metal
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

    calls = []
    original = metal.tova_fused_evict_indices

    def track(k, lineage, w, sink):
        calls.append(k.shape)
        return original(k, lineage, w, sink)

    monkeypatch.setattr(metal, "tova_fused_evict_indices", track)
    cache = KVCacheFactory.create(
        KVCacheConfig(method="tova", tova_budget=4, tova_n_sink=1, tova_backend="metal")
    )
    k = mx.ones((2, 3, 6, 7), mx.float16)
    mx.eval(*cache.update_and_fetch(k, k))
    assert calls == [(6, 5, 7)] * 2

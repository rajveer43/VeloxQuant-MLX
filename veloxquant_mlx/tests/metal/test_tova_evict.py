"""Selection identity, copy parity, and launch-boundary tests for TOVA."""

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available, tova_fused_evict
from veloxquant_mlx.quantizers.tova import _evict_mlx

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")


@pytest.mark.parametrize(
    "bh,n,d,sink",
    [(1, 2, 7, 0), (3, 259, 33, 4), (8, 513, 128, 4), (1, 4097, 64, 4), (1, 257, 256, 0)],
)
@pytest.mark.parametrize("nsg", [1, 4, 8])
def test_selection_and_exact_copy(bh, n, d, sink, nsg):
    rng = np.random.default_rng(42)
    k = rng.normal(size=(bh, n, d)).astype(np.float16)
    v = rng.normal(size=(bh, n, d)).astype(np.float16)
    weights = rng.integers(0, 4, size=(bh, n)).astype(np.float32)
    # Exact ties across SIMD groups and grid-stride iterations.
    weights[:, sink] = -1
    weights[:, -1] = -1
    weights[0, -1] = -2
    protected = weights.copy()
    protected[:, :sink] = np.inf
    evicted = protected.argmin(axis=1)
    expected = [np.stack([np.delete(x[h], evicted[h], axis=0) for h in range(bh)]) for x in (k, v)]
    inputs = mx.array(k), mx.array(v), mx.array(weights)
    for outputs in (tova_fused_evict(*inputs, sink, nsg=nsg), _evict_mlx(*inputs, sink)):
        mx.eval(*outputs)
        for out, ref in zip(outputs, expected, strict=True):
            np.testing.assert_array_equal(np.array(out), ref)


def test_noncontiguous_and_nondefault_stream():
    x = mx.arange(3 * 17 * 14).reshape(3, 17, 14).astype(mx.float16)[:, :, ::2]
    w = mx.zeros((3, 34), mx.float32)[:, ::2]
    stream = mx.new_stream(mx.gpu)
    k, v = tova_fused_evict(x, x, w, 2, stream=stream)
    mx.eval(k, v)
    np.testing.assert_array_equal(np.array(k), np.delete(np.array(x), 2, axis=1))


@pytest.mark.parametrize(
    "weights,removed",
    [
        ([float("inf")] * 4, 1),
        ([float("nan")] * 4, 1),
        ([0, float("nan"), 1, -1], 3),
        ([0, -0.0, 0.0, 0.0], 1),
    ],
)
def test_degenerate_scores_never_produce_invalid_index(weights, removed):
    x = mx.arange(4).reshape(1, 4, 1).astype(mx.float16)
    out, _ = tova_fused_evict(x, x, mx.array([weights], mx.float32), 1)
    assert out.reshape(-1).tolist() == [i for i in range(4) if i != removed]


def test_single_candidate_empty_output_and_validation():
    x = mx.ones((1, 1, 3), mx.float16)
    w = mx.ones((1, 1), mx.float32)
    assert tova_fused_evict(x, x, w, 0)[0].shape == (1, 0, 3)
    for args in [(x, x, w, 1), (x.astype(mx.float32), x, w, 0), (x, x, w[:, :0], 0)]:
        with pytest.raises(ValueError):
            tova_fused_evict(*args)
    for nsg in (32, 1.0, True):
        with pytest.raises(ValueError):
            tova_fused_evict(x, x, w, 0, nsg=nsg)

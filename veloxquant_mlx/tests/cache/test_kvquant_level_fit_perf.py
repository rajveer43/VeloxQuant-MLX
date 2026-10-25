"""KVQuant level fit must not build [BH, N, L, D] tensors (#879)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.quantizers import kvquant as kq


def _reference_fit(x: mx.array, bits: int, n_iters: int = 8) -> mx.array:
    """The original one-hot Lloyd-Max fit, kept as the equivalence oracle."""
    x32 = x.astype(mx.float32)
    n = x32.shape[1]
    L = 1 << bits
    xs = mx.sort(x32, axis=1)
    qpos = (mx.arange(L, dtype=mx.float32) + 0.5) / L
    idx = mx.clip(mx.round(qpos * (n - 1)), 0, n - 1).astype(mx.int32)
    levels = mx.take(xs, idx, axis=1)
    for _ in range(n_iters):
        diff = mx.abs(x32[:, :, None, :] - levels[:, None, :, :])
        assign = mx.argmin(diff, axis=2)
        lr = mx.arange(L)[None, None, :, None]
        one_hot = (assign[:, :, None, :] == lr).astype(mx.float32)
        counts = mx.sum(one_hot, axis=1)
        sums = mx.sum(one_hot * x32[:, :, None, :], axis=1)
        levels = mx.where(counts > 0, sums / mx.maximum(counts, 1.0), levels)
    return mx.sort(levels, axis=1)


def _reference_quantize(x: mx.array, levels: mx.array) -> mx.array:
    diff = mx.abs(x.astype(mx.float32)[:, :, None, :] - levels[:, None, :, :])
    return mx.argmin(diff, axis=2).astype(mx.int32)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("n", [16, 257])
def test_fit_matches_reference(bits, n):
    mx.random.seed(bits * 1000 + n)
    x = mx.random.normal((4, n, 32)) * mx.random.uniform(0.2, 3.0, (4, 1, 32)) + 1.5
    got = kq.fit_nuq_levels_batched(x, bits)
    want = _reference_fit(x, bits)
    mx.eval(got, want)
    assert got.shape == want.shape
    assert float(mx.max(mx.abs(got - want))) < 1e-3


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_quantize_matches_reference(bits):
    mx.random.seed(bits)
    x = mx.random.normal((4, 300, 32))
    levels = _reference_fit(x, bits)
    got = kq.quantize_nuq_batched(x, levels)
    want = _reference_quantize(x, levels)
    mx.eval(got, want)
    assert bool(mx.all(got == want).item())


def test_degenerate_constant_column_has_no_nans():
    x = mx.concatenate([mx.ones((2, 64, 4)), mx.random.normal((2, 64, 4))], axis=2)
    levels = kq.fit_nuq_levels_batched(x, 3)
    mx.eval(levels)
    assert not bool(mx.any(mx.isnan(levels)).item())
    codes = kq.quantize_nuq_batched(x, levels)
    mx.eval(codes)
    assert int(mx.max(codes).item()) < 8


def test_fit_builds_no_bh_n_l_d_tensor():
    """Peak memory of the fit must not scale with N * L * D."""
    bh, n, d, bits = 8, 4096, 128, 3
    x = mx.random.normal((bh, n, d)).astype(mx.float16)
    mx.eval(x)
    mx.synchronize()
    mx.reset_peak_memory()
    base = mx.get_peak_memory()
    mx.eval(kq.fit_nuq_levels_batched(x, bits))
    delta = mx.get_peak_memory() - base
    one_tensor = bh * n * (1 << bits) * d * 4  # one fp32 [BH, N, L, D]
    buf = bh * n * d * 4  # one fp32 [BH, N, D]
    # The one-hot fit peaked near 11 [BH, N, L, D] tensors (~90 [BH, N, D]
    # buffers); this one stays at a few dozen [BH, N, D] buffers.
    assert delta < 40 * buf, f"peak {delta} vs one [BH,N,L,D] tensor {one_tensor}"

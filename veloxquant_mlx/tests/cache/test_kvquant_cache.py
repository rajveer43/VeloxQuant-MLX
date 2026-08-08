"""Tests for KVQuantKVCache — non-uniform quantization + dense/sparse outliers.

16 tests covering:
  1.  Factory dispatch
  2.  Output shape (prefill + decode)
  3.  Values reconstructed within tolerance
  4.  NUQ lower MSE than uniform at equal bits on non-uniform (Laplacian) data
  5.  NUQ ~= uniform on genuinely uniform data (no false free-lunch claim)
  6.  Lloyd-Max convergence: distortion monotone non-increasing across iters
  7.  split_dense_sparse selects the true top-k by magnitude
  8.  Outlier isolation lowers MSE vs same-bit NUQ without isolation (heavy tails)
  9.  outlier_fraction=0 reduces to plain NUQ (no side-channel, no outliers)
  10. Level-table determinism (fixed init → identical levels)
  11. Decode after prefill — frozen key levels, correct accumulation
  12. Byte accounting: compressed < fp16
  13. effective_bits within [bits, bits + overhead] at realistic context
  14. Per-channel (key) vs per-token (value) axis correctness
  15. Determinism (end-to-end)
  16. No public `.bits` attribute (mlx_lm SDPA dispatch trap)
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kvquant_cache import KVQuantKVCache
from veloxquant_mlx.quantizers.kvquant import (
    fit_nuq_levels,
    quantize_nuq,
    dequant_nuq,
    split_dense_sparse,
    nuq_quant_dequant,
    nuq_distortion,
)
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(**kwargs) -> KVCacheConfig:
    d = dict(method="kvquant", head_dim=64, kvquant_bits=3)
    d.update(kwargs)
    return KVCacheConfig(**d)


def _laplace(B=1, H=2, S=64, D=64, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.laplace(0, 1, (B, H, S, D)).astype(np.float16))


def _mse(a, b):
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_cfg())
    assert isinstance(cache, KVQuantKVCache)


# ---------------------------------------------------------------------------
# Test 2 — output shape (prefill + decode)
# ---------------------------------------------------------------------------
def test_output_shape_prefill_decode():
    cache = KVQuantKVCache(_cfg())
    ko, vo = cache.update_and_fetch(_laplace(1, 2, 32, 64), _laplace(1, 2, 32, 64, seed=1))
    assert ko.shape == (1, 2, 32, 64) and vo.shape == (1, 2, 32, 64)
    ko2, vo2 = cache.update_and_fetch(_laplace(1, 2, 1, 64, seed=2), _laplace(1, 2, 1, 64, seed=3))
    assert ko2.shape == (1, 2, 33, 64) and vo2.shape == (1, 2, 33, 64)


# ---------------------------------------------------------------------------
# Test 3 — values reconstructed within tolerance
# ---------------------------------------------------------------------------
def test_values_reconstructed():
    cache = KVQuantKVCache(_cfg(kvquant_bits=4))
    v = _laplace(1, 2, 64, 64, seed=5)
    _, vo = cache.update_and_fetch(_laplace(1, 2, 64, 64), v)
    assert vo.shape == v.shape
    assert bool(mx.all(mx.isfinite(vo)).item())
    assert _mse(vo, v) < 0.5  # 4-bit NUQ on unit-scale Laplacian is tight


# ---------------------------------------------------------------------------
# Test 4 — NUQ beats uniform at equal bits on non-uniform data
# ---------------------------------------------------------------------------
def test_nuq_beats_uniform_on_nonuniform():
    rng = np.random.default_rng(4)
    x = mx.array(rng.laplace(0, 1, (256, 16)).astype(np.float16))
    nuq = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    uni = _group_quant_dequant(x, b=3, group_size=256)
    assert _mse(nuq, x) < _mse(uni, x), "NUQ should beat uniform on Laplacian data"


# ---------------------------------------------------------------------------
# Test 5 — NUQ ~= uniform on genuinely uniform data (no false claim)
# ---------------------------------------------------------------------------
def test_nuq_not_worse_on_uniform():
    rng = np.random.default_rng(50)
    x = mx.array(rng.uniform(-1, 1, (256, 16)).astype(np.float16))
    nuq = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    uni = _group_quant_dequant(x, b=3, group_size=256)
    # NUQ must not be materially worse than uniform where uniform is near-optimal.
    assert _mse(nuq, x) <= _mse(uni, x) * 1.2


# ---------------------------------------------------------------------------
# Test 6 — Lloyd-Max distortion monotone non-increasing
# ---------------------------------------------------------------------------
def test_lloyd_max_monotone():
    rng = np.random.default_rng(6)
    x = mx.array(rng.laplace(0, 1, (256, 8)).astype(np.float32))
    prev = None
    for it in range(1, 9):
        lv = fit_nuq_levels(x, bits=3, n_iters=it)
        d = nuq_distortion(x, lv)
        if prev is not None:
            assert d <= prev + 1e-5, f"distortion increased at iter {it}: {d} > {prev}"
        prev = d


# ---------------------------------------------------------------------------
# Test 7 — split_dense_sparse selects true top-k
# ---------------------------------------------------------------------------
def test_split_selects_top_k():
    # One column, clear outliers at known positions.
    col = np.array([0.1, 0.2, 9.0, 0.3, -8.0, 0.1, 0.2, 0.05], dtype=np.float32)
    x = mx.array(col.reshape(-1, 1))
    ds = split_dense_sparse(x, outlier_fraction=0.25)  # top 2 of 8
    mask = np.array(ds.outlier_mask.tolist()).reshape(-1).astype(bool)
    assert mask[2] and mask[4], f"expected positions 2,4 flagged, got {np.where(mask)[0]}"
    assert mask.sum() == 2


# ---------------------------------------------------------------------------
# Test 8 — outlier isolation lowers MSE on heavy-tailed data
# ---------------------------------------------------------------------------
def test_outlier_isolation_lowers_mse():
    rng = np.random.default_rng(8)
    base = rng.laplace(0, 1, (256, 8)).astype(np.float32)
    # inject a few extreme spikes
    base[rng.integers(0, 256, 5), rng.integers(0, 8, 5)] = 30.0
    x = mx.array(base.astype(np.float16))
    no_out = nuq_quant_dequant(x, bits=3, outlier_fraction=0.0)
    with_out = nuq_quant_dequant(x, bits=3, outlier_fraction=0.02)
    assert _mse(with_out, x) < _mse(no_out, x)


# ---------------------------------------------------------------------------
# Test 9 — outlier_fraction=0 reduces to plain NUQ (no outliers)
# ---------------------------------------------------------------------------
def test_outlier_fraction_zero_pure_nuq():
    cache = KVQuantKVCache(_cfg(kvquant_outlier_fraction=0.0))
    cache.update_and_fetch(_laplace(1, 2, 64, 64), _laplace(1, 2, 64, 64, seed=1))
    assert cache.outlier_count == 0


# ---------------------------------------------------------------------------
# Test 10 — level-table determinism
# ---------------------------------------------------------------------------
def test_level_table_determinism():
    rng = np.random.default_rng(10)
    x = mx.array(rng.laplace(0, 1, (256, 8)).astype(np.float32))
    l1 = fit_nuq_levels(x, bits=3, n_iters=8)
    l2 = fit_nuq_levels(x, bits=3, n_iters=8)
    np.testing.assert_array_equal(np.array(l1.tolist()), np.array(l2.tolist()))


# ---------------------------------------------------------------------------
# Test 11 — decode after prefill, frozen key levels, accumulation
# ---------------------------------------------------------------------------
def test_decode_frozen_key_levels():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 2, 20, 64), _laplace(1, 2, 20, 64, seed=1))
    frozen = [np.array(l.tolist()) for l in cache.key_levels]
    for step in range(5):
        kd = _laplace(1, 2, 1, 64, seed=100 + step)
        vd = _laplace(1, 2, 1, 64, seed=200 + step)
        ko, _ = cache.update_and_fetch(kd, vd)
        assert ko.shape[2] == 20 + step + 1
    # Key levels unchanged (refit_interval=0 → frozen).
    for a, b in zip(frozen, cache.key_levels):
        np.testing.assert_array_equal(a, np.array(b.tolist()))


# ---------------------------------------------------------------------------
# Test 12 — byte accounting compressed < fp16
# ---------------------------------------------------------------------------
def test_byte_accounting():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 2, 512, 64), _laplace(1, 2, 512, 64, seed=1))
    assert cache.compressed_key_bytes < cache.fp16_key_bytes
    assert cache.compressed_value_bytes < cache.fp16_value_bytes


# ---------------------------------------------------------------------------
# Test 13 — effective_bits within [bits, bits + overhead] at realistic context
# ---------------------------------------------------------------------------
def test_effective_bits_range():
    cache = KVQuantKVCache(_cfg(kvquant_bits=3, kvquant_outlier_fraction=0.01))
    cache.update_and_fetch(_laplace(1, 2, 1024, 64), _laplace(1, 2, 1024, 64, seed=1))
    eff = cache.effective_bits
    assert 3.0 <= eff <= 4.0, f"effective_bits={eff} out of [3.0, 4.0]"


# ---------------------------------------------------------------------------
# Test 14 — per-channel (key) vs per-token (value) axis correctness
# ---------------------------------------------------------------------------
def test_key_value_axes():
    cache = KVQuantKVCache(_cfg())
    cache.update_and_fetch(_laplace(1, 1, 64, 64), _laplace(1, 1, 64, 64, seed=1))
    # Keys: per-channel levels → [L, D] with D = head_dim columns.
    kl = cache.key_levels[0]
    assert kl.shape[0] == (1 << cache.nuq_bits) and kl.shape[1] == 64
    # Values: per-token levels (transposed space) → columns are tokens (S=64).
    vl = cache.value_levels[0]
    assert vl.shape[0] == (1 << cache.nuq_bits) and vl.shape[1] == 64


# ---------------------------------------------------------------------------
# Test 15 — determinism end-to-end
# ---------------------------------------------------------------------------
def test_determinism():
    k = _laplace(1, 2, 64, 64, seed=77)
    v = _laplace(1, 2, 64, 64, seed=88)

    def run():
        c = KVQuantKVCache(_cfg())
        ko, vo = c.update_and_fetch(k, v)
        return np.array(ko.tolist()), np.array(vo.tolist())

    k1, v1 = run()
    k2, v2 = run()
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Test 16 — no public .bits attribute (mlx_lm SDPA dispatch trap)
# ---------------------------------------------------------------------------
def test_no_bits_leak():
    """mlx_lm's SDPA checks `hasattr(cache, "bits")` to route to its
    quantized-matmul kernel, which expects mx.quantize's native tuple layout
    and a `.group_size` attribute this cache doesn't have. Exposing `.bits`
    here would silently hijack attention dispatch — see issue #87."""
    cache = KVQuantKVCache(_cfg(kvquant_bits=3))
    assert not hasattr(cache, "bits"), (
        "KVQuantKVCache must not expose .bits — would break mlx_lm SDPA dispatch"
    )
    assert hasattr(cache, "nuq_bits")
    assert cache.nuq_bits == 3

"""Tests for KIVIKVCache — asymmetric group-quant cache wrapper.

Covers mlx_lm protocol shape/dtype preservation, the fp16 residual window
(recent tokens kept exact), byte accounting, and the no-``.bits``-leak
invariant that keeps mlx_lm's SDPA on the standard fp16 path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kivi_cache import KIVIKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(method_kw=None, **cfg):
    base = dict(
        method="kivi", head_dim=128, bit_width_inlier=2, residual_length=16, kivi_group_size=32
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def test_factory_dispatch() -> None:
    cache = _make()
    assert isinstance(cache, KIVIKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make()
    k, v = _kv(1, 4, 64, 128)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 128)
    assert vo.shape == (1, 4, 64, 128)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    # mlx_lm's SDPA checks hasattr(cache, "bits") to route to a quantized
    # kernel; KIVIKVCache must not expose it (it dequantizes to fp16).
    cache = _make()
    assert not hasattr(cache, "bits")
    assert hasattr(cache, "assigned_avg_bits")


def test_residual_window_kept_fp16() -> None:
    """When S <= residual_length the whole block passes through untouched."""
    cache = _make(residual_length=64)
    k, v = _kv(1, 2, 32, 128, seed=1)  # S=32 < residual 64
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    # Exact passthrough: output equals input bit-for-bit.
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))
    assert cache.compressed_key_bytes == 0  # nothing quantized yet


def test_quantizes_aged_tokens() -> None:
    """When S > residual_length the oldest (S - r) tokens are quantized."""
    cache = _make(residual_length=16)
    k, v = _kv(1, 2, 80, 128, seed=2)  # 64 quantized, 16 residual
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    ko_np, k_np = np.array(ko), np.array(k)
    # Residual tail (last 16) is exact; quantized head differs.
    assert np.array_equal(ko_np[:, :, -16:, :], k_np[:, :, -16:, :])
    head_diff = np.mean(np.abs(ko_np[:, :, :64, :] - k_np[:, :, :64, :]))
    assert head_diff > 0.0, "quantized region should differ from fp16 input"
    assert cache.compressed_key_bytes > 0
    assert cache.residual_fp16_bytes > 0


def test_compression_ratio_below_fp16() -> None:
    """Quantized region (+ params + residual) must beat fp16."""
    cache = _make(bit_width_inlier=2, residual_length=16)
    k, v = _kv(1, 4, 256, 128, seed=3)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    total_comp = (
        cache.compressed_key_bytes + cache.compressed_value_bytes + cache.residual_fp16_bytes
    )
    total_fp16 = cache.fp16_key_bytes + cache.fp16_value_bytes
    ratio = total_fp16 / total_comp
    assert ratio > 1.5, f"end-to-end ratio {ratio:.2f} not better than fp16"


def test_decode_step_after_prefill() -> None:
    """Simulate prefill + single-token decode steps (S==1)."""
    cache = _make(residual_length=8)
    k, v = _kv(1, 2, 40, 128, seed=4)
    cache.update_and_fetch(k, v)
    for t in range(5):
        k1, v1 = _kv(1, 2, 1, 128, seed=100 + t)
        ko, vo = cache.update_and_fetch(k1, v1)
        mx.eval(ko, vo)
    assert cache.offset == 45


@pytest.mark.parametrize("b", [2, 4])
def test_higher_bits_more_compressed_bytes_but_better_quality(b: int) -> None:
    cache = _make(bit_width_inlier=b, residual_length=16)
    k, v = _kv(1, 2, 128, 128, seed=b)
    ko, _ = cache.update_and_fetch(k, v)
    mx.eval(ko)
    assert cache.assigned_avg_bits == float(b)


def test_decode_tokens_age_out_of_residual_window() -> None:
    """Regression for #86: with S==1 every decode step, tokens must still
    age out of the fp16 residual window once total cumulative length
    exceeds residual_length — the residual byte count must plateau, not
    grow unboundedly with decode steps."""
    cache = _make(residual_length=4, kivi_group_size=4)
    k, v = _kv(1, 1, 10, 128, seed=7)
    cache.update_and_fetch(k, v)  # prefill: 10 tokens, 6 quantized, 4 residual

    for t in range(50):
        k1, v1 = _kv(1, 1, 1, 128, seed=200 + t)
        cache.update_and_fetch(k1, v1)

    assert cache.offset == 60
    D, H, B = 128, 1, 1
    expected_residual = 4 * D * 2 * 2 * H * B  # residual_length * K+V * fp16 bytes
    assert cache.residual_fp16_bytes == expected_residual, (
        f"residual_fp16_bytes={cache.residual_fp16_bytes} did not plateau at "
        f"{expected_residual} after 50 decode steps past the residual window"
    )
    assert cache.compressed_key_bytes > 0

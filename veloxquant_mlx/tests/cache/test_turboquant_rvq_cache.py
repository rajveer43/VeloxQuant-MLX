"""Tests for the TurboQuantRVQKVCache mlx_lm wrapper."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache


def _build(bits: int = 1, head_dim: int = 128):
    cfg = KVCacheConfig(
        method="turboquant_rvq",
        head_dim=head_dim,
        bit_width_inlier=bits,
        seed=0,
    )
    return KVCacheFactory.create(cfg)


def test_factory_creates_rvq_cache() -> None:
    c = _build(bits=1)
    assert isinstance(c, TurboQuantRVQKVCache)


def test_update_and_fetch_preserves_shape() -> None:
    c = _build(bits=2)
    keys = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
    k_out, v_out = c.update_and_fetch(keys, vals)
    assert k_out.shape == (1, 8, 16, 128)
    assert v_out.shape == (1, 8, 16, 128)


def test_byte_accounting_matches_formula() -> None:
    """Per-vector storage = ceil(d * 2 * b / 8) + 2."""
    head_dim, bits, H, S = 128, 1, 8, 16
    c = _build(bits=bits, head_dim=head_dim)
    keys = mx.random.normal((1, H, S, head_dim)).astype(mx.float16)
    vals = mx.random.normal((1, H, S, head_dim)).astype(mx.float16)
    c.update_and_fetch(keys, vals)

    expected_per_vec = math.ceil(head_dim * 2 * bits / 8) + 2
    expected_total = expected_per_vec * H * S
    assert c.compressed_key_bytes == expected_total
    assert c.fp16_key_bytes == H * S * head_dim * 2


def test_compression_ratio_rvq_1bit() -> None:
    """RVQ 1-bit at d=128 must give 7.5× compression on keys."""
    c = _build(bits=1, head_dim=128)
    keys = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    c.update_and_fetch(keys, vals)
    ratio = c.fp16_key_bytes / c.compressed_key_bytes
    assert 7.0 < ratio < 8.0, f"unexpected ratio {ratio}"


def test_does_not_expose_public_bits_attr() -> None:
    """mlx_lm.scaled_dot_product_attention checks hasattr(cache, 'bits') to
    route to its quantized SDPA kernel — we must NOT expose .bits."""
    c = _build(bits=1)
    assert not hasattr(c, "bits"), (
        "Cache must not expose public .bits — would break mlx_lm SDPA dispatch"
    )
    assert hasattr(c, "assigned_bits")
    assert c.assigned_bits == 1


def test_rejects_list_bit_width() -> None:
    """List form must be handled by KVCacheBuilder.for_model, not the
    individual cache class."""
    cfg = KVCacheConfig(
        method="turboquant_rvq",
        head_dim=128,
        bit_width_inlier=[1, 2],
        seed=0,
    )
    with pytest.raises(TypeError, match="list-form"):
        TurboQuantRVQKVCache(cfg)

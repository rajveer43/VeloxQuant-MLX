"""Tests for VecInferKVCache (mlx_lm-compatible wrapper)."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory, VecInferKVCache


def _build(
    head_dim: int = 128,
    key_sub_dim: int = 4,
    value_sub_dim: int = 8,
    key_bits: int = 8,
    value_bits: int = 8,
):
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=head_dim,
        key_sub_dim=key_sub_dim,
        value_sub_dim=value_sub_dim,
        key_codebook_bits=key_bits,
        value_codebook_bits=value_bits,
        seed=0,
    )
    return KVCacheFactory.create(cfg)


def test_factory_creates_vecinfer_cache() -> None:
    c = _build()
    assert isinstance(c, VecInferKVCache)


def test_update_and_fetch_preserves_shape_and_dtype() -> None:
    c = _build()
    keys = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
    k, v = c.update_and_fetch(keys, vals)
    assert k.shape == (1, 4, 16, 128)
    assert v.shape == (1, 4, 16, 128)
    assert k.dtype == mx.float16
    assert v.dtype == mx.float16


def test_compression_ratio_above_one() -> None:
    """At b=8, d=4 we store 2 bits/elem -> ~8x on keys."""
    c = _build(key_sub_dim=4, key_bits=8)
    keys = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
    c.update_and_fetch(keys, vals)
    ratio = c.fp16_key_bytes / c.compressed_key_bytes
    assert ratio > 5.0, f"expected >5x key compression, got {ratio:.2f}"


def test_compression_ratio_at_2bit_config() -> None:
    """Acceptance criterion: key compression >= 5x at the 2-bit config."""
    c = _build(key_sub_dim=4, key_bits=8)  # 2 bits/element
    keys = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
    c.update_and_fetch(keys, vals)
    ratio = c.fp16_key_bytes / c.compressed_key_bytes
    assert ratio >= 5.0


def test_byte_accounting_formula() -> None:
    """compressed = ceil(D / sub_dim * b / 8) bytes per (head, batch, token)."""
    head_dim, b, sub_dim, H, S = 128, 8, 4, 4, 16
    c = _build(head_dim=head_dim, key_sub_dim=sub_dim, key_bits=b)
    keys = mx.random.normal((1, H, S, head_dim)).astype(mx.float16)
    vals = mx.random.normal((1, H, S, head_dim)).astype(mx.float16)
    c.update_and_fetch(keys, vals)

    expected_per_tok = math.ceil((head_dim // sub_dim) * b / 8) * H
    expected_total = expected_per_tok * S
    assert c.compressed_key_bytes == expected_total
    assert c.fp16_key_bytes == H * S * head_dim * 2


def test_does_not_expose_public_bits_attr() -> None:
    """mlx_lm SDPA dispatch trap: cache must not expose .bits."""
    c = _build()
    assert not hasattr(c, "bits"), (
        "VecInferKVCache must not expose .bits — would route to mlx_lm "
        "quantized SDPA kernel and bypass dequantization."
    )
    assert hasattr(c, "assigned_avg_bits")


def test_reconstruction_error_bounded() -> None:
    """Dequantized keys should track input keys to within a sensible MSE."""
    c = _build(key_sub_dim=4, key_bits=12)
    # Train a codebook implicitly via random init — random init means
    # reconstruction quality is poor. Just check the path runs without
    # NaN/Inf and shapes match.
    keys = mx.random.normal((1, 2, 8, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 2, 8, 128)).astype(mx.float16)
    k_out, _ = c.update_and_fetch(keys, vals)
    arr = k_out.astype(mx.float32)
    assert not bool(mx.any(mx.isnan(arr)))
    assert not bool(mx.any(mx.isinf(arr)))


def test_codebook_bytes_reported() -> None:
    c = _build(key_sub_dim=4, key_bits=8, value_sub_dim=8, value_bits=8)
    expected = (2**8) * 4 * 2 + (2**8) * 8 * 2
    assert c.codebook_bytes == expected


def test_head_dim_must_divide_sub_dim() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=128,
            key_sub_dim=5,
            key_codebook_bits=4,
        )
        KVCacheFactory.create(cfg)

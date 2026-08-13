"""Tests for the TurboQuantRVQKVCache mlx_lm wrapper.

Regression coverage for issue #27's packed-storage tier: this cache used to
dequantize keys to fp16 on every update_and_fetch and store the fp16 tensor
via inherited mlx_lm.models.cache.KVCache storage -- nbytes/compressed_key_bytes
were bit-width *accounting*, not the cache's actual resident size. It now
stores the two RVQ index streams bit-packed into uint32 words (see
_pack_indices/_unpack_indices in the module under test) and only
dequantizes transiently inside update_and_fetch/attend, so nbytes reflects
real packed storage.
"""

from __future__ import annotations

import copy
import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.turboquant_rvq_cache import (
    TurboQuantRVQKVCache,
    _pack_indices,
    _unpack_indices,
)
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ


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


# ---------------------------------------------------------------------------
# Packed-storage regression tests (issue #27 tier-1: real bytes, not accounting)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_pack_unpack_round_trip(bits: int) -> None:
    """_pack_indices/_unpack_indices must be lossless for every supported bit-width."""
    d = 128
    n = 37  # deliberately not a multiple of the packing group, to exercise padding
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 2**bits, size=(n, d)).astype(np.uint8)
    idx = mx.array(orig)

    packed = _pack_indices(idx, bits)
    unpacked = _unpack_indices(packed, bits, d)

    assert np.array_equal(np.array(unpacked), orig)


def test_packed_storage_matches_direct_encode_decode() -> None:
    """update_and_fetch's packed round trip must be bit-exact against calling
    TurboQuantRVQ.encode/decode directly with no cache storage involved --
    i.e. packing/unpacking introduces zero additional reconstruction error
    beyond the quantizer's own.
    """
    mx.random.seed(0)
    B, H, S, D, bits = 1, 4, 20, 128, 2

    cache = _build(bits=bits, head_dim=D)
    keys = mx.random.normal((B, H, S, D)).astype(mx.float16)
    values = mx.random.normal((B, H, S, D)).astype(mx.float16)
    k_out, _ = cache.update_and_fetch(keys, values)

    quantizer = TurboQuantRVQ(d=D, b=bits, seed=0, use_hadamard=True)
    k_flat = keys.reshape(-1, D)
    norms = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True).astype(keys.dtype)
    safe = mx.maximum(norms, mx.array(1e-4, dtype=keys.dtype))
    k_unit = (k_flat / safe).astype(mx.float16)
    ev = quantizer.encode(k_unit)
    k_hat_direct = quantizer.decode(ev)
    k_hat_direct = (k_hat_direct.astype(keys.dtype) * safe).reshape(B, H, S, D)

    assert np.array(mx.abs(k_out - k_hat_direct)).max() == 0.0


def test_growth_across_step_boundary() -> None:
    """Repeated small update_and_fetch calls (decode-shaped) must accumulate
    correctly across the step=256 pre-allocation boundary, matching
    mlx_lm.models.cache.KVCache's own growth contract.
    """
    mx.random.seed(1)
    B, H, D = 1, 4, 128
    cache = _build(bits=1, head_dim=D)

    keys = mx.random.normal((B, H, 10, D)).astype(mx.float16)
    vals = mx.random.normal((B, H, 10, D)).astype(mx.float16)
    k_out, v_out = cache.update_and_fetch(keys, vals)
    assert k_out.shape == (B, H, 10, D)

    for step in range(300):
        k1 = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        v1 = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(k1, v1)
        expected_len = 10 + step + 1
        assert k_out.shape == (B, H, expected_len, D)
        assert v_out.shape == (B, H, expected_len, D)

    assert cache.offset == 310


def test_trim() -> None:
    cache = _build(bits=2, head_dim=128)
    keys = mx.random.normal((1, 4, 15, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 15, 128)).astype(mx.float16)
    cache.update_and_fetch(keys, vals)

    assert cache.is_trimmable()
    n_trimmed = cache.trim(5)
    assert n_trimmed == 5
    assert cache.offset == 10


def test_empty() -> None:
    cache = _build(bits=1, head_dim=128)
    assert cache.empty()
    keys = mx.random.normal((1, 2, 3, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 2, 3, 128)).astype(mx.float16)
    cache.update_and_fetch(keys, vals)
    assert not cache.empty()


def test_deepcopy_isolates_state() -> None:
    """Per-request cache isolation (mlx_lm.server's fetch_nearest_cache uses
    copy.deepcopy per request, per issue #27 finding 3) must hold for the
    new packed storage the same way it held for the old fp16 storage.
    """
    mx.random.seed(2)
    cache = _build(bits=2, head_dim=128)
    keys = mx.random.normal((1, 4, 15, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 15, 128)).astype(mx.float16)
    cache.update_and_fetch(keys, vals)

    cache2 = copy.deepcopy(cache)
    assert cache2 is not cache
    assert cache2.offset == cache.offset == 15

    more_keys = mx.random.normal((1, 4, 5, 128)).astype(mx.float16)
    more_vals = mx.random.normal((1, 4, 5, 128)).astype(mx.float16)
    cache.update_and_fetch(more_keys, more_vals)

    assert cache.offset == 20
    assert cache2.offset == 15


def test_state_meta_state_round_trip_via_from_state() -> None:
    """mlx_lm.server reconstructs caches via cls.from_state(state, meta_state)
    -- verify this path preserves offset and lets generation continue.
    """
    mx.random.seed(3)
    cache = _build(bits=1, head_dim=128)
    keys = mx.random.normal((1, 2, 12, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 2, 12, 128)).astype(mx.float16)
    cache.update_and_fetch(keys, vals)

    restored = cache.__class__.from_state(cache.state, cache.meta_state)
    assert restored.offset == cache.offset == 12

    k1 = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    v1 = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    k_out, v_out = restored.update_and_fetch(k1, v1)
    assert k_out.shape == (1, 2, 13, 128)
    assert restored.offset == 13


def test_nbytes_reports_true_packed_size_not_fp16() -> None:
    """nbytes must reflect genuinely packed storage: strictly less than what
    an equivalent fp16 mlx_lm.models.cache.KVCache would report for the same
    number of cached tokens -- this is issue #27's core acceptance criterion
    (measured bytes, not bit-width accounting alongside fp16-resident storage).
    """
    from mlx_lm.models.cache import KVCache as _MLXKVCache

    B, H, S, D, bits = 1, 8, 64, 128, 1
    cache = _build(bits=bits, head_dim=D)
    keys = mx.random.normal((B, H, S, D)).astype(mx.float16)
    vals = mx.random.normal((B, H, S, D)).astype(mx.float16)
    cache.update_and_fetch(keys, vals)

    fp16_cache = _MLXKVCache()
    fp16_cache.update_and_fetch(keys, vals)

    assert cache.nbytes < fp16_cache.nbytes, (
        f"packed nbytes ({cache.nbytes}) must be strictly less than the "
        f"fp16-resident baseline ({fp16_cache.nbytes})"
    )
    # At b=1, d=128 the key stream alone should give a substantial reduction
    # (roughly 8x on keys per the module docstring); assert a conservative
    # 2x reduction on total nbytes (keys+values+norm vs keys+values fp16) to
    # avoid over-fitting the test to the exact current constant.
    assert cache.nbytes < fp16_cache.nbytes / 2

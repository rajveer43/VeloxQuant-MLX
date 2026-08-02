"""Tests for TurboQuantKVCache."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cache():
    from veloxquant_mlx.cache.base import KVCacheBuilder

    return (
        KVCacheBuilder()
        .with_method("turboquant_prod")
        .with_head_dim(64)
        .with_bit_width(inlier=2)
        .with_jl_dim(64)
        .with_seed(42)
        .build()
    )


def test_cache_append_and_len(cache) -> None:
    import mlx.core as mx

    rng = np.random.default_rng(0)
    for _ in range(10):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        cache.append(k, v)
    assert len(cache) == 10


def test_cache_attend_shape(cache) -> None:
    import mlx.core as mx
    import numpy as np

    q = mx.array(np.random.randn(64).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (64,)


def test_cache_memory_bytes(cache) -> None:
    assert cache.memory_bytes() > 0


def test_builder_validation() -> None:
    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    with pytest.raises(QuantizerConfigError):
        (
            KVCacheBuilder()
            .with_method("turboquant_prod")
            .with_head_dim(100)  # not power of 2
            .with_bit_width(inlier=2)
            .build()
        )

    with pytest.raises(QuantizerConfigError):
        (
            KVCacheBuilder()
            .with_method("turboquant_prod")
            .with_head_dim(64)
            .with_bit_width(inlier=2)
            .with_jl_dim(128)  # jl_dim > head_dim
            .build()
        )


def test_empty_cache_attend() -> None:
    import mlx.core as mx
    from veloxquant_mlx.cache.base import KVCacheBuilder

    cache = (
        KVCacheBuilder()
        .with_method("turboquant_mse")
        .with_head_dim(64)
        .with_bit_width(inlier=2)
        .build()
    )
    q = mx.array(np.random.randn(64).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (64,)


def _build_cache_with_flags(
    *,
    method: str = "turboquant_prod",
    vectorized: bool = False,
    outlier: bool = False,
    fused: bool = False,
    n_outliers: int = 4,
    n_calib: int = 8,
):
    from veloxquant_mlx.cache.base import KVCacheBuilder

    return (
        KVCacheBuilder()
        .with_method(method)
        .with_head_dim(64)
        .with_bit_width(inlier=3)
        .with_jl_dim(64)
        .with_seed(42)
        .with_vectorized_attend(vectorized)
        .with_outlier_two_stream(outlier)
        .with_fused_query_dot(fused)
        .with_n_outlier_channels(n_outliers)
        .with_n_calib_tokens(n_calib)
        .build()
    )


def test_vectorized_attend_matches_baseline() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(123)
    cache_base = _build_cache_with_flags(method="turboquant_mse", vectorized=False)
    cache_vec = _build_cache_with_flags(method="turboquant_mse", vectorized=True)

    for _ in range(64):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        cache_base.append(k, v)
        cache_vec.append(k, v)

    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out_base = cache_base.attend(q)
    out_vec = cache_vec.attend(q)
    mx.eval(out_base, out_vec)
    np.testing.assert_allclose(np.array(out_vec), np.array(out_base), rtol=5e-3, atol=5e-3)


def test_outlier_warmup_and_selection() -> None:
    import mlx.core as mx

    cache = _build_cache_with_flags(
        method="turboquant_mse",
        vectorized=True,
        outlier=True,
        n_outliers=2,
        n_calib=4,
    )
    rng = np.random.default_rng(0)
    for t in range(3):
        k = rng.standard_normal(64).astype(np.float16)
        k[5] = np.float16(8.0)
        k[17] = np.float16(-7.0)
        v = rng.standard_normal(64).astype(np.float16)
        cache.append(mx.array(k), mx.array(v))
        assert cache._outlier_idx is None

    k = rng.standard_normal(64).astype(np.float16)
    k[5] = np.float16(8.0)
    k[17] = np.float16(-7.0)
    cache.append(mx.array(k), mx.array(rng.standard_normal(64).astype(np.float16)))
    assert cache._outlier_idx is not None
    assert 5 in set(cache._outlier_idx.tolist())


def test_outlier_encode_decode_correctness() -> None:
    """Outlier channels are int8-quantized and dequantize within expected tolerance."""
    import mlx.core as mx

    n_outliers = 4
    n_calib = 4
    cache = _build_cache_with_flags(
        method="turboquant_mse",
        vectorized=True,
        outlier=True,
        n_outliers=n_outliers,
        n_calib=n_calib,
    )
    rng = np.random.default_rng(7)

    # Use channels 0..3 as outliers by giving them large constant magnitude.
    stored_keys = []
    for t in range(n_calib + 2):
        k = rng.standard_normal(64).astype(np.float16)
        for c in range(n_outliers):
            k[c] = np.float16(10.0 + c)
        v = rng.standard_normal(64).astype(np.float16)
        cache.append(mx.array(k), mx.array(v))
        stored_keys.append(k.copy())

    # After calibration the outlier index must be set.
    assert cache._outlier_idx is not None
    out_idx = cache._outlier_idx  # shape (n_outliers,)

    # Verify int8 round-trip for the last appended token.
    last_slot = (cache._head + cache._size - 1) % cache._capacity
    raw_int8 = cache._outlier_cache[last_slot]  # (n_outliers,) int8
    scale = float(cache._outlier_scales[last_slot])  # fp16 scalar
    dequant = raw_int8.astype(np.float32) * scale  # approximate fp16 values

    k_last = stored_keys[-1]
    k_out_true = k_last[out_idx].astype(np.float32)
    np.testing.assert_allclose(
        dequant,
        k_out_true,
        rtol=0.02,
        atol=0.05,
        err_msg="Outlier int8 dequant diverges from original",
    )


def test_outlier_combined_attend_reconstruction() -> None:
    """attend() with outlier two-stream: output is finite and outlier storage is populated."""
    import mlx.core as mx

    n_outliers = 4
    n_calib = 4
    cache_out = _build_cache_with_flags(
        method="turboquant_mse",
        vectorized=True,
        outlier=True,
        n_outliers=n_outliers,
        n_calib=n_calib,
    )

    rng = np.random.default_rng(99)
    n_total = n_calib + 8
    for _ in range(n_total):
        k = rng.standard_normal(64).astype(np.float16)
        k[0] = np.float16(10.0)
        k[1] = np.float16(-9.0)
        v = rng.standard_normal(64).astype(np.float16)
        cache_out.append(mx.array(k), mx.array(v))

    # Outlier channels must have been identified.
    assert cache_out._outlier_idx is not None
    assert len(cache_out._outlier_idx) == n_outliers

    # Post-calibration slots must have non-zero outlier scales.
    phys = cache_out._physical_indices(cache_out._size)
    scales = cache_out._outlier_scales[phys]
    n_post = n_total - n_calib
    assert np.sum(scales > 0) >= n_post, (
        f"Expected >= {n_post} populated outlier slots, got {np.sum(scales > 0)}"
    )

    # attend() must return a finite, correctly shaped vector.
    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out = cache_out.attend(q)
    mx.eval(out)
    assert out.shape == (64,)
    assert np.all(np.isfinite(np.array(out))), "Two-stream attend produced non-finite values"


def test_fused_query_dot_matches_baseline() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(321)
    cache_base = _build_cache_with_flags(
        method="turboquant_prod",
        vectorized=True,
        outlier=False,
        fused=False,
    )
    cache_fused = _build_cache_with_flags(
        method="turboquant_prod",
        vectorized=True,
        outlier=False,
        fused=True,
    )

    for _ in range(48):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        cache_base.append(k, v)
        cache_fused.append(k, v)

    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out_base = cache_base.attend(q)
    out_fused = cache_fused.attend(q)
    mx.eval(out_base, out_fused)
    np.testing.assert_allclose(np.array(out_fused), np.array(out_base), rtol=5e-3, atol=5e-3)

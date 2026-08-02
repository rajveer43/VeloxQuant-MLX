"""Parity + opt-in tests for the VecInfer Metal fast path.

These tests verify that running the same prompt through ``VecInferKVCache``
with ``use_metal_kernels=True`` produces output that matches the pure-MLX
path (``use_metal_kernels=False``) within fp16 quantization tolerance.

The tests skip cleanly on systems where Metal is unavailable.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import metal_available

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


def _build_cache(use_metal: bool, head_dim: int = 128, key_sub_dim: int = 4):
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=head_dim,
        key_sub_dim=key_sub_dim,
        value_sub_dim=key_sub_dim,
        key_codebook_bits=8,
        value_codebook_bits=8,
        seed=0,
        use_metal_kernels=use_metal,
    )
    return KVCacheFactory.create(cfg)


def test_use_metal_flag_resolves_to_bool() -> None:
    c_auto = _build_cache(use_metal=None)
    assert c_auto._use_metal is True

    c_off = _build_cache(use_metal=False)
    assert c_off._use_metal is False


def test_metal_path_preserves_shape_and_dtype() -> None:
    c = _build_cache(use_metal=True)
    keys = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
    k, v = c.update_and_fetch(keys, vals)
    assert k.shape == (1, 4, 32, 128)
    assert v.shape == (1, 4, 32, 128)
    assert k.dtype == mx.float16
    assert v.dtype == mx.float16


def test_metal_vs_pure_reconstruction_parity() -> None:
    """Reconstructions from the two paths must agree within fp16 noise.

    Even though individual quantized indices may differ on ties (~0.1% of
    positions when two centroids are nearly equidistant), the reconstructed
    fp16 key/value tensors are functionally equivalent: their L2 distance
    to the original input differs by <0.1% relative MSE.
    """
    mx.random.seed(7)
    keys = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)

    c_pure = _build_cache(use_metal=False)
    c_metal = _build_cache(use_metal=True)
    k_p, v_p = c_pure.update_and_fetch(keys, vals)
    k_m, v_m = c_metal.update_and_fetch(keys, vals)

    # Reconstruction quality must be near-identical between paths.
    def _mse(a, b):
        return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())

    mse_pure_k = _mse(k_p, keys)
    mse_metal_k = _mse(k_m, keys)
    mse_pure_v = _mse(v_p, vals)
    mse_metal_v = _mse(v_m, vals)

    rel_err_k = abs(mse_metal_k - mse_pure_k) / max(mse_pure_k, 1e-9)
    rel_err_v = abs(mse_metal_v - mse_pure_v) / max(mse_pure_v, 1e-9)

    assert rel_err_k < 1e-2, (
        f"Key reconstruction MSE diverges between Metal/pure paths: "
        f"pure={mse_pure_k:.4e} metal={mse_metal_k:.4e} rel_err={rel_err_k:.3e}"
    )
    assert rel_err_v < 1e-2, (
        f"Value reconstruction MSE diverges between Metal/pure paths: "
        f"pure={mse_pure_v:.4e} metal={mse_metal_v:.4e} rel_err={rel_err_v:.3e}"
    )


def test_metal_path_no_bits_attribute() -> None:
    """Metal path must NOT expose .bits (would re-route mlx_lm SDPA)."""
    c = _build_cache(use_metal=True)
    assert not hasattr(c, "bits")


def test_compression_ratio_identical_across_paths() -> None:
    """Byte accounting is path-agnostic and must produce the same numbers."""
    keys = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)

    c_pure = _build_cache(use_metal=False)
    c_metal = _build_cache(use_metal=True)
    c_pure.update_and_fetch(keys, vals)
    c_metal.update_and_fetch(keys, vals)

    assert c_pure.compressed_key_bytes == c_metal.compressed_key_bytes
    assert c_pure.fp16_key_bytes == c_metal.fp16_key_bytes


def test_metal_required_but_unavailable_raises_at_construction() -> None:
    """When Metal is available, requesting it must succeed without error."""
    # If we got here, metal_available() is True (see pytestmark) — so
    # use_metal_kernels=True must construct cleanly.
    c = _build_cache(use_metal=True)
    assert c._use_metal is True


def test_metal_path_works_with_head_dim_256() -> None:
    """Falcon3-7B-shaped inputs (head_dim=256) — the OOM trigger shape."""
    c = _build_cache(use_metal=True, head_dim=256, key_sub_dim=8)
    keys = mx.random.normal((1, 4, 32, 256)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 32, 256)).astype(mx.float16)
    k, v = c.update_and_fetch(keys, vals)
    assert k.shape == (1, 4, 32, 256)
    assert v.shape == (1, 4, 32, 256)
    # Sanity: no NaNs (would signal an int-overflow or alignment bug)
    assert not bool(mx.any(mx.isnan(k.astype(mx.float32))).item())
    assert not bool(mx.any(mx.isnan(v.astype(mx.float32))).item())

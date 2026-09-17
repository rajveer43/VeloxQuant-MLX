"""Tests for CompositeQuantizer's outlier/inlier channel split and reassembly."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.quantizers.composite import CompositeQuantizer


class _IdentityQuantizer:
    """Stub quantizer: encode/decode are the identity, optionally casting dtype.

    Lets tests pin exactly which columns end up where after CompositeQuantizer
    reassembles the outlier/inlier halves, without a real quantizer's lossy
    rounding in the way.
    """

    def __init__(self, dtype=mx.float16):
        self._dtype = dtype

    def encode(self, x):
        return EncodedVector(
            quantizer_type="identity", batch_size=x.shape[0], dim=x.shape[1], indices=x
        )

    def decode(self, ev):
        return ev.indices.astype(self._dtype)

    def estimate_inner_product(self, q, ev):
        return mx.sum(q[None, :] * ev.indices, axis=-1)


def _make_quantizer(total_dim, outlier_idx, outlier_dtype=mx.float16, inlier_dtype=mx.float16):
    return CompositeQuantizer(
        outlier_quantizer=_IdentityQuantizer(outlier_dtype),
        inlier_quantizer=_IdentityQuantizer(inlier_dtype),
        outlier_idx=np.asarray(outlier_idx, dtype=np.int32),
        total_dim=total_dim,
    )


def test_decode_places_channels_at_original_indices():
    d = 8
    outlier_idx = [1, 5]
    q = _make_quantizer(d, outlier_idx)

    x = mx.arange(d, dtype=mx.float32)[None, :]  # [[0, 1, 2, ..., 7]]
    ev = q.encode(x)
    out = q.decode(ev)
    mx.eval(out)

    assert out.shape == (1, d)
    assert np.allclose(np.array(out), np.array(x))


def test_decode_matches_encode_for_random_batch():
    d, batch = 32, 5
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(d, size=6, replace=False)
    q = _make_quantizer(d, outlier_idx)

    x_np = rng.standard_normal((batch, d)).astype(np.float32)
    x = mx.array(x_np)
    ev = q.encode(x)
    out = q.decode(ev)
    mx.eval(out)

    assert out.shape == (batch, d)
    np.testing.assert_allclose(np.array(out), x_np, atol=1e-3)


def test_decode_output_dtype_follows_outlier_quantizer():
    """Reconstruction dtype matches the outlier (high-bit-width) child's dtype."""
    d = 6
    q = _make_quantizer(d, [0, 2], outlier_dtype=mx.float32, inlier_dtype=mx.float16)

    x = mx.random.normal((3, d)).astype(mx.float32)
    ev = q.encode(x)
    out = q.decode(ev)
    mx.eval(out)

    assert out.dtype == mx.float32


def test_decode_raises_on_incomplete_encoded_vector():
    q = _make_quantizer(4, [0])
    bad_ev = EncodedVector(quantizer_type="composite", batch_size=1, dim=4)
    with pytest.raises(ValueError, match="outlier_encoded/inlier_encoded is None"):
        q.decode(bad_ev)


def test_estimate_inner_product_matches_full_dot():
    d = 10
    rng = np.random.default_rng(1)
    outlier_idx = rng.choice(d, size=3, replace=False)
    q = _make_quantizer(d, outlier_idx)

    x_np = rng.standard_normal((4, d)).astype(np.float32)
    query_np = rng.standard_normal(d).astype(np.float32)
    x = mx.array(x_np)
    query = mx.array(query_np)

    ev = q.encode(x)
    ip = q.estimate_inner_product(query, ev)
    mx.eval(ip)

    expected = x_np @ query_np
    np.testing.assert_allclose(np.array(ip), expected, atol=1e-3)

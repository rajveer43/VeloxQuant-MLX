"""Parity and bounds tests for the VecInfer codebook-gather Metal kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import vecinfer_dequant_metal

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(not metal_available(), reason="Metal kernels not available."),
]


def test_matches_codebook_gather():
    rng = np.random.default_rng(0)
    cb = mx.array(rng.standard_normal((16, 4)).astype(np.float16))
    idx = mx.array(rng.integers(0, 16, (2, 5, 3)).astype(np.uint8))
    got = vecinfer_dequant_metal(idx, cb)
    expected = cb[idx.astype(mx.int32)].reshape(2, 5, 12)
    assert np.array_equal(np.array(got), np.array(expected))


def test_out_of_range_code_is_clamped():
    cb = mx.arange(8 * 2, dtype=mx.float32).reshape(8, 2)
    idx = mx.array([[1, 200]], mx.uint32)
    got = np.array(vecinfer_dequant_metal(idx, cb))
    assert np.array_equal(got, np.array([[2, 3, 14, 15]], np.float32))

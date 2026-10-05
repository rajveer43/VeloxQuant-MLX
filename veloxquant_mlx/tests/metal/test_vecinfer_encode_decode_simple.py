"""Parity tests for the fused VecInfer value encode+decode Metal kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import (
    vecinfer_dequant_metal,
    vecinfer_encode_decode_simple_metal,
    vecinfer_quantize_metal,
)

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(not metal_available(), reason="Metal kernels not available."),
]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("D,sub_dim,n_cents", [(128, 4, 256), (64, 8, 256), (96, 2, 16), (8, 8, 4)])
@pytest.mark.parametrize("S", [1, 7, 50])
def test_matches_unfused_path(dtype, D, sub_dim, n_cents, S):
    rng = np.random.default_rng(D + S)
    v = mx.array(rng.standard_normal((2, 3, S, D)).astype(dtype))
    cb = mx.array(rng.standard_normal((n_cents, sub_dim)).astype(np.float32))

    v_hat, idx = vecinfer_encode_decode_simple_metal(v, cb, sub_dim)
    ref_idx = vecinfer_quantize_metal(v.astype(mx.float32), cb, sub_dim)
    ref_hat = vecinfer_dequant_metal(ref_idx, cb).astype(mx.float16)

    assert v_hat.shape == (2, 3, S, D) and v_hat.dtype == mx.float16
    assert idx.shape == (2, 3, S, D // sub_dim)
    assert np.array_equal(np.array(idx).astype(np.int32), np.array(ref_idx))
    assert np.array_equal(np.array(v_hat), np.array(ref_hat))


def test_empty_sequence():
    v_hat, idx = vecinfer_encode_decode_simple_metal(mx.zeros((1, 2, 0, 64)), mx.zeros((16, 4)), 4)
    assert v_hat.shape == (1, 2, 0, 64) and idx.shape == (1, 2, 0, 16)


def test_empty_codebook_rejected():
    with pytest.raises(ValueError):
        vecinfer_encode_decode_simple_metal(mx.zeros((1, 1, 2, 8)), mx.zeros((0, 4)), 4)

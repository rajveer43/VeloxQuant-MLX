"""Round-trip tests for OutlierSplitHandler (#636)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.core.context import QuantizationContext
from veloxquant_mlx.handlers import OutlierSplitHandler


def _inputs():
    arr = np.arange(16, dtype=np.float32).reshape(2, 8)
    return {"mx.array": mx.array(arr), "np.ndarray": arr}


@pytest.mark.parametrize("kind", ["mx.array", "np.ndarray"])
def test_round_trip_restores_full_width(kind):
    x = _inputs()[kind]
    h = OutlierSplitHandler(np.array([1, 5]))
    ctx = QuantizationContext(x_original=x, x_current=x, mode="encode")
    ctx = h.handle(ctx)
    assert ctx.x_current.shape == (2, 6)
    assert ctx.metadata["x_outlier"].shape == (2, 2)

    ctx.mode = "decode"
    ctx = h.handle(ctx)
    np.testing.assert_array_equal(np.array(ctx.x_current), np.arange(16).reshape(2, 8))


def test_outlier_columns_split_correctly():
    x = mx.arange(16, dtype=mx.float32).reshape(2, 8)
    h = OutlierSplitHandler(np.array([1, 5]))
    ctx = h.handle(QuantizationContext(x_original=x, x_current=x, mode="encode"))
    np.testing.assert_array_equal(np.array(ctx.metadata["x_outlier"]), [[1, 5], [9, 13]])
    np.testing.assert_array_equal(np.array(ctx.x_current)[0], [0, 2, 3, 4, 6, 7])


def test_out_of_range_outlier_index_raises():
    x = mx.zeros((2, 8))
    h = OutlierSplitHandler(np.array([1, 9]))
    with pytest.raises(ValueError, match="out of range"):
        h.handle(QuantizationContext(x_original=x, x_current=x, mode="encode"))

"""Tests for weight/quantized_linear.py."""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.weight.quantized_linear import QuantizedLinear


def test_all_zero_weight_row_produces_zero_output():
    """A zero row has no direction; it must not dequantize to arbitrary values."""
    w = mx.random.normal((8, 64))
    w[3] = 0.0
    layer = QuantizedLinear(64, 8, bits=4, bias=False)
    layer.quantize_weights(w)

    y = layer(mx.random.normal((5, 64)))
    mx.eval(y)

    assert float(mx.abs(y[:, 3]).max()) == 0.0
    assert bool(mx.all(mx.isfinite(y)))

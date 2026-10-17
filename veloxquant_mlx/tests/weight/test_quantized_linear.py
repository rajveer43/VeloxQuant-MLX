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


def test_forward_preserves_input_dtype_and_range() -> None:
    """Regression: the forward pass cast everything to fp16, so bf16/fp32
    activations above 65504 became inf and the output dtype was always fp16."""
    import mlx.core as mx

    mx.random.seed(0)
    layer = QuantizedLinear(in_features=64, out_features=32, bits=4, bias=True)
    layer.quantize_weights(mx.random.normal((32, 64)), bias=mx.random.normal((32,)))

    for dtype in (mx.bfloat16, mx.float32, mx.float16):
        x = (mx.random.normal((2, 64)) * (1e5 if dtype != mx.float16 else 1.0)).astype(dtype)
        y = layer(x)
        assert y.dtype == dtype
        assert bool(mx.all(mx.isfinite(y.astype(mx.float32))))

    # Switching dtype back and forth must not return stale-dtype weights.
    x16 = mx.random.normal((2, 64)).astype(mx.float16)
    y16_a = layer(x16)
    layer(mx.random.normal((2, 64)).astype(mx.float32))
    assert mx.allclose(layer(x16), y16_a).item()

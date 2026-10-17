"""Tests for NormalizationHandler."""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.core.context import QuantizationContext
from veloxquant_mlx.handlers import NormalizationHandler


def _roundtrip(x):
    h = NormalizationHandler()
    enc = QuantizationContext(x_original=x, mode="encode", x_current=x)
    h.handle(enc)
    dec = QuantizationContext(x_original=x, mode="decode", x_current=enc.x_current, norm=enc.norm)
    h.handle(dec)
    return enc, dec.x_current


@pytest.mark.parametrize("value", [300.0, 1e-5, 0.0])
def test_fp16_roundtrip_is_finite_and_accurate(value: float) -> None:
    """Regression: fp16 sum-of-squares overflowed (norm>256 -> x/inf = 0) or
    underflowed (tiny/zero -> eps clamp is 0 in fp16 -> 0/0 = NaN)."""
    x = mx.full((2, 16), value, dtype=mx.float16)
    enc, y = _roundtrip(x)
    assert y.dtype == mx.float16
    assert not bool(mx.any(mx.isnan(y)))
    assert float(mx.max(mx.abs(y.astype(mx.float32) - x.astype(mx.float32)))) <= 1e-3 * max(
        value, 1e-3
    )
    if value:
        assert float(mx.max(mx.abs(enc.x_current.astype(mx.float32)))) <= 1.0 + 1e-3

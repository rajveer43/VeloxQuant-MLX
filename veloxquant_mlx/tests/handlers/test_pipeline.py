"""Tests for handler chain assembly and QuantizationContext flow."""

from __future__ import annotations

import numpy as np
import pytest


def _make_codebook(b: int = 2, d: int = 64):
    from veloxquant_mlx.codebooks.base import CodebookFactory

    return CodebookFactory.create("gaussian", b=b, d=d)


def _make_rotation(d: int = 64):
    import mlx.core as mx
    from veloxquant_mlx.math.rotation import make_rotation_matrix
    from veloxquant_mlx.preconditioners.rotation import RotationPreconditioner

    Pi = mx.array(make_rotation_matrix(d, seed=0).astype(np.float16))
    return RotationPreconditioner(Pi)


def test_normalization_handler_encode() -> None:
    import mlx.core as mx
    from veloxquant_mlx.core.context import QuantizationContext
    from veloxquant_mlx.handlers.normalization import NormalizationHandler

    x = mx.array(np.array([[3.0, 4.0]], dtype=np.float16))
    ctx = QuantizationContext(x_original=x, mode="encode", x_current=x)
    handler = NormalizationHandler()
    ctx = handler.handle(ctx)

    assert ctx.norm is not None
    np.testing.assert_allclose(float(ctx.norm[0]), 5.0, atol=0.1)
    # Normalised vector should have unit norm
    norm_after = float(mx.sqrt(mx.sum(ctx.x_current**2)).item())
    np.testing.assert_allclose(norm_after, 1.0, atol=0.05)


def test_rotation_handler_roundtrip() -> None:
    import mlx.core as mx
    from veloxquant_mlx.core.context import QuantizationContext
    from veloxquant_mlx.handlers.rotation_handler import RotationHandler

    d = 64
    x = mx.array(np.random.randn(2, d).astype(np.float16))
    ctx = QuantizationContext(x_original=x, mode="encode", x_current=x)
    rot = _make_rotation(d)
    handler = RotationHandler(rot)
    ctx = handler.handle(ctx)
    # Now decode
    ctx.mode = "decode"
    ctx = handler.handle(ctx)
    # Should recover x up to fp16 precision
    np.testing.assert_allclose(
        np.array(ctx.x_current, dtype=np.float32),
        np.array(x, dtype=np.float32),
        atol=0.05,
    )


def test_scalar_quant_handler_encode_decode() -> None:
    import mlx.core as mx
    from veloxquant_mlx.core.context import QuantizationContext
    from veloxquant_mlx.handlers.scalar_quant_handler import ScalarQuantizerHandler

    d = 64
    cb = _make_codebook(b=2, d=d)
    x = mx.array(np.random.randn(3, d).astype(np.float16))
    ctx = QuantizationContext(x_original=x, mode="encode", x_current=x)
    handler = ScalarQuantizerHandler(cb)
    ctx = handler.handle(ctx)

    assert ctx.indices is not None
    assert ctx.indices.shape == (3, d)


def test_chained_normalization_rotation() -> None:
    """Normalization → Rotation chain should process context correctly."""
    import mlx.core as mx
    from veloxquant_mlx.core.context import QuantizationContext
    from veloxquant_mlx.handlers.normalization import NormalizationHandler
    from veloxquant_mlx.handlers.rotation_handler import RotationHandler

    d = 64
    x = mx.array(np.random.randn(2, d).astype(np.float16))
    ctx = QuantizationContext(x_original=x, mode="encode", x_current=x)

    h1 = NormalizationHandler()
    h2 = RotationHandler(_make_rotation(d))
    h1.set_next(h2)

    ctx = h1.handle(ctx)
    assert ctx.norm is not None
    assert ctx.rotated is not None
    assert ctx.x_current.shape == (2, d)


def test_bit_packing_handler_roundtrip() -> None:
    import mlx.core as mx
    from veloxquant_mlx.core.context import QuantizationContext
    from veloxquant_mlx.handlers.bit_pack_handler import BitPackingHandler

    d, b = 64, 2
    indices = mx.array(np.random.randint(0, 4, (2, d), dtype=np.uint8))
    ctx = QuantizationContext(
        x_original=mx.zeros((2, d)), mode="encode", x_current=mx.zeros((2, d)), indices=indices
    )
    handler = BitPackingHandler(b=b, d=d)
    ctx = handler.handle(ctx)
    assert ctx.packed_bits is not None

    # Decode
    ctx.mode = "decode"
    ctx2 = QuantizationContext(
        x_original=mx.zeros((2, d)),
        mode="decode",
        x_current=mx.zeros((2, d)),
        packed_bits=ctx.packed_bits,
    )
    ctx2 = handler.handle(ctx2)
    np.testing.assert_array_equal(np.array(indices), np.array(ctx2.indices))

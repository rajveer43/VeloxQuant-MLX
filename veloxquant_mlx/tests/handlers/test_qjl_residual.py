"""QJLResidualHandler must take its residual in the space the MSE stage worked in."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.context import QuantizationContext
from veloxquant_mlx.handlers import (
    NormalizationHandler,
    QJLResidualHandler,
    ScalarQuantizerHandler,
)
from veloxquant_mlx.math.rotation import make_jl_matrix
from veloxquant_mlx.preconditioners.jl_sketch import QJLEncoder

D = 64


def _roundtrip(x, with_qjl: bool):
    cb = CodebookFactory.create("gaussian", b=3, d=D)
    encoder = QJLEncoder(mx.array(make_jl_matrix(D, D, seed=0).astype(np.float16)))
    enc_chain = [NormalizationHandler(), ScalarQuantizerHandler(cb)]
    if with_qjl:
        enc_chain.append(QJLResidualHandler(encoder))
    for a, b in zip(enc_chain, enc_chain[1:], strict=False):
        a.set_next(b)
    enc = QuantizationContext(x_original=x, mode="encode", x_current=x)
    enc_chain[0].handle(enc)

    dec = QuantizationContext(
        x_original=x,
        mode="decode",
        x_current=enc.x_current,
        norm=enc.norm,
        indices=enc.indices,
        signs=enc.signs,
        residual_norm=enc.residual_norm,
    )
    ScalarQuantizerHandler(cb).handle(dec)
    if with_qjl:
        QJLResidualHandler(encoder).handle(dec)
    NormalizationHandler().handle(dec)
    return dec.x_current.astype(mx.float32)


def _rel_err(x, y) -> float:
    xs = x.astype(mx.float32)
    return float(mx.sqrt(mx.sum((xs - y) ** 2)) / mx.sqrt(mx.sum(xs**2)))


def test_qjl_after_normalization_does_not_blow_up() -> None:
    """Regression: the residual was x_original - x_current, mixing raw and
    unit-norm spaces; at scale 300 it overflowed to NaN and at scale 1 it
    degraded the reconstruction badly."""
    mx.random.seed(0)
    for scale in (1.0, 300.0):
        x = (mx.random.normal((8, D)) * scale).astype(mx.float32)
        y = _roundtrip(x, with_qjl=True)
        assert not bool(mx.any(mx.isnan(y)))
        # 3-bit gaussian codebook on unit-norm coords: well under 100% error.
        assert _rel_err(x, y) < 0.5


def test_qjl_does_not_worsen_mse_reconstruction() -> None:
    mx.random.seed(1)
    x = mx.random.normal((16, D)).astype(mx.float32)
    base = _rel_err(x, _roundtrip(x, with_qjl=False))
    with_qjl = _rel_err(x, _roundtrip(x, with_qjl=True))
    assert with_qjl <= base * 1.25  # QJL is unbiased-noisy; must not be far worse

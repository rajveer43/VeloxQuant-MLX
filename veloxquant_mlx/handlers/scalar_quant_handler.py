"""Handler stage applying scalar codebook quantization to the current vector.

Delegates to a ``Codebook`` (e.g. a Lloyd-Max or ``VoronoiTree``-backed
scalar codebook) to quantize ``ctx.x_current`` into ``ctx.indices`` on
encode, and dequantize back to a reconstructed ``ctx.x_current`` on decode;
on encode it also eagerly dequantizes so downstream handlers (such as
``QJLResidualHandler``) can compute a residual against the MSE
reconstruction.
"""

from __future__ import annotations

from veloxquant_mlx.core.abstractions import Codebook, QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext


class ScalarQuantizerHandler(QuantizationHandler):
    """Apply scalar codebook quantisation to x_current.

    On encode:
        ctx.indices = codebook.quantize(ctx.x_current)
        ctx.x_current = codebook.dequantize(ctx.indices)   # reconstructed

    On decode:
        ctx.x_current = codebook.dequantize(ctx.indices)

    Args:
        codebook: A Codebook instance to use for quantise/dequantise.
    """

    def __init__(self, codebook: Codebook) -> None:
        self._codebook = codebook

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Quantise and/or dequantise via the codebook.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """
        if ctx.mode == "encode":
            ctx.indices = self._codebook.quantize(ctx.x_current)
            ctx.x_current = self._codebook.dequantize(ctx.indices)
        else:
            if ctx.indices is not None:
                ctx.x_current = self._codebook.dequantize(ctx.indices)

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "scalar_quant"

    def __repr__(self) -> str:
        return f"ScalarQuantizerHandler(codebook={self._codebook!r})"

from __future__ import annotations

from veloxquant_mlx.core.abstractions import QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext
from veloxquant_mlx.preconditioners.jl_sketch import QJLEncoder


class QJLResidualHandler(QuantizationHandler):
    """Apply QJL to the residual between the original and MSE reconstruction.

    On encode:
        residual = ctx.x_original - ctx.x_current  (x_current is MSE reconstruction)
        ctx.signs, ctx.residual_norm = qjl.encode_key(residual)

    On decode:
        x_qjl = residual_norm * scale * signs @ S
        ctx.x_current = ctx.x_current + x_qjl

    The TurboQuantProd identity is:
        x̃ = x̃_mse + ‖r‖ · (√(π/2)/m) · S^T · sign(S·r)

    where r = x - x̃_mse.

    Args:
        encoder: A QJLEncoder wrapping the shared JL matrix S.
    """

    def __init__(self, encoder: QJLEncoder) -> None:
        self._encoder = encoder

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Encode or decode the QJL residual.

        Args:
            ctx: Quantization context. ctx.x_original must be set.

        Returns:
            Updated context.
        """
        import mlx.core as mx

        if ctx.mode == "encode":
            residual = ctx.x_original - ctx.x_current
            signs, r_norm = self._encoder.encode_key(residual)
            ctx.signs = signs
            ctx.residual_norm = r_norm
        else:
            if ctx.signs is not None and ctx.residual_norm is not None:
                # x_qjl = r_norm * scale/m * signs @ S
                scale = self._encoder._SCALE / self._encoder.m
                x_qjl = (
                    ctx.residual_norm[:, None]
                    * scale
                    * (ctx.signs.astype(mx.float16) @ self._encoder._S)
                )
                ctx.x_current = ctx.x_current + x_qjl

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "qjl_residual"

    def __repr__(self) -> str:
        return f"QJLResidualHandler(encoder={self._encoder!r})"

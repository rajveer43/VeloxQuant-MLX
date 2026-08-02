from __future__ import annotations

from veloxquant_mlx.core.abstractions import QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext


class NormalizationHandler(QuantizationHandler):
    """Normalise input vectors to unit L2 norm (encode) or restore scale (decode).

    On encode:
        norm = ‖x‖₂                   stored in ctx.norm
        x_current = x / norm

    On decode:
        x_current = x_current * norm   (restores original scale)

    Args:
        eps: Small value added to norm to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self._eps = eps

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Process normalisation in the context's current mode.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context (norm and x_current modified).
        """
        import mlx.core as mx

        if ctx.mode == "encode":
            norm = mx.sqrt(mx.sum(ctx.x_current * ctx.x_current, axis=-1, keepdims=True))
            norm_clipped = mx.maximum(norm, self._eps)
            ctx.norm = norm_clipped[:, 0]  # (batch,)
            ctx.x_current = ctx.x_current / norm_clipped
        else:
            # decode: restore scale
            if ctx.norm is not None:
                ctx.x_current = ctx.x_current * ctx.norm[:, None]

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "normalization"

    def __repr__(self) -> str:
        return f"NormalizationHandler(eps={self._eps})"

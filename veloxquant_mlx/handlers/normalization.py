"""Handler stage that normalizes vectors to unit L2 norm and restores scale on decode.

Typically the first stage in an encode chain: it factors each input vector
into a stored scalar norm (``ctx.norm``) and a unit-norm direction, so that
downstream stages (rotation, polar transform, scalar quantization) always
operate on scale-normalized data. On decode it multiplies the reconstructed
direction back by the stored norm.
"""

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
            # Squares and the eps clamp in float32: in fp16 the sum of squares
            # overflows once the norm passes 256 (x/inf -> 0) and the 1e-8
            # clamp underflows to 0 (0/0 -> NaN).
            x32 = ctx.x_current.astype(mx.float32)
            norm = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
            norm_clipped = mx.maximum(norm, self._eps)
            ctx.norm = norm_clipped[:, 0]  # (batch,) float32
            ctx.x_current = (x32 / norm_clipped).astype(ctx.x_current.dtype)
        else:
            # decode: restore scale
            if ctx.norm is not None:
                x32 = ctx.x_current.astype(mx.float32) * ctx.norm.astype(mx.float32)[:, None]
                ctx.x_current = x32.astype(ctx.x_current.dtype)

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        """Registry name for this handler ('normalization')."""
        return "normalization"

    def __repr__(self) -> str:
        return f"NormalizationHandler(eps={self._eps})"

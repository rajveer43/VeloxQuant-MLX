from __future__ import annotations

from veloxquant_mlx.core.abstractions import QuantizationHandler, Transform
from veloxquant_mlx.core.context import QuantizationContext, TransformResult


class PolarTransformHandler(QuantizationHandler):
    """Apply or invert the recursive polar transform.

    On encode:
        result = transform.forward(x_current)
        ctx.angles = result.angles
        ctx.final_radius = result.final_radius
        ctx.x_current is set to the concatenation of angles (for downstream handlers)

    On decode:
        result = TransformResult(angles=ctx.angles, final_radius=ctx.final_radius, ...)
        ctx.x_current = transform.inverse(result)

    Args:
        transform: A Transform instance (RecursivePolarTransform).
    """

    def __init__(self, transform: Transform) -> None:
        self._transform = transform

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Apply forward or inverse polar transform.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """

        if ctx.mode == "encode":
            result = self._transform.forward(ctx.x_current)
            ctx.angles = result.angles
            ctx.final_radius = result.final_radius
            # x_current becomes first set of angles for scalar quantiser downstream
            ctx.x_current = result.angles[0] if result.angles else ctx.x_current
            ctx.metadata["transform_result"] = result
        else:
            if ctx.angles is not None and ctx.final_radius is not None:
                n_levels = len(ctx.angles)
                result = TransformResult(
                    angles=ctx.angles,
                    final_radius=ctx.final_radius,
                    n_levels=n_levels,
                )
                ctx.x_current = self._transform.inverse(result)

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "polar_transform"

    def __repr__(self) -> str:
        return f"PolarTransformHandler(transform={self._transform!r})"

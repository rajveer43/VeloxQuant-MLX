"""Handler stage delegating to a ``Preconditioner`` for rotation/JL preconditioning.

Applies a ``Preconditioner`` (e.g. a random rotation or Johnson-Lindenstrauss
sketch) to ``ctx.x_current`` on encode, caching the rotated vector in
``ctx.rotated``, and applies the inverse transform on decode. Runs upstream
of scalar quantization so that quantizer-unfriendly channel structure
(correlated or heavy-tailed dimensions) is spread out before codebook
lookup.
"""

from __future__ import annotations

from veloxquant_mlx.core.abstractions import Preconditioner, QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext


class RotationHandler(QuantizationHandler):
    """Apply a Preconditioner (rotation/JL) to the current vector.

    On encode: ctx.x_current = preconditioner.apply(ctx.x_current)
               ctx.rotated = ctx.x_current  (saved for reference)
    On decode: ctx.x_current = preconditioner.apply_inverse(ctx.x_current)

    Args:
        preconditioner: A Preconditioner instance to delegate to.
    """

    def __init__(self, preconditioner: Preconditioner) -> None:
        self._preconditioner = preconditioner

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Apply or invert the preconditioner.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """
        if ctx.mode == "encode":
            ctx.x_current = self._preconditioner.apply(ctx.x_current)
            ctx.rotated = ctx.x_current
        else:
            ctx.x_current = self._preconditioner.apply_inverse(ctx.x_current)

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "rotation"

    def __repr__(self) -> str:
        return f"RotationHandler(preconditioner={self._preconditioner!r})"

"""Handler stage performing per-token int8 quantization for KV cache value vectors.

Unlike the key-side handlers, which route through rotation/polar/codebook
stages, value vectors use a simple dynamic per-token absmax int8 scheme: on
encode the scale is ``max(|x|) / INT8_MAX`` (clamped away from zero) and
stored in ``ctx.metadata["v_scale"]``; on decode the stored int8 values are
rescaled back to fp16. This mirrors the value-cache quantization used by
KIVI-style caches, where values tolerate coarser, cheaper quantization than
keys.
"""

from __future__ import annotations

from veloxquant_mlx.core.abstractions import QuantizationHandler
from veloxquant_mlx.core.constants import INT8_MAX
from veloxquant_mlx.core.context import QuantizationContext


class ValueQuantizerHandler(QuantizationHandler):
    """Per-token int8 quantisation for value vectors.

    On encode:
        scale = max(|x|) / INT8_MAX
        ctx.metadata["v_scale"] = scale
        ctx.x_current = round(x / scale).clip(-127, 127).astype(int8)

    On decode:
        ctx.x_current = x_current.astype(fp16) * scale

    Args:
        None. Scale is computed dynamically per call.
    """

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Quantise or dequantise value vectors.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """
        import mlx.core as mx

        if ctx.mode == "encode":
            abs_max = mx.max(mx.abs(ctx.x_current), axis=-1, keepdims=True)
            scale = abs_max / INT8_MAX
            scale = mx.maximum(scale, 1e-8)
            ctx.metadata["v_scale"] = scale
            quantised = mx.round(ctx.x_current / scale)
            quantised = mx.clip(quantised, INT8_MAX * -1, INT8_MAX)
            ctx.x_current = quantised.astype(mx.int8)
        else:
            if "v_scale" in ctx.metadata:
                scale = ctx.metadata["v_scale"]
                ctx.x_current = ctx.x_current.astype(mx.float16) * scale

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "value_quant"

    def __repr__(self) -> str:
        return "ValueQuantizerHandler()"

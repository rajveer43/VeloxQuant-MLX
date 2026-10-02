"""Handler stage that separates outlier and inlier channels for mixed-precision quantization.

Splits ``ctx.x_current`` into a small set of outlier channels (indices
supplied at construction, typically from ``OutlierDetector`` /
``SortedChannelIndex``) and the remaining inlier channels on encode, storing
both in ``ctx.metadata`` and routing the inlier portion downstream for
lower-precision quantization; on decode it recombines both portions back
into a full vector. Enables a ``CompositeQuantizer``-style pipeline where
outliers get higher fidelity than the bulk of the channels.
"""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.core.abstractions import QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext


class OutlierSplitHandler(QuantizationHandler):
    """Split input coordinates into outlier and inlier channels.

    On encode:
        Stores ctx.outlier_idx and separate x_outlier/x_inlier (plus the full
        width ``d``) in metadata.
        ctx.x_current is set to the inlier portion.

    On decode:
        Recombines outlier and inlier channels from metadata back into
        ctx.x_current using the stored outlier_idx.

    Args:
        outlier_idx: Array of channel indices to treat as outliers.
    """

    def __init__(self, outlier_idx: np.ndarray) -> None:
        self._outlier_idx = np.asarray(outlier_idx, dtype=np.int32)

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Split or recombine channels.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """
        import mlx.core as mx

        ctx.outlier_idx = self._outlier_idx

        if ctx.mode == "encode":
            x = mx.array(ctx.x_current)  # accept mx.array or np.ndarray
            d = x.shape[-1]
            if self._outlier_idx.size and (
                self._outlier_idx.min() < 0 or self._outlier_idx.max() >= d
            ):
                raise ValueError(
                    f"OutlierSplitHandler: outlier_idx out of range for width {d}: "
                    f"{self._outlier_idx.tolist()}"
                )
            inlier_idx = np.setdiff1d(np.arange(d), self._outlier_idx)
            # MLX cannot index with a numpy int array; take with an mx index.
            ctx.metadata["x_outlier"] = mx.take(x, mx.array(self._outlier_idx), axis=-1)
            ctx.metadata["x_inlier"] = mx.take(x, mx.array(inlier_idx), axis=-1)
            ctx.metadata["inlier_idx"] = inlier_idx
            # The full width must be remembered: on decode ``x_current`` is
            # only the inlier slice, so its width is not the original d (#636).
            ctx.metadata["d"] = d
            ctx.x_current = ctx.metadata["x_inlier"]
        else:
            # decode: recombine
            if "x_outlier" in ctx.metadata and "inlier_idx" in ctx.metadata:
                inlier_idx = ctx.metadata["inlier_idx"]
                x_out = ctx.metadata["x_outlier"]
                x_inlier = ctx.x_current

                d = ctx.metadata.get("d", len(inlier_idx) + len(self._outlier_idx))
                batch = x_inlier.shape[0]
                result = np.zeros((batch, d), dtype=np.float32)
                result[:, inlier_idx] = np.array(x_inlier)
                result[:, self._outlier_idx] = np.array(x_out)
                ctx.x_current = mx.array(result).astype(x_inlier.dtype)

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        """Registry name for this handler ('outlier_split')."""
        return "outlier_split"

    def __repr__(self) -> str:
        return f"OutlierSplitHandler(n_outliers={len(self._outlier_idx)})"

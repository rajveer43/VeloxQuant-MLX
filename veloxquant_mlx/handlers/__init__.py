"""Chain-of-Responsibility pipeline stages composing quantizer encode/decode paths.

Each handler subclasses ``QuantizationHandler`` (see
``veloxquant_mlx.core.abstractions``) and implements ``handle()`` to mutate a
shared ``QuantizationContext`` in place before forwarding it to the next
stage, so a quantizer is assembled by chaining handlers (normalization,
rotation, outlier splitting, polar transform, scalar/value quantization, QJL
residual, bit packing) rather than hardcoding one monolithic encode/decode
function. Re-exports every concrete handler: ``BitPackingHandler``,
``NormalizationHandler``, ``OutlierSplitHandler``, ``PolarTransformHandler``,
``QJLResidualHandler``, ``RotationHandler``, ``ScalarQuantizerHandler``, and
``ValueQuantizerHandler``.
"""

from __future__ import annotations

from veloxquant_mlx.handlers.bit_pack_handler import BitPackingHandler
from veloxquant_mlx.handlers.normalization import NormalizationHandler
from veloxquant_mlx.handlers.outlier_split import OutlierSplitHandler
from veloxquant_mlx.handlers.polar_handler import PolarTransformHandler
from veloxquant_mlx.handlers.qjl_residual_handler import QJLResidualHandler
from veloxquant_mlx.handlers.rotation_handler import RotationHandler
from veloxquant_mlx.handlers.scalar_quant_handler import ScalarQuantizerHandler
from veloxquant_mlx.handlers.value_quant_handler import ValueQuantizerHandler

__all__ = [
    "BitPackingHandler",
    "NormalizationHandler",
    "OutlierSplitHandler",
    "PolarTransformHandler",
    "QJLResidualHandler",
    "RotationHandler",
    "ScalarQuantizerHandler",
    "ValueQuantizerHandler",
]

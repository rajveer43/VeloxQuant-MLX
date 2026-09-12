"""Whole-model weight compression, built on the TurboQuant quantizer family.

Groups ``quantize_model``/``compression_report`` for replacing an
``mlx.nn`` model's Linear layers with ``QuantizedLinear`` (TurboQuant-style
rotate + Lloyd-Max weight compression), and
``save_reservoir``/``load_reservoir``/``graft_reservoir`` for serializing
and restoring a quantized model's compressed weights to/from a single flat
file.
"""

from __future__ import annotations

from veloxquant_mlx.weight.model_quantizer import compression_report, quantize_model
from veloxquant_mlx.weight.quantized_linear import QuantizedLinear
from veloxquant_mlx.weight.reservoir import graft_reservoir, load_reservoir, save_reservoir

__all__ = [
    "QuantizedLinear",
    "compression_report",
    "quantize_model",
    "save_reservoir",
    "load_reservoir",
    "graft_reservoir",
]

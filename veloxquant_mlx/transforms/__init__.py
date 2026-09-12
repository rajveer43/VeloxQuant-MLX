"""Vector-space transforms used ahead of quantization.

Currently re-exports ``RecursivePolarTransform``, the recursive
angle/radius decomposition that ``PolarQuantizer`` (see
``quantizers/polarquant.py``) applies before per-level codebook lookup.
"""

from __future__ import annotations

from veloxquant_mlx.transforms.polar import RecursivePolarTransform

__all__ = ["RecursivePolarTransform"]

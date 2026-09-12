"""Outlier-channel detection for mixed-precision KV-cache quantization.

Groups the calibration-time detector used to identify high-magnitude key
channels that warrant higher-precision treatment. Re-exports
``OutlierDetector``.
"""

from __future__ import annotations

from veloxquant_mlx.outlier.detector import OutlierDetector

__all__ = ["OutlierDetector"]

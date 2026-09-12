"""Observer hooks for instrumenting the quantization pipeline at runtime.

Each observer subscribes to :class:`~veloxquant_mlx.observers.base.QuantizationEvent`
emitted at pipeline checkpoints and accumulates a specific diagnostic:
reconstruction distortion vs. TurboQuant's theoretical bounds
(``DistortionObserver``), per-channel key-norm statistics
(``KeyNormObserver``), per-stage timing (``LatencyObserver``), and
per-stage RSS memory deltas (``MemoryObserver``). Re-exports
``QuantizationEvent``, ``DistortionObserver``, ``DistortionReport``,
``KeyNormObserver``, ``KeyNormReport``, ``LatencyObserver``, and
``MemoryObserver``.
"""

from __future__ import annotations

from veloxquant_mlx.observers.base import QuantizationEvent
from veloxquant_mlx.observers.distortion import DistortionObserver, DistortionReport
from veloxquant_mlx.observers.key_norm import KeyNormObserver, KeyNormReport
from veloxquant_mlx.observers.latency import LatencyObserver
from veloxquant_mlx.observers.memory import MemoryObserver

__all__ = [
    "QuantizationEvent",
    "DistortionObserver",
    "DistortionReport",
    "KeyNormObserver",
    "KeyNormReport",
    "LatencyObserver",
    "MemoryObserver",
]

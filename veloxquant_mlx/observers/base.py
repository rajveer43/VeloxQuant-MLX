"""Shared event payload emitted by the quantization pipeline for observer consumption.

Defines :class:`QuantizationEvent`, the single data structure every
observer in :mod:`veloxquant_mlx.observers` (distortion, latency, memory,
key-norm) consumes via its ``on_event`` hook — carrying the stage name,
input shape, elapsed time, memory delta, and stage-specific metadata for
one pipeline checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantizationEvent:
    """Data emitted by the quantization pipeline at key checkpoints.

    Attributes:
        stage: Handler name that emitted this event.
        input_shape: Shape of the input tensor at this stage.
        elapsed_ms: Wall-clock time for this stage in milliseconds.
        memory_delta_bytes: Change in process RSS during this stage.
        metadata: Stage-specific extra data.
    """

    stage: str
    input_shape: tuple
    elapsed_ms: float = 0.0
    memory_delta_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"QuantizationEvent(stage={self.stage!r}, "
            f"shape={self.input_shape}, "
            f"elapsed_ms={self.elapsed_ms:.3f})"
        )

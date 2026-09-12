from __future__ import annotations

from veloxquant_mlx.core.abstractions import QuantizationObserver
from veloxquant_mlx.observers.base import QuantizationEvent


class MemoryObserver(QuantizationObserver):
    """Tracks per-stage RSS memory changes using the event's memory_delta_bytes.

    For accurate measurement, the pipeline emitter should use psutil to
    measure RSS before and after each stage and populate memory_delta_bytes.

    Args:
        None.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, list[int]] = {}

    def on_event(self, event: QuantizationEvent) -> None:
        """Record the memory delta for this stage.

        Args:
            event: Pipeline event with memory_delta_bytes.
        """
        if event.stage not in self._deltas:
            self._deltas[event.stage] = []
        self._deltas[event.stage].append(event.memory_delta_bytes)

    def peak_delta_bytes(self) -> int:
        """Return the largest single memory delta observed across all stages."""
        all_deltas = [d for deltas in self._deltas.values() for d in deltas]
        return max(all_deltas, default=0)

    def report(self) -> dict[str, int]:
        """Return the sum of memory deltas per stage.

        Returns:
            Dict mapping stage name to total accumulated memory delta in bytes.
        """
        return {stage: sum(deltas) for stage, deltas in self._deltas.items()}

    def __repr__(self) -> str:
        return f"MemoryObserver(stages={list(self._deltas.keys())})"

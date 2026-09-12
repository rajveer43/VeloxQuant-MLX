"""Observer that records per-pipeline-stage latency histograms.

Provides :class:`LatencyObserver`, which collects every
``elapsed_ms`` sample emitted per stage name in
:class:`~veloxquant_mlx.observers.base.QuantizationEvent` and reports
mean/min/max/count summaries — useful for finding which quantization
pipeline stage (preconditioning, encoding, codebook lookup, etc.)
dominates wall-clock time.
"""

from __future__ import annotations

from collections import defaultdict

from veloxquant_mlx.core.abstractions import QuantizationObserver
from veloxquant_mlx.observers.base import QuantizationEvent


class LatencyObserver(QuantizationObserver):
    """Records per-stage timing histograms.

    Tracks all latency samples per stage in milliseconds.
    Call report() to get a summary dict.

    Args:
        None.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def on_event(self, event: QuantizationEvent) -> None:
        """Record the elapsed time for the given stage.

        Args:
            event: Pipeline event with stage name and elapsed_ms.
        """
        self._samples[event.stage].append(event.elapsed_ms)

    def report(self) -> dict[str, dict[str, float]]:
        """Return summary statistics per stage.

        Returns:
            Dict mapping stage name to {mean_ms, min_ms, max_ms, count}.
        """
        import statistics

        result = {}
        for stage, samples in self._samples.items():
            result[stage] = {
                "mean_ms": statistics.mean(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "count": len(samples),
            }
        return result

    def reset(self) -> None:
        """Clear all accumulated samples."""
        self._samples.clear()

    def __repr__(self) -> str:
        stages = list(self._samples.keys())
        return f"LatencyObserver(stages={stages})"

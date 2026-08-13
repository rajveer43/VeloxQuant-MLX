"""KeyNormObserver — records per-token key L2 norm statistics.

Complements :func:`veloxquant_mlx.allocators.calibrate_layer_sensitivities`
by providing an event-driven hook for pipelines that already use the
:class:`QuantizationObserver` framework. The calibration helper in
``allocators.ratequant`` is the direct path for most users; this observer
exists for pipelines that want to fold sensitivity tracking into an
existing observer stack alongside :class:`DistortionObserver` and
:class:`LatencyObserver`.

The observer expects events with ``metadata['key_l2_norm_sq']`` populated
(emitted by the cache during ``update_and_fetch``). When this metadata is
absent, the event is silently ignored — making the observer safe to
attach to mixed pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from veloxquant_mlx.core.abstractions import QuantizationObserver
from veloxquant_mlx.observers.base import QuantizationEvent


@dataclass
class KeyNormReport:
    """Summary statistics produced by :class:`KeyNormObserver`."""

    n_tokens: int
    mean_norm_sq: float
    min_norm_sq: float
    max_norm_sq: float

    @property
    def heterogeneity_ratio(self) -> float:
        """``max / min`` — predictor of when mixed-precision allocation helps.

        Per RateQuant Theorem 3, when this ratio (geometric variant of
        AM/GM) is far above 1, per-layer bit allocation produces measurable
        gains over uniform quantization. Below ~2.0, savings are marginal.
        """
        return self.max_norm_sq / max(self.min_norm_sq, 1e-12)


class KeyNormObserver(QuantizationObserver):
    """Accumulates per-token key L2 norm² for later RateQuant allocation.

    Usage::

        obs = KeyNormObserver()
        config.observers.append(obs)
        # ... run calibration prompts ...
        report = obs.report()
        # report.heterogeneity_ratio indicates RateQuant benefit
    """

    def __init__(self) -> None:
        self._sum_sq = 0.0
        self._min_sq = float("inf")
        self._max_sq = 0.0
        self._n_tokens = 0

    def on_event(self, event: QuantizationEvent) -> None:
        norm_sq = event.metadata.get("key_l2_norm_sq")
        if norm_sq is None:
            return
        # Accept either scalar or iterable-of-scalars
        try:
            iter(norm_sq)
            values = list(norm_sq)
        except TypeError:
            values = [float(norm_sq)]
        for v in values:
            v = float(v)
            self._sum_sq += v
            self._n_tokens += 1
            if v < self._min_sq:
                self._min_sq = v
            if v > self._max_sq:
                self._max_sq = v

    def report(self) -> KeyNormReport:
        if self._n_tokens == 0:
            return KeyNormReport(0, 0.0, 0.0, 0.0)
        return KeyNormReport(
            n_tokens=self._n_tokens,
            mean_norm_sq=self._sum_sq / self._n_tokens,
            min_norm_sq=self._min_sq,
            max_norm_sq=self._max_sq,
        )

    def reset(self) -> None:
        self._sum_sq = 0.0
        self._min_sq = float("inf")
        self._max_sq = 0.0
        self._n_tokens = 0

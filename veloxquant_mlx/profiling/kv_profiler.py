"""Transparent profiling wrapper around any :class:`KVCache`.

Wraps append_key / append_value / attend and records per-call latency and
memory footprint without touching the wrapped cache's implementation, so it
works uniformly across all registered KV-cache methods (see
``KVCacheConfig.method``). One :class:`KVCacheProfiler` instance profiles a
single layer; :func:`format_profile_table` renders a multi-layer summary in
the style requested by issue #252.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from veloxquant_mlx.core.abstractions import KVCache


@dataclass
class LayerProfile:
    """Aggregated profiling stats for a single KV-cache layer.

    Attributes:
        layer_id: Index or label identifying the layer.
        n_quantize_calls: Number of append_key calls (quantization events).
        n_dequantize_calls: Number of attend calls (dequantization events).
        quantize_ms_total: Cumulative append_key wall time, milliseconds.
        dequantize_ms_total: Cumulative attend wall time, milliseconds.
        write_ms_total: Cumulative append_value wall time, milliseconds.
        peak_memory_bytes: Largest memory_bytes() observed after any call.
        tokens_written: Number of append_key calls (proxy for tokens stored).
        fp16_baseline_bytes: What tokens_written * 2 * head_dim would cost in fp16,
            used to derive compression_ratio.
    """

    layer_id: Any
    n_quantize_calls: int = 0
    n_dequantize_calls: int = 0
    quantize_ms_total: float = 0.0
    dequantize_ms_total: float = 0.0
    write_ms_total: float = 0.0
    peak_memory_bytes: int = 0
    tokens_written: int = 0
    fp16_baseline_bytes: int = 0

    @property
    def quantize_ms_mean(self) -> float:
        return self.quantize_ms_total / self.n_quantize_calls if self.n_quantize_calls else 0.0

    @property
    def dequantize_ms_mean(self) -> float:
        return (
            self.dequantize_ms_total / self.n_dequantize_calls if self.n_dequantize_calls else 0.0
        )

    @property
    def compression_ratio(self) -> float:
        """fp16 baseline bytes divided by actual bytes (>1 means compressed)."""
        if self.peak_memory_bytes <= 0:
            return 0.0
        return self.fp16_baseline_bytes / self.peak_memory_bytes


@dataclass
class ProfileReport:
    """Profiling results across one or more layers.

    Attributes:
        layers: Per-layer profiles, in layer order.
        elapsed_s: Total wall time covered by the profiling session.
    """

    layers: list[LayerProfile] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def total_bytes_written(self) -> int:
        return sum(layer.peak_memory_bytes for layer in self.layers)

    @property
    def total_tokens(self) -> int:
        return sum(layer.tokens_written for layer in self.layers)

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.total_tokens / self.elapsed_s

    @property
    def overall_compression_ratio(self) -> float:
        fp16_total = sum(layer.fp16_baseline_bytes for layer in self.layers)
        actual_total = self.total_bytes_written
        if actual_total <= 0:
            return 0.0
        return fp16_total / actual_total


class KVCacheProfiler(KVCache):
    """Wraps a :class:`KVCache` instance, recording per-call timing and memory.

    Transparent decorator: implements the same append_key/append_value/attend/
    memory_bytes interface as the wrapped cache, so it can be substituted
    anywhere a KVCache is expected (including inside a list built by
    ``KVCacheBuilder.for_model()``).

    Args:
        cache: The KVCache instance to profile.
        head_dim: Head dimension, used to compute the fp16 compression
            baseline (2 bytes/element). Defaults to reading ``cache._d`` if
            present, else 0 (disables compression_ratio).
        layer_id: Optional label for this layer, used in reports.

    Example::

        cache = KVCacheFactory.create(config)
        profiled = KVCacheProfiler(cache, head_dim=config.head_dim, layer_id=0)
        profiled.append_key(k)
        profiled.append_value(v)
        out = profiled.attend(q)
        print(format_profile_table([profiled.profile()]))
    """

    def __init__(
        self,
        cache: KVCache,
        head_dim: int | None = None,
        layer_id: Any = 0,
    ) -> None:
        self._cache = cache
        self._head_dim = head_dim if head_dim is not None else getattr(cache, "_d", 0)
        self._profile = LayerProfile(layer_id=layer_id)

    def append_key(self, k: Any) -> None:
        t0 = time.perf_counter()
        self._cache.append_key(k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._profile.n_quantize_calls += 1
        self._profile.quantize_ms_total += elapsed_ms
        self._profile.tokens_written += 1
        self._profile.fp16_baseline_bytes += 2 * self._head_dim
        self._update_peak_memory()

    def append_value(self, v: Any) -> None:
        t0 = time.perf_counter()
        self._cache.append_value(v)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._profile.write_ms_total += elapsed_ms
        self._update_peak_memory()

    def attend(self, q: Any) -> Any:
        t0 = time.perf_counter()
        out = self._cache.attend(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._profile.n_dequantize_calls += 1
        self._profile.dequantize_ms_total += elapsed_ms
        return out

    def memory_bytes(self) -> int:
        return self._cache.memory_bytes()

    def _update_peak_memory(self) -> None:
        current = self._cache.memory_bytes()
        if current > self._profile.peak_memory_bytes:
            self._profile.peak_memory_bytes = current

    def profile(self) -> LayerProfile:
        """Return the accumulated LayerProfile for this wrapped cache."""
        return self._profile

    def reset(self) -> None:
        """Clear all accumulated profiling stats (does not affect the wrapped cache)."""
        self._profile = LayerProfile(layer_id=self._profile.layer_id)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"KVCacheProfiler({self._cache!r})"

    def __getattr__(self, name: str) -> Any:
        # Forward anything not explicitly overridden (method-specific extras
        # like fused_sdpa()) straight to the wrapped cache.
        return getattr(self._cache, name)


def profile_layers(profilers: list[KVCacheProfiler], elapsed_s: float = 0.0) -> ProfileReport:
    """Aggregate a list of KVCacheProfiler instances into one ProfileReport.

    Args:
        profilers: One profiler per model layer, in layer order.
        elapsed_s: Total wall-clock time of the profiled run, for tokens/sec.

    Returns:
        A ProfileReport summarizing all layers.
    """
    return ProfileReport(layers=[p.profile() for p in profilers], elapsed_s=elapsed_s)


def format_profile_table(report: ProfileReport) -> str:
    """Render a ProfileReport as a human-readable fixed-width table.

    Matches the format proposed in issue #252::

        Layer       Quantize   Dequantize   Memory
        ------------------------------------------------
        Layer 0     12.3 µs    8.1 µs       1.2 MB
        Layer 1     11.8 µs    7.9 µs       1.2 MB

    Args:
        report: A ProfileReport (or a bare list[LayerProfile]).

    Returns:
        Multi-line formatted string.
    """
    layers = report.layers if isinstance(report, ProfileReport) else list(report)

    header = f"{'Layer':<12}{'Quantize':<12}{'Dequantize':<13}{'Memory':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for layer in layers:
        q_us = layer.quantize_ms_mean * 1000.0
        dq_us = layer.dequantize_ms_mean * 1000.0
        mem_mb = layer.peak_memory_bytes / (1024 * 1024)
        lines.append(
            f"{'Layer ' + str(layer.layer_id):<12}"
            f"{f'{q_us:.1f} µs':<12}"
            f"{f'{dq_us:.1f} µs':<13}"
            f"{f'{mem_mb:.2f} MB':<10}"
        )

    if isinstance(report, ProfileReport) and layers:
        lines.append(sep)
        lines.append(f"Total tokens:        {report.total_tokens}")
        lines.append(f"Total memory:        {report.total_bytes_written / (1024 * 1024):.2f} MB")
        lines.append(f"Compression ratio:   {report.overall_compression_ratio:.2f}x")
        if report.elapsed_s > 0:
            lines.append(f"Tokens/sec:          {report.tokens_per_sec:.1f}")

    return "\n".join(lines)

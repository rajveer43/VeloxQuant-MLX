"""KV-cache workload replay benchmark (issue #258).

Micro-benchmarks on a single quantization/dequantization kernel can look
fast while hiding overhead that only shows up during real generation:
per-request allocation, prefill-vs-decode cost splits, and the effect of
cache growth or eviction over a long-running stream. This module replays a
small library of standardized synthetic workloads against any registered
KV-cache method (via :class:`~veloxquant_mlx.cache.base.KVCacheBuilder`) and
reports the end-to-end metrics that a serving system actually cares about:

  * TTFT (time to first token, i.e. prefill + first attend)
  * decode latency (mean / p50 / p95 per generated token)
  * tokens/sec
  * peak memory
  * compression ratio vs. an fp16 baseline
  * quantization overhead (mean append_key cost)
  * dequantization overhead (mean attend cost)

Workloads cover: single-request generation, long-context generation, batch
generation, variable sequence lengths, repeated requests, cache growth, and
cache eviction/reuse (see ``STANDARD_WORKLOADS``).

Fidelity note: this codebase's KVCache ABC (append_key/append_value/attend)
operates on one token vector at a time per stream. "Batch" and multi-stream
workloads here are replayed as independent cache instances stepped
sequentially in this process — a proxy for the memory/allocation behaviour
of concurrency, not a fused batched Metal kernel. Read tokens/sec and wall
time accordingly; peak memory, compression ratio, and per-token overheads
are unaffected by this simplification since they are summed/averaged across
streams rather than timed as a single fused call.

Usage::

    python -m veloxquant_mlx.benchmarks.workload_replay_benchmark
    python -m veloxquant_mlx.benchmarks.workload_replay_benchmark \\
        --methods turboquant_mse kivi --workloads single_request cache_growth
    python -m veloxquant_mlx.benchmarks.workload_replay_benchmark \\
        --json-out figures/workload_replay/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parents[2] / "figures" / "workload_replay"

DEFAULT_HEAD_DIM = 128
DEFAULT_BITS = 4
DEFAULT_SEED = 42
DEFAULT_METHODS = ["turboquant_prod", "turboquant_mse", "turboquant_rvq", "kivi", "qjl"]


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


@dataclass
class WorkloadScenario:
    """Declarative description of one synthetic serving pattern to replay.

    Attributes:
        name: Short identifier, used as the results-table row key.
        description: One-line human-readable summary.
        prompt_lens: Prefill length(s), in tokens, for the request(s) in this
            workload. Each length is replayed on every stream.
        n_new_tokens: Decode steps (generated tokens) per request.
        n_streams: Number of independent concurrent cache instances to
            replay (proxy for batch size — see module fidelity note).
        repeat: Number of times to replay the whole ``prompt_lens`` list.
        reuse_cache: If True, cache instances persist and keep growing
            across repeats/requests (models cache growth and reuse). If
            False, a fresh cache is built for every repeat (models
            independent/cold-start requests).
        sliding_window: If set, caches are built with this eviction window
            (see :meth:`KVCacheBuilder.with_sliding_window`), so sustained
            growth eventually forces token eviction.
    """

    name: str
    description: str
    prompt_lens: list[int]
    n_new_tokens: int
    n_streams: int = 1
    repeat: int = 1
    reuse_cache: bool = False
    sliding_window: int | None = None


STANDARD_WORKLOADS: dict[str, WorkloadScenario] = {
    "single_request": WorkloadScenario(
        name="single_request",
        description="One request: 128-token prompt, 64 decode steps.",
        prompt_lens=[128],
        n_new_tokens=64,
    ),
    "long_context": WorkloadScenario(
        name="long_context",
        description="One request with a 4096-token prompt, 32 decode steps.",
        prompt_lens=[4096],
        n_new_tokens=32,
    ),
    "batch_generation": WorkloadScenario(
        name="batch_generation",
        description="8 concurrent 256-token requests, 32 decode steps each.",
        prompt_lens=[256],
        n_new_tokens=32,
        n_streams=8,
    ),
    "variable_seq_lengths": WorkloadScenario(
        name="variable_seq_lengths",
        description="Requests of increasing length: 64/256/1024/4096 tokens.",
        prompt_lens=[64, 256, 1024, 4096],
        n_new_tokens=16,
    ),
    "repeated_requests": WorkloadScenario(
        name="repeated_requests",
        description="The same 128-token request, 10 independent cold starts.",
        prompt_lens=[128],
        n_new_tokens=16,
        repeat=10,
    ),
    "cache_growth": WorkloadScenario(
        name="cache_growth",
        description="One stream growing monotonically to ~4K tokens (8 checkpoints).",
        prompt_lens=[1],
        n_new_tokens=512,
        repeat=8,
        reuse_cache=True,
    ),
    # Known limitation (issue #274): SlidingWindowKVCache's eviction reset
    # never actually fires for any currently registered cache method, so the
    # peak_memory_bytes/compression_ratio reported below do not reflect real
    # eviction — the wrapped cache keeps growing unbounded under the hood
    # even though the wrapper's own token count looks capped. Only
    # memory_snapshots[i].n_tokens (the wrapper's own count) is reliable for
    # this workload until #274 is fixed.
    "cache_eviction_reuse": WorkloadScenario(
        name="cache_eviction_reuse",
        description="One stream growing past a 512-token eviction window (8 checkpoints).",
        prompt_lens=[1],
        n_new_tokens=256,
        repeat=8,
        reuse_cache=True,
        sliding_window=512,
    ),
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """Cache size at one checkpoint of a ``reuse_cache`` workload."""

    n_tokens: int
    memory_bytes: int


@dataclass
class WorkloadResult:
    """Aggregated metrics for one (method, workload) replay."""

    method: str
    workload: str
    n_streams: int
    total_tokens: int
    wall_time_s: float
    ttft_ms_mean: float
    decode_ms_mean: float
    decode_ms_p50: float
    decode_ms_p95: float
    tokens_per_sec: float
    peak_memory_bytes: int
    compression_ratio: float
    quantize_overhead_ms: float
    dequantize_overhead_ms: float
    memory_snapshots: list[MemorySnapshot] = field(default_factory=list)


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (no numpy dependency required)."""
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def run_workload(
    method: str,
    workload: WorkloadScenario,
    head_dim: int = DEFAULT_HEAD_DIM,
    bits: int = DEFAULT_BITS,
    jl_dim: int | None = None,
    seed: int = DEFAULT_SEED,
) -> WorkloadResult:
    """Replay one workload scenario against one KV-cache method.

    Args:
        method: A method name accepted by ``KVCacheConfig.method``.
        workload: The scenario to replay.
        head_dim: Attention head dimension (must be a power of 2).
        bits: Inlier bit-width passed to the cache builder.
        jl_dim: JL projection dimension (defaults to ``head_dim``).
        seed: Random seed for both cache construction and synthetic K/V data.

    Returns:
        A WorkloadResult with the standard replay metrics.
    """
    import numpy as np
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.profiling import KVCacheProfiler

    rng = np.random.default_rng(seed)
    jl_dim = jl_dim if jl_dim is not None else head_dim

    def _vec() -> Any:
        return mx.array(rng.standard_normal(head_dim).astype(np.float16))

    def _build_stream() -> KVCacheProfiler:
        builder = (
            KVCacheBuilder()
            .with_method(method)
            .with_head_dim(head_dim)
            .with_bit_width(inlier=bits)
            .with_jl_dim(jl_dim)
            .with_seed(seed)
        )
        if workload.sliding_window is not None:
            builder = builder.with_sliding_window(workload.sliding_window)
        return KVCacheProfiler(builder.build(), head_dim=head_dim, layer_id=0)

    streams: list[KVCacheProfiler] = []
    if workload.reuse_cache:
        streams = [_build_stream() for _ in range(workload.n_streams)]

    ttft_samples: list[float] = []
    decode_samples: list[float] = []
    snapshots: list[MemorySnapshot] = []
    total_tokens = 0

    t_start = time.perf_counter()
    for _ in range(workload.repeat):
        if not workload.reuse_cache:
            streams = [_build_stream() for _ in range(workload.n_streams)]

        for prompt_len in workload.prompt_lens:
            for stream in streams:
                t0 = time.perf_counter()
                for _ in range(prompt_len):
                    stream.append(_vec(), _vec())
                mx.eval(stream.attend(_vec()))
                ttft_samples.append((time.perf_counter() - t0) * 1000.0)
                total_tokens += prompt_len

                for _ in range(workload.n_new_tokens):
                    t1 = time.perf_counter()
                    stream.append(_vec(), _vec())
                    mx.eval(stream.attend(_vec()))
                    decode_samples.append((time.perf_counter() - t1) * 1000.0)
                    total_tokens += 1

        if workload.reuse_cache:
            snapshots.append(
                MemorySnapshot(
                    n_tokens=sum(len(s) for s in streams),
                    memory_bytes=sum(s.memory_bytes() for s in streams),
                )
            )

    wall_time_s = time.perf_counter() - t_start

    profiles = [s.profile() for s in streams]
    peak_memory = sum(p.peak_memory_bytes for p in profiles)
    fp16_total = sum(p.fp16_baseline_bytes for p in profiles)
    compression_ratio = fp16_total / peak_memory if peak_memory > 0 else 0.0
    quantize_ms = statistics.fmean(p.quantize_ms_mean for p in profiles) if profiles else 0.0
    dequantize_ms = statistics.fmean(p.dequantize_ms_mean for p in profiles) if profiles else 0.0

    return WorkloadResult(
        method=method,
        workload=workload.name,
        n_streams=workload.n_streams,
        total_tokens=total_tokens,
        wall_time_s=wall_time_s,
        ttft_ms_mean=statistics.fmean(ttft_samples) if ttft_samples else 0.0,
        decode_ms_mean=statistics.fmean(decode_samples) if decode_samples else 0.0,
        decode_ms_p50=_percentile(decode_samples, 50),
        decode_ms_p95=_percentile(decode_samples, 95),
        tokens_per_sec=total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        peak_memory_bytes=peak_memory,
        compression_ratio=compression_ratio,
        quantize_overhead_ms=quantize_ms,
        dequantize_overhead_ms=dequantize_ms,
        memory_snapshots=snapshots,
    )


def run_suite(
    methods: list[str],
    workloads: dict[str, WorkloadScenario] | None = None,
    head_dim: int = DEFAULT_HEAD_DIM,
    bits: int = DEFAULT_BITS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, WorkloadResult]]:
    """Replay every workload against every method.

    Returns:
        Nested dict: ``results[method][workload_name] -> WorkloadResult``.
    """
    workloads = STANDARD_WORKLOADS if workloads is None else workloads
    results: dict[str, dict[str, WorkloadResult]] = {}
    for method in methods:
        results[method] = {}
        for name, workload in workloads.items():
            print(f"[{method}] {name}: {workload.description}")
            results[method][name] = run_workload(
                method, workload, head_dim=head_dim, bits=bits, seed=seed
            )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_summary_table(results: dict[str, dict[str, WorkloadResult]]) -> str:
    """Render replay results as a fixed-width table, one row per (method, workload)."""
    header = (
        f"{'Method':<18}{'Workload':<22}{'TTFT(ms)':>10}{'Decode(ms)':>12}"
        f"{'p95(ms)':>10}{'Tok/s':>10}{'PeakMB':>10}{'Compr.':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for method, per_workload in results.items():
        for name, r in per_workload.items():
            lines.append(
                f"{method:<18}{name:<22}{r.ttft_ms_mean:>10.2f}{r.decode_ms_mean:>12.3f}"
                f"{r.decode_ms_p95:>10.3f}{r.tokens_per_sec:>10.1f}"
                f"{r.peak_memory_bytes / (1024 * 1024):>10.2f}{r.compression_ratio:>7.2f}x"
            )
    return "\n".join(lines)


def results_to_json(results: dict[str, dict[str, WorkloadResult]]) -> dict[str, Any]:
    """Convert a run_suite() result into a plain JSON-serializable dict."""
    return {
        method: {name: asdict(r) for name, r in per_workload.items()}
        for method, per_workload in results.items()
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-cache workload replay benchmark")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=None,
        choices=list(STANDARD_WORKLOADS),
        help="Subset of STANDARD_WORKLOADS to run (default: all).",
    )
    parser.add_argument("--head_dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Path to write full results as JSON (default: don't save).",
    )
    args = parser.parse_args()

    workloads = (
        STANDARD_WORKLOADS
        if not args.workloads
        else {name: STANDARD_WORKLOADS[name] for name in args.workloads}
    )

    results = run_suite(
        args.methods, workloads, head_dim=args.head_dim, bits=args.bits, seed=args.seed
    )

    print()
    print(format_summary_table(results))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

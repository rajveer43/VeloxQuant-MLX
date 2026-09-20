"""CLI: analytical per-method KV-cache memory estimates for a model+workload."""

from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.planning import WorkloadProfile
from veloxquant_mlx.profiling.model_profiler import profile_model_from_config


def _fmt_bytes(n: int) -> str:
    for factor, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= factor:
            return f"{n / factor:.2f} {unit}"
    return f"{n} B"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant estimate-memory",
        description=(
            "Estimate the analytic KV-cache footprint of every strategy for a "
            "model + workload, before any weights are loaded. For written "
            "numbers use the benchmark database (--benchmark-dir)."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to a HF-style config.json (needs num_hidden_layers, attention heads, head_dim)",
    )
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-query-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096, help="Context tokens")
    parser.add_argument("--generation", type=int, default=512, help="Generation tokens")
    parser.add_argument("--batch", type=int, default=1, help="Attention batch")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Benchmark-database dir; measured records override analytic estimates",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the N smallest-footprint methods (default 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    args = parser.parse_args(argv)

    if args.model_config:
        import json as _json
        from pathlib import Path

        config = _json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    else:
        config = None

    model = profile_model_from_config(
        config,
        num_layers=args.n_layers,
        num_query_heads=args.n_query_heads or args.n_kv_heads,
        num_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )
    workload = WorkloadProfile(
        context_length=args.context,
        generation_length=args.generation,
        batch_size=args.batch,
    )

    from veloxquant_mlx.planning import AutoOptimizer

    optimizer = AutoOptimizer()
    estimates = optimizer.estimate_memory(workload, model=model)

    empirical: dict[str, object] = {}
    if args.benchmark_dir:
        from veloxquant_mlx.benchmarks.benchmark_db import BenchmarkDatabase

        try:
            matches = BenchmarkDatabase(args.benchmark_dir).find_best_match(
                model, workload, hardware=optimizer.detect_hardware()
            )
            empirical = {name: rec.to_dict() for name, rec in matches.items()}
        except Exception as exc:
            empirical = {"_error": str(exc)}

    rows = sorted(
        estimates.values(), key=lambda e: (e.compressed_bytes, e.method)
    )[: args.top]

    if args.json:
        payload = {
            "model": model.to_dict(),
            "workload": workload.to_dict(),
            "strategies": {e.method: e.to_dict() for e in rows},
            "empirical": {k: v for k, v in empirical.items() if not k.startswith("_")},
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Analytic KV memory estimates for {model.model_id} ({model.architecture})")
    print(
        f"  context={workload.context_length} gen={workload.generation_length} "
        f"batch={workload.batch_size}"
    )
    baseline = rows[0].baseline_bytes if rows else 0
    print(f"  fp16 baseline (loaded): {_fmt_bytes(baseline)}")
    print(f"{'method':<16s} {'compressed':>10s} {'of baseline':>12s} {'conf':>6s}")
    for est in rows:
        print(
            f"{est.method:<16s} {_fmt_bytes(est.compressed_bytes):>10s} "
            f"{est.savings_percent:>11.1f}% {'':>1s} {est.confidence:>6s}"
        )
    if empirical:
        print("  benchmark overrides present for some methods (see --json)")


if __name__ == "__main__":
    main(sys.argv[1:])

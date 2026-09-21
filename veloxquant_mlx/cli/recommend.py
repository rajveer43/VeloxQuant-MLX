"""CLI: recommend a KV-cache method for a Mac chip + RAM + model size.

Two modes:
  * Legacy (flags --chip/--ram-gb/--model-class/--goal): the original
    heuristic recommender, unchanged.
  * Auto (--auto or --model-config): the RFC ``method="auto"`` pipeline —
    profiles the machine and model, ranks every registered strategy under an
    objective, and serves the top choices with a plain-text explanation.
"""

from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.planning import (
    AutoOptimizer,
    AutoOptimizerOptions,
    WorkloadObjective,
    WorkloadProfile,
)
from veloxquant_mlx.tools.mac_recommender import (
    ALLOWED_RAM_GB,
    MODEL_WEIGHT_GB_4BIT,
    RecommendRequest,
    recommend,
    ruleset_dict,
)

_CHIP_CHOICES = ["M1", "M2", "M3", "M4"]
_OBJECTIVE_CHOICES = [
    WorkloadObjective.MEMORY,
    WorkloadObjective.LATENCY,
    WorkloadObjective.THROUGHPUT,
    WorkloadObjective.QUALITY,
    WorkloadObjective.BALANCED,
]


def _read_model_config(path: str) -> dict:
    import json as _json
    from pathlib import Path

    config = _json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("model config must be a JSON object (config.json)")
    return config


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and print a KV-cache method recommendation.

    Dispatches to the hardware-aware auto-selector (``_main_auto``) when
    --auto or --model-config is given, otherwise runs the legacy
    --chip/--ram-gb/--model-class/--goal heuristic (mac_recommender.recommend),
    printing text or JSON (--json). --dump-ruleset prints the static ruleset
    and exits before either path runs.
    """
    parser = argparse.ArgumentParser(
        prog="veloxquant recommend",
        description=(
            "Recommend a VeloxQuant-MLX KV-cache method. Use --auto with a "
            "--model-config (config.json) for the hardware-aware auto-selector, "
            "or the legacy --chip/--ram-gb/--model-class/--goal heuristic."
        ),
    )
    # --- Auto-selector inputs ------------------------------------------------
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use the hardware-aware auto-selector (RFC method=\"auto\")",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to a HF-style config.json for architecture extraction",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Display label for the model (defaults to the config's _name_or_path)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Workload context length in tokens (default 4096)",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Workload generation length in tokens (default 512)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        choices=_OBJECTIVE_CHOICES,
        help="Optimization objective (default balanced)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print a plain-text explanation instead of the compact summary",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip live serve-tier validation of the recommended methods",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Architecture slug override when the config has no 'architectures'",
    )
    # --- Legacy heuristic inputs --------------------------------------------
    parser.add_argument(
        "--chip",
        default=None,
        choices=_CHIP_CHOICES,
        help="Apple Silicon family (legacy mode)",
    )
    parser.add_argument(
        "--ram-gb",
        type=int,
        default=None,
        choices=list(ALLOWED_RAM_GB),
        help="RAM size in GB (legacy mode)",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        choices=list(MODEL_WEIGHT_GB_4BIT),
        help="Approximate parameter class (legacy mode)",
    )
    parser.add_argument(
        "--goal",
        default=None,
        choices=[
            "everyday",
            "max_key_accounting",
            "max_context",
            "best_quality",
            "constant_memory",
        ],
        help="Optimization goal (legacy mode)",
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--n-query-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    parser.add_argument(
        "--dump-ruleset",
        action="store_true",
        help="Print static ruleset JSON and exit",
    )
    args = parser.parse_args(argv)

    if args.dump_ruleset:
        print(json.dumps(ruleset_dict(), indent=2))
        return

    auto_mode = args.auto or args.model_config is not None
    if auto_mode:
        _main_auto(args)
        return

    # --- Legacy path ----------------------------------------------------------
    missing = [name for name in ("chip", "ram_gb", "model_class", "goal") if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            f"legacy mode requires --{'/--'.join(missing)}; or use --auto for the "
            "hardware-aware selector"
        )
    req = RecommendRequest(
        chip=args.chip,
        ram_gb=args.ram_gb,
        model_class=args.model_class,
        goal=args.goal,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )
    result = recommend(req)
    payload = {
        "request": {
            "chip": req.chip,
            "ram_gb": req.ram_gb,
            "model_class": req.model_class,
            "goal": req.goal,
            "seq_len": req.seq_len,
            "n_layers": req.n_layers,
            "n_kv_heads": req.n_kv_heads,
            "head_dim": req.head_dim,
        },
        "recommendation": result.to_dict(),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    rec = payload["recommendation"]
    print("VeloxQuant-MLX method recommender")
    print(f"  chip={req.chip}  ram={req.ram_gb} GB  model~{req.model_class}  goal={req.goal}")
    print(f"  method={rec['method']}")
    print(f"  knobs={rec['knobs']}")
    print(f"  key_accounting_ratio≈{rec['key_accounting_ratio']}x")
    print(f"  resident_savings_likely={rec['resident_savings_likely']}")
    print(
        f"  kv_fp16_mb≈{rec['kv_fp16_mb']}  kv_compressed_mb_est≈{rec['kv_compressed_mb_estimate']}"
    )
    print(f"  rationale: {rec['rationale']}")
    if rec["warnings"]:
        print("  warnings:")
        for w in rec["warnings"]:
            print(f"    - {w}")


def _main_auto(args: argparse.Namespace) -> None:
    """Auto-selector path: profile, plan, explain."""
    model_config = _read_model_config(args.model_config) if args.model_config else None
    model = None
    if model_config is None:
        # Geometry-only path: build the profile from --n-layers/--heads/...
        from veloxquant_mlx.profiling.model_profiler import profile_model_from_config

        model = profile_model_from_config(
            None,
            num_layers=args.n_layers,
            num_query_heads=args.n_query_heads or args.n_kv_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            architecture=args.architecture,
        )

    workload = WorkloadProfile(
        context_length=args.context or 4096,
        generation_length=args.generation or 512,
        objective=args.objective or WorkloadObjective.BALANCED,
    )
    optimizer = AutoOptimizer(
        AutoOptimizerOptions(probe_top_n=0 if args.no_probe else 3)
    )
    result = optimizer.recommend_strategy(
        model_config, model=model, workload=workload
    )

    if args.json:
        payload = {
            "mode": "auto",
            "model": result.model.to_dict(),
            "hardware": result.hardware.to_dict(),
            "workload": result.workload.to_dict(),
            "objective": result.objective,
            "fallback_used": result.fallback_used,
            "ranked": [item.to_dict() for item in result.ranked],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"VeloxQuant-MLX auto-strategy selector (objective: {result.objective})")
    if not args.explain:
        print(f"  model: {result.model.model_id} ({result.model.architecture})")
        hw = result.hardware
        print(f"  hardware: {hw.chip}, MLX {hw.mlx_version}")
        print(f"  workload: ctx={workload.context_length} gen={workload.generation_length}")
        if result.fallback_used:
            print("  FALLBACK: no viable candidate; use the library default")
            print(f"    {optimizer.fallback_method()}")
            for name, reason in list(result.candidates.excluded.items())[:3]:
                print(f"    - {name}: {reason}")
            return
        for idx, item in enumerate(result.ranked, 1):
            est = item.memory_estimate
            print(
                f"  {idx}. {item.method}  (score {item.score:.3f}, "
                f"~{est.savings_percent:.0f}% memory vs fp16, "
                f"evidence={item.evidence})"
            )
        print("  (add --explain for the full analysis)")
    else:
        print(optimizer.explain(result))


if __name__ == "__main__":
    main(sys.argv[1:])

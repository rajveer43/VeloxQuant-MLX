"""CLI: recommend a KV-cache method for a Mac chip + RAM + model size."""

from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.tools.mac_recommender import (
    RecommendRequest,
    recommend,
    ruleset_dict,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant recommend",
        description=(
            "Recommend a VeloxQuant-MLX method from chip, RAM, model class, "
            "and goal. Estimates are accounting-aware and state when "
            "resident RAM savings are unlikely."
        ),
    )
    parser.add_argument(
        "--chip",
        required=True,
        choices=["M1", "M2", "M3", "M4"],
        help="Apple Silicon family (Pro/Max/Ultra are RAM tiers, not separate chips here)",
    )
    parser.add_argument(
        "--ram-gb",
        required=True,
        type=int,
        choices=[8, 16, 24, 32, 36, 48, 64, 128],
    )
    parser.add_argument(
        "--model-class",
        required=True,
        choices=["1B", "3B", "7B", "14B", "32B"],
        help="Approximate parameter class (4-bit weight footprint estimate)",
    )
    parser.add_argument(
        "--goal",
        required=True,
        choices=[
            "everyday",
            "max_key_accounting",
            "max_context",
            "best_quality",
            "constant_memory",
        ],
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
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


if __name__ == "__main__":
    main(sys.argv[1:])

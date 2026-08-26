"""CLI: automatically pick a KV-cache quantization config for a workload + machine (#253)."""

from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.tools.auto_kv_config import (
    HardwareProfile,
    WorkloadProfile,
    select_kv_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant autoconfig",
        description=(
            "Automatically select a KV-cache quantization method, bit-width, group size, "
            "and packing strategy from workload and hardware characteristics -- no manual "
            "goal selection required (unlike `veloxquant recommend`)."
        ),
    )
    parser.add_argument(
        "--seq-len", type=int, default=4096, help="Expected max context length in tokens"
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--ram-gb", type=float, default=16.0, help="Total unified memory in GB")
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help=(
            "GB actually available for the KV cache; default derives a conservative "
            "estimate from --ram-gb"
        ),
    )
    parser.add_argument(
        "--no-metal",
        action="store_true",
        help="Assume Metal kernel acceleration is unavailable",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    workload = WorkloadProfile(
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
    )
    hardware = HardwareProfile(
        ram_gb=args.ram_gb,
        memory_budget_gb=args.memory_budget_gb,
        metal_available=not args.no_metal,
    )
    result = select_kv_config(workload, hardware)

    if args.json:
        payload = {
            "workload": {
                "seq_len": workload.seq_len,
                "head_dim": workload.head_dim,
                "n_layers": workload.n_layers,
                "n_kv_heads": workload.n_kv_heads,
            },
            "hardware": {
                "ram_gb": hardware.ram_gb,
                "memory_budget_gb": hardware.resolved_budget_gb(),
                "metal_available": hardware.metal_available,
            },
            "result": result.to_dict(),
        }
        print(json.dumps(payload, indent=2))
        return

    print("VeloxQuant-MLX automatic KV-cache configuration")
    print(f"  seq_len={workload.seq_len}  head_dim={workload.head_dim}  ram={hardware.ram_gb} GB")
    print(f"  context_regime={result.context_regime}")
    print(f"  method={result.method}  bit_width={result.bit_width}  group_size={result.group_size}")
    print(f"  packing_strategy={result.packing_strategy}")
    print(f"  knobs={result.knobs}")
    print(f"  kv_fp16_gb≈{result.kv_fp16_gb}  memory_pressure≈{result.memory_pressure_ratio}x")
    print(f"  rationale: {result.rationale}")
    if result.warnings:
        print("  warnings:")
        for w in result.warnings:
            print(f"    - {w}")


if __name__ == "__main__":
    main(sys.argv[1:])

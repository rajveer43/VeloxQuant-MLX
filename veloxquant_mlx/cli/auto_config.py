"""CLI: pick a hardware-aware KV-cache config for a workload (issue #253)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields

from veloxquant_mlx.config.auto_config import (
    HardwareInfo,
    WorkloadSpec,
    detect_hardware_info,
    select_kv_cache_config,
)

# Per-method knob fields worth reporting, keyed by the method the selector
# can pick. KVCacheConfig is one dataclass shared by all 40+ registry
# methods, so every instance carries every other method's fields at their
# unrelated defaults (e.g. a kivi config still has a `gear_bits` field) —
# reporting only the selected method's own knobs (plus `method`/`head_dim`)
# avoids implying those unrelated defaults were a deliberate choice.
_METHOD_FIELDS: dict[str, list[str]] = {
    "turboquant_rvq": ["bit_width_inlier"],
    "kivi": ["bit_width_inlier", "kivi_group_size"],
    "kvquant": ["kvquant_bits", "kvquant_group_size", "kvquant_outlier_fraction"],
    "gear": ["gear_bits", "gear_group_size"],
}


def _config_to_dict(config: object) -> dict:
    known = {f.name for f in fields(config)}
    reported = ["method", "head_dim"] + _METHOD_FIELDS.get(config.method, [])
    return {name: getattr(config, name) for name in reported if name in known}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant auto-config",
        description=(
            "Pick a KV-cache method/bit-width/group-size for a workload "
            "(head_dim, seq_len, n_layers, batch_size), optionally biased by "
            "detected or supplied hardware memory pressure. Selects from a "
            "small pool of servable methods (turboquant_rvq, kivi, kvquant, "
            "gear) rather than the full method registry."
        ),
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--total-memory-bytes",
        type=int,
        default=None,
        help="Override detected total device memory. Omit to auto-detect via mx.device_info().",
    )
    parser.add_argument(
        "--active-memory-bytes",
        type=int,
        default=None,
        help="Override detected active (already-in-use) memory. Ignored unless --total-memory-bytes is also set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    args = parser.parse_args(argv)

    workload = WorkloadSpec(
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
    )

    if args.total_memory_bytes is not None:
        hardware = HardwareInfo(
            total_memory_bytes=args.total_memory_bytes,
            active_memory_bytes=args.active_memory_bytes or 0,
        )
    else:
        hardware = detect_hardware_info()

    result = select_kv_cache_config(workload, hardware)

    payload = {
        "workload": {
            "head_dim": workload.head_dim,
            "seq_len": workload.seq_len,
            "n_layers": workload.n_layers,
            "batch_size": workload.batch_size,
        },
        "hardware": {
            "total_memory_bytes": hardware.total_memory_bytes,
            "active_memory_bytes": hardware.active_memory_bytes,
        },
        "config": _config_to_dict(result.config),
        "reason": result.reason,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("VeloxQuant-MLX auto-config")
    print(
        f"  head_dim={workload.head_dim}  seq_len={workload.seq_len}  "
        f"n_layers={workload.n_layers}  batch_size={workload.batch_size}"
    )
    print(f"  method={payload['config']['method']}")
    print(f"  config={payload['config']}")
    print(f"  reason: {result.reason}")


if __name__ == "__main__":
    main(sys.argv[1:])

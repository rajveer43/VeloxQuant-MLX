"""CLI: report the detected Apple-Silicon hardware profile."""

from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.profiling.hardware_profiler import (
    detect_hardware_profile,
    measure_bandwidth_gbps,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant profile-hardware",
        description=(
            "Print the detected hardware profile (chip, memory, MLX/macOS "
            "versions, Metal availability) and optionally a measured bandwidth "
            "estimate."
        ),
    )
    parser.add_argument(
        "--measure-bandwidth",
        action="store_true",
        help="Run a quick MLX copy benchmark and report measured GB/s",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    args = parser.parse_args(argv)

    hw = detect_hardware_profile()
    data = hw.to_dict()
    if args.measure_bandwidth:
        measured = measure_bandwidth_gbps()
        data["measured_bandwidth_gbps"] = measured

    if args.json:
        print(json.dumps(data, indent=2))
        return

    print("VeloxQuant-MLX hardware profile")
    print(f"  chip:              {hw.chip}")
    print(f"  chip generation:   {hw.chip_generation or 'unknown'}")
    print(
        f"  total memory:      {hw.total_memory_bytes/1024**3:.2f} GiB"
        if hw.total_memory_bytes
        else "  total memory:      unknown"
    )
    print(
        f"  available memory:  {hw.available_memory_bytes/1024**3:.2f} GiB"
        if hw.available_memory_bytes
        else "  available memory:  unknown"
    )
    print(f"  nominal bandwidth: {hw.bandwidth_gbps} GB/s" if hw.bandwidth_gbps else "  nominal bandwidth: unknown")
    print(f"  MLX version:       {hw.mlx_version or 'unknown'}")
    print(f"  macOS version:     {hw.macos_version or 'unknown'}")
    print(f"  Metal available:   {hw.metal_available}")
    measured = data.get("measured_bandwidth_gbps")
    if measured is not None:
        print(f"  measured bandwidth:{measured:.1f} GB/s (naive copy benchmark)")


if __name__ == "__main__":
    main(sys.argv[1:])

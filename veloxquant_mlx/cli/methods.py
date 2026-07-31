"""``veloxquant methods`` — list KV-cache methods and their serve tiers.

The ``--json`` form is the contract the macOS control panel decodes, so the
method list and serve-tier logic live in Python only and never get duplicated
in Swift.
"""
from __future__ import annotations

import argparse
import json
import sys

from veloxquant_mlx.cache.registry import (
    DEFAULT_SERVE_METHOD,
    MethodFamily,
    ServeTier,
    list_methods,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veloxquant methods",
        description="List KV-cache methods with their serving support tier.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON (consumed by the control panel).",
    )
    parser.add_argument(
        "--servable-only", action="store_true",
        help="Hide methods that cannot be served.",
    )
    parser.add_argument(
        "--family", choices=[f.value for f in MethodFamily], default=None,
        help="Filter by method family.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    family = MethodFamily(args.family) if args.family else None
    infos = list_methods(servable_only=args.servable_only, family=family)

    if args.json:
        json.dump(
            {
                "schema_version": 1,
                "default_serve_method": DEFAULT_SERVE_METHOD,
                "accounting_only": True,
                "accounting_note": (
                    "Compression is accounting-only: caches store dequantized "
                    "fp16 tensors, so reported byte counts do not correspond to "
                    "runtime memory reduction."
                ),
                "methods": [i.to_dict() for i in infos],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    servable = [i for i in infos if i.serve_tier.is_servable]
    blocked = [i for i in infos if not i.serve_tier.is_servable]

    print(f"{len(infos)} method(s); {len(servable)} servable, {len(blocked)} unsupported.\n")
    for info in infos:
        mark = "  " if info.serve_tier.is_servable else "✗ "
        adapted = "  [adapted]" if info.is_adapted else ""
        print(f"{mark}{info.name:<18} {info.family.value:<13} {info.serve_tier.label}{adapted}")
        print(f"    {info.blurb}")
        if info.unsupported_reason:
            print(f"    reason: {info.unsupported_reason}")
        if info.paper_deviation:
            print(f"    deviation: {info.paper_deviation}")
        print()

    if any(i.serve_tier is ServeTier.ACCOUNTING_ONLY for i in infos):
        print(
            "Note: servable methods are accounting-only — byte counters measure "
            "compression\nfidelity, not runtime memory saved. See issue #27."
        )


if __name__ == "__main__":
    main()

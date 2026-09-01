"""``veloxquant profile`` — per-layer latency/memory breakdown for one run.

Wires :class:`~veloxquant_mlx.profiling.MLXCacheProfiler` around every
per-layer cache :func:`KVCacheBuilder.for_model` builds, runs a real prompt
through ``mlx_lm.generate`` with that cache list as ``prompt_cache``, then
emits the result as JSON (issue #45's control-panel contract, mirroring
``methods --json`` / ``serve``'s ``VELOXQUANT_READY`` line: stable
snake_case keys, a ``schema_version``, and an unconditional accounting-only
warning since these caches store dequantized fp16 — see ``serve.py``'s
``ACCOUNTING_WARNING`` for why that is never optional).

Servable caches implement one fused ``update_and_fetch`` call rather than
the standalone interface's separate append_key/append_value/attend, so
there is no real quantize/dequantize/write split to report per layer —
see ``MLXCacheProfiler``'s docstring. The JSON emits a single
``compute_latency_ms`` per layer instead of fabricating three numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from veloxquant_mlx.cache.registry import DEFAULT_SERVE_METHOD, get_method

SCHEMA_VERSION = 1

#: Same wording as serve.py's ACCOUNTING_WARNING / methods.py's accounting
#: note — one honesty message, not three drifting copies.
ACCOUNTING_NOTE = (
    "Compression is accounting-only. Caches store dequantized fp16 tensors, so "
    "reported byte counters measure compression fidelity, not runtime memory "
    "saved. Do not read these numbers as RSS reduction (issue #27)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veloxquant profile",
        description=(
            "Profile per-layer KV-cache latency, memory, and compression for "
            "one run of a real model + prompt, and print the result as JSON."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id (e.g. mlx-community/...) or local path.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_SERVE_METHOD,
        help=f"KV-cache method (default: {DEFAULT_SERVE_METHOD}). "
        "Run 'veloxquant methods' to list options.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=2,
        help="Inlier bit width for the cache (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Quantizer seed (default: 42).")
    parser.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt used to exercise the cache (default: a short filler sentence).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Tokens to generate while profiling (default: 64).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Method-specific KVCacheConfig field, repeatable "
        "(e.g. --set kivi_group_size=64). Run 'veloxquant methods --json' "
        "to see which fields apply to a method.",
    )
    return parser


def validate_method(method: str) -> None:
    """Reject standalone (non-mlx_lm-serving) methods before the model loads.

    ``KVCacheBuilder.for_model`` already raises for these, but that happens
    only after the (potentially large) model download/load — fail fast here
    instead, matching ``serve.py``'s ``validate_method``.
    """
    try:
        info = get_method(method)
    except KeyError as exc:
        raise SystemExit(f"error: {exc}") from None

    if not info.serve_tier.is_servable:
        reason = info.unsupported_reason or "method does not implement the mlx_lm serving contract"
        raise SystemExit(
            f"error: method {method!r} cannot be profiled through a live model run.\n"
            f"  reason: {reason}\n"
            f"  Run 'veloxquant methods --servable-only' to see valid choices "
            f"(default: {DEFAULT_SERVE_METHOD})."
        )


def _warn(message: str) -> None:
    print(f"[veloxquant profile] {message}", file=sys.stderr)


def parse_overrides(pairs: List[str]) -> dict:
    """Same FIELD=VALUE parsing as ``serve.py``'s ``parse_overrides``."""
    import dataclasses

    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.registry import describe_field

    valid = {f.name for f in dataclasses.fields(KVCacheConfig)}
    overrides: dict = {}

    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"error: --set expects FIELD=VALUE, got {pair!r}")

        name, _, raw = pair.partition("=")
        name, raw = name.strip(), raw.strip()

        if name not in valid:
            raise SystemExit(f"error: unknown config field {name!r}")
        if name in ("method", "store", "observers", "dtype"):
            raise SystemExit(f"error: {name!r} cannot be set with --set")

        schema = describe_field(name)
        if raw == "" and schema["optional"]:
            overrides[name] = None
            continue

        try:
            if schema["type"] == "int":
                overrides[name] = int(raw)
            elif schema["type"] == "float":
                overrides[name] = float(raw)
            elif schema["type"] == "bool":
                overrides[name] = raw.lower() in ("1", "true", "yes", "on")
            else:
                overrides[name] = raw
        except ValueError:
            raise SystemExit(f"error: {name!r} expects {schema['type']}, got {raw!r}") from None

    return overrides


def build_config(args: argparse.Namespace) -> Any:
    from veloxquant_mlx.cache import KVCacheConfig

    overrides = parse_overrides(getattr(args, "set", []) or [])
    return KVCacheConfig(
        method=args.method,
        bit_width_inlier=args.bits,
        seed=args.seed,
        **overrides,
    )


def run_profile(args: argparse.Namespace) -> dict:
    """Load the model, profile a real generation pass, return the JSON payload."""
    import time

    from mlx_lm import generate, load

    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.profiling import MLXCacheProfiler, format_profile_table, profile_layers

    config = build_config(args)

    _warn(f"loading model {args.model!r} ...")
    model, tokenizer = load(args.model)

    raw_caches = KVCacheBuilder.for_model(model, config)
    profilers = [MLXCacheProfiler(cache, layer_id=i) for i, cache in enumerate(raw_caches)]

    _warn(f"profiling {len(profilers)} layer cache(s) with method={args.method!r} bits={args.bits}")

    t0 = time.perf_counter()
    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        prompt_cache=profilers,
    )
    elapsed_s = time.perf_counter() - t0

    report = profile_layers(profilers, elapsed_s=elapsed_s)
    _warn("\n" + format_profile_table(report))

    layers = [
        {
            "layer_index": layer.layer_id,
            "compute_latency_ms": layer.quantize_ms_mean,
            "is_fused": layer.is_fused,
            "peak_memory_bytes": layer.peak_memory_bytes,
            "compression_ratio": layer.compression_ratio,
        }
        for layer in report.layers
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "method": args.method,
        "bits": args.bits,
        "accounting_only": True,
        "accounting_note": ACCOUNTING_NOTE,
        "layers": layers,
        "summary": {
            "total_latency_ms": sum(layer.quantize_ms_total for layer in report.layers),
            "peak_memory_bytes": report.total_bytes_written,
            "mean_compression_ratio": report.overall_compression_ratio,
            "tokens_per_second": report.tokens_per_sec,
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_method(args.method)
    payload = run_profile(args)
    print(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1:])

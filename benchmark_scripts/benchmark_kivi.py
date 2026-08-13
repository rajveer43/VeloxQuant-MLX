"""End-to-end KIVI benchmark against an fp16 baseline on MLX models.

Measures throughput, peak memory, and realized KV-cache compression for
several KIVI bit-widths against a vanilla fp16 KV cache, then writes a
``results.json`` (schema-compatible with the VecInfer benchmark) plus a
summary plot under ``figures/kivi/<model-stem>/``.

KIVI (arXiv:2402.02750) is deterministic — no codebook calibration step is
needed, unlike VecInfer.  The fp16 baseline is **always timed for real**
(the RaBitQ benchmarks recorded ``fp16_ms: 0`` which invalidated every
speedup; this script does not do that).  On Apple Silicon we expect a
*memory* win and a *throughput cost* vs fp16 — the paper's speedup comes
from a CUDA kernel that does not port to Metal.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_kivi.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


# KIVI only quantizes tokens that age out of the fp16 residual window, so a
# realistic benchmark needs a prompt whose prefill length is >> residual_length.
# We synthesize a long-context prompt by repeating a passage, then append a
# question — this exercises the quantized path the way long-context inference
# does.  Reported seq-lengths are recorded in results.json.
_PASSAGE = (
    "The key-value cache stores the attention keys and values of every past "
    "token so the model need not recompute them. Its size grows linearly with "
    "context length and, on Apple Silicon unified memory, it competes with the "
    "model weights and the operating system for the same pool. "
)
PROMPT = (_PASSAGE * 40) + (
    "\n\nGiven the passage above, explain in simple terms why the KV cache is "
    "the binding memory constraint for long-context inference on Apple Silicon, "
    "covering both the linear growth and the unified-memory contention."
)


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _peak_mb() -> float:
    try:
        return float(mx.metal.get_peak_memory()) / (1024**2)
    except Exception:
        return float("nan")


def _reset_peak() -> None:
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _hardware() -> dict:
    """Best-effort hardware record (chip + RAM) for honest provenance."""
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import subprocess

        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if chip:
            info["chip"] = chip
        if mem:
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
    except Exception:
        pass
    return info


def _build_kivi_caches(model, b: int, group_size: int, residual_length: int) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.kivi_cache import KIVIKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None:
            caches.append(_FallbackCache())
            continue
        cfg = KVCacheConfig(
            method="kivi",
            head_dim=hd,
            bit_width_inlier=b,
            kivi_group_size=group_size,
            residual_length=residual_length,
            seed=42 + i,
        )
        caches.append(KIVIKVCache(cfg))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple:
    from mlx_lm import generate

    t0 = time.time()
    try:
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=caches,
        )
    except TypeError:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return n_tok, elapsed


def _run_config(model, tokenizer, name: str, build_caches_fn, max_tokens: int) -> dict:
    print(f"\n--- {name} ---", flush=True)
    _reset_peak()
    caches = build_caches_fn()
    n_tok, elapsed = _generate(model, tokenizer, PROMPT, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    key_compressed = key_fp16 = 0
    val_compressed = val_fp16 = residual_fp16 = 0
    avg_bits = 16.0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_compressed += c.compressed_key_bytes
            key_fp16 += c.fp16_key_bytes
            val_compressed += getattr(c, "compressed_value_bytes", 0)
            val_fp16 += getattr(c, "fp16_value_bytes", 0)
            residual_fp16 += getattr(c, "residual_fp16_bytes", 0)
    if caches and hasattr(caches[0], "assigned_avg_bits"):
        avg_bits = float(caches[0].assigned_avg_bits)

    # Key-only ratio (matches vecinfer schema) and an end-to-end KV ratio
    # that *includes* the fp16 residual window so we never inflate the number.
    key_ratio = (key_fp16 / key_compressed) if key_compressed else 1.0
    total_fp16 = key_fp16 + val_fp16
    total_comp = key_compressed + val_compressed + residual_fp16
    full_kv_ratio = (total_fp16 / total_comp) if total_comp else 1.0

    print(
        f"  {n_tok} tok in {elapsed:.2f}s ({throughput:.1f} tok/s)  "
        f"peak={peak_mb:.0f}MB  key_x={key_ratio:.2f}  fullKV_x={full_kv_ratio:.2f}"
    )

    return {
        "name": name,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "key_compression": key_ratio,
        "full_kv_compression": full_kv_ratio,
        "avg_bits": avg_bits,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
    }


def _plot_summary(results: list, out_path: Path, model_label: str, hw: dict) -> None:
    names = [r["name"] for r in results]
    tput = [r["throughput_tok_s"] for r in results]
    peaks = [r["peak_mb"] for r in results]
    kratio = [r["key_compression"] for r in results]
    fullkv = [r["full_kv_compression"] for r in results]
    colors = ["#666", "#00d4ff", "#7c3aed", "#ff6b35", "#22c55e"][: len(names)]
    chip = hw.get("chip", hw.get("machine", "Apple Silicon"))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"KIVI benchmark — {model_label} ({chip})", fontsize=14)
    for ax, vals, title, ylab in [
        (axes[0], tput, "Throughput", "Tokens / second"),
        (axes[1], peaks, "Peak memory", "Peak memory (MB)"),
        (axes[2], kratio, "Key compression", "Key compression (x)"),
        (axes[3], fullkv, "Full-KV compression (incl. fp16 residual)", "Full-KV (x)"),
    ]:
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="KIVI benchmark")
    parser.add_argument("--model", required=True, help="HF model id (mlx-community/...)")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from mlx_lm import load

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/kivi") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)

    layers = getattr(model, "layers", None) or model.model.layers
    margs = getattr(model, "args", None) or model.model.args
    first_attn = None
    for L in layers:
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is not None:
            first_attn = attn
            break
    head_dim = getattr(first_attn, "head_dim", None) or (
        margs.hidden_size // margs.num_attention_heads
    )
    n_kv_heads = getattr(margs, "num_key_value_heads", None) or getattr(
        margs, "num_attention_heads", 1
    )
    n_layers = len(layers)
    hw = _hardware()
    prompt_tokens = len(tokenizer.encode(PROMPT))
    print(f"  head_dim={head_dim}, n_kv_heads={n_kv_heads}, n_layers={n_layers}")
    print(f"  prompt_tokens={prompt_tokens} (residual_length={args.residual_length})")
    print(f"  hardware={hw}")

    # KIVI configs: (label, bits).  group_size + residual shared via CLI.
    configs = [("KIVI-2bit", 2), ("KIVI-3bit", 3), ("KIVI-4bit", 4)]

    results = [
        _run_config(
            model,
            tokenizer,
            "fp16-baseline",
            lambda: _build_fp16_caches(model),
            args.max_tokens,
        )
    ]
    for label, b in configs:
        results.append(
            _run_config(
                model,
                tokenizer,
                label,
                lambda b=b: _build_kivi_caches(model, b, args.group_size, args.residual_length),
                args.max_tokens,
            )
        )

    _plot_summary(results, out_dir / "kivi_summary.png", model_stem, hw)

    payload = {
        "model": args.model,
        "head_dim": head_dim,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "max_tokens": args.max_tokens,
        "prompt_tokens": prompt_tokens,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "prompt": PROMPT[:200] + ("..." if len(PROMPT) > 200 else ""),
        "hardware": hw,
        "results": results,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {json_path}")
    for r in results:
        print(
            f"  {r['name']:<16s} {r['throughput_tok_s']:6.1f} tok/s  "
            f"{r['peak_mb']:7.1f} MB  key_x={r['key_compression']:.2f}  "
            f"fullKV_x={r['full_kv_compression']:.2f}  toks={r['tokens_generated']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

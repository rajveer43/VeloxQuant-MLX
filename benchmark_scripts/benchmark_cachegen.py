"""End-to-end CacheGen benchmark vs KIVI and fp16 baseline.

Measures throughput, peak memory, and — the headline — the entropy-coding
storage win from token-wise locality, versus fixed-width packing at the same
bit-width.

CacheGen (arXiv:2310.07240-adapted, SIGCOMM 2024) reconstructs identically to
plain group quant; its contribution is the entropy-coded byte model. The
offline harness measures the entropy savings on correlated (random-walk) vs
incompressible (iid) code streams.

Compared methods:
  - fp16_baseline   (standard mlx_lm KVCache)
  - kivi_2bit       (KIVI uniform 2-bit reference)
  - cachegen_4bit   (CacheGen 4-bit, token-delta entropy coding)
  - cachegen_3bit   (CacheGen 3-bit)

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_cachegen.py \\
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


_PASSAGE = (
    "The key-value cache stores the attention keys and values of every past "
    "token so the model need not recompute them. Its size grows linearly with "
    "context length and, on Apple Silicon unified memory, it competes with the "
    "model weights and the operating system for the same pool. "
)
PROMPT = (_PASSAGE * 40) + (
    "\n\nGiven the passage above, explain why the KV cache is the binding memory "
    "constraint for long-context inference on Apple Silicon."
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


def _chip_name() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return platform.processor()


def run_one(model, tokenizer, cache_arg, n_decode: int, label: str) -> dict:
    from mlx_lm import generate

    _reset_peak()
    t0 = time.perf_counter()
    result = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=n_decode,
        kv_cache=cache_arg,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    peak = _peak_mb()
    tokens_out = len(tokenizer.encode(result)) if isinstance(result, str) else n_decode
    tps = tokens_out / elapsed if elapsed > 0 else float("nan")
    return {
        "label": label,
        "tokens_per_sec": round(tps, 2),
        "peak_memory_mb": round(peak, 1),
        "elapsed_s": round(elapsed, 3),
        "tokens_generated": tokens_out,
    }


def build_cache(method: str, model, overrides: dict):
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

    cfg = KVCacheConfig(method=method, **overrides)
    return KVCacheBuilder.for_model(model, cfg)


def entropy_savings_offline(seed: int = 0) -> dict:
    """Offline: entropy-coded vs fixed-width bytes on correlated vs iid codes."""
    from veloxquant_mlx.quantizers.cachegen import (
        quantize_to_codes,
        entropy_coded_bytes,
        fixed_width_bytes,
    )

    rng = np.random.default_rng(seed)
    S, D = 256, 128
    walk = np.cumsum(rng.standard_normal((S, D)).astype(np.float32) * 0.12, axis=0)
    iid = rng.standard_normal((S, D)).astype(np.float32)
    out = {}
    for name, data in (("correlated", walk), ("iid", iid)):
        for b in (3, 4):
            st = quantize_to_codes(mx.array(data), bits=b, group_size=32)
            ecb = entropy_coded_bytes(st, use_delta=True)
            fwb = fixed_width_bytes(st)
            out[f"{name}_{b}bit"] = {
                "entropy_bytes": ecb,
                "fixed_width_bytes": fwb,
                "savings": round(1.0 - ecb / fwb, 4) if fwb else None,
            }
    return out


def main() -> None:
    _ensure_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--n-decode", type=int, default=200)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from mlx_lm import load

    model, tokenizer = load(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    savings = entropy_savings_offline()
    print(f"Entropy savings (offline): {savings}")

    methods = [
        ("fp16_baseline", None, {}),
        ("kivi_2bit", "kivi", {"bit_width_inlier": 2, "kivi_group_size": 32}),
        ("cachegen_4bit", "cachegen", {"cachegen_bits": 4}),
        ("cachegen_3bit", "cachegen", {"cachegen_bits": 3}),
    ]

    all_results = []
    for label, method, overrides in methods:
        trial_results = []
        for trial in range(args.trials):
            cache_arg = None if method is None else build_cache(method, model, overrides)
            res = run_one(model, tokenizer, cache_arg, args.n_decode, label)
            trial_results.append(res)
            print(f"  {label} trial {trial + 1}: {res['tokens_per_sec']:.1f} tok/s")
        avg_tps = float(np.mean([r["tokens_per_sec"] for r in trial_results]))
        avg_peak = float(np.mean([r["peak_memory_mb"] for r in trial_results]))
        all_results.append(
            {
                "label": label,
                "method": method or "fp16",
                "avg_tokens_per_sec": round(avg_tps, 2),
                "avg_peak_memory_mb": round(avg_peak, 1),
                "trials": trial_results,
            }
        )

    output = {
        "model": args.model,
        "chip": _chip_name(),
        "n_decode_tokens": args.n_decode,
        "entropy_savings_offline": savings,
        "results": all_results,
    }
    out_path = out_dir / "cachegen_benchmark.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
    _plot(all_results, args.model, out_dir)


def _plot(results: list, model_name: str, out_dir: Path) -> None:
    labels = [r["label"] for r in results]
    tps = [r["avg_tokens_per_sec"] for r in results]
    peak = [r["avg_peak_memory_mb"] for r in results]
    model_stem = Path(model_name).name
    fig_dir = out_dir / "figures" / "cachegen" / model_stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"CacheGen KV Cache — {model_stem}", fontsize=13)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    axes[0].bar(labels, tps, color=colors[: len(labels)])
    axes[0].set_title("Throughput (tok/s)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, peak, color=colors[: len(labels)])
    axes[1].set_title("Peak Memory (MB)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig_path = fig_dir / "cachegen_vs_baselines.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()

"""End-to-end MiniCache benchmark vs XQuant, KIVI, and fp16 baseline.

Measures throughput, peak memory, and the cross-layer depth-merge compression.
MiniCache (arXiv:2405.14366-adapted, NeurIPS 2024) merges adjacent middle-to-deep
layers into a shared SLERP direction + per-layer magnitudes, retaining
high-divergence token pairs. The offline harness measures adjacent-layer
direction cosine (the property MiniCache exploits) and merge reconstruction MSE.

Compared methods:
  - fp16_baseline   (standard mlx_lm KVCache)
  - kivi_2bit       (KIVI uniform 2-bit reference)
  - xquant          (cross-layer code reuse — the other cross-layer method)
  - minicache       (cross-layer SLERP merge, retention 0.9)
  - minicache_aggr  (lower retention threshold → more merging)

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_minicache.py \\
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


def merge_quality_offline(seed: int = 0) -> dict:
    """Offline: adjacent-layer cosine + merge MSE on similar vs dissimilar layers."""
    from veloxquant_mlx.quantizers.minicache import merge_pair, reconstruct_layer, merge_similarity

    rng = np.random.default_rng(seed)
    S, D = 128, 128
    base = rng.standard_normal((S, D)).astype(np.float32)
    similar = base + rng.standard_normal((S, D)).astype(np.float32) * 0.03
    dissimilar = rng.standard_normal((S, D)).astype(np.float32)
    out = {}
    for name, other in (("similar", similar), ("dissimilar", dissimilar)):
        xp, xm = mx.array(base), mx.array(other)
        sim = merge_similarity(xp, xm)
        res = merge_pair(xp, xm, retention_threshold=0.9)
        rm = reconstruct_layer(res, "merge")
        mse = float(mx.mean((rm.astype(mx.float32) - xm) ** 2).item())
        out[name] = {
            "dir_cosine": round(sim["dir_cosine"], 4),
            "merge_mse": round(mse, 6),
            "retention_rate": round(float(mx.mean(res.retained.astype(mx.float32)).item()), 4),
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

    quality = merge_quality_offline()
    print(f"Merge quality (offline): {quality}")

    methods = [
        ("fp16_baseline", None, {}),
        ("kivi_2bit", "kivi", {"bit_width_inlier": 2, "kivi_group_size": 32}),
        ("xquant", "xquant", {"xquant_base_bits": 2}),
        ("minicache", "minicache", {"minicache_retention_threshold": 0.9}),
        ("minicache_aggr", "minicache", {"minicache_retention_threshold": 0.8}),
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
        "merge_quality_offline": quality,
        "results": all_results,
    }
    out_path = out_dir / "minicache_benchmark.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
    _plot(all_results, args.model, out_dir)


def _plot(results: list, model_name: str, out_dir: Path) -> None:
    labels = [r["label"] for r in results]
    tps = [r["avg_tokens_per_sec"] for r in results]
    peak = [r["avg_peak_memory_mb"] for r in results]
    model_stem = Path(model_name).name
    fig_dir = out_dir / "figures" / "minicache" / model_stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"MiniCache KV Cache — {model_stem}", fontsize=13)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    axes[0].bar(labels, tps, color=colors[: len(labels)])
    axes[0].set_title("Throughput (tok/s)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, peak, color=colors[: len(labels)])
    axes[1].set_title("Peak Memory (MB)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig_path = fig_dir / "minicache_vs_baselines.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()

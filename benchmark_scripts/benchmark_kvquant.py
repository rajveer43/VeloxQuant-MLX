"""End-to-end KVQuant-NUQ benchmark vs KIVI (uniform), SVDq, and fp16 baseline.

Measures throughput, peak memory, and — the headline claim — reconstruction MSE
at an equal bit budget versus uniform quantization. Results are written to
``results/kvquant_benchmark.json`` and a summary plot saved under
``figures/kvquant/<model-stem>/``.

KVQuant-NUQ (arXiv:2401.18079-adapted, NeurIPS 2024) is zero-calibration: the
non-uniform signpost levels are fit online from the prefill batch via Lloyd-Max,
and a small top-magnitude outlier fraction is carved out to fp16. Keys are
quantized per-channel, values per-token.

Compared methods:
  - fp16 baseline (standard mlx_lm KVCache)
  - kivi_2bit      (KIVI uniform 2-bit, the uniform reference)
  - svdq_1_25bit   (SVDq sub-2-bit keys)
  - kvquant_2bit   (KVQuant NUQ 2-bit, outlier_fraction=0.01)
  - kvquant_3bit   (KVQuant NUQ 3-bit, outlier_fraction=0.01)
  - kvquant_3bit_no_outlier (NUQ 3-bit, outlier_fraction=0 — ablation)

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_kvquant.py \\
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


def reconstruction_mse_vs_uniform(seed: int = 0) -> dict:
    """Offline: NUQ vs uniform reconstruction MSE at equal bits on Laplacian data."""
    from veloxquant_mlx.quantizers.kvquant import nuq_quant_dequant
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    rng = np.random.default_rng(seed)
    x = mx.array(rng.laplace(0, 1, (512, 128)).astype(np.float16))
    out = {}
    for b in (2, 3, 4):
        nuq = nuq_quant_dequant(x, bits=b, outlier_fraction=0.01)
        uni = _group_quant_dequant(x, b=b, group_size=512)
        mse_nuq = float(mx.mean((x.astype(mx.float32) - nuq.astype(mx.float32)) ** 2).item())
        mse_uni = float(mx.mean((x.astype(mx.float32) - uni.astype(mx.float32)) ** 2).item())
        out[f"{b}bit"] = {
            "nuq_mse": round(mse_nuq, 6),
            "uniform_mse": round(mse_uni, 6),
            "mse_reduction": round(1.0 - mse_nuq / mse_uni, 4) if mse_uni else None,
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

    recon = reconstruction_mse_vs_uniform()
    print(f"NUQ vs uniform reconstruction MSE: {recon}")

    methods = [
        ("fp16_baseline", None, {}),
        ("kivi_2bit", "kivi", {"bit_width_inlier": 2, "kivi_group_size": 32}),
        ("svdq_1_25bit", "svdq", {"svdq_energy_threshold": 0.95, "svdq_hi_fraction": 0.25}),
        ("kvquant_2bit", "kvquant", {"kvquant_bits": 2, "kvquant_outlier_fraction": 0.01}),
        ("kvquant_3bit", "kvquant", {"kvquant_bits": 3, "kvquant_outlier_fraction": 0.01}),
        (
            "kvquant_3bit_no_outlier",
            "kvquant",
            {"kvquant_bits": 3, "kvquant_outlier_fraction": 0.0},
        ),
    ]

    all_results = []
    for label, method, overrides in methods:
        trial_results = []
        for trial in range(args.trials):
            cache_arg = None if method is None else build_cache(method, model, overrides)
            res = run_one(model, tokenizer, cache_arg, args.n_decode, label)
            trial_results.append(res)
            print(
                f"  {label} trial {trial + 1}: {res['tokens_per_sec']:.1f} tok/s, "
                f"peak {res['peak_memory_mb']:.0f} MB"
            )
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
        print(f"  {label} avg: {avg_tps:.1f} tok/s")

    fp16_entry = next((r for r in all_results if r["method"] == "fp16"), None)
    if fp16_entry:
        fp16_tps = fp16_entry["avg_tokens_per_sec"]
        for r in all_results:
            r["speedup_vs_fp16"] = (
                round(r["avg_tokens_per_sec"] / fp16_tps, 3) if fp16_tps else None
            )

    output = {
        "model": args.model,
        "chip": _chip_name(),
        "n_decode_tokens": args.n_decode,
        "nuq_vs_uniform_reconstruction_mse": recon,
        "results": all_results,
    }
    out_path = out_dir / "kvquant_benchmark.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
    _plot(all_results, args.model, out_dir)


def _plot(results: list, model_name: str, out_dir: Path) -> None:
    labels = [r["label"] for r in results]
    tps = [r["avg_tokens_per_sec"] for r in results]
    peak = [r["avg_peak_memory_mb"] for r in results]

    model_stem = Path(model_name).name
    fig_dir = out_dir / "figures" / "kvquant" / model_stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"KVQuant-NUQ KV Cache — {model_stem}", fontsize=13)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    axes[0].bar(labels, tps, color=colors[: len(labels)])
    axes[0].set_title("Throughput (tok/s)")
    axes[0].set_ylabel("Tokens / second")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, peak, color=colors[: len(labels)])
    axes[1].set_title("Peak Memory (MB)")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig_path = fig_dir / "kvquant_vs_baselines.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()

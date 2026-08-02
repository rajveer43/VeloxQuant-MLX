"""End-to-end PALU benchmark vs SVDq, KIVI, and fp16 baseline.

Measures throughput, peak memory, and — the headline claim — full-KV
compression at an equal-or-better effective bit budget than the keys-only
baselines, with both keys *and* values stored in low-rank latent form.

PALU (arXiv:2407.21118-adapted, ICLR 2025) is zero-calibration: the per-group
low-rank projections are fit online from the prefill batch via group-head SVD,
and the resulting latents are mixed-bit quantized. Unlike SVDq (keys-only,
values fp16), PALU stores BOTH tensors as low-rank latents and never
materialises full fp16 keys/values for storage.

Compared methods:
  - fp16_baseline        (standard mlx_lm KVCache)
  - kivi_2bit            (KIVI uniform 2-bit, the uniform reference)
  - svdq_1_25bit         (SVDq sub-2-bit keys, values fp16)
  - palu_lr_only         (PALU low-rank only, fp16 latents — pure rank win)
  - palu_lr_mixed        (PALU low-rank + mixed-bit, the headline config)
  - palu_aggressive      (PALU lower energy threshold, deeper compression)

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_palu.py \\
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


def reconstruction_mse_full_kv(seed: int = 0) -> dict:
    """Offline: PALU vs naive-2bit reconstruction MSE on low-rank K and V."""
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    rng = np.random.default_rng(seed)
    S, D, r = 256, 128, 16
    U = rng.standard_normal((S, r)).astype(np.float32)
    Wk = rng.standard_normal((r, D)).astype(np.float32)
    Wv = rng.standard_normal((r, D)).astype(np.float32)
    K = (U @ Wk + rng.standard_normal((S, D)) * 0.05).astype(np.float32)
    V = (U @ Wv + rng.standard_normal((S, D)) * 0.05).astype(np.float32)

    c = KVCacheFactory.create(
        KVCacheConfig(method="palu", head_dim=D, palu_rank=r, palu_n_head_groups=1)
    )
    ko, vo = c.update_and_fetch(mx.array(K[None, None]), mx.array(V[None, None]))
    mx.eval(ko, vo)

    def _mse(a, b):
        return float(mx.mean((a.astype(mx.float32) - mx.array(b)) ** 2).item())

    nk = _group_quant_dequant(mx.array(K), b=2, group_size=32)
    nv = _group_quant_dequant(mx.array(V), b=2, group_size=32)
    mx.eval(nk, nv)
    return {
        "key": {
            "palu_mse": round(_mse(ko[0, 0], K), 6),
            "naive_2bit_mse": round(_mse(nk, K), 6),
        },
        "value": {
            "palu_mse": round(_mse(vo[0, 0], V), 6),
            "naive_2bit_mse": round(_mse(nv, V), 6),
        },
        "effective_bits": round(c.assigned_avg_bits, 4),
    }


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

    recon = reconstruction_mse_full_kv()
    print(f"PALU full-KV reconstruction MSE: {recon}")

    methods = [
        ("fp16_baseline", None, {}),
        ("kivi_2bit", "kivi", {"bit_width_inlier": 2, "kivi_group_size": 32}),
        ("svdq_1_25bit", "svdq", {"svdq_energy_threshold": 0.95, "svdq_hi_fraction": 0.25}),
        ("palu_lr_only", "palu", {"palu_energy_threshold": 0.90, "palu_quantize_values": False}),
        ("palu_lr_mixed", "palu", {"palu_energy_threshold": 0.90, "palu_quantize_values": True}),
        ("palu_aggressive", "palu", {"palu_energy_threshold": 0.80, "palu_quantize_values": True}),
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
        "palu_full_kv_reconstruction_mse": recon,
        "results": all_results,
    }
    out_path = out_dir / "palu_benchmark.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
    _plot(all_results, args.model, out_dir)


def _plot(results: list, model_name: str, out_dir: Path) -> None:
    labels = [r["label"] for r in results]
    tps = [r["avg_tokens_per_sec"] for r in results]
    peak = [r["avg_peak_memory_mb"] for r in results]

    model_stem = Path(model_name).name
    fig_dir = out_dir / "figures" / "palu" / model_stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"PALU low-rank KV Cache — {model_stem}", fontsize=13)
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
    fig_path = fig_dir / "palu_vs_baselines.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()

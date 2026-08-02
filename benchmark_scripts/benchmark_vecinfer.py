"""End-to-end VecInfer benchmark against fp16 baseline on MLX models.

Compares throughput, peak memory, and KV-cache compression for three
VecInfer configurations against a vanilla fp16 KV cache. Saves a 4-panel
summary plot and a JSON dump under ``figures/vecinfer/<model-stem>/``.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py \\
        --model mlx-community/Llama-3.2-1B-Instruct-4bit

Calibration artifacts (smooth factors + codebooks) are cached under
``~/.cache/veloxquant/vecinfer/<model-id>/`` so reruns are fast. The
first run does a short calibration pass on synthetic activations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


PROMPTS = [
    "Explain why the sky appears blue in just one short sentence.",
    "List three reasons to prefer SQLite over Postgres for small apps.",
    "Write one line of Python that reverses a string named text.",
]


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


def _calibrate_artifacts(
    head_dim: int,
    n_heads: int,
    key_bits: int,
    value_bits: int,
    key_sub_dim: int,
    value_sub_dim: int,
    cache_dir: Path,
    seed: int = 42,
) -> dict:
    """Train codebooks + smooth factors on synthetic key/value samples.

    Real calibration would tap the model's actual key activations. For a
    benchmark that focuses on throughput/memory we just train on Gaussian
    samples shaped like the model's keys — codebook quality affects
    accuracy, not the metrics this script reports.
    """
    from veloxquant_mlx.allocators.vecinfer import (
        calibrate_smooth_factors,
        train_codebook,
    )

    sig = f"hd{head_dim}_h{n_heads}_kb{key_bits}_vb{value_bits}_ks{key_sub_dim}_vs{value_sub_dim}"
    cache_path = cache_dir / f"{sig}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "smooth": mx.array(data["smooth"]),
            "key_codebook": mx.array(data["key_cb"]),
            "value_codebook": mx.array(data["value_cb"]),
        }

    print(f"  [calib] training codebooks (sig={sig})...", flush=True)
    rng_np = np.random.default_rng(seed)
    n_samples = 4096
    K = mx.array(rng_np.standard_normal((n_samples, n_heads, head_dim)).astype(np.float32))
    V = mx.array(rng_np.standard_normal((n_samples, n_heads, head_dim)).astype(np.float32))
    smooth = calibrate_smooth_factors(K)

    # Pool sub-vectors across heads/tokens for codebook training.
    k_subs = mx.array(np.asarray(K).reshape(-1, key_sub_dim))
    v_subs = mx.array(np.asarray(V).reshape(-1, value_sub_dim))

    n_train = min(8000, k_subs.shape[0])
    key_cb = train_codebook(k_subs[:n_train], 2**key_bits, max_iter=15, seed=seed)
    val_cb = train_codebook(v_subs[:n_train], 2**value_bits, max_iter=15, seed=seed + 1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        smooth=np.asarray(smooth),
        key_cb=np.asarray(key_cb),
        value_cb=np.asarray(val_cb),
    )
    return {"smooth": smooth, "key_codebook": key_cb, "value_codebook": val_cb}


def _build_vecinfer_caches(
    model,
    key_bits: int,
    value_bits: int,
    key_sub_dim: int,
    value_sub_dim: int,
    artifacts: dict,
) -> list:
    """Construct one VecInferKVCache per attention layer."""
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
    from veloxquant_mlx import KVCacheConfig

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
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=key_sub_dim,
            value_sub_dim=value_sub_dim,
            key_codebook_bits=key_bits,
            value_codebook_bits=value_bits,
            smooth_factors=artifacts.get("smooth"),
            key_codebook=artifacts.get("key_codebook"),
            value_codebook=artifacts.get("value_codebook"),
            seed=42 + i,
        )
        caches.append(VecInferKVCache(cfg))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple:
    """Run a single prompt and return (tokens_generated, elapsed_seconds)."""
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
        # Older mlx_lm API
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else max_tokens
    return n_tok, elapsed


def _run_config(model, tokenizer, name: str, build_caches_fn, max_tokens: int) -> dict:
    print(f"\n--- {name} ---", flush=True)
    _reset_peak()
    total_tok = 0
    total_time = 0.0
    sample_cache = None

    for i, prompt in enumerate(PROMPTS):
        caches = build_caches_fn()
        if i == 0:
            sample_cache = caches
        n_tok, elapsed = _generate(model, tokenizer, prompt, max_tokens, caches)
        total_tok += n_tok
        total_time += elapsed
        print(
            f"  prompt {i}: {n_tok} tok in {elapsed:.2f}s ({n_tok / max(elapsed, 1e-6):.1f} tok/s)"
        )

    throughput = total_tok / max(total_time, 1e-6)
    peak_mb = _peak_mb()

    # Compression accounting from the first cache list
    key_compressed = key_fp16 = 0
    val_compressed = val_fp16 = 0
    avg_bits = 16.0
    if sample_cache is not None:
        for c in sample_cache:
            if hasattr(c, "compressed_key_bytes"):
                key_compressed += c.compressed_key_bytes
                key_fp16 += c.fp16_key_bytes
                val_compressed += getattr(c, "compressed_value_bytes", 0)
                val_fp16 += getattr(c, "fp16_value_bytes", 0)
        if hasattr(sample_cache[0], "assigned_avg_bits"):
            avg_bits = float(sample_cache[0].assigned_avg_bits)

    key_ratio = (key_fp16 / key_compressed) if key_compressed else 1.0
    val_ratio = (val_fp16 / val_compressed) if val_compressed else 1.0

    return {
        "name": name,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "key_compression": key_ratio,
        "value_compression": val_ratio,
        "avg_bits": avg_bits,
        "total_tokens": total_tok,
        "total_time_s": total_time,
    }


def _plot_summary(results: list[dict], out_path: Path, model_label: str) -> None:
    names = [r["name"] for r in results]
    throughputs = [r["throughput_tok_s"] for r in results]
    peaks = [r["peak_mb"] for r in results]
    key_ratios = [r["key_compression"] for r in results]
    avg_bits = [r["avg_bits"] for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"VecInfer benchmark — {model_label}", fontsize=14)

    colors = ["#666", "#00d4ff", "#7c3aed", "#ff6b35"][: len(names)]

    axes[0].bar(names, throughputs, color=colors)
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_title("Throughput")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, peaks, color=colors)
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("Peak memory")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(names, key_ratios, color=colors)
    axes[2].set_ylabel("Key compression (x)")
    axes[2].set_title("Key cache compression")
    axes[2].tick_params(axis="x", rotation=20)

    axes[3].bar(names, avg_bits, color=colors)
    axes[3].set_ylabel("Avg bits / element")
    axes[3].set_title("Effective bit-width")
    axes[3].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="VecInfer benchmark")
    parser.add_argument("--model", required=True, help="HF model id (mlx-community/...)")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument(
        "--output-dir", default=None, help="Defaults to figures/vecinfer/<model-stem>/"
    )
    args = parser.parse_args()

    from mlx_lm import load

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/vecinfer") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)

    # Resolve head_dim and n_heads from first attention layer
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
    n_heads = getattr(first_attn, "n_heads", None) or getattr(margs, "num_attention_heads", 1)
    print(f"  head_dim={head_dim}, n_heads={n_heads}")

    cache_root = Path(os.path.expanduser("~/.cache/veloxquant/vecinfer")) / model_stem

    # Three VecInfer configs (key bits / sub_dim / value bits / sub_dim)
    configs = [
        ("vecinfer-2bit", 8, 4, 8, 4),  # 2 bits/elem
        ("vecinfer-1.5bit", 12, 8, 12, 8),
        ("vecinfer-1bit", 8, 8, 8, 8),  # 1 bit/elem
    ]

    results = []

    # fp16 baseline
    results.append(
        _run_config(
            model,
            tokenizer,
            "fp16-baseline",
            lambda: _build_fp16_caches(model),
            args.max_tokens,
        )
    )

    for cfg_name, kb, ksd, vb, vsd in configs:
        artifacts = _calibrate_artifacts(
            head_dim=head_dim,
            n_heads=n_heads,
            key_bits=kb,
            value_bits=vb,
            key_sub_dim=ksd,
            value_sub_dim=vsd,
            cache_dir=cache_root,
        )
        results.append(
            _run_config(
                model,
                tokenizer,
                cfg_name,
                lambda kb=kb, ksd=ksd, vb=vb, vsd=vsd, art=artifacts: _build_vecinfer_caches(
                    model, kb, vb, ksd, vsd, art
                ),
                args.max_tokens,
            )
        )

    summary_path = out_dir / "vecinfer_summary.png"
    _plot_summary(results, summary_path, model_stem)

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "prompts": PROMPTS,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSummary: {summary_path}")
    print(f"Results: {json_path}")
    print("\nFinal:")
    for r in results:
        print(
            f"  {r['name']:<20s} {r['throughput_tok_s']:6.1f} tok/s  "
            f"{r['peak_mb']:7.1f} MB  "
            f"key_x={r['key_compression']:.2f}  avg_bits={r['avg_bits']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

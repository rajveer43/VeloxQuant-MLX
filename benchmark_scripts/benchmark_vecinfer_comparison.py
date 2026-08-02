"""Comparative KV-cache benchmark: fp16 vs TurboQuant vs RVQ vs VecInfer.

Runs one prompt at ``--max-tokens`` against each of 8 cache configurations
on a single MLX model. Saves a 4-panel summary plot plus ``results.json``
under ``figures/vecinfer/<model-stem>/``.

Configurations
--------------
* ``fp16-baseline`` — uncompressed reference
* ``TQ-{2,3,4}bit`` — TurboQuantProd (rotation + Lloyd-Max + QJL residual)
* ``RVQ-{1,2}bit`` — TurboQuantRVQ (two-pass residual VQ)
* ``VecInfer-{1,2}bit`` — Smooth + Hadamard + product VQ (this PR)

Usage
-----
::

    PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer_comparison.py \\
        --model mlx-community/Llama-3.1-8B-Instruct-4bit --max-tokens 200

Pass ``--models a,b,c`` to sweep multiple models in one process. After all
runs, a cross-model summary is written to
``figures/vecinfer/_summary/cross_model_comparison.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


PALETTE = {
    "fp16-baseline": "#4C72B0",
    "TQ-2bit": "#C44E52",
    "TQ-3bit": "#DD8452",
    "TQ-4bit": "#55A868",
    "RVQ-2bit": "#8172B2",
    "RVQ-1bit": "#CCB974",
    "VecInfer-2bit": "#00d4ff",
    "VecInfer-1bit": "#7c3aed",
}

DEFAULT_MODELS = [
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Falcon3-7B-Instruct-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/Phi-4-4bit",
    "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",
]

PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _peak_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Cache builders — one factory per configuration
# --------------------------------------------------------------------------
def _model_head_info(model):
    """Return (head_dim, n_kv_heads, n_heads, n_layers) by inspecting layers."""
    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    head_dim = n_kv = n_heads = None
    for L in layers:
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            continue
        head_dim = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        n_kv = getattr(attn, "n_kv_heads", None) or getattr(
            args, "num_key_value_heads", getattr(args, "num_attention_heads", 1)
        )
        n_heads = getattr(attn, "n_heads", None) or getattr(args, "num_attention_heads", n_kv)
        break
    return head_dim, n_kv, n_heads, len(layers)


def _build_fp16(model):
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FB() for _ in layers]


def _build_tq(model, bits: int):
    """TurboQuantProd (single-pass) cache list."""
    from benchmark_scripts.benchmark_core import TurboQuantMLXKVCache  # noqa
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, L in enumerate(layers):
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        n_kv = getattr(attn, "n_kv_heads", None) or getattr(args, "num_key_value_heads", 1)
        if hd is None:
            caches.append(_FB())
            continue
        caches.append(TurboQuantMLXKVCache(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=42 + i))
    return caches


def _build_rvq(model, bits: int):
    from veloxquant_mlx import KVCacheConfig, KVCacheFactory
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, L in enumerate(layers):
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None:
            caches.append(_FB())
            continue
        cfg = KVCacheConfig(
            method="turboquant_rvq",
            head_dim=hd,
            bit_width_inlier=bits,
            seed=42 + i,
        )
        caches.append(KVCacheFactory.create(cfg))
    return caches


def _vecinfer_artifacts(
    head_dim, n_heads, key_bits, value_bits, key_sub_dim, value_sub_dim, cache_dir, seed=42
):
    from veloxquant_mlx.allocators.vecinfer import (
        calibrate_smooth_factors,
        train_codebook,
    )

    sig = f"hd{head_dim}_h{n_heads}_kb{key_bits}_vb{value_bits}_ks{key_sub_dim}_vs{value_sub_dim}"
    path = cache_dir / f"{sig}.npz"
    if path.exists():
        data = np.load(path)
        return {
            "smooth": mx.array(data["smooth"]),
            "key_codebook": mx.array(data["key_cb"]),
            "value_codebook": mx.array(data["value_cb"]),
        }
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((4096, n_heads, head_dim)).astype(np.float32))
    V = mx.array(rng.standard_normal((4096, n_heads, head_dim)).astype(np.float32))
    smooth = calibrate_smooth_factors(K)
    k_subs = mx.array(np.asarray(K).reshape(-1, key_sub_dim))
    v_subs = mx.array(np.asarray(V).reshape(-1, value_sub_dim))
    n_train = min(8000, k_subs.shape[0])
    key_cb = train_codebook(k_subs[:n_train], 2**key_bits, max_iter=15, seed=seed)
    val_cb = train_codebook(v_subs[:n_train], 2**value_bits, max_iter=15, seed=seed + 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, smooth=np.asarray(smooth), key_cb=np.asarray(key_cb), value_cb=np.asarray(val_cb)
    )
    return {"smooth": smooth, "key_codebook": key_cb, "value_codebook": val_cb}


def _build_vecinfer(model, key_bits, value_bits, key_sub_dim, value_sub_dim, model_stem):
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
    from mlx_lm.models.cache import KVCache as _FB

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)
    head_dim, n_kv, n_heads, _ = _model_head_info(model)
    cache_root = Path(os.path.expanduser("~/.cache/veloxquant/vecinfer")) / model_stem
    art = _vecinfer_artifacts(
        head_dim,
        n_heads,
        key_bits,
        value_bits,
        key_sub_dim,
        value_sub_dim,
        cache_root,
    )

    caches = []
    for i, L in enumerate(layers):
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None) or (
            args.hidden_size // args.num_attention_heads if args else None
        )
        if hd is None or hd % key_sub_dim != 0 or hd % value_sub_dim != 0:
            caches.append(_FB())
            continue
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=key_sub_dim,
            value_sub_dim=value_sub_dim,
            key_codebook_bits=key_bits,
            value_codebook_bits=value_bits,
            smooth_factors=art["smooth"],
            key_codebook=art["key_codebook"],
            value_codebook=art["value_codebook"],
            seed=42 + i,
        )
        caches.append(VecInferKVCache(cfg))
    return caches


# --------------------------------------------------------------------------
# Per-config run
# --------------------------------------------------------------------------
def _run_one(model, tokenizer, label: str, builder: Callable, max_tokens: int) -> dict:
    import mlx_lm

    print(f"  [{label}] generating...", flush=True)
    try:
        caches = builder()
    except Exception as e:
        print(f"    builder failed: {e}")
        return {
            "name": label,
            "error": f"builder: {e}",
            "throughput_tok_s": 0.0,
            "peak_mb": float("nan"),
            "key_compression": 0.0,
            "tokens_generated": 0,
            "elapsed_s": 0.0,
            "key_kb": 0.0,
        }

    messages = [{"role": "user", "content": PROMPT}]
    try:
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = PROMPT

    _reset_peak()
    mx.clear_cache()

    t0 = time.perf_counter()
    try:
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt_txt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=caches,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "name": label,
            "error": str(e),
            "throughput_tok_s": 0.0,
            "peak_mb": float("nan"),
            "key_compression": 0.0,
            "tokens_generated": 0,
            "elapsed_s": 0.0,
            "key_kb": 0.0,
        }
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(response)) if response else 0

    kc = kf = 0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            kc += c.compressed_key_bytes
            kf += c.fp16_key_bytes
    ratio = (kf / kc) if kc else 1.0

    peak_mb = _peak_mb()
    print(
        f"    {n_tok} tok in {elapsed:.1f}s ({n_tok / max(elapsed, 1e-6):.1f} tok/s) "
        f"peak={peak_mb:.0f}MB key_x={ratio:.2f}"
    )
    return {
        "name": label,
        "throughput_tok_s": n_tok / max(elapsed, 1e-6),
        "peak_mb": peak_mb,
        "key_compression": ratio,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "key_kb": kc / 1024.0,
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _plot_model_summary(
    results: list, out_path: Path, model_label: str, head_dim: int, n_configs: int
):
    names = [r["name"] for r in results]
    compress = [r["key_compression"] for r in results]
    tput = [r["throughput_tok_s"] for r in results]
    toks = [r["tokens_generated"] for r in results]
    key_kb = [r["key_kb"] for r in results]
    colors = [PALETTE.get(n, "#999") for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"VecInfer comparative study — {model_label}\n"
        f"head_dim={head_dim} · {n_configs} configs · Apple Silicon MLX",
        fontsize=14,
        fontweight="bold",
    )

    def _bar(ax, vals, title, ylabel, hline=None, fmt=".1f"):
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.7)
        vmax = max(vals) if max(vals) > 0 else 1.0
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + vmax * 0.02,
                f"{v:{fmt}}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylim(0, vmax * 1.28)
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0, 0], compress, "Key Compression Ratio", "Ratio (x)", hline=1.0, fmt=".2f")
    _bar(
        axes[0, 1],
        tput,
        "Generation Throughput",
        "Tokens / second",
        hline=tput[0] if tput else None,
    )
    _bar(axes[1, 0], toks, "Tokens Generated", "Tokens", fmt="d")
    _bar(axes[1, 1], key_kb, "Compressed Key Cache Size", "KB", fmt=".0f")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def _plot_cross_model(per_model: dict, out_path: Path):
    """Side-by-side comparison across all models for the key metrics."""
    if not per_model:
        return
    config_names = sorted(
        {r["name"] for results in per_model.values() for r in results},
        key=lambda n: list(PALETTE).index(n) if n in PALETTE else 99,
    )
    models = list(per_model.keys())

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(models) * 2.4), 11))
    fig.suptitle(
        "Cross-model comparison · key compression and throughput", fontsize=15, fontweight="bold"
    )

    n_cfg = len(config_names)
    width = 0.85 / n_cfg
    x = np.arange(len(models))

    for ax_idx, (metric, ylabel, fmt) in enumerate(
        [
            ("key_compression", "Key compression (x)", ".2f"),
            ("throughput_tok_s", "Throughput (tok/s)", ".0f"),
        ]
    ):
        ax = axes[ax_idx]
        for i, cfg in enumerate(config_names):
            vals = []
            for m in models:
                hit = next((r for r in per_model[m] if r["name"] == cfg), None)
                vals.append(hit[metric] if hit else 0.0)
            ax.bar(
                x + (i - n_cfg / 2 + 0.5) * width,
                vals,
                width,
                color=PALETTE.get(cfg, "#999"),
                label=cfg,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                m.split("/")[-1].replace("-Instruct", "").replace("-4bit", "").replace("-Chat", "")
                for m in models
            ],
            fontsize=9,
            rotation=15,
            ha="right",
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9, ncol=min(n_cfg, 4), loc="upper left")
            ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _run_model(model_id: str, max_tokens: int) -> Optional[list]:
    from mlx_lm import load

    model_stem = model_id.split("/")[-1]
    out_dir = Path("figures/vecinfer") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\nLoading {model_id}...\n{'=' * 68}", flush=True)
    try:
        model, tokenizer = load(model_id)
    except Exception as e:
        print(f"  failed to load: {e}")
        return None

    head_dim, n_kv, n_heads, n_layers = _model_head_info(model)
    print(f"  head_dim={head_dim} n_kv_heads={n_kv} n_q_heads={n_heads} n_layers={n_layers}")

    # Pick a VecInfer sub_dim that divides head_dim
    vec_ksd = 4 if head_dim % 4 == 0 else (2 if head_dim % 2 == 0 else 1)
    vec_vsd = vec_ksd

    configs = [
        ("fp16-baseline", lambda: _build_fp16(model)),
        ("TQ-2bit", lambda: _build_tq(model, 2)),
        ("TQ-3bit", lambda: _build_tq(model, 3)),
        ("TQ-4bit", lambda: _build_tq(model, 4)),
        ("RVQ-2bit", lambda: _build_rvq(model, 2)),
        ("RVQ-1bit", lambda: _build_rvq(model, 1)),
        (
            "VecInfer-2bit",
            lambda: _build_vecinfer(
                model,
                key_bits=8,
                value_bits=8,
                key_sub_dim=vec_ksd,
                value_sub_dim=vec_vsd,
                model_stem=model_stem,
            ),
        ),
        (
            "VecInfer-1bit",
            lambda: _build_vecinfer(
                model,
                key_bits=8,
                value_bits=8,
                key_sub_dim=8 if head_dim % 8 == 0 else vec_ksd,
                value_sub_dim=8 if head_dim % 8 == 0 else vec_vsd,
                model_stem=model_stem,
            ),
        ),
    ]

    results = []
    for label, builder in configs:
        try:
            r = _run_one(model, tokenizer, label, builder, max_tokens)
        except Exception as e:
            traceback.print_exc()
            r = {
                "name": label,
                "error": str(e),
                "throughput_tok_s": 0.0,
                "peak_mb": float("nan"),
                "key_compression": 0.0,
                "tokens_generated": 0,
                "elapsed_s": 0.0,
                "key_kb": 0.0,
            }
        results.append(r)

    summary_path = out_dir / "comparison_summary.png"
    _plot_model_summary(results, summary_path, model_stem, head_dim, len(configs))
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": model_id,
                "head_dim": head_dim,
                "n_kv_heads": n_kv,
                "n_q_heads": n_heads,
                "n_layers": n_layers,
                "max_tokens": max_tokens,
                "prompt": PROMPT,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n  Summary: {summary_path}")
    print(f"  Results: {json_path}")
    # Release model memory before next iteration
    del model
    del tokenizer
    mx.clear_cache()
    return results


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None, help="Single HF model id (mutually exclusive with --models)"
    )
    parser.add_argument("--models", default=None, help="Comma-separated list of HF model ids")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument(
        "--use-defaults", action="store_true", help="Run on the built-in DEFAULT_MODELS list"
    )
    args = parser.parse_args()

    if args.model and args.models:
        print("error: pass --model OR --models, not both")
        return 2
    if args.use_defaults:
        targets = DEFAULT_MODELS
    elif args.models:
        targets = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        targets = [args.model]
    else:
        targets = [DEFAULT_MODELS[0]]

    print(f"Running comparison on {len(targets)} model(s):")
    for m in targets:
        print(f"  - {m}")

    per_model: dict = {}
    for m in targets:
        res = _run_model(m, args.max_tokens)
        if res is not None:
            per_model[m] = res

    if len(per_model) > 1:
        out = Path("figures/vecinfer/_summary/cross_model_comparison.png")
        _plot_cross_model(per_model, out)
        rollup = Path("figures/vecinfer/_summary/results_all.json")
        rollup.parent.mkdir(parents=True, exist_ok=True)
        with open(rollup, "w") as f:
            json.dump(
                {
                    "max_tokens": args.max_tokens,
                    "per_model": {k: v for k, v in per_model.items()},
                },
                f,
                indent=2,
            )
        print(f"\nCross-model summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

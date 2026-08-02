"""Text-model KV cache sweep across multiple models and configs.

For each (model, config) we spawn a fresh Python subprocess so MLX's compiled
graph cache is invalidated between runs (same isolation pattern as
benchmark_qwen2_vl.py). The child loads the model, runs exactly one config on
a shared text prompt, writes a JSON result, and exits. The parent aggregates
all JSON results into a comparison figure + table.

Configs:
  fp16   — baseline (mlx_lm's standard KVCache)
  rvq1   — TurboQuant RVQ 1-bit (~7.5× key compression at d=128)
  rvq2   — TurboQuant RVQ 2-bit (~3.9× key compression at d=128)
  tq4    — TurboQuant Prod 4-bit single-pass (~4.3× key compression)

Models (default set): mistral7b, qwen3_4b, qwen3_8b, phi4, llama31_8b,
                       falcon3_7b, gemma3_4b, qwen25_32b.
Select a subset with --models. Use --configs to restrict the config set.

Usage:
    # Full sweep — caches subprocess output JSON under .bench_tmp/sweep_*.json
    python3 benchmark_scripts/run_text_sweep.py

    # Single model + 2 configs:
    python3 benchmark_scripts/run_text_sweep.py --models mistral7b --configs fp16,rvq1

    # Child mode (used internally; safe to call manually for debugging):
    python3 benchmark_scripts/run_text_sweep.py --run \\
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --config rvq1 --output /tmp/out.json --label "Mistral 7B"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_scripts.benchmark_core import (
    TurboQuantMLXKVCache,
    TurboQuantRVQMLXKVCache,
)

MODEL_REGISTRY = {
    "mistral7b": ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral 7B v0.3"),
    "qwen3_4b": ("mlx-community/Qwen3-4B-4bit", "Qwen3 4B"),
    "qwen3_8b": ("mlx-community/Qwen3-8B-4bit", "Qwen3 8B"),
    "phi4": ("mlx-community/Phi-4-4bit", "Phi-4"),
    "llama31_8b": ("mlx-community/Llama-3.1-8B-Instruct-4bit", "Llama 3.1 8B"),
    "falcon3_7b": ("mlx-community/Falcon3-7B-Instruct-4bit", "Falcon3 7B"),
    "gemma3_4b": ("mlx-community/gemma-3-4b-it-4bit", "Gemma3 4B"),
    "qwen25_32b": ("mlx-community/Qwen2.5-32B-Instruct-4bit", "Qwen2.5 32B"),
}
DEFAULT_MODELS = list(MODEL_REGISTRY.keys())
CONFIG_ORDER = ["fp16", "rvq1", "rvq2", "tq4"]
CONFIG_LABELS = {
    "fp16": "fp16",
    "rvq1": "RVQ 1-bit",
    "rvq2": "RVQ 2-bit",
    "tq4": "TQ 4-bit",
}
PALETTE = {
    "fp16": "#4C72B0",
    "rvq1": "#8172B2",
    "rvq2": "#C44E52",
    "tq4": "#55A868",
}
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
    "updated_tests",
    "text_sweep",
)
TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".bench_tmp",
)


# ── Cache construction ────────────────────────────────────────────────────────


def _build_caches(model, n_layers, head_dim, n_kv_heads, config_key):
    """Return a list of caches matching `model.layers` length."""
    if config_key == "fp16":
        return None
    seeds = list(range(n_layers))
    if config_key == "rvq1":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i)
            for i in seeds
        ]
    if config_key == "rvq2":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=2, seed=i)
            for i in seeds
        ]
    if config_key == "tq4":
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=4, seed=i)
            for i in seeds
        ]
    raise ValueError(f"Unknown config: {config_key}")


# ── Child entrypoint ──────────────────────────────────────────────────────────


def run_single(model_id: str, label: str, config_key: str) -> dict:
    print(f"[child:{label}:{config_key}] loading {model_id} ...", flush=True)
    model, tokenizer = mlx_lm.load(model_id)
    cfg = model.args
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    print(
        f"[child:{label}:{config_key}] layers={n_layers} head_dim={head_dim} n_kv_heads={n_kv}",
        flush=True,
    )

    layers = getattr(model, "layers", None) or model.model.layers
    if not hasattr(model, "make_cache"):
        from mlx_lm.models.cache import KVCache

        def _default_make_cache():
            return [KVCache() for _ in layers]

        model.make_cache = _default_make_cache

    factory_caches = _build_caches(model, n_layers, head_dim, n_kv, config_key)
    injected: list = []
    if factory_caches is not None:

        def _patch(*_, **__):
            c = _build_caches(model, n_layers, head_dim, n_kv, config_key)
            injected.extend(c)
            return c

        model.make_cache = _patch

    messages = [{"role": "user", "content": PROMPT}]
    try:
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = PROMPT

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt_txt,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    tq_caches = [
        c for c in injected if isinstance(c, (TurboQuantMLXKVCache, TurboQuantRVQMLXKVCache))
    ]
    kf = sum(c.fp16_key_bytes for c in tq_caches)
    kc = sum(c.compressed_key_bytes for c in tq_caches)
    ratio_str = f"{kf / kc:.2f}×" if kc > 0 else "—"
    ratio_num = (kf / kc) if kc > 0 else 1.0
    toks = len(tokenizer.encode(response))
    tps = toks / elapsed if elapsed > 0 else 0.0

    snippet = response[:240] + ("..." if len(response) > 240 else "")
    print(
        f"[child:{label}:{config_key}] {ratio_str} | {toks} tokens "
        f"| {elapsed:.1f}s | {tps:.1f} tok/s",
        flush=True,
    )
    print(f"[child:{label}:{config_key}] response: {snippet}", flush=True)

    return {
        "model_key": label,
        "model_id": model_id,
        "config": config_key,
        "config_label": CONFIG_LABELS[config_key],
        "head_dim": head_dim,
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "tps": tps,
        "toks": toks,
        "elapsed": elapsed,
        "ratio_str": ratio_str,
        "ratio_num": ratio_num,
        "response": response,
    }


# ── Parent dispatch ───────────────────────────────────────────────────────────


def _spawn(model_id: str, label: str, config_key: str, out_json: str) -> bool:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--run",
        "--model",
        model_id,
        "--label",
        label,
        "--config",
        config_key,
        "--output",
        out_json,
    ]
    print(f"\n{'=' * 70}\n[parent] {label} / {config_key}\n{'=' * 70}", flush=True)
    proc = subprocess.run(cmd, env=os.environ.copy())
    return proc.returncode == 0


def run_sweep(model_keys: list[str], config_keys: list[str]) -> dict:
    os.makedirs(TMP_DIR, exist_ok=True)
    results: dict = {}  # results[model_key][config_key] = dict
    for mkey in model_keys:
        if mkey not in MODEL_REGISTRY:
            print(f"[parent] unknown model '{mkey}', skipping", flush=True)
            continue
        model_id, _label = MODEL_REGISTRY[mkey]
        results[mkey] = {}
        for ckey in config_keys:
            out_json = os.path.join(TMP_DIR, f"sweep_{mkey}_{ckey}.json")
            ok = _spawn(model_id, mkey, ckey, out_json)
            if not ok or not os.path.exists(out_json):
                print(f"[parent] {mkey}/{ckey} failed; skipping", flush=True)
                continue
            with open(out_json) as f:
                results[mkey][ckey] = json.load(f)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_summary(results: dict) -> None:
    print(f"\n\n{'=' * 86}")
    print(f"{'TEXT-MODEL SWEEP SUMMARY':^86}")
    print(f"{'=' * 86}")
    header = f"{'Model':<15} {'Config':<11} {'tok/s':>8} {'tokens':>8} {'compression':>13} {'vs fp16':>10}"
    print(header)
    print("-" * 86)
    for mkey, by_cfg in results.items():
        fp16 = by_cfg.get("fp16", {}).get("tps", 0.0)
        for ckey in CONFIG_ORDER:
            r = by_cfg.get(ckey)
            if r is None:
                continue
            rel = f"{r['tps'] / fp16:.2f}×" if fp16 > 0 else "—"
            print(
                f"{mkey:<15} {r['config_label']:<11} "
                f"{r['tps']:>8.1f} {r['toks']:>8} "
                f"{r['ratio_str']:>13} {rel:>10}"
            )
        print("-" * 86)


def save_figures(results: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_keys = [m for m in results if results[m]]
    if not model_keys:
        return
    cfg_keys = [c for c in CONFIG_ORDER if any(c in results[m] for m in model_keys)]

    # Throughput grouped bars
    x = np.arange(len(model_keys))
    width = 0.85 / max(len(cfg_keys), 1)
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(model_keys) + 2), 5))
    for j, c in enumerate(cfg_keys):
        vals = [results[m].get(c, {}).get("tps", 0.0) for m in model_keys]
        ax.bar(
            x + j * width - 0.42,
            vals,
            width=width,
            label=CONFIG_LABELS[c],
            color=PALETTE[c],
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=15)
    ax.set_ylabel("tok/s")
    ax.legend()
    ax.set_title("KV Cache Throughput Sweep — by Model and Config")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/sweep_throughput.png", dpi=150)
    plt.close()

    # Token completeness
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(model_keys) + 2), 5))
    for j, c in enumerate(cfg_keys):
        vals = [results[m].get(c, {}).get("toks", 0) for m in model_keys]
        ax.bar(
            x + j * width - 0.42,
            vals,
            width=width,
            label=CONFIG_LABELS[c],
            color=PALETTE[c],
            edgecolor="white",
        )
    ax.axhline(MAX_TOKENS, ls="--", color="grey", lw=1, label=f"Target ({MAX_TOKENS})")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=15)
    ax.set_ylabel("Tokens generated")
    ax.legend()
    ax.set_title("KV Cache Token Completeness — by Model and Config")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/sweep_completeness.png", dpi=150)
    plt.close()

    # Relative throughput vs fp16
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(model_keys) + 2), 5))
    for j, c in enumerate(cfg_keys):
        vals = []
        for m in model_keys:
            fp = results[m].get("fp16", {}).get("tps", 0.0)
            v = results[m].get(c, {}).get("tps", 0.0)
            vals.append(v / fp if fp > 0 else 0.0)
        ax.bar(
            x + j * width - 0.42,
            vals,
            width=width,
            label=CONFIG_LABELS[c],
            color=PALETTE[c],
            edgecolor="white",
        )
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=15)
    ax.set_ylabel("relative throughput")
    ax.legend()
    ax.set_title("KV Cache Throughput vs fp16 — by Model and Config")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/sweep_relative.png", dpi=150)
    plt.close()

    # Compression
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(model_keys) + 2), 5))
    cfg_quant = [c for c in cfg_keys if c != "fp16"]
    for j, c in enumerate(cfg_quant):
        vals = [results[m].get(c, {}).get("ratio_num", 0.0) for m in model_keys]
        ax.bar(
            x + j * (0.85 / max(len(cfg_quant), 1)) - 0.42,
            vals,
            width=0.85 / max(len(cfg_quant), 1),
            label=CONFIG_LABELS[c],
            color=PALETTE[c],
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=15)
    ax.set_ylabel("Compression ratio (×)")
    ax.legend()
    ax.set_title("Key Compression Ratio — by Model and Config")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/sweep_compression.png", dpi=150)
    plt.close()

    # Markdown-style table dump
    md_path = f"{out_dir}/sweep_table.md"
    with open(md_path, "w") as f:
        f.write("| Model | Config | tok/s | Tokens | Compression | vs fp16 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in model_keys:
            fp16_tps = results[m].get("fp16", {}).get("tps", 0.0)
            for c in cfg_keys:
                r = results[m].get(c)
                if r is None:
                    continue
                rel = f"{r['tps'] / fp16_tps:.2f}×" if fp16_tps > 0 else "—"
                f.write(
                    f"| {m} | {r['config_label']} | "
                    f"{r['tps']:.1f} | {r['toks']} | "
                    f"{r['ratio_str']} | {rel} |\n"
                )
    print(f"Figures saved to {out_dir}/")


# ── Entry ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated subset. Available: {','.join(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--configs", default=None, help=f"Comma-separated subset of {','.join(CONFIG_ORDER)}"
    )
    # Child mode
    p.add_argument(
        "--run", action="store_true", help="Child mode: run a single (model, config) and write JSON"
    )
    p.add_argument("--model", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--config", default=None, choices=CONFIG_ORDER)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.run:
        if not (args.model and args.config and args.output and args.label):
            print("--run requires --model --label --config --output", file=sys.stderr)
            sys.exit(2)
        res = run_single(args.model, args.label, args.config)
        with open(args.output, "w") as f:
            json.dump(res, f)
        return

    model_keys = args.models.split(",") if args.models else DEFAULT_MODELS
    config_keys = args.configs.split(",") if args.configs else CONFIG_ORDER
    results = run_sweep(model_keys, config_keys)
    print_summary(results)
    save_figures(results, OUT_DIR)
    # Save aggregated JSON
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

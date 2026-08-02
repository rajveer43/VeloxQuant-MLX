"""Benchmark VeloxQuant-MLX KV cache compression on Qwen2-VL.

Tests TurboQuant RVQ 2-bit, TurboQuant 4-bit, and TurboQuant 2-bit
against fp16 baseline on a text-only prompt via the VLM model wrapper.

Qwen2-VL in mlx_lm strips the visual encoder and delegates all generation
to the underlying Qwen2 language model — the KV cache interface is identical
to text-only models. This benchmark validates that our cache injection path
works through the VLM wrapper.

Model: mlx-community/Qwen2-VL-7B-Instruct-bf16 (or 2B for 8GB Macs)

The parent process spawns one fresh Python subprocess per config so MLX's
compiled-graph cache is invalidated between runs. Each child loads the model,
runs exactly one config, writes a JSON result, and exits. Use --run-config to
invoke the child mode directly.

Usage:
    # Run the full suite (parent spawns one subprocess per config):
    python3 benchmark_scripts/benchmark_qwen2_vl.py

    # Run a single config in this process (child mode used internally):
    python3 benchmark_scripts/benchmark_qwen2_vl.py \\
        --run-config rvq2 --model <id> --output /tmp/result.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_scripts.benchmark_core import (
    TurboQuantMLXKVCache,
    TurboQuantRVQMLXKVCache,
    build_vlm_caches,
)

DEFAULT_MODEL = "mlx-community/Qwen2-VL-7B-Instruct-bf16"
MAX_TOKENS = 200
PROMPT = (
    "Describe how vision transformers process images and explain "
    "the role of attention mechanisms in capturing spatial relationships."
)
CONFIG_LABELS = {
    "fp16": "fp16 baseline",
    "rvq2": "RVQ 2-bit",
    "tq4": "TurboQuant 4-bit",
    "tq2": "TurboQuant 2-bit",
}
CONFIG_ORDER = ["fp16", "rvq2", "tq4", "tq2"]
PALETTE = {
    "fp16": "#4C72B0",
    "rvq2": "#C44E52",
    "tq4": "#55A868",
    "tq2": "#DD8452",
}
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
    "updated_tests",
    "qwen2_vl",
)


# ── Prompt construction ───────────────────────────────────────────────────────


def _build_prompt(tokenizer, text):
    """Apply the model's chat template — Qwen2-VL needs the <|im_start|>... wrapper
    for the language model to know it should answer rather than continue text."""
    try:
        messages = [{"role": "user", "content": text}]
        result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return result or text
    except Exception:
        return text


def _make_factory(model, config_key):
    if config_key == "fp16":
        return None
    if config_key == "rvq2":
        return lambda: build_vlm_caches(model, bits=2, use_rvq=True)
    if config_key == "tq4":
        return lambda: build_vlm_caches(model, bits=4, use_rvq=False)
    if config_key == "tq2":
        return lambda: build_vlm_caches(model, bits=2, use_rvq=False)
    raise ValueError(f"Unknown config key: {config_key}")


def run_single(model_id: str, config_key: str) -> dict:
    """Load the model fresh and run exactly one config. Used by --run-config."""
    print(f"[child:{config_key}] Loading {model_id} ...", flush=True)
    model, tokenizer = mlx_lm.load(model_id)

    layers = getattr(model, "layers", None) or model.model.layers
    head_dim, n_kv_heads = None, None
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is not None:
            head_dim = getattr(attn, "head_dim", None) or head_dim
            n_kv_heads = getattr(attn, "n_kv_heads", None) or n_kv_heads
            if head_dim is not None and n_kv_heads is not None:
                break
    print(
        f"[child:{config_key}] layers={len(layers)} head_dim={head_dim} n_kv_heads={n_kv_heads}",
        flush=True,
    )

    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in layers]

        model.make_cache = _default_make_cache

    factory = _make_factory(model, config_key)
    injected: list = []
    if factory is not None:

        def _patch(*_, **__):
            c = factory()
            injected.extend(c)
            return c

        model.make_cache = _patch

    prompt_txt = _build_prompt(tokenizer, PROMPT)

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt_txt,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    tq = [c for c in injected if isinstance(c, (TurboQuantMLXKVCache, TurboQuantRVQMLXKVCache))]
    kf = sum(c.fp16_key_bytes for c in tq)
    kc = sum(c.compressed_key_bytes for c in tq)
    ratio = f"{kf / kc:.2f}×" if kc > 0 else "—"
    toks = len(tokenizer.encode(response))
    tps = toks / elapsed if elapsed > 0 else 0.0

    print(
        f"[child:{config_key}] {ratio} | {toks} tokens | {elapsed:.1f}s | {tps:.1f} tok/s",
        flush=True,
    )
    print(
        f"[child:{config_key}] response: {response[:300]}{'...' if len(response) > 300 else ''}",
        flush=True,
    )

    return {
        "config": config_key,
        "label": CONFIG_LABELS[config_key],
        "tps": tps,
        "toks": toks,
        "elapsed": elapsed,
        "ratio": ratio,
        "response": response,
    }


# ── Figures ────────────────────────────────────────────────────────────────────


def save_figures(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    labels = [k for k in CONFIG_ORDER if k in results]
    tps_vals = [results[k]["tps"] for k in labels]
    toks_vals = [results[k]["toks"] for k in labels]
    colors = [PALETTE.get(k, "#888888") for k in labels]
    disp = [results[k]["label"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(disp, tps_vals, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("tok/s")
    ax.set_title("Qwen2-VL — Throughput by KV Config")
    ax.set_ylim(0, max(tps_vals + [1]) * 1.2)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/throughput.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(disp, toks_vals, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.axhline(MAX_TOKENS, ls="--", color="grey", lw=1, label=f"Target ({MAX_TOKENS})")
    ax.set_ylabel("Tokens generated")
    ax.set_title("Qwen2-VL — Output Completeness")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{out_dir}/completeness.png", dpi=150)
    plt.close()

    fp16_tps = results.get("fp16", {}).get("tps", 1.0) or 1.0
    rel_vals = [v / fp16_tps for v in tps_vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(disp, rel_vals, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f×", padding=3, fontsize=9)
    ax.axhline(1.0, ls="--", color="grey", lw=1, label="fp16 baseline")
    ax.set_ylabel("Relative throughput")
    ax.set_title("Qwen2-VL — Throughput vs fp16")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{out_dir}/relative_throughput.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    header = ["Config", "tok/s", "Tokens", "Compression", "vs fp16"]
    table_data = []
    for k in labels:
        r = results[k]
        rel = f"{r['tps'] / fp16_tps:.2f}×" if fp16_tps > 0 else "—"
        table_data.append([r["label"], f"{r['tps']:.1f}", str(r["toks"]), r["ratio"], rel])
    tbl = ax.table(cellText=table_data, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax.set_title("Qwen2-VL KV Cache Benchmark Summary", pad=12)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/summary_table.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to {out_dir}/")


# ── Parent dispatch ───────────────────────────────────────────────────────────


def _spawn_child(model_id: str, config_key: str, out_json: str) -> bool:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--run-config",
        config_key,
        "--model",
        model_id,
        "--output",
        out_json,
    ]
    print(f"\n{'=' * 64}\n[parent] spawning {config_key} → {out_json}\n{'=' * 64}", flush=True)
    proc = subprocess.run(cmd, env=os.environ.copy())
    if proc.returncode != 0:
        print(f"[parent] {config_key} child exited with code {proc.returncode}", flush=True)
        return False
    return True


def run_benchmark(model_id: str, configs: list[str] | None = None) -> dict:
    keys = configs or CONFIG_ORDER
    results: dict = {}
    tmpdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".bench_tmp",
    )
    os.makedirs(tmpdir, exist_ok=True)

    for key in keys:
        out_json = os.path.join(tmpdir, f"qwen2_vl_{key}.json")
        ok = _spawn_child(model_id, key, out_json)
        if not ok or not os.path.exists(out_json):
            print(f"[parent] skipping {key} — no result file", flush=True)
            continue
        with open(out_json) as f:
            results[key] = json.load(f)

    print(f"\n{'=' * 64}")
    print(f"SUMMARY — {model_id}")
    print(f"{'=' * 64}")
    print(f"{'Config':<24}  {'tok/s':>7}  {'tokens':>7}  {'compression':>13}")
    print(f"{'-' * 64}")
    for key in CONFIG_ORDER:
        if key not in results:
            continue
        r = results[key]
        print(f"{r['label']:<24}  {r['tps']:>7.1f}  {r['toks']:>7}  {r['ratio']:>13}")

    if results:
        save_figures(results, OUT_DIR)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VeloxQuant VLM benchmark (Qwen2-VL)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--run-config",
        default=None,
        choices=CONFIG_ORDER,
        help="Child mode: run exactly this one config and write JSON",
    )
    parser.add_argument(
        "--output", default=None, help="(child mode) write result JSON to this path"
    )
    parser.add_argument(
        "--configs", default=None, help="Comma-separated subset of configs to run (parent mode)"
    )
    args = parser.parse_args()

    if args.run_config is not None:
        if args.output is None:
            print("--output required with --run-config", file=sys.stderr)
            sys.exit(2)
        result = run_single(args.model, args.run_config)
        with open(args.output, "w") as f:
            json.dump(result, f)
        return

    cfg_list = args.configs.split(",") if args.configs else None
    run_benchmark(args.model, cfg_list)


if __name__ == "__main__":
    main()

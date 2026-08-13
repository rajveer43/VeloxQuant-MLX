"""Full benchmark report generator — 8 models × 6 configs.

Produces 6 publication-quality figures per model under
figures/2026-05-12/<model_folder>/ using the v3 benchmark pipeline that
includes RVQ 1-bit alongside fp16 / TQ 2-3-4-bit / RVQ 2-bit.

Architecture
────────────
Parent process iterates models × configs, spawning one fresh Python
subprocess per (model, config). This guarantees MLX's compiled-graph cache
is clean for every run — the graph-reuse bug that corrupted quantized outputs
when multiple configs ran back-to-back in the same process is fully avoided.

Each child:
  1. Loads the model fresh
  2. Runs mlx_lm.generate() with the appropriate KV-cache wrapper
  3. Writes a compact JSON to .bench_tmp/full_<model>_<config>.json
  4. Exits

The parent then collects all 6 JSONs for a model, calls
benchmark_core.run_benchmark_v3_from_results(), which recomputes synthetic
quality curves and generates the 6 PNGs.

Idempotency
───────────
• If figures/2026-05-12/<folder>/fig6_full_report.png already exists and
  --force is not set, that model is skipped entirely.
• If .bench_tmp/full_<model>_<config>.json already exists (from a prior
  interrupted run) and --force is not set, that child is not re-spawned.
  This lets you resume a 90-minute sweep after a crash.

Usage
─────
  # Full sweep (all 8 models, all 6 configs):
  python3 benchmark_scripts/run_full_reports.py

  # Subset:
  python3 benchmark_scripts/run_full_reports.py --models mistral7b,qwen3_4b

  # Specific configs only:
  python3 benchmark_scripts/run_full_reports.py --configs fp16,rvq1,rvq2

  # Re-run even if outputs exist:
  python3 benchmark_scripts/run_full_reports.py --force

  # Child mode (internal use / debugging):
  python3 benchmark_scripts/run_full_reports.py --run \\
      --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
      --model-key mistral7b --config rvq1 \\
      --output /tmp/out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import mlx.core as mx
import mlx_lm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_scripts.benchmark_core import (
    TurboQuantMLXKVCache,
    TurboQuantRVQMLXKVCache,
    _MLXKVCache,
    _read_model_cfg,
    run_benchmark_v3_from_results,
)

# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gemma4": ("mlx-community/gemma-3-4b-it-4bit", "Gemma3 4B"),
    "qwen3_4b": ("mlx-community/Qwen3-4B-4bit", "Qwen3 4B"),
    "mistral7b": ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral 7B v0.3"),
    "falcon3_7b": ("mlx-community/Falcon3-7B-Instruct-4bit", "Falcon3 7B"),
    "phi4": ("mlx-community/Phi-4-4bit", "Phi-4"),
    "qwen3_8b": ("mlx-community/Qwen3-8B-4bit", "Qwen3 8B"),
    "llama31_8b": ("mlx-community/Llama-3.1-8B-Instruct-4bit", "Llama 3.1 8B"),
    "qwen25_32b": ("mlx-community/Qwen2.5-32B-Instruct-4bit", "Qwen2.5 32B"),
}
DEFAULT_MODEL_ORDER = list(MODEL_REGISTRY.keys())

CONFIG_ORDER = ["fp16", "tq2", "tq3", "tq4", "rvq2", "rvq1"]
CONFIG_LABELS = {
    "fp16": "fp16 baseline",
    "tq2": "TurboQuant 2-bit",
    "tq3": "TurboQuant 3-bit",
    "tq4": "TurboQuant 4-bit",
    "rvq2": "RVQ 2-bit ★",
    "rvq1": "RVQ 1-bit ★",
}

PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200

OUT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
    "2026-05-12",
)
TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".bench_tmp",
)


# ── Cache factory ─────────────────────────────────────────────────────────────


def _make_caches(config_key, n_layers, head_dim, n_kv_heads):
    if config_key == "fp16":
        return None
    seeds = list(range(n_layers))
    if config_key == "tq2":
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=2, seed=i)
            for i in seeds
        ]
    if config_key == "tq3":
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=3, seed=i)
            for i in seeds
        ]
    if config_key == "tq4":
        return [
            TurboQuantMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=4, seed=i)
            for i in seeds
        ]
    if config_key == "rvq2":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=2, seed=i)
            for i in seeds
        ]
    if config_key == "rvq1":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i)
            for i in seeds
        ]
    raise ValueError(f"Unknown config: {config_key}")


# ── Child mode ────────────────────────────────────────────────────────────────


def run_single(model_id: str, model_key: str, config_key: str) -> dict:
    tag = f"{model_key}/{config_key}"
    print(f"[{tag}] loading {model_id} ...", flush=True)
    model, tokenizer = mlx_lm.load(model_id)

    head_dim, n_kv_heads, n_layers = _read_model_cfg(model)
    print(f"[{tag}] layers={n_layers} head_dim={head_dim} n_kv_heads={n_kv_heads}", flush=True)

    # Ensure model.make_cache exists
    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            return [_MLXKVCache() for _ in layers]

        model.make_cache = _default_make_cache

    injected: list = []
    cache_list = _make_caches(config_key, n_layers, head_dim, n_kv_heads)
    if cache_list is not None:

        def _patch(*_, **__):
            c = _make_caches(config_key, n_layers, head_dim, n_kv_heads)
            injected.extend(c)
            return c

        model.make_cache = _patch

    # Build prompt
    try:
        prompt_txt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
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

    tq = [c for c in injected if isinstance(c, (TurboQuantMLXKVCache, TurboQuantRVQMLXKVCache))]
    fp16_key_bytes = sum(c.fp16_key_bytes for c in tq)
    compressed_key_bytes = sum(c.compressed_key_bytes for c in tq)
    ratio_num = fp16_key_bytes / compressed_key_bytes if compressed_key_bytes > 0 else 1.0
    ratio_str = f"{ratio_num:.2f}×" if compressed_key_bytes > 0 else "—"
    toks = len(tokenizer.encode(response))
    tps = toks / elapsed if elapsed > 0 else 0.0

    print(f"[{tag}] {ratio_str} | {toks} tokens | {elapsed:.1f}s | {tps:.1f} tok/s", flush=True)
    print(f"[{tag}] → {response[:200]}{'...' if len(response) > 200 else ''}", flush=True)

    return {
        "model_key": model_key,
        "model_id": model_id,
        "config": config_key,
        "config_label": CONFIG_LABELS[config_key],
        "head_dim": head_dim,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "tps": tps,
        "toks": toks,
        "elapsed": elapsed,
        "ratio_str": ratio_str,
        "ratio_num": ratio_num,
        "fp16_key_bytes": fp16_key_bytes,
        "compressed_key_bytes": compressed_key_bytes,
        "response": response,
    }


# ── Parent helpers ────────────────────────────────────────────────────────────


def _tmp_path(model_key: str, config_key: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, f"full_{model_key}_{config_key}.json")


def _spawn(model_id: str, model_key: str, config_key: str, out_json: str) -> bool:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--run",
        "--model",
        model_id,
        "--model-key",
        model_key,
        "--config",
        config_key,
        "--output",
        out_json,
    ]
    print(f"\n{'─' * 70}\n[parent] {model_key} / {config_key}\n{'─' * 70}", flush=True)
    proc = subprocess.run(cmd, env=os.environ.copy())
    return proc.returncode == 0


def run_model(model_key: str, config_keys: list[str], force: bool) -> bool:
    """Run all configs for one model; generate figures. Returns True on success."""
    model_id, model_label = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(OUT_BASE, model_key)
    sentinel = os.path.join(out_dir, "fig6_full_report.png")

    if not force and os.path.exists(sentinel):
        print(
            f"\n[parent] {model_key}: already done ({sentinel}). Use --force to re-run.", flush=True
        )
        return True

    # Spawn one child per config
    for ckey in config_keys:
        tmp = _tmp_path(model_key, ckey)
        if not force and os.path.exists(tmp):
            print(f"[parent] {model_key}/{ckey}: cached ({tmp})", flush=True)
            continue
        ok = _spawn(model_id, model_key, ckey, tmp)
        if not ok:
            print(f"[parent] {model_key}/{ckey}: child failed", flush=True)

    # Collect results
    results: dict[str, dict] = {}
    for ckey in config_keys:
        tmp = _tmp_path(model_key, ckey)
        if not os.path.exists(tmp):
            continue
        try:
            with open(tmp) as f:
                results[ckey] = json.load(f)
        except Exception as e:
            print(f"[parent] could not read {tmp}: {e}", flush=True)

    if not results:
        print(f"[parent] {model_key}: no results collected — skipping figures", flush=True)
        return False

    # Infer model dims from any available result
    sample = next(iter(results.values()))
    head_dim = sample["head_dim"]
    n_kv = sample["n_kv_heads"]
    n_layers = sample["n_layers"]

    run_benchmark_v3_from_results(
        results_by_config=results,
        out_dir=out_dir,
        model_label=model_label,
        head_dim=head_dim,
        n_kv_heads=n_kv,
        n_layers=n_layers,
    )
    return True


# ── Global summary table ──────────────────────────────────────────────────────


def print_global_summary(model_keys: list[str], config_keys: list[str]) -> None:
    print(f"\n\n{'═' * 90}")
    print(f"{'GLOBAL SWEEP SUMMARY — TurboQuant KV Cache v3':^90}")
    print(f"{'═' * 90}")
    header = (
        f"{'Model':<14} {'Config':<12} {'tok/s':>8} "
        f"{'tokens':>8} {'compression':>13} {'vs fp16':>10}"
    )
    print(header)
    print("─" * 90)

    for mkey in model_keys:
        if mkey not in MODEL_REGISTRY:
            continue
        fp16_tps = None
        for ckey in config_keys:
            tmp = _tmp_path(mkey, ckey)
            if not os.path.exists(tmp):
                continue
            try:
                with open(tmp) as f:
                    r = json.load(f)
            except Exception:
                continue
            if ckey == "fp16":
                fp16_tps = r["tps"]
            rel = f"{r['tps'] / fp16_tps:.2f}×" if fp16_tps and fp16_tps > 0 else "—"
            print(
                f"  {mkey:<12} {r['config_label']:<12} "
                f"{r['tps']:>8.1f} {r['toks']:>8} "
                f"{r['ratio_str']:>13} {rel:>10}"
            )
        print("─" * 90)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Run full 6-config v3 benchmark report for 8 text models"
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys "
        f"(default: all {len(MODEL_REGISTRY)}). "
        f"Available: {','.join(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--configs",
        default=None,
        help=f"Comma-separated configs (default: all 6). Available: {','.join(CONFIG_ORDER)}",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if outputs already exist")

    # Child mode arguments
    p.add_argument(
        "--run", action="store_true", help="Child mode: run a single (model, config) and write JSON"
    )
    p.add_argument("--model", default=None, help="(child) HF model id")
    p.add_argument("--model-key", default=None, dest="model_key", help="(child) short model key")
    p.add_argument("--config", default=None, choices=CONFIG_ORDER, help="(child) config key")
    p.add_argument("--output", default=None, help="(child) output JSON path")

    args = p.parse_args()

    # ── Child mode ────────────────────────────────────────────────────────────
    if args.run:
        missing = [
            n
            for n in ("model", "model_key", "config", "output")
            if not getattr(args, n.replace("-", "_"))
        ]
        if missing:
            print(f"--run requires: {', '.join('--' + m for m in missing)}", file=sys.stderr)
            sys.exit(2)
        result = run_single(args.model, args.model_key, args.config)
        with open(args.output, "w") as f:
            json.dump(result, f)
        return

    # ── Parent mode ───────────────────────────────────────────────────────────
    model_keys = args.models.split(",") if args.models else DEFAULT_MODEL_ORDER
    config_keys = args.configs.split(",") if args.configs else CONFIG_ORDER

    unknown_m = [m for m in model_keys if m not in MODEL_REGISTRY]
    unknown_c = [c for c in config_keys if c not in CONFIG_ORDER]
    if unknown_m:
        print(f"Unknown models: {unknown_m}. Available: {list(MODEL_REGISTRY)}", file=sys.stderr)
        sys.exit(1)
    if unknown_c:
        print(f"Unknown configs: {unknown_c}. Available: {CONFIG_ORDER}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_BASE, exist_ok=True)
    print(
        f"\nStarting v3 sweep: {len(model_keys)} models × {len(config_keys)} configs"
        f" = {len(model_keys) * len(config_keys)} child processes"
    )
    print(f"Output base: {OUT_BASE}\n")

    for mkey in model_keys:
        run_model(mkey, config_keys, args.force)

    print_global_summary(model_keys, config_keys)
    print(f"\nAll figures in: {OUT_BASE}/")


if __name__ == "__main__":
    main()

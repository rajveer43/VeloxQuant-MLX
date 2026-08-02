"""RateQuant V2 trial runner — improved per-paper implementation.

Runs a 4-config sweep against a small set of models (default: falcon3_7b,
gemma4) to validate the V2 improvements:

  fp16        — baseline
  rvq1        — TurboQuantRVQ 1-bit (the prior best)
  rvq1o       — RVQ 1-bit + Outlier-Token side buffer (already validated)
  rvqrqv2     — RVQ + RateQuant V2 (activation-sensitivity allocation)

Output: figures/<today>/<model_name>/ with the same 6-figure layout as
the existing outlier_ratequant pipeline.

Subprocess isolation: one child per (model, config) to avoid the MLX
graph-reuse bug. Each child loads the model fresh, runs one config,
writes a JSON to .bench_tmp/rqv2_<model>_<config>.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import mlx_lm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_scripts.benchmark_core import (
    TurboQuantRVQMLXKVCache,
    _MLXKVCache,
    _read_model_cfg,
)
from benchmark_scripts.outlier_ratequant_core import (
    OutlierTokenRVQMLXKVCache,
    run_outlier_ratequant_v4_from_results,
)
from benchmark_scripts.ratequant_v2_core import (
    RateQuantV2RVQMLXKVCache,
    allocate_bits_ratequant_v2,
    calibrate_layer_sensitivities,
)


# ── Registry: trial set is intentionally small ────────────────────────────────

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "falcon3_7b": ("mlx-community/Falcon3-7B-Instruct-4bit", "Falcon3 7B"),
    "gemma4": ("mlx-community/gemma-3-4b-it-4bit", "Gemma3 4B"),
}
DEFAULT_MODEL_ORDER = list(MODEL_REGISTRY.keys())

CONFIG_ORDER = ["fp16", "rvq1", "rvq1o", "rvqrqv2"]
CONFIG_LABELS = {
    "fp16": "fp16 baseline",
    "rvq1": "RVQ 1-bit",
    "rvq1o": "RVQ 1-bit + Outlier",
    "rvqrqv2": "RVQ + RateQuant V2",
}
# Align with the v4 figure pipeline's expected keys
CONFIG_TO_V4_KEY = {
    "fp16": "fp16",
    "rvq1": "rvq1",
    "rvq1o": "rvq1o",
    "rvqrqv2": "rvqrq",
}

PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200

TODAY = datetime.date.today().isoformat()
OUT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
    TODAY,
)
TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".bench_tmp",
)


# ── Cache factory ─────────────────────────────────────────────────────────────


def _make_caches(
    config_key: str,
    model,
    tokenizer,
    n_layers: int,
    head_dim: int,
    n_kv_heads: int,
    ratequant_target: float,
    sigma_k: float,
):
    if config_key == "fp16":
        return None
    seeds = list(range(n_layers))
    if config_key == "rvq1":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i)
            for i in seeds
        ]
    if config_key == "rvq1o":
        return [
            OutlierTokenRVQMLXKVCache(
                n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i, sigma_k=sigma_k
            )
            for i in seeds
        ]
    if config_key == "rvqrqv2":
        print(f"[rqv2] running calibration ({n_layers} layers)...", flush=True)
        weights = calibrate_layer_sensitivities(
            model,
            tokenizer,
            n_sequences=8,
            seq_len=256,
            verbose=True,
        )
        alloc = allocate_bits_ratequant_v2(
            sensitivities=weights,
            target_avg_bits=ratequant_target,
            beta=3.5,
            bit_choices=(1, 2, 3),
            verbose=True,
        )
        print(f"[rqv2] per-layer bit allocation: {alloc}", flush=True)
        return [
            RateQuantV2RVQMLXKVCache(
                n_kv_heads=n_kv_heads, head_dim=head_dim, bits=alloc[i], seed=i
            )
            for i in seeds
        ]
    raise ValueError(f"Unknown config: {config_key}")


# ── Child mode ────────────────────────────────────────────────────────────────


def run_single(
    model_id: str, model_key: str, config_key: str, ratequant_target: float, sigma_k: float
) -> dict:
    tag = f"{model_key}/{config_key}"
    print(f"[{tag}] loading {model_id} ...", flush=True)
    model, tokenizer = mlx_lm.load(model_id)

    head_dim, n_kv_heads, n_layers = _read_model_cfg(model)
    print(f"[{tag}] layers={n_layers} head_dim={head_dim} n_kv_heads={n_kv_heads}", flush=True)

    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    if not hasattr(model, "make_cache"):

        def _default_make_cache():
            return [_MLXKVCache() for _ in layers]

        model.make_cache = _default_make_cache

    # Build caches BEFORE generation. For rvqrqv2 the calibration happens here.
    injected: list = []
    cache_list = _make_caches(
        config_key, model, tokenizer, n_layers, head_dim, n_kv_heads, ratequant_target, sigma_k
    )
    if cache_list is not None:
        # The first build was for calibration; on each generate() call mlx_lm
        # will request a fresh cache list, so we patch make_cache to return
        # a NEW set with the same allocation.
        bit_alloc = [getattr(c, "assigned_bits", getattr(c, "_bits", 0)) for c in cache_list]

        def _patch(*_, **__):
            c = _make_caches_post_calib(
                config_key, n_layers, head_dim, n_kv_heads, sigma_k, bit_alloc
            )
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

    # Aggregate byte accounting
    quant = [
        c
        for c in injected
        if isinstance(
            c, (TurboQuantRVQMLXKVCache, OutlierTokenRVQMLXKVCache, RateQuantV2RVQMLXKVCache)
        )
    ]
    fp16_key_bytes = sum(c.fp16_key_bytes for c in quant)
    compressed_key_bytes = sum(c.compressed_key_bytes for c in quant)
    ratio_num = fp16_key_bytes / compressed_key_bytes if compressed_key_bytes > 0 else 1.0
    ratio_str = f"{ratio_num:.2f}×" if compressed_key_bytes > 0 else "—"
    toks = len(tokenizer.encode(response))
    tps = toks / elapsed if elapsed > 0 else 0.0

    outlier_fraction = 0.0
    o_caches = [c for c in injected if isinstance(c, OutlierTokenRVQMLXKVCache)]
    if o_caches:
        outlier_fraction = sum(c.outlier_fraction for c in o_caches) / len(o_caches)

    rq_caches = [c for c in injected if isinstance(c, RateQuantV2RVQMLXKVCache)]
    avg_bits = sum(c.assigned_bits for c in rq_caches) / len(rq_caches) if rq_caches else 0.0
    bit_distribution = [c.assigned_bits for c in rq_caches] if rq_caches else []

    print(
        f"[{tag}] {ratio_str} | {toks} tokens | {elapsed:.1f}s | {tps:.1f} tok/s"
        + (f" | outlier {outlier_fraction * 100:.2f}%" if o_caches else "")
        + (f" | b̄={avg_bits:.2f}" if rq_caches else ""),
        flush=True,
    )
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
        "outlier_fraction": outlier_fraction,
        "ratequant_avg_bits": avg_bits,
        "bit_distribution": bit_distribution,
        "response": response,
    }


def _make_caches_post_calib(
    config_key: str,
    n_layers: int,
    head_dim: int,
    n_kv_heads: int,
    sigma_k: float,
    bit_alloc: list[int],
):
    """Re-build cache list using a pre-computed bit allocation (no re-calib)."""
    seeds = list(range(n_layers))
    if config_key == "rvq1":
        return [
            TurboQuantRVQMLXKVCache(n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i)
            for i in seeds
        ]
    if config_key == "rvq1o":
        return [
            OutlierTokenRVQMLXKVCache(
                n_kv_heads=n_kv_heads, head_dim=head_dim, bits=1, seed=i, sigma_k=sigma_k
            )
            for i in seeds
        ]
    if config_key == "rvqrqv2":
        return [
            RateQuantV2RVQMLXKVCache(
                n_kv_heads=n_kv_heads, head_dim=head_dim, bits=bit_alloc[i], seed=i
            )
            for i in seeds
        ]
    raise ValueError(f"Unknown: {config_key}")


# ── Parent helpers ────────────────────────────────────────────────────────────


def _tmp_path(model_key: str, config_key: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, f"rqv2_{model_key}_{config_key}.json")


def _spawn(
    model_id: str,
    model_key: str,
    config_key: str,
    out_json: str,
    ratequant_target: float,
    sigma_k: float,
) -> bool:
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
        "--ratequant-target",
        str(ratequant_target),
        "--sigma-k",
        str(sigma_k),
    ]
    print(f"\n{'─' * 70}\n[parent] {model_key} / {config_key}\n{'─' * 70}", flush=True)
    proc = subprocess.run(cmd, env=os.environ.copy())
    return proc.returncode == 0


def run_model(
    model_key: str, config_keys: list[str], force: bool, ratequant_target: float, sigma_k: float
) -> bool:
    model_id, model_label = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(OUT_BASE, model_key)
    sentinel = os.path.join(out_dir, "fig6_full_report.png")

    if not force and os.path.exists(sentinel):
        print(
            f"\n[parent] {model_key}: already done ({sentinel}). Use --force to re-run.", flush=True
        )
        return True

    for ckey in config_keys:
        tmp = _tmp_path(model_key, ckey)
        if not force and os.path.exists(tmp):
            print(f"[parent] {model_key}/{ckey}: cached ({tmp})", flush=True)
            continue
        ok = _spawn(model_id, model_key, ckey, tmp, ratequant_target, sigma_k)
        if not ok:
            print(f"[parent] {model_key}/{ckey}: child failed", flush=True)

    # Collect results, remapped to v4 figure-pipeline keys
    results: dict[str, dict] = {}
    for ckey in config_keys:
        tmp = _tmp_path(model_key, ckey)
        if not os.path.exists(tmp):
            continue
        try:
            with open(tmp) as f:
                d = json.load(f)
            results[CONFIG_TO_V4_KEY[ckey]] = d
        except Exception as e:
            print(f"[parent] could not read {tmp}: {e}", flush=True)

    if not results:
        print(f"[parent] {model_key}: no results — skipping figures", flush=True)
        return False

    sample = next(iter(results.values()))
    run_outlier_ratequant_v4_from_results(
        results_by_config=results,
        out_dir=out_dir,
        model_label=f"{model_label} (RateQuant V2)",
        head_dim=sample["head_dim"],
        n_kv_heads=sample["n_kv_heads"],
        n_layers=sample["n_layers"],
        ratequant_target=ratequant_target,
    )
    return True


def print_global_summary(model_keys: list[str], config_keys: list[str]) -> None:
    print(f"\n\n{'═' * 94}")
    print(f"{'GLOBAL SUMMARY — RateQuant V2 Trial':^94}")
    print(f"{'═' * 94}")
    header = (
        f"{'Model':<14} {'Config':<24} {'tok/s':>8} "
        f"{'tokens':>8} {'compression':>13} {'vs fp16':>10} "
        f"{'extra':>14}"
    )
    print(header)
    print("─" * 94)
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
            extra = ""
            if ckey == "rvq1o":
                extra = f"outlier {r.get('outlier_fraction', 0.0) * 100:.1f}%"
            elif ckey == "rvqrqv2":
                extra = f"b̄ = {r.get('ratequant_avg_bits', 0.0):.2f}"
            print(
                f"  {mkey:<12} {r['config_label']:<24} "
                f"{r['tps']:>8.1f} {r['toks']:>8} "
                f"{r['ratio_str']:>13} {rel:>10} "
                f"{extra:>14}"
            )
        print("─" * 94)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="RateQuant V2 trial — 4 configs on falcon3_7b & gemma4")
    p.add_argument(
        "--models",
        default=None,
        help=f"Default: all trial models. Available: {','.join(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--configs", default=None, help=f"Default: all 4. Available: {','.join(CONFIG_ORDER)}"
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--ratequant-target", type=float, default=1.5)
    p.add_argument("--sigma-k", type=float, default=2.5)

    p.add_argument("--run", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--model-key", default=None, dest="model_key")
    p.add_argument("--config", default=None, choices=CONFIG_ORDER)
    p.add_argument("--output", default=None)

    args = p.parse_args()

    if args.run:
        missing = [
            n
            for n in ("model", "model_key", "config", "output")
            if not getattr(args, n.replace("-", "_"))
        ]
        if missing:
            print(f"--run requires: {', '.join('--' + m for m in missing)}", file=sys.stderr)
            sys.exit(2)
        result = run_single(
            args.model, args.model_key, args.config, args.ratequant_target, args.sigma_k
        )
        with open(args.output, "w") as f:
            json.dump(result, f)
        return

    model_keys = args.models.split(",") if args.models else DEFAULT_MODEL_ORDER
    config_keys = args.configs.split(",") if args.configs else CONFIG_ORDER

    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"\nRateQuant V2 trial: {len(model_keys)} models × {len(config_keys)} configs")
    print(f"Output base: {OUT_BASE}")
    print(f"RateQuant target b̄={args.ratequant_target}, outlier σ-k={args.sigma_k}\n")

    for mkey in model_keys:
        run_model(mkey, config_keys, args.force, args.ratequant_target, args.sigma_k)

    print_global_summary(model_keys, config_keys)
    print(f"\nAll figures in: {OUT_BASE}/")


if __name__ == "__main__":
    main()

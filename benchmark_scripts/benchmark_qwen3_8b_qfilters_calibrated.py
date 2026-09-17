"""Follow-up to benchmark_qwen3_8b_qfilters_honest.py: does real calibration fix it?

The honest QFilters post found a fully reproducible generation collapse
(repeated "is is is" / "====" tokens) once the cache exceeded its eviction
budget, using the *fallback* filter path -- direction estimated from
observed keys, sign ambiguous, because the cache never sees queries.

calibrate_qwen3_8b_qfilters.py has since produced a real calibration
artifact using this repo's qfilters_calibration module: the paper's actual
mechanism (query-SVD, sign-fixed via Theorem 3.3, averaged across GQA
groups). This script reruns the identical long-prompt protocol with FOUR
arms instead of three -- fp16, fallback (uncalibrated), and calibrated,
each further split by the Metal kernel toggle where relevant -- to see
whether real calibration actually fixes the collapse, partially helps, or
makes no difference at this budget.

Only the long prompt is run here (short stayed under budget and never
evicted in the original post, so it isn't where the failure mode lives).

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_qfilters_calibrated.py \\
        --model mlx-community/Qwen3-8B-4bit \\
        --calibration figures/qwen3_8b_qfilters_calibrated/qfilters_qwen3_8b.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_PASSAGE = (
    "The key-value cache stores the attention keys and values of every past "
    "token so the model need not recompute them. Its size grows linearly with "
    "context length and, on Apple Silicon unified memory, it competes with the "
    "model weights and the operating system for the same pool. "
)

LONG_PROMPT = (_PASSAGE * 40) + (
    "\n\nGiven the passage above, explain in simple terms why the KV cache "
    "is the binding memory constraint for long-context inference on Apple "
    "Silicon, covering both the linear growth and the unified-memory "
    "contention."
)


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


def _hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if chip:
            info["chip"] = chip
        if mem:
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
    except Exception:
        pass
    return info


def _gpu_contention_check() -> list[str]:
    suspects = []
    try:
        out = subprocess.run(["ps", "-Ao", "comm"], capture_output=True, text=True, timeout=5).stdout
        keywords = ("python", "ollama", "mlx", "llama", "lmstudio")
        for line in out.splitlines():
            low = line.lower()
            if any(k in low for k in keywords) and "benchmark_qwen3_8b" not in low:
                suspects.append(line.strip())
    except Exception:
        pass
    return suspects


def _build_qfilters_caches(model, budget: int, seed: int, use_metal: bool, filters_per_layer):
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None) or model.model.args

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        hd = getattr(attn, "head_dim", None) if attn is not None else None
        if hd is None:
            hd = args.hidden_size // args.num_attention_heads
        cfg = KVCacheConfig(
            method="qfilters",
            head_dim=hd,
            seed=seed + i,
            qfilters_budget=budget,
            use_metal_kernels=use_metal,
        )
        f = filters_per_layer[i] if filters_per_layer is not None else None
        caches.append(QFiltersKVCache(cfg, filters=f))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple[str, int, float]:
    from mlx_lm import generate

    t0 = time.time()
    out = generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, prompt_cache=caches,
    )
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return out, n_tok, elapsed


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _run_once(model, tokenizer, arm: str, prompt: str, max_tokens: int, budget: int, seed: int,
              calibrated_filters) -> dict:
    _reset_peak()
    if arm == "fp16":
        caches = _build_fp16_caches(model)
    elif arm == "fallback_off":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=False, filters_per_layer=None)
    elif arm == "fallback_on":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=True, filters_per_layer=None)
    elif arm == "calibrated_off":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=False, filters_per_layer=calibrated_filters)
    elif arm == "calibrated_on":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=True, filters_per_layer=calibrated_filters)
    else:
        raise ValueError(arm)

    text, n_tok, elapsed = _generate(model, tokenizer, prompt, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    kept_bytes = full_bytes = 0
    for c in caches:
        if hasattr(c, "qfilters_kept_bytes"):
            kept_bytes += c.qfilters_kept_bytes
            full_bytes += c.full_seq_bytes
    compression = (full_bytes / kept_bytes) if kept_bytes else 1.0

    return {
        "arm": arm, "text": text, "tokens_generated": n_tok, "elapsed_s": elapsed,
        "throughput_tok_s": throughput, "peak_mb": peak_mb, "compression_ratio": compression,
    }


def _summarize(runs: list[dict]) -> dict:
    tputs = [r["throughput_tok_s"] for r in runs]
    peaks = [r["peak_mb"] for r in runs]
    return {
        "n_repeats": len(runs),
        "throughput_median": statistics.median(tputs),
        "throughput_min": min(tputs),
        "throughput_max": max(tputs),
        "peak_mb_median": statistics.median(peaks),
        "peak_mb_min": min(peaks),
        "peak_mb_max": max(peaks),
        "compression_ratio": runs[-1]["compression_ratio"],
        "texts": [r["text"] for r in runs],
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Calibrated vs fallback QFilters on Qwen3-8B (long prompt)")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--calibration", default="figures/qwen3_8b_qfilters_calibrated/qfilters_qwen3_8b.npz")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_qfilters_calibrated") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = _hardware()
    contenders = _gpu_contention_check()
    print(f"hardware={hw}")
    if contenders:
        print("WARNING: other GPU/LLM-adjacent processes detected -- results may be contended:")
        for c in contenders:
            print(f"    {c}")
    else:
        print("No obvious contending processes detected.")

    print(f"Loading {args.model}...", flush=True)
    from mlx_lm import load

    model, tokenizer = load(args.model)

    from veloxquant_mlx.quantizers.qfilters_calibration import load_qfilters

    calibration = load_qfilters(args.calibration, expect_model_id=args.model)
    print(f"Loaded calibration: n_layers={calibration.n_layers} model_id={calibration.model_id}")

    layers = getattr(model, "layers", None) or model.model.layers
    margs = getattr(model, "args", None) or model.model.args
    n_layers = len(layers)
    n_kv_heads = getattr(margs, "num_key_value_heads", None) or getattr(margs, "num_attention_heads", 1)
    head_dim = getattr(margs, "head_dim", None) or (margs.hidden_size // margs.num_attention_heads)
    print(f"  n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim} budget={args.budget}")

    arms = ["fp16", "fallback_off", "fallback_on", "calibrated_off", "calibrated_on"]
    collected: dict[str, list[dict]] = {a: [] for a in arms}
    for rep in range(args.repeats):
        for arm in arms:
            print(f"  rep {rep + 1}/{args.repeats} arm={arm}", flush=True)
            collected[arm].append(_run_once(
                model, tokenizer, arm, LONG_PROMPT, args.max_tokens, args.budget, args.seed,
                calibration.filters,
            ))

    summaries = {arm: _summarize(runs) for arm, runs in collected.items()}

    fp16_text = summaries["fp16"]["texts"][0]

    payload = {
        "model": args.model, "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "qfilters_budget": args.budget, "max_tokens": args.max_tokens, "repeats": args.repeats,
        "calibration_artifact": args.calibration, "calibration_dataset": calibration.dataset,
        "hardware": hw, "gpu_contention_suspects": contenders,
        "prompt_tokens": len(tokenizer.encode(LONG_PROMPT)),
        "summaries": summaries,
        "fallback_on_vs_off_identical_text": (
            set(summaries["fallback_off"]["texts"]) == set(summaries["fallback_on"]["texts"])
            and len(set(summaries["fallback_off"]["texts"])) == 1
        ),
        "calibrated_on_vs_off_identical_text": (
            set(summaries["calibrated_off"]["texts"]) == set(summaries["calibrated_on"]["texts"])
            and len(set(summaries["calibrated_off"]["texts"])) == 1
        ),
        "fp16_vs_calibrated_word_overlap": _word_overlap(fp16_text, summaries["calibrated_on"]["texts"][0]),
        "fp16_vs_fallback_word_overlap": _word_overlap(fp16_text, summaries["fallback_on"]["texts"][0]),
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults: {json_path}")
    for arm, s in summaries.items():
        print(
            f"  {arm:<16s} tok/s median={s['throughput_median']:.2f} "
            f"min={s['throughput_min']:.2f} max={s['throughput_max']:.2f}  "
            f"peak median={s['peak_mb_median']:.0f}MB  compression_x={s['compression_ratio']:.2f}"
        )
    print(f"\nfallback on vs off identical text: {payload['fallback_on_vs_off_identical_text']}")
    print(f"calibrated on vs off identical text: {payload['calibrated_on_vs_off_identical_text']}")
    print(f"fp16 vs fallback word overlap: {payload['fp16_vs_fallback_word_overlap']:.3f}")
    print(f"fp16 vs calibrated word overlap: {payload['fp16_vs_calibrated_word_overlap']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

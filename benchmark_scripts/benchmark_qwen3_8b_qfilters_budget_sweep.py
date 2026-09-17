"""Sweep qfilters_budget to find where QFilters output stops being coherent.

Third post in the QFilters sub-series (honest benchmark -> calibration
follow-up -> this). Both earlier posts found generation collapse at the
method's default qfilters_budget=512 against a 2,238-token prompt, and a
smoke test confirmed that setting the budget above the prompt's token count
(so nothing is evicted) restores fluent, near-fp16 output. That's a binary
result -- "broken at 512, fine once nothing is evicted" -- and doesn't say
where between those two points quality actually degrades.

This script sweeps qfilters_budget across a range spanning "far below the
prompt" to "above the prompt" and reports throughput, peak memory,
compression ratio, and fp16-word-overlap at each point, using the
CALIBRATED filter path only (established in the calibration follow-up post
as faster and no worse than the fallback path at equal budget -- there is
no reason to re-litigate that comparison here).

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_qfilters_budget_sweep.py \\
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


def _build_qfilters_caches(model, budget: int, seed: int, filters_per_layer):
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
            method="qfilters", head_dim=hd, seed=seed + i, qfilters_budget=budget,
        )
        caches.append(QFiltersKVCache(cfg, filters=filters_per_layer[i]))
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


def _run_once(model, tokenizer, budget: int, prompt: str, max_tokens: int, seed: int, filters) -> dict:
    _reset_peak()
    caches = _build_qfilters_caches(model, budget, seed, filters)
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
        "text": text, "tokens_generated": n_tok, "elapsed_s": elapsed,
        "throughput_tok_s": throughput, "peak_mb": peak_mb, "compression_ratio": compression,
    }


def _summarize(runs: list[dict], fp16_text: str) -> dict:
    tputs = [r["throughput_tok_s"] for r in runs]
    peaks = [r["peak_mb"] for r in runs]
    overlaps = [_word_overlap(fp16_text, r["text"]) for r in runs]
    texts = [r["text"] for r in runs]
    return {
        "n_repeats": len(runs),
        "throughput_median": statistics.median(tputs),
        "throughput_min": min(tputs),
        "throughput_max": max(tputs),
        "peak_mb_median": statistics.median(peaks),
        "peak_mb_min": min(peaks),
        "peak_mb_max": max(peaks),
        "compression_ratio": runs[-1]["compression_ratio"],
        "word_overlap_median": statistics.median(overlaps),
        "word_overlap_min": min(overlaps),
        "word_overlap_max": max(overlaps),
        "n_unique_texts": len(set(texts)),
        "texts": texts,
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Sweep qfilters_budget to find the coherence breaking point")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--calibration", default="figures/qwen3_8b_qfilters_calibrated/qfilters_qwen3_8b.npz")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budgets", type=int, nargs="+",
        default=[512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560],
        help="qfilters_budget values to sweep (prompt is ~2238 tokens; "
             "values above that should show ~no eviction, compression_x~=1.0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_qfilters_budget_sweep") / model_stem
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

    prompt_tokens = len(tokenizer.encode(LONG_PROMPT))
    print(f"prompt_tokens={prompt_tokens}  sweeping budgets: {args.budgets}")

    # fp16 reference, interleaved with each budget's repeats rather than
    # computed once, so it shares the same noise floor as everything else.
    fp16_runs = []
    for rep in range(args.repeats):
        print(f"  fp16 rep {rep + 1}/{args.repeats}", flush=True)
        caches = _build_fp16_caches(model)
        _reset_peak()
        text, n_tok, elapsed = _generate(model, tokenizer, LONG_PROMPT, args.max_tokens, caches)
        fp16_runs.append({
            "text": text, "throughput_tok_s": n_tok / max(elapsed, 1e-6), "peak_mb": _peak_mb(),
        })
    fp16_text = fp16_runs[0]["text"]
    fp16_tput = statistics.median(r["throughput_tok_s"] for r in fp16_runs)
    fp16_peak = statistics.median(r["peak_mb"] for r in fp16_runs)
    print(f"  fp16 tok/s median={fp16_tput:.2f} peak median={fp16_peak:.0f}MB")

    sweep = {}
    for budget in args.budgets:
        print(f"\n=== budget={budget} ===", flush=True)
        runs = []
        for rep in range(args.repeats):
            print(f"  rep {rep + 1}/{args.repeats}", flush=True)
            runs.append(_run_once(model, tokenizer, budget, LONG_PROMPT, args.max_tokens, args.seed, calibration.filters))
        s = _summarize(runs, fp16_text)
        sweep[str(budget)] = s
        print(
            f"  tok/s median={s['throughput_median']:.2f}  peak median={s['peak_mb_median']:.0f}MB  "
            f"compression_x={s['compression_ratio']:.2f}  "
            f"word_overlap median={s['word_overlap_median']:.3f} "
            f"(min={s['word_overlap_min']:.3f} max={s['word_overlap_max']:.3f})  "
            f"n_unique_texts={s['n_unique_texts']}"
        )

    payload = {
        "model": args.model, "prompt_tokens": prompt_tokens, "max_tokens": args.max_tokens,
        "repeats": args.repeats, "budgets": args.budgets,
        "calibration_artifact": args.calibration,
        "hardware": hw, "gpu_contention_suspects": contenders,
        "fp16": {
            "throughput_median": fp16_tput, "peak_mb_median": fp16_peak,
            "texts": [r["text"] for r in fp16_runs],
        },
        "sweep": sweep,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults: {json_path}")
    print(f"\nprompt_tokens={prompt_tokens}")
    print(f"fp16              tok/s median={fp16_tput:.2f}  peak median={fp16_peak:.0f}MB")
    for budget in args.budgets:
        s = sweep[str(budget)]
        flag = " <- prompt fits" if budget >= prompt_tokens else ""
        print(
            f"budget={budget:<6d} tok/s median={s['throughput_median']:.2f}  "
            f"peak median={s['peak_mb_median']:.0f}MB  compression_x={s['compression_ratio']:.2f}  "
            f"word_overlap={s['word_overlap_median']:.3f}{flag}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

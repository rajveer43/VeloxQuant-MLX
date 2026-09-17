"""Honest end-to-end QFilters benchmark on Qwen3-8B-4bit.

Third post in the same series as benchmark_qwen3_8b_kivi_honest.py and
benchmark_qwen3_8b_turboquant_rvq_honest.py -- same model, same protocol,
a structurally different method and a different failure mode to look for.

KIVI and TurboQuantRVQ both *quantize* every key/value -- information is
approximated but nothing is discarded, so fp16 and the quantized arms
should produce very similar (KIVI: byte-identical) text at any bit-width
that doesn't collapse quality. QFiltersKVCache is an *eviction* method: once
the cache exceeds `qfilters_budget` tokens, the lowest-scoring tokens are
dropped and their information is gone, not approximated. So this run
explicitly does NOT expect fp16 and QFilters text to match once a prompt's
token count exceeds the budget -- that divergence, and whether the surviving
output still reads as coherent, is the actual question this script is built
to surface, not a bug to rule out.

Two independent things are being tested and must not be conflated:
  1. The Metal fused-evict kernel (qfilters_on) vs the pure-MLX eviction
     path (qfilters_off) -- the module docstring claims these "agree
     bit-for-bit" (same tie-breaking convention). That predicts byte-identical
     text between qfilters_on and qfilters_off, exactly like the KIVI/
     TurboQuantRVQ kernel checks.
  2. QFilters (either path) vs fp16 -- NOT expected to match once the budget
     is exceeded, by construction. This is measured and reported honestly,
     including a lexical-overlap coherence proxy, rather than silently
     treated as a pass/fail check.

This run uses the *fallback* (uncalibrated) filter path -- filters=None,
estimated via SVD of the first `qfilters_calib_tokens` observed keys -- since
no calibrated query-SVD filters exist for Qwen3-8B in this repo and the
docstring is explicit that the fallback is a real, documented mode (not a
test-only stub, unlike VecInfer's random-init codebook path, which is why
VecInfer was not chosen for this run).

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_qfilters_honest.py \\
        --model mlx-community/Qwen3-8B-4bit
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

PROMPTS = {
    "short": (_PASSAGE * 4) + "\n\nSummarize the passage above in two sentences.",
    "long": (_PASSAGE * 40)
    + (
        "\n\nGiven the passage above, explain in simple terms why the KV cache "
        "is the binding memory constraint for long-context inference on Apple "
        "Silicon, covering both the linear growth and the unified-memory "
        "contention."
    ),
}


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


def _build_qfilters_caches(model, budget: int, seed: int, use_metal: bool):
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
        # filters=None -> fallback key-SVD path (see module docstring).
        caches.append(QFiltersKVCache(cfg, filters=None))
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


def _run_once(model, tokenizer, arm: str, prompt: str, max_tokens: int, budget: int, seed: int) -> dict:
    _reset_peak()
    if arm == "fp16":
        caches = _build_fp16_caches(model)
    elif arm == "qfilters_off":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=False)
    elif arm == "qfilters_on":
        caches = _build_qfilters_caches(model, budget, seed, use_metal=True)
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
        "arm": arm,
        "text": text,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "compression_ratio": compression,
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


def _run_prompt_length(model, tokenizer, prompt_name: str, prompt: str, max_tokens: int,
                        repeats: int, budget: int, seed: int) -> dict:
    arms = ["fp16", "qfilters_off", "qfilters_on"]
    collected: dict[str, list[dict]] = {a: [] for a in arms}
    for rep in range(repeats):
        for arm in arms:
            print(f"  [{prompt_name}] rep {rep + 1}/{repeats} arm={arm}", flush=True)
            collected[arm].append(_run_once(model, tokenizer, arm, prompt, max_tokens, budget, seed))

    summaries = {arm: _summarize(runs) for arm, runs in collected.items()}

    off_texts = set(summaries["qfilters_off"]["texts"])
    on_texts = set(summaries["qfilters_on"]["texts"])
    kernel_identical = off_texts == on_texts and len(off_texts) == 1

    fp16_text = summaries["fp16"]["texts"][0]
    qf_text = summaries["qfilters_on"]["texts"][0]
    overlap = _word_overlap(fp16_text, qf_text)

    return {
        "prompt_name": prompt_name,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "summaries": summaries,
        "qfilters_on_vs_off_identical_text": kernel_identical,
        "fp16_vs_qfilters_word_overlap": overlap,
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Honest QFilters benchmark on Qwen3-8B")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=int, default=512,
                         help="qfilters_budget; default 512 (this method's own default). "
                              "The short prompt (~231 tok) stays under budget -- no eviction. "
                              "The long prompt (~2238 tok) exceeds it -- heavy eviction, by design.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_qfilters_honest") / model_stem
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

    layers = getattr(model, "layers", None) or model.model.layers
    margs = getattr(model, "args", None) or model.model.args
    n_layers = len(layers)
    n_kv_heads = getattr(margs, "num_key_value_heads", None) or getattr(margs, "num_attention_heads", 1)
    head_dim = getattr(margs, "head_dim", None) or (margs.hidden_size // margs.num_attention_heads)
    print(f"  n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim} budget={args.budget}")

    results = {}
    for name, prompt in PROMPTS.items():
        print(f"\n=== prompt length: {name} ===", flush=True)
        results[name] = _run_prompt_length(
            model, tokenizer, name, prompt, args.max_tokens, args.repeats, args.budget, args.seed,
        )

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "qfilters_budget": args.budget,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "hardware": hw,
        "gpu_contention_suspects": contenders,
        "results": results,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults: {json_path}")
    for name, r in results.items():
        print(f"\n--- {name} (prompt_tokens={r['prompt_tokens']}) ---")
        for arm, s in r["summaries"].items():
            print(
                f"  {arm:<14s} tok/s median={s['throughput_median']:.1f} "
                f"min={s['throughput_min']:.1f} max={s['throughput_max']:.1f}  "
                f"peak median={s['peak_mb_median']:.0f}MB "
                f"(min={s['peak_mb_min']:.0f} max={s['peak_mb_max']:.0f})  "
                f"compression_x={s['compression_ratio']:.2f}"
            )
        print(f"  qfilters_on vs qfilters_off identical text: {r['qfilters_on_vs_off_identical_text']}")
        print(f"  fp16 vs qfilters word overlap: {r['fp16_vs_qfilters_word_overlap']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

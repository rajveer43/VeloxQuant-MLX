"""Honest end-to-end TurboQuantRVQ benchmark on Qwen3-8B-4bit.

Companion to benchmark_qwen3_8b_kivi_honest.py, same model and same
protocol, different method and a different question. KIVI quantizes keys
then immediately dequantizes back to fp16 before attention -- the live
tensor is fp16 at all times, so KIVI's "compression ratio" is a hypothetical
byte-accounting number, not a memory-savings claim (see kivi_cache.py's own
docstring). TurboQuantRVQ is structurally different: keys are stored
**packed** (two bit-packed uint32 RVQ-index streams + a shared fp16 norm)
and only dequantized on fetch -- the same accepted pattern as mlx_lm's own
native QuantizedKVCache. So unlike the KIVI run, a peak-memory reduction
here is a real claim to test, not one ruled out by construction.

turboquant_rvq_cache.py's own docstring already reports one measured data
point: -12.8% peak memory vs fp16 on a 1B model at a single 4002-token
prompt. This run checks whether that holds at a bigger model (Qwen3-8B) and
across two prompt lengths, with the same noise-floor discipline as the KIVI
post (interleaved repeats, median/min/max, no single-shot conclusions).

Metal kernel toggle: unlike KIVIKVCache, TurboQuantRVQKVCache has no public
use_metal_kernels config field -- the fused quantize+pack kernel
(_use_metal_pack) is auto-detected from head_dim (power-of-two, <=1024) at
construction time and is a pure perf knob (bit-identical to the MLX path,
per the module docstring's issue #251 reference). This script forces it off
for the "off" arm by overwriting the instance attribute post-construction,
the same latch the class itself flips on a kernel-side failure.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_turboquant_rvq_honest.py \\
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


def _build_rvq_caches(model, b: int, seed: int, use_metal: bool):
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None) or model.model.args

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        hd = getattr(attn, "head_dim", None) if attn is not None else None
        if hd is None:
            hd = args.hidden_size // args.num_attention_heads
        cfg = KVCacheConfig(
            method="turboquant_rvq",
            head_dim=hd,
            bit_width_inlier=b,
            seed=seed + i,
        )
        cache = TurboQuantRVQKVCache(cfg)
        # No public toggle for the fused pack kernel -- force the same latch
        # the class flips on a kernel-side failure, for a clean A/B. Only
        # ever narrows eligibility (never forces it on beyond what the
        # constructor already auto-detected), matching the class's own
        # fallback semantics.
        cache._use_metal_pack = bool(use_metal) and cache._use_metal_pack
        caches.append(cache)
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


def _run_once(model, tokenizer, arm: str, prompt: str, max_tokens: int, b: int, seed: int) -> dict:
    _reset_peak()
    if arm == "fp16":
        caches = _build_fp16_caches(model)
    elif arm == "rvq_off":
        caches = _build_rvq_caches(model, b, seed, use_metal=False)
    elif arm == "rvq_on":
        caches = _build_rvq_caches(model, b, seed, use_metal=True)
    else:
        raise ValueError(arm)

    text, n_tok, elapsed = _generate(model, tokenizer, prompt, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    key_compressed = key_fp16 = 0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_compressed += c.compressed_key_bytes
            key_fp16 += c.fp16_key_bytes
    key_ratio = (key_fp16 / key_compressed) if key_compressed else 1.0

    return {
        "arm": arm,
        "text": text,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "key_compression": key_ratio,
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
        "key_compression": runs[-1]["key_compression"],
        "texts": [r["text"] for r in runs],
    }


def _run_prompt_length(model, tokenizer, prompt_name: str, prompt: str, max_tokens: int,
                        repeats: int, b: int, seed: int) -> dict:
    arms = ["fp16", "rvq_off", "rvq_on"]
    collected: dict[str, list[dict]] = {a: [] for a in arms}
    for rep in range(repeats):
        for arm in arms:
            print(f"  [{prompt_name}] rep {rep + 1}/{repeats} arm={arm}", flush=True)
            collected[arm].append(_run_once(model, tokenizer, arm, prompt, max_tokens, b, seed))

    summaries = {arm: _summarize(runs) for arm, runs in collected.items()}

    off_texts = set(summaries["rvq_off"]["texts"])
    on_texts = set(summaries["rvq_on"]["texts"])
    identical = off_texts == on_texts and len(off_texts) == 1

    return {
        "prompt_name": prompt_name,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "summaries": summaries,
        "rvq_on_vs_off_identical_text": identical,
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Honest TurboQuantRVQ benchmark on Qwen3-8B")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bits", type=int, default=2,
                         help="TurboQuantRVQ bit_width_inlier; default 2 (this method's "
                              "own default -- unlike KIVI, no documented quality collapse "
                              "at this width was found in the cache's docstring)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_turboquant_rvq_honest") / model_stem
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
    print(f"  n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim}")

    is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
    print(f"  head_dim is power-of-two (fused-pack eligible): {is_pow2}")

    results = {}
    for name, prompt in PROMPTS.items():
        print(f"\n=== prompt length: {name} ===", flush=True)
        results[name] = _run_prompt_length(
            model, tokenizer, name, prompt, args.max_tokens, args.repeats, args.bits, args.seed,
        )

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bits": args.bits,
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
                f"  {arm:<10s} tok/s median={s['throughput_median']:.1f} "
                f"min={s['throughput_min']:.1f} max={s['throughput_max']:.1f}  "
                f"peak median={s['peak_mb_median']:.0f}MB "
                f"(min={s['peak_mb_min']:.0f} max={s['peak_mb_max']:.0f})  "
                f"key_x={s['key_compression']:.2f}"
            )
        print(f"  rvq_on vs rvq_off identical text: {r['rvq_on_vs_off_identical_text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

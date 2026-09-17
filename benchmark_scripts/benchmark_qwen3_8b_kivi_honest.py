"""Honest end-to-end KIVI Metal-kernel benchmark on Qwen3-8B-4bit.

Model choice (see docs-site guide / this script's companion report for the
full writeup): the newest Qwen releases at benchmark time (Qwen3.5, Qwen3.6)
use a hybrid linear-attention / full-attention architecture loaded via
mlx_vlm, which this repo's KIVI cache and mlx_lm-based benchmark harness are
not built against -- KIVI's per-channel/per-token KV quantization assumes
standard full attention at every layer. Qwen3-8B is the newest model that is
both (a) plain causal, full GQA attention every layer, mlx_lm-native, and
(b) small enough to fit this machine's ~19GB working-set cap (4.61GB weights
at 4-bit; 36 layers x 8 KV heads x 128 head_dim).

Protocol (matches docs-site/blog/2026-08-12-kivi-metal-kernel-honest-benchmark.md
Generation 4, the protocol that post's own earlier generations converged on
after three false conclusions):

  1. kernel-off vs kernel-on vs fp16 baseline, end-to-end (not op-level).
  2. Interleaved A/B repeats (not blocked) so drift/thermal state hits all
     arms equally, reporting median/min/max to expose the noise floor.
  3. Two prompt lengths, so a short-context reading alone can't hide real
     headroom (see the nsg-autotune project memory: short prompts sit near
     the minimum of this repo's headroom curves).
  4. Byte-identical output check between kernel-on/off (the two paths are
     documented bit-identical -- see kivi_cache.py -- so this is a
     regression tripwire, not an open question).
  5. bit_width_inlier=4, not the KIVI default of 2: kivi_cache.py's own
     docstring documents that b=2 produces negative logit cosine similarity
     (effectively uncorrelated output) by 64 decode steps on a small model.
     b=4 is measured safe (0.996) in the same doc. Benchmarking a config
     known to degrade output quality would make any throughput number
     meaningless.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_kivi_honest.py \\
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

# Two prompt lengths -- short and long -- so a single short-context reading
# can't stand in for the whole picture (see module docstring point 3).
PROMPTS = {
    "short": (_PASSAGE * 4)
    + "\n\nSummarize the passage above in two sentences.",
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
        return float(mx.metal.get_peak_memory()) / (1024**2)
    except Exception:
        return float("nan")


def _reset_peak() -> None:
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if chip:
            info["chip"] = chip
        if mem:
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
    except Exception:
        pass
    return info


def _gpu_contention_check() -> list[str]:
    """Best-effort list of other GPU-heavy processes running right now.

    Not exhaustive -- just enough to catch the exact failure mode
    documented in the KIVI post (an LLM benchmark left running in the
    background produced a spurious 15x/5x contention swing).
    """
    suspects = []
    try:
        out = subprocess.run(
            ["ps", "-Ao", "comm"], capture_output=True, text=True, timeout=5
        ).stdout
        keywords = ("python", "ollama", "mlx", "llama", "lmstudio")
        for line in out.splitlines():
            low = line.lower()
            if any(k in low for k in keywords) and "benchmark_qwen3_8b" not in low:
                suspects.append(line.strip())
    except Exception:
        pass
    return suspects


def _build_kivi_caches(model, b: int, group_size: int, residual_length: int, use_metal: bool):
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.kivi_cache import KIVIKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None) or model.model.args

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        hd = getattr(attn, "head_dim", None) if attn is not None else None
        if hd is None:
            hd = args.hidden_size // args.num_attention_heads
        cfg = KVCacheConfig(
            method="kivi",
            head_dim=hd,
            bit_width_inlier=b,
            kivi_group_size=group_size,
            residual_length=residual_length,
            seed=42 + i,
            use_metal_kernels=use_metal,
        )
        caches.append(KIVIKVCache(cfg))
    return caches


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers = getattr(model, "layers", None) or model.model.layers
    return [_FallbackCache() for _ in layers]


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple[str, int, float]:
    from mlx_lm import generate

    t0 = time.time()
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
        prompt_cache=caches,
    )
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return out, n_tok, elapsed


def _run_once(model, tokenizer, arm: str, prompt: str, max_tokens: int, b: int,
              group_size: int, residual_length: int) -> dict:
    _reset_peak()
    if arm == "fp16":
        caches = _build_fp16_caches(model)
    elif arm == "kivi_off":
        caches = _build_kivi_caches(model, b, group_size, residual_length, use_metal=False)
    elif arm == "kivi_on":
        caches = _build_kivi_caches(model, b, group_size, residual_length, use_metal=True)
    else:
        raise ValueError(arm)

    text, n_tok, elapsed = _generate(model, tokenizer, prompt, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    key_compressed = key_fp16 = val_compressed = val_fp16 = residual_fp16 = 0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_compressed += c.compressed_key_bytes
            key_fp16 += c.fp16_key_bytes
            val_compressed += getattr(c, "compressed_value_bytes", 0)
            val_fp16 += getattr(c, "fp16_value_bytes", 0)
            residual_fp16 += getattr(c, "residual_fp16_bytes", 0)
    key_ratio = (key_fp16 / key_compressed) if key_compressed else 1.0
    total_fp16 = key_fp16 + val_fp16
    total_comp = key_compressed + val_compressed + residual_fp16
    full_kv_ratio = (total_fp16 / total_comp) if total_comp else 1.0

    return {
        "arm": arm,
        "text": text,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "key_compression": key_ratio,
        "full_kv_compression": full_kv_ratio,
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
        "key_compression": runs[-1]["key_compression"],
        "full_kv_compression": runs[-1]["full_kv_compression"],
        "texts": [r["text"] for r in runs],
    }


def _run_prompt_length(model, tokenizer, prompt_name: str, prompt: str, max_tokens: int,
                        repeats: int, b: int, group_size: int, residual_length: int) -> dict:
    arms = ["fp16", "kivi_off", "kivi_on"]
    # Interleaved: one full round through all arms, repeated `repeats` times,
    # rather than running all repeats of one arm before moving to the next.
    # This is the fix for Lie #2 in the reference post (blocked runs let a
    # background contention episode land unevenly across arms).
    collected: dict[str, list[dict]] = {a: [] for a in arms}
    for rep in range(repeats):
        for arm in arms:
            print(f"  [{prompt_name}] rep {rep + 1}/{repeats} arm={arm}", flush=True)
            collected[arm].append(
                _run_once(model, tokenizer, arm, prompt, max_tokens, b, group_size, residual_length)
            )

    summaries = {arm: _summarize(runs) for arm, runs in collected.items()}

    # Byte-identical check: kivi_off and kivi_on are documented bit-identical
    # (same quant/dequant math, Metal kernel is a pure perf toggle) -- this
    # asserts that held on this run, on this model, rather than assuming it.
    off_texts = set(summaries["kivi_off"]["texts"])
    on_texts = set(summaries["kivi_on"]["texts"])
    identical = off_texts == on_texts and len(off_texts) == 1

    return {
        "prompt_name": prompt_name,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "summaries": summaries,
        "kivi_on_vs_off_identical_text": identical,
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Honest KIVI Metal-kernel benchmark on Qwen3-8B")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bits", type=int, default=4,
                         help="KIVI bit_width_inlier; default 4, NOT the KIVI-default 2 "
                              "(b=2 is documented unsafe for long decode -- see module docstring)")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_kivi_honest") / model_stem
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

    results = {}
    for name, prompt in PROMPTS.items():
        print(f"\n=== prompt length: {name} ===", flush=True)
        results[name] = _run_prompt_length(
            model, tokenizer, name, prompt, args.max_tokens, args.repeats,
            args.bits, args.group_size, args.residual_length,
        )

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bits": args.bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
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
                f"peak={s['peak_mb_median']:.0f}MB  "
                f"key_x={s['key_compression']:.2f} fullKV_x={s['full_kv_compression']:.2f}"
            )
        print(f"  kivi_on vs kivi_off identical text: {r['kivi_on_vs_off_identical_text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

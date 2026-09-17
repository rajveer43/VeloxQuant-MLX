"""Honest end-to-end VecInfer benchmark on Qwen3-8B-4bit.

Fourth post in the same series as the KIVI, TurboQuantRVQ, and QFilters
Qwen3-8B posts -- same model, same protocol, another structurally different
method. VecInfer is product vector-quantization: keys get a per-(head,
channel) smooth scale plus a Walsh-Hadamard rotation, then are encoded
against a trained codebook and immediately dequantized (and inverse-
transformed) before attention runs -- the live tensor is fp16 at all times,
same accepted "quantize then dequantize immediately" pattern as KIVI, so
(like KIVI, unlike TurboQuantRVQ) any memory-savings claim here is a
hypothetical byte-accounting number, not a measurement of resident bytes.

Calibration caveat, stated up front rather than discovered in the results:
VecInferKVCache's own docstring says a random-initialized codebook is "only
useful for shape/wiring tests." No calibration tooling for real model
activations exists in this repo yet. What DOES exist -- reused here from
benchmark_vecinfer.py -- is k-means-style codebook training
(train_codebook / calibrate_smooth_factors) on synthetic Gaussian samples
shaped like the model's keys/values, not the model's actual activations.
That is a real, trained codebook (not literally random), but it is still
not calibrated against Qwen3-8B's real key/value distribution. This run
uses that synthetic-calibration path and reports it as exactly that --
better than random-init, not equivalent to real calibration -- the same
honesty standard the QFilters post applied to its own uncalibrated
fallback-filter path.

Metal kernel toggle: VecInferKVCache has a public three-state
use_metal_kernels field on KVCacheConfig (None=auto, True=require,
False=force-off), same shape as KIVI's toggle -- no private-attribute
override needed for a clean A/B, unlike TurboQuantRVQ or QFilters.

This run does NOT enable the fused_sdpa path (codebook-indices-only
storage, no fp16 K_hat at rest) -- that is a separate, more invasive
optimization requiring mlx_lm to be monkey-patched
(patch_mlx_lm_for_fused_sdpa) and a stricter shape constraint
(n_sub <= 16; Qwen3-8B's head_dim=128 with the default key_sub_dim=4 gives
n_sub=32, which does not qualify without also changing key_sub_dim). Adding
that as a third axis on top of kernel-on/off and calibration quality would
conflate three separate questions in one run.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_qwen3_8b_vecinfer_honest.py \\
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
import numpy as np


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


def _calibrate_artifacts(
    head_dim: int, n_heads: int, key_bits: int, value_bits: int,
    key_sub_dim: int, value_sub_dim: int, cache_dir: Path, seed: int = 42,
) -> dict:
    """Synthetic-Gaussian codebook training, reused from benchmark_vecinfer.py.

    NOT real calibration against Qwen3-8B's actual key/value activations --
    see module docstring. Cached to disk so reruns are fast.
    """
    from veloxquant_mlx.allocators.vecinfer import calibrate_smooth_factors, train_codebook

    sig = f"hd{head_dim}_h{n_heads}_kb{key_bits}_vb{value_bits}_ks{key_sub_dim}_vs{value_sub_dim}"
    cache_path = cache_dir / f"{sig}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "smooth": mx.array(data["smooth"]),
            "key_codebook": mx.array(data["key_cb"]),
            "value_codebook": mx.array(data["value_cb"]),
        }

    print(f"  [calib] training codebooks (sig={sig})...", flush=True)
    rng_np = np.random.default_rng(seed)
    n_samples = 4096
    K = mx.array(rng_np.standard_normal((n_samples, n_heads, head_dim)).astype(np.float32))
    V = mx.array(rng_np.standard_normal((n_samples, n_heads, head_dim)).astype(np.float32))
    smooth = calibrate_smooth_factors(K)

    k_subs = mx.array(np.asarray(K).reshape(-1, key_sub_dim))
    v_subs = mx.array(np.asarray(V).reshape(-1, value_sub_dim))
    n_train = min(8000, k_subs.shape[0])
    key_cb = train_codebook(k_subs[:n_train], 2**key_bits, max_iter=15, seed=seed)
    val_cb = train_codebook(v_subs[:n_train], 2**value_bits, max_iter=15, seed=seed + 1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, smooth=np.asarray(smooth), key_cb=np.asarray(key_cb), value_cb=np.asarray(val_cb))
    return {"smooth": smooth, "key_codebook": key_cb, "value_codebook": val_cb}


def _build_vecinfer_caches(model, artifacts: dict, key_bits: int, value_bits: int,
                            key_sub_dim: int, value_sub_dim: int, seed: int, use_metal) -> list:
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None) or model.model.args

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        hd = getattr(attn, "head_dim", None) if attn is not None else None
        if hd is None:
            hd = args.hidden_size // args.num_attention_heads
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=key_sub_dim,
            value_sub_dim=value_sub_dim,
            key_codebook_bits=key_bits,
            value_codebook_bits=value_bits,
            smooth_factors=artifacts["smooth"],
            key_codebook=artifacts["key_codebook"],
            value_codebook=artifacts["value_codebook"],
            seed=seed + i,
            use_metal_kernels=use_metal,
        )
        caches.append(VecInferKVCache(cfg))
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


def _run_once(model, tokenizer, arm: str, prompt: str, max_tokens: int, artifacts: dict,
              key_bits: int, value_bits: int, key_sub_dim: int, value_sub_dim: int, seed: int) -> dict:
    _reset_peak()
    if arm == "fp16":
        caches = _build_fp16_caches(model)
    elif arm == "vecinfer_off":
        caches = _build_vecinfer_caches(model, artifacts, key_bits, value_bits, key_sub_dim, value_sub_dim, seed, use_metal=False)
    elif arm == "vecinfer_on":
        caches = _build_vecinfer_caches(model, artifacts, key_bits, value_bits, key_sub_dim, value_sub_dim, seed, use_metal=True)
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
        "arm": arm, "text": text, "tokens_generated": n_tok, "elapsed_s": elapsed,
        "throughput_tok_s": throughput, "peak_mb": peak_mb, "key_compression": key_ratio,
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


def _run_prompt_length(model, tokenizer, prompt_name: str, prompt: str, max_tokens: int, repeats: int,
                        artifacts: dict, key_bits: int, value_bits: int, key_sub_dim: int,
                        value_sub_dim: int, seed: int) -> dict:
    arms = ["fp16", "vecinfer_off", "vecinfer_on"]
    collected: dict[str, list[dict]] = {a: [] for a in arms}
    for rep in range(repeats):
        for arm in arms:
            print(f"  [{prompt_name}] rep {rep + 1}/{repeats} arm={arm}", flush=True)
            collected[arm].append(_run_once(model, tokenizer, arm, prompt, max_tokens, artifacts,
                                             key_bits, value_bits, key_sub_dim, value_sub_dim, seed))

    summaries = {arm: _summarize(runs) for arm, runs in collected.items()}
    off_texts = set(summaries["vecinfer_off"]["texts"])
    on_texts = set(summaries["vecinfer_on"]["texts"])
    kernel_identical = off_texts == on_texts and len(off_texts) == 1

    return {
        "prompt_name": prompt_name,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "summaries": summaries,
        "vecinfer_on_vs_off_identical_text": kernel_identical,
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(description="Honest VecInfer benchmark on Qwen3-8B")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--key-bits", type=int, default=12, help="key_codebook_bits; default 12 (this method's own default)")
    parser.add_argument("--value-bits", type=int, default=8, help="value_codebook_bits; default 8 (this method's own default)")
    parser.add_argument("--key-sub-dim", type=int, default=4)
    parser.add_argument("--value-sub-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calib-dir", default=None)
    args = parser.parse_args()

    model_stem = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("figures/qwen3_8b_vecinfer_honest") / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir) if args.calib_dir else Path.home() / ".cache" / "veloxquant" / "vecinfer"

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

    artifacts = _calibrate_artifacts(
        head_dim, n_kv_heads, args.key_bits, args.value_bits,
        args.key_sub_dim, args.value_sub_dim, calib_dir, args.seed,
    )

    results = {}
    for name, prompt in PROMPTS.items():
        print(f"\n=== prompt length: {name} ===", flush=True)
        results[name] = _run_prompt_length(
            model, tokenizer, name, prompt, args.max_tokens, args.repeats,
            artifacts, args.key_bits, args.value_bits, args.key_sub_dim, args.value_sub_dim, args.seed,
        )

    payload = {
        "model": args.model, "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "key_codebook_bits": args.key_bits, "value_codebook_bits": args.value_bits,
        "key_sub_dim": args.key_sub_dim, "value_sub_dim": args.value_sub_dim,
        "calibration": "synthetic-gaussian (NOT real model activations, see script docstring)",
        "max_tokens": args.max_tokens, "repeats": args.repeats,
        "hardware": hw, "gpu_contention_suspects": contenders, "results": results,
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
                f"key_x={s['key_compression']:.2f}"
            )
        print(f"  vecinfer_on vs vecinfer_off identical text: {r['vecinfer_on_vs_off_identical_text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

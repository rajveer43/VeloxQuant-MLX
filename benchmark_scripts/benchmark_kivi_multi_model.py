"""Multi-model real-model validation for KIVI Metal kernel warmup (issue #250, PR #268).

Same measurements as ``benchmark_kivi_warmup.py`` (kernel compile cost, TTFT,
prefill, decode throughput, perplexity), run across every locally-cached
mlx-community instruct model that's small/mid enough to iterate on in one
sitting. Large models (>= 32B params, VLMs, non-causal-LM checkpoints) are
deliberately excluded -- this is a breadth check across architectures/sizes,
not a full leaderboard run.

Usage:
    python benchmark_scripts/benchmark_kivi_multi_model.py
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.metal import _kivi_quant
from veloxquant_mlx.metal._warmup import warmup_for_config

RESULTS_PATH = Path(__file__).parents[1] / "kivi_multi_model_benchmark_results.json"

# Cached, causal-LM, single-GPU-friendly models spanning a range of sizes and
# families (Llama, Qwen, Mistral, Falcon, Phi, Gemma, DeepSeek-MoE). Skips
# Qwen2.5-32B-Instruct-4bit (too slow to run x3 configs each on a laptop),
# Qwen2-VL (VLM, different call path), and encoder-only/embedding models.
MODELS = [
    "mlx-community/SmolLM2-135M-Instruct",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Falcon3-7B-Instruct-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",
]

# KIVI's default residual_length=128 keeps the most recent 128 tokens in the
# fp16 residual window -- only tokens older than that get quantized. This
# sample is long enough (~430+ tokens) that it forces at least one full
# kivi_group_size=32 block to flush through quantization during prefill.
LONG_SAMPLE = (
    """
In mathematics, the Riemann hypothesis is a conjecture that the Riemann zeta function
has its zeros only at the negative even integers and complex numbers with real part
one half. Many consider it to be the most important unsolved problem in pure mathematics.
It was proposed by Bernhard Riemann in 1859 in a landmark paper, and it remains unsolved
to this day. The hypothesis states that the nontrivial zeros of the zeta function all
lie on the critical line in the complex plane. Large language models are neural networks
trained on vast text corpora to predict the next token in a sequence. At inference time,
these models maintain a key-value cache that stores intermediate attention states to
avoid recomputing them for each new token. The cache grows linearly with sequence length
and can consume several gigabytes of memory for long contexts, making it the dominant
memory bottleneck for long-context serving. Quantizing the key-value cache to fewer bits
per value reduces this memory footprint substantially, often by four to eight times
compared to a full-precision floating point representation, at the cost of a small,
usually tolerable, increase in perplexity or downstream task error. Group-wise asymmetric
quantization schemes such as KIVI compute a separate minimum and maximum per small group
of channels or tokens, which keeps the quantization error bounded even when outlier
values are present in some channels. Apple Silicon integrates the CPU and GPU on a single
die with shared unified memory, eliminating the PCIe bandwidth bottleneck present on
discrete GPU setups, which makes it practical to run these compressed representations
efficiently even on consumer hardware with limited total memory capacity. Custom Metal
compute kernels written directly in Metal Shading Language can fuse several of these
quantization steps -- computing group minimums and maximums, clipping, rounding, and
reconstructing values -- into a single GPU dispatch, avoiding the overhead of materializing
several full-size intermediate tensors that a naive implementation using only high level
array operations would otherwise allocate on every single decode step of generation.
"""
).strip()

PROMPT = LONG_SAMPLE + "\n\nIn summary, the key insight from all of this is that"


def load_model(model_id: str):
    from mlx_lm import load

    print(f"  Loading {model_id} ...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"    loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def compute_perplexity_stable(model, tokenizer, text: str, max_tokens: int = 512) -> float:
    tokens = tokenizer.encode(text)[:max_tokens]
    if len(tokens) < 4:
        return float("nan")
    input_ids = mx.array(tokens[:-1], dtype=mx.int32)[None]
    targets = tokens[1:]
    cache = model.make_cache()
    logits = model(input_ids, cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)
    logits_np = np.array(logits[0].astype(mx.float32))
    total_nll = 0.0
    for t, tgt in enumerate(targets):
        lg = logits_np[t]
        lg_shifted = lg - lg.max()
        log_sum_exp = np.log(np.sum(np.exp(lg_shifted)))
        total_nll -= lg_shifted[tgt] - log_sum_exp
    return math.exp(total_nll / len(targets))


def measure_kernel_compile_cost(config: KVCacheConfig) -> dict:
    _kivi_quant._cache.clear()
    t0 = time.perf_counter()
    warmup_for_config(config)
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    warmup_for_config(config)
    warm_s = time.perf_counter() - t0

    return {"cold_compile_s": cold_s, "warm_recall_s": warm_s}


def measure_ttft(model, tokenizer, prompt: str) -> float:
    from mlx_lm import generate

    t0 = time.perf_counter()
    generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    mx.eval()
    return time.perf_counter() - t0


def measure_prefill_and_decode(model, tokenizer, prompt: str, n_new_tokens: int = 64) -> dict:
    from mlx_lm import generate

    prompt_tokens = tokenizer.encode(prompt)

    t0 = time.perf_counter()
    generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    mx.eval()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=n_new_tokens, verbose=False)
    mx.eval()
    total_s = time.perf_counter() - t0

    n_out = max(len(tokenizer.encode(out)), 1)
    decode_s = max(total_s - prefill_s, 1e-9)
    return {
        "prefill_s": prefill_s,
        "decode_tokens_per_sec": (n_out - 1) / decode_s if n_out > 1 else float("nan"),
        "n_prompt_tokens": len(prompt_tokens),
        "n_new_tokens": n_out,
    }


def run_config(model_id: str, label: str, config: KVCacheConfig | None) -> dict:
    print(f"\n  --- {label} ---")

    compile_cost = None
    if config is not None:
        compile_cost = measure_kernel_compile_cost(config)

    _kivi_quant._cache.clear()
    model, tokenizer = load_model(model_id)

    if config is not None:
        try:
            patch_model_kv_cache(model, config)
        except Exception as e:
            print(f"    SKIPPED (patch failed: {e})")
            del model, tokenizer
            gc.collect()
            return {"label": label, "error": str(e)}

    try:
        cold_ttft = measure_ttft(model, tokenizer, PROMPT)
        warm_ttft = measure_ttft(model, tokenizer, PROMPT)
        latency = measure_prefill_and_decode(model, tokenizer, PROMPT)
        ppl = compute_perplexity_stable(model, tokenizer, LONG_SAMPLE)
        n_kernel_variants = len(_kivi_quant._cache)
    except Exception as e:
        print(f"    FAILED: {e}")
        del model, tokenizer
        gc.collect()
        return {"label": label, "error": str(e)}

    result = {
        "label": label,
        "kernel_compile": compile_cost,
        "cold_ttft_s": cold_ttft,
        "warm_ttft_s": warm_ttft,
        "ttft_delta_s": cold_ttft - warm_ttft,
        **latency,
        "perplexity": ppl,
        "n_kivi_kernel_variants_compiled": n_kernel_variants,
    }
    print(
        f"    ttft(cold/warm)={cold_ttft * 1e3:.0f}/{warm_ttft * 1e3:.0f}ms  "
        f"prefill={latency['prefill_s'] * 1e3:.0f}ms  "
        f"decode={latency['decode_tokens_per_sec']:.1f}tok/s  "
        f"ppl={ppl:.2f}"
    )
    del model, tokenizer
    gc.collect()
    return result


def run_model(model_id: str) -> dict:
    print(f"\n=== {model_id} ===")
    try:
        configs = {
            "fp16 (baseline)": None,
            "KIVI b=2": KVCacheConfig(method="kivi", bit_width_inlier=2, seed=42),
            "KIVI b=4": KVCacheConfig(method="kivi", bit_width_inlier=4, seed=42),
        }
        results = {label: run_config(model_id, label, cfg) for label, cfg in configs.items()}
        return {"model": model_id, "results": results}
    except Exception as e:
        print(f"  MODEL FAILED: {e}")
        return {"model": model_id, "error": str(e)}


def main() -> None:
    all_results = []
    for model_id in MODELS:
        all_results.append(run_model(model_id))
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== Summary ===")
    header = f"{'model':<42}{'config':<18}{'cold TTFT':>11}{'decode t/s':>12}{'ppl':>10}"
    print(header)
    for entry in all_results:
        if "error" in entry:
            print(f"{entry['model']:<42}  ERROR: {entry['error']}")
            continue
        for label, r in entry["results"].items():
            if "error" in r:
                print(f"{entry['model']:<42}{label:<18}  ERROR: {r['error']}")
                continue
            print(
                f"{entry['model']:<42}{label:<18}"
                f"{r['cold_ttft_s'] * 1e3:>11.1f}"
                f"{r['decode_tokens_per_sec']:>12.1f}"
                f"{r['perplexity']:>10.3f}"
            )


if __name__ == "__main__":
    main()

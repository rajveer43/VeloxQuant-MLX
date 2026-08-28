"""Real-model validation for KIVI Metal kernel warmup (issue #250, PR #268).

Runs mlx-community/Llama-3.2-1B-Instruct-4bit (already cached locally, small
enough to iterate on) through the real serving path --
``patch_model_kv_cache`` -> ``KVCacheBuilder.for_model`` -> ``mlx_lm.generate``
-- and measures, for fp16 baseline vs KIVI b=2 and b=4:

  1. Kernel compile cost -- direct, isolated measurement of
     ``warmup_for_config``'s wall-clock cost with an empty kernel cache vs.
     a pre-warmed one. This is the actual thing PR #268 moves out of the
     generation hot path, measured directly rather than inferred from
     generate()-level noise.
  2. TTFT              -- wall-clock to the first generated token, on a
     *long* prompt (>> kivi_group_size * several groups, well past
     residual_length) so KIVI's quantized path is actually exercised
     during prefill -- a short prompt never leaves the fp16 residual
     window and silently never calls the Metal kernel at all.
  3. Prefill latency    -- wall-clock for the prompt forward pass alone.
  4. Decode throughput  -- tokens/sec over a fixed generation length.
  5. Perplexity         -- numerically-stable causal LM perplexity on the
     same long sample (quality regression guard -- KIVI's quant/dequant
     must not silently corrupt the cache under the warmup path).

MMLU is intentionally out of scope for this script: it requires
``pip install lm_eval`` and is a much longer-running eval; see the issue
follow-up before adding it here.

Usage:
    python benchmark_scripts/benchmark_kivi_warmup.py
"""

from __future__ import annotations

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

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
RESULTS_PATH = Path(__file__).parents[1] / "kivi_warmup_benchmark_results.json"

# KIVI's default residual_length=128 keeps the most recent 128 tokens in the
# fp16 residual window -- only tokens older than that get quantized. A short
# prompt never ages out of that window, so it never actually calls the
# Metal kernel. This sample is long enough (~600+ tokens) that a real
# generation run pushes well past 128 tokens of *total* context, forcing at
# least one full kivi_group_size=32 block to flush through quantization.
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

# Plain continuation (not an instruction) so the base model has no reason to
# emit EOS quickly -- an instruct-style "summarize this" prompt finishes in
# a handful of tokens, which starves the decode-throughput measurement.
PROMPT = LONG_SAMPLE + "\n\nIn summary, the key insight from all of this is that"


def load_model():
    from mlx_lm import load

    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def compute_perplexity_stable(model, tokenizer, text: str, max_tokens: int = 512) -> float:
    """Perplexity computed THROUGH model.make_cache() -- a bare model(input_ids)
    call with no cache argument bypasses caching entirely (mlx_lm models
    default cache=None to "no KV cache, full self-attention recompute"), so it
    would silently measure the base model regardless of which method is
    patched in. Feeding an explicit cache list is what actually routes
    through KIVIKVCache.update_and_fetch / quant_dequant_along.
    """
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
    logits_np = np.array(logits[0], dtype=np.float32)
    total_nll = 0.0
    for t, tgt in enumerate(targets):
        lg = logits_np[t]
        lg_shifted = lg - lg.max()
        log_sum_exp = np.log(np.sum(np.exp(lg_shifted)))
        total_nll -= lg_shifted[tgt] - log_sum_exp
    return math.exp(total_nll / len(targets))


def measure_kernel_compile_cost(config: KVCacheConfig) -> dict:
    """Directly isolates the Metal shader-compile cost warmup moves out of
    the generation hot path, independent of any generate()-level noise
    (tokenizer, sampling, Python dispatch overhead that's identical whether
    or not the kernel cache is warm)."""
    _kivi_quant._cache.clear()
    t0 = time.perf_counter()
    warmup_for_config(config)
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    warmup_for_config(config)  # kernel cache already populated by the call above
    warm_s = time.perf_counter() - t0

    return {"cold_compile_s": cold_s, "warm_recall_s": warm_s}


def measure_ttft(model, tokenizer, prompt: str) -> float:
    """Wall-clock seconds to the first generated token (prefill + 1 decode step)."""
    from mlx_lm import generate

    t0 = time.perf_counter()
    generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    mx.eval()
    return time.perf_counter() - t0


def measure_prefill_and_decode(model, tokenizer, prompt: str, n_new_tokens: int = 64) -> dict:
    """``generate()`` returns only the generated completion text (not
    prompt+completion), so its token count -- not a prompt-length
    subtraction -- is the number of new tokens actually produced. The
    completion can be shorter than ``n_new_tokens`` if the model emits EOS
    first."""
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


def run_config(label: str, config: KVCacheConfig | None) -> dict:
    """Loads a fresh model instance per config so patch state never leaks
    between configs (fp16 baseline uses mlx_lm's own default cache path,
    which only exists pre-patch)."""
    print(f"\n=== {label} ===")

    compile_cost = None
    if config is not None:
        compile_cost = measure_kernel_compile_cost(config)
        print(f"  cold kernel compile: {compile_cost['cold_compile_s'] * 1e3:.1f} ms")
        print(f"  warm kernel recall:  {compile_cost['warm_recall_s'] * 1e3:.3f} ms")

    _kivi_quant._cache.clear()
    model, tokenizer = load_model()

    if config is not None:
        patch_model_kv_cache(model, config)

    cold_ttft = measure_ttft(model, tokenizer, PROMPT)
    warm_ttft = measure_ttft(model, tokenizer, PROMPT)
    latency = measure_prefill_and_decode(model, tokenizer, PROMPT)
    ppl = compute_perplexity_stable(model, tokenizer, LONG_SAMPLE)
    n_kernel_variants = len(_kivi_quant._cache)

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
    print(f"  cold TTFT:  {cold_ttft * 1e3:.1f} ms")
    print(f"  warm TTFT:  {warm_ttft * 1e3:.1f} ms")
    print(f"  prefill:    {latency['prefill_s'] * 1e3:.1f} ms")
    print(f"  decode:     {latency['decode_tokens_per_sec']:.1f} tok/s")
    print(f"  perplexity: {ppl:.3f}")
    print(f"  KIVI kernel variants compiled: {n_kernel_variants}")
    return result


def main() -> None:
    results = []
    results.append(run_config("fp16 (baseline)", None))
    results.append(
        run_config("KIVI b=2", KVCacheConfig(method="kivi", bit_width_inlier=2, seed=42))
    )
    results.append(
        run_config("KIVI b=4", KVCacheConfig(method="kivi", bit_width_inlier=4, seed=42))
    )

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== Summary ===")
    header = (
        f"{'label':<18}{'compile ms':>12}{'recall ms':>11}"
        f"{'cold TTFT':>11}{'warm TTFT':>11}{'ppl':>10}{'#kernels':>10}"
    )
    print(header)
    for r in results:
        kc = r["kernel_compile"]
        compile_ms = f"{kc['cold_compile_s'] * 1e3:.1f}" if kc else "n/a"
        recall_ms = f"{kc['warm_recall_s'] * 1e3:.3f}" if kc else "n/a"
        print(
            f"{r['label']:<18}{compile_ms:>12}{recall_ms:>11}"
            f"{r['cold_ttft_s'] * 1e3:>11.1f}{r['warm_ttft_s'] * 1e3:>11.1f}"
            f"{r['perplexity']:>10.3f}{r['n_kivi_kernel_variants_compiled']:>10}"
        )


if __name__ == "__main__":
    main()

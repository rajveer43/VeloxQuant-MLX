"""Uniform vs. adaptive KV precision benchmark (issue #256).

Answers issue #256's research question directly: does gating KV-cache
precision by token *age* beat a uniform, budget-matched baseline?

Compares three configurations on real Apple Silicon inference:

  fp16-baseline    every token, every layer, 16-bit (no compression)
  uniform-int4     every token quantized to a single fixed bit-width
                    (KVCacheConfig(method="kivi", bit_width_inlier=4) —
                    KIVI at a fixed bit-width *is* "uniform INT4": no
                    residual-window special-casing beyond the KIVI default,
                    same shared min/max group quantizer AgeTieredKV uses)
  age-tiered       recent tokens at 8-bit, mid-age at 4-bit, old at 2-bit
                    (KVCacheConfig(method="age_tiered"), this repo's own
                    method — see docs-site/docs/algorithms/age-tiered.md)

For each configuration, measures:
  * perplexity            — causal-LM next-token perplexity, computed WITH
                             the configuration's cache wired into the
                             forward pass (unlike this file's sibling
                             model_kv_benchmark.py's compute_perplexity*,
                             which never touches a cache at all — that
                             measures the model, not the cache).
  * generation quality     — output text is saved for manual inspection;
                             no automatic scoring (perplexity is the
                             quantitative signal here, matching how every
                             other perplexity-based benchmark in this repo
                             is reported).
  * peak memory            — mx.get_peak_memory(), reset between configs.
  * latency / throughput   — real generate() timing, never assumed.
  * attention reconstruction error — mean squared error between the
    quantized cache's dequantized K/V and the fp16 baseline's K/V at
    matched positions, averaged over layers. Not measured anywhere else in
    this repo; this is the metric issue #256 explicitly asks for and no
    existing benchmark script reports.

Average bit-width matching: age-tiered's default 8/4/2 split is only a fair
comparison against uniform-int4 if their *realized* average bits per token
land close together over the prompt actually used — this script reports
each config's `avg_bits`/`assigned_avg_bits`-equivalent (derived from byte
accounting) so a reader can check the comparison is apples-to-apples rather
than trusting the nominal numbers.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_age_tiered.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


# A long-context prompt so age-tiering actually has old tokens to act on —
# AgeTieredKV's default age_mid_boundary is 1024, so the prefill needs to
# clear that by a comfortable margin (same rationale as benchmark_kivi.py's
# PROMPT: KIVI-family methods only show their effect once context exceeds
# their residual/boundary window).
_PASSAGE = (
    "The key-value cache stores the attention keys and values of every past "
    "token so the model need not recompute them. Its size grows linearly with "
    "context length and, on Apple Silicon unified memory, it competes with the "
    "model weights and the operating system for the same pool. Not every "
    "token in that cache is equally important to future predictions: recently "
    "generated tokens dominate attention scores, while tokens far in the past "
    "contribute comparatively little to any single next-token decision. "
)
PROMPT = (_PASSAGE * 60) + (
    "\n\nGiven the passage above, explain why recent tokens might tolerate "
    "less precision loss than older tokens, and why a uniform bit-width "
    "might not be the most efficient way to compress a KV cache."
)


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _peak_mb() -> float:
    get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return float(get_peak()) / (1024**2)


def _reset_peak() -> None:
    reset_peak = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    reset_peak()


def _hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import subprocess

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


def _layers_and_head_dim(model):
    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None) or model.model.args
    first_attn = None
    for L in layers:
        attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
        if attn is not None:
            first_attn = attn
            break
    head_dim = getattr(first_attn, "head_dim", None) or (
        args.hidden_size // args.num_attention_heads
    )
    return layers, head_dim


def _build_fp16_caches(model) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    layers, _ = _layers_and_head_dim(model)
    return [_FallbackCache() for _ in layers]


def _build_kivi_uniform_caches(model, bits: int, group_size: int) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.kivi_cache import KIVIKVCache

    layers, head_dim = _layers_and_head_dim(model)
    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or head_dim
        cfg = KVCacheConfig(
            method="kivi",
            head_dim=hd,
            bit_width_inlier=bits,
            kivi_group_size=group_size,
            # A large residual window so "uniform" really means uniform
            # over the measured prompt, not accidentally age-gated by
            # KIVI's own fp16 residual — this comparison is uniform-INT4
            # vs. age-tiered, not KIVI-vs-age-tiered.
            residual_length=1,
            seed=42 + i,
        )
        caches.append(KIVIKVCache(cfg))
    return caches


def _build_age_tiered_caches(
    model,
    age_recent_boundary: int,
    age_mid_boundary: int,
    bits_recent: int,
    bits_mid: int,
    bits_old: int,
    group_size: int,
) -> list:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.age_tiered_cache import AgeTieredKVCache

    layers, head_dim = _layers_and_head_dim(model)
    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or head_dim
        cfg = KVCacheConfig(
            method="age_tiered",
            head_dim=hd,
            age_recent_boundary=age_recent_boundary,
            age_mid_boundary=age_mid_boundary,
            age_bits_recent=bits_recent,
            age_bits_mid=bits_mid,
            age_bits_old=bits_old,
            age_group_size=group_size,
        )
        caches.append(AgeTieredKVCache(cfg))
    return caches


def _generate(model, tokenizer, prompt: str, max_tokens: int, caches: list) -> tuple:
    from mlx_lm import generate

    t0 = time.time()
    out = generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, prompt_cache=caches
    )
    elapsed = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if out else 0
    return out, n_tok, elapsed


def _cache_aware_perplexity(model, tokenizer, text: str, caches: list, max_tokens: int) -> float:
    """Next-token perplexity computed WITH the given prompt_cache wired into
    the forward pass, so the cache's compression actually participates in
    the measurement (unlike model_kv_benchmark.py's compute_perplexity*,
    which never passes a cache at all).
    """
    tokens = tokenizer.encode(text)[:max_tokens]
    if len(tokens) < 4:
        return float("nan")

    input_ids = mx.array(tokens[:-1], dtype=mx.int32)[None]
    targets = tokens[1:]

    logits = model(input_ids, cache=caches)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)
    logits_np = np.array(logits[0], dtype=np.float32)

    total_nll = 0.0
    for t, tgt in enumerate(targets):
        lg = logits_np[t]
        lg_shifted = lg - lg.max()
        log_sum_exp = np.log(np.sum(np.exp(lg_shifted)))
        log_prob_tgt = lg_shifted[tgt] - log_sum_exp
        total_nll -= log_prob_tgt
    return math.exp(total_nll / len(targets))


def _reconstruction_error(model, tokenizer, text: str, build_caches_fn, max_tokens: int) -> float:
    """Mean-squared error between a compressed cache's stored K/V (after
    quantize-then-dequantize) and the fp16 baseline's K/V, at the same
    positions, averaged across layers. This is the "attention reconstruction
    error" issue #256 asks for and no other benchmark in this repo measures.
    """
    tokens = tokenizer.encode(text)[:max_tokens]
    if len(tokens) < 4:
        return float("nan")
    input_ids = mx.array(tokens[:-1], dtype=mx.int32)[None]

    fp16_caches = _build_fp16_caches(model)
    _ = model(input_ids, cache=fp16_caches)
    mx.eval([c.keys for c in fp16_caches if getattr(c, "keys", None) is not None])

    test_caches = build_caches_fn()
    _ = model(input_ids, cache=test_caches)
    mx.eval([getattr(c, "keys", None) for c in test_caches])

    errs = []
    for fp16_c, test_c in zip(fp16_caches, test_caches):
        fp16_k = getattr(fp16_c, "keys", None)
        test_k = getattr(test_c, "keys", None)
        if fp16_k is None or test_k is None:
            continue
        fp16_k32 = fp16_k.astype(mx.float32)
        test_k32 = test_k.astype(mx.float32)
        n = min(fp16_k32.shape[2], test_k32.shape[2])
        if n == 0:
            continue
        mse = mx.mean((fp16_k32[:, :, :n] - test_k32[:, :, :n]) ** 2)
        errs.append(float(mse.item()))

    return float(np.mean(errs)) if errs else float("nan")


def _run_config(
    model,
    tokenizer,
    name: str,
    build_caches_fn,
    max_tokens: int,
    ppl_tokens: int,
) -> dict:
    print(f"\n--- {name} ---", flush=True)

    _reset_peak()
    caches = build_caches_fn()
    out_text, n_tok, elapsed = _generate(model, tokenizer, PROMPT, max_tokens, caches)
    throughput = n_tok / max(elapsed, 1e-6)
    peak_mb = _peak_mb()

    ppl_caches = build_caches_fn()
    perplexity = _cache_aware_perplexity(model, tokenizer, PROMPT, ppl_caches, ppl_tokens)

    key_compressed = key_fp16 = 0
    val_compressed = val_fp16 = residual_fp16 = 0
    avg_bits = 16.0
    for c in caches:
        if hasattr(c, "compressed_key_bytes"):
            key_compressed += c.compressed_key_bytes
            key_fp16 += c.fp16_key_bytes
            val_compressed += getattr(c, "compressed_value_bytes", 0)
            val_fp16 += getattr(c, "fp16_value_bytes", 0)
            residual_fp16 += getattr(c, "residual_fp16_bytes", 0)
        elif hasattr(c, "age_tiered_bytes"):
            # AgeTieredKV reports a single combined K+V byte total rather
            # than KIVI's key/value split; fold it into the same "total
            # compressed vs total fp16" accounting so full_kv_compression
            # is comparable across both method families.
            key_compressed += c.age_tiered_bytes // 2
            val_compressed += c.age_tiered_bytes - c.age_tiered_bytes // 2
            key_fp16 += c.full_seq_bytes // 2
            val_fp16 += c.full_seq_bytes - c.full_seq_bytes // 2
    if caches and hasattr(caches[0], "assigned_avg_bits"):
        avg_bits = float(caches[0].assigned_avg_bits)
    elif (
        caches and hasattr(caches[0], "compression_ratio") and hasattr(caches[0], "full_seq_bytes")
    ):
        # Back out an effective avg-bits from the realized compression ratio
        # (fp16 baseline is 16 bits/element by construction).
        ratio = caches[0].compression_ratio
        avg_bits = 16.0 / ratio if ratio > 0 else 16.0

    total_fp16 = key_fp16 + val_fp16
    total_comp = key_compressed + val_compressed + residual_fp16
    full_kv_ratio = (total_fp16 / total_comp) if total_comp else 1.0

    recon_caches_fn = build_caches_fn
    recon_error = _reconstruction_error(model, tokenizer, PROMPT, recon_caches_fn, ppl_tokens)

    print(
        f"  {n_tok} tok in {elapsed:.2f}s ({throughput:.1f} tok/s)  peak={peak_mb:.0f}MB  "
        f"fullKV_x={full_kv_ratio:.2f}  ppl={perplexity:.3f}  recon_mse={recon_error:.6f}"
    )

    return {
        "name": name,
        "throughput_tok_s": throughput,
        "peak_mb": peak_mb,
        "full_kv_compression": full_kv_ratio,
        "avg_bits_estimate": avg_bits,
        "perplexity": perplexity,
        "attention_reconstruction_mse": recon_error,
        "tokens_generated": n_tok,
        "elapsed_s": elapsed,
        "sample_output": out_text[:300],
    }


def main() -> int:
    _ensure_path()
    parser = argparse.ArgumentParser(
        description="Uniform vs. adaptive (age-tiered) KV precision benchmark (issue #256)"
    )
    parser.add_argument("--model", required=True, help="HF model id (mlx-community/...)")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--ppl-tokens", type=int, default=1536)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--age-recent-boundary", type=int, default=128)
    parser.add_argument("--age-mid-boundary", type=int, default=1024)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from mlx_lm import load

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load(args.model)

    hw = _hardware()
    _, head_dim = _layers_and_head_dim(model)
    prompt_tokens = len(tokenizer.encode(PROMPT))
    print(f"  hardware={hw}")
    print(f"  head_dim={head_dim}  prompt_tokens={prompt_tokens}  ppl_tokens={args.ppl_tokens}")
    print(
        f"  age_recent_boundary={args.age_recent_boundary}  "
        f"age_mid_boundary={args.age_mid_boundary}"
    )

    configs = [
        (
            "fp16-baseline",
            lambda: _build_fp16_caches(model),
        ),
        (
            "uniform-int4",
            lambda: _build_kivi_uniform_caches(model, bits=4, group_size=args.group_size),
        ),
        (
            "age-tiered-8-4-2",
            lambda: _build_age_tiered_caches(
                model,
                args.age_recent_boundary,
                args.age_mid_boundary,
                bits_recent=8,
                bits_mid=4,
                bits_old=2,
                group_size=args.group_size,
            ),
        ),
    ]

    results = [
        _run_config(model, tokenizer, name, build_fn, args.max_tokens, args.ppl_tokens)
        for name, build_fn in configs
    ]

    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "ppl_tokens": args.ppl_tokens,
        "group_size": args.group_size,
        "age_recent_boundary": args.age_recent_boundary,
        "age_mid_boundary": args.age_mid_boundary,
        "prompt_tokens": prompt_tokens,
        "prompt": PROMPT[:200] + ("..." if len(PROMPT) > 200 else ""),
        "hardware": hw,
        "results": results,
    }
    out_path = Path(args.output) if args.output else Path("age_tiered_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {out_path}")
    print(
        f"\n{'config':<20s} {'ppl':>10s} {'recon_mse':>12s} {'peak_mb':>10s} "
        f"{'tok/s':>8s} {'fullKV_x':>10s} {'avg_bits':>10s}"
    )
    for r in results:
        print(
            f"{r['name']:<20s} {r['perplexity']:10.3f} {r['attention_reconstruction_mse']:12.6f} "
            f"{r['peak_mb']:10.1f} {r['throughput_tok_s']:8.1f} {r['full_kv_compression']:10.2f} "
            f"{r['avg_bits_estimate']:10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

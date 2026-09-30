"""Fused KIVI decode-attend vs. the decode paths production actually runs.

benchmark_real_model_scalar_attend.py compares scalar_fused_decode_attend
against dequantizing the whole quantized history to fp16 on every step.
Production never does that: KIVIKVCache quantizes-then-dequantizes into its
fp16 buffer once and runs plain SDPA, which costs the same as a plain fp16
KVCache. This script times the fused path (same harness cache and SDPA
patch) against:

  * plain fp16 ``KVCache`` -- what KIVIKVCache's decode actually costs;
  * mlx_lm's ``QuantizedKVCache(bits=4)`` -- the built-in cache that does
    save live memory.

The fused arm attends only over whole quantized groups (the harness cache
has no fp16 residual), i.e. over fewer tokens than the other arms, which
favors it. See docs/KV_KERNEL_ROOFLINE_FINDINGS.md for results.

Usage (run as a module so the harness import resolves):
    python -m benchmark_scripts.benchmark_fused_decode_vs_production
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from benchmark_scripts.benchmark_real_model_scalar_attend import (
    MODEL_ID,
    _make_caches,
    _patch_sdpa_for_scalar_attend,
    _unpatch_sdpa,
)

CONFIGS = [(1, 256), (1, 2048), (1, 4096), (4, 2048), (16, 256), (16, 1024)]
N_DECODE = 30


def _run(model, ids, make_cache, B, n_decode):
    caches = make_cache()
    mx.clear_cache()
    mx.reset_peak_memory()
    logits = model(mx.array([ids] * B), cache=caches)
    nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(nxt)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        logits = model(nxt, cache=caches)
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(nxt)
    mx.synchronize()
    return n_decode * B / (time.perf_counter() - t0), mx.get_peak_memory() / 1e9


def main() -> None:
    model, tokenizer = load(MODEL_ID)
    text = (
        "The history of artificial intelligence began in antiquity, with myths and legends "
        "of artificial beings endowed with intelligence or consciousness by master craftsmen. "
    ) * 400
    base_ids = tokenizer.encode(text)

    arms = {
        "fp16 KVCache": (lambda: [KVCache() for _ in model.layers], False),
        "mlx QuantizedKVCache 4b": (
            lambda: [QuantizedKVCache(group_size=64, bits=4) for _ in model.layers],
            False,
        ),
        "fused KIVI 2b": (lambda: _make_caches(model), True),
    }

    print(f"{MODEL_ID}, {N_DECODE} decode steps, MLX {mx.__version__}\n")
    print("| B | prompt | arm | decode tok/s | vs fp16 | peak GB |")
    print("|---|---|---|---|---|---|")
    for B, L in CONFIGS:
        ids = (base_ids * (L // len(base_ids) + 1))[:L]
        fp16_tps = None
        for name, (make_cache, fused) in arms.items():
            if fused:
                _patch_sdpa_for_scalar_attend(model, nsg=None, use_fused=True)
            try:
                _run(model, ids[:256], make_cache, B, n_decode=3)  # warmup / JIT
                tps, peak = _run(model, ids, make_cache, B, N_DECODE)
            finally:
                _unpatch_sdpa()
            fp16_tps = fp16_tps or tps
            print(
                f"| {B} | {L} | {name} | {tps:.1f} | {tps / fp16_tps:.2f}x | {peak:.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()

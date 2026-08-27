"""Benchmark: mlx_lm.generate() driven through PoolBackedKVCache vs stock KVCache.

This is the end-to-end benchmark `benchmark_block_pool_real_model.py`
couldn't do: that script could only run the allocator *alongside* a real
generation (same shape, unrelated call), because PoolBackedKVCache did not
exist yet — PooledKVCache only wraps VeloxQuant's own standalone-method
cache interface, which mlx_lm.generate() never drives.

PoolBackedKVCache implements mlx_lm's update_and_fetch protocol directly
(see veloxquant_mlx/memory/pool_backed_cache.py), so it plugs into
mlx_lm.generate() as a prompt_cache exactly like a stock KVCache. This
script runs the *same* prompt/model through both and reports:

  - generation tokens/sec and peak memory for each (should be near-identical
    — the pool changes allocation bookkeeping, not the math)
  - the pool's own AllocationStats for a single request
  - allocation-count / reuse stats across two *sequential* requests sharing
    one pool, to demonstrate real cross-request block reuse on a real model

Usage::

    python benchmark_scripts/benchmark_pool_backed_kvcache.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit --max-tokens 200

Writes figures/block_pool/pool_backed_kvcache_results.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, build_pooled_caches

PROMPT = (
    "Explain the theory of relativity in simple terms, covering both "
    "special and general relativity with examples a high-school student "
    "could follow."
)


def _hardware() -> dict:
    """Best-effort hardware record (chip + RAM) for honest provenance."""
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


def _run_generation(model, tokenizer, prompt_cache, max_tokens: int) -> dict:
    mx.reset_peak_memory()
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    t0 = time.perf_counter()
    last = None
    text = ""
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prompt_cache=prompt_cache
    ):
        last = response
        text += response.text  # response.text is a per-token delta, not cumulative
    elapsed = time.perf_counter() - t0

    return {
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tokens": last.generation_tokens,
        "generation_tps": last.generation_tps,
        "wall_clock_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "text_preview": text[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-blocks", type=int, default=8192)
    parser.add_argument("--out-dir", type=str, default="figures/block_pool")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)

    print(f"Running mlx_lm.generate() with a stock KVCache (max_tokens={args.max_tokens}) ...")
    stock_cache = make_prompt_cache(model)
    stock = _run_generation(model, tokenizer, stock_cache, args.max_tokens)
    print(
        f"  stock:  generation={stock['generation_tokens']} tok @ "
        f"{stock['generation_tps']:.1f} tok/s | peak={stock['peak_memory_gb']:.3f} GB"
    )

    print("Running the same generation through PoolBackedKVCache (request A) ...")
    pool = BlockPoolAllocator(PoolConfig(block_size=args.block_size, n_blocks=args.n_blocks))
    cache_a = build_pooled_caches(model, pool, owner=1)
    pooled_a = _run_generation(model, tokenizer, cache_a, args.max_tokens)
    stats_after_a = repr(pool.stats)
    print(
        f"  pooled (req A): generation={pooled_a['generation_tokens']} tok @ "
        f"{pooled_a['generation_tps']:.1f} tok/s | peak={pooled_a['peak_memory_gb']:.3f} GB"
    )
    print(f"  pool stats after request A: {stats_after_a}")
    cache_a[0].release()
    stats_after_release_a = repr(pool.stats)
    print(f"  pool stats after releasing A: {stats_after_release_a}")

    print("Running a second, sequential request through the same pool (request B) ...")
    n_allocations_before_b = pool.stats.n_allocations
    n_reused_before_b = pool.stats.n_reused
    cache_b = build_pooled_caches(model, pool, owner=2)
    pooled_b = _run_generation(model, tokenizer, cache_b, args.max_tokens)
    stats_after_b = pool.stats
    print(f"  pool stats after request B: {stats_after_b}")
    b_allocations = stats_after_b.n_allocations - n_allocations_before_b
    b_reused = stats_after_b.n_reused - n_reused_before_b
    reuse_fraction = b_reused / b_allocations if b_allocations else 0.0
    print(
        f"  request B: {b_allocations} allocations, {b_reused} satisfied by "
        f"reuse ({reuse_fraction:.1%})"
    )
    cache_b[0].release()

    payload = {
        "model": args.model,
        "block_size": args.block_size,
        "n_blocks": args.n_blocks,
        "hardware": _hardware(),
        "note": (
            "stock and pooled_request_a run the identical prompt/max_tokens "
            "through mlx_lm.generate() with a stock KVCache vs. "
            "PoolBackedKVCache respectively -- generation_tps and "
            "peak_memory_gb should be near-identical between them, since "
            "the pool changes allocation bookkeeping (which shows up in "
            "pool_stats_*), not the attention/decode math. "
            "pooled_request_b is a second, sequential request sharing the "
            "same pool after request A released its blocks -- "
            "reuse_fraction_request_b is the fraction of request B's "
            "allocations that came from blocks request A had freed, "
            "measured on a real model's real generation rather than a "
            "synthetic replay."
        ),
        "stock": stock,
        "pooled_request_a": pooled_a,
        "pool_stats_after_request_a": stats_after_a,
        "pool_stats_after_releasing_a": stats_after_release_a,
        "pooled_request_b": pooled_b,
        "pool_stats_after_request_b": repr(stats_after_b),
        "request_b_allocations": b_allocations,
        "request_b_reused": b_reused,
        "reuse_fraction_request_b": reuse_fraction,
    }
    json_path = out_dir / "pool_backed_kvcache_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {json_path}")


if __name__ == "__main__":
    main()

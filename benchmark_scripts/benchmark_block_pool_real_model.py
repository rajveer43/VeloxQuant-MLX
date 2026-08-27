"""Benchmark: KV-cache block pool allocator sized to a real model's generation.

`benchmark_block_pool.py` compares the allocator against a naive baseline
on an arbitrary synthetic (head_dim, sequence length) — useful for a quick
apples-to-apples allocator comparison, but not tied to any real model's
actual KV-cache growth pattern.

This script instead:

  1. Loads a real `mlx_lm` model and runs `mlx_lm.generate()` normally
     (no VeloxQuant cache wiring — plain fp16 mlx_lm caches), recording
     real prompt length, generation length, generation tokens/sec, and
     `mx.get_peak_memory()`.
  2. Reads the model's real per-layer shape (n_layers, n_kv_heads,
     head_dim) off the loaded model.
  3. Replays that exact (n_layers, n_kv_heads, head_dim, generation_length)
     profile through BlockPoolAllocator and the naive per-token baseline,
     one pool per KV head per layer (mirroring how mlx_lm allocates one
     cache per layer today), to get allocation-count / reuse /
     fragmentation numbers that reflect this model's real decode length
     rather than an arbitrary synthetic one.

Why not run mlx_lm.generate() *through* the pool directly: PooledKVCache
wraps VeloxQuant's own KVCache ABC (append_key/append_value/attend), which
only the 5 "standalone" methods (turboquant_prod/mse, polar, qjl, spectral
— see STANDALONE_METHODS in veloxquant_mlx/cache/base.py) implement.
mlx_lm.generate() drives caches via a different protocol
(update_and_fetch), so there is no existing integration that puts the pool
directly in a real model's decode hot path yet. See CHANGELOG.md for the
follow-up this benchmark's results.json documents.

Usage::

    python benchmark_scripts/benchmark_block_pool_real_model.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit --max-tokens 256

Writes figures/block_pool/real_model_results.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import mlx_lm

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig

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


def _model_shape(model) -> dict:
    """Extract (n_layers, n_kv_heads, head_dim) from a loaded mlx_lm model."""
    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)

    first_attn = None
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is not None:
            first_attn = attn
            break

    head_dim = getattr(first_attn, "head_dim", None) or (
        args.hidden_size // args.num_attention_heads
    )
    n_kv_heads = getattr(args, "num_key_value_heads", None) or getattr(
        args, "num_attention_heads", 1
    )
    return {
        "n_layers": len(layers),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
    }


def _run_generation(model, tokenizer, max_tokens: int) -> dict:
    mx.reset_peak_memory()
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    t0 = time.perf_counter()
    last = None
    for response in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        last = response
    elapsed = time.perf_counter() - t0

    return {
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tokens": last.generation_tokens,
        "generation_tps": last.generation_tps,
        "wall_clock_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def _replay_naive(n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int) -> dict:
    """One fresh mx.array buffer per token per (layer, kv_head) pair, no reuse.

    Mirrors how each of the ~40 veloxquant_mlx/cache/*.py classes today
    pre-allocates its own capacity-sized buffer independently, and how a
    plain mlx_lm KVCache grows via repeated concatenation — no shared,
    reusable storage across layers or across requests.
    """
    n_streams = n_layers * n_kv_heads  # one K (or V) buffer series per (layer, head)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    n_allocations = 0
    for _ in range(n_streams):
        for _ in range(n_tokens):
            mx.zeros((1, head_dim), dtype=mx.float16)
            n_allocations += 1
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_tokens = n_streams * n_tokens
    return {
        "label": "naive-per-token-alloc",
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else float("inf"),
        "n_allocations": n_allocations,
        "allocations_per_request": n_allocations / n_streams,
        "python_peak_bytes": py_peak,
        "fragmentation": 0.0,
    }


def _replay_pooled(
    n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int, block_size: int
) -> dict:
    """Same (layer, kv_head) x token workload, but through the block pool.

    One pool per (layer, head)-stream would defeat the point of pooling
    (no cross-stream reuse); instead every stream shares one pool sized to
    this model's real n_layers * n_kv_heads concurrency, checking blocks
    out/in as each stream's generation proceeds — the way a real
    multi-layer decode step advances every layer's cache by one token in
    lock-step.
    """
    n_streams = n_layers * n_kv_heads
    n_blocks_per_stream = -(-n_tokens // block_size) + 1
    pool = BlockPoolAllocator(
        PoolConfig(
            block_size=block_size,
            n_blocks=n_blocks_per_stream * n_streams,
            separate_kv=False,
        )
    )

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    used_in_block = {s: 0 for s in range(n_streams)}
    current_block = {s: None for s in range(n_streams)}
    for _ in range(n_tokens):
        for s in range(n_streams):
            if current_block[s] is None or used_in_block[s] == block_size:
                (current_block[s],) = pool.allocate(stream="kv", n_tokens=1, owner=s, format="fp16")
                used_in_block[s] = 0
            used_in_block[s] += 1
            current_block[s].n_used = used_in_block[s]
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_tokens = n_streams * n_tokens
    return {
        "label": "block-pool",
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else float("inf"),
        "n_allocations": pool.stats.n_allocations,
        "allocations_per_request": pool.stats.n_allocations / n_streams,
        "n_reused": pool.stats.n_reused,
        "n_exhausted": pool.stats.n_exhausted,
        "peak_blocks_in_use": pool.stats.peak_blocks_in_use,
        "python_peak_bytes": py_peak,
        "fragmentation": pool.stats.fragmentation(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="figures/block_pool")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)
    shape = _model_shape(model)
    print(
        f"  n_layers={shape['n_layers']}, n_kv_heads={shape['n_kv_heads']}, "
        f"head_dim={shape['head_dim']}"
    )

    print(f"Running mlx_lm.generate() (max_tokens={args.max_tokens}) ...")
    gen = _run_generation(model, tokenizer, args.max_tokens)
    print(
        f"  prompt={gen['prompt_tokens']} tok @ {gen['prompt_tps']:.1f} tok/s | "
        f"generation={gen['generation_tokens']} tok @ {gen['generation_tps']:.1f} tok/s | "
        f"peak={gen['peak_memory_gb']:.3f} GB"
    )

    n_tokens = gen["generation_tokens"]
    print(
        f"Replaying allocator workload at this model's real shape "
        f"(n_streams={shape['n_layers'] * shape['n_kv_heads']}, n_tokens={n_tokens}) ..."
    )
    naive = _replay_naive(shape["n_layers"], shape["n_kv_heads"], shape["head_dim"], n_tokens)
    pooled = _replay_pooled(
        shape["n_layers"], shape["n_kv_heads"], shape["head_dim"], n_tokens, args.block_size
    )

    payload = {
        "model": args.model,
        "block_size": args.block_size,
        "model_shape": shape,
        "hardware": _hardware(),
        "generation": gen,
        "note": (
            "generation.* is real mlx_lm.generate() decode throughput/memory on "
            "plain fp16 caches (no VeloxQuant cache wiring — none of the pool "
            "code sits in this decode's hot path). allocator_replay.* re-plays "
            "the block pool and naive baseline at this model's *real* "
            "(n_layers, n_kv_heads, head_dim, generation_length), so the "
            "allocation-count/reuse/fragmentation numbers reflect a real "
            "model's real decode length rather than an arbitrary synthetic "
            "one. tokens_per_sec under allocator_replay is allocator-side "
            "append throughput on synthetic zero-filled vectors, not "
            "generation.generation_tps; mx.zeros() is lazy/cheap in MLX so "
            "the naive baseline's raw allocator throughput here is not "
            "penalized for the malloc/free churn a real allocator would pay "
            "under memory pressure or an eager backend — read "
            "allocations_per_request and n_reused as the honest signal, the "
            "same caveat as benchmark_block_pool.py."
        ),
        "allocator_replay": [naive, pooled],
    }
    json_path = out_dir / "real_model_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {json_path}")
    for r in payload["allocator_replay"]:
        print(
            f"  {r['label']:>22}: {r['n_allocations']:>8} allocs "
            f"({r['allocations_per_request']:.2f}/stream) | "
            f"frag={r['fragmentation']:.3f}"
            + (f" | reused={r['n_reused']}" if "n_reused" in r else "")
        )


if __name__ == "__main__":
    main()

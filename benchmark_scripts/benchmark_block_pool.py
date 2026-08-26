"""Benchmark: KV-cache block pool allocator vs naive per-request allocation.

Issue #249 asks for a KV-cache-specific block pool allocator and a
comparison against the current (naive) implementation on:

  - allocation latency
  - peak memory usage
  - fragmentation
  - tokens/sec
  - number of allocations per request

This benchmark is deliberately model-free: allocator behavior does not
depend on which quantization method or LLM is in use, and a synthetic
multi-request decode workload lets the comparison run on any machine
without downloading a checkpoint. Reported "tokens/sec" measures how many
token-appends the allocator can service per second (its own overhead),
not end-to-end model decode throughput — see the printed caveat and the
`note` field in results.json.

Usage::

    python benchmark_scripts/benchmark_block_pool.py --n-requests 200 \\
        --tokens-per-request 256 --head-dim 128

Writes figures/block_pool/results.json.
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

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.mlx_storage import MLXBlockStorage


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


class _NaiveAllocator:
    """Baseline: one fresh mx.array buffer per request, grown on demand.

    Mirrors the "current implementation" issue #249 compares against — each
    per-layer cache in this codebase pre-allocates its own capacity-sized
    numpy/mx buffer independently (see e.g. TurboQuantKVCache.__init__),
    with no cross-request reuse of freed storage.
    """

    def __init__(self, head_dim: int) -> None:
        self.head_dim = head_dim
        self.n_allocations = 0
        self._live: dict[int, list] = {}

    def allocate(self, owner: int, n_tokens: int) -> None:
        # One new buffer per allocate() call, exactly n_tokens rows — no
        # rounding to a block, no reuse of any previously freed buffer.
        buf = mx.zeros((n_tokens, self.head_dim), dtype=mx.float16)
        self._live.setdefault(owner, []).append(buf)
        self.n_allocations += 1

    def free_all(self, owner: int) -> None:
        self._live.pop(owner, None)  # buffers become garbage, no reuse

    def resident_bytes(self) -> int:
        total = 0
        for bufs in self._live.values():
            for b in bufs:
                total += b.shape[0] * b.shape[1] * 2
        return total


def _run_naive(n_requests: int, tokens_per_request: int, head_dim: int) -> dict:
    alloc = _NaiveAllocator(head_dim)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    peak_resident = 0
    for req in range(n_requests):
        # One allocate() call per token appended — no batching, matching the
        # "malloc on every append" behavior a block pool eliminates.
        for _ in range(tokens_per_request):
            alloc.allocate(owner=req, n_tokens=1)
        peak_resident = max(peak_resident, alloc.resident_bytes())
        alloc.free_all(owner=req)
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_tokens_total = n_requests * tokens_per_request
    return {
        "label": "naive-per-token-alloc",
        "elapsed_s": elapsed,
        "tokens_per_sec": n_tokens_total / elapsed if elapsed > 0 else float("inf"),
        "n_allocations": alloc.n_allocations,
        "allocations_per_request": alloc.n_allocations / n_requests,
        "peak_resident_bytes": peak_resident,
        "python_peak_bytes": py_peak,
        "fragmentation": 0.0,  # no block concept -> not applicable
    }


def _run_pooled(n_requests: int, tokens_per_request: int, head_dim: int, block_size: int) -> dict:
    # Sized to exactly fit one request's worth of K blocks resident at a time
    # plus slack, so free_all() lets the next request's blocks come from the
    # just-freed set (the reuse path) instead of ever growing the pool.
    n_blocks_per_stream = -(-tokens_per_request // block_size) + 2
    pool = BlockPoolAllocator(
        PoolConfig(block_size=block_size, n_blocks=n_blocks_per_stream * 2, separate_kv=True)
    )
    storage = MLXBlockStorage(pool, head_dim=head_dim)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    peak_resident = 0
    row = mx.zeros((1, head_dim), dtype=mx.float16)
    for req in range(n_requests):
        used_in_block = 0
        blocks: list = []
        for _ in range(tokens_per_request):
            if not blocks or used_in_block == block_size:
                (block,) = pool.allocate(stream="k", n_tokens=1, owner=req, format="fp16")
                blocks.append(block)
                used_in_block = 0
            storage.write(blocks[-1], offset=used_in_block, values=row)
            used_in_block += 1
            blocks[-1].n_used = used_in_block
        peak_resident = max(peak_resident, storage.resident_bytes())
        pool.free_all(owner=req)
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_tokens_total = n_requests * tokens_per_request
    return {
        "label": "block-pool",
        "elapsed_s": elapsed,
        "tokens_per_sec": n_tokens_total / elapsed if elapsed > 0 else float("inf"),
        "n_allocations": pool.stats.n_allocations,
        "allocations_per_request": pool.stats.n_allocations / n_requests,
        "n_reused": pool.stats.n_reused,
        "n_exhausted": pool.stats.n_exhausted,
        "peak_blocks_in_use": pool.stats.peak_blocks_in_use,
        "peak_resident_bytes": peak_resident,
        "python_peak_bytes": py_peak,
        "fragmentation": pool.stats.fragmentation(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--tokens-per-request", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="figures/block_pool")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running naive vs block-pool allocator: {args.n_requests} requests x "
        f"{args.tokens_per_request} tokens, head_dim={args.head_dim}, "
        f"block_size={args.block_size}"
    )

    naive = _run_naive(args.n_requests, args.tokens_per_request, args.head_dim)
    pooled = _run_pooled(args.n_requests, args.tokens_per_request, args.head_dim, args.block_size)

    payload = {
        "n_requests": args.n_requests,
        "tokens_per_request": args.tokens_per_request,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "hardware": _hardware(),
        "note": (
            "tokens_per_sec measures allocator-side append throughput on "
            "synthetic zero-filled vectors, not end-to-end mlx_lm decode "
            "throughput. Use it to compare allocator overhead only. On this "
            "microbenchmark mx.zeros() is lazy/cheap, so the naive baseline's "
            "raw tok/s is not penalized for the malloc/free churn a real "
            "allocator would pay under memory pressure or with eager backends; "
            "the metrics that isolate the pool's actual benefit are "
            "allocations_per_request (drops by ~block_size x) and n_reused "
            "(steady-state allocations satisfied from freed blocks instead of "
            "new memory)."
        ),
        "results": [naive, pooled],
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {json_path}")
    for r in payload["results"]:
        print(
            f"  {r['label']:>22}: {r['tokens_per_sec']:>10,.0f} tok/s | "
            f"{r['n_allocations']:>8} allocs ({r['allocations_per_request']:.2f}/req) | "
            f"peak={r['peak_resident_bytes'] / 1024:.1f} KiB | "
            f"frag={r['fragmentation']:.3f}"
        )


if __name__ == "__main__":
    main()

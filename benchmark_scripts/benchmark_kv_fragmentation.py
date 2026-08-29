"""Benchmark: KV-cache memory fragmentation under long-context generation.

Issue #254 notes that compression ratio alone does not capture actual
memory efficiency: a theoretically compressed cache can still consume much
more resident memory than its logical size because of allocation
granularity, padding, temporary buffers, and fragmentation.

This benchmark grows a *single* KV-cache sequence (mirroring real
prompt-processing / long-context decode, not many short requests) out to
each of several context lengths and reports, for both a naive
per-token-malloc allocator and the block-pool allocator:

  - logical_bytes:      exact bytes needed for the tokens actually stored
                         (n_tokens * head_dim * bytes_per_element)
  - physical_bytes:     bytes actually resident in allocated buffers,
                         including block padding
  - fragmentation:      physical_bytes / logical_bytes - 1 (0 == no waste)
  - temp_peak_bytes:    largest *transient* rise in MLX's own peak resident
                         memory (mx.get_peak_memory()) observed during a
                         single grow step, capturing scratch/copy buffers a
                         naive reallocating strategy pays for that a pool
                         avoids by writing in place
  - peak_physical_bytes: high-water mark of physical_bytes across the
                         whole run (not just the final size)

across context lengths: 1K, 4K, 8K, 16K, 32K(+) tokens.

This is deliberately model-free (allocator behavior does not depend on
which quantization method or LLM is in use), matching the approach in
benchmark_block_pool.py, so it runs on any machine without downloading a
checkpoint.

Usage::

    python benchmark_scripts/benchmark_kv_fragmentation.py \\
        --head-dim 128 --block-size 16

Writes figures/block_pool/fragmentation_results.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path

import mlx.core as mx

from veloxquant_mlx.memory.block_pool import BlockPoolAllocator, PoolConfig
from veloxquant_mlx.memory.mlx_storage import MLXBlockStorage

DEFAULT_CONTEXT_LENGTHS = (1024, 4096, 8192, 16384, 32768)

# Tokens appended per grow step. Chosen to be smaller than the block size
# so pool fragmentation (partially-filled trailing block) is visible, while
# still being coarse enough that a full 32K sweep runs quickly.
GROW_STEP_TOKENS = 8

BYTES_PER_ELEMENT_FP16 = 2


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


class _NaiveGrowingBuffer:
    """Baseline: reallocates a fresh, exactly-sized buffer on every grow step.

    Mirrors a from-scratch concatenation strategy (no pre-allocated
    capacity, no block reuse) — the "current implementation" comparison
    point issue #249 established and this benchmark extends across context
    lengths instead of across request counts.
    """

    def __init__(self, head_dim: int) -> None:
        self.head_dim = head_dim
        self.n_tokens = 0
        self.n_allocations = 0
        self._buf = None

    def grow(self, n_new_tokens: int) -> None:
        # Full copy-and-grow: allocate a new (n_tokens + n_new) buffer and
        # copy the old contents in, exactly what a naive concatenate-based
        # cache does on each append batch.
        new_buf = mx.zeros((self.n_tokens + n_new_tokens, self.head_dim), dtype=mx.float16)
        if self._buf is not None:
            new_buf[: self.n_tokens] = self._buf
        mx.eval(new_buf)
        self._buf = new_buf
        self.n_tokens += n_new_tokens
        self.n_allocations += 1

    def physical_bytes(self) -> int:
        if self._buf is None:
            return 0
        return self._buf.shape[0] * self._buf.shape[1] * BYTES_PER_ELEMENT_FP16


class _PooledGrowingBuffer:
    """Block-pool-backed growing sequence: appends allocate whole blocks only
    when the current tail block is full, and never reallocates existing data.
    """

    def __init__(self, head_dim: int, block_size: int, n_blocks: int) -> None:
        self.pool = BlockPoolAllocator(
            PoolConfig(block_size=block_size, n_blocks=n_blocks, separate_kv=False)
        )
        self.storage = MLXBlockStorage(self.pool, head_dim=head_dim)
        self.owner = 0
        self.block_size = block_size
        self.n_tokens = 0
        self._blocks: list = []
        self._used_in_tail = 0

    def grow(self, n_new_tokens: int) -> None:
        row = mx.zeros((1, self.storage.head_dim), dtype=mx.float16)
        remaining = n_new_tokens
        while remaining > 0:
            if not self._blocks or self._used_in_tail == self.block_size:
                (block,) = self.pool.allocate(
                    stream="kv", n_tokens=1, owner=self.owner, format="fp16"
                )
                self._blocks.append(block)
                self._used_in_tail = 0
            self.storage.write(self._blocks[-1], offset=self._used_in_tail, values=row)
            self._used_in_tail += 1
            self._blocks[-1].n_used = self._used_in_tail
            remaining -= 1
        mx.eval(self.storage.buffer_for(self._blocks[-1]))
        self.n_tokens += n_new_tokens

    def physical_bytes(self) -> int:
        return self.storage.resident_bytes()


def _run_sweep_for_allocator(
    label: str,
    make_buffer,
    context_length: int,
    grow_step: int,
) -> dict:
    gc.collect()

    buf = make_buffer()
    peak_physical = 0
    temp_peak_bytes = 0
    n_tokens = 0
    while n_tokens < context_length:
        step = min(grow_step, context_length - n_tokens)
        physical_before = buf.physical_bytes()
        mx.reset_peak_memory()
        buf.grow(step)
        # Transient/scratch signal: how far MLX's actual peak resident
        # memory during this one grow step exceeded the buffer state
        # already resident going in. A naive copy-and-grow pays for the
        # old buffer plus the new (bigger) buffer simultaneously here; a
        # pool that writes in place pays only for the one new block.
        step_peak = mx.get_peak_memory()
        temp_peak_bytes = max(temp_peak_bytes, step_peak - physical_before)
        n_tokens += step
        peak_physical = max(peak_physical, buf.physical_bytes())

    head_dim = buf.storage.head_dim if isinstance(buf, _PooledGrowingBuffer) else buf.head_dim
    logical_bytes = n_tokens * head_dim * BYTES_PER_ELEMENT_FP16
    physical_bytes = buf.physical_bytes()
    fragmentation = (physical_bytes / logical_bytes - 1.0) if logical_bytes > 0 else 0.0

    result = {
        "label": label,
        "context_length": context_length,
        "n_tokens": n_tokens,
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "peak_physical_bytes": peak_physical,
        "temp_peak_bytes": temp_peak_bytes,
        "fragmentation": fragmentation,
        "n_allocations": getattr(buf, "n_allocations", None),
    }
    if isinstance(buf, _PooledGrowingBuffer):
        result["n_allocations"] = buf.pool.stats.n_allocations
        result["n_reused"] = buf.pool.stats.n_reused
        result["block_fragmentation"] = buf.pool.stats.fragmentation()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help="Context lengths (in tokens) to sweep.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--grow-step", type=int, default=GROW_STEP_TOKENS)
    parser.add_argument("--out-dir", type=str, default="figures/block_pool")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Sweeping KV-cache fragmentation across context lengths "
        f"{args.context_lengths}, head_dim={args.head_dim}, "
        f"block_size={args.block_size}"
    )

    results = []
    for context_length in args.context_lengths:
        n_blocks = -(-context_length // args.block_size) + 4  # headroom above exact fit

        naive = _run_sweep_for_allocator(
            "naive-realloc",
            lambda: _NaiveGrowingBuffer(args.head_dim),
            context_length,
            args.grow_step,
        )
        pooled = _run_sweep_for_allocator(
            "block-pool",
            lambda n_blocks=n_blocks: _PooledGrowingBuffer(
                args.head_dim, args.block_size, n_blocks
            ),
            context_length,
            args.grow_step,
        )
        results.append({"context_length": context_length, "naive": naive, "pooled": pooled})

        print(
            f"  {context_length:>6} tok | "
            f"naive: phys={naive['physical_bytes'] / 1024:.1f} KiB "
            f"frag={naive['fragmentation']:.3f} temp_peak={naive['temp_peak_bytes'] / 1024:.1f} KiB | "
            f"pool: phys={pooled['physical_bytes'] / 1024:.1f} KiB "
            f"frag={pooled['fragmentation']:.3f} temp_peak={pooled['temp_peak_bytes'] / 1024:.1f} KiB"
        )

    payload = {
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "grow_step": args.grow_step,
        "hardware": _hardware(),
        "note": (
            "logical_bytes is the exact fp16 footprint of the tokens stored "
            "(n_tokens * head_dim * 2). physical_bytes is what is actually "
            "resident in allocated buffers, including block/allocation "
            "padding. fragmentation = physical/logical - 1 (0 == no waste "
            "beyond the logical size). temp_peak_bytes is the largest "
            "*transient* rise in MLX's own peak resident memory "
            "(mx.get_peak_memory()) observed during a single grow step, "
            "capturing scratch/copy buffers a naive copy-and-grow strategy "
            "pays for on every append that a pool avoids by writing in "
            "place. naive-realloc reallocates and "
            "copies the whole sequence on every grow step (worst case for "
            "temporary memory); block-pool only allocates a new block when "
            "the current tail block is full, and never copies existing "
            "data. This benchmark is model-free and synthetic: it isolates "
            "allocator-level memory behavior from any particular "
            "quantization method's compression math."
        ),
        "results": results,
    }
    json_path = out_dir / "fragmentation_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults: {json_path}")


if __name__ == "__main__":
    main()

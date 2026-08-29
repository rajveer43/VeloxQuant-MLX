---
slug: kv-cache-fragmentation
title: "Compression Ratio Lies About Memory. Here's the Allocator That Doesn't."
date: 2026-08-29
authors: rajveer
tags: [memory, allocator, kv-cache, apple-silicon, mlx, benchmarking]
---

# Compression Ratio Lies About Memory. Here's the Allocator That Doesn't.

*A 5×–8× KV-cache compression ratio tells you almost nothing about resident memory if the allocator underneath it copies the whole sequence on every append. This is what happens when you measure the allocator instead of the arithmetic — on a synthetic sweep out to 32K tokens, and on six real models from 135M to 7B parameters.*

---

Every KV-cache method in this library reports a compression ratio: KIVI gets 4×–8×, TurboQuant gets more. Those numbers describe the *arithmetic* — how many bytes a token's key/value vector costs once quantized. They describe nothing about what actually sits resident in memory while the cache is growing.

Those are different questions. A cache that logically needs 256 KiB for 1024 tokens can still make your process's peak memory spike far past that, because of how the buffer holding those bytes got built. The compression ratio is deaf to this; a good allocator is the difference between "safe to run" and "swaps your Mac to death" at the same reported ratio.

This post is about the allocator side specifically — not compression math, not a quantization method, just: given a cache that's growing token by token, how much memory does the *mechanism of growing it* actually spend, and how does that scale with context length?

## The two allocation strategies

There are two honest ways to grow a sequence's KV storage:

**Copy-and-grow.** Keep one contiguous buffer. Every time you append tokens, allocate a new buffer sized for the whole sequence so far plus the new tokens, copy the old contents in, discard the old buffer. This is the naive baseline and — worth saying plainly — it's also what `mlx_lm`'s stock `KVCache` does, just with a fixed 256-token growth chunk instead of growing by exactly the new token count.

**Block pool.** Pre-allocate fixed-size blocks up front. Appending tokens writes into the current block; only allocate a new block when the current one fills up. Existing data is never copied — a full block, once written, is never touched again until it's freed. This library's [`BlockPoolAllocator`](/docs/api/memory-api) (issue #249, `veloxquant_mlx/memory/block_pool.py`) implements this, and [`PoolBackedKVCache`](/docs/api/memory-api#poolbackedkvcache) (`veloxquant_mlx/memory/pool_backed_cache.py`) wires it directly into `mlx_lm.generate()`'s `update_and_fetch` protocol as a drop-in replacement for the stock cache.

Copy-and-grow's problem isn't the final resident size — both strategies end up holding roughly the same number of logical bytes once the sequence stops growing. The problem is what happens *during* every single append: for one moment, both the old buffer and the new, bigger buffer are alive in memory simultaneously. That moment's cost scales with how much history you're copying, which means it scales with context length. A pool's per-append cost is one new block, a constant, regardless of how long the sequence already is.

That's a claim you can measure directly, so this benchmark measures it directly.

## Benchmark 1: synthetic, allocator only, out to 32K tokens

`benchmark_scripts/benchmark_kv_fragmentation.py` grows a single sequence — mirroring real prompt processing, not a swarm of short requests — out to each of 1K, 4K, 8K, 16K, and 32K tokens, using both strategies, and reports:

- **logical_bytes** — exact bytes the stored tokens need (`n_tokens × head_dim × 2` for fp16)
- **physical_bytes** — bytes actually resident, including any block padding
- **fragmentation** — `physical / logical - 1`
- **temp_peak_bytes** — the largest *transient* rise in MLX's own peak resident memory (`mx.get_peak_memory()`) during a single append, isolating the scratch/copy cost from the steady-state footprint
- **peak_physical_bytes** — high-water mark across the whole run

It's deliberately model-free — allocator behavior doesn't depend on which quantization method or LLM is in use — so it runs anywhere without a checkpoint download, following the same approach as the existing block-pool benchmark from issue #249.

Steady-state resident size is identical between the two strategies at every context length, as expected — that part was never in question:

| context | naive physical | pool physical |
|---|---|---|
| 1,024 tok | 256.0 KiB | 256.0 KiB |
| 4,096 tok | 1,024.0 KiB | 1,024.0 KiB |
| 8,192 tok | 2,048.0 KiB | 2,048.0 KiB |
| 16,384 tok | 4,096.0 KiB | 4,096.0 KiB |
| 32,768 tok | 8,192.0 KiB | 8,192.0 KiB |

The transient cost — what each *append* pays, not what the sequence holds at rest — is where the two strategies diverge, and it diverges a lot:

| context | naive temp_peak | pool temp_peak | ratio |
|---|---|---|---|
| 1,024 tok | 514.0 KiB | 6.3 KiB | **82×** |
| 4,096 tok | 1,958.0 KiB | 6.3 KiB | **313×** |
| 8,192 tok | 3,948.0 KiB | 6.3 KiB | **630×** |
| 16,384 tok | 8,200.0 KiB | 6.3 KiB | **1,310×** |
| 32,768 tok | 14,206.0 KiB | 6.3 KiB | **2,269×** |

The pool's transient cost is flat — one new block, regardless of how much history already exists. The naive strategy's transient cost grows with context length, because "copy the whole sequence" gets more expensive the longer the sequence is. This is the O(n) tax that a compression-ratio number simply never sees: it lives entirely in the mechanics of growing the buffer, not in how many bits each element costs.

The gap is roughly linear in context length, which is exactly what "copy the whole history on every append" predicts. At 32K tokens the naive path's single append briefly holds ~14 MB of scratch state for a cache whose steady-state footprint is 8 MB — nearly double, for one call, on every single token.

**Fragmentation**, separately: with the default settings (block_size=16 dividing every default context length evenly), the pool shows 0% padding waste — no free lunch was hidden by rounding. Feeding it a context length that doesn't divide evenly makes the real cost visible: at 999 tokens with block_size=16, the pool rounds up to 63 blocks (1,008 slots) for 999 tokens actually stored, a 0.9% overhead. The naive strategy, allocating exact-sized buffers, shows 0% fragmentation by construction — it has no block concept to round against. That's the honest trade a block allocator makes: bounded, small, predictable padding in exchange for eliminating the O(n) copy tax above. 0.9% is a good trade.

## Benchmark 2: real models, end to end

The synthetic sweep isolates the allocator, which is useful precisely because it removes every other variable — but it's also exactly the kind of clean, single-variable result the [previous benchmarking post on this blog](/docs/blog/kivi-metal-kernel-honest-benchmark) is about being suspicious of. A microbenchmark can be honest and still not predict what happens when a real model's actual attention math, actual memory allocator, and actual generation loop are all running at once. So: six models, real generation, through `mlx_lm.generate()`.

`benchmark_scripts/benchmark_pool_backed_kvcache.py` runs the identical prompt and `max_tokens` through a stock `KVCache` and then through `PoolBackedKVCache`, on the same model, and reports generation throughput and peak memory for both — plus a second, sequential request sharing the same pool, to check that blocks freed by the first request actually get reused rather than triggering fresh allocation.

Six models, 300 generated tokens each, block_size=16:

| model | params | stock tok/s | pool tok/s | Δ tok/s | stock peak | pool peak | Δ peak | reuse (2nd request) |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 135M | 265.4 | 257.5 | −3.0% | 0.315 GB | 0.318 GB | +1.1% | 100% |
| Qwen2.5-0.5B-4bit | 0.5B | 254.7 | 252.3 | −0.9% | 0.354 GB | 0.358 GB | +1.2% | 100% |
| Llama-3.2-1B-4bit | 1B | 122.3 | 123.1 | +0.6% | 0.838 GB | 0.849 GB | +1.3% | 100% |
| gemma-3-4b-4bit | 4B | 42.9 | 42.7 | −0.6% | 2.666 GB | 2.707 GB | +1.5% | 100% |
| Mistral-7B-v0.3-4bit | 7B | 22.8 | 24.5 | +7.4% | 4.231 GB | 4.269 GB | +0.9% | 100% |
| Qwen2.5-7B-4bit | 7B | 23.9 | 23.9 | +0.1% | 4.429 GB | 4.451 GB | +0.5% | 100% |

Three things worth saying about this table, in order of how much they matter.

**Throughput moves both directions and lands within ±7.4%.** That's noise, not signal — the [KIVI Metal kernel post](/docs/blog/kivi-metal-kernel-honest-benchmark) measured a ±25% run-to-run spread on an *unchanged* fp16 baseline from thermal state and scheduling alone, on the same hardware. Every delta here sits well inside that floor. The pool changes allocation bookkeeping — which block ids get handed out, when — not the attention or decode math, so parity, not a speedup, is the expected and correct result. (Mistral's +7.4% is the single largest positive delta here and it's still smaller than the established noise floor; it is not being read as a real finding.)

**Peak memory rises a small, consistent amount — +0.5% to +1.5% — across every model.** This is real, not noise, and it's not free: `PoolBackedKVCache` grows in `block_size`-token chunks tracked through the pool's bookkeeping structures (one `Block` object and dict entries per allocation), which costs a little Python/object overhead on top of the stock cache's plain `mx.array` growth. It's the price of getting per-block allocation visibility — fragmentation stats, reuse counts, exhaustion tracking — and at under 2% of peak memory on every model tested, it's a small price.

**Every model shows 100% block reuse on the second sequential request.** Request A generates, releases its blocks back to the pool; request B — a fresh, independent generation sharing the same pool — draws every single block it needs from that just-freed set rather than growing the pool further. This is the steady-state behavior a block pool exists to produce: a long-running server serving many sequential requests should see its pool stop growing entirely once it's warm. Six real models, six architectures (Dense down to 135M, up through 7B), same result.

## What this does and doesn't claim

It doesn't claim `PoolBackedKVCache` makes generation faster — it doesn't, by design, and the small memory overhead it does add is honestly reported, not hidden. What it claims is narrower and, I think, more useful: **the block-pool allocator eliminates an O(n)-with-context transient memory cost that a naive growing buffer pays on every single append**, at a fixed, small, and now-measured cost in steady-state memory — and this holds up identically whether you isolate the allocator on a synthetic sweep to 32K tokens, or run six real models end to end through `mlx_lm.generate()`.

Compression ratio and allocator overhead are answering different questions. A method's bit-width tells you what a token *should* cost. The allocator tells you what growing the buffer that holds that token actually spends, on every step, for as long as the sequence keeps growing. Both numbers belong in the report — this one is about the second.

---

*Benchmarks: `benchmark_scripts/benchmark_kv_fragmentation.py` (synthetic allocator sweep) and `benchmark_scripts/benchmark_pool_backed_kvcache.py` (real-model, end-to-end). Allocator lives in `veloxquant_mlx/memory/block_pool.py` and `veloxquant_mlx/memory/pool_backed_cache.py`. See the [memory/block-pool API reference](/docs/api/memory-api) for usage. All measurements on an Apple M4 with MLX.*

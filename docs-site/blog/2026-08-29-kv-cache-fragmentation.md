---
slug: kv-cache-fragmentation
title: "Compression Ratio Lies About Memory. Here's the Allocator That Doesn't."
date: 2026-08-29
authors: rajveer
tags: [memory, allocator, kv-cache, apple-silicon, mlx, benchmarking]
---

# Compression Ratio Lies About Memory. Here's the Allocator That Doesn't.

*A 5×–8× KV-cache compression ratio tells you almost nothing about resident memory: it describes the arithmetic, not the mechanism of growing the buffer that arithmetic runs on. This is what happens when you measure the allocator instead — on a synthetic sweep out to 32K tokens, and on six real models from 135M to 7B parameters — including a first draft of this benchmark that compared against a strawman baseline no shipped cache actually uses, and the correction once I checked.*

---

Every KV-cache method in this library reports a compression ratio: KIVI gets 4×–8×, TurboQuant gets more. Those numbers describe the *arithmetic* — how many bytes a token's key/value vector costs once quantized. They describe nothing about what actually sits resident in memory while the cache is growing.

Those are different questions. A cache that logically needs 256 KiB for 1024 tokens can still make your process's peak memory spike far past that, because of how the buffer holding those bytes got built. The compression ratio is deaf to this; a good allocator is the difference between "safe to run" and "swaps your Mac to death" at the same reported ratio.

This post is about the allocator side specifically — not compression math, not a quantization method, just: given a cache that's growing token by token, how much memory does the *mechanism of growing it* actually spend, and how does that scale with context length?

## The two allocation strategies

There are two ways to grow a sequence's KV storage that are actually worth comparing — and getting the first one wrong is exactly the mistake an earlier draft of this post made, so it's worth being specific.

**Chunked concatenate.** This is what `mlx_lm`'s stock `KVCache` actually does (`mlx_lm.models.cache.KVCache.update_and_fetch`): pre-allocate a `step`-token chunk (256 by default), write new tokens into it, and only when that chunk fills up, allocate a fresh `step`-sized chunk and `mx.concatenate()` it onto the existing buffer. It does **not** recopy the whole sequence on every append — an earlier version of this benchmark compared the pool against exactly that strawman (full recopy every append), which no shipped cache does, and it made the pool's advantage look far larger than it is. This post's numbers are from the corrected baseline: real chunked-concatenate growth, at the same chunk size as the pool's `block_size`, so the comparison isolates one variable — concatenate-a-new-chunk vs. checkout-a-block — instead of also smuggling in a chunk-size difference.

**Block pool.** Pre-allocate fixed-size blocks up front. Appending tokens writes into the current block; only allocate a new block when the current one fills up. Existing data is never copied or concatenated — a full block, once written, is never touched again until it's freed. This library's [`BlockPoolAllocator`](/docs/api/memory-api) (issue #249, `veloxquant_mlx/memory/block_pool.py`) implements this, and [`PoolBackedKVCache`](/docs/api/memory-api#poolbackedkvcache) (`veloxquant_mlx/memory/pool_backed_cache.py`) wires it directly into `mlx_lm.generate()`'s `update_and_fetch` protocol as a drop-in replacement for the stock cache.

Both strategies end up holding the same number of resident bytes once the sequence stops growing — that was never in question. The difference is in what each *append that triggers a new allocation* costs: `mx.concatenate([old_buffer, new_chunk])` must materialize a fresh array holding both, so for one moment the old buffer and the new concatenated buffer are both alive in memory. The old buffer's size scales with how much history exists, so that moment's cost scales with context length. The pool's per-allocation cost is one new block, a constant, regardless of how long the sequence already is.

That's a claim you can measure directly, so this benchmark measures it directly — and reports the actual run-to-run noise in that measurement, not one favorable sample.

## Benchmark 1: synthetic, allocator only, out to 32K tokens

`benchmark_scripts/benchmark_kv_fragmentation.py` grows a single sequence — mirroring real prompt processing, not a swarm of short requests — out to each of 1K, 4K, 8K, 16K, and 32K tokens, using both strategies at matched chunk/block size, and reports:

- **logical_bytes** — exact bytes the stored tokens need (`n_tokens × head_dim × 2` for fp16)
- **physical_bytes** — bytes actually resident, including any block padding
- **fragmentation** — `physical / logical - 1`
- **temp_peak_bytes** — the largest *transient* rise in MLX's own peak resident memory (`mx.get_peak_memory()`) during a single append, isolating the scratch/concatenate cost from the steady-state footprint
- **peak_physical_bytes** — high-water mark across the whole run

It's deliberately model-free — allocator behavior doesn't depend on which quantization method or LLM is in use — so it runs anywhere without a checkpoint download, following the same approach as the existing block-pool benchmark from issue #249.

Steady-state resident size is identical between the two strategies at every context length, as expected:

| context | stock physical | pool physical |
|---|---|---|
| 1,024 tok | 256.0 KiB | 256.0 KiB |
| 4,096 tok | 1,024.0 KiB | 1,024.0 KiB |
| 8,192 tok | 2,048.0 KiB | 2,048.0 KiB |
| 16,384 tok | 4,096.0 KiB | 4,096.0 KiB |
| 32,768 tok | 8,192.0 KiB | 8,192.0 KiB |

The transient cost — what a chunk-filling append pays, not what the sequence holds at rest — is where the two strategies diverge. Because this measurement is itself noisy (allocator/scheduler variance, same lesson as every prior post on this blog), the numbers below are medians across 3 full runs, with the observed range:

| context | stock temp_peak (median, range) | pool temp_peak | ratio (median) |
|---|---|---|---|
| 1,024 tok | 272.0 KiB (272.0–272.0) | 4.3 KiB | **64×** |
| 4,096 tok | 1,040.0 KiB (1,040.0–1,040.0) | 4.3 KiB | **244×** |
| 8,192 tok | 2,380.0 KiB (2,064.0–3,468.0) | 4.3 KiB | **~560×** |
| 16,384 tok | 4,112.0 KiB (4,112.0–6,828.0) | 4.3 KiB | **~970×** |
| 32,768 tok | 15,384.0 KiB (14,544.0–16,356.0) | 4.3 KiB | **~3,600×** |

Two things to take from this, in order of confidence.

**High confidence: the pool's transient cost is flat and the stock cache's is not.** Across all 3 runs and every context length, pool never moved off 4.3 KiB — one new block, independent of history length, exactly as the mechanism predicts. Stock's temp_peak grew with context length in every run, because a fresh `mx.concatenate([old, new_chunk])` briefly holds the old buffer (which scales with context) plus the new one. This qualitative relationship — flat vs. growing — held in 100% of samples and is the actual finding.

**Lower confidence: the exact ratio at a given context length.** The range column is real variance from a single unchanged code path (compare to the ±25% baseline spread reported in the [KIVI Metal kernel post](/docs/blog/kivi-metal-kernel-honest-benchmark) on this same hardware) — at 8,192 tokens the ratio could reasonably be reported as anywhere from ~470× to ~810× depending which of the 3 runs you'd picked. Treat "~3,600× at 32K" as an order-of-magnitude statement, not a precise multiplier — the *direction and existence* of the gap is solid, the specific digit after the tilde is not something to build an argument on.

**Fragmentation**, separately, is a cleaner measurement with no such noise: with the default settings (block_size=16 dividing every default context length evenly), the pool shows 0% padding waste — no free lunch was hidden by rounding. Feeding it a context length that doesn't divide evenly makes the real cost visible: at 999 tokens with block_size=16, the pool rounds up to 63 blocks (1,008 slots) for 999 tokens actually stored, a 0.9% overhead. The stock strategy, chunked at the same granularity, would show equivalent rounding once its own chunk boundary is crossed — fragmentation here is a property of chunk/block size, not of which allocation strategy is used. That's the honest trade a fixed-granularity allocator makes: bounded, small, predictable padding, in exchange for whatever growth-mechanism benefit the strategy provides. 0.9% is a good trade either way.

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

## The mistake, and why it mattered

The first version of this benchmark's synthetic baseline reallocated and copied the *entire* sequence on every single append — not the fixed-chunk `mx.concatenate` growth `mlx_lm`'s actual stock cache uses, which only reallocates once a chunk fills up. That baseline produced numbers like "82×–2,269× less transient memory," and every one of those numbers was comparing the pool against a strategy nobody ships. Once corrected to grow at the same chunk size mlx_lm actually uses, the gap is real but far smaller — roughly 64× at 1K tokens, growing to somewhere in the high-hundreds to low-thousands× by 32K, with real run-to-run noise on the exact figure at any single context length. The qualitative shape — pool flat, stock growing with context — survived the correction intact. The specific multiplier didn't, and shouldn't have been trusted at face value in the first place.

## What this does and doesn't claim

It doesn't claim `PoolBackedKVCache` makes generation faster — it doesn't, by design, and the small memory overhead it does add is honestly reported, not hidden. It doesn't claim a precise multiplier for the transient-memory gap — that number is genuinely noisy, and the range above is reported instead of a single favorable sample. What it claims is narrower: **the block-pool allocator's per-append transient cost stays flat as context grows, while chunked-concatenate growth's transient cost grows with it** — measured against `mlx_lm`'s real growth strategy, not a strawman, and confirmed on six real models end to end through `mlx_lm.generate()`, where peak-memory overhead is a small, consistent, honestly-reported +0.5%–+1.5%.

Compression ratio and allocator overhead are answering different questions. A method's bit-width tells you what a token *should* cost. The allocator tells you what growing the buffer that holds that token actually spends, on every step, for as long as the sequence keeps growing. Both numbers belong in the report — this one is about the second, and about how easy it is to get the second one wrong by picking the wrong thing to compare against.

---

*Benchmarks: `benchmark_scripts/benchmark_kv_fragmentation.py` (synthetic allocator sweep) and `benchmark_scripts/benchmark_pool_backed_kvcache.py` (real-model, end-to-end). Allocator lives in `veloxquant_mlx/memory/block_pool.py` and `veloxquant_mlx/memory/pool_backed_cache.py`. See the [memory/block-pool API reference](/docs/api/memory-api) for usage. All measurements on an Apple M4 with MLX.*

---
slug: qwen3-8b-kivi-honest-benchmark
title: "A Kernel That Wasn't There"
description: "An honest end-to-end benchmark of the KIVI Metal kernel against Qwen3-8B on Apple M4 -- bit-exact, fast at the op level, and invisible end-to-end, confirming the earlier 3B-model finding on a second, larger model."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, kivi, benchmarking, kv-cache]
---

# A Kernel That Wasn't There

*I benchmarked VeloxQuant-MLX's KIVI Metal kernel against Qwen3-8B on an Apple M4 -- and the honest result is that it changes nothing you'd notice, for reasons that were predictable before I ran anything.*

---

There's a specific kind of benchmark result that's more useful than a win: a clean, well-measured null. Not "it didn't work," not "the code is broken" -- but "I measured this carefully, twice, at two context lengths, with the noise floor exposed, and the answer is that nothing moved."

That's what happened when I ran this repo's KIVI KV-cache quantization kernel against Qwen3-8B on an Apple M4. This post is the full record -- including the part where the model I actually wanted to test turned out to be disqualified before I wrote a single line.

---

## The starting question

An earlier post on this project, [A 5.65× Metal Kernel That Made My LLM Exactly 0% Faster](/blog/kivi-metal-kernel-honest-benchmark), found that a bit-exact, 5.65x-faster-at-the-op-level KIVI kernel produced no measurable end-to-end difference on a 3B model, and explained precisely why: quantization is roughly 1-2% of total runtime, so even a large kernel-level win is invisible once Amdahl's law gets to it.

The obvious follow-up: does that hold on a bigger, newer model? I set out to benchmark the newest capable open-weight model I could find, run the same honest protocol, and report whatever came out -- good, bad, or nothing.

## Picking a model turned out to be the interesting part

"Newest" and "will actually work with this benchmark" are not the same list, and finding that out took longer than running the benchmark itself.

The newest Qwen releases at the time -- **Qwen3.5** and **Qwen3.6** -- looked like the obvious pick. Recent, well-supported, official MLX conversions available within hours of release. Then I actually read their configs.

:::danger[Disqualified: Qwen3.5-4B]
Multimodal -- loads via `mlx_vlm`, not `mlx_lm`. VeloxQuant's KIVI cache and this benchmark harness are built against `mlx_lm`'s generation loop; they aren't wired into the vision-language pipeline at all.
:::

:::danger[Disqualified: Qwen3.6-27B / 35B-A3B]
Same `mlx_vlm` problem, plus a deeper one: hybrid architecture, mostly **Gated DeltaNet** (linear attention) layers with full attention only every 4th layer. KIVI's per-channel/per-token quantization assumes standard full attention at *every* layer -- most of this model's layers aren't the thing KIVI was designed to compress. And at 27B-35B params, it doesn't fit a 24GB machine's ~19GB working-set budget at 4-bit anyway.
:::

Which left **Qwen3-8B** -- the previous generation, but the newest model that's actually compatible: plain causal decoder, full grouped-query attention at every layer, native `mlx_lm` support, and a 4.61GB 4-bit footprint that leaves comfortable room for KV cache on a 24GB Mac. Not the frontier release, but the newest one where "benchmark the kernel" is even a coherent sentence.

> The newest model isn't always the right model for the question you're asking.

## Setting up an honest run

The earlier KIVI post is explicit about four separate ways its own benchmarks lied -- a lazy-eval sync point that inflated a number 50x, GPU contention from a stray background process that flipped a result by 15x, cache reuse across repeated calls that produced four different "conclusions" from the same kernel, and a 127 MB "memory saving" that was allocator noise. This run was built to avoid repeating any of them:

- **fp16 vs KIVI-off vs KIVI-on**, end-to-end -- not an isolated op-level microbenchmark.
- **Interleaved repeats**, not blocked: one full round through all three arms, five times, so thermal drift and scheduling noise land on every arm equally instead of stacking against one.
- **Median / min / max reported**, not a single best-of-N number, so the noise floor is visible rather than hidden.
- **Two prompt lengths** -- 231 tokens and 2,238 tokens -- because a short context can sit at the flat part of a kernel's headroom curve and understate what's really there.
- **Byte-identical output check** between kernel-on and kernel-off, since the two code paths are documented bit-exact and this should hold in practice, not just on paper.
- **A contention check** before trusting anything -- scanning for other GPU-heavy processes running at the same time.

One more decision worth flagging: KIVI's own default bit-width is 2 bits. This repo's own docstring for the KIVI cache documents that at 2 bits, logit cosine similarity against the fp16 baseline goes *negative* -- effectively uncorrelated output -- by 64 decode steps on a small model. Benchmarking throughput on a config known to break output quality would produce a number attached to a model that isn't really answering questions anymore. This run used **4 bits** instead, where the same measurement shows 0.996 similarity -- safe, and still a real compression config, not a strawman.

## The numbers

Run against `mlx-community/Qwen3-8B-4bit` on Apple M4 (24 GB), 36 layers, 8 KV heads, head_dim 128:

```text
--- short (prompt_tokens=231) ---
  fp16       tok/s median=17.7  min=14.7  max=17.8  peak=4725MB  key_x=1.00 fullKV_x=1.00
  kivi_off   tok/s median=17.6  min=17.3  max=17.8  peak=4649MB  key_x=3.90 fullKV_x=2.29
  kivi_on    tok/s median=17.6  min=17.3  max=17.8  peak=4752MB  key_x=3.90 fullKV_x=2.29
  kivi_on vs kivi_off identical text: True

--- long (prompt_tokens=2238) ---
  fp16       tok/s median=5.9   min=5.9   max=6.8   peak=5226MB  key_x=1.00 fullKV_x=1.00
  kivi_off   tok/s median=5.9   min=5.7   max=6.3   peak=5355MB  key_x=3.28 fullKV_x=3.05
  kivi_on    tok/s median=5.9   min=5.8   max=6.2   peak=5274MB  key_x=3.28 fullKV_x=3.05
  kivi_on vs kivi_off identical text: True
```

| prompt | arm | tok/s median | min-max | peak MB | key x | full-KV x |
|---|---|---|---|---|---|---|
| short (231 tok) | fp16 | 17.71 | 14.66-17.85 | 4725 | 1.00x | 1.00x |
| short | KIVI off | 17.62 | 17.29-17.83 | 4649 | 3.90x | 2.29x |
| short | KIVI on | 17.63 | 17.31-17.81 | 4752 | 3.90x | 2.29x |
| long (2,238 tok) | fp16 | 5.93 | 5.86-6.81 | 5226 | 1.00x | 1.00x |
| long | KIVI off | 5.86 | 5.73-6.29 | 5355 | 3.28x | 3.05x |
| long | KIVI on | 5.92 | 5.76-6.16 | 5274 | 3.28x | 3.05x |

:::tip[Metal kernel, on vs. off]
**1.0005x** at the short prompt, **1.011x** at the long one. Both are comfortably inside the noise floor described below. Output text was **byte-identical** between kernel-on and kernel-off across all ten repeats, at both prompt lengths -- the documented bit-exactness held in practice, not just on paper. No contending GPU processes were running during the measured repeats.
:::

### Why the noise floor matters more than the ratio

Look at the min-max spread on the *unchanged* fp16 baseline -- same code, same weights, nothing different between runs: **14.66 to 17.85 tok/s** on the short prompt, **5.86 to 6.81 tok/s** on the long one. That's an **18% spread** and a **16% spread**, respectively, from thermal state and scheduling jitter alone. The kernel's entire measured effect -- 0.05% to 1.1% -- is roughly fifteen to three-hundred times smaller than that spread. Reporting "KIVI-on is 1.1% faster" without also reporting "and the baseline itself varies by 16% run to run" would be technically true and practically misleading. This is the same lesson the earlier post's Generation 3 table made explicit: looking for a 1% effect through 16-25% baseline variance is like weighing a signature on a bathroom scale.

---

## Why this was predictable

None of this should be surprising if you do the arithmetic first, which is the actual point of the earlier post. Quantization overhead in KIVI is a small fraction of total wall-clock time -- the model's matmuls and attention dominate. A kernel that speeds up quantization itself, however dramatically at the op level, is speeding up a sliver of the total.

> A 5x speedup on 2% of the work is a 2 x 0.05 = 10% change to nothing measurable once repeat variance is larger than the effect. Amdahl's law doesn't negotiate.

The memory numbers tell a related story. Peak memory didn't drop with KIVI enabled -- if anything it moved a little in both directions, which is allocator noise, not signal. This is expected, not a bug: this repo's KIVI cache quantizes *then immediately dequantizes back to fp16* before attention runs, because the downstream attention call needs a standard fp16 tensor. The `key x` / `full-KV x` columns above are a *hypothetical* packed-byte accounting -- "what this would cost if actually stored compressed" -- not a measurement of what's resident in memory. The tensor sitting in GPU memory is fp16 the entire time. Reducing that would need packed `uint8` storage plus attention that reads packed codes directly, which is a different, larger project than a fused quant/dequant kernel.

## So why ship a kernel that does nothing end-to-end?

Genuinely fair question, and the earlier post's answer still holds here:

- **It's free.** Bit-exact, byte-identical output confirmed across ten repeats on this model too. It never makes anything slower.
- **It removes a floor, not a ceiling.** Quantization is ~1-2% of runtime *today*. If everything around it gets faster -- better attention kernels, better matmuls -- that slice grows, and fixed per-op costs start to matter more.
- **The op-level win is real, even though it's invisible.** A kernel can be honestly fast at what it does and still be inconsequential to what the user experiences. Both facts belong in the same report.

## What I'd take from this

:::tip[Model selection is part of the methodology]
"Newest" and "correct fit for the test" aren't the same axis. Checking architecture compatibility (attention pattern, loader library, size envelope) *before* writing a benchmark script saved a wasted run against Qwen3.6 -- and the reasoning for ruling it out is as much a finding as the throughput numbers are.
:::

:::tip[Report the noise floor, always]
A ratio without its baseline variance is not a result. 1.1% next to a 16% baseline spread is a null result reported honestly; 1.1% reported alone looks like a finding.
:::

:::tip[Compression ratio is not memory savings]
If your quantized value gets dequantized back to full precision before the next op touches it, your "compression ratio" is an accounting fiction until something downstream actually consumes the packed form.
:::

The honest headline is the same shape as the one from the 3B run: a correct, bit-exact, measurably-fast-at-the-op-level Metal kernel, invisible end-to-end, for reasons you can derive with arithmetic before running a single generation. Running it again on a bigger, newer-generation model didn't change the conclusion -- it strengthened it, because now it's two models, two very different context lengths, and the same result both times.

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), KIVI `bit_width_inlier=4`, `group_size=32`, `residual_length=32`, 5 interleaved repeats per arm, 120 max tokens. Benchmark script: [`benchmark_scripts/benchmark_qwen3_8b_kivi_honest.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_kivi_honest.py). See the [KIVI algorithm reference](/algorithms/kivi) for the method itself, and the [companion 3B-model post](/blog/kivi-metal-kernel-honest-benchmark) for the original kernel writeup.*

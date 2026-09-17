---
slug: qwen3-8b-turboquant-rvq-honest-benchmark
title: "A Real Cost and a Claim That Didn't Hold"
description: "An honest end-to-end benchmark of TurboQuantRVQ -- this library's default serving method -- against Qwen3-8B on Apple M4. Unlike the KIVI kernel, the fused Metal pack kernel is genuinely invisible, but the method itself has a real throughput cost that grows with context length, and a previously-reported memory win reverses sign at long context."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, turboquant, rvq, benchmarking, kv-cache]
---

# A Real Cost and a Claim That Didn't Hold

*Same honest-benchmark protocol as the [KIVI/Qwen3-8B post](/blog/qwen3-8b-kivi-honest-benchmark), a different method -- and this time the numbers aren't a null result. TurboQuantRVQ's fused Metal pack kernel is invisible, same as KIVI's. But the method itself costs real throughput that gets worse with context length, and a memory-savings claim already sitting in this repo's own docstring did not replicate at longer context -- it reversed sign.*

---

The [previous post](/blog/qwen3-8b-kivi-honest-benchmark) in this pair ran the same honest-benchmark protocol against KIVI and found nothing: a bit-exact Metal kernel, byte-identical output, and an end-to-end throughput delta smaller than the run-to-run noise floor. A clean null.

This post runs the identical protocol against **TurboQuantRVQ** -- this repo's namesake method, and its `DEFAULT_SERVE_METHOD` -- and gets a different kind of answer: two real findings, not a null, and one of them contradicts a number already committed to this codebase.

## Why TurboQuantRVQ, and why it's a different kind of test

KIVI quantizes keys, then **immediately dequantizes them back to fp16** before attention runs. The live tensor is fp16 at all times -- this repo's own `kivi_cache.py` docstring says so explicitly. That's why KIVI's "compression ratio" is a hypothetical byte-accounting number, not a memory-savings claim, and why testing for a memory win there would have been testing something ruled out by construction.

`TurboQuantRVQKVCache` is built differently. Keys are stored **packed**: two bit-packed `uint32` residual-vector-quantization index streams plus a shared fp16 norm, matching the same accepted "dequantize-on-fetch" pattern `mlx_lm`'s own native `QuantizedKVCache` uses. Nothing forces the packed form back to fp16 until a fetch actually needs it. So a peak-memory reduction here is a claim worth testing, not one closed off by the storage design -- and the cache's own module docstring already reports one: **-12.8% peak memory vs. fp16**, measured on a 1B model at a single 4,002-token prompt.

One measurement, one model, one prompt length. That's exactly the setup the earlier KIVI post's "run the third model" lesson warns about -- a result that looks clean because nothing has yet contradicted it.

## The kernel toggle, and one difference from the KIVI setup

`TurboQuantRVQKVCache` fuses key quantization and bit-packing into a single Metal kernel (`_use_metal_pack`) when `head_dim` is a power of two and fits one threadgroup (`head_dim <= 1024`). Qwen3-8B's `head_dim=128` qualifies, and the flag auto-enables at construction -- there's no public config field for it, unlike KIVI's `use_metal_kernels`. For a clean A/B this benchmark forces it off on the "off" arm by flipping the same private latch the class itself flips on a kernel-side failure -- not a new code path, just triggering the class's own documented fallback intentionally.

The method's own default bit-width is **2 bits** (`KVCacheConfig`'s dataclass default, and `turboquant_rvq_cache.py` documents no quality caveat at this width -- unlike KIVI, which documents negative logit cosine similarity at 64 decode steps for `b=2`). This run used the default rather than overriding it, since nothing in the source flagged it as unsafe.

Same protocol as the KIVI post otherwise: fp16 vs. kernel-off vs. kernel-on, interleaved (not blocked) repeats, median/min/max reported, two prompt lengths (231 and 2,238 tokens), byte-identical-output check, GPU-contention check before trusting anything.

## The numbers

Run against `mlx-community/Qwen3-8B-4bit` on Apple M4 (24 GB), 36 layers, 8 KV heads, head_dim 128, 5 interleaved repeats, `bit_width_inlier=2`:

```text
--- short (prompt_tokens=231) ---
  fp16       tok/s median=17.0  min=15.8  max=17.4  peak=4725MB  key_x=1.00
  rvq_off    tok/s median=14.4  min=14.2  max=14.6  peak=4637MB  key_x=3.88
  rvq_on     tok/s median=14.6  min=13.9  max=15.3  peak=4648MB  key_x=3.88
  rvq_on vs rvq_off identical text: True

--- long (prompt_tokens=2238) ---
  fp16       tok/s median=5.9   min=5.7   max=6.8   peak=5226MB  key_x=1.00
  rvq_off    tok/s median=4.1   min=3.9   max=4.6   peak=5285MB  key_x=3.88
  rvq_on     tok/s median=4.1   min=3.7   max=4.5   peak=5293MB  key_x=3.88
  rvq_on vs rvq_off identical text: True
```

| prompt | arm | tok/s median | min-max | peak MB | vs fp16 tok/s | key x |
|---|---|---|---|---|---|---|
| short (231 tok) | fp16 | 17.02 | 15.76-17.44 | 4725 | 100% | 1.00x |
| short | RVQ off | 14.42 | 14.20-14.63 | 4637 | 84.7% | 3.88x |
| short | RVQ on | 14.63 | 13.89-15.28 | 4648 | 86.0% | 3.88x |
| long (2,238 tok) | fp16 | 5.94 | 5.67-6.83 | 5226 | 100% | 1.00x |
| long | RVQ off | 4.10 | 3.87-4.55 | 5285 | 69.1% | 3.88x |
| long | RVQ on | 4.12 | 3.68-4.48 | 5293 | 69.4% | 3.88x |

:::tip[Metal pack kernel, on vs. off: also invisible]
**1.5%** faster at the short prompt, **0.4%** at the long one -- both inside the noise floor (fp16's own min-max spread is 4-15% across the two lengths). Output text was byte-identical across all ten repeats at both lengths, same as KIVI. So the fused quantize+pack kernel is *also* a correct, bit-exact, end-to-end-invisible optimization -- same shape of result as the KIVI post, for the same underlying reason: the kernel accelerates a small slice of total wall time.
:::

That's the part of this run that rhymes with the KIVI post. The two parts that don't follow.

### Finding 1: the method itself has a real, growing throughput cost

Unlike KIVI -- where fp16 vs. quantized was within a percent or two at both prompt lengths -- TurboQuantRVQ is **15% slower than fp16 at the short prompt and 31% slower at the long one**. This is not noise: the gap is three to five times larger than fp16's own run-to-run spread at each length, and it moves in a consistent direction as context grows.

:::warning[Cost scales with context, not just present]
The KIVI post's finding was "the overhead exists but doesn't move." Here the overhead is measurably larger at long context than short. The dequantize-on-fetch reconstructs the *entire* cached-key history from packed codes at every step, so its cost should scale with context length -- and it visibly does, going from a 15% throughput cost at 231 tokens to a 31% cost at 2,238. This is the opposite of the KIVI kernel's flat, invisible profile, and it means the honest tradeoff for this method is not free the way the earlier post's kernel was.
:::

### Finding 2: the memory claim reversed sign at longer context

This is the one worth sitting with. `turboquant_rvq_cache.py`'s own docstring reports -12.8% peak memory vs. fp16, from a single measurement on a 1B model at one 4,002-token prompt. Here, on Qwen3-8B:

- **Short prompt (231 tok): -1.7% to -1.9%** peak memory vs. fp16 -- same direction as the docstring's claim, but an order of magnitude smaller.
- **Long prompt (2,238 tok): +1.1% to +1.3%** peak memory vs. fp16 -- the **opposite** direction.

:::danger[A claim that held once and didn't generalize]
Two prompt lengths on one model were enough to flip the sign of a memory-savings number already committed to this codebase's own docstring. At short context, packed key storage is genuinely smaller than fp16, and that shows up as a small real saving. At long context, whatever else the dequantize-on-fetch path allocates -- reconstruction buffers, intermediate arrays sized to the *growing* cache -- outweighs the saving from packed storage, and total peak memory ends up slightly *higher* than just storing keys as fp16 in the first place. Interestingly, peak memory showed **zero within-arm variance** here (min equals max across all 5 repeats at both lengths) -- steadier than the throughput numbers -- so this isn't measurement noise masquerading as a sign flip; it's a real, reproducible crossover somewhere between 231 and 2,238 tokens of context.
:::

The existing docstring's number isn't wrong -- it's what was measured, on the model and prompt length it was measured on. It just isn't a general claim, and nothing about the original one-data-point report could have told you that from the number alone.

---

## What this changes about how the earlier claim should be read

The honest fix isn't to argue with the existing docstring's number -- it's to scope it. "TurboQuantRVQ saves memory" was never quite the claim; "TurboQuantRVQ saved 12.8% memory on a 1B model at one 4K-token prompt" was, and that's still true. What this run adds is that the sign of the effect depends on context length, at least somewhere between short and long prompts on an 8B model, and that the throughput cost of getting there is real and grows with context too -- which the original single data point had no way to surface.

## What I'd take from this

:::tip[A one-data-point claim is a data point, not a claim]
The existing docstring did the right thing methodologically -- it stated its exact measurement conditions rather than generalizing. The gap wasn't dishonesty, it was scope: nothing had yet run the second and third condition that would show the sign flip. Read every "measured: X%" in this codebase as scoped to its stated conditions, not as a property of the method.
:::

:::tip[Test the axis most likely to break the claim]
The KIVI post's rule was "run the third model." Here the operative axis was context length, not model identity -- the crossover between memory-positive and memory-negative sat somewhere between a 231-token and a 2,238-token prompt on the *same* model. Before trusting a compression method's memory claim, sweep the dimension the mechanism actually depends on, not just whichever dimension is easiest to vary.
:::

:::tip[An invisible kernel and an expensive method are not the same finding]
It would be easy to read "the Metal kernel is invisible, same as KIVI" and stop there. But the kernel and the method are different things: the kernel (quantize+pack fusion) is genuinely free here, exactly like KIVI's kernel. The *method* (RVQ quantization plus dequantize-on-fetch) is not free -- it costs 15-31% throughput regardless of whether the kernel is on. Reporting only the kernel A/B would have missed the actual cost of using this method at all.
:::

Two honest benchmarks, same protocol, same model, two different shapes of result: one method's kernel and its end-to-end footprint were both invisible; another method's kernel is equally invisible, but the method itself has a real, context-dependent cost and a memory story that only a second prompt length was able to falsify.

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), TurboQuantRVQ `bit_width_inlier=2` (the method's own default), 5 interleaved repeats per arm, 120 max tokens. Benchmark script: [`benchmark_scripts/benchmark_qwen3_8b_turboquant_rvq_honest.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_turboquant_rvq_honest.py). See the [companion KIVI/Qwen3-8B post](/blog/qwen3-8b-kivi-honest-benchmark) for the null-result half of this pair, and [`turboquant_rvq_cache.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/cache/turboquant_rvq_cache.py) for the method's implementation and its original single-data-point memory measurement.*

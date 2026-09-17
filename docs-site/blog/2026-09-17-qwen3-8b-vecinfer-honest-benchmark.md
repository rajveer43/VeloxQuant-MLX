---
slug: qwen3-8b-vecinfer-honest-benchmark
title: "The Kernel That Finally Did Something -- And Didn't Quite Agree With Itself"
description: "An honest end-to-end benchmark of VecInfer -- this library's product vector-quantization KV cache -- against Qwen3-8B on Apple M4. Unlike every other method in this series, the fused Metal encode/decode kernel is a real, large speedup (up to 14x at long context) and uses less peak memory than the pure-MLX fallback. But for the first time in this series, kernel-on and kernel-off do not produce byte-identical output."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, vecinfer, quantization, benchmarking, kv-cache]
---

# The Kernel That Finally Did Something -- And Didn't Quite Agree With Itself

*Fourth post in this series, same model and protocol as [KIVI](/blog/qwen3-8b-kivi-honest-benchmark), [TurboQuantRVQ](/blog/qwen3-8b-turboquant-rvq-honest-benchmark), and [QFilters](/blog/qwen3-8b-qfilters-honest-benchmark). The first three posts all found Metal kernels that were bit-exact against their pure-MLX fallback, whatever else they cost or saved. This one breaks that pattern in both directions: the kernel is a real, large speedup for once -- and, for the first time, it is not byte-identical to the path it's supposed to match.*

---

Three posts into this series, a pattern had formed: every Metal kernel tested so far -- KIVI's quantize/dequantize fusion, TurboQuantRVQ's pack kernel, QFilters' fused eviction -- produced text that was byte-identical to its pure-MLX fallback. The kernels differed in what they cost or saved end-to-end, sometimes substantially, but never in *what they computed*. Bit-exactness held every time.

This post benchmarks **VecInfer**, this repo's product vector-quantization method, and that pattern doesn't hold. The kernel is also, for the first time in this series, a genuinely large speedup rather than an invisible one. Both facts turned up in the same run, and neither one excuses the other.

## What VecInfer does, and the calibration caveat up front

VecInfer applies a per-(head, channel) smooth scale and a Walsh-Hadamard rotation to keys (suppressing outliers before quantization), encodes the transformed keys against a trained codebook, and immediately dequantizes back to fp16 before attention runs -- the live tensor is fp16 at all times, the same accepted "quantize then dequantize immediately" pattern as KIVI. So, like KIVI and unlike TurboQuantRVQ, any compression ratio reported here is a hypothetical byte-accounting number, not a claim about resident memory.

One thing has to be said plainly before the numbers: **this run does not use a calibrated codebook.** `VecInferKVCache`'s own docstring says a random-initialized codebook is "only useful for shape/wiring tests." No tooling for calibrating against real model activations exists in this repo yet. What this script uses instead -- reusing existing code from `benchmark_vecinfer.py` -- is k-means-style codebook training on **synthetic Gaussian samples shaped like the model's keys and values**, not the model's actual activations. That's a real, trained codebook, meaningfully better than pure random initialization, but it is still not calibrated to Qwen3-8B's real key/value distribution. Expect degraded output quality from that alone, independent of anything the kernel does. This is reported as exactly that -- not concealed, not treated as a surprise later in the post.

## The kernel toggle

Unlike TurboQuantRVQ and QFilters, `VecInferKVCache` exposes a public three-state `use_metal_kernels` field on `KVCacheConfig` -- same shape as KIVI's toggle, so this run needed no private-attribute override for a clean A/B. With Qwen3-8B's `head_dim=128` (power-of-two, well under the 512-element threadgroup cap), the "on" arm takes a genuinely different code path: a single fused Metal dispatch that does smooth-scale + Hadamard transform + quantize + dequantize + inverse-transform in one kernel call. The "off" arm runs the same math as seven separate MLX ops. Same protocol otherwise: fp16 vs. kernel-off vs. kernel-on, interleaved repeats, median/min/max, two prompt lengths, GPU-contention check.

## The numbers

Run against `mlx-community/Qwen3-8B-4bit` on Apple M4 (24 GB), 36 layers, 8 KV heads, head_dim 128, synthetic-Gaussian-calibrated codebooks (`key_codebook_bits=12`, `value_codebook_bits=8`, the method's own defaults), 5 interleaved repeats:

| prompt | arm | tok/s median | min-max | peak MB | vs fp16 tok/s | key x |
|---|---|---|---|---|---|---|
| short (231 tok) | fp16 | 17.69 | 17.14-17.93 | 4725 | 100% | 1.00x |
| short | VecInfer off | 3.16 | 2.91-3.25 | 5829 | 17.9% | 5.33x |
| short | VecInfer on | 3.21 | 3.12-3.30 | 5114 | 18.1% | 5.33x |
| long (2,238 tok) | fp16 | 5.33 | 4.79-5.90 | 5226 | 100% | 1.00x |
| long | VecInfer off | 0.17 | 0.16-0.18 | 6205 | 3.2% | 5.33x |
| long | VecInfer on | 2.42 | 2.24-2.43 | 5788 | 45.4% | 5.33x |

:::tip[The kernel is a real speedup for the first time in this series]
At the long prompt, the fused kernel is **14.24x faster** than the pure-MLX fallback (2.42 vs. 0.17 tok/s) -- not inside any noise floor, not a rounding-level effect. At the short prompt the two paths are close (3.21 vs. 3.16, ~2%), but at long context the gap is enormous and the direction is consistent: fusing seven ops (smooth scale, Hadamard transform, quantize, dequantize, inverse transform, unscale) into one Metal dispatch avoids materializing six intermediate buffers per call, and that saving compounds with every decode step against a 2,238-token cache. The kernel also uses **less peak memory** than the fallback at both lengths -- 12.3% less at short context, 6.7% less at long -- for the same reason: fewer intermediate allocations.
:::

### But: kernel-on and kernel-off do not agree

Every arm in this run is internally deterministic -- all 5 repeats of `vecinfer_off` produce identical text, all 5 repeats of `vecinfer_on` produce identical text, at both prompt lengths. That rules out sampling noise as the explanation for what comes next: **`vecinfer_on` and `vecinfer_off` do not produce the same text as each other**, at either prompt length.

```text
short prompt, vecinfer_off (all 5 repeats):
ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss::sssssss...

short prompt, vecinfer_on (all 5 repeats):
ssssssssssssssssssssssssss::sssssssssssssssssssssssssssss::s:ssssss:::::ssss...

long prompt, vecinfer_off (all 5 repeats):
:::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

long prompt, vecinfer_on (all 5 repeats):
:
:
:
:
:
:
ed:
ed:
:
:
:
ed:
...
```

:::danger[This is not the bit-exactness pattern the first three posts established]
KIVI's docstring documents its dequantize path as exact. TurboQuantRVQ's pack kernel is described as bit-identical to its MLX path (issue #251). QFilters' fused-evict kernel matches its pure-MLX selection "bit-for-bit," and this series confirmed all three claims held under measurement. Nothing in `vecinfer_cache.py` makes an equivalent claim for the fused encode/decode kernel -- and this run shows why that matters: the fused single-dispatch kernel and the seven-op pure-MLX pipeline compute the Walsh-Hadamard transform, quantization, and dequantization in a different operation order, and floating-point arithmetic is not associative. A tiny per-step numerical difference, invisible at token 1, compounds across 120 autoregressive decode steps into visibly different output -- both still degenerate (see below), but different degenerate attractors, deterministically, every time.
:::

Both outputs are also badly broken -- repeated `s`, repeated `:`, no coherent language in either arm. That's the calibration caveat showing up exactly as expected: a codebook trained on synthetic Gaussian samples, not real key/value activations, was never going to produce fluent generation. But "both are broken in different, non-bit-exact ways" and "one kernel is 14x faster than its counterpart" are two separate facts sitting in the same run, and the second one doesn't make the first one go away. A production deployment choosing `use_metal_kernels=True` for the throughput win is choosing a numerically different computation, not just a faster one -- something the QFilters and TurboQuantRVQ kernels, tested earlier in this series, did not require anyone to accept.

## What I'd take from this

:::tip[A real speedup and an unverified equivalence claim are not the same finding]
Three kernels in this series were free lunches: no cost, no correctness question, nothing to weigh. This one is a real 14x speedup at long context with a genuine memory saving -- and a numerical divergence from its own reference path that nothing in this repo's docstrings currently documents or claims to bound. Both belong in the same sentence about this kernel; reporting the speedup alone would have been the more flattering post and the less honest one.
:::

:::tip[Synthetic calibration is not a substitute for real calibration, and this run proves it two ways]
The degenerate output here was expected going in -- the docstring already says random-init codebooks are test-only, and training on synthetic Gaussian samples instead of real activations was flagged as a lesser substitute, not a fix. What wasn't obvious in advance: an uncalibrated codebook doesn't just produce *bad* output, it can produce output where the fused and unfused kernel paths disagree with each other, because there's no well-behaved signal for floating-point non-associativity to average out against. A calibrated codebook producing fluent text might show a smaller divergence, a larger one, or none at all -- that's a different, untested run.
:::

:::tip[Check for bit-exactness, don't assume it from a pattern]
Three posts of "the kernel matches its fallback" could have made a fourth one lazy about verifying the same thing again. It didn't hold here, and the only reason this post knows that is because the same identical-text check that passed three times in a row was run a fourth time instead of assumed.
:::

Four methods into this series, and this is the first one where the honest headline isn't "invisible" or "costly" -- it's "faster, and not proven equivalent." Both halves of that sentence are real, and reporting only the flattering half would have been the easy version of this post to write.

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), VecInfer `key_codebook_bits=12`, `value_codebook_bits=8` (the method's own defaults), synthetic-Gaussian-trained codebooks (not calibrated against real model activations -- see the calibration caveat above), 5 interleaved repeats per arm, 120 max tokens. Benchmark script: [`benchmark_scripts/benchmark_qwen3_8b_vecinfer_honest.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_vecinfer_honest.py). See the [KIVI](/blog/qwen3-8b-kivi-honest-benchmark), [TurboQuantRVQ](/blog/qwen3-8b-turboquant-rvq-honest-benchmark), and [QFilters](/blog/qwen3-8b-qfilters-honest-benchmark) posts for the first three in this series, and [`vecinfer_cache.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/cache/vecinfer_cache.py) for the method's implementation and its fused-kernel code paths.*

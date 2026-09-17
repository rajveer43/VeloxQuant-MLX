---
slug: qwen3-8b-qfilters-honest-benchmark
title: "The Output That Stopped Being Output"
description: "An honest end-to-end benchmark of QFilters -- this library's query-agnostic KV-cache eviction method -- against Qwen3-8B on Apple M4. The fused Metal eviction kernel is bit-exact, same as every other method in this series. But eviction itself has a real throughput cost even when nothing is evicted, and once the cache exceeds budget, generation degenerates into repeated tokens -- fully reproducibly, across all five repeats."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, qfilters, eviction, benchmarking, kv-cache, coherence]
---

# The Output That Stopped Being Output

*Third post in this series, same model and protocol as the [KIVI](/blog/qwen3-8b-kivi-honest-benchmark) and [TurboQuantRVQ](/blog/qwen3-8b-turboquant-rvq-honest-benchmark) posts before it. This time the method doesn't quantize anything -- it evicts. And once the cache runs out of budget, the honest result isn't a percentage, it's forty repeated equals signs.*

---

The [first post](/blog/qwen3-8b-kivi-honest-benchmark) in this series found a Metal kernel with zero end-to-end effect. The [second](/blog/qwen3-8b-turboquant-rvq-honest-benchmark) found a real throughput cost and a memory claim that flipped sign at longer context. Both methods, though, were **quantization** -- every key and value survives, just approximated. Nothing in either post could produce genuinely broken output, because nothing was ever thrown away.

This post benchmarks **QFilters**, this repo's query-agnostic eviction method, which works differently on purpose: past a fixed token budget, it *drops* the lowest-scoring cached tokens outright. Their information doesn't get approximated -- it's gone. That makes a different question testable, one the first two posts structurally couldn't ask: does the cache stay coherent once it starts forgetting things?

## Why eviction is a different kind of test

`QFiltersKVCache` scores every cached key by its projection onto a frozen per-head direction -- the "Q-Filter" -- and evicts the lowest-scoring tokens once the cache exceeds `qfilters_budget` (default 512 tokens, including protected leading "sink" positions). This repo's implementation is explicit that it's **adapted, not a faithful port** of the arXiv:2503.02812 preprint, and offers two ways to get the scoring direction:

- **Calibrated filters** (query-SVD, the mechanism the paper actually specifies) -- frozen before the first token, correct by construction, path-independent.
- **Fallback** (`filters=None`) -- SVD-estimated from the first `qfilters_calib_tokens` observed keys, which recovers the dominant axis but not its orientation. The module docstring calls this exactly what it is: a real, documented mode, not a stand-in.

No calibrated filters exist for Qwen3-8B in this repo, so this run used the fallback path -- the honest choice given what's actually available, not the paper's strongest configuration. That's flagged here the same way the TurboQuantRVQ post flagged using the method's own uncalibrated default bit-width: report what you actually ran, not the best-case version of it.

VecInfer -- this repo's other codebook-based method with a public Metal toggle -- was considered and set aside for this run. Its docstring is explicit that a random-initialized codebook is "for tests only," and no calibration tooling exists in this repo yet to produce a real one. Benchmarking coherence against a method whose codebook is documented as non-functional would have answered a different, less honest question than the one this post is actually asking.

## Setup

Same protocol as the previous two posts: fp16 vs. kernel-off vs. kernel-on, interleaved (not blocked) repeats, median/min/max reported, two prompt lengths, a GPU-contention check before trusting anything. One addition specific to eviction: **two separate comparisons that must not be conflated.**

:::info[Two different questions, two different expectations]
1. **`qfilters_on` vs. `qfilters_off`** (fused Metal eviction kernel vs. the pure-MLX path) -- the module docstring claims these agree bit-for-bit, same tie-breaking convention. This *should* produce identical text, exactly like the KIVI and TurboQuantRVQ kernel checks.
2. **QFilters vs. fp16** -- this should **not** be expected to match once the cache exceeds budget. Tokens are being discarded, not approximated. Divergence here isn't a bug to rule out; it's the actual thing this post measures.
:::

The short prompt (231 tokens) stays under the 512-token budget -- no eviction happens at all. The long prompt (2,238 tokens) blows well past it. That contrast is the point: one prompt length isolates pure per-step scoring overhead, the other tests what happens once the method actually does its job.

## The numbers

Run against `mlx-community/Qwen3-8B-4bit` on Apple M4 (24 GB), 36 layers, 8 KV heads, head_dim 128, `qfilters_budget=512`, fallback (uncalibrated) filters, 5 interleaved repeats:

| prompt | arm | tok/s median | min-max | peak MB | vs fp16 tok/s | compression |
|---|---|---|---|---|---|---|
| short (231 tok, no eviction) | fp16 | 16.91 | 16.77-18.11 | 4725 | 100% | 1.00x |
| short | QFilters off | 11.85 | 11.59-12.35 | 4933 | 70.1% | 1.00x |
| short | QFilters on | 11.78 | 11.29-12.36 | 4933 | 69.7% | 1.00x |
| long (2,238 tok, heavy eviction) | fp16 | 5.58 | 5.48-5.85 | 5226 | 100% | 1.00x |
| long | QFilters off | 0.89 | 0.88-0.91 | 5267 | 16.0% | 4.60x |
| long | QFilters on | 0.89 | 0.83-0.90 | 5266 | 15.9% | 4.60x |

:::tip[Metal eviction kernel, on vs. off: bit-exact again]
Byte-identical output across all ten repeats at both prompt lengths -- the third method in this series where the fused Metal kernel and the pure-MLX fallback produce indistinguishable text. The pattern established by KIVI and reconfirmed by TurboQuantRVQ holds a third time: this repo's Metal kernels are correct by the bit-exactness standard they claim, consistently, across three structurally different methods.
:::

### Finding 1: overhead exists even with zero evictions

At the short prompt, nothing gets evicted -- the cache never reaches its 512-token budget. And yet QFilters is **30% slower than fp16**, with peak memory **4.4% higher**, not lower. This is pure per-step scoring cost: projecting every cached key onto the filter direction, tracking per-head state, maintaining the SVD estimation buffers for the fallback path -- all of it runs whether or not anything ends up evicted. Unlike KIVI (roughly free) or TurboQuantRVQ (a real but smaller 15% cost at this same prompt length), QFilters' overhead is front-loaded into bookkeeping that has nothing to do with the compression it eventually delivers.

### Finding 2: at long context, throughput drops 84% and compression still doesn't shrink peak memory

Once eviction is actually happening, throughput falls to **16% of fp16's** -- a bigger relative cost than either quantization method in this series produced at any context length. Compression is real here (4.60x on the retained K/V, since evicted tokens' bytes are actually gone, not just packed tighter), but peak memory is still **0.78% higher than fp16**, not lower. The same shape of result as the TurboQuantRVQ post's memory-claim reversal, but more extreme: even a genuine 4.6x compression ratio on retained tokens doesn't translate into a net memory win, because whatever eviction bookkeeping costs at 2,238 tokens of context outweighs it.

### Finding 3: the coherence question has a real, fully reproducible answer

This is the part that doesn't fit in a percentage. Here is what fp16 generated on the long prompt (first 200 characters, identical across all 5 repeats):

```text
fp16:

Also, explain in simple terms what the KV cache is, and what it does.

Finally, explain in simple terms what "long-context inference" means.

Please make sure your explanations are clear and conci...
```

Here is what QFilters generated on the same prompt -- identical across all 5 repeats, and identical between the Metal kernel on and off:

```text
qfilters (on and off, all 5 repeats):
===============.========================================================================= is is is is is is is is is is is is is is is is is is is is===========
```

:::danger[Not noise -- a deterministic failure mode]
This isn't a one-off bad sample. All five repeats of `qfilters_off` and all five of `qfilters_on` produced the **exact same degenerate string**, character for character. That rules out the ordinary explanations -- sampling variance, a rare unlucky seed, thermal-state luck. Once the fallback filter's SVD-estimated direction evicts enough of the wrong tokens, the model's context is damaged badly enough that generation collapses into token repetition (`is is is is...`) and separator spam (`====`), and it does so exactly the same way every time. That's what a genuinely broken KV cache looks like at the output layer, not a subtle quality regression you'd need an eval harness to catch.
:::

Worth being precise about what this does and doesn't indict. This is the **uncalibrated fallback path** -- the docstring is explicit that a key-SVD-estimated direction "recovers the dominant axis but not its orientation," i.e., it can evict in the wrong direction from the start. It is not a claim that QFilters with real calibrated query-SVD filters would fail the same way; that's a different, untested configuration. It's also not a claim about the eviction *mechanism* being unsound -- a budget of 512 tokens against a prompt that needs most of its 2,238 tokens of context to answer a question **about that passage** is close to an adversarial setup by construction. What it is: an honest measurement of what actually ships if this cache is deployed at its own default budget, with the calibration path that's actually available in this repo today, against a plausible long-context prompt.

The fp16 baseline here isn't polished either -- it echoes the prompt's own instructions rather than producing a tight two-paragraph explanation, which is a model/prompt-fit issue independent of any cache method. But "echoes the prompt" and "===== is is is is is =====" are not the same category of failure, and the gap between them is the actual finding.

---

## What I'd take from this

:::tip[Quantization and eviction are different bets]
KIVI and TurboQuantRVQ can be wrong about a percentage -- fp16 similarity, throughput, memory. QFilters, past its budget, can be wrong about whether the output is language at all. Reporting "compression ratio: 4.6x" without also reporting what the output looked like at that ratio would have been true and almost useless.
:::

:::tip[Overhead and compression are separable, and the short prompt proves it]
A method can cost real throughput and memory even in the exact case where it does nothing -- zero evictions, zero compression, and still 30% slower with higher peak memory than fp16. If a benchmark only runs prompts long enough to trigger the mechanism, this cost is invisible. The short prompt in this run existed specifically to catch it.
:::

:::tip[Determinism across repeats is itself informative]
Five bit-identical repeats of a degenerate output is stronger evidence than one bad sample and stronger evidence than five different bad samples. It says the failure is structural -- reachable from this exact configuration every time -- not a rare unlucky draw this benchmark happened to catch once.
:::

Three methods into this series, three different shapes of honest result: a kernel that changes nothing, a method with a real and growing cost and a memory claim that reverses sign, and a method where the Metal kernel is once again bit-exact but the method itself, run at its own default budget with the calibration path actually available, can turn a long-context prompt into forty characters of `=`.

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), QFilters `qfilters_budget=512` (the method's own default), fallback (uncalibrated) filter path, 5 interleaved repeats per arm, 120 max tokens. Benchmark script: [`benchmark_scripts/benchmark_qwen3_8b_qfilters_honest.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_qfilters_honest.py). See the [KIVI/Qwen3-8B post](/blog/qwen3-8b-kivi-honest-benchmark) and [TurboQuantRVQ/Qwen3-8B post](/blog/qwen3-8b-turboquant-rvq-honest-benchmark) for the first two posts in this series, and [`qfilters_cache.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/cache/qfilters_cache.py) for the method's implementation, its calibrated-vs-fallback filter distinction, and its own documented limitations.*

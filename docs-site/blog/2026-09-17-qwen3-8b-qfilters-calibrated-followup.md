---
slug: qwen3-8b-qfilters-calibrated-followup
title: "Calibration Changed the Failure, Not the Outcome"
description: "A follow-up to the QFilters honest benchmark: does this repo's real query-SVD calibration module fix the generation collapse the earlier post found? Real calibration is 3.6x faster than the uncalibrated fallback and measurably different -- but at this eviction budget, on this prompt, the output is still not coherent, just degenerate in a different shape."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, qfilters, eviction, benchmarking, kv-cache, coherence, calibration]
---

# Calibration Changed the Failure, Not the Outcome

*Direct follow-up to [The Output That Stopped Being Output](/blog/qwen3-8b-qfilters-honest-benchmark). That post found QFilters producing a fully reproducible generation collapse on a long prompt, using the fallback (uncalibrated) filter path. This one asks the obvious next question: does this repo's real calibration module fix it? Short answer -- it changes what the collapse looks like and makes it substantially faster, but on this exact configuration, it does not produce coherent output.*

---

The [QFilters post](/blog/qwen3-8b-qfilters-honest-benchmark) was explicit about running the uncalibrated fallback path -- key-SVD-estimated filter direction, sign ambiguous, because a KV cache never sees queries. That was flagged as a real limitation, not a hidden one. What made it worth a follow-up is that this repo isn't missing a fix for that limitation: `veloxquant_mlx/quantizers/qfilters_calibration.py` already implements the paper's actual mechanism (arXiv:2503.02812 §3.2) -- hook each layer's query projection, gather real query activations, take the SVD's top right-singular vector per head, sign-fix it against the paper's Theorem 3.3 (`kappa^h > 0`), average query-head filters down to KV heads for GQA. It just wasn't used in the first post.

This post uses it.

## What calibration actually involved

`calibrate_qwen3_8b_qfilters.py` ran Qwen3-8B forward over five ~2,048-token calibration passages (general prose about transformers, memory bandwidth, unified memory, vector quantization, and eviction itself -- topically adjacent to the benchmark prompt but not the same text), captured real query activations per layer via a hook on `q_proj`, and computed one `[8, 128]` filter per layer (32 query heads averaged down to 8 KV heads). This is a one-time, pre-deployment pass -- the paper's own cost estimate is 20 samples of length 2048; this run used 5, on the smaller side but the same order of magnitude.

:::info[This is not a claim about calibration-corpus quality]
The calibration passages here are repo-local synthetic prose, not the Pile (what the paper's own experiments used). A better-matched, larger, more diverse calibration corpus is a real variable this post did not sweep. What this post answers is narrower: given the calibration mechanism this repo already implements, run correctly, does it fix the specific collapse the earlier post measured?
:::

## The comparison

Same long prompt (2,238 tokens) and same `qfilters_budget=512` as the original post, five arms instead of three: `fp16`, `fallback_off`/`fallback_on` (the original post's uncalibrated path, Metal kernel off/on), and `calibrated_off`/`calibrated_on` (this post's real query-SVD filters, Metal kernel off/on). 5 interleaved repeats per arm.

| arm | tok/s median | min-max | peak MB | vs fp16 | compression |
|---|---|---|---|---|---|
| fp16 | 6.47 | 5.40-7.05 | 5226 | 100% | 1.00x |
| fallback (off) | 0.95 | 0.86-1.12 | 5267 | 14.7% | 4.60x |
| fallback (on) | 0.91 | 0.83-1.12 | 5266 | 14.1% | 4.60x |
| calibrated (off) | 3.41 | 3.14-4.23 | 5565 | 52.7% | 4.60x |
| calibrated (on) | 3.32 | 0.26-4.09 | 5547 | 51.3% | 4.60x |

:::tip[Real calibration is 3.6x faster than the fallback, at the same compression]
`calibrated_off` runs at 3.41 tok/s median vs. `fallback_off`'s 0.95 -- a **3.6x speedup**, at an identical 4.60x compression ratio (the budget and eviction count are the same; only the scoring direction changed). This wasn't the thing being tested for, but it's a real, reproducible difference: the calibrated filter evicts along a more decisive direction, which appears to change the per-step cost of the eviction path itself, not just its selection quality. Peak memory is about 6% higher with calibration (5565MB vs. 5267MB) -- a real, if modest, cost for that speedup.
:::

:::warning[One outlier worth naming instead of hiding]
`calibrated_on`'s five repeats ranged 0.26 to 4.09 tok/s -- one rep came in far below the other four (which clustered 3.3-4.1). Every other arm in this run, and every arm across all four posts in this series, has shown tight repeat-to-repeat clustering. This looks like a single stray contention or thermal event rather than a structural property of calibrated eviction, and the median (which this table reports) is barely moved by it -- but a benchmark that only ran once could have reported either 0.26 or 4.09 as "the" number, and both would have been wrong. This is the same noise-floor discipline the first post in this series built the whole protocol around, applied here.
:::

### The actual output

Every arm was internally deterministic -- 5/5 identical repeats, same as every method tested in this series. Here's what changed and what didn't:

```text
fallback (both kernel arms, all 5 repeats):
===============.========================================================================= is is is is is is is is is is is is is is is is is is is is===========

calibrated (both kernel arms, all 5 repeats):
...........,. the..........,. the. the. and.,..... the.,. the. the. and. the. the. the. and. the. the. the. the. the. the. the. and. the. the. the. the...
```

Calibration produced a **different** deterministic collapse, not a working one. `fp16`-vs-`calibrated` word overlap is actually *lower* (0.000) than `fp16`-vs-`fallback` (0.014) -- by that specific metric, calibration didn't move the output closer to the uncompressed baseline at all.

:::danger[Not a fix, but not nothing either]
There's a real, if easy to over-read, qualitative shift: the fallback path collapses into `=` and a single repeated verb; the calibrated path collapses into real English function words (`the`, `and`) and sentence-ending punctuation, with more variety in what repeats. That's a different kind of broken -- closer to the shape of degenerate language-model output (function-word loops are a known failure mode even in undamaged models under some decoding settings) than to pure symbol repetition. But "closer to the shape of a failure mode we recognize" is not "coherent," and nothing here should be read as "calibration solved it." At `qfilters_budget=512` against a 2,238-token prompt whose question requires reasoning over content spread across that whole passage, this specific configuration still does not produce usable output, calibrated or not.
:::

## What this changes about the original post's finding

The original QFilters post's headline claim holds: past this method's default budget, on this prompt, output collapses -- deterministically, reproducibly. What this follow-up adds is that the collapse is **sensitive to filter quality** (it's not simply what eviction always looks like at this budget, since a better filter changed its shape and roughly quadrupled the achievable compression-adjusted throughput) but is **not resolved by filter quality alone** at this budget. The next honest variable to sweep isn't calibration -- it's the budget itself, or a larger/better-matched calibration corpus, or both. Neither was tested here.

## What I'd take from this

:::tip[Test the fix, don't assume it from the docstring]
"A real calibration module exists in this repo" was true before this post ran and would have been an easy thing to cite as sufficient. It wasn't -- the module does what it says (real query-SVD, sign-fixed, GQA-averaged), and using it correctly still didn't produce coherent output at this budget. The only way to know that was to run it.
:::

:::tip[A speedup and a coherence fix are independent axes, again]
The VecInfer post found a real kernel speedup with a correctness gap. This post finds a real calibration-driven speedup (3.6x) with no coherence fix. Two different mechanisms, same lesson: throughput moving in the right direction says nothing about whether the text is any good, and the only way to know is to read the text.
:::

:::tip[A negative result on the fix is still a result]
The header of this post could have been "calibration solves it" (flattering, and false) or nothing at all (if the run had been quietly discarded for not showing the hoped-for improvement). Reporting "it's different and faster but still not coherent" is the accurate middle ground, and it's the more useful one for anyone deciding whether to trust QFilters at this budget in production.
:::

:::info[So what budget do you actually need?]
This post held `qfilters_budget=512` fixed and varied the filter quality. The next post in this sub-series does the opposite -- [It's a Cliff, Not a Slope](/blog/qwen3-8b-qfilters-budget-sweep) holds calibration fixed and sweeps the budget from 512 up to and past the prompt length, and finds the transition from broken to coherent is a sharp threshold around 80-92% of the prompt's token count, not a gradual slope.
:::

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), QFilters `qfilters_budget=512`, 5 interleaved repeats, 120 max tokens, long prompt only (2,238 tokens -- the configuration that exceeds budget). Calibration script: [`benchmark_scripts/calibrate_qwen3_8b_qfilters.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/calibrate_qwen3_8b_qfilters.py). Comparison script: [`benchmark_scripts/benchmark_qwen3_8b_qfilters_calibrated.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_qfilters_calibrated.py). See the [original QFilters post](/blog/qwen3-8b-qfilters-honest-benchmark) for the finding this follows up on, and [`qfilters_calibration.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/quantizers/qfilters_calibration.py) for the calibration mechanism itself.*

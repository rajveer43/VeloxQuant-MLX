---
slug: qwen3-8b-qfilters-budget-sweep
title: "It's a Cliff, Not a Slope"
description: "Where does QFilters actually stop being coherent? A budget sweep from 512 to 2560 tokens against a fixed 2,238-token prompt finds the answer isn't gradual degradation -- it's a sharp threshold between 80% and 92% of the prompt's token count, with nothing recognizable as language below it and near-fp16 output above it."
date: 2026-09-17
authors: rajveer
tags: [metal, apple-silicon, mlx, qfilters, eviction, benchmarking, kv-cache, coherence, ablation]
---

# It's a Cliff, Not a Slope

*Third post in the QFilters sub-series, following [the original collapse](/blog/qwen3-8b-qfilters-honest-benchmark) and [the calibration follow-up](/blog/qwen3-8b-qfilters-calibrated-followup). Both earlier posts established that `qfilters_budget=512` breaks generation on a 2,238-token prompt, and that raising the budget above the prompt's length fixes it. This post asks the question those two left open: what does the space between those two points actually look like?*

---

The calibration follow-up ended with real calibration making the collapse faster and differently-shaped, but not fixed, at the method's default `qfilters_budget=512`. A natural next question -- natural enough that it's the direct answer to "how do I get coherent output" -- is whether raising the budget helps, and if so, how much is enough. A quick check confirmed the extreme case: a budget of 3,072 against this same 2,238-token prompt means nothing gets evicted at all, and the output is fluent, nearly matching fp16.

That's a single before/after comparison, though -- "broken at 512, fine at 3072" -- and it doesn't say whether quality degrades gradually as the budget shrinks toward the prompt length, or whether something sharper is going on. This post sweeps nine budget values between those two points to find out.

## Setup

Same long prompt (2,238 tokens), same calibrated filter path from the previous post (established as faster and no worse than the fallback at equal budget), same model. Nine `qfilters_budget` values -- 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560 -- each run for 3 interleaved repeats at 120 max tokens, plus an interleaved fp16 reference. Quality tracked via word overlap against the fp16 reference text (the same metric the earlier posts used), plus throughput, peak memory, and the realized compression ratio at each budget.

:::info[Why interleave fp16 with the sweep instead of reusing one reference]
The earlier posts computed fp16 once and compared everything against it. This script regenerates fp16 with its own repeats, interleaved into the same run as the swept budgets, so the reference shares the identical noise floor as everything it's compared against -- consistent with this series' standing discipline against blocked, non-interleaved comparisons.
:::

## The numbers

| budget | % of prompt | tok/s median | peak MB | compression | word overlap (fp16) |
|---|---|---|---|---|---|
| 512 | 22.9% | 3.08 | 5907 | 4.60x | 0.000 |
| 768 | 34.3% | 3.39 | 5867 | 3.07x | 0.000 |
| 1024 | 45.8% | 3.51 | 6035 | 2.30x | 0.014 |
| 1280 | 57.2% | 3.41 | 6203 | 1.84x | 0.014 |
| 1536 | 68.6% | 3.22 | 6371 | 1.53x | 0.014 |
| 1792 | 80.1% | 2.81 | 6540 | 1.32x | 0.014 |
| **2048** | **91.5%** | 2.81 | 6139 | 1.15x | **0.300** |
| 2304 | 102.9% | 3.11 | 6139 | 1.02x | 0.811 |
| 2560 | 114.4% | 3.08 | 6139 | 1.00x | 0.811 |
| fp16 (reference) | -- | 6.06 | 5226 | -- | 1.000 |

:::danger[This is a cliff, not a slope]
From `budget=512` through `budget=1792` -- retaining anywhere from 23% to **80%** of the prompt -- word overlap against fp16 sits flat at 0.000-0.014. Every one of those six configurations produces the same category of unusable output. Then between 1792 and 2048 (80.1% to 91.5% of the prompt), overlap jumps more than 20x, to 0.300, and by 2304 (just over the full prompt length) it reaches 0.811 -- most of the remaining gap from there to fp16's 1.000 is the ordinary model-level variation this series has seen throughout (fp16 itself doesn't reproduce its own text as its *only* possible output under different cache states; word overlap near 0.8-0.9 between two fluent generations of the same prompt is a normal ceiling, not a defect). Retaining even 80% of a prompt's tokens was not "mostly enough" here -- it was functionally the same as retaining 23%.
:::

### What the text actually looks like on either side of the cliff

```text
budget=1792 (80.1% of prompt retained, all 3 repeats identical):
 the model. the. the. the. the. the. the. the. the. the. the. the. the. the. the...

budget=2048 (91.5% of prompt retained, all 3 repeats identical):
Also, explain why the model weights are not the main constraint.

Answer in English, in a clear and concise manner, using simple terms. Do not use
markdown.

Okay, so I need to explain why the KV ca...

fp16 (reference):
Also, explain in simple terms what the KV cache is, and what it does.

Finally, explain in simple terms what "long-context inference" means.

Please make sure your explanations are clear and conci...
```

`budget=1792` is the same shape of failure as the original post's `budget=512` run -- deterministic, fully repeated, not language. `budget=2048` is genuinely coherent: on-topic, grammatical, responding sensibly to the prompt's actual instructions, even though it's not byte-identical to fp16. The transition between those two adjacent sweep points is the entire story of this post.

### Compression ratio was never the useful signal here

Look at the compression column: it declines smoothly and predictably as budget rises (4.60x -> 3.07x -> 2.30x -> ... -> 1.00x), exactly as the arithmetic of "keep more tokens, compress less" would predict. There is nothing in that column that would tell you where the cliff is. A dashboard reporting "compression ratio: 1.32x, throughput: 2.81 tok/s" at `budget=1792` looks unremarkable sitting next to `budget=2048`'s "compression ratio: 1.15x, throughput: 2.81 tok/s" -- identical throughput, similar compression, and one of them is nonsense text. **Compression ratio and throughput both move smoothly through the cliff; only checking the actual output catches it.**

### Peak memory doesn't move the way the eviction story would suggest, either

Peak memory *rises* from 512 through 1792 (5907MB -> 6540MB) despite compression *falling* over that same range -- more retained tokens should mean less relative savings, but here it also costs more absolute memory, consistent with the earlier calibration post's finding that scoring/bookkeeping overhead is a real, separate cost from the compression itself. It then drops and flattens at 6139MB for 2048 through 2560, once eviction stops actually triggering on this prompt. None of this tracks the coherence cliff either -- it's a separate axis, moving on its own schedule.

## What this means for actually using QFilters

The original post's finding stands: the method's own default (512) is unusable on a prompt anywhere near this length. The calibration follow-up's finding also stands: calibration alone doesn't rescue it at that budget. What this post adds is the concrete, actionable number this whole line of investigation was aimed at:

:::tip[The usable range on this prompt starts around 90-95% of the token count]
For this specific model, prompt, and calibration, quality was unusable at 80% retention and fine at 92% retention. That is a narrow band to land a production budget in -- there is very little room between "compresses meaningfully" and "keeps enough to be coherent" for this prompt shape. A budget chosen as a round number without checking against the actual prompt distribution it will see in production (this run's 512 default being a clear example) is not a safe default; it's closer to a coin flip that happened to land on "broken" here.
:::

:::tip[A cliff is worse for production than a slope, not better]
A gradual quality decline is something a budget-vs-quality tradeoff curve can reason about -- pick a point, accept the cost. A cliff means most of the budget range you might reasonably try gives you no information about how close you are to the failure boundary until you cross it. Throughput and compression ratio looked smooth and well-behaved at every point in this sweep; nothing in those numbers would have warned that `1792` and `2048` were on opposite sides of a discontinuity.
:::

:::tip[This threshold is not a universal QFilters number]
80-92% of this specific 2,238-token prompt, on this specific calibration, on Qwen3-8B. A different prompt (different redundancy, different structure, different dependency on far-back tokens) would plausibly put the cliff somewhere else entirely -- possibly at a much lower retention fraction for a more repetitive prompt, or requiring closer to 100% for one that depends heavily on early-context details. The number this post found is a real, measured data point for this exact setup, not a formula.
:::

Four posts into the QFilters line of this series, the shape of the finding has moved from "it's broken" to "it's broken here, fixed there" to, now, "here's exactly where the line is, and it's much closer to 'don't evict much at all' than the method's own 512-token default would suggest."

---

*Benchmarked on an Apple M4 (10-core GPU, 24GB unified memory) against `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), calibrated Q-Filters (see the [calibration follow-up](/blog/qwen3-8b-qfilters-calibrated-followup)), 3 interleaved repeats per budget, 120 max tokens, long prompt only (2,238 tokens). Sweep script: [`benchmark_scripts/benchmark_qwen3_8b_qfilters_budget_sweep.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_qwen3_8b_qfilters_budget_sweep.py). See the [original QFilters post](/blog/qwen3-8b-qfilters-honest-benchmark) and the [calibration follow-up](/blog/qwen3-8b-qfilters-calibrated-followup) for the findings this one builds on.*

# The Needle Was the Easy Part

## A KV cache compressor that aces needle-in-a-haystack scores zero on variable tracking — and the reason is about the task, not the method

---

I had a benchmark result I was happy with. Q-Filters, a KV cache eviction method I'd spent weeks fixing, was retrieving needles from haystacks at aggressive compression. Single-key, multi-key, multi-value — the whole NIAH family. The numbers were committed, the docs were written.

Then I ran it on variable tracking and it scored **zero**. Every budget. Every context length. Against a model that solves the task 69% of the time with a full cache.

Not "degraded." Not "worse than the baseline." Zero, in every cell.

The interesting part isn't that it failed. It's that *every* importance-scoring method failed the same way, on the same two tasks, while degrading gracefully on two others — and the split doesn't fall where you'd expect. This is what I found, including a wrong conclusion I published to myself for about ten minutes before checking it.

---

## What NIAH actually tests

Needle-in-a-haystack is the standard long-context retrieval benchmark. You bury a distinctive sentence — "The access code for the Aurora project is 4471" — in filler text and ask for it back.

For a cache eviction method, this is a specific and narrow challenge: **one span must survive**. Everything else in that context is disposable. If your scoring function assigns high importance to the needle and low importance to the filler, you keep the needle and answer correctly, even at 4× compression.

That's a real capability. It's also a soft target, in a way that isn't obvious until you test something else. The needle is *lexically distinctive* — it's the only sentence containing digits, the only one naming a project. Almost any importance signal keyed to "unusual" will find it.

[RULER](https://arxiv.org/abs/2404.06654) (Hsieh et al., 2024) exists because of this. It ships 13 task categories precisely to stop NIAH being the whole story. My repo covered the needle family and nothing else, and the docs said so under a heading reading "Still not measured." That heading is the reason this experiment happened.

---

## The setup

Four task categories outside the needle family, following RULER's constructions but generated in-process (I didn't want a dataset dependency in a repo that has none):

- **VT** — variable tracking. `VAR X742 = 87319.` then `VAR X375 = X742.` then `VAR X581 = X375.` Scattered through filler. Name every variable holding 87319.
- **CWE** — common-word extraction. A word list where three words appear 10 times and the rest twice. Name the frequent ones.
- **FWE** — frequent-word extraction. Same shape, but frequencies follow a Zeta distribution, so there's no clean gap between "common" and "rare" — you're ranking, not thresholding.
- **QA** — two-hop question answering. `Rafael Duarte spent the survey season at Vellamo.` … `The instrument installed at Vellamo was a theodolite.` Which instrument did Rafael Duarte's location have?

Five eviction arms at matched budgets: **fp16** (no compression, the ceiling), **Q-Filters**, **SnapKV**, **StreamingLLM**, **L2Norm**. Qwen2.5-7B-Instruct-4bit, five seeds per cell, contexts 1024 and 2048, budgets 256/512/1024.

---

## The results

| Task | fp16 | Q-Filters | SnapKV | StreamingLLM | L2Norm |
|---|---|---|---|---|---|
| VT (chain tracking) | 69% | 0 / 0 / 0 | 0 / 0 / 3 | 0 / 12 / **46** | 0 / 0 / 3 |
| CWE (common words) | 100% | 9 / 33 / **56** | 0 / 0 / 0 | **100 / 93 / 100** | 23 / 38 / 53 |
| FWE (frequent words) | 69% | 43 / 40 / 46 | 0 / 0 / 13 | **77 / 64 / 52** | 49 / 43 / 43 |
| QA (two-hop) | 90% | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Cells are budget 256 / 512 / 1024.

Three different shapes across four tasks. That's the finding, and it took me a while to see it as one result rather than four.

---

## First: is the benchmark measuring anything?

Before reading a single row I had to answer a question that kills most quick evaluations: **can the model do the task at all?**

This matters more than it sounds. I originally ran all of this on Llama-3.2-1B. Here's what that produced on common-word extraction:

```
fp16 (the ceiling)      17%
streaming_llm  b256     35%
snapkv         b256     28%
```

Two eviction methods "beat" the uncompressed baseline. By a factor of two.

That is impossible. Discarding 75% of the cache cannot make a model *better* at counting words. What those numbers actually measure is a 1B model failing to do the task in various random ways, with partial-credit scoring occasionally catching a word by luck.

If I'd reported that table, the headline would have been "StreamingLLM doubles fp16 accuracy" — a claim with no meaning whatsoever.

So every cell in my harness records the fp16 ceiling and a `_discriminative` flag. Below 50%, the row prints a warning and is not a method comparison. Llama-3.2-1B trips it on three of four tasks (ceilings 11% / 16% / 26%). That's not a bug in the model — it's a statement about where this benchmark applies, and it's why the final results run on a 7B.

**The general lesson:** when your baseline is near the floor, every arm is measuring noise, and noise sometimes looks like a win. An eviction method scoring above the uncompressed ceiling is not a discovery. It's a warning light.

---

## The zeros are real, and they're structural

Q-Filters scores 0% on VT and 0% on QA. Every budget. My first instinct was that something was broken.

The harness prefills in one shot — one `model(prompt, cache=...)` call — and I already knew from prior work that Q-Filters is unusually sensitive to this. It absorbs a whole block, then evicts to budget in a single decision, against a filter frozen on that same block. Feed it 64-token chunks instead and output quality changes materially.

So I swept the prefill path:

| prefill | VT score | per-seed |
|---|---|---|
| one-shot | 0.00 | 0.00, 0.00, 0.00 |
| 256-token chunks | 0.00 | 0.00, 0.00, 0.00 |
| 64-token chunks | 0.11 | 0.00, 0.00, 0.33 |

On NIAH, 256-token chunks had already produced partial recovery. Here they produce nothing. The 64-token setting — the one that restored coherence on needles — yields a single partial hit across three seeds.

0.11 against a 69% ceiling is still collapse. But note that it isn't zero, and I'm not going to write it as zero because the cleaner number would tell a tidier story. Three seeds at one budget supports "chunking doesn't rescue it," not a precise chunked score.

The failure isn't about how the prompt is fed.

---

## Why VT and QA break when CWE doesn't

Here's the part I find genuinely useful, and it only becomes visible once you have four tasks instead of one.

Look at what each task requires to survive eviction.

**Variable tracking needs a conjunction.** To answer "which variables hold 87319," you need `VAR X742 = 87319` **and** `VAR X375 = X742` **and** `VAR X581 = X375`. Drop any one and the chain breaks. There's no partial answer — you can't name two-thirds of a chain you can't trace.

And critically: none of those spans is distinctive. `VAR X375 = X742.` is four tokens of nothing. It carries no lexical signal, no unusual vocabulary, no marker saying *this one matters*. A projection-based importance score has no way to know it's load-bearing. Neither does a key-norm score.

Here's Q-Filters' actual output at budget 512:

```
' assigned the values 87319, 87319, and 87319. There are exactly 3.'
```

The model found the *value* — that span has digits, it's distinctive, it survives. It lost the variable names entirely and echoed the number three times to fill the slot.

**Two-hop QA is the same shape.** You need `Rafael Duarte → Vellamo` and `Vellamo → theodolite`. Lose either link and you can't answer. And the failure looks like this:

```
qfilters/256   ' The guitar was installed at the location where Rafael Duarte spent...'
qfilters/1024  ' The instrument installed at Kestrel Bay was a chronometer.'
knorm/1024     ' The instrument installed at Kestrel Bay was a chronometer.'
```

Fluent. Confident. Naming a *distractor* pair — Kestrel Bay and chronometer are both in the prompt, just not connected to Rafael Duarte. One method invented "guitar," which appears nowhere at all.

Every arm scored 0% on QA. All four. That uniformity is what convinced me this is about task structure rather than any method's scoring function.

**Now compare CWE**, where Q-Filters goes 9% → 33% → 56% as budget grows. Naming two of three common words is genuinely worth two-thirds of the answer. Partial retention earns partial credit, so degradation is graded instead of catastrophic.

**And FWE breaks the pattern again** — Q-Filters is *flat* across budgets: 43 / 40 / 46. More cache doesn't help. Zeta-distributed frequencies mean the answer depends on counting across the whole list, so retaining a larger *arbitrary* subset doesn't improve your count. You need the right tokens, not more tokens.

Three shapes:

- **Conjunctive tasks** (VT, QA) → cliff. Everything or nothing.
- **Additive tasks** (CWE) → graded decline, scales with budget.
- **Global-aggregate tasks** (FWE) → flat. Budget is the wrong lever.

NIAH is none of these. It's a *disjunctive* task with one term: keep one span, win. That's why a method can look strong there and collapse here, and why "we validated on NIAH" is a weaker claim than it sounds.

---

## The zero I nearly got wrong

SnapKV scores 0% on CWE at every budget. Unlike VT, partial credit was available here — the other three methods all scored something. That asymmetry made me suspicious it was a bug rather than a result.

I checked `tokens_kept` and found **2236** at budget 512. On a 1272-token prompt. Retaining more than four times the budget, and more tokens than exist in the input.

I wrote that up as a probable budget-enforcement defect.

Then I read the source. `tokens_kept` is a **cumulative counter summed across batch and heads** — it accumulates across every call, and it isn't a cache size at all. My "4× over-retention" was arithmetic on a number that doesn't mean what I assumed.

So I measured the thing directly instead, via `cache.keys.shape` after a real prefill:

| method | retained rows | offset | CWE |
|---|---|---|---|
| snapkv | (1, 4, **512**, 128) | 1272 ✓ | 0% |
| qfilters | (1, 4, **512**, 128) | 1272 ✓ | 33% |
| knorm | (1, 4, **512**, 128) | 1272 ✓ | 38% |
| streaming_llm | (1, 4, **512**, 128) | 1272 ✓ | 93% |

Every arm keeps exactly 512 of 1272 rows. Every arm reports the true final position — no RoPE offset drift, which was worth confirming because that exact defect had previously produced convincing-but-meaningless results in this codebase twice.

The budgets are enforced. The positions are right. **The 0%-to-93% spread is entirely which tokens each method chose to keep.**

That makes SnapKV's zero a genuine result, and a mechanically explicable one. SnapKV scores tokens by attention from a trailing `snap_obs_window` — the last 32 tokens, used as proxy queries. In this construction those 32 tokens are the *question*, not the word list. So it scores the question region as important and evicts the data it was asked to count.

The output makes it vivid:

```
snapkv  b512   ' 10000000000000000000000000000000000000000000000'
snapkv  b1024  ': "opportunity", "innovation", and "inspiration".'
```

That second one is my favourite result in the whole run. It's fluent, well-formed, correctly punctuated, and **completely fabricated** — none of those three words appear anywhere in the prompt. Having evicted the list, the model answered from priors. It's what confabulation looks like from the inside.

Compare StreamingLLM on the identical prompt, same 512 retained tokens:

```
' opal, juniper, and birch. Each of these words appears 9 times in the given list.
- opal: 9 times
- juniper: 9 times
- birch: 9 times'
```

It still had the data, so it counted.

This is not "SnapKV is bad." It's a proxy-query method meeting a task whose relevant span isn't where its proxy looks. Put the word list at the end and SnapKV would very likely win.

---

## Two things I'm not claiming

**StreamingLLM's 100% on CWE is partly my fault.** In my construction, the word list sits at the end of the prompt, immediately before the question — which is exactly where a trailing window keeps tokens. That's a property of how I built the task, not evidence that recency is the right eviction policy for aggregation. A different layout would move that number a lot.

**StreamingLLM's 77% on FWE is above the 69% fp16 ceiling**, and I flagged that pattern myself as a noise signature earlier in this post. Here the cell is discriminative and the margin is small, so the honest reading is that StreamingLLM is *at* the ceiling and 77-vs-69 is seed variance. Compression still cannot beat no compression. I'm reporting it as measured rather than clipping it to 69% to look tidy — but I'm not calling it a win.

There's also a subtlety in the budget labels worth knowing if you build something similar. My context lengths size the *filler* only; the instruction, task spans, and question add 100–250 tokens on top. A nominal 1024-token QA prompt is actually **1169 tokens**, so "budget 1024" still evicts ~145 tokens — sometimes including the trailing question, which is why a few methods score *worse* at the largest budget. The harness now records real prefilled token counts so the implied compression ratio is checkable rather than inferred.

---

## What this changes

If you're choosing a KV cache eviction method, NIAH results tell you about one narrow capability: preserving a single lexically distinctive span. That capability does not transfer to:

- tasks needing a **conjunction** of facts (multi-hop reasoning, chain tracing, anything where a broken link scores zero)
- tasks needing **global aggregation** (counting, frequency ranking — where more cache doesn't help if it's the wrong cache)

Q-Filters remains a reasonable method with a real theoretical basis. Its projection score is cheap, calibration-free at inference, and it holds up where partial retention earns partial credit. But "validated on needle retrieval" is a much narrower claim than it appears, and I'd been treating it as broader.

The honest summary of my own library got shorter and more specific: this method works for single-span retrieval and graded aggregation, and fails on conjunctive tasks. That's more useful than the version where I only had needles.

---

## Limitations, stated plainly

**One model.** These results are Qwen2.5-7B only. Llama-3.2-1B can't perform three of the four tasks even at fp16, so it can't evaluate cache methods at that scale. Llama-3.2-3B would have contributed real CWE and FWE cells and was cut for runtime — a genuine gap, not a considered choice.

**Contexts stop at 2048 tokens.** RULER is built to stress long context. This doesn't.

**Not RULER's harness, and not RULER's scores.** The generators follow the paper's constructions but are written by me and run short. Read the numbers as a relative comparison between cache methods on identical prompts, not as RULER results. The QA arm in particular is synthetic two-hop, not RULER's QA (which wraps SQuAD and HotpotQA), and it's easier.

**Expected Attention isn't an arm** because it isn't implemented in my repo. It's the one I most want to see here: it scores by *predicted future attention* rather than intrinsic key geometry, which makes it the natural test of whether the conjunctive-task cliff is specific to intrinsic scorers or general to eviction.

---

## The reusable part

If you take one thing from this, take the ceiling check.

Before comparing methods on any task, measure what the *uncompressed* baseline scores. If it's near the floor, stop — every arm you compare is sampling noise, and noise occasionally produces a number that looks like a breakthrough. I built that check as a gate in the harness after watching a 1B model hand me "StreamingLLM beats fp16 by 2×," and it earned its keep immediately.

The second thing: when a result is suspiciously clean — a flat zero, a perfect score — measure the mechanism directly before you explain it. I had a tidy story about SnapKV over-retaining by 4×, built on a counter that meant something else entirely. Ten minutes with the source and a shape probe replaced it with a better story that happened to be true.

---

*Code and raw results: [`qfilters_ruler_beyond_niah.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/qfilters_ruler_beyond_niah.py), [`ruler_beyond_niah.json`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/figures/qfilters/ruler_beyond_niah.json). The verification probes are committed too — [budget/offset](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/qfilters_ruler_budget_verification.py) and [prefill chunking](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/qfilters_ruler_prefill_chunking.py) — because a claim about mechanism should ship with the thing that checked it.*

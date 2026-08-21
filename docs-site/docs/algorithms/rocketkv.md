---
id: rocketkv
title: RocketKV-adapted
sidebar_label: RocketKV-adapted
slug: /algorithms/rocketkv
---

# RocketKV-adapted — Two-Stage Compression (SnapKV Eviction + Hybrid Sparse Attention)

**Method id:** `rocketkv` · **New in 0.56.0** · *Inspired by* ["RocketKV:
Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"
(Behnam, Fu, Zhao, Tsai, Yu, Tumanov; NVIDIA / Georgia Tech; ICML 2025,
arXiv:2502.14051)](https://arxiv.org/abs/2502.14051) — **RocketKV-adapted
(VeloxQuant-MLX implementation)**, not a faithful port.

## The problem this solves

Every eviction method in this library ([H2O](../algorithms/h2o),
[SnapKV](../algorithms/snapkv), [PyramidKV](../algorithms/pyramidkv), …) has
to commit to which tokens matter *once*, using only what's visible at the
point of eviction. Push the retained budget low enough and accuracy falls
off a cliff — the paper's own measurement on Mistral-7B/qasper shows every
practical method losing meaningfully more accuracy than an oracle top-k
scheme once the budget drops under ~1024 tokens, even though the oracle
itself stays flat down to 256.

The paper's insight: run a **cheap, coarse eviction pass first**, then a
**second, much more accurate dynamic selection pass** over what survived.
Dynamic top-k prediction is hard over the full sequence, but easy over an
already-filtered candidate set — a random attention head in their qasper
analysis needed a top-256 union across *all* decoding steps of at most 1200
unique indices, out of a sequence that reached 25,000 tokens. RocketKV
exploits that gap directly: stage 1 throws away everything **except** that
small surviving set (reusing [SnapKV-adapted](../algorithms/snapkv)
verbatim — the paper adopts SnapKV as-is for this stage), and stage 2 runs a
fresh top-k approximation, called **Hybrid Sparse Attention (HSA)**, over
just the survivors at every decode step.

## Where it sits — the mechanism gap

| Method family | When selection happens | Selection accuracy at low budgets |
|---|---|---|
| Eviction only (H2O, SnapKV, PyramidKV, …) | Once, at prefill | Degrades sharply below ~1024 tokens |
| Dynamic-only (Quest-style, SparQ-style) | Every decode step, over the FULL cache | Also degrades — one-dimensional approximation |
| **RocketKV-adapted** | **Both** — coarse evict once, then dynamic top-k over the survivors every step | **Matches oracle top-k far more closely at the same budget** |

Unlike single-dimension dynamic methods, HSA approximates attention scores
with a **two-dimensional** reduction: Quest-style paged element-wise
max/min summaries along the *sequence* dimension, combined with SparQ-style
top-magnitude channel selection along the *head* dimension. The two
combine into a tighter per-page score bound than either reduction gets
alone.

## Quick start

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="rocketkv",
    head_dim=model.args.head_dim or (model.args.hidden_size // model.args.num_attention_heads),
    rocketkv_compression_ratio=8.0,  # overall target ratio; adaptively split across both stages
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Summarize the attached document.",
    max_tokens=200,
)
```

## The one setting worth tuning: `rocketkv_compression_ratio`

The paper's §3.6 adaptive decomposition formula
(`r = clip(0.2 + 0.06*log2(c), 0.2, 0.8)`) splits your target ratio `c` into
a stage-1 (eviction) ratio `c^r` and a stage-2 (HSA) ratio `c^(1-r)`
automatically — you set one number and the split follows the paper's own
worked logic (their example: `c=64` splits to `10.3x` at stage 1, `6.2x` at
stage 2). `rocketkv_page_size` and `rocketkv_head_topk1` let you override
the derived HSA split directly if you want manual control instead.

| `rocketkv_compression_ratio` | What happens |
|---|---|
| Small (`2`–`4`) | Most of the compression stays in stage 1 is avoided — `r` stays near its floor of `0.2`, so stage 1 barely compresses and stage 2 does most of the work |
| Large (`64`–`400`) | `r` climbs toward its ceiling of `0.8` — stage 1 does most of the compression, since SnapKV's exact-attention selection is more reliable than HSA's approximation at extreme ratios |

## Should you use this?

**It's a good fit if** you're already using SnapKV and hitting an accuracy
wall as you push the token budget down — RocketKV keeps SnapKV's storage
savings and layers a second, cheaper-to-compute selection pass on top that
the paper shows closes most of the remaining gap to oracle top-k.

**Look elsewhere if:**

- **You want RocketKV-MT's multi-turn behavior** — the paper's variant that
  skips permanent eviction so earlier-turn tokens survive for later
  queries. Not implemented here (see
  [issue #239](https://github.com/rajveer43/VeloxQuant-MLX/issues/239)); use
  plain [SnapKV-adapted](../algorithms/snapkv) or accept the same
  multi-turn accuracy risk any eviction method carries.
- **You want a fused decode kernel** — HSA's page gather/attend runs in
  eager MLX ops each step here, not the paper's tiled FlashAttention-style
  kernel.

## Adaptation notes — what we do NOT implement

- **Key-as-query proxy**, inherited unchanged from
  [SnapKV-adapted](../algorithms/snapkv) for stage 1, and used again for
  stage 2's per-step HSA query — a cache wrapper's `update_and_fetch` only
  ever sees keys and values, never the model's true query vector.
- **Page-granularity selection, not token-granularity.** The paper's own
  Algorithm 1 selects whole pages (`k2` of them), so a selected page
  contributes `page_size` tokens to the sparse-attention set, not exactly
  one. This matches the paper, not a simplification introduced here.
- **No fused kernel.** Reconstruction/gathering happens eagerly in MLX ops
  every decode step.
- **No RocketKV-MT.** The multi-turn variant is a separate, not-yet-built
  method — see [issue #239](https://github.com/rajveer43/VeloxQuant-MLX/issues/239).

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_rocketkv.py` and
`veloxquant_mlx/tests/cache/test_rocketkv_cache.py`:

- **`test_hsa_scores_bound_true_attention_within_page`** — HSA's per-page
  approximate score is a genuine upper bound on every token's true
  dot-product within that page (the paper's Step 2 invariant), verified
  numerically against brute-force dot products.
- **`test_append_matches_full_rebuild_exact_pages`** /
  **`test_append_single_token_at_a_time_matches_rebuild`** — the
  incremental paged-summary update used at decode produces bit-identical
  results to rebuilding the summary from scratch on the full sequence.
- **`test_split_compression_ratio_matches_paper_worked_example`** — the
  adaptive decomposition reproduces the paper's own `c=64 → 10.3x/6.2x`
  worked example.
- **`test_prefill_evicts_when_over_budget`** /
  **`test_chunked_prefill_reenforces_budget`** — stage-1 eviction behaves
  like [SnapKVKVCache](../algorithms/snapkv), including re-enforcing the
  budget across `mlx_lm`'s chunked-prefill calls.
- **`test_offset_tracks_true_position_not_row_count`** — the same
  RoPE-correctness split `SnapKVKVCache` needed (issue #171) holds here too.
- Byte accounting, determinism, `for_model` config propagation.

**No model-level benchmark has been run.**
`benchmark_scripts/benchmark_rocketkv.py` is offline-synthetic — attention
selection-overlap and byte-accounting comparisons against
[SnapKV-adapted](../algorithms/snapkv) at matched budgets, not perplexity or
throughput on a real model, and not a reproduction of the paper's
A100/H100 LongBench/NIAH/RULER numbers.

## When to use it

| Method | Selection passes | Dynamic (per-step) selection | Verified venue |
|--------|:---:|:---:|:---:|
| [SnapKV-adapted](../algorithms/snapkv) | 1 (prefill only) | No | Yes |
| [H2O-adapted](../algorithms/h2o) | Ongoing (every step, cumulative) | Yes (score update, not re-selection) | Yes |
| **RocketKV-adapted** | **2 (prefill evict + per-step HSA)** | **Yes (HSA re-selects every step)** | **Yes** |

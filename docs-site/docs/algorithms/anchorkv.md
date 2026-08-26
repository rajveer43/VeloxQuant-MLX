---
id: anchorkv
title: AnchorKV-adapted
sidebar_label: AnchorKV-adapted
slug: /algorithms/anchorkv
---

# AnchorKV-adapted — Anchor-Residual Compression, No Eviction

**Method id:** `anchorkv` · **New in 0.53.0** · *Inspired by* ["AnchorKV:
Anchor-Residual KV Cache Compression" (Khalaf, Shamshoum, Hodos, Sieradzki,
Schuster; Technion; arXiv:2608.02901v1)](https://arxiv.org/abs/2608.02901) —
**AnchorKV-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

:::warning[No verified peer-reviewed venue]
This is the **second** method in VeloxQuant-MLX (after
[NestedKV-adapted](../algorithms/nestedkv)) that does not trace to a
verified peer-reviewed venue. As of 2026-08-20, the paper is a single arXiv
revision (submitted 2026-08-03, "Preprint. Under review." on its own first
page) with no Comments/journal-ref field indicating acceptance anywhere.
This ships as a **second, one-time, user-directed exception** to the
standing venue-verification rule, following the exact precedent set for
NestedKV. See
[`NEW_METHOD_SURVEY_V22.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/paper/research/surveys/NEW_METHOD_SURVEY_V22.md)
for the full rationale. The rule itself is unchanged for every method after
this one.
:::

## The problem this solves

Every other compression method in this library picks one of two extremes.
**Eviction** methods ([H2O](../algorithms/h2o), [SnapKV](../algorithms/snapkv),
[PyramidKV](../algorithms/pyramidkv), [NestedKV](../algorithms/nestedkv), …)
score tokens and throw the low-scoring ones away completely. That gets you
very high compression, but a discarded token is gone for good — if a later
question needs exactly that token, there's nothing left to answer it with.
**Quantization** methods ([KIVI](../algorithms/kivi), [SVDq](../algorithms/svdq),
[GEAR](../algorithms/gear), …) keep every token but store it in fewer bits.
Nothing is ever unreachable, but accuracy falls off sharply once you push
below about 2 bits per value, which caps how far this axis can go.

AnchorKV-adapted tries to avoid choosing between those two costs. It never
removes a token from the cache — every position stays available to every
future query — but it doesn't store every token at the same cost either.
A small number of tokens (**anchors**) are kept exactly. Every other token
is represented as a cheap pointer to its closest anchor plus one scalar
number (a "how far along that direction" coefficient) — a few bytes instead
of a full vector. Whatever byte budget is left over after that buys a small
correction (a **residual**) for the handful of tokens whose cheap
approximation would hurt the model's output the most.

## Where it sits — the mechanism gap

| Method family | Every token reachable later? | Per-token cost |
|---|:---:|---|
| Eviction (H2O, SnapKV, PyramidKV, …) | **No** — dropped tokens are gone | Full precision (kept) or zero (dropped) |
| Quantization (KIVI, SVDq, GEAR, …) | Yes | Uniform low bit-width for every token |
| **AnchorKV-adapted** | **Yes** | **Non-uniform: anchor+coefficient for most, anchor+coefficient+residual for the highest-impact tokens** |

The one user-facing knob is `anchorkv_theta`: the fraction of the
uncompressed cache you want to keep. Everything else — how many anchors,
which tokens get a residual — is derived from that single number.

## Quick start

No calibration step is needed — unlike [A2ATS-adapted](../algorithms/a2ats),
AnchorKV-adapted builds its anchors and residual budget directly from the
prompt itself, once, at the end of prefill.

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="anchorkv",
    head_dim=model.args.head_dim or (model.args.hidden_size // model.args.num_attention_heads),
    anchorkv_theta=0.1,  # retain ~10% of the uncompressed byte cost
    anchorkv_window=32,  # trailing tokens always kept as anchors
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

:::danger[Right now this doesn't shrink your process's actual memory use]
Like [A2ATS-adapted](../algorithms/a2ats), `compression_ratio` here is real
byte-accounting arithmetic describing the compressed representation — but
`update_and_fetch` reconstructs a full dense fp16 tensor on every call so it
can hand it to the standard `mlx_lm` attention path. The paper's own memory
win comes from a fused kernel that reconstructs each tile only when
consumed and never materializes the dense cache — that fused path is **not**
implemented here (see [Adaptation notes](#adaptation-notes--what-we-do-not-implement)).
If you need resident RAM to drop today, use an eviction method (🔻RSS in the
[method library](https://github.com/rajveer43/VeloxQuant-MLX#method-library)) or
[VecInfer](../algorithms/vecinfer).
:::

## The one setting worth tuning: `anchorkv_theta`

| `anchorkv_theta` | What happens |
|---|---|
| Larger (say `0.5`) | More residual slots, closer to the uncompressed cache, smaller savings |
| Smaller (say `0.02`) | Very few or zero residual slots — mostly pure anchor projection, largest savings |
| `1.0` or above | Enough budget for every non-anchor token to get a residual |

At very short contexts, `anchorkv_theta` below roughly `0.2` can floor to
**zero** residual slots — anchors and their per-token bookkeeping (one index
+ one coefficient per side) already consume the whole budget before any
residuals are affordable. This is expected at small scale and is not a bug;
see the comment above `THETAS` in
[`benchmark_scripts/benchmark_anchorkv.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_anchorkv.py)
for a worked example.

## Should you use this?

**It's a good fit if** you want deep compression (well past what 2-bit
quantization can do) but can't tolerate the risk of a token being
permanently gone when a later query needs it — long documents with
scattered, unpredictable retrieval needs, multi-turn sessions where you
don't know in advance which earlier turn will get referenced again.

**Look elsewhere if:**

- **You need your process's RSS to drop today**, not just the accounting
  number — pick an eviction method (🔻RSS) or [VecInfer](../algorithms/vecinfer).
- **Your context is short enough that `anchorkv_theta` floors to zero
  residual slots** — at that point you're paying anchor-projection error
  with none of the residual correction that makes the method work; a
  uniform low-bit quantizer may do better at that scale.
- **You want the paper's own reported ~19× decode-memory reduction** — that
  depends on a fused tiled-reconstruction kernel not implemented here.

## Adaptation notes — what we do NOT implement

- **Key-as-query proxy.** The paper's anchor scoring and residual-utility
  estimate both use the prompt's true trailing *query* vectors. A cache
  wrapper only sees keys and values at `update_and_fetch` — never queries —
  so, following the same convention already used by
  [SnapKV-adapted](../algorithms/snapkv), the trailing `anchorkv_window` key
  rows stand in as proxy queries for both anchor selection and residual
  scoring.
- **No fused decode kernel.** Reconstruction happens eagerly in plain MLX
  ops on every `update_and_fetch` call, not inside a tiled attention kernel
  that never materializes the dense cache. The byte accounting
  (`anchorkv_bytes`, `compression_ratio`) is exact for the compressed
  representation; the transient dense tensor built to satisfy `mlx_lm`'s
  `KVCache` contract is not what's being measured.
- **One-shot prefill compression.** Anchors and the residual budget are
  fixed once, at the end of prefill — this matches the paper's own design
  (not a simplification), but it means decode tokens are always appended at
  full fp16 and never retroactively re-anchored.
- **No model-level benchmark.** The paper's RULER / LongBench /
  Needle-in-a-Haystack numbers (Llama-3.1-8B/70B-Instruct,
  Mistral-Small-3.1-24B-Instruct, NVIDIA A100-80GB) are the paper's own.
  `benchmark_scripts/benchmark_anchorkv.py` is an offline, synthetic,
  cache-primitive-level comparison against [H2O](../algorithms/h2o) at
  matched byte budgets — not a reproduction of the paper's results.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_anchorkv.py` and
`veloxquant_mlx/tests/cache/test_anchorkv_cache.py`:

- **`test_prefill_never_drops_tokens`** / **`test_decode_tokens_always_appended`**
  / **`test_tokens_kept_equals_tokens_total_always`** — direct proof that,
  unlike every eviction method in this repo, output token count never falls
  below input token count at any `anchorkv_theta`.
- **`test_anchors_reconstruct_exactly_without_residual`** — an anchor
  projected onto itself has coefficient 1 and zero residual.
- **`test_projection_reduces_norm_or_matches`** — the anchor projection plus
  residual reconstructs the original vector exactly (the orthogonal
  decomposition holds).
- **`test_residual_codec_roundtrip_reduces_error`** — the quantized residual
  is strictly closer to the true residual than storing nothing.
- **`test_allocate_residual_budget_pools_across_heads`** — the residual
  budget correctly favors the head with higher estimated utility rather
  than splitting evenly.
- Byte accounting, determinism, `for_model` config propagation (all 6
  `anchorkv_*` fields).

**No model-level benchmark has been run.**
`benchmark_scripts/benchmark_anchorkv.py` is offline-synthetic and
deterministic in all non-timing fields — attention-output relative-error
comparisons only, not perplexity or throughput on a real model.

## When to use it

| Method | Every token reachable? | Bounded during decode | Verified venue |
|--------|:---:|:---:|:---:|
| [H2O](../algorithms/h2o) | No | Yes | Yes |
| [SnapKV-adapted](../algorithms/snapkv) | No | No (one-shot prefill) | Yes |
| [NestedKV-adapted](../algorithms/nestedkv) | No | No (one-shot prefill) | No |
| **AnchorKV-adapted** | **Yes** | No (one-shot prefill) | **No (this method + NestedKV only)** |

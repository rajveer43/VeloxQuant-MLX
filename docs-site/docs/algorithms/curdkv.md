---
id: curdkv
title: CurDKV-adapted
sidebar_label: CurDKV-adapted
slug: /algorithms/curdkv
---

# CurDKV-adapted — Value-Aware Leverage-Score Eviction

**Method id:** `curdkv` · **New in 0.36.0** · *Inspired by* ["Value-Guided KV
Compression for LLMs via Approximated CUR Decomposition" (Sengupta,
Chaudhary, Chakraborty; **NeurIPS 2025**, confirmed poster,
arXiv:2509.15038)](https://arxiv.org/abs/2509.15038) —
**CurDKV-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

CurDKV joins the repo's token-eviction family
([H2O](../algorithms/h2o), [SnapKV](../algorithms/snapkv),
[TOVA](../algorithms/tova), [KNorm](../algorithms/knorm),
[Q-Filters](../algorithms/qfilters), and more) with a genuinely new mechanism
axis: instead of scoring a token by its **key** side alone, it derives a
**leverage score** from the joint key-and-value structure of the proxy
attention output, so a token's *value* contribution — not just its key or
attention-mass profile — determines whether it survives eviction.

## Where it sits — the mechanism gap

| Method | Scoring signal | Value-aware? |
|---|---|---|
| [H2O](../algorithms/h2o) | cumulative softmax attention-mass over **keys** | No |
| [KNorm](../algorithms/knorm) | intrinsic **key**-vector norm | No |
| [Q-Filters](../algorithms/qfilters) | frozen per-head **key**-SVD projection direction | No |
| [Keyformer](../algorithms/keyformer) | Gumbel-regularized attention over **keys** | No |
| [MorphKV](../algorithms/morphkv) | recent-window correlation over **keys** | No |
| [KVzip](../algorithms/kvzip) | context-reconstruction reliance (**key**-based probe) | No |
| **CurDKV-adapted** | **leverage score over the joint (key, value) block** | **Yes** |

Every eviction method already in this repo scores a token using only its key
side. **None of them fold the value vector's own contribution into the
retention decision.** A token whose key looks "important" by any of the
above criteria, but whose value vector is near-zero or points away from the
accumulated output direction, is indistinguishable — under every existing
method — from a token whose value genuinely matters. CurDKV-adapted closes
that gap: two tokens with **identical keys** but **different values**
receive **different leverage scores** here (see
`test_curdkv.py::test_identical_keys_different_values_diverge`), a
distinction none of the methods above can make by construction.

## :warning: The honesty crux — read this first

1. **Key-as-query proxy, not the true query vector.** The paper derives
   leverage scores from the ground-truth `softmax(QK^T)V` attention-output
   matrix, built from the model's real query vectors. The cache wrapper
   never sees the true query, so — exactly like
   [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv) — the incoming
   key vector stands in for it.
2. **An SVD-based, energy-weighted leverage-score estimator, not the
   paper's own CUR sampling algorithm.** We build the proxy
   attention-weighted value block (`weighted_values[i] = attn[i] *
   values[i]`) and estimate each row's leverage as an **energy-weighted**
   sum over its projection onto the block's leading left singular vectors:
   `l_i = Σⱼ (sⱼ² / Σs²) · U[i,j]²`. This is a standard, generically-cited
   leverage-score stand-in (Mahoney & Drineas-style), **not** a
   reproduction of the paper's specific CUR row/column sampling routine.
3. **Energy-weighting, not a hard top-k/bottom-(n−k) split, and this is
   load-bearing.** A hard rank cutoff degenerates whenever the retained
   rank reaches the block size `n`: the left singular vectors of a
   full-rank `[n, k]` block with `k ≥ n` form a complete orthogonal basis,
   and every row of an orthogonal matrix has unit norm **by construction**
   — silently erasing the magnitude signal this estimator exists to
   capture, no matter how small the tail singular values actually are.
   Weighting each singular direction by its own energy (`s_j²`) avoids this
   degeneracy without relying on a brittle rank threshold.
4. **New tokens are seeded with their own leverage score, not a flat 0.**
   H2O seeds a newly-appended token's score at exactly 0 (it "begins
   accumulating next step"). CurDKV cannot reuse that convention as-is:
   leverage scores can legitimately be **exactly 0** for a genuinely
   negligible-value token (unlike H2O's softmax weights, which are never
   exactly 0), so a flat-0 seed would let a negligible-value newcomer tie
   forever with already-negligible survivors, and index-based tie-breaking
   would then protect whichever one happens to sit at the lower array
   index — arrival order, not value, would decide the outcome. Seeding a
   new token with its **own** leverage score within the block it is joining
   fixes this: a negligible-value newcomer is evicted on the very next
   over-budget step instead of parking at a permanent tie.
5. **Uniform budget across all heads**, matching the repo's existing
   eviction convention (H2O, SnapKV, etc.), not any per-head tuning the
   paper may use.
6. Nothing here is validated on a trained model. The paper's headline
   numbers (**up to 9.6% higher accuracy than SOTA baselines, up to 40%
   latency reduction** under aggressive compression) are the **paper's, on
   trained models** — never quoted as this repo's own.

## The planted-geometry observable (pinned)

Two token classes, near-identical keys (so any key-only scorer treats them
alike), sharply divergent values:

- **Class 1** — key aligned with a common direction, **large,
  output-relevant value**.
- **Class 2** — the same key alignment, **near-zero value**.

At a tight token budget, CurDKV-adapted correctly retains class-1 tokens
over class-2 tokens — a rate of **8/8 trials** across seeds in
`test_curdkv.py::test_planted_geometry_curdkv_prefers_value_relevant_tokens`,
each trial retaining **all** budget slots with class-1 tokens. On the
**same** planted geometry, [H2O](../algorithms/h2o) — given the identical
keys — cannot tell the classes apart and evicts near-uniformly
(`test_h2o_blind_spot_on_same_planted_geometry` demonstrates the baseline
actually has this blind spot, not just asserts CurDKV is good in isolation).
This is the clean, always-true claim: **two tokens with identical keys and
divergent values get different CurDKV leverage scores by construction**,
which a key-only method cannot achieve regardless of budget or arrival
order.

We do **not** claim CurDKV-adapted strictly dominates H2O in general — see
the benchmark section below for the honestly-reported, more nuanced picture
across geometries.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="curdkv",
    head_dim=128,
    curdkv_budget=512,  # max tokens kept at any time (sinks + non-sinks)
    curdkv_n_sink=4,  # leading positions never evicted
    curdkv_rank_cap=16,  # leading singular directions considered per leverage-score estimate
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`CurDKVKVCache` per attention layer. No `.bits` attribute (an eviction
cache, not a quantizer) — stores and returns fp16 K/V directly.

## How it works

For every incoming token (prefill and decode both go through the same
loop — no prefill-only phase, unlike [SnapKV](../algorithms/snapkv)):

1. Compute proxy attention weights of the incoming key against all
   currently-stored keys — the same softmax formulation
   [H2O](../algorithms/h2o) uses.
2. Form the proxy attention-weighted value block
   (`weighted_values[i] = attn[i] * values[i]`).
3. Estimate leverage scores via an energy-weighted sum over the block's
   leading `min(curdkv_rank_cap, n, D)` left singular vectors.
4. Accumulate the leverage score into each existing token's cumulative
   score; seed the new token's own score with its self-leverage within the
   resulting block (not a flat 0 — see honesty crux, point 4).
5. If the cache now exceeds `curdkv_budget`, permanently evict the
   lowest-cumulative-score non-sink token (sink protection identical to
   [H2O](../algorithms/h2o): the first `curdkv_n_sink` positions get `+inf`
   protection before the `argmin`).

## Byte accounting

- `curdkv_kept_bytes` — fp16 bytes for currently retained K + V tokens.
- `full_seq_bytes` — hypothetical fp16 cost if all tokens were kept.
- `compression_ratio` — `full_seq_bytes / curdkv_kept_bytes` (> 1 = savings).
- `tokens_seen` — total token positions ever passed to `update_and_fetch`.
- `tokens_kept` — tokens currently in the first (B=0, H=0) head's cache.

## Benchmark — honestly reported, including the surprising part

`benchmark_scripts/benchmark_curdkv.py` (results in
`figures/curdkv/results.json`) sweeps sequence length
(40/80) and token budget (6/8) across `geometry ∈
{planted_value_divergence, correlated}`, comparing CurDKV-adapted against
H2O at the **same matched token budget**, reporting **class-2
(value-irrelevant) token retention rate** — lower is better at deprioritizing
value-irrelevant tokens.

- **`planted_value_divergence`**: CurDKV retains class-2 tokens at a mean
  rate of **≈0.17** across the sweep vs H2O's **≈0.50** — consistent with
  the pinned unit-test result above.
- **`correlated`** (key-distinctiveness and value-magnitude driven by the
  same per-token importance scalar, so a key-only scorer has a real,
  non-degenerate signal here too): CurDKV still retains fewer class-2
  tokens (**≈0.09**) than H2O (**≈0.51**). This was **not** the initially
  expected null-control result — the honest reading is that CurDKV is never
  structurally worse off (it uses the same key-similarity signal H2O uses,
  plus the value signal), and that H2O's single-token incremental eviction
  with exact-tie `argmin` tie-breaking is itself prone to persistent
  near-uniform splits on tightly-clustered synthetic key geometries at
  small budgets — a property of H2O's eviction dynamics in this small-N
  regime, not a claim about CurDKV's mechanism specifically. The always-true
  claim stays scoped to `planted_value_divergence` and to the direct
  same-key/divergent-value test above.

Deterministic in all non-`_ms` fields, verified by diffing two runs.
Offline-synthetic; loads no model.

## Adaptation notes — what we do NOT implement

- The paper's ground-truth `softmax(QK^T)V` attention-output matrix computed
  from real query vectors — key-as-query proxy instead, the same limitation
  [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv)/
  [Keyformer](../algorithms/keyformer)/[MorphKV](../algorithms/morphkv)/
  [KVzip](../algorithms/kvzip) already document.
- The paper's specific CUR sketching/sampling algorithm — a standard,
  generically-cited SVD-based, energy-weighted leverage-score approximation
  instead.
- Any per-head budget tuning beyond the repo's existing uniform-budget
  eviction convention.
- Any trained-model perplexity/throughput/accuracy benchmark. The paper's
  headline numbers (up to 9.6% higher accuracy than SOTA baselines, up to
  40% latency reduction) are the paper's — not reproduced here.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_curdkv.py` (23 tests) and
`veloxquant_mlx/tests/cache/test_curdkv_cache.py` (16 tests):

- Bootstrap, budget enforcement (never exceeded, exact boundary), sink
  protection (including a sink token with deliberately negligible value —
  confirms sink protection overrides leverage score, as it must).
- Degenerate all-zero-value block produces finite (non-NaN/inf) leverage
  scores.
- **Two tokens with identical keys but different values receive different
  leverage scores** — direct proof of value-awareness
  (`test_identical_keys_different_values_diverge`).
- **Planted two-class geometry**: CurDKV retains value-relevant tokens
  preferentially in 8/8 trials across seeds; H2O, given the same keys,
  does not (`test_h2o_blind_spot_on_same_planted_geometry`).
- Byte accounting, determinism (no RNG in the leverage-score path itself),
  `for_model` config propagation, multi-head/multi-batch shape correctness,
  prefill and decode through the same eviction loop.

**No model-level benchmark has been run.**
`benchmark_scripts/benchmark_curdkv.py` is offline-synthetic and
deterministic in all non-timing fields — class-2 retention-rate numbers
only, not perplexity or throughput on a real model.

## When to use it

CurDKV-adapted is for workloads where you suspect eviction decisions are
being made purely on key/attention-mass grounds and want a value-aware
tiebreaker — a token that "looks" important by key geometry alone but
contributes little to the actual output is exactly the case existing
key-only eviction methods ([H2O](../algorithms/h2o),
[KNorm](../algorithms/knorm), [Q-Filters](../algorithms/qfilters)) cannot
correct for.

| Method | Scoring signal | Value-aware |
|--------|------------------|----------------------|
| [H2O](../algorithms/h2o) | cumulative attention-mass over keys | No |
| [KNorm](../algorithms/knorm) | key-vector norm | No |
| [Q-Filters](../algorithms/qfilters) | frozen key-SVD projection | No |
| **CurDKV-adapted** | **leverage score over joint (key, value)** | **Yes** |

See also [NestedKV-adapted](../algorithms/nestedkv), which takes the
*ensembling* axis CurDKV doesn't: instead of one joint (key, value) score,
NestedKV combines **three** independent key-only signals (stable/episodic/
current) via a head-adaptive blend and surprise-gated route. The two axes
are orthogonal — value-awareness (CurDKV) vs. multi-scale ensembling
(NestedKV) — and neither subsumes the other.

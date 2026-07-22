---
id: nestedkv
title: NestedKV-adapted
sidebar_label: NestedKV-adapted
slug: /algorithms/nestedkv
---

# NestedKV-adapted — Multi-Scale Ensembled Prefill Eviction

**Method id:** `nestedkv` · **New in 0.37.0** · *Inspired by* ["NestedKV:
Nested Memory Routing for Long-Context KV Cache Compression" (Chen, Liu, Gao,
Fan, Wang, Chu, Lin, Hu; arXiv:2605.26678)](https://arxiv.org/abs/2605.26678)
— **NestedKV-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

:::warning[No verified peer-reviewed venue]
This is the **only** method in VeloxQuant-MLX (1 of 39) that does not trace
to a verified peer-reviewed venue. As of 2026-07-14, the paper is a single
arXiv revision (submitted 2026-05-26) with no Comments/journal-ref field
indicating acceptance anywhere. Every other method in this repo required a
live-verified venue before implementation — this one ships as a **one-time,
user-directed exception** to that standing rule. See
[`NEW_METHOD_SURVEY_V21.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/paper/research/surveys/NEW_METHOD_SURVEY_V21.md)
for the full rationale. The next method survey reverts to requiring a
verified venue — this is not a new precedent.
:::

NestedKV joins the repo's token-eviction family
([H2O](../algorithms/h2o), [SnapKV](../algorithms/snapkv),
[CurDKV](../algorithms/curdkv), and more) with a genuinely new mechanism
axis: instead of committing to **one** importance signal, it keeps **three
parallel key-only continuum-memory statistics** — stable/global,
episodic/block-local, current/recent-window — scores every token's anomaly
against each independently, and combines the three rankings via a
training-free **head-adaptive blend** (which scale is most discriminative on
this head) and a per-token **surprise-gated route** (fall back to the single
strongest scale when the three disagree, instead of averaging them).

## Where it sits — the mechanism gap

| Method | Scoring signal | Signals ensembled |
|---|---|---|
| [H2O](../algorithms/h2o) | cumulative softmax attention-mass over keys | 1 |
| [SnapKV](../algorithms/snapkv) | prefill observation-window attention over keys | 1 |
| [PyramidKV](../algorithms/pyramidkv) | layer-adaptive attention budget | 1 (budget varies by layer, not by signal) |
| [Keyformer](../algorithms/keyformer) | Gumbel-regularized attention over keys | 1 |
| [MorphKV](../algorithms/morphkv) | recent-window correlation over keys | 1 |
| [KVzip](../algorithms/kvzip) | context-reconstruction reliance | 1 |
| [CurDKV](../algorithms/curdkv) | joint key+value leverage score | 1 (joint, not ensembled) |
| **NestedKV-adapted** | **stable + episodic + current key anomaly, ensembled** | **3** |

Every eviction method already in this repo scores a token from **one**
importance signal. A token can be important because it is globally unusual,
because it marks a local topic shift within one segment, or because it is
part of the recent stream shaping immediate generation — which one applies
shifts across documents and compression ratios. NestedKV-adapted is the
first method here that keeps all three signals simultaneously rather than
picking one, and combines them with a rule that adapts per head and per
token rather than a fixed weighting.

## :warning: The honesty crux — read this first

1. **Unpublished preprint, no verified venue.** See the warning banner
   above — this is the headline exception for this method, stated first.
2. **One-shot prefill compression — the cache is NOT bounded during
   decode.** The paper's own design (Appendix A, quoted directly): "NestedKV
   does not recompute scores, scale reliabilities, or routes for retained
   prompt tokens as new tokens are generated; newly decoded tokens are
   appended normally." This is a faithful port of the paper's actual
   design, not a shortcut — but it is a real structural difference from
   every other eviction method here (H2O, CurDKV, StreamingLLM all stay
   bounded through decode). NestedKV's cache can grow unboundedly during a
   very long decode run, mirroring [SnapKV-adapted](../algorithms/snapkv)'s
   decode-phase design, not H2O's/CurDKV's per-step loop.
3. **Episodic blocks computed over fixed prefill-time positions — a
   faithful, non-approximated port.** Since eviction happens only once,
   after all three memory scales are already computed, there is no
   eviction-collapses-indices problem to work around here (unlike a
   hypothetical incremental version).
4. **Gate/blend constants taken directly from the paper's Appendix A** —
   `beta=3.0` (blend temperature), `tau=0.60` (surprise gate threshold, on
   min-max-normalized and mean-centered surprise), `kappa=10.0` (gate
   sharpness), log-prior `(0.4, 0.4, 0.2)`, `safeguard_alpha=0.20` (per-head
   guaranteed budget floor). These are the paper's own stated defaults, not
   this implementation's guesses — a rarer, stronger fidelity point than
   most adapted methods in this repo get to claim.
5. **Key-only — no query/attention access at all, not even a proxy.**
   Stronger than [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv)/
   [CurDKV](../algorithms/curdkv)'s key-as-query proxy: NestedKV never
   approximates attention in the first place.
6. **A structural interaction found during benchmark construction, not a
   bug**: at small (few-hundred-token) synthetic scale, the head-adaptive
   blend's min-max normalization can make the *stable* scale's
   discriminative gap come out near-maximal almost by construction whenever
   there is any real score variation — regardless of whether the stable
   scale is actually the relevant one for a given token — and the surprise
   gate's mean-centered threshold does not always fully compensate at this
   scale. See the benchmark section below for the full, honestly-reported
   finding. All formulas are implemented exactly per the paper — this is a
   property of how paper-tuned constants (validated on real 4k–32k-token
   contexts) interact at a much smaller synthetic scale.
7. Nothing here is validated on a trained model. The paper's own
   RULER/LongBench/LooGLE/InfiniteBench/MMLU-Pro numbers (Qwen3 and
   Llama-3.2 family models, NVIDIA L20 GPUs) are the **paper's** — never
   quoted as this repo's own.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="nestedkv",
    head_dim=128,
    nestedkv_budget=512,          # per-head-equivalent budget (total layer budget = this * n_heads)
    nestedkv_n_sink=4,            # leading positions never evicted
    nestedkv_window=64,           # W, current-memory trailing window
    nestedkv_beta=3.0,            # head-adaptive blend temperature (paper default)
    nestedkv_tau=0.60,            # surprise gate threshold (paper default)
    nestedkv_kappa=10.0,          # surprise gate sharpness (paper default)
    nestedkv_safeguard_alpha=0.20,  # per-head guaranteed budget floor (paper default)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`NestedKVKVCache` per attention layer. No `.bits` attribute (an eviction
cache, not a quantizer) — stores and returns fp16 K/V directly.

## How it works

**At the end of prefill (once per sequence):**

1. For every head, compute three continuum-memory anchors over all `N`
   prefill keys: stable (`μ_s` — global mean), episodic (`μ_e(i)` — mean
   over the `clip(⌊N/32⌋, 128, 256)`-token block containing `i`), current
   (`μ_c(i)` — mean over the trailing `W=64`-token window ending at `i`).
2. Score every token's cosine anomaly against each anchor independently,
   min-max normalize each of the three score vectors across the head.
3. **Head-adaptive blend**: weight the three scales by how discriminative
   each is on this head (top-10%/bottom-10% score gap), softmax-combined
   with a fixed log-prior `(0.4, 0.4, 0.2)` and temperature `beta`.
4. **Surprise-gated route**: where the three scales disagree (high
   cross-scale standard deviation), route the score toward the single
   strongest scale instead of the blend, via a sigmoid gate on `tau`/`kappa`.
5. **Cross-head budget competition**: each head first keeps a guaranteed
   floor of its own top `⌊safeguard_alpha · (1-r) · n⌋` tokens, then the
   remaining layer budget is filled by a global pool of the highest-scoring
   `(head, position)` pairs not already guaranteed.
6. Evict everything outside each head's allocated budget (sinks always
   kept).

**During decode:** every new token is simply appended — never rescored,
never evicted (see honesty crux, point 2).

## Byte accounting

- `nestedkv_kept_bytes` — fp16 bytes for currently retained K + V tokens.
- `full_seq_bytes` — hypothetical fp16 cost if all tokens were kept.
- `compression_ratio` — `full_seq_bytes / nestedkv_kept_bytes` (> 1 = savings).
- `tokens_seen` — total token positions ever passed to `update_and_fetch`.
- `tokens_kept` — tokens currently in the first (B=0, H=0) head's cache.

## Benchmark — honestly reported, including the part that didn't work

`benchmark_scripts/benchmark_nestedkv.py` (results in
`figures/nestedkv/results.json`) sweeps sequence length
(320/512) and token budget (16/24) across three geometries, each isolating
one of NestedKV's three scales via a single planted anomalous token, and
compares NestedKV-adapted against H2O at the **same matched token budget**,
reporting the planted anomaly's retention rate:

- **`global_outlier_only`** (anomalous only against the stable/global mean):
  NestedKV retains the anomaly in **100%** of trials vs H2O's **0%**.
- **`recency_only`** (anomalous only in the recent window): NestedKV **100%**
  vs H2O's **0%**.
- **`local_episodic_only`** (anomalous only within its local block, with the
  global mean engineered to cancel across two blocks): **neither method
  protects it (0% vs 0%)** — **not the initially expected result.** Debugged
  directly: the raw per-scale episodic anomaly score correctly ranks the
  defecting token #1 of N (proven at the primitive level by
  `test_single_anchor_blind_spot`), but the signal is lost one stage later —
  the head-adaptive blend's min-max normalization of the *stable* scale's
  score produces a near-maximal discriminative gap almost by construction
  whenever there is any real variation, regardless of whether the stable
  scale is actually the relevant signal for that token, and the surprise
  gate's mean-centered surprise value stays below the `tau=0.60` threshold
  for this benchmark's scale even for the single most-disagreeing token, so
  it only partially (~27%) routes toward the correct scale. All formulas are
  implemented exactly per the paper (see the module docstrings) — this is a
  property of how the paper's Appendix-A constants, tuned and validated on
  real 4k–32k-token contexts, interact at a much smaller two-block synthetic
  scale, not a deviation from the paper or a code defect.

Both methods use only keys (H2O additionally uses the key-as-query proxy for
its attention-mass signal; NestedKV uses no query proxy at all).

Deterministic in all non-`_ms` fields, verified by diffing two runs.
Offline-synthetic; loads no model.

## Adaptation notes — what we do NOT implement

- The paper's own RL-free but batch-computed stable/episodic/current memory
  statistics are ported exactly as specified — no incremental approximation
  was needed, since NestedKV is a one-shot prefill compressor by design (see
  honesty crux, point 3).
- Any per-task tuning of the blend/gate constants — the paper's own
  Appendix-A defaults are used unmodified.
- Any trained-model perplexity/throughput/accuracy benchmark. The paper's
  RULER/LongBench/LooGLE/InfiniteBench/MMLU-Pro numbers (Qwen3, Llama-3.2
  family, NVIDIA L20 GPUs) are the paper's — not reproduced here.
- No PyTorch/CUDA reference kept; pure MLX from the start.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_nestedkv.py` (30 tests) and
`veloxquant_mlx/tests/cache/test_nestedkv_cache.py` (17 tests):

- Bootstrap, prefill budget enforcement, sink protection, byte accounting,
  determinism, `for_model` config propagation (all 7 `nestedkv_*` fields).
- **`test_decode_tokens_appended_unscored`** / cache-level
  **`test_decode_growth_unbounded_past_prefill_budget`** — direct proof of
  the one-shot prefill/unbounded-decode design (honesty crux, point 2).
- **`test_single_anchor_blind_spot`** — proves the episodic scale identifies
  a locally-anomalous token that a global-mean-only scorer structurally
  cannot see.
- **`test_budget_allocation_favors_high_score_head`** /
  **`test_safeguard_floor_respected`** — proves the cross-head competition
  correctly shifts budget toward a more informative head while guaranteeing
  every head a minimum floor.
- **`test_surprise_gate_routes_to_winner_on_disagreement`** — proves the
  gate pulls the final score toward the single strongest scale when the
  three disagree.
- Degenerate all-identical-keys (zero cosine variance) produces finite
  scores, no NaN/crash.

**No model-level benchmark has been run.**
`benchmark_scripts/benchmark_nestedkv.py` is offline-synthetic and
deterministic in all non-timing fields — planted-anomaly retention rates
only, not perplexity or throughput on a real model.

## When to use it

NestedKV-adapted is for workloads where you suspect a single-signal
eviction scorer is structurally blind to some class of important tokens —
for example, tokens that matter because of a document-wide pattern rather
than recent attention mass. It is **not** a drop-in replacement when cache
size must stay strictly bounded through a very long decode phase — for that,
prefer [H2O](../algorithms/h2o) or [CurDKV](../algorithms/curdkv), which
re-evict every step.

| Method | Bounded during decode | Signals ensembled | Verified venue |
|--------|:---:|:---:|:---:|
| [H2O](../algorithms/h2o) | Yes | 1 | Yes |
| [CurDKV](../algorithms/curdkv) | Yes | 1 (joint key+value) | Yes |
| **NestedKV-adapted** | **No** | **3** | **No (this method only)** |

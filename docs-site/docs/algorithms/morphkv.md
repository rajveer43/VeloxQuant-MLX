# MorphKV — Recent-Window Correlation Retention

**Method id:** `morphkv` · **New in 0.33.0** · *Inspired by* ["Dialogue Without
Limits: Constant-Sized KV Caches for Extended Responses in LLMs" (Ghadia et al.,
ICML 2025, arXiv:2503.00979)](https://arxiv.org/abs/2503.00979) —
**MorphKV-adapted (VeloxQuant-MLX implementation)**. The estimator is
deliberately changed from the paper's (see [Adaptation notes](#adaptation-notes)).

The paper's contribution is a **retention rule**: keep a constant-size cache by
ranking stored tokens against the attention pattern of a *sliding window of
recent tokens*, eliminating the **"early-token bias"** of cumulative scoring —
where tokens that were heavy hitters early dominate the keep set and crowd out
what the model is *currently* attending to.

## Where it sits — the proxy-attention scorer family

MorphKV joins the repo's largest eviction family. It shares the H2O/TOVA
scaffolding — proxy-attention over stored keys with protected-sink top-budget
eviction — but introduces a **new axis**: the ranking signal is correlation with
a *window* of recent tokens, not cumulative history (H2O) or a single latest
query (TOVA).

| Scorer class | Signal | Methods |
|---|---|---|
| Attention / proxy | softmax weights (true or key-as-query proxy) | [SnapKV](../algorithms/snapkv) · [H2O](../algorithms/h2o) · [TOVA](../algorithms/tova) · [PyramidKV](../algorithms/pyramidkv) · [SqueezeAttention](../algorithms/squeeze) · [ChunkKV](../algorithms/chunkkv) · [CaM](../algorithms/cam) · [Keyformer](../algorithms/keyformer) · **MorphKV** |
| Structural | position only (sinks, recency) | [StreamingLLM](../algorithms/streaming_llm) · sink · sliding-window |
| Intrinsic | the stored key itself (L2 norm) | [L2Norm](../algorithms/knorm) |
| Projection | key's projection onto a frozen per-head direction | [Q-Filters](../algorithms/qfilters) |

The distinguishing axis across the proxy family:

| Method | Ranks a stored token against… |
|---|---|
| [H2O](../algorithms/h2o) | **cumulative** attention over all history (inertial; early-token bias) |
| [TOVA](../algorithms/tova) | the **single most recent** query (memoryless) |
| **MorphKV** | a **sliding window** of the most recent tokens (recent-context aware) |

### `morphkv_window = 1` **is** TOVA-adapted

Setting the window to one reduces the recent-relevance signal to the newest
key's attention over the keep set — the [TOVA](../algorithms/tova) latest-token
ranking, bit-for-bit. That is the honest reduction: a dedicated test asserts the
`window=1` kept set equals TOVA's over an identical stream. We do **not** claim
any H2O collapse — MorphKV recomputes from the live window and never becomes
H2O's cumulative-forever rule, so no such equivalence is asserted.

## :warning: The honesty crux — read this first

1. **Proxy query.** Like [H2O](../algorithms/h2o)/[TOVA](../algorithms/tova)/[SnapKV](../algorithms/snapkv)/[Keyformer](../algorithms/keyformer),
   a cache never sees the true query vector, so incoming **keys** are used as
   proxy queries to estimate the attention each stored key receives. The paper
   uses the model's real attention patterns. Documented substitution, not the
   paper's math.
2. **Constant-size, recomputed — not accumulated.** We store no cumulative score
   array. Each step, retention is recomputed from the live keep set and a window
   of the last `morphkv_window` key rows (the trailing recent tokens, themselves
   protected). This *is* the mechanism: a fixed budget refreshed against recent
   context, not a growing accumulator.
3. **Not validated on a trained model.** The rule's benefit is measured only
   under constructed "topic-shift" geometry, with a stable control where it has
   nothing to re-target. The paper's headline accuracy/memory numbers are the
   **paper's, on trained models** — never reproduced or claimed here.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="morphkv",
    head_dim=128,
    morphkv_budget=512,   # max tokens kept (incl. sinks)
    morphkv_n_sink=4,     # leading positions never evicted
    morphkv_window=8,     # trailing recent-attention window; 1 = latest-token (TOVA)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`MorphKVKVCache` per attention layer. MorphKV is fully deterministic (no RNG).

## How it works

Per incoming token (prefill and decode alike):

1. Append the new token to the cache.
2. If over `morphkv_budget`: take the last `morphkv_window` stored keys as the
   **recent window**, and for each stored token compute its mean proxy-attention
   mass under that window (`_recent_relevance` — how much the recent context
   attends to it).
3. Force the leading `morphkv_n_sink` sinks and the trailing `morphkv_window`
   recent tokens to survive, and evict the non-protected token with the
   **lowest** recent-relevance. Constant-size: `n_kept <= budget` after every
   token.

No cumulative score is carried across steps — the ranking is recomputed fresh
each step from the live keep set and recent window. With `morphkv_window=1` the
recent window is the single newest key, and this is exactly the
[TOVA](../algorithms/tova) latest-token argmin.

Byte accounting mirrors H2O's — `morphkv_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`. The recent window is a view
into the stored keys, not extra payload, so only K + V are counted.

## Adaptation notes

**What we do NOT implement:**
- The model's real attention logits — replaced by the key-as-query proxy
  (crux 1), same approximation as H2O/TOVA/Keyformer-adapted.
- The paper's exact refresh cadence / per-head adaptive budgets.
- RoPE position-ID remapping after eviction (same as every eviction method here).
- Per-head windows / budgets (uniform across heads).

**Design choices:**
- The trailing `morphkv_window` tokens are protected from eviction — they are
  the recency context that drives the ranking.
- Leading `morphkv_n_sink` tokens are protected as attention sinks.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_morphkv.py` (19 tests) and
`veloxquant_mlx/tests/cache/test_morphkv_cache.py` (13 tests):

- **`window = 1` collapses onto TOVA-adapted** — the kept set equals TOVA's,
  bit-for-bit, over an identical stream. A wider window is shown capable of
  retaining a different set, so the window axis is not vacuous.
- Constant-size budget is never exceeded, token-by-token or in a prefill block,
  across batch/head shapes.
- Sinks and the trailing recent window survive heavy eviction; bad
  budget/window/sink bounds raise at build time.
- The full run is reproducible (no RNG).
- **Topic-shift mechanism:** with an early block on axis A and a late block on a
  distinct axis B that the recent window reads, MorphKV retains the axis-B region
  at a **materially higher rate than a cumulative H2O-style baseline** — a
  statistical mechanism claim, with a null "stable" control where it shows no
  advantage.

The offline harness in `benchmark_scripts/benchmark_morphkv.py` (results in
`figures/morphkv/results.json`) sweeps sequence length
(256/512) and budget (32/64) across `window ∈ {1, 8, 32}`, an H2O cumulative
cross-check, and random eviction, under two data regimes:

- **`topic_shift` geometry:** cumulative H2O scoring retains **~0%** of the
  recent-relevant (axis-B) region — fully captured by the stale early axis-A
  heavy hitters, exactly the "early-token bias" the paper describes. The recent
  signal is deliberately made weak and per-token noisy, so a single latest token
  (`window=1` == TOVA) only partly re-targets (~0.15–0.37), while **averaging
  over a wider window** (`window=8/32`) cancels the noise and reliably surfaces
  axis B (~0.39–0.74). This recent-relevant retention rate is the mechanism's
  clean, direct observable, and it is where the *window* earns its keep over the
  latest-token reference.
- **`stable` geometry** (all traffic on one axis from token 0): every arm keeps
  the same region (retention 1.0), so there is nothing to re-target and MorphKV
  is neutral. Reporting this control is the point — the rule is not a free win.

The downstream probe-attention **perturbation** is a noisier, regime-dependent
secondary effect that does **not** uniformly improve; it is reported as-is
rather than cherry-picked. **No model-level benchmark has been run** — these are
offline-synthetic retention-rate, output-perturbation and byte-accounting
numbers, not perplexity or throughput on a real model.

## When to use it

MorphKV is for workloads whose focus **shifts** over a long generation —
extended dialogues, multi-turn or multi-topic responses — where what matters now
is not what mattered at the start. Where [H2O](../algorithms/h2o)'s cumulative
scoring would cling to stale early heavy hitters, MorphKV re-targets the cache
toward what the recent window is reading. If the focus is *stable* (heavy hitters
stay heavy), plain [H2O](../algorithms/h2o) is simpler; if you want purely the
most-recent query to decide, set `morphkv_window=1` and you are running
[TOVA](../algorithms/tova).

| Method | Score | Recent-context aware | Path-independent |
|--------|-------|----------------------|------------------|
| [H2O](../algorithms/h2o) | cumulative proxy-attention mass | no (inertial) | no |
| [TOVA](../algorithms/tova) | single latest query's attention | latest only | no |
| **MorphKV** | recent-**window** proxy-attention correlation | **yes (window)** | no |

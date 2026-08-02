# KVzip — Context-Reconstruction Reliance Retention

**Method id:** `kvzip` · **New in 0.34.0** · *Inspired by* ["KVzip:
Query-Agnostic KV Cache Compression with Context Reconstruction" (Kim et al.,
NeurIPS 2025 Oral, arXiv:2505.23416)](https://arxiv.org/abs/2505.23416) —
**KVzip-adapted (VeloxQuant-MLX implementation)**. The estimator is deliberately
changed from the paper's (see [Adaptation notes](#adaptation-notes)).

The paper's contribution is a **retention rule**: keep a constant-size cache by
ranking stored tokens according to how much the model relies on them to
**reconstruct its own context** — a *query-agnostic* importance profile computed
once and reused across all future queries — then evict the least-relied-upon
pairs.

## Where it sits — the proxy-attention scorer family

KVzip joins the repo's largest eviction family. It shares the H2O/TOVA/MorphKV
scaffolding — proxy-attention over stored keys with protected-sink top-budget
eviction — but introduces a **new axis**: the ranking signal is
*reconstruction reliance* (attention received from a fixed reconstruction probe),
not attention received from a query.

| Scorer class | Signal | Methods |
|---|---|---|
| Attention / proxy | softmax weights (true or key-as-query proxy) | [SnapKV](../algorithms/snapkv) · [H2O](../algorithms/h2o) · [TOVA](../algorithms/tova) · [PyramidKV](../algorithms/pyramidkv) · [SqueezeAttention](../algorithms/squeeze) · [ChunkKV](../algorithms/chunkkv) · [CaM](../algorithms/cam) · [Keyformer](../algorithms/keyformer) · [MorphKV](../algorithms/morphkv) · **KVzip** |
| Structural | position only (sinks, recency) | [StreamingLLM](../algorithms/streaming_llm) · sink · sliding-window |
| Intrinsic | the stored key itself (L2 norm) | [L2Norm](../algorithms/knorm) |
| Projection | key's projection onto a frozen per-head direction | [Q-Filters](../algorithms/qfilters) |

The distinguishing axis across the proxy family — **what** each method ranks a
stored token against:

| Method | Ranks a stored token against… |
|---|---|
| [H2O](../algorithms/h2o) | **cumulative** attention over all history (inertial; early-token bias) |
| [TOVA](../algorithms/tova) | the **single most recent** query (memoryless) |
| [MorphKV](../algorithms/morphkv) | a **sliding window** of the most recent tokens (recent-context aware) |
| **KVzip** | a fixed **reconstruction probe** — how much rebuilding the context relies on it (query-agnostic) |

Every other proxy scorer ranks by *attention received from a query* (cumulative,
latest, or windowed). KVzip is the first to rank by *reconstruction reliance* —
attention received from a reconstruction probe that is not a downstream query.

### `kvzip_probe = "latest"` **is** TOVA-adapted

Setting the probe to `"latest"` makes the reconstruction probe the single
most-recent key, so the reconstruction importance reduces to that key's attention
over the keep set — the [TOVA](../algorithms/tova) latest-token ranking,
bit-for-bit. That is the honest reduction: a dedicated test asserts the
`probe="latest"` kept set equals TOVA's over an identical stream. We do **not**
claim any H2O collapse — KVzip recomputes from the live keep set and never becomes
H2O's cumulative-forever rule, so no such equivalence is asserted.

## :warning: The honesty crux — read this first

1. **Proxy reconstruction.** A cache never runs the real model to reconstruct
   text. Like [H2O](../algorithms/h2o)/[TOVA](../algorithms/tova)/[MorphKV](../algorithms/morphkv),
   the stored **keys** are used as proxy reconstruction queries to estimate the
   attention each stored key receives. The paper uses the model's real
   reconstruction forward passes. Documented substitution, not the paper's math.
2. **Query-agnostic, recomputed — not accumulated.** We store no cumulative score
   array. Each step, reconstruction importance is recomputed from the live keep
   set against the probe. Query-agnostic in the paper's sense (the probe is not a
   downstream query); constant, not a growing accumulator.
3. **Not validated on a trained model.** The rule's benefit is measured only
   under constructed "reconstruction-shift" geometry, with a flat control where it
   has nothing to re-target. The paper's headline numbers (3–4× reduction, ~2×
   decode latency, negligible loss up to 170K tokens on LLaMA3.1/Qwen2.5/Gemma3)
   are the **paper's, on trained models** — never reproduced or claimed here.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="kvzip",
    head_dim=128,
    kvzip_budget=512,  # max tokens kept (incl. sinks)
    kvzip_n_sink=4,  # leading positions never evicted
    kvzip_probe="context",  # reconstruction probe; "latest" = latest-token (TOVA)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`KVzipKVCache` per attention layer. KVzip is fully deterministic (no RNG).

## How it works

Per incoming token (prefill and decode alike):

1. Append the new token to the cache.
2. If over `kvzip_budget`: compute each stored token's **reconstruction reliance**
   (`_reconstruction_importance`) — the **maximum** proxy-attention it receives
   across the reconstruction-probe rows. With `kvzip_probe="context"` the probe is
   the full kept set (reconstruct the context from itself); with
   `kvzip_probe="latest"` the probe is the single most-recent key.
3. Force the leading `kvzip_n_sink` sinks to survive, and evict the non-sink token
   with the **lowest** reconstruction reliance. Constant-size: `n_kept <= budget`
   after every token.

No cumulative score is carried across steps — the ranking is recomputed fresh
each step from the live keep set against the probe. With `kvzip_probe="latest"`
the probe is the single newest key, and this is exactly the
[TOVA](../algorithms/tova) latest-token argmin.

Byte accounting mirrors H2O's — `kvzip_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`. The reconstruction probe reuses
the stored keys, not extra payload, so only K + V are counted.

## Adaptation notes

**What we do NOT implement:**
- The model's real context-reconstruction forward passes — replaced by the
  key-as-reconstruction-probe proxy (crux 1).
- Head-level context-independent scoring / DuoAttention-style head compression.
- RoPE position-ID remapping after eviction (same as every eviction method here).
- Per-head probes / budgets (uniform across heads).

**Design choices:**
- Importance is the **max** proxy-attention over the probe rows (a token critical
  to *any* reconstruction position survives), following the paper's max-over-probe.
- Leading `kvzip_n_sink` tokens are protected as attention sinks. Unlike MorphKV,
  no trailing window is force-protected — a token survives only if the
  reconstruction probe relies on it.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_kvzip.py` (19 tests) and
`veloxquant_mlx/tests/cache/test_kvzip_cache.py` (13 tests):

- **`probe = "latest"` collapses onto TOVA-adapted** — the kept set equals TOVA's,
  bit-for-bit, over an identical stream. The default `context` probe is shown
  capable of retaining a different set, so the reconstruction axis is not vacuous.
- Constant-size budget is never exceeded, token-by-token or in a prefill block,
  across batch/head shapes.
- Sinks survive heavy eviction; bad budget/sink bounds and an invalid probe raise
  at build time.
- The full run is reproducible (no RNG).
- **Reconstruction-geometry mechanism:** with an early block on axis A and a
  mutually-reinforcing reconstruction-critical cluster on a distinct axis B, KVzip
  retains the axis-B region at a **materially higher rate than a cumulative
  H2O-style baseline** — a statistical mechanism claim, with a null "flat" control
  where it shows no advantage.

The offline harness in `benchmark_scripts/benchmark_kvzip.py` (results in
`figures/kvzip/results.json`) sweeps sequence length
(256/512) and budget (32/64) across `probe ∈ {latest, context}`, an H2O cumulative
cross-check, and random eviction, under two data regimes:

- **`reconstruction_shift` geometry:** cumulative H2O scoring retains **~0.017**
  of the reconstruction-critical (axis-B) region — fully captured by the stale
  early axis-A heavy hitters, exactly the early-token bias. The axis-B signal is
  deliberately made weak and per-token noisy, so a single latest token
  (`probe="latest"` == TOVA) only partly re-targets (**~0.248**), while the
  **full-context reconstruction probe** aggregates the cluster's mutual
  reinforcement and reliably surfaces axis B (**~0.609**). This
  reconstruction-critical retention rate is the mechanism's clean, direct
  observable, and it is where the *context* probe earns its keep over the
  latest-token reference.
- **`flat` geometry** (all traffic on one axis from token 0): every arm keeps the
  same region (retention 1.0), so there is nothing to re-target and KVzip is
  neutral. Reporting this control is the point — the rule is not a free win.

The downstream probe-attention **perturbation** is a noisier, regime-dependent
secondary effect that does **not** uniformly improve; it is reported as-is rather
than cherry-picked. **No model-level benchmark has been run** — these are
offline-synthetic retention-rate, output-perturbation and byte-accounting
numbers, not perplexity or throughput on a real model.

## When to use it

KVzip is for workloads where a compressed cache must serve **diverse, unknown
future queries** — its importance profile is query-agnostic, so the same cache
answers many different downstream questions about a long context. Where
[H2O](../algorithms/h2o)'s cumulative scoring clings to stale early heavy hitters,
KVzip retains what the model relies on to reconstruct the whole context. If you
want purely the most-recent query to decide, set `kvzip_probe="latest"` and you
are running [TOVA](../algorithms/tova).

| Method | Score | Query-agnostic | Path-independent |
|--------|-------|----------------|------------------|
| [H2O](../algorithms/h2o) | cumulative proxy-attention mass | no (per-query) | no |
| [TOVA](../algorithms/tova) | single latest query's attention | no (per-query) | no |
| [MorphKV](../algorithms/morphkv) | recent-window proxy-attention correlation | no (per-query) | no |
| **KVzip** | reconstruction-probe reliance (max over probe) | **yes (reconstruction probe)** | no |

# L2Norm — Intrinsic Key-Norm Eviction

**Method id:** `knorm` · **New in 0.29.0** · *Inspired by* ["A Simple and
Effective L2 Norm-Based Strategy for KV Cache Compression" (arXiv:2406.11430,
EMNLP 2024)](https://arxiv.org/abs/2406.11430) — **L2Norm-adapted
(VeloxQuant-MLX implementation)**, faithful to the scoring rule; the
correlation it relies on is the paper's empirical claim about trained models
(see [Adaptation notes](#adaptation-notes)).

The paper reports a consistent — and counterintuitive — correlation in
trained decoder LMs: **a low L2 norm of a key embedding usually leads to a
high attention score during decoding**. A KV pair's influence is therefore
largely determined by the key itself, *before it is ever queried*. Eviction
follows directly: rank cached tokens by key norm, **keep the lowest-norm
ones**, evict the highest-norm ones. No attention scores, no proxies, no
training, no calibration.

## A third scorer class

Every eviction method in the repo scores tokens one of two ways — this adds
a third:

| Scorer class | Signal | Methods |
|---|---|---|
| Attention / proxy | softmax weights (true or key-as-query proxy) | [SnapKV](../algorithms/snapkv) · [H2O](../algorithms/h2o) · [TOVA](../algorithms/tova) · [PyramidKV](../algorithms/pyramidkv) · [SqueezeAttention](../algorithms/squeeze) · [ChunkKV](../algorithms/chunkkv) · [CaM](../algorithms/cam) |
| Structural | position only (sinks, recency) | [StreamingLLM](../algorithms/streaming_llm) · sink · sliding-window |
| **Intrinsic** | **the stored key itself (L2 norm)** | **L2Norm** |
| Projection | key's projection onto a frozen per-head direction | [Q-Filters](../algorithms/qfilters) |

Two consequences fall out of the score being intrinsic (computed once at
insertion, never updated):

1. **No per-token loop.** Eviction is one protected top-k per incoming
   block — measured **~100–800× faster** than H2O-adapted's per-token
   softmax-over-cache update at prefill (0.3 ms vs 240 ms at S=1024 in the
   committed benchmark).
2. **Path independence** (`knorm_recent=0`): evicting the current
   worst-scoring non-sink token whenever over budget is the classic "keep k
   best with a heap" algorithm, so prefill-in-one-block and token-by-token
   decode produce **bit-for-bit identical caches**. No accumulating-score
   method (H2O, TOVA) has this property; it is pinned by test.

Note the **sign inversion** vs [ChunkKV](../algorithms/chunkkv)'s `key_norm`
scoring option and [ZipCache](../algorithms/zipcache)'s saliency proxy, which
treat *high* norm as important. The inversion (low norm attracts attention)
is exactly the paper's empirical content.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="knorm",
    head_dim=128,
    knorm_budget=512,   # max tokens kept (incl. sinks)
    knorm_n_sink=4,     # leading positions never evicted
    knorm_recent=0,     # trailing protected window (0 = paper-faithful)
    knorm_keep="low",   # "low" = paper finding; "high" = inverted ablation
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`L2NormKVCache` per attention layer.

## How it works

Per `update_and_fetch` block (prefill or decode, same path):

1. Compute the L2 norm of each incoming key row — once; norms are never
   recomputed or updated.
2. Concatenate onto the kept set.
3. If over `knorm_budget`: keep the `budget` lowest-norm positions
   (`knorm_keep="high"` inverts the ranking) in one top-k, with sinks (first
   `knorm_n_sink` positions) and the optional trailing `knorm_recent` window
   forced to survive. Kept tokens preserve original temporal order.

Byte accounting mirrors H2O's: `knorm_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`.

## Adaptation notes

**Fidelity to the paper:** the scoring rule is implemented exactly — rank by
key L2 norm, keep lowest. The clean part of this adaptation is that the
paper's actual signal is **fully observable at the cache level** (unlike
SnapKV/H2O/TOVA, whose true signal — real query vectors — must be proxied
with the incoming key). Fewer disclaimers than any eviction method in this
repo.

**The one big caveat, stated plainly:** the low-norm ⇒ high-attention
correlation is an *empirical property of trained models*. It is not a
mathematical identity, and it is **not reproducible on isotropic synthetic
data** — on plain Gaussian keys the direction actually reverses (softmax
favors high-norm keys), and the committed benchmark shows keep-low
*underperforming random eviction* in that control. The method's value rests
entirely on trained-LM key geometry, which we attribute to the paper and do
not claim to have independently verified.

**What we do NOT implement:**
- Per-layer/per-task compression-rate tuning from the paper's evaluation
  sweeps — one uniform `knorm_budget` (per-layer overrides remain possible
  through the standard config mechanics).
- RoPE position-ID remapping after eviction (same as every eviction method
  here).
- Per-head budgets (uniform across heads, same as H2O/TOVA/CaM).

**Extensions beyond the paper (both off by default):**
- `knorm_recent` — protects the most recent tokens StreamingLLM-style.
  Enabling it breaks the path-independence property.
- `knorm_keep="high"` — the inverted scorer, shipped as a config value
  because it is exactly the ablation arm the benchmark needs.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_knorm.py` (10 tests) and
`veloxquant_mlx/tests/cache/test_knorm_cache.py` (14 tests):

- Over budget, the kept set equals the budget lowest-norm positions
  (verified against a manual numpy ranking), in original temporal order
- Sinks survive even with the highest norms; `recent` window protection;
  `n_sink + recent >= budget` raises at build time
- Norm immutability — a token's stored norm never changes across updates
- **Path independence: block vs token-by-token arrival produce bit-for-bit
  identical kept keys and values** (primitives and full cache wrapper)
- Mechanism test under paper-like geometry: with low-norm keys aligned to
  the probe-query cluster and high-norm keys anti-aligned (the correlation
  the paper reports, constructed explicitly), keep-low's attention output is
  strictly closer to the full-cache output than keep-high's
- Budget enforcement, byte accounting, determinism, `for_model` wiring

The offline harness in `benchmark_scripts/benchmark_knorm.py` (results in
`figures/knorm/results.json`) sweeps sequence length
(256–1024) and budget (64/128) across four arms — keep-low, keep-high,
random eviction, H2O-adapted — under two data regimes:

- **paper-like geometry** (low-norm keys aligned with the query cluster):
  keep-low wins every row — mean perturbation advantage **+0.17 vs random
  eviction and +0.21 vs keep-high**, and it also beats H2O-adapted on most
  rows at the same budget.
- **isotropic control** (plain Gaussian keys, no norm signal): the
  advantage doesn't just vanish — it **reverses** (keep-low is ~0.07 worse
  than random), because softmax favors high-norm keys on isotropic data.
  Reported in full; this is the regime the paper's trained-LM geometry does
  not resemble.
- **Speed:** the intrinsic score needs no per-token softmax-over-cache loop
  — 0.3–1.2 ms per prefill block vs H2O-adapted's 37–275 ms on the same
  inputs (M-series, offline harness).

**No model-level benchmark has been run.** These are offline-synthetic,
output-perturbation and byte-accounting numbers — not perplexity or
throughput on a real model, and they validate the machinery, not the
paper's correlation claim.

## When to use it

L2Norm is the lightest-weight importance-based eviction in the library: if
you want importance eviction (not just recency) with zero per-step scoring
cost and provably grouping-independent behavior, it is the natural first
choice — *provided you trust the paper's finding for your model family*. If
you'd rather pay per-step compute for a score that reacts to the actual
query stream, use [H2O](../algorithms/h2o) (cumulative) or
[TOVA](../algorithms/tova) (memoryless) instead.

| Method | Score | Per-step cost | Path-independent |
|--------|-------|---------------|------------------|
| H2O | cumulative proxy-attention mass | softmax over cache | no |
| TOVA | current-step proxy attention | softmax over cache | no |
| **L2Norm** | intrinsic key norm | none (norm at insertion) | **yes** (`recent=0`) |
| [Q-Filters](../algorithms/qfilters) | projection onto frozen key-SVD direction | none (after calibration) | no |

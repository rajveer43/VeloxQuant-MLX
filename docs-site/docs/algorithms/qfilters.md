# Q-Filters — Query-Agnostic Projection Eviction

**Method id:** `qfilters` · **New in 0.31.0** · *Inspired by* ["Q-Filters:
Leveraging QK Geometry for Efficient KV Cache Compression"
(arXiv:2503.02812)](https://arxiv.org/abs/2503.02812) — **Q-Filters-adapted
(VeloxQuant-MLX implementation)**, a *preprint* (no venue). The estimator is
deliberately changed from the paper's (see [Adaptation notes](#adaptation-notes)).

The paper's premise: for a trained attention head the (Query, Key) joint
distribution is anisotropic, so there is a single per-head direction — the
*Q-Filter* — onto which a key's projection predicts the attention that key
will receive. Ranking cached keys by that projection approximates
attention-based importance **without computing attention and without a query
at eviction time**.

## A fourth scorer class

Q-Filters adds a scorer class the repo otherwise lacks:

| Scorer class | Signal | Methods |
|---|---|---|
| Attention / proxy | softmax weights (true or key-as-query proxy) | [SnapKV](../algorithms/snapkv) · [H2O](../algorithms/h2o) · [TOVA](../algorithms/tova) · [PyramidKV](../algorithms/pyramidkv) · [SqueezeAttention](../algorithms/squeeze) · [ChunkKV](../algorithms/chunkkv) · [CaM](../algorithms/cam) |
| Structural | position only (sinks, recency) | [StreamingLLM](../algorithms/streaming_llm) · sink · sliding-window |
| Intrinsic | the stored key itself (L2 norm) | [L2Norm](../algorithms/knorm) |
| **Projection** | **key's projection onto a frozen per-head direction** | **Q-Filters** |

## :warning: The honesty crux — read this first

The paper estimates the filter **offline, from the SVD of a sample of query
vectors**. A cache-side library **never sees query vectors** — only the K/V
passed to `update_and_fetch`. So this adaptation substitutes a *different
estimator of the same head-geometry direction*: the **top singular vector of
the first `qfilters_calib_tokens` observed keys**, computed once and frozen.

This substitution has a concrete, measured consequence: **the key-SVD
recovers the dominant axis but not which end of it is important.** The sign
that would tell "high projection = attended" from "low projection = attended"
is exactly what a query disambiguates — and the cache has no query. In the
committed benchmark the key-SVD direction recovers the planted axis with
`filter_cosine ≈ 0.97`, yet whether `qfilters_sign=+1` or `-1` is the good
arm flips from row to row. **The `qfilters_sign` knob is therefore a genuine
ablation, not a cosmetic one**; nothing here is claimed equivalent to the
paper's query-derived filter.

## Path dependence (contrast with L2Norm)

Unlike [L2Norm](../algorithms/knorm), the kept set is **not** path-independent.
The filter is estimated from whichever chunk first crosses `calib_tokens`, so
prefill-in-one-block and token-by-token decode can freeze *different*
directions and diverge. There is deliberately **no prefill/decode bit-for-bit
equivalence guarantee** — the tests assert only the weaker, true property
that *given the same frozen filter*, scoring and eviction are order-invariant.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="qfilters",
    head_dim=128,
    qfilters_budget=512,        # max tokens kept (incl. sinks)
    qfilters_n_sink=4,          # leading positions never evicted
    qfilters_recent=0,          # trailing protected window (extension, off)
    qfilters_calib_tokens=128,  # tokens observed before the filter freezes
    qfilters_sign=1,            # +1 = paper direction; -1 = inverted ablation
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`QFiltersKVCache` per attention layer.

## How it works

Per `update_and_fetch` block:

1. Concatenate incoming K/V onto the kept set.
2. **Before `calib_tokens` keys have been observed:** keep everything (the
   filter does not exist yet — no eviction can happen).
3. **Once enough keys are seen:** estimate the filter as the top singular
   vector of the observed keys, freeze it, and score every stored token by
   `sign · (key · filter_dir)` (float32). Scores are computed once at
   insertion and never updated.
4. If over `qfilters_budget`: keep the `budget` **highest-scoring** positions
   in one top-k, with sinks (first `qfilters_n_sink`) and the optional
   trailing `qfilters_recent` window forced to survive. Kept tokens preserve
   original temporal order.

Byte accounting mirrors L2Norm's — `qfilters_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept` — and additionally counts
the frozen `filter_dir` in full (`head_dim × 4` bytes, float32, per head).

## Adaptation notes

**What we do NOT implement:**
- **Query-derived filter estimation** (the paper's actual mechanism) —
  replaced by the key-SVD substitute above. This is the crux, not a footnote.
- Any offline calibration corpus; the per-head filter is estimated online
  from observed traffic and frozen.
- RoPE position-ID remapping after eviction (same as every eviction method
  here).
- Per-head budgets (uniform across heads, same as H2O/TOVA/CaM/L2Norm).

**Extensions beyond the paper (off by default):**
- `qfilters_recent` — protects the most recent tokens StreamingLLM-style.
- `qfilters_sign=-1` — the inverted scorer, shipped as a config value because
  the key-SVD's sign ambiguity makes it a real ablation arm (see the crux).

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_qfilters.py` (12 tests) and
`veloxquant_mlx/tests/cache/test_qfilters_cache.py` (15 tests):

- `estimate_filter_dir` recovers a planted dominant direction (cosine > 0.99)
- Over budget, the kept set equals the budget highest-projection positions
  (verified against a manual numpy ranking against the *frozen* direction),
  in original temporal order
- Pre-calibration passthrough: below `calib_tokens` nothing is evicted even
  over budget
- Sinks and `recent` window survive; `n_sink + recent >= budget` and invalid
  `sign` raise at build time
- Frozen-filter determinism — a stored score never changes once the filter is
  set
- **Path dependence handled honestly:** prefill vs decode may differ; the
  test asserts both stay within budget and both freeze a valid unit-norm
  filter — *not* bit-for-bit equality
- Mechanism test under paper-like geometry: important tokens carry a large
  projection onto the dominant axis and align with the probe-query cluster;
  the *correct-sign* cache's attention output beats random eviction by a wide
  margin (the anisotropy is the paper's claim, constructed here explicitly)

The offline harness in `benchmark_scripts/benchmark_qfilters.py` (results in
`figures/qfilters/results.json`) sweeps sequence length
(256–1024) and budget (64/128) across sign±1, best-of-sign, KNorm-adapted,
H2O-adapted and random arms under two data regimes:

- **paper-like geometry:** the key-SVD direction recovers the planted axis
  with **mean `filter_cosine ≈ 0.97`**, and the best-sign Q-Filter beats
  random eviction by **mean perturbation +0.16** — but which raw sign arm is
  the good one flips from row to row, the direct evidence of the key-only
  estimator's sign ambiguity.
- **isotropic control** (plain Gaussian keys, no dominant importance axis):
  the raw single-sign arms hover at random; any small residual advantage in
  best-of-sign is the best-of-two selection bonus, not an importance signal.
  Reported in full — no fabricated advantage.

**No model-level benchmark has been run.** These are offline-synthetic,
output-perturbation and byte-accounting numbers — not perplexity or
throughput on a real model, and they validate the machinery under constructed
geometry, not the paper's anisotropy claim.

## When to use it

Q-Filters is the repo's projection-based eviction: a per-head direction gives
importance eviction with zero per-step scoring cost, like
[L2Norm](../algorithms/knorm), but keyed to head geometry rather than raw
norm. Two honest caveats before reaching for it: the direction is estimated
from **keys**, not the paper's queries (so its sign may need the
`qfilters_sign` ablation to land correctly for your model), and the kept set
is **path-dependent**. If you want a grouping-independent, sign-unambiguous
intrinsic scorer, prefer [L2Norm](../algorithms/knorm); if you want a score
that reacts to the actual query stream, use [H2O](../algorithms/h2o).

| Method | Score | Per-step cost | Path-independent |
|--------|-------|---------------|------------------|
| H2O | cumulative proxy-attention mass | softmax over cache | no |
| [L2Norm](../algorithms/knorm) | intrinsic key norm | none | yes (`recent=0`) |
| **Q-Filters** | projection onto frozen key-SVD direction | none (after calibration) | **no** |

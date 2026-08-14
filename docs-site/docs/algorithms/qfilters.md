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

## Two filter sources

Which filter you use decides what guarantees you get. **Calibrate if you can.**

| | **Calibrated (paper-faithful)** | **Key-SVD fallback** |
|---|---|---|
| Filter from | SVD of **query** activations, offline (paper Eq. 1) | SVD of the first `qfilters_calib_tokens` observed **keys** |
| Sign | correct by construction (Thm 3.3, `κʰ > 0`) | **ambiguous** — `qfilters_sign` is a real ablation |
| Warm-up | none; evicts from token 0 | budget exceeded until calibration completes |
| Path-independent | **yes** | no |
| Needs | one offline pass per model | nothing |

### Calibrated — the paper's actual mechanism

Paper §3.2 step 1: gather `Qʰ` activations, take their SVD (Eq. 1), and keep
the sign-fixed top right vector `v₁⁺ = sgn(1ᵀu₁)·v₁`. Under GQA the filters of
each query-head group are averaged onto their KV head.

```python
from veloxquant_mlx.quantizers.qfilters_calibration import (
    collect_query_activations, compute_qfilters, average_gqa_filters,
    QFiltersCalibration, save_qfilters,
)

acts = collect_query_activations(model, tokenizer, calibration_texts)  # per layer [H_q, N, D]
filters = [average_gqa_filters(compute_qfilters(a), n_kv_heads=8) for a in acts]
save_qfilters(
    QFiltersCalibration(filters, model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        n_samples=3000, dataset="pile-subset"),
    "qfilters_llama32_3b.npz",
)
```

Two details that are easy to get wrong and are implemented per the paper:

- **No mean-centering.** Observation 3.1 is about the query cloud's *drift*
  away from the origin. Centering subtracts exactly that signal and returns
  the top *variance* axis instead — a different direction. (`test_does_not_mean_center`
  pins this, and shows a centering estimator picking the wrong axis on the same data.)
- **Sign anchoring** on `sgn(1ᵀu₁)`, which is what makes `κʰ > 0` hold and so
  makes "higher projection = more attention" a valid ranking rule.

Cost is negligible and one-time: paper §4.2 uses 20 samples × 2048 tokens with
3k SVD samples per head; filters total `l × n_H × d_H` parameters (~36000×
smaller than Llama-3.2-1B's weights).

### Key-SVD fallback

With no artifact, the filter is estimated from observed keys and frozen. It
recovers the dominant *axis* but **not which end of it matters** — the sign is
precisely what a query disambiguates, and the cache has no query. Worse, the
fallback mean-centers, so a head whose geometry is pure drift (the paper's
`ε = −1` case) leaves it with no signal at all: measured cosine to the planted
direction drops to `< 0.3` (`test_key_svd_loses_a_pure_drift_direction`). On
this path `qfilters_sign` is a **genuine ablation knob**, not a cosmetic one,
and nothing is claimed equivalent to the paper's filter.

## Path dependence (fallback only)

On the fallback path the filter is estimated from whichever chunk first
crosses `calib_tokens`, so prefill-in-one-block and token-by-token decode can
freeze *different* directions and diverge — there is deliberately **no
bit-for-bit equivalence guarantee** there, only order-invariance given the
same frozen filter.

With a **calibrated** filter the direction never depends on traffic, so that
divergence cannot arise: prefill and decode produce bit-identical kept sets
(`test_calibrated_filter_is_path_independent`).

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="qfilters",
    head_dim=128,
    qfilters_budget=512,  # max tokens kept (incl. sinks)
    qfilters_n_sink=4,  # leading positions never evicted
    qfilters_recent=0,  # trailing protected window (extension, off)
    qfilters_calib_tokens=128,  # fallback only: tokens before the filter freezes
    qfilters_sign=1,  # +1 = paper direction; -1 = inverted ablation
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

To use paper-faithful calibrated filters, pass them per layer — this skips the
calibration window and makes eviction path-independent:

```python
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache
from veloxquant_mlx.quantizers.qfilters_calibration import load_qfilters

cal = load_qfilters("qfilters_llama32_3b.npz", expect_model_id=model_id)
caches = [QFiltersKVCache(config, filters=f) for f in cal.filters]
model.make_cache = lambda *_a, **_k: caches
```

`expect_model_id` is checked on load: filters are model-specific, and reusing
another model's artifact would degrade quality with no visible error.

Single-layer, no coordinator — the default `for_model` path returns one
`QFiltersKVCache` per attention layer.

## How it works

Per `update_and_fetch` block:

1. Concatenate incoming K/V onto the kept set.
2. **Calibrated:** the filter already exists — go straight to step 4.
   **Fallback, before `calib_tokens` keys:** keep everything (no filter yet,
   so no eviction can happen); once enough keys are seen, estimate it from the
   key SVD and freeze.
3. Score every stored token by `sign · (key · filter_dir)` (float32). Scores
   are computed once at insertion and never updated.
4. If over `qfilters_budget`: keep the `budget` **highest-scoring** positions
   in one top-k, with sinks (first `qfilters_n_sink`) and the optional
   trailing `qfilters_recent` window forced to survive. Kept tokens preserve
   original temporal order.

Byte accounting mirrors L2Norm's — `qfilters_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept` — and additionally counts
the frozen `filter_dir` in full (`head_dim × 4` bytes, float32, per head).

## Metal kernels

Steps 3–4 have a fused two-dispatch GPU path
(`veloxquant_mlx/metal/_qfilters_evict.py`), batched across all `(batch, head)`
pairs at once, alongside the existing H2O and Keyformer eviction kernels:

1. `qfilters_score.metal` — the full `[BH, n_total]` projection score array,
   accumulated in fp32 over fp16 keys, with sink/recent rows forced to `+inf`.
2. `qfilters_evict_apply.metal` — compacts survivors against a keep-threshold,
   one threadgroup per group, preserving temporal order.

Unlike H2O/Keyformer — which evict exactly one row per token, so dispatch 1 is
an argmin and dispatch 2 a closed-form index shift — Q-Filters evicts a whole
block at once, so there is no closed form and survivors are gathered via a
cooperative scan. The keep-threshold itself is chosen with `mx.sort` on the MLX
side, since a top-k is a primitive MLX already implements well.

**The tie trap:** thresholding with `score >= thresh` keeps *more* than
`budget` rows whenever the threshold value repeats — and duplicates are the
norm here, because every protected row shares `+inf`. Overflowing would
silently break the cache's size guarantee, so rows are admitted in two tiers
(strictly-greater always; equal-to-threshold only while quota remains,
lowest index first), reproducing `mx.argsort`'s tie-break. Both behaviors are
pinned by `test_tied_scores_do_not_overflow_budget` and
`test_all_protected_rows_tie_at_infinity`.

Budgets above `QFILTERS_MAX_BUDGET` (4096) raise rather than truncate — the
survivor index list is staged in threadgroup memory, which needs a
compile-time bound.

## Adaptation notes

**What we do NOT implement:**
- RoPE position-ID remapping after eviction (same as every eviction method
  here) — so the Metal apply kernel copies keys bit-identically, with no
  re-rotation step.
- Per-head budgets (uniform across heads, same as H2O/TOVA/CaM/L2Norm).

**Extensions beyond the paper (off by default):**
- `qfilters_recent` — protects the most recent tokens StreamingLLM-style.
- `qfilters_sign=-1` — the inverted scorer, meaningful as an ablation arm on
  the key-SVD fallback path, where the sign is ambiguous. With calibrated
  filters the paper's `+1` is correct by construction.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_qfilters.py` (12),
`veloxquant_mlx/tests/quantizers/test_qfilters_calibration.py` (20),
`veloxquant_mlx/tests/cache/test_qfilters_cache.py` (20) and
`veloxquant_mlx/tests/metal/test_qfilters_evict.py` (20).

**Calibration (query-SVD, paper-faithful):**

- `compute_qfilters` recovers a planted query-drift direction **with its sign**
  (signed cosine > 0.99), and tracks the orientation when the drift is negated
- No mean-centering: on data where drift and centered-variance point along
  different axes, the paper's estimator returns the drift axis while a
  centering estimator returns the variance axis
- The contrast that motivates the module: on pure-drift key geometry the
  key-SVD fallback's cosine to the planted direction falls **below 0.3**, while
  query-SVD stays above 0.99 on the same geometry
- GQA group-averaging keeps filters unit-norm; artifact round-trips, and
  version / missing-layer / model-id mismatches all raise

**Cache, calibrated path:**

- Evicts from token 0 with no calibration window (fallback still holds all 80
  tokens at the same point)
- **Prefill vs. decode is bit-for-bit identical** — the path-independence the
  fallback cannot provide
- Kept set equals the budget highest-projection rows against the filter
- Filters whose head layout disagrees with the layer raise rather than
  silently mis-evicting

**Metal kernels:** scores match `sign · (keys @ filter_dir)` in fp32; the
evicted set matches an argsort reference for keys *and* values across four
shape/protection configurations; budget is respected exactly when all scores
tie and when the kept set is entirely `+inf` protected rows.

**Fallback path (unchanged):**

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
output-perturbation and byte-accounting numbers — not perplexity or throughput
on a real model. The calibration path is validated on constructed query
geometry and on `collect_query_activations`' hook mechanics, **not** by
measuring the paper's anisotropy claim on real trained weights, and no
end-to-end perplexity or TTFT comparison against the paper's Figures 5/10
exists here yet. Treat the paper's reported accuracy and TTFT numbers as the
paper's, not as reproduced.

## When to use it

Q-Filters is the repo's projection-based eviction: a per-head direction gives
importance eviction with zero per-step scoring cost, like
[L2Norm](../algorithms/knorm), but keyed to head geometry rather than raw norm.

**With calibrated filters** it is the closest thing here to the paper's method:
sign-correct, path-independent, and free of a warm-up window — at the cost of
one offline pass per model. **Without them** the key-SVD fallback keeps working
with zero setup, but its sign is ambiguous (expect to try `qfilters_sign=±1`)
and its kept set is path-dependent.

If you want a grouping-independent, sign-unambiguous intrinsic scorer with no
calibration at all, prefer [L2Norm](../algorithms/knorm); if you want a score
that reacts to the actual query stream, use [H2O](../algorithms/h2o).

| Method | Score | Per-step cost | Path-independent |
|--------|-------|---------------|------------------|
| H2O | cumulative proxy-attention mass | softmax over cache | no |
| [L2Norm](../algorithms/knorm) | intrinsic key norm | none | yes (`recent=0`) |
| **Q-Filters** (calibrated) | projection onto the query-SVD filter | none | **yes** |
| **Q-Filters** (fallback) | projection onto frozen key-SVD direction | none (after calibration) | no |

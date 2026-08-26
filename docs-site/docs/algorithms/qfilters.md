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
    collect_query_activations,
    compute_qfilters,
    average_gqa_filters,
    QFiltersCalibration,
    save_qfilters,
)

acts = collect_query_activations(model, tokenizer, calibration_texts)  # per layer [H_q, N, D]
filters = [average_gqa_filters(compute_qfilters(a), n_kv_heads=8) for a in acts]
save_qfilters(
    QFiltersCalibration(
        filters,
        model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        n_samples=3000,
        dataset="pile-subset",
    ),
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
    qfilters_recent=0,  # trailing protected window; set ~budget/4 for generation
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

### Wired into the cache's hot path (#179)

Through 0.44.x these kernels existed and were tested (`test_qfilters_evict.py`)
but `QFiltersKVCache` never called them — the calibrated/batched eviction
branch always ran the pure-MLX `mx.argsort` / `mx.take_along_axis` selection
in `qfilters_update_batched`, even on a machine where Metal was available.
The B×H Python loop itself was already removed for the calibrated path back
in #173; what #179 found still on the table was this second layer — the
already-written fused kernel sitting unused.

`QFiltersKVCache` now resolves a three-state `use_metal_kernels` flag at
construction (`None` auto-detects, `True` requires Metal and raises
immediately if unavailable or if `qfilters_budget` exceeds
`QFILTERS_MAX_BUDGET`, `False` forces the pure-MLX path — same convention as
`VecInferKVCache`). When Metal is selected and a group is actually over
budget, `_update_batched` calls `qfilters_fused_evict` instead of
`qfilters_update_batched`; both share the same scoring formula and
tie-breaking convention, so `test_metal_path_matches_pure_mlx_path` (Metal
hardware only) asserts the two are bit-for-bit interchangeable. Under-budget
calls and the key-SVD fallback's per-head loop are untouched either way — the
fallback's path-dependence still requires per-group filter freezing that a
single shared-filter selection (Metal or not) cannot express.

This has **not been benchmarked** — no TTFT/decode-throughput numbers exist
yet for the fused-kernel path vs. the pure-MLX batched path it now sits
alongside. That measurement needs real Apple Silicon; see
[Still not measured](#still-not-measured).

## Adaptation notes

**What we do NOT implement:**
- RoPE position-ID **renumbering** after eviction. Survivors keep their
  original absolute positions, so the cache reports the true token position and
  RoPE stays correct with no re-rotation — which is why the Metal apply kernel
  copies keys bit-identically (H2O/Keyformer renumber, so they need a
  delta-rotation pass). Positions do become non-contiguous where tokens were
  dropped; the paper does not address this either.
- Per-head budgets (uniform across heads, same as H2O/TOVA/CaM/L2Norm).

**Extensions beyond the paper (off by default):**
- `qfilters_recent` — protects the most recent tokens StreamingLLM-style.
  **Effectively required for open-ended generation** (16× perplexity swing —
  see [Generation perplexity](#generation-perplexity)); left off by default so
  the default configuration stays paper-faithful.
- `qfilters_sign=-1` — the inverted scorer, meaningful as an ablation arm on
  the key-SVD fallback path, where the sign is ambiguous. With calibrated
  filters the paper's `+1` is correct by construction.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_qfilters.py` (12),
`veloxquant_mlx/tests/quantizers/test_qfilters_calibration.py` (20),
`veloxquant_mlx/tests/cache/test_qfilters_cache.py` (33) and
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

## Real-model validation

Measured on trained MLX weights — scripts in
`benchmark_scripts/qfilters_real_model_anisotropy.py` and
`qfilters_real_model_attention_corr.py`, raw numbers in
`figures/qfilters/real_model_results.json`.

### The paper's anisotropy holds (Observations 3.1 / 3.2)

| Model | `E⟨Q,uʰ⟩ > 0` | `\|E⟨Q,v₁⟩\|` vs. other components | Top-component energy |
|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | **100%** of 512 heads | 44.4× | 90.6% |
| Llama-3.2-3B-Instruct-4bit | **100%** of 672 heads | 52.3× | 84.7% |
| Qwen2.5-7B-Instruct-4bit | **100%** of 784 heads | 47.9× | 81.3% |

Observation 3.1 holds in **every head measured** — that universal positivity is
`κʰ > 0`, the property the sign correctness rests on. Observation 3.2 holds
too: the leading component's mean projection is ~50× the rest, which have
near-zero mean, matching Figure 2c. Qwen2.5 is the paper's §5 limitation case
(QKV bias), yet the anisotropy is present there as well. Anisotropy alone does
not guarantee an end-to-end win, though: see the Qwen budget sweep below, where
the advantage disappears at mild compression.

### The calibrated filter predicts real attention (Figure 4)

Spearman correlation against **true** mean attention `Sʰₜ` from real attention
maps on held-out text:

| Scorer | Llama-3.2-1B (128 KV heads) | Llama-3.2-3B (224 KV heads) | Qwen2.5-7B (112 KV heads) |
|---|---|---|---|
| **Q-Filters, calibrated (query-SVD)** | **+0.783** (100% sign-correct) | **+0.863** (100% sign-correct) | **+0.850** (100% sign-correct) |
| K-norm ([L2Norm](../algorithms/knorm)) | +0.460 (94.5%) | +0.410 (94.6%) | +0.402 (86.6%) |
| Q-Filters, key-SVD fallback | **−0.032** (46.1%) | **−0.008** (49.1%) | **−0.039** (48.2%) |

The calibrated filter tracks true attention strongly and beats K-norm,
reproducing the paper's Figure 4 ordering. The key-SVD fallback is
**statistically indistinguishable from noise** — its sign is correct less than
half the time, i.e. worse than a coin flip. This is the sharpest available
evidence that the query-SVD estimator is not a refinement of the key-side one
but a categorically different signal.

### Generation perplexity

Token-by-token generation with eviction active (paper §4 / Figure 5 setup),
1024 tokens of continuous prose. Script:
`benchmark_scripts/qfilters_real_model_perplexity.py`.

**Llama-3.2-1B** (fp16 baseline **4.050**):

| Budget | Calibrated | Fallback | Gap to fp16 closed |
|---|---|---|---|
| 256 (~4×) | **8.476** | 13.358 | **52%** |
| 128 (~8×) | **16.307** | 23.645 | **37%** |
| 64 (~16×) | **25.933** | 31.274 | **20%** |

**Llama-3.2-3B** (fp16 baseline **3.305**):

| Budget | Calibrated | Fallback | Gap to fp16 closed |
|---|---|---|---|
| 256 (~4×) | **5.076** | 7.264 | **55%** |
| 128 (~8×) | **10.046** | 14.909 | **42%** |

The calibrated filter wins at every budget on both models, matching the
attention-correlation ordering. But note the absolute numbers: perplexity is
still well above fp16. This is a **policy comparison, not a reproduction of
Figure 5** — no RoPE renumbering, uniform per-head budgets, and a much smaller
calibration set than the paper's §4.2.

#### Qwen2.5-7B and a wider budget sweep (#176)

The rows above use a 1024-token eval, which caps the usable budget: at
budget 1024 the cache never fills, no eviction runs, and every arm returns the
fp16 baseline. The sweep below therefore uses a longer **~3485-token** corpus,
so budgets 256/512/1024 sit at roughly 13.6×/6.8×/3.4× compression.
**These numbers are not comparable to the 1024-token tables above.**
Llama-3.2-1B was re-run on the same corpus to keep the comparison matched.

**Qwen2.5-7B** (fp16 baseline **2.325**):

| Budget | Calibrated | Fallback | L2Norm | Gap to fp16 closed |
|---|---|---|---|---|
| 256 (~13.6×) | **10.478** | 11.660 | 11.297 | 13% |
| 512 (~6.8×) | **7.000** | 8.416 | 7.751 | 23% |
| 1024 (~3.4×) | 4.015 | 4.010 | **3.560** | −0% |

**Llama-3.2-1B** (fp16 baseline **3.445**):

| Budget | Calibrated | Fallback | L2Norm | Gap to fp16 closed |
|---|---|---|---|---|
| 256 (~13.6×) | **27.671** | 34.631 | 30.836 | 22% |
| 512 (~6.8×) | **15.975** | 24.126 | 16.855 | 39% |
| 1024 (~3.4×) | 6.818 | 9.237 | **6.518** | 42% |

Two findings, one positive and one not:

The query-SVD advantage **does** survive at 7B scale. Qwen's Spearman
correlation (+0.850) is close to Llama-3.2-3B's, and the calibrated filter beats
the fallback at budgets 256 and 512.

It does **not** hold uniformly across compression ratios. At budget 1024 on Qwen
the calibrated and fallback arms are tied (4.015 vs 4.010) and plain L2Norm beats
both. The advantage is also consistently smaller on Qwen than on Llama-3.2-1B at
matched compression — 13% vs 22% at budget 256, 23% vs 39% at budget 512. That is
consistent with the paper's §5, which lists Qwen-2.5 as a limitation case because
of its QKV projection bias. If you are running Q-Filters on Qwen at mild
compression, measure against L2Norm before assuming the calibrated filter helps.

#### `qfilters_recent` is effectively required for generation

The single largest factor. Pure projection ranking is a *long-range importance*
signal, and it will happily evict the immediately-preceding tokens — which is
what next-token prediction leans on hardest. Llama-3.2-1B, budget 128,
calibrated:

| `qfilters_recent` | 0 | 32 | 64 | 96 |
|---|---|---|---|---|
| perplexity | **263.2** | **16.3** | 20.3 | 24.9 |

A 16× swing. The paper evaluates retrieval and NIAH tasks where recency matters
far less, so this does not contradict it — but if you use Q-Filters for
open-ended generation, set `qfilters_recent` (≈ budget/4 worked best here).
It remains off by default to keep the default paper-faithful.

#### Comparison arms and the RoPE precondition (#183)

An eviction cache may only join these comparisons once its post-eviction RoPE
handling is verified. `mlx_lm` rotates both the query and the incoming key at
`offset=cache.offset` *before* `update_and_fetch` runs, so a cache that reports
the retained-row count instead of the true token position stalls its offset at
`budget` the moment eviction begins, and the drift grows without bound. An arm
in that state measures position drift rather than eviction quality — which
would make whichever method is correct look good for the wrong reason.

All four arms report the true absolute position, verified at zero drift through
a 200-token prefill plus 120 decode steps at budget 64, and pinned as a shared
contract in `tests/cache/test_eviction_rope_contract.py`. They reach it two
different ways, both valid:

| arm | positions after eviction | needs re-rotation | recency guard |
|---|---|---|---|
| Q-Filters | preserved (gapped) | no | `qfilters_recent` (default 0) |
| TOVA | preserved (gapped) | no | none — intrinsic |
| L2Norm | preserved (gapped) | no | `knorm_recent` (default 0) |
| H2O | **renumbered** (gap-free) | **yes** | `h2o_grace` (default 16) |

**Intentional differences from the original algorithms.** H2O renumbers
survivors to a contiguous layout, so it de-rotates and re-rotates kept keys
via `rope_remap_positions`; `h2o_rope_base` must match the model's own RoPE
base or that correction will not cancel. Q-Filters, TOVA and L2Norm leave
positions untouched, so surviving keys keep the rotation they were stored
with and non-contiguous gaps are expected — neither the Q-Filters nor the
TOVA paper addresses renumbering.

H2O and TOVA are run at budget and `n_sink` only, with **no** trailing-window
override. That is faithful rather than an oversight: H2O's accumulated-attention
score and TOVA's last-query attention both carry intrinsic recency bias, so
unlike pure projection ranking they do not need the trailing guard that
Q-Filters requires to stay coherent (see `qfilters_recent` above).

#### Eviction semantics: one-shot vs incremental (#172, #183)

`qfilters_update` absorbs a whole block of `S` tokens and evicts down to budget
**once**, rather than evicting after every sub-block. On the calibrated path this
is not an approximation of incremental eviction — it is the *same function*. A
token is scored once against a filter frozen before token 0, and its score never
changes while cached, so top-k under a fixed key is order-invariant. Feeding a
prompt as one block, in 256-token chunks, or one token at a time retains
bit-identical K/V and the same `offset`
(`test_calibrated_kept_set_is_invariant_to_prefill_chunk_size`).

Incremental eviction is therefore pure overhead on that path. Prefilling 4096
tokens, 8 heads, D=128, budget 512:

| prefill chunk | eviction calls | prefill time |
|---|---|---|
| one-shot (4096) | 1 | **1.8 ms** |
| 1024 | 4 | 3.4 ms |
| 256 | 16 | 7.9 ms |
| 64 | 64 | 23.7 ms |
| 16 | 256 | 65.8 ms |

Up to **36× slower for identical output**. One-shot stays the default; there is
no cache-pressure or adaptive variant to add, because there is no quality
difference to trade against.

The **key-SVD fallback** is the genuine exception, and its path-dependence is a
property of the *estimator*, not of the eviction step: the direction is frozen
from whichever chunk first crosses `qfilters_calib_tokens`, so different
chunkings freeze different directions (cosine to the one-shot direction measured
at +0.65 / +0.43 / +0.56 for chunks of 512 / 256 / 64) and retain different
token sets. The budget bound and `offset` hold at every chunk size regardless
(`test_fallback_respects_budget_at_every_prefill_chunk_size`). Chunked prefill
on the fallback is not more correct than one-shot — just different. If you need
chunk-invariance, supply calibrated filters.

Chunk size was originally suspected as the cause of degenerate `"the the the
the"` generation. It is not: the retained set is chunk-invariant on the
calibrated path. The actual cause is `qfilters_recent=0` — see above.

#### RoPE positions had to be fixed first (#171)

`mlx_lm` rotates both the query and the incoming key at `offset=cache.offset`
*before* `update_and_fetch` runs. The base class left `offset` equal to the
retained-row count, so it stalled at `budget` once eviction began: at
budget 64 the position drift reached **+135 by token 199** and grew without
bound. Fixing it moved calibrated perplexity from **598.5 → 17.6**.

Q-Filters **preserves** original positions (it drops rows but never renumbers
them) and RoPE is relative, so reporting the true position is sufficient and
survivors need no re-rotation — unlike H2O/Keyformer, which renumber and
therefore need a delta-rotation pass.

#### Beyond NIAH: the other RULER categories (#177)

The retrieval evidence above is the needle family only. That leaves the
question of whether the method generalises past single-span lookup, which
these four task categories answer. Script:
`benchmark_scripts/qfilters_ruler_beyond_niah.py`, raw numbers in
`figures/qfilters/ruler_beyond_niah.json`.

Qwen2.5-7B-Instruct-4bit, 5 seeds per cell, mean over contexts 1024 and 2048,
every arm at a matched budget. These are **not** RULER's own harness or its
scores — the generators follow the constructions in Hsieh et al.
(arXiv:2404.06654) but are written here and run at short contexts, so read
them as a relative comparison between cache methods, not as RULER numbers.
`qa_synthetic` is not RULER's QA task, which wraps SQuAD/HotpotQA; it is
synthetic two-hop QA and is easier.

| Task | fp16 | Q-Filters | SnapKV | StreamingLLM | L2Norm |
|---|---|---|---|---|---|
| VT (chain tracking) | 69% | 0 / 0 / 0 | 0 / 0 / 3 | 0 / 12 / **46** | 0 / 0 / 3 |
| CWE (common words) | 100% | 9 / 33 / **56** | 0 / 0 / 0 | **100 / 93 / 100** | 23 / 38 / 53 |
| FWE (frequent words) | 69% | 43 / 40 / 46 | 0 / 0 / 13 | **77 / 64 / 52** | 49 / 43 / 43 |
| QA (two-hop) | 90% | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Cells are budget 256 / 512 / 1024. fp16 ignores the budget and is the
ceiling. All four task rows clear the harness's discriminative gate (fp16
well above the floor), so each is a method comparison rather than a
measurement of what the model cannot do.

**Q-Filters does not generalise past needle retrieval.** It scores 0% on
variable tracking and 0% on two-hop QA at every budget, against ceilings of
69% and 90%. This is not a prefill artifact: chunking the prefill — the
path-dependence lever that restored coherence in the NIAH harness — leaves VT
at 0.00 one-shot, 0.00 at 256-token chunks and 0.11 at 64-token chunks
(`qfilters_ruler_prefill_chunking.py`).

**The failures are task-shaped, not method-shaped.** VT and QA need a
specific *conjunction* to survive: a whole assignment chain, or both hops of
person → place → instrument. Partial retention earns nothing, so every arm
bottoms out. CWE and FWE award partial credit, and there Q-Filters degrades
gracefully instead — 9 → 33 → 56% on CWE, scaling cleanly with budget. Three
task categories, three different shapes; NIAH alone shows none of this.

**Budget does not always help.** On FWE, Q-Filters is flat across budgets
(43 / 40 / 46%) and so is L2Norm. Zeta-distributed frequencies make the answer
depend on counting the whole list, so retaining a larger arbitrary subset
does not improve the count.

**SnapKV floors on the three aggregation tasks** because its trailing
`snap_obs_window` lands on the question text rather than on the data it must
aggregate, so the proxy queries score the wrong region as important. Having
evicted the word list, the model answers from priors: at budget 1024 on CWE it
returns a fluent, well-formed and entirely fabricated `"opportunity",
"innovation", "inspiration"` — none of which appear in the prompt. This is a
proxy-query method meeting a task whose relevant span is not where its proxy
looks, not a general claim about SnapKV.

Two caveats on the StreamingLLM column, which otherwise looks like a clean
win. Its 100% on CWE is partly a property of the construction: the word list
sits at the end of the prompt, immediately before the question, which is
exactly where a trailing window keeps tokens. And its 77% on FWE sits *above*
the 69% fp16 ceiling — that is seed variance at the ceiling, not compression
beating no compression, and it is reported as measured rather than clipped.

Every arm was verified to compress equally before these numbers were read:
at budget 512 on a 1272-token CWE prompt, all four retain exactly 512 rows
and report offset 1272 (`qfilters_ruler_budget_verification.py`). No arm is
advantaged by keeping more, and none carries the `offset` defect #171/#174
fixed, so the 0%–100% spread is entirely *which* tokens each method selected.

One caution on reading budgets. The harness's context lengths size the filler
only; instruction, task spans and question add 100–250 tokens, so a nominal
1024-token QA prompt is 1169 tokens and budget 1024 still evicts ~145 of them
— enough to break a hop. `prompt_tokens` in the JSON records the real
prefilled length per task and context.

#### Still not measured

**RULER's remaining task categories.** #177 covers variable tracking,
common/frequent-word extraction and QA (above); RULER ships 13 tasks, and the
rest — multi-hop tracing variants and the aggregation tasks at long context —
are not covered, nor is any context beyond 2048 tokens.
**Model coverage for the RULER results is a single model.** Llama-3.2-1B
cannot perform VT, CWE or FWE even at fp16 (ceilings 11% / 16% / 26%), so
those cells cannot evaluate cache methods at that scale; Llama-3.2-3B would
have contributed genuine CWE and FWE cells (ceilings 100% / 81%) and was cut
for runtime.
**Expected Attention** is not implemented in this repo, so it cannot be a
comparison arm — tracked in #178. [L2Norm](../algorithms/knorm) previously
carried the same `offset` defect and was excluded for that reason; #174
generalised the `_true_offset` fix to it, so it is now included as a fair
comparison arm.

**TTFT / decode throughput of the fused-Metal-kernel path** (#179). The
kernel wiring above is correctness-tested (bit-for-bit parity against the
pure-MLX batched path, Metal hardware only) but not benchmarked — no
before/after TTFT or tok/s numbers exist yet for it, on any model size. The
`n_total <= budget` case and the key-SVD fallback loop never touch the
kernel, so any speedup is bounded to the calibrated path's over-budget
steps specifically. Measuring this needs real Apple Silicon.

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

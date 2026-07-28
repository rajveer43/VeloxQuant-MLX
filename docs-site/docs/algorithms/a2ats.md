---
id: a2ats
title: A2ATS-adapted
sidebar_label: A2ATS-adapted
slug: /algorithms/a2ats
---

# A2ATS-adapted — Windowed RoPE + Query-Aware Retrieval VQ

**Method id:** `a2ats` · **New in 0.39.0** · *Inspired by* ["A2ATS:
Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and
Query-Aware Vector Quantization" (He, Xing, Wang, Xu, Wu, Zhou, Liu, Xue, Li —
**ACL 2025 Findings**)](https://aclanthology.org/2025.findings-acl.644/) —
**A2ATS-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

A2ATS-adapted joins the repo's vector-quantization family
([VecInfer](../algorithms/vecinfer), [CommVQ-adapted](../algorithms/commvq), [RaBitQ](../algorithms/rabitq),
[NSNQuant](../algorithms/nsnquant)) with a mechanism no existing method
combines: **RoPE-position-aware windowing of the compression scheme itself**,
plus **query-aware codebook assignment** for a retrieval-fraction subset of
tokens. It is a normal-track method — a live-verified peer-reviewed venue, no
exception needed (unlike [AMC-adapted](../algorithms/amc) or
[NestedKV-adapted](../algorithms/nestedkv), which shipped as one-time venue
exceptions).

## Where it sits — the mechanism gap

| Method | RoPE handling | Query-aware? | Selection axis |
|---|---|:---:|---|
| [VecInfer-adapted](../algorithms/vecinfer) | None — smooth + Hadamard transform only | No | Codebook only |
| [CommVQ-adapted](../algorithms/commvq) | Codebook-constraint (train pre-RoPE, apply once at decode) | No | Codebook only |
| **A2ATS-adapted** | **Distance-gated: exact within a trailing window, key left unrotated outside it (constant `R_b` rides on the query)** | **Yes (retrieval-fraction subset)** | **Codebook + per-token retrieval split** |

[CommVQ-adapted](../algorithms/commvq) solves RoPE by constraining *what the codebook can represent*
(centroids trained in a pre-RoPE frame, uniform treatment of every position).
A2ATS-adapted instead changes *when* exact-vs-approximate RoPE is paid for,
gated by each token's distance from the current decode position — a
genuinely different axis, and in principle composable with [CommVQ-adapted](../algorithms/commvq)'s
approach (not attempted here).

## :warning: The honesty crux — read this first

1. **No query visible at cache level.** Like every other query-aware method
   in this repo ([AMC-adapted](../algorithms/amc)'s
   `amc_use_query_saliency`, [H2O](../algorithms/h2o)'s key-as-query proxy,
   [SnapKV](../algorithms/snapkv)'s prefill window), `update_and_fetch` only
   ever receives keys and values — the true decode-time query is not part of
   the mlx_lm cache protocol. This port substitutes the incoming key vector
   itself as a proxy query for both the retrieval-set split and the
   query-aware codebook assignment. Same category of approximation as those
   methods, not a new one.
2. **Windowed RoPE has a real, nonzero cost — measured directly, not just
   asserted.** The benchmark below shows windowed RoPE is worse than
   always-exact RoPE in *every* geometry tested, not only the long-range
   one. This survived the [#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)
   correctness fixes: with Eq. (12) implemented properly the *near* bucket is
   now numerically identical to always-exact (max gap `5e-07`), so the entire
   penalty is far tokens — which are the overwhelming majority of any long
   sequence. Replacing each far token's true relative position with one
   constant `b` is simply lossy, and sweeping `b` does not remove it.
3. **Query-aware assignment trades reconstruction fidelity for a property
   this benchmark cannot measure.** `a2ats_beta=1.0` reduces exactly to
   plain nearest-centroid VQ; any `beta<1.0` necessarily moves away from the
   pure-reconstruction optimum. The benchmark shows this plainly: query-aware
   assignment has *higher* reconstruction MSE than plain VQ in every row
   measured. The intended payoff — better downstream retrieval/attention
   quality for the query-relevant subset — is not something an offline
   reconstruction-MSE benchmark can show.
4. **The cosine-blend path is not the paper's estimator.** Paper Eq. (13)/(14)
   minimizes an `H`-weighted quadratic form, where `H = E[q̃ᵀq̃]` is the query
   second-moment matrix (§3.2's whole argument is that `H ∦ I`). That
   objective **is** implemented — `a2ats_h_weighted_assignment`, exact via the
   Eq. (15)–(18) Cholesky identity — but it needs a calibrated `H`, supplied
   via `a2ats_query_h`. Without one, the cache falls back to a cosine blend
   between reconstruction error and query-centroid alignment, which is a
   *substitute*, not an approximation of Eq. (14): its cosine term is a
   constant per-centroid bias rather than a per-token coupling, and its
   `beta` is scale-dependent. See
   [#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29), finding 4.
5. **Per-step re-rotation is `O(total_tokens)`, not `O(S)`.** Eq. (11) makes
   near/far a function of the *advancing* decode query, so the split cannot be
   precomputed and baked into the stored cache — doing so freezes each token's
   class at write time and silently defeats the distance gating. This port
   therefore stores pre-RoPE keys and re-applies windowed RoPE to the whole
   accumulated cache every step. Far tokens are returned unrotated, so only
   the `window`-sized near slice does nontrivial work, but the pass itself is
   over the full cache.
6. **Retrieval set gets preferential codebook assignment, not eviction.**
   Every token is quantized and retained; the retrieval-fraction split only
   changes which centroid a token is matched against. No token is ever
   dropped — a compression-only method (same framing as
   [AMC-adapted](../algorithms/amc)).
7. **Offline codebook calibration required**, same footgun class as
   [VecInfer-adapted](../algorithms/vecinfer)/[CommVQ-adapted](../algorithms/commvq)/
   [Palu](../algorithms/palu)/[SVDq](../algorithms/svdq)/
   [AMC-adapted](../algorithms/amc): the default random-init codebook exists
   only so wiring/shape tests don't require a calibration pass. Using
   `a2ats` in production without a codebook trained on representative data
   (`a2ats_codebook` config field) degrades to near-random quantization.
8. **No CUDA kernel fusion reproduced.** Same MLX/Metal disclaimer as every
   other VQ-family method here: the benefit on Apple Silicon is memory
   footprint, not throughput — the paper's own numbers assume a fused
   kernel this port does not have.
9. Nothing here is validated on a trained model or real hardware. The
   paper's own retrieval-accuracy and throughput numbers are measured on
   real long-context LLM workloads this repo does not have.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_codebook_bits=8,          # codebook size 2^bits
    a2ats_sub_dim=8,                 # VQ sub-vector width
    a2ats_window=128,                # trailing exact-RoPE window w (positions)
    a2ats_b=2048,                    # constant far-token relative position b — independent of w
    a2ats_use_query_aware=True,      # paper's primary reported path (default ON)
    a2ats_beta=0.5,                  # query/reconstruction blend, in [0, 1] (cosine-blend fallback)
    a2ats_retrieval_fraction=0.20,   # fraction of tokens routed to query-aware assignment
    # a2ats_codebook=my_calibrated_codebook,  # REQUIRED for real use — see honesty crux, point 7
    # a2ats_query_h=my_query_second_moment,   # enables the paper's Eq. 14 assignment — see point 4
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Plain nearest-centroid path (no query-awareness, closer to
[VecInfer-adapted](../algorithms/vecinfer)'s default behavior):

```python
config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_use_query_aware=False,
)
```

Single-layer, no coordinator — the default `for_model` path returns one
`A2ATSKVCache` per attention layer. No `.bits` attribute — stores and returns
fp16 K/V directly.

### The paper's Eq. (14) assignment

To use the paper's actual query-aware objective rather than the cosine-blend
fallback, calibrate a query second-moment matrix `H` offline and pass it:

```python
from veloxquant_mlx.quantizers.a2ats import a2ats_query_second_moment

# collected_queries: [M, a2ats_sub_dim] post-RoPE query sub-vectors from a
# representative dataset (the paper's §4.2 offline calibration step)
h = a2ats_query_second_moment(collected_queries)

config = KVCacheConfig(method="a2ats", head_dim=128, a2ats_sub_dim=8, a2ats_query_h=h)
```

With `a2ats_query_h` set, retrieval-set tokens are assigned by minimizing
`(k̃ − c) H (k̃ − c)ᵀ` (Eq. 14), computed exactly via the Eq. (15)–(18)
Cholesky identity. `a2ats_beta` is then unused. Without it, the cosine-blend
substitute runs instead — see honesty crux point 4.

## How it works

Every call to `update_and_fetch` — prefill batch or single decode token
alike — runs:

1. **Retrieval-set split** (query-aware path only): the top
   `a2ats_retrieval_fraction` of tokens by proxy-query cosine similarity form
   the retrieval set; the rest are the bulk. Implemented via
   [`dsa.MaxHeap`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/dsa/heap.py)
   top-k selection, the same pattern
   [AMC-adapted](../algorithms/amc)'s `amc_assign_tiers` uses.
2. **Codebook assignment**: retrieval-set tokens get **query-aware**
   assignment — the paper's `H`-weighted Eq. (14) objective when
   `a2ats_query_h` is supplied, otherwise the `a2ats_beta` cosine blend
   (honesty crux point 4); bulk tokens get **plain** nearest-centroid
   assignment (identical to
   [VecInfer-adapted](../algorithms/vecinfer)'s `quantize_vq`).
3. **Dequantization** reconstructs the pre-RoPE key vector from its assigned
   centroid. This pre-RoPE reconstruction is what gets **stored**.
4. **Windowed RoPE, re-applied every step** to the whole accumulated cache
   against the current decode position: tokens within `a2ats_window`
   positions get *exact* RoPE at their own position; tokens outside it are
   returned **unrotated** (paper Eq. 12, `k̃ = k`), with the constant `R_b`
   that encodes "far" relative position belonging on the *query* side
   (Eq. 11, `u_ij = q_i R_b k_j^T`) — exposed as
   `A2ATSKVCache.far_query_rope` since the cache never sees the query.
   `a2ats_window<=0` degrades to all-unrotated; `a2ats_window` at or beyond
   the sequence length degrades to always-exact (equivalent to
   [CommVQ-adapted](../algorithms/commvq)'s uniform treatment).

   The rotation is deliberately *not* baked in at write time: Eq. (11) makes
   near/far a function of the advancing decode query, so freezing it at write
   time defeats the gating entirely (honesty crux point 5).
5. Values follow a plain nearest-centroid VQ path — no RoPE (values are
   never position-rotated), no retrieval-set preference. Same choice
   [ZipCache-adapted](../algorithms/zipcache)/[Palu](../algorithms/palu)
   make for their "values follow the safer default" fields.

## Byte accounting

- `compressed_key_bytes` / `compressed_value_bytes` — actual stored bytes.
- `fp16_key_bytes` / `fp16_value_bytes` — hypothetical full-rank fp16 cost.
- `compression_ratio` — combined fp16 / compressed ratio (> 1 = savings).
- `codebook_bytes` — static codebook overhead (fp16 storage, amortized).
- `assigned_avg_bits` — effective bits/element, excluding codebook overhead.
- `tokens_seen` / `tokens_retrieved` — cumulative counts, for observability.

## Benchmark — honestly reported, including the parts that didn't win

`benchmark_scripts/benchmark_a2ats.py` (results in
`figures/a2ats/results.json`) sweeps sequence length
(200/400) across two geometries, comparing windowed vs. always-exact RoPE,
and query-aware vs. plain nearest-centroid VQ assignment, all at the same
codebook/sub_dim.

RoPE is measured as error in the **attention score** `u_ij = q̃_i k̃_j^T`,
not in the key vector. That distinction matters: under Eq. (12) far keys are
deliberately left unrotated with `R_b` carried on the query, so scoring far
*keys* against exact-RoPE'd keys would count the method's own design as
error. (The pre-[#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)
benchmark did exactly that, against an implementation that also rotated far
keys wrongly.)

| Geometry | Bucket | Windowed | Always-exact | Ratio |
|---|---|---:|---:|---:|
| `local_recency` | overall | 3.929 | 1.347 | **2.9x** |
| | near (n=16) | 8.248850 | 8.248850 | **1.000000x** |
| | far (n=184–384) | 3.830 | 1.080 | 3.5x |
| `long_range_dependent` | overall | 8.384 | 1.014 | **8.3x** |
| | near (n=16) | 1.090521 | 1.090522 | **1.000001x** |
| | far (n=184–384) | 8.762 | 1.006 | 8.7x |

- **Windowed RoPE is worse than always-exact RoPE in every geometry
  measured**, and this *survived* the #29 correctness fixes — it is a real
  property of the approximation, not an artifact of the bugs. The near/far
  split is the evidence: with Eq. (12) implemented correctly the **near
  bucket is numerically identical** to always-exact (max gap `5e-07`), so
  the entire penalty comes from far tokens, which are ~92% of a 200-token
  sequence and ~96% of a 400-token one. Replacing each far token's true
  relative position with a single constant `b` is intrinsically lossy;
  sweeping `b` from 8 to 2048 does not remove the gap (it is smallest at
  small `b` and grows with it).
- **Query-aware assignment has higher reconstruction MSE than plain
  nearest-centroid VQ in every row measured** — mathematically expected
  (`a2ats_beta=1.0` reduces exactly to plain VQ; any lower `beta` trades
  reconstruction accuracy for query alignment), not a bug. This benchmark
  cannot measure the intended payoff (downstream retrieval/attention
  quality), only reconstruction fidelity — readers should not conclude
  query-aware assignment is "better" from these numbers. Note this row
  exercises the cosine-blend fallback, not the paper's `H`-weighted Eq. (14)
  path, which needs a calibrated `H` the synthetic harness does not have.

Deterministic in all non-`ms` fields, verified by diffing two runs.
Offline-synthetic; loads no model, no mlx_lm generation. **Not** a
reproduction of the paper's own retrieval-accuracy or throughput numbers.

## Adaptation notes — what we do NOT implement

- No CUDA kernel fusion; pure MLX from the start, same as every VQ-family
  method here.
- `A2ATSKVCache` does not auto-invoke a codebook training routine — callers
  must train a codebook on representative data (e.g. via
  `veloxquant_mlx.allocators.vecinfer.train_codebook`) and pass it via
  `a2ats_codebook` for real use (see honesty crux, point 7).
- No composition with [CommVQ-adapted](../algorithms/commvq)'s RoPE-commuting codebook constraint —
  the two RoPE-handling strategies are independent axes and not combined
  here.
- No automatic `H` calibration. `a2ats_query_second_moment` computes `H` from
  queries you collect, but nothing in this repo collects them for you — the
  paper's §4.2 offline pass over a representative dataset is the caller's job.
  Without `a2ats_query_h`, the cosine-blend fallback runs instead (honesty
  crux point 4).
- No query-side `R_b` application inside the cache. `far_query_rope` is
  exposed, but the mlx_lm cache protocol never hands `update_and_fetch` a
  query, so composing Eq. (11)'s two halves is left to callers with real
  query access.
- Any trained-model perplexity/throughput/retrieval-accuracy benchmark. The
  paper's own numbers are measured on real long-context LLM workloads this
  repo does not have.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_a2ats_rope.py` (17 tests),
`veloxquant_mlx/tests/quantizers/test_a2ats.py` (19 tests), and
`veloxquant_mlx/tests/cache/test_a2ats_cache.py` (31 tests):

- **`test_windowed_rope_within_window_matches_exact_rope`** /
  **`test_windowed_rope_outside_window_returns_key_unrotated`** — direct
  proof the window boundary behaves as documented: near tokens match exact
  RoPE, far tokens come back in the pre-RoPE frame (Eq. 12). The second test
  asserts *equality with the input*, not merely difference from exact RoPE —
  the weaker form let the pre-#29 `R_window` bug pass.
- **`test_windowed_rope_far_tokens_are_position_independent`** — proves the
  property WRoPE exists for: far keys carry no positional information, so a
  shared codebook can quantize them (§3.1, Observation 2).
- **`test_far_query_rope_b_is_independent_of_window`** /
  **`test_far_query_rope_reconstructs_paper_attention_score`** — proves `b`
  is a real knob separate from `w`, and that the two halves compose back into
  Eq. (11)'s `u_ij = q_i R_b k_j^T`.
- **`test_token_rotation_updates_as_decode_position_advances`** — proves a
  token's near/far class tracks the advancing query rather than freezing at
  write time (#29, finding 3).
- **`test_h_weighted_assignment_matches_bruteforce_eq14`** /
  **`test_h_identity_reduces_to_plain_nearest_centroid`** /
  **`test_h_weighted_differs_from_plain_vq_under_anisotropic_h`** — proves
  the Cholesky route computes Eq. (14) exactly (not approximately), that it
  reduces to plain VQ exactly when `H ∝ I` (§3.2's premise), and that it
  genuinely diverges from plain VQ when `H` is anisotropic — otherwise the
  paper's contribution would be vacuous.
- **`test_window_zero_always_unrotated`** /
  **`test_window_exceeds_seqlen_always_exact`** — proves both documented
  degradation boundaries.
- **`test_beta_one_reduces_to_nearest_centroid`** /
  **`test_query_aware_prefers_relevant_centroid_over_nearest`** — proves
  the query-aware blend does something a pure-reconstruction assignment
  cannot, and that `beta=1.0` is a true reduction to the plain-VQ baseline.
- **`test_retrieval_set_picks_most_similar_to_query`** — direct proof the
  retrieval-set split actually selects the query-relevant tokens on a
  hand-constructed similarity ranking, not arbitrary ones.
- **Config-validation tests written first**, per this repo's own lesson from
  a same-session bug hunt that found 5 sibling methods shipped without
  bounds-checking their fraction-valued config fields: `a2ats_beta` and
  `a2ats_retrieval_fraction` are validated to `[0, 1]` at construction time.
- Standard suite: byte accounting, determinism (including across mixed
  prefill+decode), `for_model` config propagation (all `a2ats_*` fields),
  factory dispatch, factory smoke test.

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_a2ats.py`
is offline-synthetic and deterministic in all non-timing fields —
reconstruction MSE only, not perplexity, retrieval accuracy, or throughput
on a real model.

## When to use it

A2ATS-adapted is for long-context workloads with **strong positional
locality**, where the tokens a query actually needs sit close to the current
decode position — the case where the windowed-RoPE approximation costs the
least. Tokens inside the window are now handled *exactly* (the near bucket is
numerically identical to always-exact RoPE), so the approximation is free for
whatever fraction of the query-relevant mass falls inside `a2ats_window`;
raising `a2ats_window` trades memory-access savings for fidelity along a
predictable curve.

It remains a poor fit for workloads with genuinely long-range dependencies:
the benchmark's `long_range_dependent` geometry shows the cost is largest
exactly there (8.3x vs. 2.9x), and that gap survived the
[#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29) fixes rather
than being an artifact of them. For those, prefer
[VecInfer-adapted](../algorithms/vecinfer) or [CommVQ-adapted](../algorithms/commvq), which apply
exact RoPE uniformly regardless of distance.

| Method | RoPE cost model | Query-aware | Verified venue |
|--------|---|:---:|:---:|
| [VecInfer-adapted](../algorithms/vecinfer) | N/A (no RoPE handling) | No | Yes |
| [CommVQ-adapted](../algorithms/commvq) | Uniform exact (codebook-constrained) | No | Yes |
| **A2ATS-adapted** | **Distance-gated: exact near, unrotated far** | **Yes (retrieval subset)** | **Yes** |

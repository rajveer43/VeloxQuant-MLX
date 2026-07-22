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
| **A2ATS-adapted** | **Distance-gated: exact within a trailing window, fixed-offset approximate outside it** | **Yes (retrieval-fraction subset)** | **Codebook + per-token retrieval split** |

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
   one. See the benchmark section for the actual numbers.
3. **Query-aware assignment trades reconstruction fidelity for a property
   this benchmark cannot measure.** `a2ats_beta=1.0` reduces exactly to
   plain nearest-centroid VQ; any `beta<1.0` necessarily moves away from the
   pure-reconstruction optimum. The benchmark shows this plainly: query-aware
   assignment has *higher* reconstruction MSE than plain VQ in every row
   measured. The intended payoff — better downstream retrieval/attention
   quality for the query-relevant subset — is not something an offline
   reconstruction-MSE benchmark can show.
4. **Retrieval set gets preferential codebook assignment, not eviction.**
   Every token is quantized and retained; the retrieval-fraction split only
   changes which centroid a token is matched against. No token is ever
   dropped — a compression-only method (same framing as
   [AMC-adapted](../algorithms/amc)).
5. **Offline codebook calibration required**, same footgun class as
   [VecInfer-adapted](../algorithms/vecinfer)/[CommVQ-adapted](../algorithms/commvq)/
   [Palu](../algorithms/palu)/[SVDq](../algorithms/svdq)/
   [AMC-adapted](../algorithms/amc): the default random-init codebook exists
   only so wiring/shape tests don't require a calibration pass. Using
   `a2ats` in production without a codebook trained on representative data
   (`a2ats_codebook` config field) degrades to near-random quantization.
6. **No CUDA kernel fusion reproduced.** Same MLX/Metal disclaimer as every
   other VQ-family method here: the benefit on Apple Silicon is memory
   footprint, not throughput — the paper's own numbers assume a fused
   kernel this port does not have.
7. Nothing here is validated on a trained model or real hardware. The
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
    a2ats_window=128,                # trailing exact-RoPE window (positions)
    a2ats_use_query_aware=True,      # paper's primary reported path (default ON)
    a2ats_beta=0.5,                  # query/reconstruction blend, in [0, 1]
    a2ats_retrieval_fraction=0.20,   # fraction of tokens routed to query-aware assignment
    # a2ats_codebook=my_calibrated_codebook,  # REQUIRED for real use — see honesty crux, point 5
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
   nearest-centroid assignment (`a2ats_beta`-weighted blend of reconstruction
   error and query-cosine-similarity); bulk tokens get **plain**
   nearest-centroid assignment (identical to
   [VecInfer-adapted](../algorithms/vecinfer)'s `quantize_vq`).
3. **Dequantization** reconstructs the pre-RoPE key vector from its assigned
   centroid.
4. **Windowed RoPE reconstruction**: tokens within `a2ats_window` positions
   of the current decode step get *exact* RoPE at their own position; tokens
   outside the window get a single shared *fixed-offset approximate*
   rotation (computed once at the window's trailing edge). `a2ats_window<=0`
   degrades to always-approximate; `a2ats_window` at or beyond the sequence
   length degrades to always-exact (equivalent to [CommVQ-adapted](../algorithms/commvq)'s uniform
   treatment).
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
(200/400) across two geometries, comparing windowed vs. always-exact RoPE
reconstruction, and query-aware vs. plain nearest-centroid VQ assignment,
all at the same codebook/sub_dim:

- **Windowed RoPE is worse than always-exact RoPE in every geometry
  measured** — roughly **2.8x** higher MSE on `local_recency` (where the
  query-relevant tokens sit inside the window) and roughly **4.4x** higher
  on `long_range_dependent` (where they sit outside it). The gap widens
  substantially in the long-range case, consistent with the approximation
  paying its cost exactly where relevant tokens are farthest — but the cost
  is real even in the favorable geometry, because every far token in a long
  sequence still gets the coarse shared rotation regardless of what's
  "relevant."
- **Query-aware assignment has higher reconstruction MSE than plain
  nearest-centroid VQ in every row measured** — mathematically expected
  (`a2ats_beta=1.0` reduces exactly to plain VQ; any lower `beta` trades
  reconstruction accuracy for query alignment), not a bug. This benchmark
  cannot measure the intended payoff (downstream retrieval/attention
  quality), only reconstruction fidelity — readers should not conclude
  query-aware assignment is "better" from these numbers.

Deterministic in all non-`ms` fields, verified by diffing two runs.
Offline-synthetic; loads no model, no mlx_lm generation. **Not** a
reproduction of the paper's own retrieval-accuracy or throughput numbers.

## Adaptation notes — what we do NOT implement

- No CUDA kernel fusion; pure MLX from the start, same as every VQ-family
  method here.
- `A2ATSKVCache` does not auto-invoke a codebook training routine — callers
  must train a codebook on representative data (e.g. via
  `veloxquant_mlx.allocators.vecinfer.train_codebook`) and pass it via
  `a2ats_codebook` for real use (see honesty crux, point 5).
- No composition with [CommVQ-adapted](../algorithms/commvq)'s RoPE-commuting codebook constraint —
  the two RoPE-handling strategies are independent axes and not combined
  here.
- Any trained-model perplexity/throughput/retrieval-accuracy benchmark. The
  paper's own numbers are measured on real long-context LLM workloads this
  repo does not have.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_a2ats_rope.py` (13 tests),
`veloxquant_mlx/tests/quantizers/test_a2ats.py` (13 tests), and
`veloxquant_mlx/tests/cache/test_a2ats_cache.py` (25 tests):

- **`test_windowed_rope_within_window_matches_exact_rope`** /
  **`test_windowed_rope_outside_window_uses_fixed_offset`** — direct proof
  the window boundary behaves as documented: near tokens match exact RoPE,
  far tokens genuinely differ (not an accidental match, which this repo has
  caught before — see AMC-adapted's saturated-clamp bug fix).
- **`test_window_zero_always_approximate`** /
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
least. It is a poor fit for workloads with genuinely long-range dependencies
(the benchmark's `long_range_dependent` geometry shows the approximation's
cost is largest exactly there) — for those, prefer
[VecInfer-adapted](../algorithms/vecinfer) or [CommVQ-adapted](../algorithms/commvq), which apply
exact RoPE uniformly regardless of distance.

| Method | RoPE cost model | Query-aware | Verified venue |
|--------|---|:---:|:---:|
| [VecInfer-adapted](../algorithms/vecinfer) | N/A (no RoPE handling) | No | Yes |
| [CommVQ-adapted](../algorithms/commvq) | Uniform exact (codebook-constrained) | No | Yes |
| **A2ATS-adapted** | **Distance-gated exact/approximate** | **Yes (retrieval subset)** | **Yes** |

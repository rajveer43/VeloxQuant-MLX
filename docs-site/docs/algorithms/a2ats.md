---
id: a2ats
title: A2ATS-adapted
sidebar_label: A2ATS-adapted
slug: /algorithms/a2ats
---

# A2ATS-adapted

A2ATS-adapted compresses the KV cache with **vector quantization**, and skips
the expensive position-encoding work for tokens far behind the one you're
currently generating. Recent tokens get exact treatment; distant ones get a
cheap approximation.

It's built for long contexts where **the tokens that matter are usually
nearby** — chat histories, code completion, running summaries.

:::warning[Needs a calibration pass]
Like [VecInfer](../algorithms/vecinfer), [CommVQ-adapted](../algorithms/commvq),
and [Palu](../algorithms/palu), this method needs a codebook trained on
representative data. Without one it falls back to a random codebook that
exists only so the plumbing tests can run — quality will be poor. See
[Calibration](#calibration-one-time-setup).
:::

:::note[Memory, not speed]
On Apple Silicon the win is **memory footprint**, not decode throughput. The
paper's speedup numbers assume a fused CUDA kernel this port doesn't have —
same disclaimer as every VQ-family method here.
:::

## Quick start

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_window=128,     # recent tokens get exact position encoding
    a2ats_codebook=my_calibrated_codebook,   # see Calibration below
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

That's the whole setup. The defaults for everything else are the paper's own
values and are reasonable starting points.

## Should you use this?

**Good fit:**

- Long contexts where relevant tokens are usually recent — conversations,
  iterative editing, code in the current file
- You can run a one-time offline calibration pass
- Memory is your constraint, not latency

**Poor fit:**

- Genuine long-range lookups — retrieval over a large document, "what did the
  contract say on page 3." The approximation costs the most exactly there;
  our benchmark measures **8.3x** higher attention-score error in that
  geometry versus 2.9x for the local case.
- You can't calibrate → try [TurboQuant RVQ](../algorithms/rvq), which needs
  no calibration
- You want uniform accuracy regardless of distance →
  [VecInfer](../algorithms/vecinfer) or [CommVQ-adapted](../algorithms/commvq)
  apply exact position encoding everywhere

### Tuning `a2ats_window`

This is the main dial, and it behaves predictably: **tokens inside the window
are handled exactly** — bit-identical to never approximating at all. So the
window is the fraction of your context you're paying full price for.

| `a2ats_window` | Effect |
|---|---|
| Larger (e.g. 512) | More accuracy, less savings |
| Smaller (e.g. 64) | More savings, more error on tokens that fall outside |
| `>= context length` | Degrades to always-exact — no approximation at all |
| `<= 0` | Everything approximated (not recommended) |

If you know roughly how far back your queries reach, set the window to cover
it.

## Calibration (one-time setup)

Train a codebook on a representative sample of your model's key activations
and persist it:

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.allocators.vecinfer import train_codebook

sub_dim = 8          # must match a2ats_sub_dim
bits = 8             # must match a2ats_codebook_bits

# Collect real key activations from a calibration prompt set —
# shape [n_tokens, n_heads, head_dim].
keys_calib = mx.array(np.random.default_rng(0).standard_normal(
    (4096, 8, 128)).astype(np.float32))

codebook = train_codebook(
    keys_calib.reshape(-1, sub_dim), n_centroids=2 ** bits, seed=42
)
np.savez("a2ats_codebook.npz", codebook=np.asarray(codebook))
```

Then load it in your config:

```python
data = np.load("a2ats_codebook.npz")
config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_codebook=mx.array(data["codebook"]),
)
```

### Optional: query-aware calibration

A2ATS can also bias codebook selection toward the directions your queries
actually look in. This needs a second calibration artifact — the query
second-moment matrix:

```python
from veloxquant_mlx.quantizers.a2ats import a2ats_query_second_moment

# collected_queries: [n_tokens, a2ats_sub_dim] query sub-vectors
h = a2ats_query_second_moment(collected_queries)

config = KVCacheConfig(
    method="a2ats", head_dim=128, a2ats_sub_dim=8,
    a2ats_codebook=..., a2ats_query_h=h,
)
```

With `a2ats_query_h` supplied you get the paper's actual query-aware
objective. Without it, a simpler cosine-similarity approximation runs
instead — see [Honest limitations](#honest-limitations) for what differs.

To turn query-awareness off entirely and use plain nearest-centroid
quantization:

```python
config = KVCacheConfig(method="a2ats", head_dim=128, a2ats_use_query_aware=False)
```

## Configuration reference

`KVCacheConfig` fields (when `method="a2ats"`):

| Parameter | Type | Description |
|---|---|---|
| `head_dim` | `int` | **Required.** Must be even and divisible by `a2ats_sub_dim` |
| `a2ats_window` | `int` | Recent-token window given exact position encoding. Default `128` |
| `a2ats_b` | `int` | Stand-in distance used for tokens outside the window. Default `2048` |
| `a2ats_codebook` | `mx.array \| np.ndarray \| None` | Trained codebook. Random if `None` (testing only) |
| `a2ats_codebook_bits` | `int` | Codebook size = `2**bits`. Default `8` |
| `a2ats_sub_dim` | `int` | Sub-vector width. Default `8` |
| `a2ats_use_query_aware` | `bool` | Query-aware assignment for the retrieval subset. Default `True` |
| `a2ats_query_h` | `mx.array \| np.ndarray \| None` | Query second-moment matrix, shape `[sub_dim, sub_dim]`. Enables the paper's exact objective |
| `a2ats_beta` | `float` | Query/reconstruction blend in `[0, 1]`, used only in the fallback path. Default `0.5` |
| `a2ats_retrieval_fraction` | `float` | Fraction of tokens routed to query-aware assignment, in `[0, 1]`. Default `0.20` |
| `a2ats_rope_base` | `float` | Position-encoding frequency base. Default `10000.0` |

Single-layer, no coordinator — `for_model` returns one `A2ATSKVCache` per
attention layer. No `.bits` attribute; stores and returns fp16 K/V directly.

## Measuring what you got

Every cache exposes running counters:

```python
cache = caches[0]
print(f"compression: {cache.compression_ratio:.1f}x")
print(f"effective bits/element: {cache.assigned_avg_bits:.2f}")
print(f"codebook overhead: {cache.codebook_bytes / 1024:.1f} KB")
print(f"tokens seen: {cache.tokens_seen}")
```

| Property | Meaning |
|---|---|
| `compression_ratio` | fp16 bytes ÷ compressed bytes (>1 means savings) |
| `compressed_key_bytes` / `compressed_value_bytes` | Actual stored bytes |
| `fp16_key_bytes` / `fp16_value_bytes` | What fp16 would have cost |
| `codebook_bytes` | One-time codebook overhead, amortized across tokens |
| `assigned_avg_bits` | Effective bits per element, excluding codebook |
| `tokens_seen` / `tokens_retrieved` | Cumulative counts, for observability |

## Honest limitations

Read this before deploying. None of it is hidden in a footnote.

**1. The windowing approximation genuinely costs accuracy.** It is not free,
and our benchmark says so in both geometries tested — 2.9x higher
attention-score error when relevant tokens are nearby, 8.3x when they're far.
Tokens *inside* the window are exact; everything outside pays. See
[Benchmark](#benchmark) for the full numbers.

**2. The cache can't see your real queries.** The mlx_lm cache protocol only
hands `update_and_fetch` keys and values, so "query-aware" here uses the
incoming key vector as a stand-in. Same approximation category as
[AMC-adapted](../algorithms/amc), [H2O](../algorithms/h2o), and
[SnapKV](../algorithms/snapkv) — not a new one, but it does mean the
query-awareness is weaker than the paper's.

**3. Without `a2ats_query_h`, query-aware assignment is a simplification.**
The paper minimizes a query-second-moment-weighted objective. That is
implemented and exact — but only runs when you supply `a2ats_query_h`.
Otherwise a cosine-similarity blend runs instead, which differs in two ways:
its query term applies the same bias to every token rather than coupling per
token, and `a2ats_beta` becomes sensitive to the scale of your key vectors.

**4. Query-aware assignment trades reconstruction accuracy for relevance.**
`a2ats_beta=1.0` is exactly plain quantization; anything lower deliberately
moves away from the lowest-reconstruction-error choice. Our benchmark shows
query-aware assignment with *higher* reconstruction error in every row — that
is expected, not a bug. The intended payoff is retrieval quality, which an
offline reconstruction benchmark cannot measure. **Don't read those rows as
"query-aware is worse."**

**5. Nothing is dropped.** Every token is quantized and kept. The
"retrieval fraction" only changes which centroid a token is matched
against — this is a compression method, not an eviction method.

**6. Per-step cost is proportional to total context, not new tokens.**
Whether a token counts as near or far depends on where you are *now*, so it
can't be precomputed once and frozen. Each decode step re-derives it across
the whole cache. Distant tokens skip the expensive path, so the constant is
small, but the pass itself is `O(total_tokens)`.

**7. Not validated on a trained model.** Our benchmark is offline and
synthetic. The paper's retrieval-accuracy and throughput numbers come from
real long-context workloads this repo doesn't have.

## Benchmark

`benchmark_scripts/benchmark_a2ats.py` (results in
`figures/a2ats/results.json`) sweeps sequence length across two geometries.
Error is measured on **attention scores**, which is what the approximation
actually affects.

| Geometry | Tokens measured | Windowed | Always-exact | Ratio |
|---|---|---:|---:|---:|
| `local_recency` | overall | 3.929 | 1.347 | **2.9x** |
| | inside window (16) | 8.248850 | 8.248850 | **1.000000x** |
| | outside window | 3.830 | 1.080 | 3.5x |
| `long_range_dependent` | overall | 8.384 | 1.014 | **8.3x** |
| | inside window (16) | 1.090521 | 1.090522 | **1.000001x** |
| | outside window | 8.762 | 1.006 | 8.7x |

**How to read this:** the "inside window" rows are identical to always-exact
(max difference `5e-07`) — the window really is exact. All the error comes
from outside it, and in a long sequence that's most of your tokens (~92% at
200 tokens, ~96% at 400). That is the trade you are making, stated plainly.

The gap doesn't come from a tunable being set badly: sweeping `a2ats_b` from
8 to 2048 never closes it. Collapsing every distant token's true position
into one stand-in value is simply lossy.

Deterministic in all non-timing fields, verified by diffing two runs.
Offline-synthetic — loads no model. **Not** a reproduction of the paper's
retrieval-accuracy or throughput numbers.

## How it works

Each call to `update_and_fetch` runs:

1. **Retrieval-set split** (query-aware path only) — the top
   `a2ats_retrieval_fraction` of tokens by proxy-query similarity, selected
   via [`dsa.MaxHeap`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/dsa/heap.py)
   top-k, the same pattern [AMC-adapted](../algorithms/amc) uses.
2. **Codebook assignment** — the retrieval set gets query-aware assignment
   (the paper's exact objective with `a2ats_query_h`, otherwise the cosine
   blend); everything else gets plain nearest-centroid, identical to
   [VecInfer](../algorithms/vecinfer)'s `quantize_vq`.
3. **Dequantization** reconstructs the key from its centroid. This is what
   gets **stored** — before any position encoding.
4. **Windowed position encoding, re-applied every step** across the whole
   cache against the current position. Tokens within `a2ats_window` get exact
   encoding at their own position; tokens outside are returned unencoded,
   with the constant stand-in rotation belonging on the query side instead
   (exposed as `A2ATSKVCache.far_query_rope`, since the cache never sees the
   query).
5. **Values** take a plain nearest-centroid path — no position encoding
   (values are never position-rotated), no retrieval preference. Same choice
   [ZipCache-adapted](../algorithms/zipcache)/[Palu](../algorithms/palu) make.

## Where it sits

| Method | Position-encoding handling | Query-aware | Selection axis |
|---|---|:---:|---|
| [VecInfer](../algorithms/vecinfer) | None — smooth + Hadamard only | No | Codebook only |
| [CommVQ-adapted](../algorithms/commvq) | Codebook-constrained, uniform | No | Codebook only |
| **A2ATS-adapted** | **Distance-gated: exact nearby, approximate far** | **Yes (subset)** | **Codebook + retrieval split** |

[CommVQ-adapted](../algorithms/commvq) solves position encoding by
constraining *what the codebook can represent*, treating every position
uniformly. A2ATS-adapted instead changes *when* exact encoding is paid for,
gated by distance — a different axis, and in principle composable with
CommVQ's approach (not attempted here).

## For contributors — paper fidelity

<details>
<summary>Correspondence to the source paper, and where this port deviates</summary>

*Inspired by* ["A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu,
Wu, Zhou, Liu, Xue, Li — **ACL 2025 Findings**)](https://aclanthology.org/2025.findings-acl.644/).
This is **A2ATS-adapted (VeloxQuant-MLX implementation)**, not a faithful
port. It is a normal-track method — a live-verified peer-reviewed venue, no
exception needed (unlike [AMC-adapted](../algorithms/amc) or
[NestedKV-adapted](../algorithms/nestedkv)).

**Windowed RoPE (Eq. 11–12).** Far keys are returned **unrotated**
(`k̃_i = k_i`, Eq. 12); the constant `R_b` encoding "far" relative position
rides on the **query** (Eq. 11, `u_ij = q_i R_b k_j^T`). This decoupling is
what makes a shared codebook viable across inputs (§3.1, Observation 2).
`a2ats_window` is `w`; `a2ats_b` is `b` — independent knobs, as in the
paper's §5.1 (`w=64`, `b=2048`).

**Query-aware VQ (Eq. 13–18).** `a2ats_h_weighted_assignment` minimizes
`(k̃ − c) H (k̃ − c)ᵀ` where `H = E[q̃ᵀq̃]`, computed exactly via the
Eq. (15)–(18) Cholesky identity — not an approximation. Requires
`a2ats_query_h`. The cosine blend used otherwise is a *substitute*: its
cosine term is a constant per-centroid bias rather than a per-token coupling,
and `beta` is scale-dependent.

**Per-step re-rotation.** Eq. (11) makes near/far a function of the advancing
decode query, so the split can't be baked into stored keys — doing so freezes
each token's class at write time. This port stores pre-RoPE keys and
re-applies windowed RoPE each step, costing an `O(total_tokens)` pass.

**Not implemented:**

- No CUDA kernel fusion; pure MLX throughout.
- No automatic codebook or `H` calibration — `a2ats_query_second_moment`
  computes `H` from queries you collect, but nothing collects them for you.
- No query-side `R_b` inside the cache. `far_query_rope` is exposed, but
  composing Eq. (11)'s two halves is left to callers with real query access.
- No composition with [CommVQ-adapted](../algorithms/commvq)'s
  RoPE-commuting codebook constraint.
- No trained-model perplexity/throughput/retrieval-accuracy benchmark.

**Evidence.** 67 tests across
`tests/quantizers/test_a2ats_rope.py` (17),
`tests/quantizers/test_a2ats.py` (19), and
`tests/cache/test_a2ats_cache.py` (31):

- `test_windowed_rope_outside_window_returns_key_unrotated` — far keys equal
  the *input*, not merely differ from exact RoPE. The weaker form let a real
  bug pass ([#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)).
- `test_windowed_rope_far_tokens_are_position_independent` — far keys carry
  no positional information, the §3.1 shared-codebook premise.
- `test_far_query_rope_b_is_independent_of_window` /
  `test_far_query_rope_reconstructs_paper_attention_score` — `b` is a real
  separate knob, and the halves compose back into Eq. (11).
- `test_token_rotation_updates_as_decode_position_advances` — near/far tracks
  the advancing query rather than freezing at write time.
- `test_h_weighted_assignment_matches_bruteforce_eq14` /
  `test_h_identity_reduces_to_plain_nearest_centroid` /
  `test_h_weighted_differs_from_plain_vq_under_anisotropic_h` — the Cholesky
  route computes Eq. (14) exactly, reduces to plain VQ when `H ∝ I` (§3.2's
  premise), and diverges under anisotropic `H`.
- `test_window_zero_always_unrotated` /
  `test_window_exceeds_seqlen_always_exact` — both degradation boundaries.
- Config validation, byte accounting, determinism across mixed
  prefill+decode, `for_model` propagation, factory dispatch.

</details>

## See also

- [Calibration guide](../guides/calibration)
- [VecInfer](../algorithms/vecinfer) — uniform exact position encoding
- [CommVQ-adapted](../algorithms/commvq) — codebook-constrained alternative
- [Algorithm overview](../algorithms/overview)

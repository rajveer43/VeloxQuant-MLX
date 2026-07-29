---
id: a2ats
title: A2ATS-adapted
sidebar_label: A2ATS-adapted
slug: /algorithms/a2ats
---

# A2ATS-adapted

A2ATS-adapted compresses the KV cache with vector quantization, and skips the
position-encoding work for tokens far behind the one you're currently
generating. Recent tokens are handled exactly; distant ones get a cheap
approximation. It suits long contexts where the tokens that matter are usually
nearby — chat histories, code completion, running summaries.

:::warning[Needs a calibration pass]
Like [VecInfer](../algorithms/vecinfer), [CommVQ-adapted](../algorithms/commvq),
and [Palu](../algorithms/palu), this method needs a codebook trained on
representative data. Without one it falls back to a random codebook that exists
only so the plumbing tests can run, and quality will be poor. See
[Calibration](#calibration-one-time-setup).
:::

:::note[Memory, not speed]
On Apple Silicon the win is memory footprint rather than decode throughput. The
paper's speedup numbers assume a fused CUDA kernel this port doesn't have — the
same disclaimer as every VQ-family method here.
:::

:::danger[The savings are modelled, not yet realised in RAM]
`compression_ratio` reports what the quantized codes *would* cost. It is not
the resident memory of your process. `update_and_fetch` dequantizes each token
back to fp16 and stores that in the parent buffer, because the near/far split
has to be recomputed against the live decode position every step
([How it works](#how-it-works), step 4). So today the cache holds fp16 and the
counters describe a saving the storage layer doesn't take yet.

What you do get is a faithful measurement of the accuracy cost and the
achievable footprint of this configuration. Treat the numbers as sizing for a
codes-resident backend, not as RAM you'll see in Activity Monitor. If you need
an actual reduction in resident memory right now, use
[VecInfer](../algorithms/vecinfer) or [TurboQuant RVQ](../algorithms/rvq).
:::

## Quick start

Work through [Calibration](#calibration-one-time-setup) first — without a
trained codebook this produces near-random quantization. Once you have
`a2ats_codebook.npz`:

```python
import mlx_lm
import numpy as np
import mlx.core as mx
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# head_dim must match the model. Llama-3.2-3B uses 128; read it off the config
# rather than guessing, since a mismatch raises at construction time.
head_dim = model.args.head_dim or (model.args.hidden_size // model.args.num_attention_heads)

data = np.load("a2ats_codebook.npz")

config = KVCacheConfig(
    method="a2ats",
    head_dim=head_dim,
    a2ats_window=128,     # recent tokens get exact position encoding
    a2ats_codebook=mx.array(data["codebook"]),
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Summarise this conversation so far.",
    max_tokens=200,
)
```

The defaults for everything else are the paper's own values, and are reasonable
starting points.

`head_dim` must be even and divisible by `a2ats_sub_dim` (default `8`); both
are checked when the cache is built, so a wrong value fails immediately rather
than silently degrading. `head_dim=128` satisfies this, as does `64`.

## Should you use this?

Good fit:

- Long contexts where relevant tokens are usually recent — conversations,
  iterative editing, code in the current file
- You can run a one-time offline calibration pass
- Memory is your constraint, not latency

Poor fit:

- Genuine long-range lookups, such as retrieval over a large document or "what
  did the contract say on page 3". The approximation costs the most exactly
  there: our benchmark measures 8.3x higher attention-score error in that
  geometry, versus 2.9x for the local case.
- You can't calibrate. Try [TurboQuant RVQ](../algorithms/rvq), which needs no
  calibration.
- You want uniform accuracy regardless of distance.
  [VecInfer](../algorithms/vecinfer) and [CommVQ-adapted](../algorithms/commvq)
  apply exact position encoding everywhere.

### Tuning `a2ats_window`

This is the main dial. Tokens inside the window are handled exactly, bit-identical
to never approximating at all, so the window is the fraction of your context
you're paying full price for.

| `a2ats_window` | Effect |
|---|---|
| Larger (e.g. 512) | More accuracy, less savings |
| Smaller (e.g. 64) | More savings, more error on tokens that fall outside |
| `>= context length` | Degrades to always-exact — no approximation at all |
| `<= 0` | Everything approximated (not recommended) |

If you know roughly how far back your queries reach, set the window to cover it.

## Calibration (one-time setup)

Train a codebook on a representative sample of your model's key activations and
persist it:

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
actually look in. This needs a second calibration artifact, the query
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

With `a2ats_query_h` supplied you get the paper's actual query-aware objective.
Without it, a simpler cosine-similarity approximation runs instead; see
[Limitations](#limitations) for what differs.

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

Single-layer, no coordinator: `for_model` returns one `A2ATSKVCache` per
attention layer. There is no `.bits` attribute; the cache stores and returns
fp16 K/V directly.

## Measuring compression

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
| `compression_ratio` | fp16 bytes ÷ code bytes, for keys and values together |
| `compressed_key_bytes` / `compressed_value_bytes` | What the codes occupy |
| `fp16_key_bytes` / `fp16_value_bytes` | What fp16 would have cost |
| `codebook_bytes` | One-time codebook overhead, amortized across tokens |
| `assigned_avg_bits` | Effective bits per element, excluding codebook |
| `tokens_seen` / `tokens_retrieved` | Cumulative counts, for observability |

All of these are modelled sizes, not resident memory — see the callout at the
top of the page.

### What the defaults give you

Compression is fixed by your configuration, not by the data, so you can compute
it up front: each token costs `head_dim / a2ats_sub_dim` indices of
`a2ats_codebook_bits` each, which works out to
`assigned_avg_bits = a2ats_codebook_bits / a2ats_sub_dim`. Keys and values are
accounted identically.

| `a2ats_sub_dim` | `a2ats_codebook_bits` | Bits/element | vs fp16 | Codebook |
|---|---|---:|---:|---:|
| `8` (default) | `8` (default) | 1.00 | 16x | 4 KB |
| `4` | `8` | 2.00 | 8x | 2 KB |
| `8` | `4` | 0.50 | 32x | 0.25 KB |

The codebook is a fixed cost, stored fp16 as `2**bits × sub_dim` entries. It's
per cache and independent of context length, so it amortizes to nothing on any
real sequence. Higher ratios mean coarser centroids and more reconstruction
error, so raising `sub_dim` past the default trades quality for footprint.

## Troubleshooting

**`head_dim=... must be even`** — RoPE rotates dimensions in pairs. Read
`head_dim` off the model config rather than hardcoding it.

**`head_dim=... not divisible by a2ats_sub_dim`** — the head dimension is split
into equal sub-vectors. With the default `a2ats_sub_dim=8`, `head_dim` must be a
multiple of 8. Either fix `head_dim` or pick a `sub_dim` that divides it.

**Codebook shape errors, or output quality far worse than expected** — the
`sub_dim` and `bits` you calibrated with must match the config you run with. A
codebook trained at `sub_dim=8` is not usable at `sub_dim=4`. If you didn't pass
`a2ats_codebook` at all, you're on the random fallback and output will look
broken; that path exists only so shape tests can run.

**`a2ats_query_h` rejected for its shape** — it must be `[sub_dim, sub_dim]`,
matching your configured `a2ats_sub_dim`.

**Output is fine but you see no memory savings** — expected today. See the
callout at the top of the page.

**Decode feels slower as context grows** — also expected: the windowed
re-rotation is an `O(total_tokens)` pass per step. See the last limitation
below.

## Limitations

Worth reading before you deploy this.

**The windowing approximation costs accuracy.** Our benchmark shows it in both
geometries tested: 2.9x higher attention-score error when relevant tokens are
nearby, 8.3x when they're far. Tokens inside the window are exact; everything
outside pays. See [Benchmark](#benchmark) for the full numbers.

**The cache can't see your real queries.** The mlx_lm cache protocol only hands
`update_and_fetch` keys and values, so "query-aware" here uses the incoming key
vector as a stand-in. This is the same approximation category as
[AMC-adapted](../algorithms/amc), [H2O](../algorithms/h2o), and
[SnapKV](../algorithms/snapkv), so it isn't a new problem, but it does mean the
query-awareness is weaker than the paper's.

**Without `a2ats_query_h`, query-aware assignment is a simplification.** The
paper minimizes a query-second-moment-weighted objective. That is implemented
exactly, but only runs when you supply `a2ats_query_h`. Otherwise a
cosine-similarity blend runs instead, which differs in two ways: its query term
applies the same bias to every token rather than coupling per token, and
`a2ats_beta` becomes sensitive to the scale of your key vectors.

**Query-aware assignment trades reconstruction accuracy for relevance.**
`a2ats_beta=1.0` is exactly plain quantization; anything lower deliberately
moves away from the lowest-reconstruction-error choice. Our benchmark shows
query-aware assignment with higher reconstruction error in every row, which is
expected rather than a bug. The intended payoff is retrieval quality, and an
offline reconstruction benchmark can't measure that. Don't read those rows as
"query-aware is worse".

**Nothing is dropped.** Every token is quantized and kept. The retrieval
fraction only changes which centroid a token is matched against. This is a
compression method, not an eviction method.

**Per-step cost is proportional to total context, not new tokens.** Whether a
token counts as near or far depends on where you are now, so it can't be
precomputed once and frozen. Each decode step re-derives it across the whole
cache. Distant tokens skip the expensive path, so the constant is small, but the
pass itself is `O(total_tokens)`.

**Not validated on a trained model.** Our benchmark is offline and synthetic.
The paper's retrieval-accuracy and throughput numbers come from real
long-context workloads this repo doesn't have.

## Benchmark

`benchmark_scripts/benchmark_a2ats.py` (results in `figures/a2ats/results.json`)
sweeps sequence length across two geometries. Error is measured on attention
scores, which is what the approximation actually affects.

| Geometry | Tokens measured | Windowed | Always-exact | Ratio |
|---|---|---:|---:|---:|
| `local_recency` | overall | 3.929 | 1.347 | **2.9x** |
| | inside window (16) | 8.248850 | 8.248850 | **1.000000x** |
| | outside window | 3.830 | 1.080 | 3.5x |
| `long_range_dependent` | overall | 8.384 | 1.014 | **8.3x** |
| | inside window (16) | 1.090521 | 1.090522 | **1.000001x** |
| | outside window | 8.762 | 1.006 | 8.7x |

The "inside window" rows are identical to always-exact, to a maximum difference
of `5e-07`, so the window really is exact. All the error comes from outside it,
and in a long sequence that's most of your tokens: roughly 92% at 200 tokens and
96% at 400. That's the trade you're making.

The gap isn't a tunable set badly. Sweeping `a2ats_b` from 8 to 2048 never
closes it — collapsing every distant token's true position into one stand-in
value is simply lossy.

The benchmark is deterministic in all non-timing fields, verified by diffing two
runs. It's offline-synthetic and loads no model, so it does not reproduce the
paper's retrieval-accuracy or throughput numbers.

## How it works

Each call to `update_and_fetch` runs:

1. **Retrieval-set split** (query-aware path only) — the top
   `a2ats_retrieval_fraction` of tokens by proxy-query similarity, selected via
   [`dsa.MaxHeap`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/dsa/heap.py)
   top-k, the same pattern [AMC-adapted](../algorithms/amc) uses.
2. **Codebook assignment** — the retrieval set gets query-aware assignment (the
   paper's exact objective with `a2ats_query_h`, otherwise the cosine blend);
   everything else gets plain nearest-centroid, identical to
   [VecInfer](../algorithms/vecinfer)'s `quantize_vq`.
3. **Dequantization** reconstructs the key from its centroid. This is what gets
   stored, before any position encoding.
4. **Windowed position encoding, re-applied every step** across the whole cache
   against the current position. Tokens within `a2ats_window` get exact encoding
   at their own position; tokens outside are returned unencoded, with the
   constant stand-in rotation belonging on the query side instead (exposed as
   `A2ATSKVCache.far_query_rope`, since the cache never sees the query).
5. **Values** take a plain nearest-centroid path: no position encoding, since
   values are never position-rotated, and no retrieval preference. This is the
   same choice [ZipCache-adapted](../algorithms/zipcache) and
   [Palu](../algorithms/palu) make.

## Where it sits

| Method | Position-encoding handling | Query-aware | Selection axis |
|---|---|:---:|---|
| [VecInfer](../algorithms/vecinfer) | None — smooth + Hadamard only | No | Codebook only |
| [CommVQ-adapted](../algorithms/commvq) | Codebook-constrained, uniform | No | Codebook only |
| **A2ATS-adapted** | **Distance-gated: exact nearby, approximate far** | **Yes (subset)** | **Codebook + retrieval split** |

[CommVQ-adapted](../algorithms/commvq) solves position encoding by constraining
what the codebook can represent, treating every position uniformly.
A2ATS-adapted instead changes when exact encoding is paid for, gated by
distance. That's a different axis, and in principle composable with CommVQ's
approach, though we haven't attempted it.

## For contributors — paper fidelity

<details>
<summary>Correspondence to the source paper, and where this port deviates</summary>

*Inspired by* ["A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu, Wu,
Zhou, Liu, Xue, Li — **ACL 2025 Findings**)](https://aclanthology.org/2025.findings-acl.644/).
This is A2ATS-adapted (VeloxQuant-MLX implementation), not a faithful port. It's
a normal-track method — a live-verified peer-reviewed venue, no exception needed
(unlike [AMC-adapted](../algorithms/amc) or
[NestedKV-adapted](../algorithms/nestedkv)).

**Windowed RoPE (Eq. 11–12).** Far keys are returned unrotated (`k̃_i = k_i`,
Eq. 12); the constant `R_b` encoding "far" relative position rides on the query
(Eq. 11, `u_ij = q_i R_b k_j^T`). This decoupling is what makes a shared
codebook viable across inputs (§3.1, Observation 2). `a2ats_window` is `w` and
`a2ats_b` is `b`, independent knobs as in the paper's §5.1 (`w=64`, `b=2048`).

**Query-aware VQ (Eq. 13–18).** `a2ats_h_weighted_assignment` minimizes
`(k̃ − c) H (k̃ − c)ᵀ` where `H = E[q̃ᵀq̃]`, computed exactly via the
Eq. (15)–(18) Cholesky identity rather than approximated. Requires
`a2ats_query_h`. The cosine blend used otherwise is a substitute: its cosine
term is a constant per-centroid bias rather than a per-token coupling, and
`beta` is scale-dependent.

**Per-step re-rotation.** Eq. (11) makes near/far a function of the advancing
decode query, so the split can't be baked into stored keys — doing so freezes
each token's class at write time. This port stores pre-RoPE keys and re-applies
windowed RoPE each step, costing an `O(total_tokens)` pass.

**Not implemented:**

- No CUDA kernel fusion; pure MLX throughout.
- No automatic codebook or `H` calibration. `a2ats_query_second_moment` computes
  `H` from queries you collect, but nothing collects them for you.
- No query-side `R_b` inside the cache. `far_query_rope` is exposed, but
  composing Eq. (11)'s two halves is left to callers with real query access.
- No composition with [CommVQ-adapted](../algorithms/commvq)'s RoPE-commuting
  codebook constraint.
- No trained-model perplexity/throughput/retrieval-accuracy benchmark.

**Evidence.** 67 tests across `tests/quantizers/test_a2ats_rope.py` (17),
`tests/quantizers/test_a2ats.py` (19), and `tests/cache/test_a2ats_cache.py`
(31):

- `test_windowed_rope_outside_window_returns_key_unrotated` — far keys equal the
  *input*, not merely differ from exact RoPE. The weaker form let a real bug
  pass ([#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)).
- `test_windowed_rope_far_tokens_are_position_independent` — far keys carry no
  positional information, the §3.1 shared-codebook premise.
- `test_far_query_rope_b_is_independent_of_window` /
  `test_far_query_rope_reconstructs_paper_attention_score` — `b` is a real
  separate knob, and the halves compose back into Eq. (11).
- `test_token_rotation_updates_as_decode_position_advances` — near/far tracks the
  advancing query rather than freezing at write time.
- `test_h_weighted_assignment_matches_bruteforce_eq14` /
  `test_h_identity_reduces_to_plain_nearest_centroid` /
  `test_h_weighted_differs_from_plain_vq_under_anisotropic_h` — the Cholesky
  route computes Eq. (14) exactly, reduces to plain VQ when `H ∝ I` (§3.2's
  premise), and diverges under anisotropic `H`.
- `test_window_zero_always_unrotated` /
  `test_window_exceeds_seqlen_always_exact` — both degradation boundaries.
- Config validation, byte accounting, determinism across mixed prefill+decode,
  `for_model` propagation, factory dispatch.

</details>

## See also

- [Calibration guide](../guides/calibration)
- [VecInfer](../algorithms/vecinfer) — uniform exact position encoding
- [CommVQ-adapted](../algorithms/commvq) — codebook-constrained alternative
- [Algorithm overview](../algorithms/overview)

---
id: gear
title: GEAR
sidebar_label: GEAR
slug: /algorithms/gear
---

# GEAR

**Method id:** `gear` · *Inspired by* [GEAR: An Efficient KV Cache Compression
Recipe for Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527)
(Kang, Zhang et al.) — **GEAR-adapted (VeloxQuant-MLX implementation)**, not a
faithful port. See [When it helps — and when it actively doesn't](#when-it-helps--and-when-it-actively-doesnt)
below before you reach for it.

:::info[What GEAR is for]
Every other bit-width method in this library picks a bit-width and lives with
the quantization error it causes. GEAR is different: it's an **add-on** that
sits on top of an ultra-low-bit quantizer (2-bit, even 1-bit) and reconstructs
most of the accuracy that low bit-width would normally lose — so you can push
compression lower than usual without the quality hit. Use it when you want
"2-bit quality closer to 4-bit" rather than "the smallest possible cache."
:::

## The idea in one picture

Take one head's keys or values as a matrix `X`. Instead of just quantizing it
and accepting the error, GEAR decomposes it three ways:

```
X   ≈   Quant_b(X)     +     L · R          +          S
        └─ base ─┘            └ low-rank ┘             └ sparse ┘
     "most entries,        "the error's           "the few outlier
      compressed hard"      common pattern"        entries quantization
                                                     mangles the worst"
```

1. **Base** — quantize almost everything to an ultra-low bit-width (2–4 bits).
   This is where the bulk of the memory savings come from, and where most of
   the error is introduced.
2. **Low-rank residual** — the *error* from step 1 (`X` minus what the base
   quantizer reconstructs) is not random noise. It's structured: attention
   heads encode correlated information, so the error itself is well
   approximated by a small number of rank-1 patterns. GEAR captures that
   pattern in two small matrices `L` and `R` and adds it back.
3. **Sparse outliers** — a handful of entries (~0.5–1%) are true outliers even
   after the low-rank correction. GEAR stores those few values exactly and
   patches them back in.

The result: a cache that's still mostly ultra-low-bit, but with most of the
accuracy of a much higher bit-width, because the two small correction terms
recover what the base quantizer threw away.

## Quickstart

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(
    method="gear",
    head_dim=128,
    gear_bits=2,  # ultra-low base bit-width
    gear_rank=4,  # low-rank correction size — keep this small
    gear_sparse_fraction=0.005,  # top 0.5% of residual entries kept exact
    gear_group_size=32,
    gear_quantize_values=True,  # apply GEAR to values too (False = keys only)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Explain the theory of relativity in simple terms.",
    max_tokens=200,
)
```

**Config fields at a glance**

| Field | Default | What it controls |
|---|---|---|
| `gear_bits` | `2` | Base bit-width. Lower = more compression, more base error for GEAR to recover. |
| `gear_rank` | `None` (auto) | Size of the low-rank correction `L·R`. `None` picks a rank automatically from `gear_energy_threshold`; set explicitly (2–8) to control the memory/quality trade directly. **Keep this small** — see [the section below](#when-it-helps--and-when-it-actively-doesnt). |
| `gear_energy_threshold` | `0.90` | Used only when `gear_rank=None`: how much of the error's "energy" the auto-picked rank must capture. |
| `gear_sparse_fraction` | `0.01` | Fraction of residual entries kept as exact outliers. `0` disables the sparse term. |
| `gear_group_size` | `32` | Token/channel group size for the base quantizer. |
| `gear_quantize_values` | `True` | Set `False` to apply GEAR to keys only and leave values at fp16. |

## How it works, in the cache

The base quantizer follows the paper's own scheme, called **KCVT**: **keys are
quantized per-channel**, **values are quantized per-token** — the same
asymmetry [KIVI](../algorithms/kivi) uses, because the paper builds directly on
that observation about where each tensor's outlier structure lives. GEAR then
composes its low-rank + sparse correction on top of whichever axis is correct
for that tensor.

Per head, on every `update_and_fetch` call (the whole prompt at prefill, one
new token at each decode step):

1. Base-quantize with the KCVT-appropriate axis and dequantize, giving the
   base reconstruction.
2. Form the residual: `E = X − base_reconstruction`.
3. Truncated SVD of `E` gives the low-rank correction `L · R` (rank chosen by
   `gear_rank`, or automatically via `gear_energy_threshold`). Subtract it —
   what's left is the post-low-rank residual.
4. Keep the top `gear_sparse_fraction` of that residual by magnitude as the
   exact sparse correction `S`.
5. Reconstruct as `base + L·R + S` in fp16 and hand it to the underlying
   `mlx_lm` cache, so attention runs on the normal fp16 path.

## When it helps — and when it actively doesn't

This is the single most important thing to know before turning GEAR on, and it
comes from actually running it end-to-end on a real model, not just
synthetic data.

:::warning[GEAR needs a batch to amortize its overhead — single-token decode does not have one]
The low-rank correction costs `(N + D) × rank × 2` bytes, where `N` is however
many tokens that specific `update_and_fetch` call compressed. At **prefill**,
`N` is the whole prompt (hundreds to thousands of tokens), so that fixed cost
is amortized across all of them and the low-rank term is nearly free per
token. At **decode**, this wrapper compresses one new token at a time (`N=1`)
— see [Fidelity to the paper](#fidelity-to-the-paper--whats-adapted-and-why)
for why there's no streaming buffer to batch decode tokens together — so the
*same* fixed-size `L`/`R` overhead is paid **for a single token**, and it is
bigger than that one token's entire fp16 cost. There is no rank small enough
to fix this for `N=1`; it is a structural property of the low-rank term, not a
tuning problem.

Measured on Mistral-7B-Instruct-v0.3 (`head_dim=128`), `gear_bits=2`,
`gear_rank=4`:

| Workload shape | Key-cache compression vs fp16 |
|---|---|
| Long prompt (~800 tokens), short generation (8 tokens) — **prefill-dominated** | **3.96×** |
| Short prompt, longer generation (60 tokens) — **decode-dominated** | **0.81×** (worse than fp16) |

**Rule of thumb:** GEAR is a net win when your workload is prefill-heavy
(long-context understanding, summarization, RAG over a long retrieved
context, one-shot classification) and a net loss when it's decode-heavy (long
free-form generation, extended chat, chain-of-thought). If you mostly generate
long outputs, prefer a bit-width method without this per-call overhead —
[KIVI](../algorithms/kivi) or [VecInfer](../algorithms/vecinfer) — or use GEAR
with `gear_rank=0, gear_sparse_fraction=0` (pure base quantizer, no
correction, no per-token overhead) for the decode-heavy portion.
:::

Beyond the prefill/decode split, three more factors move the needle:

- **Rank.** Bigger rank recovers more accuracy but costs more bytes per call.
  The paper finds rank 4 (prefill) is already enough for near-lossless
  accuracy — going much higher has diminishing quality returns and a real
  memory cost.
- **Head dimension.** Small head dims (64 and below, common on smaller/newer
  models with high head counts) leave less room for the low-rank term to pay
  for itself even at prefill, because `D` is part of the fixed overhead too.
  Larger head dims (128, the paper's own setup) amortize better.
- **`gear_quantize_values=False`.** If only keys need the quality recovery,
  turning off value compression halves the overhead for the same accuracy
  gain on keys.

## Fidelity to the paper — what's adapted and why

This is not a faithful port. Differences, and why they exist:

- **KCVT base backbone.** ✅ Matches the paper: per-channel keys, per-token
  values (as of this implementation — earlier versions of this library used a
  generic per-token quantizer for both, which understated the paper's actual
  design).
- **No streaming buffer.** ❌ Not ported. The paper computes its low-rank
  correction on a buffer of `nb=20` newly-generated tokens at a time during
  decode (§3, "Streaming Buffer") specifically so the low-rank overhead is
  amortized across 20 tokens instead of 1. This implementation computes the
  correction on whatever batch `update_and_fetch` receives, which is 1 token
  at decode — this is exactly why the decode-time regression documented above
  exists. Porting a real token buffer would fix it; that's tracked as future
  work, not silently worked around.
- **No fused CUDA dequant-attend kernel.** ❌ Not ported. The paper fuses
  dequantization into the attention kernel so the *working set during
  attention* stays compressed. This implementation reconstructs to fp16 and
  calls MLX's standard attention — the *stored* cache shrinks (when the
  prefill/decode math above is favorable), but the peak memory *during*
  attention is the full fp16 tensor either way.
- **Per-call residual SVD, not a calibration pass.** The paper and this
  implementation agree here: no separate calibration step, no dataset
  dependency — the low-rank correction is computed fresh from whatever data
  is in front of it, which is why GEAR works "out of the box" like the other
  zero-calibration methods in this library.

## Evidence

- **23 unit tests** in `veloxquant_mlx/tests/quantizers/test_gear.py` (19) and
  `veloxquant_mlx/tests/cache/test_gear_cache.py` (11), covering: GEAR beats
  base-quant-alone (the core claim), low-rank-alone and sparse-alone each help,
  `rank=0, sparse=0` collapses exactly to plain base quantization, a
  known-rank residual is recovered to numerical precision, sparse selection
  picks true outliers, byte-accounting ordering and component sums, the KCVT
  axis is actually different per tensor kind (not a cosmetic label), the
  single-token-decode-exceeds-fp16 property above, and determinism.
- **Offline synthetic benchmark**
  (`benchmark_scripts/benchmark_gear.py` → `results_gear.json`, run against
  the KCVT backbone) measures 11–38% MSE improvement over base-quant-alone
  across 2/3/4-bit configs, for both the key (channel) and value (token) axes
  independently.
- **Real end-to-end generation**, `mlx_lm.generate` on
  `mlx-community/Mistral-7B-Instruct-v0.3-4bit`: produces coherent output at
  `gear_bits=2`; the prefill-vs-decode compression numbers in the table above
  are from this run, not synthetic data.

## When to use it

| Method | Reconstruction | Compresses / recovers via | Best fit |
|---|---|---|---|
| KIVI-2bit | group quant, no correction | fixed 2-bit packing | steady compression at any sequence shape |
| CacheGen | identical to group quant | entropy coding of deltas | storage, not quality |
| **GEAR** | base quant **+ error feedback** | low-rank residual + sparse outliers | **prefill-heavy** workloads at ultra-low bits |

Reach for GEAR when: your workload is prefill-dominated (long context in,
short answer out), you want lower bit-widths than usual without the usual
accuracy cliff, and your model's head dimension is reasonably large (≥128).

Reach for something else when: you're doing long free-form generation or
extended chat (decode-dominated) — GEAR's per-token overhead will make your
cache *larger* than fp16 in that regime, not smaller. [KIVI](../algorithms/kivi)
or [VecInfer](../algorithms/vecinfer) give steady compression regardless of
prefill/decode balance.

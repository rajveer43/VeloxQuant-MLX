---
id: age-tiered
title: AgeTieredKV
sidebar_label: AgeTieredKV
slug: /algorithms/age-tiered
---

# AgeTieredKV

AgeTieredKV is this library's own method, not a port of a published paper.
It exists to answer a specific research question raised in
[issue #256](https://github.com/rajveer43/VeloxQuant-MLX/issues/256): **does
KV-cache precision need to be uniform, or can a token's position/age be used
to assign it a coarser bit-width without a worse quality/memory tradeoff
than a uniform budget-matched baseline?**

:::info[Where this sits relative to existing methods]
[KIVI](./kivi) already treats recency specially — a fixed-length fp16
residual window for the newest tokens, with everything older quantized at
one fixed bit-width. That is a **two-level** scheme (fp16 vs. quantized)
gated purely by age. [AMC](./amc) already implements a **three-tier**
discrete precision ladder, but tiers by per-token activation *saliency*, not
position. AgeTieredKV combines the two: three quantization tiers, gated
purely by age.
:::

## How it works

Every token is retained — like AMC, and unlike the eviction-family methods
(H2O, SnapKV, PyramidKV, ...) — only its bit-width changes as it ages. A
token's age is `current_position - token_position` (age `0` = written this
step). Two boundaries split age into three tiers:

1. **Recent** (`age < age_recent_boundary`) — `age_bits_recent` (default 8-bit)
2. **Mid** (`age_recent_boundary <= age < age_mid_boundary`) — `age_bits_mid` (default 4-bit)
3. **Old** (`age >= age_mid_boundary`) — `age_bits_old` (default 2-bit)

Each tier's quantization reuses this library's existing asymmetric min/max
group quantizer — the same primitive KIVI and AMC's Low/Mid tiers both use,
not a new numeric scheme:

```
zero  = min(group)
scale = (max(group) - min(group)) / (2**b - 1)
q     = round((group - zero) / scale)      # uint, [0, 2**b - 1]
recon = q * scale + zero
```

Unlike AMC, there is no rank masking here — the only varying signal is
bit-width. On every step, the whole per-(batch, head) buffer is re-tiered
from its current age and re-quantized wherever a token has just crossed a
boundary, so a token is never coarser than its current age entitles — the
same "flush on boundary crossing" model KIVI uses for its residual window,
extended to three tiers instead of two.

AgeTieredKV is fully deterministic — pure min/max group quantization, no
codebook, no RNG — so it adds no run-to-run variance.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="age_tiered",
    age_recent_boundary=128,  # age < this -> 8-bit
    age_mid_boundary=1024,  # age < this -> 4-bit; older -> 2-bit
    age_bits_recent=8,
    age_bits_mid=4,
    age_bits_old=2,
    age_group_size=32,  # min/max group size, shared across tiers
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=120)
```

Or from the CLI, via [`veloxquant profile`](../guides/profiling#the-veloxquant-profile-cli):

```bash
veloxquant profile \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --method age_tiered \
  --set age_recent_boundary=128 --set age_mid_boundary=1024 \
  --prompt "..." --max-tokens 120
```

## Comparing uniform vs. adaptive

Issue #256's proposed experiment is: compare a uniform INT4 cache against an
adaptive recent→INT8 / mid→INT4 / old→INT2 cache, matched so both land near
the same *average* bit-width, and measure perplexity, generation quality,
memory, latency, and attention reconstruction error. A uniform INT4
equivalent is `KVCacheConfig(method="kivi", bit_width_inlier=4)` (or
`age_tiered` with all three tier bit-widths set equal); the adaptive
configuration is `age_tiered` with its default 8/4/2 split. Both configs
plug into the same `for_model` + `generate` pattern above, so any of this
repo's existing perplexity/latency/memory benchmark harnesses
(`veloxquant_mlx/benchmarks/model_kv_benchmark.py`) apply unchanged to both
sides of the comparison.

### Measured result: the default 8/4/2 split loses to uniform INT4

:::danger[Adaptive tiering did not beat uniform INT4 in this measurement]
Apple M4, `mlx-community/Llama-3.2-1B-Instruct-4bit`, a ~511-token prefill
(`age_recent_boundary=32`, `age_mid_boundary=256`), perplexity computed with
each configuration's cache wired directly into the forward pass. Source:
`benchmark_scripts/benchmark_age_tiered.py`, `age_tiered_benchmark_results.json`.

| Config | Recent/Mid/Old bits | Realized avg bits | Perplexity | Attention-recon MSE |
|---|---|---|---|---|
| fp16 baseline | 16/16/16 | 16.0 | 2.20 | 0.0 |
| uniform-int4 (KIVI) | 4/4/4 | 4.0 | 2.23 | 0.028 |
| **age-tiered (default)** | **8/4/2** | **2.12** | **13.20** | **1.13** |
| age-tiered (mild) | 8/4/4 | ~4.5 | 1.23<sup>†</sup> | — |

<sup>†</sup> measured on a shorter 511-token prefill in a separate isolated
run against the same fp16 baseline (ppl 1.23); the two prefills used
different text, so this row is not directly comparable to the ppl column
above — it is included to show 8/4/4 tracks uniform-4bit closely (ppl 1.2334
vs. 1.2334, both ~fp16), not to claim a specific ppl delta against the 2.20
baseline in the same row.

**This directly answers issue #256's research question, and the answer is
not the one a naive average-bit-width comparison predicts.** The default
8/4/2 split has a *lower* average bit-width than uniform-int4 (2.12 vs 4.0)
but is dramatically worse — ~6× the perplexity, ~40× the reconstruction
error. Isolating the effect (a controlled sweep at fixed
`age_recent_boundary`/`age_mid_boundary`, varying only the OLD tier's
bit-width) shows why: quantization error does not scale linearly with bits.
On synthetic key-scale data, mean-squared quantization error was ~0.0002 at
8-bit, ~0.054 at 4-bit, and ~1.36 at 2-bit — roughly a 25× jump from 4-bit to
2-bit, not the ~4× a linear bits-vs-error intuition would suggest. **A
2-bit tier for old tokens is not "a bit worse than 4-bit" — it is a
qualitatively different, much steeper regime**, and spending a few tokens'
worth of extra bit-width there costs far more accuracy than the memory
saved is worth, at least at this model scale and this age-boundary
configuration.

The 8/4/4 configuration (no 2-bit tier at all) essentially matches
uniform-int4's quality while still being *nominally* adaptive — but at that
point the "adaptive" part of the scheme is doing very little, since 4-bit
is 4-bit whether or not it is gated by age. **The practical takeaway: this
implementation does not find evidence that position/age-gated multi-tier
precision beats a well-chosen uniform bit-width, once the tiers are matched
for realized error rather than nominal bit-width** — the 2-bit tier's cliff
dominates any savings from the higher-precision recent/mid tiers, on this
model and this prompt. Whether this holds on larger models, different age
boundaries, or values (not just keys) is not established here — reproduce
with `benchmark_scripts/benchmark_age_tiered.py` on your own model/config
before generalizing further.
:::

## Honest scope

:::warning[Compression is accounting-only, same as every other method here]
- Like every method in this repo, K/V are quantized-then-dequantized to
  fp16 before SDPA — `compression_ratio` and `age_tiered_bytes` measure
  compression fidelity (byte accounting for K + V given each token's tier),
  not runtime RSS saved. See the shared `ACCOUNTING_NOTE` in `serve.py` /
  `cli/profile.py`.
- Re-quantizing the whole buffer on every step (rather than only the newly
  boundary-crossing slice) is `O(n)` in cache length per call, the same
  cost KIVI's own flush-on-boundary model pays for its two-tier scheme,
  scaled to three tiers. This has not been optimized for very long contexts.
- No Metal-specific kernel exists for this method yet (unlike KIVI's fused
  group-quant kernel) — expect no throughput win over fp16, and possibly a
  cost, on Apple Silicon.
- `is_trimmable()` is `False` — `mlx_lm.server`'s prompt-cache trimming is
  unavailable for the same reason it's unavailable for AMC and the
  eviction-family methods (#152): `trim()` would roll back offset
  bookkeeping without reverting the internal per-token tier state.
:::

See also: [KIVI](./kivi) — the two-tier (fp16 residual vs. one fixed
bit-width) special case this method generalizes.

See also: [AMC](./amc) — the three-tier precision ladder this method
borrows its tier-count/quantize machinery from, tiering by saliency instead
of age.

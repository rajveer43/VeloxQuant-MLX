---
id: adakv
title: AdaKV-proxy — Per-Head Adaptive Bit Allocation
sidebar_label: AdaKV-proxy
slug: /algorithms/adakv
---

# AdaKV-proxy — Per-Head Adaptive Bit Allocation

**Method id:** `adakv` · **Available since:** v0.12.0 · **Calibration:** none —
works on any model immediately · *Proxy adaptation of* [Ada-KV
(arXiv:2407.11550, NeurIPS 2025)](https://arxiv.org/abs/2407.11550), not a
faithful port — see [Adaptation notes](#adaptation-notes).

## The idea in one minute

A language model keeps notes on every word it has read so it doesn't have to
reread them. Those notes are the **KV cache**, and in a long conversation they
can take more memory than the model itself. Storing them with fewer bits per
number — **quantization** — shrinks them, at some cost in accuracy.

The models do this reading through many parallel **attention heads**, each
watching for a different kind of pattern. The usual approach, [KIVI](./kivi),
gives every head the same number of bits. But heads aren't equally fragile:
squeeze some and nothing happens, squeeze others and quality drops.

AdaKV-proxy spends the same average budget and divides it unevenly. Fragile
heads get 4 bits, robust ones get 2, and the average still lands on the target
you asked for. Nothing is deleted — every token is kept, just stored at
different precision depending on which head it belongs to.

If it helps: same total budget as KIVI, allocated where it does the most good.

:::warning[Pick a target strictly between `lo_bit` and `hi_bit`]
Adaptation only works when `adakv_target_avg_bits` sits **strictly inside**
`(adakv_lo_bit, adakv_hi_bit)`. At either endpoint there's only one way to hit
the budget — give every head the same width — so the allocation flattens and
you're running plain [KIVI](./kivi) no matter what the importance signal says.

The default `2.5` is chosen for this reason. Setting `2.0` or `4.0` with the
default range emits a `UserWarning` rather than silently flattening.
:::

## Quick start

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="adakv",
    head_dim=128,               # your model's head dimension
    adakv_target_avg_bits=2.5,  # must be strictly inside (lo_bit, hi_bit)
)
caches = KVCacheBuilder.for_model(model, config)

output = generate(model, tokenizer, prompt="Tell me about KV caches",
                  kv_cache=caches)
```

There's no calibration step and no artifact to train — the method measures what
it needs as your text flows through. The defaults are reasonable; the only knob
most people should touch is `adakv_target_avg_bits`.

## What you actually save

Two things trip people up here, so they're worth stating plainly before the
numbers.

**Only keys are compressed.** Values pass through untouched at fp16. Since keys
and values are roughly half the cache each, halving the key size shrinks the
whole cache by much less than the key figure suggests.

**Quantization has overhead.** Numbers are compressed in groups, and each group
stores two extra fp16 values (a scale and an offset) to reconstruct from. Those
aren't free, and at small group sizes they're a large fraction of the total.

Putting both together, for a 1024-token sequence with `head_dim=128` and the
default `adakv_group_size=32`:

| `target_avg_bits` | Naive figure | Real key saving | Whole cache |
|---:|---:|---:|---:|
| 2.0 | 8.0x | 5.3x | 1.68x |
| **2.5** (default) | 6.4x | **4.6x** | **1.64x** |
| 3.0 | 5.3x | 4.0x | 1.60x |
| 4.0 | 4.0x | 3.2x | 1.52x |

The "naive figure" is just `16 ÷ target` — what you'd get if scales and offsets
were free. Expect the middle column in practice, and the right-hand column for
your actual memory use.

`adakv_group_size` is the lever on that overhead, and it's a real one:

| `adakv_group_size` | Key saving at 2.5 bits | Overhead |
|---:|---:|---:|
| 16 | 3.6x | 44% |
| **32** (default) | **4.6x** | 29% |
| 64 | 5.3x | 17% |
| 128 | 5.8x | 9% |

Larger groups mean less overhead but coarser quantization, since one scale and
offset now have to cover a wider spread of values. The default is a middle
setting, not a maximum.

:::note[Memory, not speed]
This won't make generation faster. Keys are decompressed back to fp16 before
attention runs, so the arithmetic is unchanged and you're paying a little extra
to compress and decompress. The win is footprint.
:::

## Choosing the importance signal

The whole method rests on deciding which heads deserve more bits. There are two
signals available, set via `adakv_importance`, and here's the part to
understand: **they rank heads in roughly opposite orders.** This isn't a tuning
detail — picking one is picking what you believe "important" means.

### `"norm_variance"` (default) — which heads are hardest to compress

For each head, this tracks how much the size of its key vectors varies from
token to token.

The logic: quantization works by fitting a range of values into a few bits. A
head whose values swing wildly needs a wide range, so each bit covers a coarser
step and more precision is lost. Giving that head extra bits buys back real
accuracy. A head whose values are all similar is easy to compress tightly.

This is computed continuously from running totals as tokens arrive — no
calibration, no attention weights, negligible cost.

:::note[This is not the original paper's criterion]
Ada-KV gives more budget to heads whose attention is **spread out** across many
tokens. High key-norm variance is usually the signature of the opposite — a few
outlier tokens dominating a head, which is a **concentrated** head, exactly the
kind the paper would give *less* to. The two signals are anti-correlated, and
there's a test in the suite that verifies this.

`norm_variance` is a defensible objective in its own right — it optimizes for
quantization error rather than attention coverage — but it is not an
approximation of the paper's. If you want the paper's ranking, use
`attention_entropy`. An earlier version of this page described norm-variance as
a proxy for attention entropy, which was wrong.
:::

### `"attention_entropy"` — the paper's criterion

This estimates how widely each head spreads its attention, and gives more bits
to the heads that spread it widest, matching Ada-KV §3.3.

There's a caveat. The real measure needs the model's queries, and the cache
never sees them — it only receives keys and values. So this substitutes recent
keys as stand-ins for queries, the same trick [SnapKV-adapted](./snapkv) uses.
It's still an approximation, but unlike `norm_variance` it at least points in
the direction the paper intended.

It costs more: one attention matrix per head during prefill. And since a single
token has no attention distribution of its own, the estimate made during prefill
is reused for the rest of the generation.

**Which to pick:** `norm_variance` if you want the cheapest option and your goal
is minimizing reconstruction error. `attention_entropy` if you want to follow
the paper's reasoning about which heads matter. They will disagree.

## How the budget is divided

Recomputed at the end of prefill and again each decode step:

1. **Rank the heads** by importance and spread those ranks into a bounded,
   mean-centred range. Only the *ordering* survives this step — the raw
   magnitudes are discarded.
2. **Place** each head at `target + spread × rank`, with `spread` as wide as it
   can be while keeping every head inside `[lo_bit, hi_bit]`. The average lands
   on your target by construction.
3. **Snap** each head to the nearest allowed value from `{lo_bit, mid_bit,
   hi_bit}`.
4. **Correct the rounding.** If snapping pushed the total off target, move the
   head sitting closest to a boundary one step in the corrective direction,
   ties broken by importance, until no single move gets the total closer.

Step 1 is worth explaining, since it looks like a detour. Key-norm variances
span orders of magnitude, so scaling them directly and clamping pins almost
every head to the floor or ceiling and throws away the ordering that steps 3–4
exist to act on. Ranks are bounded and order-preserving, so one extreme head
can't flatten everyone else.

Each head's keys are then quantized with KIVI-style asymmetric min/max group
quantization at its assigned width, and reconstructed to fp16 for attention.

### Inspecting what happened

```python
cache = caches[0]
print(cache.head_bits)           # e.g. [2, 4, 3, 2, ...] per-head assignment
print(cache.assigned_avg_bits)   # should land near your target
print(cache.head_importance)     # the scores driving the split
print(cache.importance_mode)     # which signal is active
```

If `head_bits` comes back all-identical, your target is at an endpoint — see the
warning at the top.

## Configuration reference

**The one most people set:**

| Parameter | Default | Description |
|---|---|---|
| `adakv_target_avg_bits` | `2.5` | Average bits per element. **Must be strictly inside `(lo_bit, hi_bit)`** or the allocation goes uniform. |

**Worth knowing about:**

| Parameter | Default | Description |
|---|---|---|
| `adakv_importance` | `"norm_variance"` | `"norm_variance"` (compression sensitivity) or `"attention_entropy"` (the paper's dispersion criterion). These disagree — see above. |
| `adakv_group_size` | `32` | Tokens per quantization group. Bigger means less overhead but coarser rounding. |
| `adakv_lo_bit` | `2` | Fewest bits any head can get. |
| `adakv_hi_bit` | `4` | Most bits any head can get. |
| `adakv_mid_bit` | `3` | Middle tier. Set equal to `hi_bit` for a two-tier `{lo, hi}` set. |
| `adakv_obs_window` | `32` | Observation window for `"attention_entropy"`. Ignored otherwise. |
| `adakv_update_interval` | `1` | Wired through config but not yet honoured — the allocation is recomputed every step regardless. See [What is not implemented](#what-is-not-implemented). |

### Tuning the target

With the default allowed set `{2, 3, 4}`:

| `target_avg_bits` | Adapts? | Typical spread |
|---|---|---|
| 2.0 | **No** — equals `lo_bit` | every head 2-bit (identical to KIVI) |
| 2.25 | Yes | mostly 2-bit, a few 3-bit |
| **2.5** (default) | Yes | mix of 2 and 3-bit |
| 3.0 | Yes | mix of 2, 3 and 4-bit |
| 3.5 | Yes | mix of 3 and 4-bit |
| 4.0 | **No** — equals `hi_bit` | every head 4-bit (identical to KIVI) |

If you want a 2.0-bit average *and* adaptation, drop `adakv_lo_bit` to `1` so
that 2.0 sits strictly inside the range.

## Troubleshooting

**Every head got the same bit-width.** Your target is at an endpoint of
`(lo_bit, hi_bit)`. A `UserWarning` is emitted for this. Move the target
inward, or widen the range.

**Memory didn't drop as much as the compression figure suggested.** Expected —
values are never compressed, and group parameters add overhead. See
[What you actually save](#what-you-actually-save).

**Generation isn't faster.** Also expected. This trades a little compute for
footprint; keys are expanded back to fp16 before attention.

**Quality dropped more than expected at the same average bits as KIVI.** Try
switching `adakv_importance`. If the default signal is ranking your model's
heads badly, an unlucky allocation can underperform a uniform one.

## Compared to related methods

| | AdaKV-proxy | [Kitty](./kitty) | [SVDq](./svdq) | [KIVI](./kivi) |
|---|---|---|---|---|
| What varies | per **head** | per **channel** | latent (SVD) | nothing (uniform) |
| Effective key bits | your target (2.25–3.5) | ~2.5 | ~1.25 | 2.0 |
| Calibration | none | none | SVD at prefill | none |
| Importance signal | key-norm variance or attention entropy (online) | per-channel variance (online) | singular values | none |
| Values compressed | no | no | no | yes |

**Over KIVI** — when some heads are far more quantization-sensitive than others.
Same average budget, concentrated where it helps.

**Over Kitty** — when the variation across heads matters more than variation
across channels within a head. They're complementary axes: Kitty moves bits
around inside a head, AdaKV-proxy moves them between heads. AdaKV-proxy also
lets you state an exact average-bit target rather than deriving it.

**Neither** — use Kitty if per-channel variance dominates, or SVDq if you need
sub-2-bit keys and can absorb an SVD at prefill.

## Adaptation notes

This is a **proxy** for Ada-KV, and the differences are substantive.

**1. It allocates bits, not eviction budget.** The real Ada-KV adapts how many
tokens each head *keeps*, deciding this from softmax attention weights. Those
weights aren't available inside the `update_and_fetch` contract. This port
adapts per-head *precision* instead, which fits the cache-only contract and
preserves the core idea — more resources to the heads that need them — but it is
a different mechanism. Nothing is ever evicted here.

**2. The default importance signal isn't the paper's.** As described
[above](#choosing-the-importance-signal), `norm_variance` is anti-correlated
with Ada-KV's attention-dispersion criterion. `attention_entropy` carries the
paper's sign at extra cost.

**3. Recomputed online.** The allocation comes from running accumulators updated
every step, not a one-time offline pass.

**4. No variable-length attention needed.** The paper's §3.5 flattened layout and
custom CUDA kernels exist to handle heads holding different token counts. Since
this port varies *bits* rather than *token counts*, every head holds the same
number of tokens and standard attention applies unchanged — the efficiency
problem the paper's §4.4 solves doesn't arise.

**5. Not validated on a trained model.** The paper's accuracy and throughput
results come from real long-context workloads this repo doesn't run. Treat the
compression figures here as arithmetic (they are), and the quality implications
as untested.

### What is not implemented

- **True head-adaptive eviction** — needs softmax attention weights, outside the
  cache contract. Documented as theoretical basis only.
- **Cross-layer budget sharing** — a layer of uniformly unimportant heads could
  in principle donate budget elsewhere. Out of scope; it would break the
  one-wrapper-per-layer contract.
- **`update_interval > 1`** — the field is wired through config but the
  assignment is recomputed every step regardless.

## Evidence

<details>
<summary>Test coverage behind the claims on this page</summary>

| Claim | Test | Status |
|---|---|---|
| High-importance heads receive more bits | `test_high_importance_heads_get_more_bits` | Verified |
| The default config actually adapts (regression, issue #31) | `test_default_config_is_adaptive` | Verified |
| Degenerate target warns instead of silently flattening | `test_degenerate_target_warns_and_is_uniform` | Verified |
| Allocation is monotone in importance (no saturation) | `test_allocation_monotone_in_importance` | Verified |
| Allocation is permutation-equivariant over distinct importances | `test_allocation_permutation_equivariant` | Verified |
| Budget met exactly where the allowed set permits | `test_budget_met_exactly` | Verified |
| `attention_entropy` ranks dispersed above sparse (paper's sign) | `test_attention_entropy_ranks_dispersed_above_sparse` | Verified |
| `norm_variance` is anti-correlated with the paper's criterion | `test_norm_variance_is_anticorrelated_with_paper_criterion` | Verified (documented caveat) |
| Average bits matches target within ±0.5 | `test_average_bits_matches_target` | Verified |
| Equal importance degrades to uniform allocation | `test_equal_importance_uniform_allocation` | Verified |
| Assigned bits beat `lo_bit` on the high-importance head | `test_high_importance_head_lower_mse_than_lo_bit` | Verified |
| Running norm accumulator matches ground-truth variance | `test_running_norm_accumulator_correctness` | Verified |
| `assigned_avg_bits` stays within `[lo_bit, hi_bit]` | `test_assigned_avg_bits_in_range` | Verified |
| Single-head model trivially assigns target | `test_single_head_assigns_target` | Verified |
| Output shape preserved (prefill + decode) | Tests 2, 3, 10 | Verified |
| Values unchanged | Test 4 | Verified |
| Determinism | Test 14 | Verified |
| Throughput vs KIVI/Kitty on M-series | `benchmark_scripts/benchmark_adakv.py` | Run locally |

</details>

## Next steps

- [Kitty](./kitty) — per-channel mixed precision within a head (complementary)
- [KIVI](./kivi) — the uniform 2-bit baseline this layers on
- [SVDq](./svdq) — sub-2-bit keys via offline SVD
- [Algorithm overview](./overview) — full method comparison
- [mlx_lm integration guide](../guides/mlx-lm-integration)

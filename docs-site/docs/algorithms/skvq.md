# SKVQ — Sliding-Window Reorder + Clip Quantization

**Method id:** `skvq` · **New in 0.30.0** · *Inspired by* ["SKVQ:
Sliding-window Key and Value Cache Quantization for Large Language Models"
(arXiv:2405.06219, COLM 2024)](https://arxiv.org/abs/2405.06219) — **SKVQ-adapted
(VeloxQuant-MLX implementation)**, not a faithful port: the paper's offline
calibration is replaced by first-chunk statistics (see
[Adaptation notes](#adaptation-notes)).

SKVQ pushes plain asymmetric group quantization to very low bit-widths with
two mechanisms that are new to this library, behind a sliding fp16 window
that is not:

1. **Channel reordering** — permute the head-dim channels so that channels
   with similar dynamic range share a quantization group. A group's min/max
   then fits every member tightly, instead of one wide channel stretching
   the scale for fifteen narrow ones.
2. **Clipped dynamic quantization** — shrink each group's quantization
   window by a factor α ∈ (0, 1] centered on the group midpoint, saturating
   a few extreme values to buy finer resolution for everything else. α is
   chosen **per group** by grid search against reconstruction error; α = 1
   (no clipping) is always in the grid, so the search never loses to plain
   min/max under its own metric.
3. **Sliding window + sink filter** — the most recent `skvq_window` tokens
   stay fp16; tokens aging out are quantized **once and frozen** (the
   chunk-flush idiom shared with [NSNQuant](../algorithms/nsnquant)). The
   first `skvq_n_sink` tokens (attention sinks) stay fp16 forever.

## Where it sits in the quantization family

| Trick | Method |
|---|---|
| Per-channel scales (groups along tokens) | [KIVI](../algorithms/kivi) keys |
| Outlier isolation to fp16 | [KVQuant-NUQ](../algorithms/kvquant) |
| Non-uniform signpost levels | [KVQuant-NUQ](../algorithms/kvquant) |
| **Regroup channels by statistics** | **SKVQ** |
| **Clip the range instead of covering it** | **SKVQ** |

Both K and V are quantized with **per-token** groups along the channel axis
(the KIVI *value* scheme). Reordering is what makes that viable for keys:
without it, one dominant channel per group stretches every scale — with it,
dominant channels share groups with each other.

Because the permutations are frozen after the first flushed chunk and every
other step is deterministic, prefill and token-by-token decode produce
**bit-for-bit identical caches** (pinned by test) — the same
path-independence property as NSNQuant's chunk flush.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="skvq",
    head_dim=128,
    skvq_bits_key=2,        # paper setting
    skvq_bits_value=2,      # paper uses 1.5; we ship integer bits
    skvq_group_size=32,     # channels per quant group
    skvq_window=128,        # fp16 sliding window (= flush chunk size)
    skvq_n_sink=5,          # attention-sink filter, kept fp16
    skvq_reorder=True,      # channel reordering (False = ablation)
    skvq_clip_search=True,  # per-group clip grid search
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`SKVQKVCache` per attention layer.

## How it works

Per `update_and_fetch` (prefill or decode, same path): append the incoming
block at fp16, then, while at least `skvq_window` tokens sit past the
quantized frontier, flush that chunk:

1. **First flush only:** compute per-head channel permutations for K and V
   from this chunk (sort channels by dynamic range) and freeze them.
2. Permute channels → per-token clipped group quant → dequant → inverse
   permute; overwrite the chunk's fp16 storage with the round-trip.
3. **Chunk 0 only:** restore the first `skvq_n_sink` rows to exact fp16.

The chosen clip factor is folded into each group's stored `(lo, scale)` —
nothing extra is kept. Byte accounting (`compressed_*_bytes`,
`residual_fp16_bytes`, `perm_bytes`, `assigned_avg_bits`) counts codes,
fp16 metadata per (token, group), the fp16 window + sinks, and the one-off
int32 permutations.

## Adaptation notes

**What the paper does that we do not:**

- **Offline calibration.** The paper clusters channels with KMeans over
  distribution features computed on 256×4096 WikiText-2 samples, and finds
  α by minimizing attention-output MSE offline. We compute the permutation
  from the **first flushed chunk** of live traffic (sorting on per-channel
  dynamic range — the 1-D analogue of KMeans grouping) and search α **per
  group at flush time** against reconstruction MSE.
- **Weight-fused permutation.** The paper hides the reorder inside the
  attention projection weights; we permute explicitly and invert on
  dequantization. Mathematically identical round-trip, two extra gathers
  per flushed chunk.
- **1.5-bit values and FP8 metadata.** Both are CUDA packing artifacts in
  the paper; we ship integer bit-widths and fp16 metadata, all counted in
  the byte accounting.
- **Fused kernels.** The paper's throughput gains do not port to Metal —
  on Apple Silicon the win is *memory* (same caveat as KIVI/NSNQuant).

**The honest caveat:** channel reordering only pays when channels are
*heterogeneous*. That real transformer K/V have a few dominant channels is
the paper's premise (shared with KIVI and KVQuant) — our benchmark
constructs that regime synthetically and also runs a homogeneous control
where reordering buys nothing (measured −0.3%). The premise itself is
attributed to the papers, not validated here.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_skvq.py` (13 tests) and
`veloxquant_mlx/tests/cache/test_skvq_cache.py` (18 tests):

- α=1 clip window equals plain asymmetric min/max group quantization
  (verified against a manual numpy reference)
- Per-group clip search is never worse than α=1 under reconstruction MSE,
  and strictly better on outlier-heavy groups
- Reordering reduces round-trip MSE under smooth channel heterogeneity and
  collapses per-channel normalized error when outlier channels fill whole
  groups; the homogeneous control shows no comparable win
- **Prefill vs token-by-token decode produce bit-for-bit identical
  caches**; permutations are frozen from chunk 0; flushed chunks never
  change; sink rows stay fp16-exact
- Byte accounting matches the closed form; build-time validation; max_ctx
  guard; `for_model` wiring

The offline harness in `benchmark_scripts/benchmark_skvq.py` (results in
`figures/skvq/results.json`) sweeps sequence length
(512/1024) × bits (2/4) × regime, with ablation arms (reorder off, clip
off, both off) and the repo's KIVI as reference, at matched
bits/group/window:

- **heterogeneous channels** (2.5-decade smooth scale spread): reordering
  cuts key MSE by a further **16.9%** on top of clip search, and cuts
  per-channel normalized error ~450× (0.21 vs 91.5 at S=512, 2-bit);
  clip search adds **14.0%** on top of reordering.
- **homogeneous control:** reordering's effect is **−0.3%** — nothing, as
  it should be. Clip search still helps (~15%) since it addresses
  within-group Gaussian tails, not channel structure.
- **KIVI reference:** KIVI's per-channel key scheme is intrinsically immune
  to channel heterogeneity and **wins several heterogeneous rows outright**
  (e.g. lower perturbation at S=512) — reported as measured. KIVI also
  always holds the trailing 128 tokens fp16 (its incoming-block
  simplification), while SKVQ quantizes everything that ages out — at
  S ≡ 0 (mod window) SKVQ has quantized 100% of aged tokens and still lands
  within ~10% of KIVI's key MSE at 2 bits.

**No model-level benchmark has been run.** These are offline-synthetic
reconstruction/perturbation and byte-accounting numbers — they validate the
machinery, not the papers' claims about real KV statistics.

## When to use it

| Method | Key scheme | Decode-time story | Window |
|--------|-----------|-------------------|--------|
| [KIVI](../algorithms/kivi) | per-channel groups along tokens | incoming-block only (trailing block stays fp16) | fp16 residual |
| [NSNQuant](../algorithms/nsnquant) | universal-codebook VQ | chunk flush, quantize-once | fp16 chunk buffer |
| **SKVQ** | per-token groups + reorder + clip | chunk flush, quantize-once | fp16 sliding window + sink filter |

Choose SKVQ when you want very low-bit uniform quantization with dynamic
per-token scales *and* your model family shows the classic dominant-channel
key structure; choose KIVI for the simplest tuning-free baseline (its
per-channel keys handle heterogeneity by construction); choose NSNQuant
when you want VQ-level rates with zero data dependence.

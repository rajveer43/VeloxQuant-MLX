---
slug: crosslayer-decode-batching-results
title: "Batching Decode Attention Across Requests Got Us 3.83x Real Throughput — Across Layers Got Us Nothing"
description: "Full benchmark results for issue #307's batched decode-attend kernel: 1.0-4.5x kernel-level gains from cross-layer batching, a structural wall blocking real single-request decode, and 1.50-3.83x real end-to-end throughput from request batching on Qwen3-4B."
date: 2026-09-05
authors: rajveer
tags: [metal, apple-silicon, mlx, attention, kv-cache, benchmarking, results]
---

*The full benchmark results for issue #307's cross-layer/multi-request batched decode-attend kernel: kernel-level numbers, the real-model end-to-end tokens/sec table, and why one axis of the same optimization won while the other structurally cannot.*

---

Here is the headline number: **3.83x real decode throughput** on an actual model, from a kernel that already shipped in this repo and nobody had wired into a real serving path. And here is the number right next to it that matters just as much: **1.0x** — because the other half of the same optimization idea, which looked identical on paper, cannot ever produce a real speedup, for a reason baked into how every transformer computes.

This is the results writeup. For the story of how these numbers were found — the failed baseline, the two test-harness bugs caught before trusting them, the model-source-reading that closed off half the idea — see the companion post, [Batching Decode Attention Across Layers Sounded Great — Until the Residual Stream Said No](/docs/blog/crosslayer-decode-batching).

---

## The setup

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` had already shown `scalar_fused_decode_attend` — this repo's fused group-affine decode+attention kernel — is occupancy-bound at realistic decode shapes: a single decode step for one request dispatches only 8-32 Metal threadgroups, far too few to fill a 10-core Apple GPU. The doc's own Recommendation #2 named the fix: dispatch more threadgroups per call, either by batching multiple **layers** or multiple **requests** into one launch. This benchmark tests both, on real hardware and, for the request axis, a real model.

All numbers: one base 10-core Apple M4, 24 GB unified memory.

---

## Result 1: cross-layer batching, kernel level — a real, consistent win

`scalar_fused_decode_attend_batched` adds one outermost `NL` (layer count) axis to the existing kernel's grid, so one dispatch covers every transformer layer's decode-attend call instead of 28-80 separate launches. Correctness: **bit-identical**, not just close, to looping the single-layer kernel `NL` times and stacking the outputs — verified for `NL ∈ {1, 4, 32}` plus an adversarial `NL=3, B=2` combined-indexing test.

| H_kv | H_q/H_kv | S_kv | NL | sequential ms | batched ms | speedup |
|---|---|---|---|---:|---:|---:|
| 2 | 1 | 128 | 80 | 1.01 | 0.44 | 2.28x |
| 2 | 1 | 16384 | 32 | 53.88 | 15.66 | **3.44x** |
| 2 | 1 | 16384 | 80 | 107.75 | 23.89 | **4.51x** |
| 8 | 1 | 16384 | 80 | 289.66 | 83.91 | **3.45x** |
| 8 | 8 | 16384 | 32 | 501.15 | 229.55 | 2.18x |
| 8 | 8 | 16384 | 80 | 1247.28 | 322.54 | **3.87x** |
| 2 | 8 | 2048 | 32 | 11.80 | 11.28 | 1.05x |

Every shape tested landed as a win — no null or negative result, unlike two prior attempts at fixing this same occupancy problem (GQA head-packing measured 2.7-4.7x *slower*; a SIMD-shuffle alternative was ruled out architecturally before being built). The win grows with `S_kv`, topping out at 4.5x at 16k tokens across 80 layers, and shrinks toward ~1.0-1.3x at small `S_kv` combined with a high query/kv-head ratio, where fixed dispatch overhead eats a larger share of both numbers.

## Result 2: the stacking tax that erases it for real single-request serving

Batching requires the caller to `mx.stack` each layer's tensors into one buffer before dispatch. That cost is real and doesn't scale predictably:

| S_kv | NL | mx.stack cost |
|---|---|---:|
| 128 | 32 | 1.19 ms |
| 2048 | 32 | 25.38 ms |
| 16384 | 32 | 14.91 ms |
| 16384 | 80 | 36.41 ms |

At `S_kv=2048, NL=32`, the ~0.6ms kernel-level saving is dwarfed by a 25ms stacking cost. Reading `mlx_lm`'s actual `KVCache` source confirmed this isn't a one-time cost: it builds one independent cache object *per layer* (`[KVCache() for _ in range(num_layers)]`), with no shared layer-stacked buffer anywhere — so a real integration would pay the stacking cost **every decode step**, not once.

## Result 3: the structural wall — 1.0x, permanently

The real blocker for single-request decode isn't the stacking cost — it's underneath it. Reading `mlx_lm`'s model code directly:

```python
# TransformerBlock.__call__
r = self.self_attn(self.input_layernorm(x), mask, cache)
h = x + r
r = self.mlp(self.post_attention_layernorm(h))
out = h + r
return out
```

Layer `L+1`'s attention input needs layer `L`'s **complete** block output — attention, residual, MLP, residual — not just its attention output. No reordering of a standard decoder transformer lets `N` layers' attention be grouped into one dispatch while still computing the same model. This closes cross-layer batching for real single-request decode as a **structural dead end**, independent of hardware, kernel quality, or how the stacking cost might be optimized away later. Speedup for this path, permanently: **1.0x** — it cannot apply to the case it was aimed at.

## Result 4: the request-batching half — real, positive, measured end-to-end on Qwen3-4B

The other half of the same original lever — batching across concurrent *requests* rather than layers — has no residual-stream dependency to block it, and it doesn't even need the new batched kernel: the *existing*, already-shipped `scalar_fused_decode_attend` already has a `B` axis in its dispatch grid. Nothing in this repo had ever routed real generation through it, though — the shipped `KIVIKVCache` dequantizes to fp16 and calls standard SDPA instead.

Measured on `mlx-community/Qwen3-4B-4bit` (36 layers, `H_q=32, H_kv=8`), real prompts, real greedy decoding, both arms verified to produce **bit-identical output tokens** before any timing was trusted:

| B (concurrent requests) | decode tok/s — dequant+SDPA (today's path) | decode tok/s — fused kernel | speedup |
|---|---:|---:|---:|
| 1 | 17.1 | 25.5 | 1.50x |
| 4 | 25.3 | 48.1 | 1.90x |
| 16 | 34.3 | 108.1 | 3.16x |
| **32** | **39.1** | **149.9** | **3.83x** |

TTFT was unaffected in both arms at every batch size — expected, since prefill never touches this decode-only kernel. The speedup climbing with `B` (1.50x → 3.83x) is the exact occupancy signature the roofline document's synthetic sweep predicted, now confirmed through a real forward pass on a real model instead of an isolated kernel call.

---

## Four findings worth pulling out

### 1. Two axes of the "same" optimization can have opposite outcomes

Cross-layer batching and request batching both raise threadgroup count by exactly the same mechanism (a new grid axis). One is permanently blocked by the residual stream; the other works cleanly. Symmetry in the kernel design does not imply symmetry in the real-world result — the two axes had to be tested separately, on real models, to find that out.

### 2. Most of the win comes from skipping dequantization, not from the attend loop itself

At the actual real-model shape (`S_kv≈64, B=4`), the fused kernel measured only **~1.15x** faster than a plain fp16 cache with *no quantization at all* — versus **2.57x** faster than the KIVI dequant-then-SDPA baseline it's meant to replace. Most of the headroom in the 1.50x-3.83x end-to-end numbers above comes from skipping the fp16 materialization step specifically, not from the attend computation being dramatically cheaper in absolute FLOPs.

### 3. Test-harness bugs can look exactly like real results if you don't check tokens

Two bugs surfaced while building the real-model benchmark, and both would have silently produced a wrong number if uncaught: a full-history requantization cost (~25ms/step across 36 layers) that swamped the kernel's own cost, and a baseline arm that was accidentally comparing against a *non-quantized* fp16 cache instead of the real KIVI dequant path. Both were caught only because both arms' output tokens were compared for exact equality before any timing number was trusted — a cheap check that would have been easy to skip.

### 4. A closed negative result is as valuable as an open positive one

The cross-layer half of this work produced no usable speedup, but it produced a permanent answer: don't revisit this axis for single-request serving, on any model, on any future hardware — the constraint is in the model's math, not this GPU or this kernel. That's a stronger, more durable finding than "didn't try it" or "not yet integrated," and it's reported with the same detail as the positive result above rather than left out.

---

## What this means in practice

**If you're serving one request at a time**, this work doesn't change anything for you today — the cross-layer kernel exists and is tested, but there's no path to a real speedup from it under standard transformer architectures.

**If you're serving multiple concurrent requests** through a KIVI-quantized (or similarly structured) KV cache, routing decode-step attention through `scalar_fused_decode_attend` instead of dequantize-then-SDPA is a real, verified win that grows with batch size — up to 3.83x at `B=32` in this benchmark. The cache used here is a minimal test harness (no fp16 residual window), so a production integration — wired through `KVCacheBuilder`, with a real residual window, tested against variable-length concurrent requests — is the natural next step and isn't shipped in this PR.

---

## Reproducibility

```bash
# Kernel-level benchmark (Results 1-2 above), self-calibrated bandwidth peak
python benchmark_scripts/benchmark_crosslayer_decode_batch.py

# Real-model end-to-end benchmark (Result 4 above) — must run as a module,
# not by path, or it can resolve to a stale installed copy of the package
python -m benchmark_scripts.benchmark_real_model_scalar_attend

# Correctness (72 tests, 16 new for this work)
pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v -k batched
```

Full methodology, additional shapes, and the complete honesty caveats (what the benchmark cache does and doesn't represent about production KIVI) are in `docs/KV_KERNEL_ROOFLINE_FINDINGS.md`'s two new addenda: "cross-layer batched decode-attend dispatch (issue #307, part 1)" and the real-model follow-up immediately after it.

*VeloxQuant-MLX is MIT licensed. Hardware: Apple M4, 24GB unified memory. Model: mlx-community/Qwen3-4B-4bit.*

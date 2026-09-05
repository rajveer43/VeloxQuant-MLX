---
slug: crosslayer-decode-batching
title: "Batching Decode Attention Across Layers Sounded Great — Until the Residual Stream Said No"
description: "Building issue #307's cross-layer batched decode-attend kernel found a real, bit-verified kernel win, then a structural wall for real single-request decode — and a real 1.5-3.8x win on the untried request-batching half, measured end-to-end on Qwen3-4B."
date: 2026-09-05
authors: rajveer
tags: [metal, apple-silicon, mlx, attention, kv-cache, benchmarking, roofline]
---

*Issue #307's own roofline analysis named one unblocked, unattempted lever for `scalar_fused_decode_attend`'s occupancy problem: batch multiple layers' independent decode-attend calls into one Metal dispatch. Building it produced a real, measured kernel-level win — and then a real-model test found the lever can't reach where it matters most, for a reason baked into how every transformer works. The other half of the same lever could, and did, on a real model.*

---

## The problem this picks up

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` already established that `scalar_fused_decode_attend` — the fused group-affine decode+attention kernel behind this repo's KIVI-style KV cache — is not memory-bound, not compute-bound, and not launch-bound. It's **occupancy-bound**: a real decode step (`B=1, H_kv=8, S_q=1`) dispatches only 8 threadgroups, nowhere near enough to fill a 10-core Apple GPU, no matter how efficient the kernel's inner loop is or how much data any one threadgroup has to move.

Two attempts at fixing this from inside the kernel had already failed, honestly and with numbers: GQA head-packing (share decoded K/V across query heads sharing a kv-head) measured 2.7-4.7x *slower*, because it traded away threadgroup count to save decode arithmetic — the wrong trade when occupancy, not arithmetic, is the bottleneck. A `simd_shuffle`-based alternative was ruled out architecturally before it was even built: cross-lane shuffles only work within one threadgroup, and under full-occupancy dispatch a threadgroup only ever holds one query head's data.

What the roofline doc's own Recommendation #2 named as the actual unblocked lever: **dispatch more threadgroups per call by processing multiple layers or multiple requests in one kernel launch** — batching across what's currently separate per-layer Python-level calls, rather than trying to make one tiny dispatch stream memory faster. This post is about building both halves of that lever, and getting two very different answers.

## Part 1: batching across layers — a real kernel win, verified bit-for-bit

A real decode step calls `scalar_fused_decode_attend` once *per transformer layer* — typically 28 to 80 times for current open-weight models — and those calls are independent at the Metal-dispatch level: each layer's own K/V cache and post-attention Q projection are already sitting in memory before any of these calls run. Nothing about attention's data dependencies forces them to be separate launches.

`scalar_fused_decode_attend_batched` adds exactly one thing to the existing kernel: a new outermost `NL` (num_layers) grid axis, so threadgroup count becomes `NL * B * H_kv * S_q` instead of `B * H_kv * S_q`. The indexing change in the Metal source is small and mechanical —

```cpp
uint sq_idx  = tg % S_q;
uint hkv_idx = (tg / S_q) % H_kv;
uint b_idx   = (tg / (S_q * H_kv)) % B;
uint l_idx   = tg / (S_q * H_kv * B);

uint bh_kv = (l_idx * B + b_idx) * H_kv + hkv_idx;   // one new leading stride term
```

— and every buffer offset that touches `bh_kv`, `q`, or `out` gets the same added `l_idx * (...)` leading term. Same math, same memory layout, same per-(layer, batch, kv-head, query-position) work per threadgroup. The caller pre-stacks each layer's tensors along a new leading axis; the kernel doesn't gather anything itself.

The load-bearing correctness check: calling the batched kernel once must be **bit-identical**, not just within tolerance, to calling the single-layer kernel `NL` times in a loop and stacking the outputs.

```python
@pytest.mark.parametrize("NL", [1, 4, 32])
def test_scalar_attend_batched_parity_vs_single_layer_loop(NL):
    ...
    max_abs = np.abs(got_np.astype(np.float32) - ref.astype(np.float32)).max()
    assert max_abs == 0.0, f"NL={NL}: batched must be bit-identical to the single-layer loop"
```

It passed on the first real run, along with an adversarial `NL=3, B=2` test specifically built to catch a swapped layer/batch stride order, and a non-broadcast test confirming layer 1's output isn't secretly layer 0's data repeated (the kind of copy-paste stride bug that a same-content test would miss entirely).

The result was a consistent, positive win everywhere tested — no null result, unlike both prior attempts:

| H_kv | H_q/H_kv | S_kv | NL | sequential ms | batched ms | speedup |
|------|----------|------|----|---------------:|-----------:|--------:|
| 2 | 1 | 16384 | 80 | 107.75 | 23.89 | **4.51x** |
| 8 | 1 | 16384 | 80 | 289.66 | 83.91 | **3.45x** |
| 8 | 8 | 16384 | 80 | 1247.28 | 322.54 | **3.87x** |
| 2 | 8 | 2048 | 32 | 11.80 | 11.28 | 1.05x |

The win grows with `S_kv` (up to 4.5x at 16k tokens) and shrinks toward ~1.0-1.3x at small `S_kv` combined with a high `H_q/H_kv` ratio, where fixed dispatch overhead is a larger share of both numbers. Achieved bandwidth in the batched case reached up to ~40% of a freshly self-calibrated peak — still short of memory-bound, meaning occupancy remains the governing constraint even after batching, just a much less severe one.

But a kernel-level win isn't the same thing as a real speedup, and the next two steps are why.

## Part 2: the stacking cost, and a KVCache that won't cooperate

Batching requires the caller to hand over layer-stacked tensors — `mx.stack`ing `NL` independent per-layer arrays before every dispatch. That's not free, and it doesn't scale the way you'd guess:

| S_kv | NL | mx.stack ms |
|------|----|--------------:|
| 128 | 32 | 1.19 |
| 2048 | 32 | 25.38 |
| 16384 | 32 | 14.91 |
| 16384 | 80 | 36.41 |

At `S_kv=2048, NL=32`, a ~0.6ms kernel-level saving is dwarfed by a 25ms stacking cost — if that stacking has to be paid on every decode step, the kernel win is meaningless.

Whether it does depends entirely on the KV-cache implementation underneath, so instead of assuming an answer, I read `mlx_lm`'s actual cache source:

```python
# mlx_lm/models/cache.py
return [KVCache() for _ in range(num_layers)]
```

One independent `KVCache` object per layer, each with its own independently-growing `.keys`/`.values` buffer. There is no shared, layer-stacked backing buffer anywhere in the stock cache — which means a real integration built on it would need to `mx.stack()` every layer's state **fresh, on every single decode step**, not once at cache-construction time. The stacking cost above is a per-step tax, not a one-time layout change. Same amortization failure mode this repo already found for a sibling kernel (`fused_sdpa`'s dispatcher is a documented no-op for exactly this reason) — just arriving via a different mechanism.

## Part 3: the wall — why cross-layer batching can never reach real single-request decode

Even setting the stacking cost aside, there's a harder problem underneath it, and it took reading `mlx_lm`'s model code directly to see it clearly:

```python
# mlx_lm/models/llama.py — LlamaModel.__call__
for layer, cache in zip(self.layers, cache):
    mask = swa_mask if layer.use_sliding else fa_mask
    h = layer(h, mask, cache=cache)
```

```python
# TransformerBlock.__call__
r = self.self_attn(self.input_layernorm(x), mask, cache)
h = x + r
r = self.mlp(self.post_attention_layernorm(h))
out = h + r
return out
```

Layer `L+1`'s attention input is `q_proj(norm(layer_L_output))` — and `layer_L_output` is layer `L`'s **entire block output**: attention, residual, MLP, residual. Not just its attention output. There is no reordering of a standard pre-norm decoder transformer that lets `N` layers' attention be grouped into one dispatch while still computing the same model. "Run all layers' Q/K/V projections first, batch the attend call, then finish the MLPs" isn't a valid optimization here — layer 2's Q/K/V projections need layer 1's MLP output, which needs layer 1's attention output already. Any patch that tried this would silently change what the model computes, which fails the correctness bar this repo holds everywhere else.

This is a structural fact about the residual stream, not a missing engineering patch. **Cross-layer batched decode-attend cannot speed up real single-request decode latency on any standard transformer — full stop, regardless of how good the kernel or the batching mechanism gets.** That's a stronger, more useful conclusion than "not yet integrated": it closes this half of the lever as a dead end, not a pending follow-up.

## Part 4: the other half of the lever actually works — measured on a real model

Recommendation #2 named two axes: batch across layers, *or* across requests. Concurrent requests have no residual-stream dependency between them — they're independent by construction — so this half was never blocked by anything in Part 3. And it turns out it doesn't even need the new batched kernel: the *original*, already-shipped `scalar_fused_decode_attend` already carries a `B` (batch/request) axis in its dispatch grid (`n_tg = B * H_kv * S_q`). Nobody had ever wired real generation through it, though — `KIVIKVCache` dequantizes to fp16 and hands off to standard SDPA instead, the same pattern that made the fused kernel a no-op for the sibling VecInfer cache.

So I built a minimal cache that keeps KIVI-quantized codes live and exposes them, and monkeypatched `mlx_lm`'s SDPA dispatch to route real decode-step attention through the fused kernel when it applies:

```python
def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    if S_q == 1 and isinstance(cache, _ScalarAttendKIVICache) and cache._k_codes is not None:
        if use_fused:
            return scalar_fused_decode_attend(
                queries, *cache.quantized_state(), GROUP_SIZE, scale, nsg=nsg
            )
        else:
            k_hat, v_hat = cache.dequantized_kv(heads_per_kv)  # what KIVI does today
            return _original_sdpa(queries, k_hat, v_hat, cache=None, scale=scale, mask=None)
    return _original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask)
```

Two bugs surfaced immediately, and both mattered enough to fix before trusting any number. First: re-quantizing the *entire* growing K/V history from scratch every decode step cost ~25ms/step across 36 layers — pure test-harness overhead that would have silently drowned the kernel's own ~12ms/step in noise. Fixed by quantizing only the newly-aged `group_size`-aligned block each step, mirroring the incremental-flush discipline this repo's real `KIVIKVCache` already uses. Second, and sneakier: the "baseline" arm was accidentally attending over the cache's raw fp16 buffer instead of dequantized KIVI codes — comparing the fused kernel against a plain fp16 cache with *no quantization at all*, not against the KIVI dequant+SDPA path it's actually meant to replace. Fixed by giving the cache an explicit `dequantized_kv()` method both arms could be checked against, and verifying — before trusting a single timing number — that both arms decode to **exactly the same tokens**:

```
dequant-baseline route counts: {'decode_fused': 0, 'decode_dequant_baseline': 360, 'fallback': 0}
decoded (dequant baseline):  sentence is00000000
fused route counts: {'decode_fused': 360, 'decode_dequant_baseline': 360, 'fallback': 0}
decoded (fused):  sentence is00000000
tokens match: True
```

With that confirmed, the real numbers, on `mlx-community/Qwen3-4B-4bit` (36 layers, `H_q=32, H_kv=8`), real prompts, real greedy decoding, real end-to-end tokens/sec including embeddings, MLPs, o_proj, and sampling — not an isolated kernel call:

| B (concurrent requests) | decode tok/s, dequant+SDPA | decode tok/s, fused kernel | speedup |
|---|---|---|---|
| 1  | 17.1  | 25.5  | **1.50x** |
| 4  | 25.3  | 48.1  | **1.90x** |
| 16 | 34.3  | 108.1 | **3.16x** |
| 32 | 39.1  | 149.9 | **3.83x** |

The speedup growing with batch size — 1.50x at B=1, up to 3.83x at B=32 — is exactly the occupancy signature the synthetic roofline sweep predicted, now confirmed through an actual forward pass instead of an isolated kernel call. TTFT was unaffected in both arms, as it should be: prefill uses `S_q > 1` and never touches this decode-only kernel.

Two caveats worth stating rather than glossing over. The isolated per-layer comparison at this shape showed the fused kernel only ~1.15x faster than a **plain fp16 cache with no quantization at all** — most of the 2.57x-over-dequant-baseline margin comes specifically from avoiding the dequant materialization step, not from the attend computation itself being dramatically faster in absolute terms. And the cache built for this test has no fp16 residual window, so its *output quality* isn't representative of production KIVI — the timing comparison is sound because both arms decode from identical (lossy) quantized state and were verified token-for-token, but "1.5-3.8x faster" describes the kernel swap, not a production-ready serving stack.

## Where this leaves things

Two symmetric findings, both earned by actually building and measuring rather than assuming: cross-layer batching is a structural dead end for the case that matters most (a single request's decode latency), closed for good reasons rather than left open as unfinished work — and the request-batching half of the same original lever is real, positive, and now confirmed on an actual model rather than a synthetic sweep, using a kernel that was already sitting in the codebase unused. A production-grade version — a real residual-window cache, wired through `KVCacheBuilder`, tested against variable-length concurrent requests rather than identical-length ones — is the natural next step, and isn't attempted here.

*Full details, tables, and reproduction commands: `docs/KV_KERNEL_ROOFLINE_FINDINGS.md`'s two new addenda ("cross-layer batched decode-attend dispatch" and the real-model follow-up immediately after it). Correctness: `pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v -k batched` (72/72 passing, zero regressions in the existing suite). Benchmarks: `python benchmark_scripts/benchmark_crosslayer_decode_batch.py` and `python -m benchmark_scripts.benchmark_real_model_scalar_attend`. All numbers from one base 10-core Apple M4, 24 GB.*

*For the results-first version of this post — full benchmark tables, the headline numbers, and the practical "what this means for you" guidance without the narrative — see [Batching Decode Attention Across Requests Got Us 3.83x Real Throughput — Across Layers Got Us Nothing](/docs/blog/crosslayer-decode-batching-results).*

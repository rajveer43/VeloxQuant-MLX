---
slug: tensorops-research
title: "TensorOps Research: What We Learned Optimizing KV Caches"
description: "Reading Apple's full Metal 4 spec to speed up a KV cache attention kernel: two hardware matmul paths failed, but two smaller fixes shipped."
date: 2026-06-20
authors: rajveer
tags: [metal, apple-silicon, mlx, gpu, research]
---
# I Read the Entire Metal 4 Spec Looking for a GPU Trick. Here Is Every Dead End, and the One Thing That Actually Worked.

*A deep-dive into Apple's Metal Shading Language specification, what TensorOps promised, why it didn't work through MLX, and the two real improvements we shipped from three sessions of research.*

---

## Where This Story Starts

A few weeks ago I shipped a FlashAttention-style Metal kernel for VeloxQuant-MLX that was correct, fast in isolation, and completely useless end-to-end. The blog post about that mistake is [here](https://medium.com/@rajveer.rathod1301). The short version: I built a fused dequant+attention kernel that beat `mx.fast.scaled_dot_product_attention` by 1.3× in benchmarks — then discovered mlx_lm had already eliminated the dequant cost via a persistent fp16 K_hat buffer, making my kernel 3-4× slower than the baseline it was supposed to beat.

I kept the kernel in the library as an opt-in API. It's correct, it's tested, and it loses.

After writing that post, a reader suggested I look at the Metal Shading Language specification — specifically Metal 4, which Apple released with macOS Sequoia. The argument was: Metal 4 adds hardware tensor operations that could replace the slow part of the kernel. Maybe there was a path to winning that I hadn't found yet.

So I read the spec. All 346 pages of it.

This is what I found.

---

## The Kernel's Hot Path

To understand why the spec research mattered, I need to explain the bottleneck.

The fused SDPA kernel computes attention directly from VecInfer's compressed codebook indices without materializing the fp16 key matrix. For each query, it needs to compute a **Look-Up Table** first:

```
LUT[sub, centroid] = q_sub_vector · codebook_row[centroid]
```

For VecInfer's default config (`n_sub=16`, `sub_dim=8`, `n_centroids=256`), this is a `[16, 8] @ [8, 256]` matrix multiply — 4,096 dot products. In the current kernel, 32 GPU lanes stripe these across the SIMD group: each lane computes 128 scalar dot products independently.

This LUT precompute is Phase 0. Everything else — the online softmax, the V accumulation — comes after. If the LUT is slow, everything is slow.

The Metal 4 spec describes two potential hardware paths to speed this up:

1. **`simdgroup_float8x8`** (Metal 2.3+, Section 2.4 / 6.7): 8×8 hardware matmul tiles via `simdgroup_multiply_accumulate`. Available today.
2. **TensorOps `matmul2d`** (Metal 4+, Section 7.2): A full hardware matrix multiply API with a `cooperative_tensor` destination. Potentially much faster.

I tested both.

---

## Attempt 1: `simdgroup_float8x8`

The Metal spec (Section 6.7, Table 6.9) shows `simdgroup_float8x8` as a cooperative 8×8 float matrix multiply tile. The `<metal_simdgroup_matrix>` header is accessible via MLX's `header=` parameter:

```python
k = mx.fast.metal_kernel(
    name="my_kernel",
    source=src,
    header="#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
    ...
)
```

The tiling plan for our LUT: `n_sub=16` rows / 8 = 2 row-tiles, `n_centroids=256` cols / 8 = 32 col-tiles, `sub_dim=8` = 1 K-tile. Total: 64 hardware matmul operations.

I implemented it. Correctness test: zero diff vs reference.

Then I benchmarked it against the current scalar loop:

```
scalar loop:     212 µs per LUT precompute
simdgroup 8×8:   255 µs per LUT precompute
```

**The hardware matrix multiply was slower.**

The reason is protocol overhead. `simdgroup_float8x8` is a cooperative operation — all 32 lanes must execute each tile in lock-step. For our 64 tile iterations, that's 64 synchronization points. The scalar loop has zero synchronization: each lane independently computes 128 dot products in parallel. For a small matrix like `[16,8]@[8,256]`, the cooperation overhead dominates the compute savings.

`simdgroup_matrix` wins at large, batched matmuls (MLX uses it for GEMM with 128×128 tiles). For our 16×256 LUT, it's the wrong tool. Reverted.

---

## Attempt 2: Metal 4 TensorOps

Section 7.2 of the spec describes `tensor_ops::matmul2d` — a hardware-accelerated matrix multiply that operates on `tensor<>` types and writes to a `cooperative_tensor` destination held in thread registers. The pitch is exactly right: no threadgroup memory round-trip, hardware tensor units, single API call.

The example from the spec:

```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

[[ kernel ]] void matrixMultiply(
    tensor<device half, dextents<int, 2>> a [[ buffer(0) ]],
    tensor<device half, dextents<int, 2>> b [[ buffer(1) ]],
    tensor<device half, dextents<int, 2>> c [[ buffer(2) ]]) {

    constexpr auto desc = tensor_ops::matmul2d_descriptor(64, 32, 0);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    matmulOp.run(a, b, c);
}
```

Clean. Exactly what we need.

I confirmed the header is accessible:

```python
k = mx.fast.metal_kernel(
    name="test",
    source=src,
    header="""
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp;
    """,
)
# Compiles. Header is reachable.
```

And Metal 4 is available:

```
Metal version: 400.0   (M4, macOS Sequoia)
```

Then I tried to actually use `matmul2d`. Three blockers, in order of discovery:

**Blocker 1: Type support.**

Table 7.3 of the spec lists supported type combinations. `float/float/float` is listed — but when I tried it:

```
static_assert failed: "Unsupported type"
```

Table 7.4 clarifies: `bfloat/bfloat/bfloat` and several mixed-precision combinations require **OS 26.1 and later**. That's iOS/macOS naming — it maps to macOS 26.1 (not released yet as of this writing). The `float/float/float` path in Table 7.3 is supported, but only with certain `execution_scope` + K-dimension combinations that are hardware-dependent.

**Blocker 2: `tensor_handle` vs `tensor_inline`.**

The spec's `matmul2d` example uses tensors declared as kernel parameters with `[[buffer(N)]]` attributes — these are `tensor_handle` type. MLX's `metal_kernel` generates the function signature automatically: it only creates raw pointer buffers (`const device float* a [[buffer(0)]]`), not `tensor<device half, ..., tensor_handle>` parameters.

The only tensor type you can construct at runtime from a pointer is `tensor_inline`. But `cooperative_tensor.store()` only accepts `tensor_handle` targets for device memory writes. The round-trip `cooperative_tensor → tensor_inline → device output` is blocked:

```
error: candidate template ignored: could not match 'tensor_handle' against 'tensor_inline'
```

**Blocker 3: Dynamic K hangs the GPU compiler.**

When I tried `K=0` (dynamic length, matching the spec example exactly), the MLX JIT compilation hung. The TensorOps template instantiation with `dynamic_length_v<int>` appears to trigger a very long (possibly infinite) compile path under MLX's inline Metal JIT. The process never returned.

**Summary:** TensorOps is architecturally incompatible with MLX's `mx.fast.metal_kernel` API. The API generates raw pointer buffers; TensorOps requires tensor-typed formal parameters. The mismatch is fundamental, not a workaround.

---

## What Actually Worked

Two improvements from the spec research did ship.

### 1. `metal::precise::exp` — a correctness fix hiding as a performance question

Section 8.2 of the spec describes rounding mode. Section 8.3 covers floating-point exceptions. Table 8.2 documents accuracy under fast math.

The relevant line: `exp()` in fast math mode (`-fmetal-math-mode=fast`) does not guarantee `exp(-INFINITY) = 0.0`. The spec's ULP table for fast math lists relaxed accuracy bounds for transcendentals.

Our kernel uses `exp(score - running_max)` for the online softmax. When a lane is masked (causal or sliding-window), we set `score = -INFINITY`. In fast math mode, `exp(-INFINITY)` may not be exactly `0.0` — which would corrupt the softmax denominator.

The fix: use the `metal::precise::` namespace to force IEEE-compliant `exp` regardless of compiler math mode:

```metal
// Before (math-mode dependent):
float w = exp(score - m_new);

// After (always correct):
float w = metal::precise::exp(score - m_new);
```

MLX's `metal_kernel` API has no parameter for compiler flags, so `-fmetal-math-mode=relaxed` isn't accessible. The namespace workaround is better anyway — it's surgical, affects only these two `exp` calls, and documents intent in the code.

### 2. `simd_broadcast_first` — eliminating two threadgroup barriers per tile

Section 6.9.2 of the spec (Table 6.14) lists the full SIMD-group permute function set. One entry:

```
simd_broadcast_first(x)  →  broadcasts lane 0's value to all lanes
                             without a threadgroup barrier
```

The original kernel used threadgroup memory to share the running max and rescale factor:

```metal
// Before: two threadgroup variables, two barriers per tile
if (lane == 0) {
    tg_m_shared = m_new;
    tg_factor   = factor;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
m_new  = tg_m_shared;
factor = tg_factor;
```

With `simd_broadcast_first`, both threadgroup variables disappear entirely — `running_m` becomes a lane-local float that all 32 lanes keep synchronized:

```metal
// After: no threadgroup variables, no barriers for scalar sharing
float m_new  = simd_broadcast_first(max(running_m, tile_max));
float factor = simd_broadcast_first(
    isfinite(running_m) ? metal::precise::exp(running_m - m_new) : 0.0f);
running_m = m_new;
```

With S_kv=4096 and 128 tiles, this removes 256 threadgroup barriers from the hot loop. Threadgroup barriers are expensive — they serialize the entire threadgroup and flush threadgroup memory. Removing them reduces both latency and the register pressure from storing shared state.

Both of these are in the current kernel. 9 parity tests pass. The improvements are real even if the end-to-end situation hasn't changed.

---

## The Actual Answer to "What Is Section 7.2 Useful For?"

TensorOps would be transformative **if** MLX supported tensor-typed kernel parameters. The current `mx.fast.metal_kernel` API exposes only raw device pointers — the `[[buffer(N)]]` binding that TensorOps needs is auto-generated as `const device float*`, not `tensor<device half, ..., tensor_handle>`.

To use TensorOps for our LUT precompute, MLX would need one of:

1. **Support `tensor<>` as a formal parameter type** in `metal_kernel`'s auto-generated signature. Something like `input_tensor_types=[("a", mx.float16, 2)]` that generates `tensor<device half, dextents<int,2>> a [[buffer(0)]]`.

2. **A new `mx.fast.metal_tensor_kernel` variant** that accepts tensor operands natively and dispatches via TensorOps internally.

This is exactly the GitHub issue we filed at [ml-explore/mlx](https://github.com/ml-explore/mlx). The issue covers three requests — compiler options, integer template parameters, and Metal 4 tensor type access — all confirmed by direct testing.

---

## The Broader Pattern

Three sessions, three attempts at the LUT precompute, three different techniques:

| Attempt | Technique | Result |
|---|---|---|
| Original | Scalar loop, 32 lanes stripe independently | Baseline |
| Attempt 1 | `simdgroup_float8x8`, cooperative 8×8 tiles | 20 µs slower — protocol overhead wins |
| Attempt 2 | TensorOps `matmul2d`, hardware tensor units | API incompatible with MLX's kernel wrapper |

The pattern: each attempt was technically sound, correctly implemented, and blocked by something orthogonal to the GPU math.

- Simdgroup matrix: the hardware works, the tile size is wrong.
- TensorOps: the hardware works, the API binding doesn't exist.

In both cases, the blocker wasn't that the hardware was slow. The blocker was that the **interface** between our code and the hardware had a constraint we couldn't see until we hit it.

The right mental model for GPU kernel work on Apple Silicon: there are three layers — the math you want to do, the hardware that can do it, and the API that connects them. Breakthroughs happen at the API layer, not the math layer. The math for attention has been solved. The hardware for matrix multiply has been built. The gap is the binding.

That gap is the GitHub issue. If MLX adds `tensor<>` support to `metal_kernel`, this whole investigation becomes a one-afternoon project. Until then, the scalar LUT is the fastest thing we can write.

---

## What Is in the Library Now

[veloxquant_mlx/metal/fused_sdpa.py](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/metal/fused_sdpa.py) has:

- `metal::precise::exp` for both softmax `exp` calls — correctness guarantee regardless of MLX math mode
- `simd_broadcast_first` replacing threadgroup barriers for `running_m` — 256 fewer barriers at S_kv=4096
- `tg_m_shared` and `tg_factor` threadgroup variables removed — smaller threadgroup memory footprint
- All 9 parity tests passing: causal, non-causal, sliding-window, GQA, short-sequence, long-sequence, dispatcher patch

The end-to-end situation is unchanged from [the previous post](https://medium.com/@rajveer.rathod1301) — the kernel only helps if mlx_lm exposes a way to skip K_hat materialization, which requires an upstream change.

But the kernel is now more correct and slightly better engineered. That's what reading 346 pages of a GPU spec gets you when the hardware feature you wanted is one API version away.

---

## The One Practical Takeaway

Before spending time implementing a GPU optimization, answer this question:

**Which layer is blocking you — the math, the hardware, or the API?**

If the math is solved and the hardware exists, the answer is almost always the API. Find the API gap first. File the issue or write the binding. Don't write the kernel until the API exists to call it from.

I wrote the kernel first. I found the API gap last. Three sessions later.

---

## Links

- VeloxQuant-MLX on PyPI: [pypi.org/project/VeloxQuant-MLX](https://pypi.org/project/VeloxQuant-MLX)
- GitHub: [github.com/rajveer43/VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX)
- The kernel: [veloxquant_mlx/metal/fused_sdpa.py](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/metal/fused_sdpa.py)
- MLX issue filed: [github.com/ml-explore/mlx/issues](https://github.com/ml-explore/mlx/issues)
- Previous post (Phase 2 mistake): [Medium — I Spent 8 Hours Writing a FlashAttention Kernel](https://medium.com/@rajveer.rathod1301)
- Previous post (Phase 1 win): [Medium — I Wrote a Metal Kernel to Stop My Mac From OOMing](https://medium.com/@rajveer.rathod1301)

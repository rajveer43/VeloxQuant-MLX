---
slug: metal-kernels
title: "I Wrote a Metal Kernel to Stop My Mac From OOMing"
date: 2026-05-25
authors: rajveer
tags: [metal, apple-silicon, mlx, gpu, performance]
---
# I Wrote a Metal Kernel to Stop My Mac From OOMing on LLM Inference — Here's a 13× Speedup and 98% Memory Reduction

*How a 30-line Metal compute shader replaced the worst hot path in VeloxQuant-MLX 0.5.1, what I learned about Apple Silicon kernel launch overhead, and why this matters if you run LLMs locally on Mac.*

---

## The Bug That Wouldn't Die

A few weeks back I shipped VeloxQuant-MLX 0.5.0 — a Python library that compresses the KV cache for any model you load through `mlx_lm`. The headline algorithm is **VecInfer**, which uses product vector quantization to squeeze keys down to 1 bit per element. That is **16× compression**. Sounds great.

It worked beautifully on Llama-3.1-8B, Mistral-7B, Qwen2.5-7B, Phi-4 — every model with `head_dim=128`. And then I tested Falcon3-7B.

```
[VecInfer-2bit] generating...
  Out of memory: requested 712 MB, available 0
```

Falcon3-7B has `head_dim=256`. The chunked nearest-centroid search at the heart of `quantize_vq` allocates a tensor of shape `[chunk_size, n_centroids, sub_dim]` on every chunk. For Falcon's geometry that's a multi-hundred-megabyte intermediate — at every single token, on every layer, on every step. The GPU runs out of memory before generating a single token.

I shipped 0.5.0 with the OOM marked as a known limitation. It bothered me. I knew the fix conceptually — accumulate the squared distance in registers, never materialize the diff matrix — but doing that meant writing a Metal compute shader, and I had never written one.

This post is what happened when I did.

---

## What Even Is a KV Cache And Why Should You Care

Quick recap. Every transformer layer needs to remember the keys and values it computed for every token it's already seen. For a 7B model with 32 layers, 8 KV heads, and head_dim=128, generating an 8,000-token response means storing:

```
32 layers × 8 heads × 8000 tokens × 128 dims × 2 (K + V) × 2 bytes (fp16)
≈ 1 GB
```

On a 16 GB MacBook running the model weights (~5 GB at 4-bit) plus the OS and your app, that 1 GB is the difference between a fluent response and a hard crash. **The KV cache is the silent killer of long-context inference on Mac.**

KV-cache *quantization* — storing those keys and values at fewer bits — is the answer. There are several flavors. The aggressive one I shipped, VecInfer, uses **product vector quantization**:

1. Split each `[head_dim]` key vector into small sub-vectors of length `sub_dim` (typically 4 or 8).
2. Pre-train a codebook of K-means centroids on calibration data.
3. At inference, encode each sub-vector as the index of its nearest centroid.

A 128-dim fp16 key (256 bytes) becomes 16 indices at 8 bits each (16 bytes). That's the 16× compression.

The hot operation is step 3: finding the nearest centroid. On every layer, on every token, you do a vectorized argmin against the codebook. That's `quantize_vq`.

---

## What `quantize_vq` Was Doing Wrong

Here's what the pure-MLX implementation looks like (paraphrased):

```python
def quantize_vq(x, codebook, sub_dim):
    # x: [N, sub_dim]    -- the sub-vectors to encode
    # codebook: [n_centroids, sub_dim]
    diff = x[:, None, :] - codebook[None, :, :]  # [N, n_centroids, sub_dim]
    d2 = mx.sum(diff * diff, axis=-1)  # [N, n_centroids]
    return mx.argmin(d2, axis=-1)  # [N]
```

That `diff` tensor is the killer. Its shape is `[N, n_centroids, sub_dim]`. For Falcon3-7B-shape inputs:
- `N = 4096 tokens × 4 KV heads × 64 sub-vectors per head = 1,048,576`
- `n_centroids = 256`
- `sub_dim = 4`
- Total: 1,048,576 × 256 × 4 × 2 bytes (fp16) = **2.1 GB intermediate**

The implementation tries to mitigate this by chunking N — processing 4,096 sub-vectors at a time — but even one chunk is still ~32 MB, and a 7B model's GPU memory pressure means even that gets fragmented and OOMs in practice.

What you actually want is for each thread to compute the argmin **in registers**, only writing out a single uint32 index. No intermediate tensor. Total intermediate memory: zero.

That's exactly what a Metal compute kernel can do.

---

## What Is MLX `mx.fast.metal_kernel`?

MLX (Apple's array library for Apple Silicon) has a feature most people don't know about: `mx.fast.metal_kernel`. It lets you write a Metal Shading Language function inline as a Python string and have MLX JIT-compile it, manage the buffer bindings, and dispatch it on the GPU.

The whole thing takes a few lines of Python:

```python
kernel = mx.fast.metal_kernel(
    name="vecinfer_quantize",
    input_names=["x", "codebook"],
    output_names=["out"],
    source=METAL_SOURCE,  # a string of MSL
)

result = kernel(
    inputs=[x, codebook],
    output_shapes=[(N,)],
    output_dtypes=[mx.uint32],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
)
```

MLX handles all the boilerplate: function signature generation, dtype binding, threadgroup memory, dispatch encoding. You write the kernel body. It's the easiest GPU programming experience I've ever had — closer to writing a Python function than to traditional CUDA.

---

## The Kernel: 18 Lines of MSL

Here's the entire fused-argmin kernel that replaces that 2 GB intermediate tensor:

```metal
uint vec_idx = thread_position_in_grid.x;
uint N_total = x_shape[0];
if (vec_idx >= N_total) {
    return;
}

uint n_centroids = codebook_shape[0];
uint sub_dim     = codebook_shape[1];
uint x_base      = vec_idx * sub_dim;

// Track running argmin in registers — never materialize the diff matrix.
float best_dist = INFINITY;
uint  best_idx  = 0;

for (uint c = 0; c < n_centroids; ++c) {
    uint cb_base = c * sub_dim;
    float dist = 0.0f;
    for (uint i = 0; i < sub_dim; ++i) {
        float d = float(x[x_base + i]) - float(codebook[cb_base + i]);
        dist += d * d;
    }
    if (dist < best_dist) {
        best_dist = dist;
        best_idx  = c;
    }
}

out[vec_idx] = best_idx;
```

That's it. Each GPU thread handles one sub-vector. It loops over all centroids, accumulates squared distance in a single float register, tracks the running minimum, and writes one uint32 index. The intermediate "diff matrix" never exists anywhere except in those two register-resident floats per thread.

Memory complexity: `O(N)` total output, vs `O(N × n_centroids × sub_dim)` for the Python path.

---

## The Numbers

I wrote a benchmark script — `scripts/plot_metal_benchmarks.py` in the repo — that runs both paths across realistic shapes and saves figures. Here are the results.

### Throughput: 6.9–14.7× Speedup

| Shape | pure-MLX | Metal | Speedup |
|---|---:|---:|---:|
| S=128, D=128 | 3.64 ms | 0.53 ms | **6.9×** |
| S=512, D=128 | 13.5 ms | 1.26 ms | **10.7×** |
| S=2048, D=128 | 55.1 ms | 4.18 ms | **13.2×** |
| S=8192, D=128 | 228.6 ms | 15.6 ms | **14.7×** |
| S=1024, D=256 | 27.0 ms | 2.23 ms | **12.1×** |
| S=4096, D=256 | 108.8 ms | 7.98 ms | **13.6×** |

The speedup scales with sequence length — longer contexts (where the Python path is bandwidth-bound on those huge diff tensors) get bigger wins. At `S=8192, D=128` we go from 228 ms per call to 16 ms per call. Per call. Multiply by 32 layers × 1 quantize per step × hundreds of tokens and you're talking minutes saved per long generation.

### Memory: 729 MB → 12 MB

At the Falcon3-7B OOM trigger shape (`head_dim=256, n_centroids=256, sub_dim=4, S=4096`):

| Path | Peak memory |
|---|---:|
| Pure-MLX `quantize_vq` | **729.3 MB** |
| Metal `vecinfer_quantize_metal` | **12.0 MB** |
| Reduction | **98.4%** (saved 717 MB) |

This is the result that matters. The kernel doesn't just make existing models faster — it makes models that previously OOMed actually run.

### Correctness: Bit-Exact on fp32, MSE-Identical on fp16

This is where I had to be careful. The Metal kernel and the pure-MLX path don't produce identical indices on fp16 inputs — about **0.1% of indices differ**.

Why? When two centroids are nearly equidistant from a point, the choice of "nearest" depends on the order of floating-point operations. The pure-MLX path does the subtract in fp16 (because the inputs are fp16); the Metal kernel promotes to fp32 inside the inner loop. When the tiebreaker happens at the 5th decimal place, the two paths pick different winners.

But here's the thing: **the reconstruction quality is identical**. I validated this by reconstructing keys from both index sets and measuring MSE against the original input:

```
B=1 H=8 S=2048 D=128 sub_dim=8 n_c=256 dtype=float16
  idx_diff = 0.104%
  mse_ref = 3.7211e-01    mse_metal = 3.7211e-01
  rel_err = 5.61e-07
```

Reconstruction MSE matches to **7 decimal places**. The two paths produce functionally identical compressed representations — they just disagree on which arbitrary tie-breaker to pick.

The parity tests in `veloxquant_mlx/tests/cache/test_vecinfer_metal_parity.py` validate this directly: assert that reconstruction MSE is within 1% relative error, not that indices match.

---

## What I Got Wrong on the First Try

I want to be honest about the missteps, because they're the actually interesting part.

### Mistake 1: I Wrote the Dequant Kernel First

My first instinct was to write a Metal kernel for `dequantize_vq` — the operation that takes codebook indices and reconstructs the float vectors. It's conceptually simpler (just a gather), so I started there.

After getting bit-exact correctness, I benchmarked it:

```
shape                                pure-mlx     metal    speedup
B=1 H=8 S=128 n_sub=16 sub_dim=8       223.3 µs   185.6 µs   1.20x
B=1 H=8 S=512                          183.6 µs   209.3 µs   0.88x
B=1 H=8 S=2048                         258.3 µs   275.9 µs   0.94x
B=1 H=8 S=8192                         467.8 µs   577.6 µs   0.81x
```

**My kernel was slower than MLX's `mx.take`.** That stung. After staring at the numbers for an hour, the reason became obvious: MLX's `mx.take` is already a highly tuned Metal gather kernel under the hood. There is no "Python overhead" to eliminate. The pure-MLX path *is* a Metal kernel. My kernel was duplicating it badly.

**The lesson:** before writing a custom kernel, profile to find the operation that has actual Python/intermediate-tensor overhead. `mx.take` does not. `quantize_vq` does, because of the `[N, n_centroids, sub_dim]` materialization. The 30-line MSL shader had to fuse an *algorithm* — argmin over distances — not just replace a builtin.

I kept the dequant kernel as a building block for Phase 2 (fused dequant+SDPA), but the headline result is the quantize kernel.

### Mistake 2: Wrong Threadgroup Layout

My first quantize kernel dispatched **one thread per (input_vector, sub_dim_component)** pair. That made each thread tiny — one subtract, one square, one accumulate — and meant launching `N × sub_dim` threads. For typical shapes, that's millions of threads.

Apple Silicon GPUs have 32-wide SIMD groups and an internal cost per thread launch. Launching 8× more threads than you need is pure overhead.

The fix was to dispatch **one thread per input vector** and let each thread loop over all sub_dim components in registers. Same total work, 8× fewer thread launches, much better register reuse. That's the layout in the kernel above.

### Mistake 3: I Assumed End-to-End Would Always Be Faster

After validating the kernel was 13× faster on synthetic shapes, I ran the full benchmark on SmolLM2-135M (a 135-million-parameter tiny model) expecting to see a speedup in end-to-end token generation.

I got the opposite. The Metal path was **slower** end-to-end — 75 tok/s vs 178 tok/s for the pure-MLX path.

The reason: Metal kernel dispatch has a fixed per-launch overhead of roughly 50–200 µs on Apple Silicon. SmolLM2 has 30 layers, each doing 2 quantize calls per token, so that's ~60 kernel launches per generated token. The per-launch overhead exceeded the work each kernel did.

**The kernel is designed for the regime where it matters: 7B+ models with realistic context lengths, where each `quantize_vq` call is doing milliseconds of work.** On those, the launch overhead is negligible relative to the kernel runtime, and you get the full 10–14× speedup.

This is a limitation of MLX's kernel launch path — MLX doesn't yet expose a way to amortize launch overhead across multiple layers in a single dispatch. That's a Phase 3 problem and probably out of scope for a Python-level library.

---

## How to Use This Today

VeloxQuant-MLX 0.5.1 is on PyPI. Install:

```bash
pip install --upgrade VeloxQuant-MLX
```

The Metal kernels are **on by default** when available. No code changes needed. Your existing `VecInferKVCache` calls auto-detect Metal and use the fast path:

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheFactory

model, tokenizer = mlx_lm.load("mlx-community/Falcon3-7B-Instruct-4bit")

# Metal auto-detected. To force off for debugging: use_metal_kernels=False
config = KVCacheConfig(
    method="vecinfer",
    head_dim=256,
    key_sub_dim=4,
    value_sub_dim=4,
    key_codebook_bits=8,
    value_codebook_bits=8,
    smooth_factors=calibrated_smooth_factors,
    key_codebook=calibrated_key_codebook,
    use_metal_kernels=None,  # None = auto, True = require, False = forbid
)
```

The new `use_metal_kernels` flag is three-state:
- `None` (default) — auto-detect; use Metal if available, silently fall back if not
- `True` — require Metal; raise at construction time if unavailable
- `False` — forbid Metal; use the pure-MLX path (for parity testing and debugging)

To verify the speedup on your own machine:

```bash
git clone https://github.com/rajveer43/VeloxQuant-MLX
cd VeloxQuant-MLX
PYTHONPATH=. python scripts/plot_metal_benchmarks.py
# Produces figures/metal/summary.png with your hardware's numbers
```

---

## What's Next: Phase 2

The quantize kernel is the biggest single win, but it's not the end. **Phase 2 is fusing dequantize + scaled-dot-product-attention** into a single kernel.

Right now, even with Phase 1, the cache still materializes the full fp16 key tensor on every attention call. The dequant happens — efficiently, since `mx.take` is already fast — but we hold the result in GPU memory long enough to pass it to MLX's SDPA. For very long contexts, that materialized key tensor is still significant memory pressure.

The Phase 2 kernel would:
1. Take codebook indices, the per-query LUT (`q_tilde @ codebook.T`), and value indices
2. Compute attention scores directly via LUT lookup, never materializing fp16 keys
3. Compute the softmax-weighted value sum in-kernel
4. Output the final attention result in one fused pass

This is what the VecInfer paper's CUDA kernel does. Porting it to Metal is the goal. If you've written Metal compute shaders before and want to collaborate, the GitHub issue is open.

---

## The Meta-Lesson: Custom Kernels Are More Accessible Than You Think

I had never written a Metal shader before this project. The mental model is straightforward once you get past the syntax:

1. **Identify the operation with materialization overhead** (not just a slow Python loop — those are usually wrapped in optimized C++ already; look for operations that create big intermediate tensors)
2. **Write the algorithm with the intermediate as register-state instead of memory-state** (running min, running sum, running argmin)
3. **Dispatch one thread per output element**, not per input or per output-component
4. **Validate with reconstruction error**, not bit-exact equality, when fp16 is involved
5. **Benchmark at realistic shapes**, not toy shapes — kernel launch overhead can dominate for small workloads

Total time investment for this Phase 1: about 6 hours of focused work, including the two failed approaches above. The resulting kernel unblocks `head_dim=256` models that previously OOMed, gives a 10–14× speedup on the hot path, and is 30 lines of MSL.

If you've been hesitant to write custom GPU kernels because it sounds intimidating — `mx.fast.metal_kernel` makes the bar way lower than it used to be on CUDA. Try it.

---

## TL;DR

- VeloxQuant-MLX 0.5.1 adds a Metal compute kernel for `quantize_vq`, the hot path in VecInfer KV-cache compression
- **13× faster** on realistic shapes (S=2048+)
- **98% less peak memory** at the Falcon3-7B OOM trigger configuration
- **Drop-in, zero API change** — auto-detected when Metal is available
- Free, MIT-licensed, on PyPI: `pip install VeloxQuant-MLX`
- The kernel is 30 lines of Metal Shading Language inside Python
- Phase 2 (fused dequant+SDPA attention kernel) is next

GitHub: [github.com/rajveer43/VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX)
PyPI: [pypi.org/project/VeloxQuant-MLX](https://pypi.org/project/VeloxQuant-MLX)
Benchmark figures: [`figures/metal/summary.png`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/figures/metal/summary.png) in the repo

If this saves your Mac from OOMing tonight, leave a star — or open an issue if it doesn't.

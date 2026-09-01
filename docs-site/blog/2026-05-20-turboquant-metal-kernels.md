---
slug: turboquant-metal-kernels
title: "TurboQuant + Metal Kernels: The Combined Writeup"
description: "Five custom Metal kernels for TurboQuant KV cache quantization on Apple Silicon, including MLX dispatch gotchas and a parallel Hadamard transform bug."
date: 2026-05-20
authors: rajveer
tags: [metal, apple-silicon, mlx, gpu, turboquant, performance]
---
# Building Custom Metal Kernels for LLM KV Cache Compression on Apple Silicon

*How I wrote five hand-tuned Metal compute kernels in MLX for TurboQuant — and what every bug taught me about Apple GPU programming.*

---

## The Problem

My Mac was choking on long-context LLM inference.

Not because the model was too large — I had already quantized the weights. The bottleneck was the **KV cache**. At 8k context, a single layer's key cache is `[1, 32, 8192, 128]` in fp16 — over 67 MB per layer, 2 GB across 32 layers. On Apple Silicon, where the GPU and CPU share the same physical memory, that pressure is immediate and painful.

VeloxQuant-MLX already had several compression algorithms: TurboQuantRVQ (7.5× via two-stage scalar RVQ), QJL (16× via 1-bit Johnson-Lindenstrauss sketching), and VecInfer (16× via product VQ). But they were all running through pure MLX graph operations — no custom GPU kernels. The hot paths were either slow or allocating huge intermediate tensors.

The fix: write the hot paths in **Metal Shading Language** and JIT-compile them via `mx.fast.metal_kernel`.

This is the story of how I did it, what broke, and what I learned.

---

## The Stack

Before diving into the kernels, here's the relevant context:

- **MLX** — Apple's NumPy-style ML framework with lazy evaluation and Metal GPU backend
- **`mx.fast.metal_kernel`** — Python API to write raw Metal Shading Language compute shaders that plug into MLX's lazy graph
- **TurboQuant** — a family of KV cache quantization algorithms (MSE, Prod, RVQ) implemented in VeloxQuant-MLX
- **QJL** — Quantized Johnson-Lindenstrauss: compress keys to 1-bit sign sketches + a scalar norm

The goal was to replace the slowest pure-MLX operations with Metal kernels that live in five focused submodules:

| Submodule | What it does |
|---|---|
| `_bit_packing.py` | Pack/unpack b-bit indices into uint8 bytes |
| `_scalar_quant.py` | Nearest-centroid quantize, dequantize, fused Hadamard+quant |
| `_qjl.py` | QJL sign encode and inner product scoring |
| `_rvq_attend.py` | Fused RVQ key decode + FlashAttention-style online softmax |

---

## How `mx.fast.metal_kernel` Works

Before showing any kernel code, there's one thing you need to understand about the API — because getting it wrong produces silent, subtle bugs.

### The API in 30 seconds

```python
import mlx.core as mx

kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["x", "y"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        out[i] = x[i] + y[i];
    """,
)

result = kernel(
    inputs=[a, b],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(N,)],
    output_dtypes=[mx.float32],
)
```

The `source` string is raw Metal Shading Language — no `kernel` keyword, no function signature. MLX wraps it. Shape information is injected automatically: inside the kernel, `x_shape[0]` gives you the first dimension of `x`.

### The #1 Gotcha: Grid = Total Threads

This is the single most important thing to get right, and the MLX documentation is easy to misread on this point.

In standard Metal (Obj-C / Swift), you call `dispatchThreadgroups(n_groups, threadsPerThreadgroup: tg_size)` — so the grid is in *threadgroup* units.

**MLX uses `dispatchThreads` — the grid is in *total thread* units.**

That means if you want B threadgroups of T threads each:

```python
# WRONG — only dispatches 1 thread per threadgroup
grid=(B, 1, 1), threadgroup=(T, 1, 1)

# CORRECT — dispatches B threadgroups of T threads each
grid=(B * T, 1, 1), threadgroup=(T, 1, 1)
```

I made this mistake on four out of five kernels. The symptom was identical every time: **only the first batch element had correct output; everything else was zero**. It looked like a memory layout bug or an indexing error, not a dispatch error. I spent hours debugging before I found it.

### The Lazy Graph Contract

`mx.fast.metal_kernel` returns a lazy node — nothing runs until `mx.eval()` is called. `mx.eval()` internally:

1. Encodes the compute command into a `MTLCommandBuffer`
2. Calls `commandBuffer.commit()` to submit to the GPU
3. Calls `commandBuffer.waitUntilCompleted()` to synchronize

You never write any of this yourself. MLX owns the entire Metal command buffer lifecycle.

---

## Kernel 1: Bit-Packing — 30× Over NumPy

### The problem

TurboQuantRVQ stores KV cache keys as b-bit indices (b ∈ \{1, 2, 4\}). The pure-Python path used a loop to pack these into uint8 bytes. At 65k elements it was ~8 ms — unacceptable.

### The kernel

```metal
constexpr int  ELEMS_PER_BYTE = 8 / B_BITS;
constexpr uint MASK           = (1u << B_BITS) - 1u;

uint byte_idx = thread_position_in_grid.x;
uint base     = byte_idx * ELEMS_PER_BYTE;

uint packed_byte = 0u;
for (int i = 0; i < ELEMS_PER_BYTE; ++i) {
    uint val = uint(indices[base + i]) & MASK;
    packed_byte |= (val << (i * B_BITS));
}
packed[byte_idx] = uint8_t(packed_byte);
```

One thread per output byte. `B_BITS` is a **template parameter** — a compile-time integer constant. This lets the compiler statically unroll the inner loop (2 iterations for b=4, 4 for b=2, 8 for b=1) and inline the constants.

The dispatch:

```python
grid = ((n_bytes, 1, 1),)
threadgroup = ((min(256, n_bytes), 1, 1),)
```

### Results

| N | NumPy | Metal | Speedup |
|---|---|---|---|
| 4,096 | 0.52 ms | 0.18 ms | 2.9× |
| 16,384 | 2.1 ms | 0.17 ms | 12.5× |
| 65,536 | 8.4 ms | 0.28 ms | **29.5×** |

The kernel dispatch overhead is ~0.17 ms regardless of N. Below ~2k elements NumPy wins because there's nothing to hide the launch cost behind. Above 16k elements, Metal dominates by an order of magnitude.

---

## Kernel 2: Scalar Quantize / Dequantize — 11× Over NumPy

### The problem

TurboQuantMSE quantizes each key dimension independently against a Lloyd-Max codebook. The pure-MLX path computed `|x - centroids|²` as a full `[N, 2^b]` matrix, then took `argmin` — allocating a tensor that was `2^b` times the input size.

### The quantize kernel

```metal
constexpr int N_CENTS = 1 << B_BITS;

uint  elem      = thread_position_in_grid.x;
float val       = float(x[elem]);
int   best      = 0;
float best_dist = INFINITY;

for (int j = 0; j < N_CENTS; ++j) {
    float d    = val - centroids[j];
    float dist = d * d;
    if (dist < best_dist) { best_dist = dist; best = j; }
}
indices[elem] = uint8_t(best);
```

One thread per element. The centroid scan lives entirely in registers — no intermediate allocation. With `B_BITS` as a template, the loop body is known at compile time: the compiler generates 2, 4, 8, or 16 iterations of straight-line code.

### The dequantize kernel

Even simpler — a pure gather:

```metal
uint elem   = thread_position_in_grid.x;
x_hat[elem] = half(centroids[uint(indices[elem])]);
```

### Results

| N | NumPy argmin | Metal | Speedup |
|---|---|---|---|
| 16,384 | 0.21 ms | 0.17 ms | 1.2× |
| 65,536 | 0.86 ms | 0.19 ms | 4.5× |
| 262,144 | 3.5 ms | 0.31 ms | **11.3×** |

---

## Kernel 3: Fused Hadamard + Quantize — The Hardest One

### The problem

TurboQuantMSE (with Hadamard preconditioner) runs:

```
y = diag * H * x / sqrt(D)    [randomized Hadamard rotation]
idx = argmin_k |y - c_k|²     [nearest-centroid quantize]
```

Two separate dispatches, with a `[B, D]` fp16 intermediate between them. Fusing them into one kernel eliminates that allocation and the round-trip to GPU memory.

### The kernel design

Walsh-Hadamard Transform (WHT) is an in-place butterfly — each pass halves the stride. On GPU, D threads share a threadgroup, and each butterfly step needs a barrier.

```metal
threadgroup float buf[MAX_D];   // static threadgroup memory; MAX_D injected at compile time

// 1. Load + diagonal sign flip
float v = float(x[tg * D + lane]);
v *= float(diag[lane]);
buf[lane] = v;
threadgroup_barrier(mem_flags::mem_threadgroup);

// 2. In-place WHT — range-based parallel butterfly
for (uint stride = 1; stride < D; stride <<= 1) {
    uint local    = lane % (stride << 1u);
    bool is_upper = local >= stride;
    uint partner  = is_upper ? (lane - stride) : (lane + stride);
    float a = buf[lane];
    float b = buf[partner];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[lane] = is_upper ? (b - a) : (a + b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 3. Scale
float y = buf[lane] * metal::rsqrt(float(D));

// 4. Nearest-centroid argmin (register-local)
int   best      = 0;
float best_dist = INFINITY;
for (int j = 0; j < N_CENTS; ++j) {
    float d    = y - centroids[j];
    float dist = d * d;
    if (dist < best_dist) { best_dist = dist; best = j; }
}
indices[tg * D + lane] = uint8_t(best);
```

The threadgroup array `buf[MAX_D]` requires `MAX_D` to be a compile-time constant — which is why it's injected as a `#define` in the kernel header:

```python
_hadamard_quantize_kernel = mx.fast.metal_kernel(
    ...
    header=f"#define MAX_D {D}\n",
    source=_HADAMARD_QUANTIZE_SRC,
)
```

### The butterfly bug

My first implementation used:

```metal
uint partner = lane ^ stride;    // XOR butterfly
```

This looked right — it's the standard Cooley-Tukey bit-reversal trick. But on GPU, it produced ~90% index mismatch vs the sequential reference.

The problem: `lane ^ stride` traverses the WHT in **bit-reversal order**, which is fine for sequential execution (because you can reorder the output at the end), but on GPU where lanes run simultaneously, XOR pairing creates **data races** within a butterfly pass — some lanes read values that other lanes in the same pass are simultaneously writing.

The fix is a **range-based butterfly** that unambiguously partitions each pass into non-overlapping upper/lower pairs:

```metal
uint local    = lane % (stride << 1u);
bool is_upper = local >= stride;
uint partner  = is_upper ? (lane - stride) : (lane + stride);
float a = buf[lane];
float b = buf[partner];          // read BEFORE the barrier write below
threadgroup_barrier(mem_flags::mem_threadgroup);
buf[lane] = is_upper ? (b - a) : (a + b);
```

Reading `a` and `b` before the barrier guarantees both values come from the previous pass. After this fix, 100% of indices matched the reference.

### Grid

The grid uses B threadgroups of D threads — **not** B × D total:

```python
# Wrong:
grid=(B, 1, 1), threadgroup=(D, 1, 1)   # only 1 thread per threadgroup!

# Correct:
grid=(B * D, 1, 1), threadgroup=(D, 1, 1)   # B threadgroups of D threads
```

---

## Kernel 4: QJL Encode — Simdgroup Sign Packing

### The problem

QJL encoding requires:
1. For each key vector `x[b]`, compute `sign(S @ x[b])` for all m sketch dimensions — giving m bits
2. Pack those m bits into m/8 uint8 bytes (LSB-first)
3. Compute `‖x[b]‖` (one scalar per key)

The pure-MLX path materialized the full `[B, m]` float matrix `S @ x.T` before sign-taking — `m * d * B * 4` bytes, growing linearly with batch and sketch size.

### Simdgroup design

Each simdgroup (32 lanes) handles 32 consecutive sketch dimensions. Lane `j` computes `dot(S[simd_blk*32 + j, :], x[b, :])` via a scalar loop:

```metal
uint b_idx    = flat_tg / n_simd_per_batch;
uint simd_blk = flat_tg % n_simd_per_batch;
uint sketch_j = simd_blk * 32u + lane;

float dot_val = 0.0f;
if (sketch_j < m) {
    uint S_row = sketch_j * d;
    uint x_row = b_idx   * d;
    for (uint i = 0; i < d; ++i) {
        dot_val += float(S[S_row + i]) * float(x[x_row + i]);
    }
}
```

After the dot product, all 32 lanes **cooperate to pack 32 sign bits into 4 bytes** using `simd_shuffle`:

```metal
uint sign_bit    = (dot_val >= 0.0f) ? 1u : 0u;
uint byte_in_blk = lane / 8u;
uint bit_in_byte = lane % 8u;

uint packed_byte = 0u;
for (uint bit = 0; bit < 8u; ++bit) {
    uint src = byte_in_blk * 8u + bit;
    packed_byte |= (simd_shuffle(sign_bit, src) << bit);
}

if (bit_in_byte == 0 && sketch_j < m) {
    packed_signs[out_byte] = uint8_t(packed_byte);
}
```

`simd_shuffle(val, lane_id)` broadcasts `sign_bit` from lane `src` to the current lane — no shared memory needed. Lane 0 (of each byte group) does the final write.

The norm is computed cooperatively by simd_blk 0:

```metal
if (simd_blk == 0) {
    float x_sq = 0.0f;
    for (uint i = lane; i < d; i += 32u) {
        float v = float(x[x_row + i]);
        x_sq += v * v;
    }
    float norm_sq = simd_sum(x_sq);
    if (lane == 0) norms[b_idx] = half(metal::sqrt(norm_sq));
}
```

### Grid (the bug, again)

```python
n_simd_per_batch = (m + 31) // 32
n_total_threads  = B * n_simd_per_batch * 32   # ← must multiply by 32
grid=(n_total_threads, 1, 1), threadgroup=(32, 1, 1)
```

Without the `* 32`, only `B * n_simd_per_batch` total threads dispatched — meaning only the first simdgroup ran, and only the first key had any output.

---

## Kernel 5: Fused RVQ Decode + Attend — Online Softmax Without Materializing K

### The problem

Attention with a quantized KV cache normally requires two dispatches:

1. Decode all compressed keys → `K_hat` tensor `[B, H, S_kv, D]` (fp16, potentially GBs)
2. Run `softmax(q @ K_hat.T / sqrt(D)) @ V`

The `K_hat` tensor is allocated, filled, used once, and thrown away. For RVQ keys this is unavoidable in the two-dispatch design — but we can fuse everything into a single FlashAttention-style pass that decodes keys **on the fly** without ever materializing `K_hat`.

### Design

Each threadgroup handles one query position `(b, h, sq)`. Lanes stripe across the D-dimensional vectors in steps of TG = min(D, 32):

```metal
float running_m = -INFINITY;   // online softmax running max
float running_d = 0.0f;        // online softmax running denominator
float my_out[8];               // per-lane output accumulator
for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;

for (uint sk = 0; sk < S_kv; ++sk) {
    // 1. Decode key on-the-fly: k[i] = cents1[idx1[i]] + cents2[idx2[i]]
    float partial_dot = 0.0f;
    for (uint i = tg_lane; i < D; i += TG) {
        float ki = centroids1[uint(k_indices1[k_off])]
                 + centroids2[uint(k_indices2[k_off])];
        partial_dot += float(q[q_base + i]) * ki;
    }
    float score = simd_sum(partial_dot) * inv_sqrt_d;

    // 2. Online softmax update (Dao et al. FlashAttention)
    float m_new  = metal::max(running_m, score);
    float factor = metal::exp(running_m - m_new);
    float w      = metal::exp(score     - m_new);
    running_d    = running_d * factor + w;
    running_m    = m_new;

    // 3. Rescale and accumulate value
    for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;
    for (uint i = tg_lane; i < D; i += TG) {
        float vi    = float(v_codebook[cb_off]);
        uint  out_i = (i - tg_lane) / TG;
        my_out[out_i] += w * vi;
    }
}

// 4. Normalize and write
for (uint i = tg_lane; i < D; i += TG) {
    uint out_i   = (i - tg_lane) / TG;
    out[out_off] = half(my_out[out_i] / running_d);
}
```

`simd_sum(partial_dot)` broadcasts the full dot product to all lanes in the simdgroup — this is the SIMD-level reduction that gives the correct score without any threadgroup memory.

The local accumulator index `out_i = (i - tg_lane) / TG` is the critical piece: lane 0 owns dims \{0, TG, 2×TG, ...\}, lane 1 owns \{1, TG+1, ...\}, and `out_i` is the position within that lane's private array.

---

## The Benchmarks

After fixing all the dispatch bugs, here are the results on Apple M-series (figures saved to `figures/metal/turboquant_kernels/`):

| Kernel | Peak speedup vs NumPy | Notes |
|---|---|---|
| `turboquant_bit_pack` (b=4, N=65k) | **29.5×** | NumPy loop vs Metal one-thread-per-byte |
| `turboquant_scalar_quantize` (N=256k) | **11.3×** | Eliminates `[N, 2^b]` diff tensor |
| `turboquant_hadamard_quantize` (D=1024) | 1.1× | Fused saves 1 allocation; WHT itself is fast |
| `qjl_encode` (B=256) | 0.2× (small B); ~1× (large B) | `np.packbits` is BLAS-level; Metal overhead dominates at `B<64` |
| `turboquant_fused_rvq_decode_attend` | — | No NumPy baseline (different algorithm) |

**Memory savings** are the bigger story for the RVQ attend kernel — it eliminates the `[B, H, S_kv, D]` fp16 `K_hat` tensor entirely. At `S_kv=4096, H=32, D=128` that's 33 MB per layer, ~1 GB across a 32-layer model, allocated and freed every forward pass.

**1-bit bit-packing alone gives 16× memory compression** on the key cache (1 bit per dimension vs fp16). Combined with the Metal kernel's 30× throughput advantage, the packing/unpacking step goes from a bottleneck to essentially free.

---

## What I Learned

### 1. Grid = total threads is the most common MLX Metal mistake

Every tutorial and reference for Metal uses `dispatchThreadgroups`. MLX uses `dispatchThreads`. These are different. If your output is correct for the first batch element and zero elsewhere, check your grid first.

### 2. XOR butterflies are wrong for parallel WHT

The standard sequential WHT uses `pair = i ^ stride`. On GPU this causes data races within a butterfly pass because multiple threads simultaneously read from and write to overlapping pairs. Use range-based pairing (`local = lane % (stride*2); is_upper = local >= stride`) and read both values before the barrier.

### 3. `simd_sum` and `simd_shuffle` are your first tools, not shared memory

For reductions and broadcasts within a simdgroup (32 lanes), `simd_sum` and `simd_shuffle` are zero-cost compared to `threadgroup_barrier` + shared memory. Design around simdgroups first; only escalate to threadgroup memory when you need communication beyond 32 lanes.

### 4. Template parameters unlock static unrolling

`template <int B_BITS>` turns runtime constants into compile-time constants. The inner loop over centroids becomes 2, 4, 8, or 16 unrolled iterations — no branch, no loop counter. This is how Metal kernels beat NumPy at large N despite higher launch overhead: the arithmetic is genuinely faster.

### 5. You don't manage `commandBuffer`

MLX handles `commandBuffer.commit()` and `commandBuffer.waitUntilCompleted()` inside `mx.eval()`. You never touch Metal command buffers when using `mx.fast.metal_kernel`. This is by design — MLX's lazy graph batches multiple kernel dispatches into one command buffer where possible.

### 6. The launch overhead is real and ~0.17 ms

Every Metal kernel dispatch costs ~0.17 ms regardless of work size. For small N (< ~2k elements), NumPy is faster. For large N (> ~16k), Metal wins by 10–30×. Design your batching strategy accordingly — combine small operations into a single larger kernel rather than dispatching many small ones.

---

## Code Organization

The five kernels are organized into focused submodules under `veloxquant_mlx/metal/`:

```
metal/
├── __init__.py          # lazy re-exports
├── kernels.py           # thin facade — imports from all submodules
├── _bit_packing.py      # turboquant_bit_pack, turboquant_bit_unpack
├── _scalar_quant.py     # turboquant_scalar_quantize, _dequantize, _hadamard_quantize
├── _qjl.py              # qjl_encode, qjl_inner_product
├── _rvq_attend.py       # turboquant_fused_rvq_decode_attend
└── _vecinfer.py         # vecinfer_dequant_metal, vecinfer_quantize_metal, ...
```

Each submodule has its own `_cache: dict = {}` for the kernel singleton pattern — build the `MTLComputePipelineState` once on first call, reuse forever:

```python
def _pack_kernel(b: int):
    key = ("bit_pack", b)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_bit_pack_b{b}",
            input_names=["indices"],
            output_names=["packed"],
            source=_PACK_SRC,
        )
    return _cache[key]
```

`kernels.py` is now a 47-line re-export facade:

```python
from veloxquant_mlx.metal._bit_packing import turboquant_bit_pack, turboquant_bit_unpack
from veloxquant_mlx.metal._scalar_quant import turboquant_scalar_quantize, ...
from veloxquant_mlx.metal._qjl import qjl_encode, qjl_inner_product
from veloxquant_mlx.metal._rvq_attend import turboquant_fused_rvq_decode_attend
```

All 40 tests pass after the restructuring — the facade is transparent to callers.

---

## The Broader Point

Apple Silicon is a genuinely good target for this kind of work. Unified memory means you don't pay PCIe bandwidth to move data between CPU and GPU — the Metal kernel reads the same bytes your Python code just wrote. The simdgroup primitives (`simd_sum`, `simd_shuffle`) are clean and well-documented. And `mx.fast.metal_kernel` makes the iteration loop fast: write Metal source in Python, evaluate, fix, repeat.

The hard part isn't the Metal itself — it's understanding how MLX dispatches kernels. Once you internalize "grid = total threads, not threadgroups" and "lazy graph, so nothing runs until mx.eval()", the rest is straightforward shader programming.

The full source is in [VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX) under `veloxquant_mlx/metal/`. The benchmark script is at `veloxquant_mlx/benchmarks/metal_kernel_benchmark.py` and produces all the figures discussed here.

---

## References

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"
- [FlashAttention (NeurIPS 2022)](https://arxiv.org/abs/2205.14135) — Dao et al., online softmax algorithm
- [Apple MLX](https://github.com/ml-explore/mlx) — the framework
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — simd_sum, simd_shuffle, threadgroup_barrier reference

---

*Code: [github.com/rajveer43/VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX) · Previous post: [I Wrote a Metal Kernel to Stop My Mac From OOMing on LLM Inference](/blog/metal-kernels)*

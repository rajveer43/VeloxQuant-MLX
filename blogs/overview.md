# Why llama.cpp, Ollama, and LM Studio Can't Fix Your Mac's Memory Problem — And What I Built to Fix It

> **TL;DR:** The KV cache is the last major unoptimised bottleneck for local LLM inference on Apple Silicon. llama.cpp, Ollama, and LM Studio don't compress it. I built VeloxQuant-MLX to fix that — 9 quantization algorithms, custom Metal GPU kernels, up to 16× compression, plug into `mlx_lm` with 3 lines.

---

## The wall every Mac LLM user hits

You've got a MacBook with an M-series chip. You've downloaded llama.cpp, or Ollama, or LM Studio. You load a 7B model, start a conversation, and for the first few hundred tokens everything feels fast.

Then you push it. A long document. A complex coding task. A multi-turn conversation that's been going for a while. And suddenly — generation slows to a crawl, the fans spin up, or the process just dies.

This isn't a bug in those tools. It's a fundamental memory problem that none of them solve. And it lives in a part of the inference stack that almost nobody talks about: **the KV cache**.

---

## What the KV cache actually is

To understand the problem, you need to understand what happens inside a transformer when it generates text.

Every token in your context window requires the model to compute two matrices at every layer — a **key** and a **value**. These represent what that token "means" in context. Without caching, generating each new token would require recomputing these for every prior token — quadratic cost that makes long-context generation completely impractical.

So instead, the model stores them. Every time a token is processed, its key and value matrices are written to a cache. Future tokens look them up instead of recomputing them.

This is the KV cache. And it grows with every token you generate.

The math is simple:

```
memory = num_layers × num_kv_heads × head_dim × seq_len × 2 (K+V) × 2 bytes (fp16)
```

For **Llama-3.1-8B at 8k context: 4.2 GB**. At 32k context: **16.8 GB**. On a machine with 16 GB of total unified memory.

---

## Why Apple Silicon makes this worse, not better

On a discrete GPU setup — say, an NVIDIA card with 24 GB VRAM — the GPU has its own dedicated memory pool. The KV cache lives there, separate from your system RAM.

Apple Silicon doesn't work that way. M-series chips use **unified memory**: the CPU, GPU, and Neural Engine all share the same physical memory pool. Your model weights, your KV cache, your browser tabs, your IDE, your OS — all competing for the same 16 or 24 GB.

A 7B model at 4-bit quantisation takes roughly 4–5 GB. Add OS and background processes (~4 GB), Chrome with a few tabs (~2 GB), your IDE (~1 GB). You're at 11–12 GB before you've generated a single token. On a 16 GB machine, you have 4 GB left for the KV cache — which runs out around 6k context for a 7B model.

On a 24 GB machine it's better, but not solved. During the development of VeloxQuant-MLX, I ran a benchmark sweep across 8 models. **Qwen2.5-32B failed every quantization config** because the weights alone (~17.5 GB) plus OS and development tools left negative headroom for activations and cache buffers. The memory watchdog had to kill the process before it triggered a kernel panic:

```
[07:28:09] free=62MB   inactive=1094MB pressure=unknown
[07:28:19] free=63MB   inactive=891MB  pressure=unknown
[07:28:19] CRITICAL: only 954MB available — killing PID 21438
```

System recovered to 10.2 GB free immediately after the kill, with no GPU stall.

---

## What existing tools do about this: essentially nothing

This is the part that surprised me most when I started digging.

**llama.cpp** has done extraordinary work on attention kernel optimisation, weight quantisation, batching, and speculative decoding. But the KV cache is stored at fp16 by default. There is experimental support (`--cache-type-k q8_0`) but it's not the default, not Metal-optimised, and the quantisation methods are basic scalar quantisation with no adaptation to actual key distributions.

**Ollama** builds on top of llama.cpp and inherits all of this. The UX is excellent. The KV cache situation is unchanged.

**LM Studio** adds a polished GUI and nice model management. Same story on the cache.

**mlx_lm** — Apple's own MLX-based inference library — is the most natural fit for Apple Silicon and does excellent work. But its default KV cache is full fp16. No built-in compression.

All of these tools have optimised everything *around* the KV cache. None of them have solved the KV cache itself.

---

## The insight: compress the cache, not just the weights

Most quantisation work in the LLM ecosystem targets **weight quantisation** — compressing model parameters from fp16 down to 8-bit, 4-bit, or lower. This is what GGUF, GPTQ, AWQ, and most Ollama models use.

Weight quantisation is a one-time operation. You compress once, save, load the compressed version.

KV cache quantisation is different. It happens **at inference time**, on every token, for every layer. The keys and values are different every generation — you can't pre-compute anything.

The key insight is that while weight distributions are relatively uniform, **KV cache distributions are not**. Keys tend to follow Gaussian or Laplacian distributions after a rotation transform. This structure can be exploited with much more aggressive compression than weight quantisation — down to **1 bit per dimension** with surprisingly low quality loss.

The reason this works at such extreme bit rates: the attention computation only needs an *approximation* of the inner product `q · k`. Small errors in the reconstructed key vector translate to small errors in the attention score, which average out across many keys in the softmax.

---

## What I built: VeloxQuant-MLX

VeloxQuant-MLX is a KV cache compression library for Apple Silicon, built on top of MLX. It implements nine quantisation algorithms — from zero-calibration 1-bit methods to mixed-precision allocators — all backed by custom Metal GPU kernels compiled at runtime.

---

### The algorithms

#### TurboQuant RVQ — the recommended default

Zero calibration. Works on any model immediately. Uses Residual Vector Quantisation with analytical Gaussian and Laplacian codebooks precomputed from distribution theory.

Two residual passes at 1 bit each → **7.5× compression** with cosine similarity above 0.97.

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Write a 3000-word analysis of the transformer architecture.",
    max_tokens=3000,
)
```

#### VecInfer — 16× with Metal acceleration

Trades a 2-minute calibration step for **16× compression** via Product Vector Quantisation. Per-channel smooth scaling handles outlier dimensions that defeat standard VQ. The hot path runs through a Metal GPU kernel that is **13× faster** than equivalent MLX Python ops.

Standout result: **Qwen2.5-7B VecInfer-1bit exceeds fp16 throughput** at 16× compression, likely due to its strong GQA ratio reducing the number of KV heads.

#### RateQuant — best accuracy per bit

Runs a 90-second sensitivity probe across all transformer layers, learns which layers are most sensitive to quantisation noise, then uses reverse-waterfilling to allocate more bits to sensitive layers and fewer to insensitive ones.

At 2.0 average bits, RateQuant achieves **2.7× lower perplexity degradation** than uniform 2-bit quantisation at identical memory cost.

| Model | Sensitivity ratio | Allocation | Result vs fp16 |
|---|---|---|---|
| Falcon3-7B | 6.48× | 14 × b=2, 14 × b=1 | **100%** at 5.22× compression |
| Gemma3-4B | 14.39× | 3 × b=3, 11 × b=2, 20 × b=1 | **91%** at 5.22× compression |

#### SpectralQuant — best quality at long context

Keys in transformer models concentrate ~96% of their variance in just 3–4% of dimensions. SpectralQuant rotates keys into their eigenvector basis via SVD, then applies separate codebooks to the "signal" dimensions (high variance) and "noise" dimensions (low variance).

| Model | SpectralQuant | TurboQuant 3-bit | Quality gain |
|---|---|---|---|
| Qwen2.5-0.5B | 0.9072 cosim | 0.8329 cosim | **+7.4pp** |
| Gemma 4 4B | 0.8625 cosim | 0.7581 cosim | **+10.4pp** |

At 16k context the rotation trick matters enormously — quantisation errors accumulate over long sequences, and aligning the codec to the actual variance structure cuts that accumulation dramatically.

#### RaBitQ — maximum context length

1-bit key compression via randomised Hadamard transform + binary sign packing + IVF clustering. Pairs with 4-bit MSE scalar quantisation on values for **6× full KV compression**.

| Method | KV memory @ 1024 tok | Compression | Context @ 8 GB |
|---|---|---|---|
| fp16 baseline | 117.4 MB | 1× | ~17k tokens |
| RaBitQ keys + MSE-b4 values | **19.7 MB** | **6×** | **~103k tokens** |

6× more context in the same RAM budget.

#### CommVQ — RoPE-compatible exact VQ (ICML 2025)

Standard VQ breaks with RoPE positional encodings because `quantize(rotate(x)) ≠ rotate(quantize(x))`. CommVQ (Apple ML Research, ICML 2025) trains codebooks on pre-RoPE keys with a commutativity projection in the EM M-step. RoPE is applied exactly at decode time. **64× key compression** with exact positional encoding.

---

### The Metal kernels

All the compression math in the world doesn't help if the quantisation itself is slow. VeloxQuant-MLX compiles nine Metal GPU kernels at runtime using `mx.fast.metal_kernel`:

| Kernel | What it does | Speedup |
|---|---|---|
| `vecinfer_quantize_metal` | Fused nearest-centroid product VQ | **13×** |
| `rabitq_hamming_score` | XOR + popcount Hamming distance | **11×** |
| `turboquant_hadamard_quantize` | WHT rotation + scalar quant fused | **8.6×** |
| `turboquant_fused_rvq_decode_attend` | RVQ decode + attention in one dispatch | **6.9×** |
| `metal_fused_sdpa` | Dequant + scaled dot-product attention | avoids fp16 materialisation |

The fused SDPA kernel is the most impactful. Without fusion, dequantisation creates a full fp16 key matrix in memory before attention — which defeats much of the point of compression. The fused path keeps the cache compressed all the way until attention scores are computed.

The 30-line Metal kernel that powers VecInfer's 13× speedup:

```metal
// One thread per sub-vector. Argmin lives in registers — no diff tensor.
uint vec_idx = thread_position_in_grid.x;
float best_dist = INFINITY;
uint  best_idx  = 0;

for (uint c = 0; c < n_centroids; ++c) {
    float dist = 0.0f;
    for (uint i = 0; i < sub_dim; ++i) {
        float d = float(x[x_base + i]) - float(codebook[cb_base + i]);
        dist += d * d;
    }
    if (dist < best_dist) { best_dist = dist; best_idx = c; }
}
out[vec_idx] = best_idx;
```

The key insight: the `[N, n_centroids, sub_dim]` diff tensor is never materialised. The argmin accumulator lives entirely in thread-local registers — **98% peak memory reduction** at long context shapes.

---

## Real numbers across 10 models

End-to-end `mlx_lm.generate`, 200-token prompt, 120-token generation, Apple M-series:

| Model | fp16 tok/s | RVQ-1bit tok/s | VecInfer-1bit tok/s | VecInfer compression |
|---|---|---|---|---|
| Llama-3.2-3B | 47.6 | **46.2** | 40.2 | 16× |
| Llama-3.1-8B | 20.5 | **20.6** | 19.6 | 16× |
| Mistral-7B | 23.6 | **22.8** | 9.8 | 16× |
| Qwen2.5-7B | 21.0 | 20.7 | **21.5** ⬆ exceeds fp16 | 16× |
| Falcon3-7B | 17.3 | **21.7** | 17.0 | 16× |
| Gemma-3-4B | 26.0 | 24.2 | **22.6** | 16× |

**RVQ-1bit** is the safe default — within 5% of fp16 on most 7–8B models with zero calibration. **VecInfer-1bit** wins on compression (always 16×) and throughput on models with strong Grouped Query Attention ratios.

---

## What this means in practice

On a **16 GB MacBook M3**:
- Before: 7B model maxes out around 6k context before generation slows
- After with RVQ 1-bit: that same cache now fits in ~450 MB, leaving headroom for 50k+ token contexts

On a **24 GB MacBook M3 Pro**:
- Before: 13B models are borderline; long conversations risk OOM
- After: 13B at 32k context fits comfortably; the cache that would have consumed 8 GB uses 500 MB

---

## Quick decision guide

| Goal | Method |
|---|---|
| No calibration, get started now | `turboquant_rvq` b=1 — 7.5× compression |
| Maximum compression | `vecinfer` 1-bit — 16× (needs 2 min calibration) |
| Best quality at any bit rate | `spectral` b=3 — 5.33× with +7–10pp cosine sim gain |
| Best accuracy per average bit | `ratequant` — 2.7× better than uniform at same memory |
| Maximum context length | `rabitq` keys + MSE-b4 values — 6× full KV, 6× more context |
| RoPE-compatible exact VQ | `comm_vq` — 64× key compression, exact positions |

---

## Try it

```bash
pip install VeloxQuant-MLX
```

Requires macOS on Apple M1 or later · Python 3.11+ · MLX ≥ 0.18

📖 Documentation: https://veloxquant-mlx.netlify.app/docs/
📦 PyPI: https://pypi.org/project/VeloxQuant-MLX/
⭐ GitHub: https://github.com/rajveer43/turboquant_mac_implementation

---

## What's next

- **Vision-language model support** — visual tokens have fundamentally different KV distributions; heavy outliers require adapted algorithms
- **Per-head RateQuant granularity** — the paper allocates bits per layer-head group; current implementation is per-layer, leaving ~30% accuracy improvement on the table
- **Gradient-based sensitivity** for RateQuant — more accurate than activation-based, at the cost of a slightly longer calibration pass

---

The KV cache is the last major unoptimised bottleneck for local LLM inference on Apple Silicon. llama.cpp, Ollama, and LM Studio have built excellent tools — but none of them have solved this. VeloxQuant-MLX is my attempt.

If you're running models locally on a Mac and hitting memory walls — at any context length, on any model — I'd genuinely like to hear about it.

---

*MIT License · Built for Apple Silicon · Engineered for speed*

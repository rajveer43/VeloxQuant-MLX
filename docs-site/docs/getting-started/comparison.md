---
id: comparison
title: VeloxQuant-MLX vs. llama.cpp vs. plain mlx_lm
sidebar_label: vs. llama.cpp / mlx_lm
slug: /getting-started/comparison
---

# VeloxQuant-MLX vs. llama.cpp vs. plain mlx_lm

If you already run local LLMs on Apple Silicon, you're probably using `llama.cpp` (or Ollama/LM Studio, which wrap it) or `mlx_lm` directly. Both work today without VeloxQuant-MLX. This page is the honest answer to "why would I add another dependency?"

:::info[Short version]
`llama.cpp` quantizes the KV cache too — but with one fixed scheme applied uniformly to every model and layer, and no eviction. Plain `mlx_lm` doesn't quantize the KV cache at all; it stores it fp16, full size, always. VeloxQuant-MLX gives you 41 selectable compression methods (quantization *and* eviction *and* cross-layer merging), each independently tuned per layer, on top of the `mlx_lm` you're probably already using.
:::

## The options, side by side

| | Plain `mlx_lm` | `llama.cpp` / Ollama / LM Studio | VeloxQuant-MLX |
|---|---|---|---|
| What it is | Model loading + generation library | Standalone inference runtime (C/C++) | KV cache compression library on top of `mlx_lm` |
| KV cache precision | fp16 (no compression) | Fixed: `q8_0` or `q4_0` via `--cache-type-k` / `--cache-type-v` | 1–8 bit, chosen per method |
| Compression scheme | None | One uniform per-tensor quant type, same scheme for every layer | 41 methods — VQ, RVQ, non-uniform, low-rank, entropy coding, mixed-precision |
| Token eviction (drop stale tokens) | No | No | Yes — SnapKV, StreamingLLM, H2O, TOVA, and 8 more |
| Cross-layer compression | No | No | Yes — XQuant (code reuse), MiniCache (SLERP merge), xKV (shared subspace) |
| Per-layer / per-head tuning | N/A | No — one setting for the whole model | Yes — method and bit-width are configurable per layer |
| Calibration step | N/A | None | Optional — most methods need none; a few (VecInfer, SpectralQuant) train a codebook once |
| Runtime | Python (MLX, Metal) | C/C++ (Metal backend on macOS) | Python (MLX, Metal) — same runtime as plain `mlx_lm` |
| Model format | MLX (safetensors) | GGUF | MLX (safetensors) — same models plain `mlx_lm` already loads |
| Integration | Native | Native | 3 extra lines on top of `mlx_lm` |

## Where llama.cpp actually wins

Be clear-eyed about this: `llama.cpp` is not a strawman.

- **It's the more mature, more portable project.** It runs on everything — Apple Silicon, x86, Linux, Windows, mobile — not just Apple Silicon with Metal.
- **Its KV cache quantization (`q4_0`/`q8_0`) is production-tested at massive scale.** Ollama and LM Studio both build on it, and millions of users run it daily without issue.
- **GGUF is a mature, widely supported model format** with a large pre-quantized model catalog.
- **Zero configuration.** `--cache-type-k q4_0` and you're done — there's no method to choose because there's only one.

If you're not on Apple Silicon, or you want the most battle-tested path with the least decision-making, `llama.cpp`-based tooling is the right default. VeloxQuant-MLX doesn't try to replace that.

## Where VeloxQuant-MLX wins

The gap is specifically in **how much control you have over the memory/quality tradeoff**, not raw compatibility:

- **llama.cpp's KV quantization is one fixed scheme for the whole model.** `q4_0` is a uniform 4-bit block quantizer — the same scheme whether the layer is shallow (broad attention) or deep (narrow attention), whether the head is RoPE-sensitive or not. VeloxQuant-MLX's 41 methods exist because no single scheme is optimal everywhere: CommVQ preserves RoPE exactly, PolarQuant fits geometric key clusters, PyramidKV gives early layers a bigger budget than deep ones — none of that is expressible as a `--cache-type-k` flag.
- **llama.cpp has no token eviction.** It quantizes every token's KV pair, forever. VeloxQuant-MLX's eviction methods (StreamingLLM, SnapKV, H2O, and 9 more) drop stale tokens entirely for constant-memory long-context generation — a different lever llama.cpp doesn't expose at all.
- **Compression ceiling is higher.** `q4_0` gets you to 4 bits per element. VeloxQuant-MLX's 1-bit methods (TurboQuant RVQ, RaBitQ, QJL) go further — up to **16× key cache compression** (VecInfer, head_dim=128) — because they're built specifically for the sub-4-bit regime instead of adapting a general block quantizer down to it.
- **You stay in the `mlx_lm` ecosystem.** If you're already loading MLX-format models with `mlx_lm`, VeloxQuant-MLX is three lines, not a runtime switch to GGUF and a different toolchain.

## Real numbers (VeloxQuant-MLX, measured)

These are from this library's own benchmark suite, run against real production models (Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon) — not projections:

| Metric | Value | Notes |
|---|---|---|
| Max key cache compression | **16×** | VecInfer-1bit, head_dim=128 |
| Metal kernel speedup | **13×** | `quantize_vq` at S=2048 (range 6.9–14.7× across S=128–8192) |
| Peak memory reduction | **98%** | 729 MB → 12 MB, Falcon3-7B shape at the OOM-trigger context length |
| RVQ-1bit compression | **7.5×** | near-zero throughput cost |
| FP16 throughput retained | **100%** | Qwen2.5-7B at 16× compression |
| KIVI-2bit full-KV compression | **~4×** | incl. fp16 residual window; 100–106% of fp16 throughput |
| CommVQ key compression | **64×** | RoPE-commutative VQ, D=128, n_cb=4 |

We don't have head-to-head throughput numbers against `llama.cpp`'s `q4_0`/`q8_0` cache in this repo — different runtime, different hardware paths, and an apples-to-apples run hasn't been published here yet. The honest comparison today is architectural (fixed scheme vs. 41 selectable ones, no eviction vs. 11 eviction methods), not a benchmark race.

## Which one should you use?

```
Are you on Apple Silicon and already using mlx_lm?
├── No  → llama.cpp / Ollama / LM Studio (broader hardware support, zero setup)
└── Yes →
    Is fp16 KV memory (or a flat q4_0/q8_0 cache) already good enough
    for your context length?
    ├── Yes → Stick with what you have — don't add a dependency you don't need
    └── No, you need more compression, eviction, or per-layer tuning →
        VeloxQuant-MLX
```

In short: reach for VeloxQuant-MLX when your problem is *memory* — you've hit a wall a single fixed 4-bit cache can't solve, or you need eviction and cross-layer tricks `llama.cpp` doesn't have. If you haven't hit that wall, you don't need this library.

## Next steps

- [5-minute quickstart](./quickstart)
- [Algorithm overview](../algorithms/overview) — all 41 methods
- [mlx_lm integration guide](../guides/mlx-lm-integration)

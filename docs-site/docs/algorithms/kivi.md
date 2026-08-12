---
id: kivi
title: KIVI
sidebar_label: KIVI
slug: /algorithms/kivi
---

# KIVI

KIVI is VeloxQuant-MLX's re-implementation of the field's most widely-cited
KV-cache quantization baseline: ["KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) (Liu, Yuan et
al., **ICML 2024**). It is included so every other algorithm in this library
can be measured against a recognized reference point.

:::info[Why a baseline?]
KIVI is not the highest-compression method here (VecInfer-2bit reaches 8×
key compression vs KIVI-2bit's ~5.8×). Its value is being **calibration-free,
deterministic, and the number reviewers expect to see compared against.**
:::

## How it works

KIVI's insight is **asymmetry** between keys and values:

1. **Keys are quantized per channel** — the quantization group runs along the
   token axis, with one `(scale, zero)` pair per channel-group. Key
   distributions have a few high-variance channels, so per-channel scales keep
   them accurate.
2. **Values are quantized per token** — the group runs along the channel axis,
   one `(scale, zero)` per token-group.
3. **fp16 residual window** — the most recent `residual_length` tokens are kept
   at full precision. Newly generated tokens dominate attention and are cheap
   to keep exact; they are quantized only once they age out of the window.
4. **Group-aligned flushes** — aged-out tokens are quantized in blocks that are
   whole multiples of `kivi_group_size`, per the paper's Algorithm 1. This is
   what makes per-channel key groups meaningful: flushing one token at a time
   would put a single element in every group, making `min == max` and the
   round-trip lossless-but-uncompressed. It also makes the result independent
   of chunking — one-shot prefill and token-by-token decode produce
   bit-identical caches.

Each group uses asymmetric min/max quantization:

```
zero  = min(group)
scale = (max(group) - min(group)) / (2**b - 1)
q     = round((group - zero) / scale)      # uint, [0, 2**b - 1]
recon = q * scale + zero
```

KIVI is **fully deterministic** — no codebook training, no rotation, no RNG —
so it adds no run-to-run variance.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="kivi",
    bit_width_inlier=2,  # KIVI default
    kivi_group_size=32,  # min/max group size
    residual_length=32,  # recent tokens kept in fp16
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=120)
```

## Measured results

Apple M4, `max_tokens≈120`, `residual_length=32`, long-context prompt.
Source: `figures/kivi/<model>/results.json`.

| Model | KIVI-2bit key comp. | full-KV comp. | throughput vs fp16 |
|---|---|---|---|
| Llama-3.2-3B-4bit | 5.46× | 4.84× | 101% |
| Qwen2.5-7B-4bit | _pending re-run_ | _pending re-run_ | _pending re-run_ |
| Mistral-7B-4bit | _pending re-run_ | _pending re-run_ | _pending re-run_ |

Full-KV compression includes the fp16 residual window, so it is not inflated.

:::note[Re-measured after #162]
The Llama row was re-measured on Apple M4 after the residual-buffer fix
([#162](https://github.com/rajveer43/VeloxQuant-MLX/issues/162)). Before that
fix, decode-time keys were quantized one token at a time, which collapsed each
per-channel group to a single element and left those keys bit-exact fp16 while
still billing them as 2-bit — so the earlier numbers in this table were not
reproducible from the code. The remaining rows are pending a re-run.
:::

## Honest scope

:::warning[Memory, not raw speed; storage, not peak]
- KIVI's published *speedup* comes from a CUDA kernel that **does not port to
  Metal**. On Apple Silicon the win is memory; throughput is at-or-near fp16
  because the min/max arithmetic is cheap on a memory-bound decode path.
- Compression only manifests **once context exceeds the residual window** — at
  short prompts the whole prefill stays fp16 and the realized ratio is 1.0×
  (correct behavior, not a bug).
- **Peak runtime memory is not reduced** (sometimes marginally higher): keys
  are dequantized to fp16 before SDPA, so the compression is in cache-storage
  accounting, not the peak fp16 working set.
- At 2 bits, raw-key reconstruction cosine on synthetic unit-norm Gaussian keys
  is ~0.93 — KIVI 2-bit is genuinely lossy, which is exactly why the fp16
  residual window exists.
:::

See `figures/kivi/fig4_vs_existing.png` for the KIVI-vs-VecInfer comparison.

See also: [NSNQuant](../algorithms/nsnquant) — the other residual-window wrapper; it differs because it adapts the data to a fixed universal codebook (NSN + Hadamard Gaussianization) rather than fitting scalar min/max scales to the data.

See also: [SKVQ](../algorithms/skvq) — the reorder+clip evolution of this same group quantizer (COLM 2024): it quantizes *both* K and V with per-token groups (the KIVI value scheme), made viable for keys by sorting channels into like-range groups, and shrinks each group's min/max window by a per-group searched clip factor. Its sliding window also quantizes decode tokens once they age out, where this wrapper's residual logic only touches the incoming block.

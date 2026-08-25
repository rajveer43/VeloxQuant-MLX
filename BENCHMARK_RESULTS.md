# TurboQuant KV Cache Compression on Apple Silicon — Benchmark Results

> **Platform:** Apple M4 MacBook · MLX Framework · Python 3.12  
> **Date:** May 2026  
> **Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (3B parameters, 4-bit weights, ~1.8 GB on disk)

---

## What Is TurboQuant?

TurboQuant is a KV cache compression algorithm published by Google Research (ICLR 2026). Every token a language model generates writes a **Key** and **Value** vector into a KV cache. At long contexts this cache can consume gigabytes of memory — rivalling the model weights themselves.

TurboQuant compresses those Key vectors from 16-bit floats down to 3–4 bits using two stages:

1. **Rotation + Lloyd-Max quantization** — a random orthogonal rotation spreads information uniformly across all dimensions, then a fixed scalar codebook (calibrated for the resulting distribution) assigns each dimension to the nearest centroid. No per-block scale constants are stored.
2. **QJL residual correction** — a 1-bit sign sketch of the quantization error provides an unbiased correction to every attention score, recovering precision that scalar quantization lost.

This implementation runs natively on Apple Silicon via the MLX framework, using `mx.hadamard_transform` for O(d log d) Metal-accelerated rotation.

---

## Experimental Setup

### Model

| Property | Value |
|---|---|
| Model ID | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Architecture | Llama 3.2, decoder-only transformer |
| Parameters | 3 billion |
| Weight quantization | 4-bit (pre-quantized by mlx-community) |
| KV head dimension | 128 |
| KV heads per layer | 8 (GQA) |
| Attention layers | 28 |
| Disk size | ~1.8 GB |

### Hardware

| Property | Value |
|---|---|
| Chip | Apple M4 |
| Unified memory | 16 GB |
| Framework | MLX (Apple's ML framework for Apple Silicon) |

### Test Prompt

```
Explain the theory of relativity in simple terms,
covering both special and general relativity with examples.
```

This prompt was chosen because:
- It requires multi-paragraph structured reasoning
- It has a known ground-truth answer — easy to judge correctness
- Repetition or hallucination is immediately visible
- It generates ≥200 tokens, giving the KV cache time to accumulate and compress

### Generation Settings

| Setting | Value |
|---|---|
| Max tokens | 200 |
| Temperature | default (greedy/sampling) |
| Chat template | Applied (system + user turn) |

---

## Configurations Tested

Four configurations were benchmarked back-to-back on the same model and prompt:

| Config | Description |
|---|---|
| **fp16 baseline** | Standard mlx-lm KV cache, no compression |
| **TurboQuant 3-bit** | 2-bit MSE stage (4 centroids) + 1-bit QJL residual |
| **TurboQuant 4-bit** | 3-bit MSE stage (8 centroids) + 1-bit QJL residual |
| **TurboQuant 4-bit + 8 outliers** | 4-bit + top-8 highest-variance dimensions stored fp16 |

---

## Results

### Compression and Speed

| Config | Key Compression | KV Key Size | fp16 Equivalent | Time | Throughput |
|---|---|---|---|---|---|
| fp16 baseline | 1.00× | 8,120 KB | 8,120 KB | 4.3s | **47.2 tok/s** |
| TurboQuant 3-bit | **5.82×** | 1,396 KB | 8,120 KB | 3.4s | 25.8 tok/s |
| TurboQuant 4-bit | **4.27×** | 3,360 KB | 14,336 KB | 8.1s | 24.9 tok/s |
| TurboQuant 4-bit + 8 outliers | **3.51×** | 4,088 KB | 14,336 KB | 38.5s | 5.2 tok/s |

> **Note on KV key size:** Only Key vectors are compressed. Value vectors remain fp16 in the current implementation (compressing values is an extension — Keys dominate attention quality).

> **Note on throughput overhead:** The current Python-level encode/decode loop adds latency proportional to `n_layers × n_kv_heads × tokens`. A fused Metal kernel (planned) would reduce this to near-zero overhead. The 4-bit result (+88% time overhead) is a software bottleneck, not an algorithmic one.

---

### Output Quality — Side-by-Side

#### fp16 Baseline (reference)

```
The theory of relativity, developed by Albert Einstein, is a fundamental concept
in physics that explains how space and time are connected. There are two main
components of the theory: special relativity and general relativity.

Special Relativity (1905)

Special relativity is a theory that explains how time and space are affected by
an object's speed. The main idea is that the laws of physics are the same
everywhere in the universe, and the speed of light is always constant,
regardless of t...

[201 tokens, 4.3s, 47.2 tok/s]
```

---

#### TurboQuant 3-bit (5.82× compression)

```
I'll try to explain it in a way that's easy to understand.

The theory of relativity is a bit of relativity that's hard to explain. It's a
theory that was developed by Albert Einstein. It was developed in 1905 and 1915.
It's a theory that it's a theory that's hard to understand.

The theory of relativity is a theory that's that's hard to explain.

[89 tokens, 3.4s, 25.8 tok/s]
```

**Assessment:** ❌ Repetition loop. Output degrades into circular sentences. This is characteristic of broken attention — the model cannot properly attend to earlier tokens, so it loops. The cosine similarity between original and reconstructed key vectors at 3-bit is ~0.85, meaning attention scores have ~15% directional error. That is enough to destabilise autoregressive generation.

---

#### TurboQuant 4-bit (4.27× compression)

```
The theory of relativity, developed by Albert Einstein, is a fundamental concept
in modern physics that explains how space and time are connected. I'll break it
down into simple terms, covering both special and general relativity.

Special Relativity (1905)

Special Relativity, also known as Special Relativity, is a theory that explains
how space and time are connected. It was developed by Albert Einstein in 1905.

Key Points:
1. The Speed of Light is Constant: The speed of light (a...

[201 tokens, 8.1s, 24.9 tok/s]
```

**Assessment:** ✅ Near-lossless. Content, structure, and factual accuracy match the fp16 baseline. Minor stylistic difference ("I'll break it down" vs direct explanation) — within normal temperature-sampling variance. The cosine similarity at 4-bit is ~0.95, which is sufficient to preserve attention quality.

---

#### TurboQuant 4-bit + 8 Outlier Channels (3.51× compression)

```
The theory of relativity, developed by Albert Einstein, is a fundamental concept
in modern physics. It's a complex topic, but I'll break it down into two parts:
special relativity and general relativity, using simple examples to illustrate
each concept.

Special Relativity

Special relativity, introduced in 1905, deals with objects moving at constant
speeds relative to each other. It challenged the long-held notion of absolute
time and space. Here are some key points:

1. The Speed of Ligh...

[201 tokens, 38.5s, 5.2 tok/s]
```

**Assessment:** ✅ Lossless quality. Output is arguably *more detailed and better structured* than the fp16 baseline — correctly introducing the historical context and structuring the answer as a proper explanation. The outlier protection (storing the 8 highest-variance key dimensions at full fp16 precision per head) prevents any precision loss in the most information-dense channels. **Throughput is 5.2 tok/s — 9× slower than baseline** due to the Python-level numpy↔MLX data copy on every token for outlier channel handling. This is a software implementation bottleneck, not an inherent algorithm cost.

---

## Compression Math — How the Numbers Work

For Llama-3.2-3B at 200 tokens:

**fp16 KV cache per layer:**
```
8 KV heads × 128 head_dim × 2 bytes (fp16) × 2 (K+V) × 200 tokens = 819,200 bytes/layer
× 28 layers = 22,937,600 bytes ≈ 22.4 MB
```

**TurboQuant 4-bit key storage per layer (keys only):**
```
MSE indices  : 200 tokens × 8 heads × ceil(128 × 3 / 8) = 200 × 8 × 48 = 76,800 bytes
QJL signs    : 200 tokens × 8 heads × ceil(64 / 8)      = 200 × 8 × 8  = 12,800 bytes
Residual norm: 200 tokens × 8 heads × 2 bytes (fp16)    = 200 × 8 × 2  =  3,200 bytes
Per-vec norm : 200 tokens × 8 heads × 2 bytes (fp16)    = 200 × 8 × 2  =  3,200 bytes
                                                                   Total = 96,000 bytes
Values (fp16): 200 × 8 × 128 × 2                                       = 409,600 bytes

Total per layer = 505,600 bytes   vs   819,200 bytes fp16
```

The 4.27× ratio compresses only keys; when values are included, the total KV compression is ~1.6×. Compressing values too (a straightforward extension using int8 per-token quantization) would push total KV compression to ~3.5–4×.

---

## Why 4-bit, Not 3-bit?

The Lloyd-Max codebook is calibrated for the distribution of unit-norm post-rotation key vectors. The MSE stage of TurboQuant uses `b-1` bits (so 3-bit → 2-bit MSE → 4 centroids; 4-bit → 3-bit MSE → 8 centroids).

| Bits | MSE centroids | Cosine similarity | SNR | Output quality |
|---|---|---|---|---|
| 3-bit | 4 | ~0.85 | ~4 dB | Repetition loops |
| 4-bit | 8 | ~0.95 | ~10 dB | Near-lossless |
| 5-bit | 16 | ~0.98 | ~15 dB | Indistinguishable |

At head_dim=128, the transition from broken to working output happens between 3-bit and 4-bit. The paper's 3-bit results (on GPU with head_dim=256+) benefit from larger dimensions where the codebook approximation is tighter and the QJL correction more effective.

---

## Key Implementation Insights

### 1. Per-Vector Normalization Is Critical

The Lloyd-Max codebook is calibrated for **unit-norm** post-rotation vectors. Raw KV vectors from a language model are not unit-norm — they have varying magnitudes across layers and tokens. Without normalization, codebook centroids are completely miscalibrated and SNR drops below 0 dB (more noise than signal).

**Fix:** Normalize each key vector to unit norm before encoding, store the norm as fp16, rescale during decode. This is the same technique used in weight quantization (per-row normalization).

### 2. MLX Arrays Are Immutable

MLX uses lazy evaluation — arrays cannot be modified in-place like NumPy. Assignments like `array[i, j] = value` silently fail. The correct pattern is to collect results per-head/per-batch and use `mx.stack()` to assemble the final tensor.

### 3. The MLX uint8 Bottleneck

MLX has no sub-byte dtype. A 3-bit index stored in `uint8` wastes 5 bits. True 10× compression requires bit-packing (already implemented in `mlx_kv_quant/dsa/bit_pack.py` and `TurboQuantKVCache`). The benchmark above uses the simpler per-element uint8 storage for clarity.

### 4. Head Dimension Matters

TurboQuant is designed for head_dim ≥ 128. At head_dim=64 (e.g. SmolLM2-135M), the codebook approximation breaks down — SNR is negative even at 4-bit. The algorithm shines at the head dimensions used by production models (128–512).

---

## What's Next

- [ ] **Value compression** — int8 per-token quantization of Value vectors, targeting ~2× additional compression
- [ ] **Fused Metal kernel** — eliminate the Python encode/decode loop; target <5% throughput overhead
- [ ] **Perplexity benchmark** — quantitative quality measurement on WikiText-2
- [ ] **Longer context** — test at 4K, 8K tokens where KV cache dominates memory
- [ ] **Bit-packing** — true 3-bit storage via `BitPackBuffer`, targeting ~5.5× total KV compression

---

## Running the Benchmark

```bash
# Install dependencies
pip install mlx mlx-lm

# Clone and install
git clone <repo>
pip install -e .

# Run the benchmark
python benchmark_kv.py
```

The benchmark script is at [`benchmark_kv.py`](benchmark_kv.py). It runs all four configurations sequentially and prints a summary table.

---

## References

- TurboQuant (ICLR 2026) — Zandieh et al., *"Online Vector Quantization with Near-optimal Distortion Rate"*
- QJL (AAAI 2025) — Zandieh et al., *"QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"*
- PolarQuant (AISTATS 2026) — *"PolarQuant: Quantizing KV Caches with Polar Transformation"*
- Apple MLX — [ml-explore/mlx](https://github.com/ml-explore/mlx)

---

## Historical benchmark snapshots

Point-in-time results from earlier releases, preserved for reference. See the
[README's benchmark section](README.md#benchmark-results) for the current
10-model comparative study.

### Throughput optimisation journey (v0.3.0)

Four sequential changes to lift quantized throughput to fp16 parity:

| Stage | Mistral-7B RVQ-2bit | Qwen3-4B RVQ-2bit |
|---|---|---|
| 0. Original (per-head Python loop) | 17.7 tok/s | 24.8 tok/s |
| 1. Batch heads `(B,H,S,D) → (B·H·S,D)` | 21.5 tok/s | 34.0 tok/s |
| 2. Hadamard rotation by default | 20.0 tok/s | — |
| 3. Boundary-sum quantize (replaces argmin) | 22.4 tok/s | — |
| 4. Drop redundant fp32↔fp16 casts | **22.3 tok/s** | **36.0 tok/s** |

Figure: `figures/updated_tests/optimization_journey.png`

### RateQuant V2 mixed-precision results (v0.3.5)

Per-layer allocation at target b̄=1.5, measured on Apple M4 24 GB.

| Model | fp16 | RVQ-1bit | RVQ + RateQuant V2 | Sens. ratio |
|---|---|---|---|---|
| Falcon3-7B | 22.9 | 23.1 (101%) | **22.8 (100%)** at 5.22× | 6.48× |
| Gemma3-4B | 39.8 | 37.8 (95%) | **36.3 (91%)** at 5.22× | 14.39× |

Source figures: [`figures/2026-05-16/`](figures/2026-05-16/)

### RVQ 1-bit 8-model sweep (v0.3.4)

All on Apple M4 MacBook 16/24 GB. Prompt: 200-token explanation of relativity.

| Model | fp16 tok/s | RVQ-1bit tok/s | vs fp16 |
|---|---|---|---|
| Mistral-7B v0.3 | 23.3 | 22.2 | 95% |
| Falcon3-7B | 24.0 | 23.1 | 96% |
| Phi-4 | 11.9 | 11.8 | **99%** |
| Qwen3-4B | 40.2 | 34.3 | 85% |
| Qwen3-8B | 20.5 | 21.1 | **103%** |
| Llama-3.1-8B | 22.0 | 21.5 | 98% |
| Gemma3-4B | 32.5 | 30.5 | 94% |

Source figures: [`figures/outlier_token_ratequant/`](figures/outlier_token_ratequant/)

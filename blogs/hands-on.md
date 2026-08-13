# Run Faster LLMs on Your Mac: A Hands-On Guide to VeloxQuant-MLX

*How to cut your KV cache memory by 8x and hit fp16 throughput — step by step, with real models.*

---

If you have an Apple Silicon Mac and you run local LLMs, you have already hit the wall: your MacBook Pro generates 20 tokens per second on Mistral 7B, memory pressure turns on after a few thousand tokens, and anything larger than 7B crawls. The bottleneck is not compute — it is memory bandwidth. Every token requires loading the entire KV cache for every layer. Make the cache smaller, and the model gets faster.

VeloxQuant-MLX is a drop-in KV cache quantization library for MLX. It compresses the KV cache to 2, 3, or 4 bits per value while keeping text quality close to full-precision. In this guide you will install it, run your first compressed model, pick the right algorithm for your use case, and understand what the benchmark numbers mean. No machine learning background required.

---

## What You Need

- Apple Silicon Mac (M1 or later)
- Python 3.11 or 3.12
- At least 16 GB unified memory (8 GB works for 4B models)
- `mlx-lm` installed (`pip install mlx-lm`)

---

## Step 1: Install VeloxQuant-MLX

```bash
pip install VeloxQuant-MLX
```

That installs the `mlx_kv_quant` package plus the `veloxquant` CLI. Verify it worked:

```bash
veloxquant --help
```

You should see the precompute and benchmark subcommands.

---

## Step 2: Your First Compressed Inference

The fastest path uses the `KVCacheBuilder`. Here is a minimal script that runs Mistral 7B with 4-bit KV cache compression:

```python
import mlx.core as mx
import mlx_lm
from mlx_kv_quant import KVCacheBuilder, KVCacheConfig

# Load the model normally
model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Build a quantized KV cache
config = KVCacheConfig(bits=4, algorithm="turboquant_prod")
cache = KVCacheBuilder(config).for_model(model).build()

# Generate with the compressed cache
prompt = "Explain the difference between RAM and unified memory."
tokens = mlx_lm.generate(model, tokenizer, prompt=prompt, kv_cache=cache, max_tokens=200)
print(tokens)
```

That is the entire integration. The cache is transparent to `mlx_lm.generate()` — you do not change anything else.

---

## Step 3: Understanding the Algorithm Choices

VeloxQuant-MLX ships five quantization algorithms. They trade reconstruction quality against memory and speed:

| Algorithm name | Bits/dim | Best for | Notes |
|---|---|---|---|
| `turboquant_mse` | b | Fast inference, b >= 3 | MSE-optimal Lloyd-Max codebook |
| `turboquant_prod` | b + 1 | Inner-product tasks, RAG | Adds 1-bit QJL residual correction |
| `turboquant_rvq` | 2b | 2-bit where quality matters | Two-pass residual; cosine 0.98 at b=2 |
| `polarquant` | b | Long contexts | Spherical Lloyd-Max, no norm storage |
| `qjl` | 1 | Ultra-low memory | Sign sketch only; rough approximation |

**Rule of thumb:**

- For most uses at 3–4 bit: use `turboquant_prod`. It is the default and the most tested.
- At 2 bits: always use `turboquant_rvq`. The single-pass algorithms fall apart at 2-bit; RVQ does not.
- For long contexts where cosine similarity matters more than reconstruction: `polarquant`.
- To stress-test memory limits: `qjl`.

---

## Step 4: Using RVQ 2-Bit (the Interesting One)

The headline feature in v0.3.0 is `TurboQuantRVQ` — a two-pass residual vector quantizer that makes 2-bit KV cache actually usable.

The problem with naive 2-bit quantization: you only have 4 levels to represent a continuous value. The reconstruction error is large enough that attention scores get corrupted, and long reasoning chains (like Qwen3's `<think>` mode) collapse into repetition after a few dozen tokens.

RVQ fixes this by running two codebooks. The first pass quantizes the rotated vector. The second pass quantizes the residual — the error left over from the first pass — using a Laplacian-fit codebook (which matches the residual distribution better than a Gaussian one). The result: cosine similarity goes from 0.69 to 0.98 at 2 bits.

Here is how to use it:

```python
from mlx_kv_quant import KVCacheBuilder, KVCacheConfig

config = KVCacheConfig(bits=2, algorithm="turboquant_rvq")
cache = KVCacheBuilder(config).for_model(model).build()
```

That's it. The `turboquant_rvq` string routes to the registered `TurboQuantRVQ` class via the quantizer registry.

**When should you use RVQ 2-bit?**

Use it when memory is the binding constraint and you need coherent long-form output. It delivers:

- Mistral 7B at 2-bit: 22.3 tok/s — matches fp16 throughput (22.1 tok/s) on an M4 MacBook
- Qwen3 4B at 2-bit: 36.0 tok/s (92% of fp16) with full thinking-mode output

Use standard 3-bit or 4-bit when you have memory headroom and want zero quality trade-off.

---

## Step 5: Running Benchmarks on Your Mac

VeloxQuant-MLX ships benchmark scripts for several models. These measure tok/s and output completeness across five configurations: fp16, 2-bit RVQ, 3-bit, 4-bit, and 2-bit single-pass (for comparison).

**Qwen3 4B** (good for 16GB Macs):

```bash
python3 benchmark_qwen3_4b_v2.py
```

**Mistral 7B** (needs 16GB+, comfortable at 32GB):

```bash
python3 benchmark_mistral7b_v2.py
```

Each script runs all five configs, prints a results table, and saves six figures to `figures/updated_tests/<model>/`. The figures include:

- Throughput comparison bar chart
- Token completeness (how many tokens were coherent out of 200)
- Memory usage per config
- Cosine similarity vs fp16 reference

Here is what our numbers look like on an M4 MacBook (16GB):

**Mistral 7B throughput:**

| Config | tok/s | Memory |
|---|---|---|
| fp16 | 22.1 | 14.2 GB |
| RVQ 2-bit | 22.3 | 3.8 GB |
| 3-bit | 21.8 | 5.4 GB |
| 4-bit | 21.6 | 7.1 GB |

Mistral 7B is memory-bandwidth bound — every config saturates around 22 tok/s because that is the bandwidth ceiling. But RVQ 2-bit uses 10x less cache memory than fp16.

**Qwen3 4B thinking-mode throughput:**

| Config | tok/s | Tokens (out of 200) |
|---|---|---|
| fp16 | 39.2 | 200 / 200 |
| RVQ 2-bit | 36.0 | 199 / 200 |
| 3-bit | 37.1 | 200 / 200 |
| 4-bit TQ | 24.8 | 50 / 200 |

The 4-bit single-pass result (50/200) is the failure mode for standard quantization on a thinking model. RVQ 2-bit produces full coherent output and runs faster.

---

## Step 6: Integrating with mlx-lm Properly

For real usage you want the cache to persist across generation steps. Here is the pattern for a chat loop:

```python
import mlx.core as mx
import mlx_lm
from mlx_kv_quant import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Qwen3-4B")

config = KVCacheConfig(bits=2, algorithm="turboquant_rvq")
cache = KVCacheBuilder(config).for_model(model).build()

messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        kv_cache=cache,
        max_tokens=512,
        verbose=False,
    )

    messages.append({"role": "assistant", "content": response})
    print(f"Assistant: {response}\n")
```

The `kv_cache` argument is passed directly to `mlx_lm.generate()`. No other changes. The cache grows with each turn and is quantized transparently.

---

## Step 7: Precomputing Codebooks for Repeated Use

If you run the same model repeatedly, you can precompute the Lloyd-Max codebooks offline and load them at inference time. This eliminates the one-time calibration cost at startup:

```bash
# Precompute and save codebooks for Mistral 7B, 4-bit
veloxquant precompute \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --bits 4 \
  --algorithm turboquant_prod \
  --output codebooks/mistral7b_4bit/

# Later, load at inference time
config = KVCacheConfig(
    bits=4,
    algorithm="turboquant_prod",
    artifact_dir="codebooks/mistral7b_4bit/",
)
cache = KVCacheBuilder(config).for_model(model).build()
```

Codebook size is tiny (kilobytes). You can commit them to your project directory and avoid the calibration step entirely on every run.

---

## Step 8: Picking the Right Bit Width

Here is the practical decision tree:

```
Do you have >= 32GB memory?
  Yes → fp16 is fine; use VeloxQuant only for very long contexts
  No  → continue

Is your model >= 7B parameters?
  Yes → 3-bit TurboQuant (turboquant_prod)
  No  → 2-bit RVQ (turboquant_rvq)

Are you doing RAG or similarity search over the KV cache?
  Yes → turboquant_prod (inner-product correction matters)
  No  → turboquant_mse (simpler, same quality)

Do you need to run 13B+ on a 16GB Mac?
  Yes → 2-bit RVQ — the only option that keeps quality
```

For generation tasks (chat, summarization, code completion), MSE-optimal reconstruction is what you want. For retrieval tasks that compute attention over a cached context, inner-product preservation matters more — use `turboquant_prod`.

---

## Step 9: What the Optimization Journey Looked Like

Getting from 17.7 tok/s to 22.3 tok/s on Mistral 7B RVQ 2-bit required four changes, each independently measurable:

**1. Batch all heads into one MLX call (+22%)**

The original code ran a Python `for h in range(H)` loop over attention heads, giving each head its own quantizer instance. For Mistral 7B (8 heads, 32 layers) that meant 256 small kernel dispatches per token. The fix: reshape `(B, H, S, D)` to `(B*H*S, D)` and call one shared quantizer once. MLX already handles batched input — we just stopped fragmenting it.

**2. Switch to Hadamard rotation**

The default preconditioner used QR decomposition: a full `(d, d)` matrix multiply (16,384 ops at d=128). `mx.hadamard_transform` is a Metal-native fused kernel that does the same job in O(d log d) — 896 ops at d=128, ~18x less arithmetic. Quality is mathematically identical because Hadamard with random ±1 diagonal is also a Haar-equivalent rotation.

**3. Replace broadcast-argmin with boundary-sum**

Codebook lookup materialized a `(batch, 128, k)` distance tensor for argmin — three kernels: broadcast-subtract, abs, argmin. Lloyd-Max boundaries are just midpoints between sorted centroids. The nearest centroid index is the number of boundaries the value exceeds: one comparison and one sum, two kernels total. Index match vs the old path: 100.0000%.

**4. Drop redundant fp32 casts**

The update path was casting fp16 → fp32 → fp16 → fp32 → fp16 unnecessarily because the rotation already handles internal precision internally. Removing the round-trips saved ~1.05x per call and is invisible to output quality.

The full write-up with stage-by-stage numbers is in [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md).

---

## Step 10: Understanding Memory Numbers

KV cache memory scales with: `2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element`

For Mistral 7B (32 layers, 8 KV heads, head_dim 128) at sequence length 2048:

- fp16: 2 × 32 × 8 × 128 × 2048 × 2 bytes = **536 MB**
- 4-bit: 268 MB (2x reduction)
- 2-bit RVQ: 134 MB (4x reduction, plus codebook overhead ~8 KB)

At longer contexts (32K tokens), these numbers scale linearly:

- fp16: 8.4 GB for context alone
- 2-bit RVQ: 2.1 GB

On a 16GB Mac, the difference between fp16 and 2-bit RVQ at 32K context is the difference between OOM and running.

---

## Common Questions

**Does this work with quantized models (4-bit weights)?**

Yes. The KV cache quantization is independent of weight quantization. You can run a 4-bit weight model with a 2-bit KV cache. The two compressions are additive: `mlx-community/Mistral-7B-Instruct-v0.3-4bit` loads as a 4-bit weight model; the KV cache is then further compressed by VeloxQuant.

**Will I notice quality degradation?**

At 4-bit and 3-bit: barely measurable on generation tasks. At 2-bit with RVQ: cosine similarity 0.98 (vs 1.0 for fp16). On our Qwen3 4B thinking-mode test, RVQ 2-bit produced 199/200 coherent tokens vs fp16's 200/200. One token difference across a 200-token reasoning chain.

At 2-bit without RVQ (single-pass): noticeable. Thinking-mode models collapse early. Use RVQ.

**What models are supported?**

Any model that uses standard multi-head attention with `mlx_lm.make_cache()`. Tested: Mistral 7B, Qwen3 4B, Qwen3 8B, Qwen2.5 32B, Falcon3 7B, Phi-4, Gemma-4. Not supported: architectures with multi-latent attention (DeepSeek-V2) or non-standard cache formats.

**Does it work with streaming generation?**

Yes. The cache is stateful — each `generate()` call updates it. Stream tokens with the standard `mlx_lm` streaming API; the cache update happens transparently inside the model forward pass.

---

*VeloxQuant-MLX is MIT licensed. Contributions welcome — especially benchmark results for new models and hardware.*

# CacheGen — Entropy-Coded KV Cache via Token Locality

**Method id:** `cachegen` · **New in 0.16.0** · *Inspired by* [CacheGen (arXiv:2310.07240,
SIGCOMM 2024)](https://arxiv.org/abs/2310.07240) — **CacheGen-adapted (VeloxQuant-MLX
implementation)**, not a faithful port.

CacheGen is the first method in VeloxQuant-MLX to **entropy-code** the quantized
KV cache. Every other method packs codes at a fixed bit-width; CacheGen exploits
the cache's distributional structure — adjacent tokens' KV are similar, so the
*delta* between consecutive tokens' codes is low-entropy and compresses below the
fixed bit-width. The reconstruction is identical to plain group quant (lossless
over the codes); the win is in storage.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="cachegen",
    head_dim=128,
    cachegen_bits=4,  # base group-quant bit-width
    cachegen_group_size=32,
    cachegen_use_delta=True,  # token-delta transform before entropy coding
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## How it works

CacheGen builds on three observations from the paper:

1. **Token-wise locality** — adjacent tokens' KV vectors are similar, so the
   per-token *delta* of the quantized codes concentrates near zero and is far
   more compressible than the raw codes.
2. **Layer-wise sensitivity** — deeper layers tolerate coarser quantization;
   `cachegen_bits` can be set per layer through the builder.
3. **Arithmetic coding** — the low-entropy delta stream is compressed toward its
   Shannon entropy.

The pipeline per head:

1. Asymmetric min/max group quantize keys/values to integer codes (the same
   scheme as KIVI), exposing the codes.
2. Apply the reversible **token-delta** transform along the sequence axis.
3. Measure the **Shannon entropy** of the delta symbol stream; report the
   compressed size from it.
4. Reconstruct fp16 from the codes (identical to plain group quant).

## Adaptation notes

**Fidelity to the paper:** This is a VeloxQuant-MLX adaptation, not a faithful
port. Adaptations:

- **No serial range codec.** A true per-step arithmetic coder is sequential and
  would bottleneck MLX's parallel decode while adding no quality. Instead the
  entropy-coded byte size is **modelled from the measured Shannon entropy** of
  the delta stream — an honest estimate of what an ideal coder achieves, reported
  through `compressed_*_bytes`. The reconstructed tensors are exact (the entropy
  layer is lossless over the codes).
- **Never-worse-than-fixed-width cap.** A real arithmetic coder falls back to raw
  packing when the stream is incompressible, so the estimate is capped at the
  fixed-width packed size. On iid (incompressible) data the savings are exactly
  0%, never negative.

**Known limitation:** The win is a **storage** win, realized only on
token-correlated data (as real KV is). It does not reduce the working-set memory
at attend time (codes are dequantized to fp16 for SDPA), and on Apple Silicon's
bandwidth-bound decode it is lower-leverage than the low-rank/cross-layer methods.
No model-level benchmarks have been run yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_cachegen_cache.py` (12 tests) and
`veloxquant_mlx/tests/quantizers/test_cachegen.py` (9 tests):

- Reconstruction matches plain group quant exactly (lossless over codes)
- Token-delta transform is reversible (prefix-sum recovers the codes)
- Delta entropy < raw entropy on token-correlated data
- `entropy_savings > 0` on correlated data; `compressed < fixed_width`
- Savings never negative on iid data (the cap); `compressed <= fixed_width`
- Shannon entropy primitives: 0 for constants, 1 bit for 50/50, bounded by
  log2(alphabet)
- Byte-accounting ordering: `compressed <= fixed_width < fp16`
- Decode after prefill, determinism

The offline harness in `benchmark_scripts/benchmark_cachegen.py` reports ~17%
entropy savings on correlated 3-bit data and exactly 0% on iid data —
**synthetic, not model-level.**

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_cachegen.py`
is the planned script; until its `results.json` is committed, no throughput or
perplexity figures are claimed.

## When to use it

CacheGen is a **storage-compression** layer for token-correlated workloads — long,
coherent contexts where adjacent tokens' KV move slowly. It is orthogonal to the
quality-vs-bits methods: it does not change the reconstructed values, only how
compactly the codes are stored. For bandwidth-bound decode on Apple Silicon,
prefer [PALU](../algorithms/palu) or [SVDq](../algorithms/svdq); use CacheGen
when stored cache size (e.g. for offload/streaming) is the binding constraint.

| Method | Reconstruction | Compresses via | Win |
|--------|----------------|----------------|-----|
| KIVI-2bit | group quant | fixed 2-bit packing | bandwidth |
| CacheGen | identical to group quant | entropy coding of deltas | storage |

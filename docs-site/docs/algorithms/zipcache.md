# ZipCache — Saliency-Adaptive Per-Token Mixed Precision

**Method id:** `zipcache` · **New in 0.18.0** · *Inspired by* [ZipCache (arXiv:2405.14256)](https://arxiv.org/abs/2405.14256)
(He et al., NeurIPS 2024) — **ZipCache-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

ZipCache-adapted is the first method in VeloxQuant-MLX on the **per-token
mixed bit-width** axis. Three existing methods route tokens using the key-norm
proxy signal, but each makes a different decision:

| Method | Signal | Decision | Outcome |
|--------|--------|----------|---------|
| KIVI-Sink | key L2-norm | top-k positions → fp16 | binary: quantized vs not |
| AdaKV-proxy | mean key-norm per head | head-level budget reallocation | per-head bit budget |
| ZipCache-adapted | key L2-norm per token | hi_bits vs lo_bits — **both quantized** | per-token bit-width |

Both the hi-bit and lo-bit tokens remain quantized (not fp16). The salient
(high-norm) fraction gets finer quantization; the rest gets aggressive
compression. The effective average key rate is:

```
avg_bits = hi_fraction × hi_bits + (1 − hi_fraction) × lo_bits
```

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="zipcache",
    head_dim=128,
    zipcache_hi_bits=4,  # bit-width for salient (high-norm) tokens
    zipcache_lo_bits=2,  # bit-width for non-salient tokens
    zipcache_hi_fraction=0.20,  # fraction of tokens routed to hi_bits
    zipcache_group_size=32,  # token group size for min/max quantization
    zipcache_quantize_values=True,  # apply mixed-precision to values too
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zipcache_hi_bits` | `4` | Bit-width for salient (high-norm) tokens |
| `zipcache_lo_bits` | `2` | Bit-width for non-salient tokens |
| `zipcache_hi_fraction` | `0.20` | Fraction of tokens routed to `hi_bits` (0 = all lo, 1 = all hi) |
| `zipcache_group_size` | `32` | Token group size for min/max group quantization |
| `zipcache_quantize_values` | `True` | Apply uniform `hi_bits` to values too (`False` = keys-only) |

## How it works

Per head, per `update_and_fetch` call:

1. Compute the L2-norm of each incoming key token: `norm[i] = ‖k_i‖₂`.
2. Build a saliency mask: the top `ceil(S × hi_fraction)` tokens by norm are
   marked *salient* (True).
3. Quantize the salient rows at `hi_bits` with asymmetric min/max group quant.
4. Quantize the non-salient rows at `lo_bits` with the same scheme.
5. Store both sets of codes + their group (scale, zero) params + the bool mask.
6. Reconstruct fp16 by dequantizing each group and scattering rows back to
   their original positions. Hand fp16 K/V to the parent `mlx_lm` cache.

No `bits` attribute is exposed — SDPA stays on the clean fp16 path. The
`compression_ratio` and `effective_avg_bits` properties report the mixed-bit
accounting.

## Proxy limitation

The paper's true saliency signal is the **normalized attention score**, which
is not observable by a cache wrapper (only K and V tensors are visible at
`update_and_fetch` time). Key L2-norm is a proxy: attention-sink tokens
(highest attention weight) also exhibit large key norms. This proxy is weaker
than true attention scores — stated plainly, never hidden.

This is the third use of the key-norm proxy in this repo (after KIVI-Sink and
AdaKV-proxy). Each use makes a different decision from the same signal.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_zipcache_cache.py` (11 tests) and
`veloxquant_mlx/tests/quantizers/test_zipcache.py` (16 tests):

- Saliency mask selects exactly the top-`hi_fraction` tokens by key-norm
- 4-bit channel quant cosine > 0.995; 2-bit cosine > 0.8
- `compress` + `reconstruct` preserves `[S, D]` shape and fp16 dtype
- `hi_fraction=1.0` (all hi_bits) has lower MSE than `hi_fraction=0.0` (all lo_bits)
- Mixed-bit stored bytes < fp16 bytes (always)
- Mixed-bit stored bytes >= all-lo-bit baseline (hi-bit overhead is accounted)
- `effective_avg_bits` in `[lo_bits, hi_bits]` range
- Values-off path passes values through losslessly with zero `compressed_value_bytes`
- `hi_fraction=0` and `hi_fraction=1` edge cases run without error
- Decode accumulation: sequential single-token calls grow `seq_len` correctly
- Determinism and `for_model` config propagation

The offline harness in `benchmark_scripts/benchmark_zipcache.py` measures
reconstruction MSE vs uniform-lo-bit and uniform-hi-bit baselines on synthetic
high-norm-outlier data — **synthetic, not model-level.**

**No model-level benchmark has been run.** Until `results_zipcache.json` is
committed, no throughput or perplexity figures are claimed.

## When to use it

ZipCache-adapted is a **per-token quality allocation** layer: use it when a
small fraction of tokens are disproportionately important (attention-sink or
content-rich positions) and the rest can be compressed aggressively. It
operates entirely within the quantized space (unlike KIVI-Sink's fp16
protection) and is orthogonal to cross-layer methods (XQuant, MiniCache) and
error-feedback methods (GEAR).

| Method | Quality mechanism | Both tensors? | Overhead |
|--------|------------------|---------------|---------|
| KIVI-2bit | uniform group quant | ✅ | none (pure quant) |
| KIVI-Sink | fp16 for top-k sinks | ✅ | fp16 pool for sinks |
| ZipCache-adapted | hi_bits for top-norm tokens | ✅ | mask + hi-bit param overhead |
| GEAR | base quant + low-rank + sparse correction | ✅ | L, R factors + sparse triples |

# GEAR — Error-Feedback KV Cache (Residual Low-Rank + Sparse Outliers)

**Method id:** `gear` · **New in 0.17.0** · *Inspired by* [GEAR (arXiv:2403.05527)](https://arxiv.org/abs/2403.05527)
(Kang et al.) — **GEAR-adapted (VeloxQuant-MLX implementation)**, not a faithful
port.

GEAR is the first method in VeloxQuant-MLX on the **error-feedback** axis. Every
other method picks a bit-width or a cache layout and lives with the
quantization error; GEAR makes *any* ultra-low-bit base quantizer near-lossless
by reconstructing what it threw away. For a KV matrix `X` it stores the
three-part decomposition:

```
X  ~=  Quant_b(X)  +  L . R  +  S
```

- `Quant_b(X)` — the **base**: most entries at ultra-low precision (the repo's
  shared asymmetric min/max group quant, the same scheme as KIVI/CacheGen).
- `L . R` — a **low-rank** approximation of the quantization residual
  `E = X - dequant(Quant_b(X))`. The residual of a coherent KV matrix is itself
  low-rank, so a small rank recovers most of the lost signal cheaply.
- `S` — a **sparse** matrix correcting the top-`rho` outlier entries by
  magnitude that the low-rank term could not absorb.

Unlike CacheGen (whose reconstruction is identical to group quant and whose win
is a storage-byte model), GEAR's reconstruction is a genuine lossy
reconstruction that **recovers quality** the base bit-width alone would lose.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="gear",
    head_dim=128,
    gear_bits=2,  # ultra-low base bit-width
    gear_rank=8,  # residual low-rank (keep small: this is the premise)
    gear_energy_threshold=0.90,  # used when gear_rank is None
    gear_sparse_fraction=0.005,  # top-|residual| fraction kept exact
    gear_group_size=32,
    gear_quantize_values=True,  # apply GEAR to values too (False = keys only)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## How it works

Per head, per `update_and_fetch` call (the prefill batch when sequence length is
greater than 1, a single token at decode):

1. Base group-quantize the keys/values to integer codes and dequantize, giving
   the base reconstruction.
2. Form the residual `E = X - base_recon`.
3. Truncated SVD of `E` gives `L . R` (rank chosen by `gear_rank`, or by
   `gear_energy_threshold` when rank is None). Subtract it: the post-low-rank
   residual is what remains.
4. Keep the top-`gear_sparse_fraction` entries of the post-low-rank residual by
   magnitude as the sparse correction `S`.
5. Reconstruct fp16 as `base_recon + L . R + S` and hand it to the parent
   `mlx_lm` cache, so SDPA stays on the clean fp16 path.

The shared truncated-SVD helper (`_quant_utils._truncated_svd`) is the same one
SVDq and PALU use — GEAR applies it to the quantization *error* rather than the
*signal*.

## Adaptation notes

**Fidelity to the paper:** This is a VeloxQuant-MLX adaptation, not a faithful
port. Adaptations:

- **No fused dequant kernel.** GEAR's reference implementation streams and fuses
  dequant into attention with a custom CUDA kernel. We reconstruct fp16 then call
  MLX SDPA. Consequence: the *stored* cache shrinks, but the working set *during*
  attention is the reconstructed fp16 K/V — attend-time peak memory is not
  reduced, only the stored cache size.
- **Per-call residual SVD.** The residual SVD is computed on the tensor the cache
  holds at each call, reusing the SVDq/PALU prefill-SVD pattern. No separate
  calibration pass.
- **Borrowed base quantizer.** The base `Quant_b` is the repo's shared group
  quant, so GEAR composes over an existing, already-tested quantizer plus an
  error-feedback layer.

**Overhead caveat:** the low-rank factors cost `(N + D) * r * 2` bytes and the
sparse triples `nnz * 6` bytes. For these to stay below the fp16 budget the rank
must be genuinely *low* relative to `D` (the GEAR premise). On tiny head dims
with a near-`D/2` rank the error-feedback overhead can exceed fp16 — keep
`gear_rank` small (or use `gear_energy_threshold`). The overhead is reported
honestly through `compressed_*_bytes`; it is not silently hidden.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_gear_cache.py` (10 tests) and
`veloxquant_mlx/tests/quantizers/test_gear.py` (13 tests):

- GEAR reconstruction MSE **strictly below** base-quant-alone on low-rank +
  outlier data (the core claim)
- Low-rank-alone and sparse-alone each reduce error vs base; `rank=0, sparse=0`
  collapses exactly to the base group quant
- A genuinely rank-`r` residual is recovered by the low-rank term to `< eps`;
  sparse selection picks the true top-magnitude entries
- Byte-accounting ordering `base_only <= compressed <= fp16` at realistic head
  dim with low rank; component byte sum matches
- `error_recovery_ratio` in `(0, 1]`; values-off path leaves values fp16;
  decode accumulation; determinism; build via both `create` and `for_model`

The offline harness in `benchmark_scripts/benchmark_gear.py` measures
reconstruction-MSE improvement, stored bytes, and error-recovery on synthetic
low-rank-plus-outlier data — **synthetic, not model-level.**

**No model-level benchmark has been run.** Until `results_gear.json` is committed,
no throughput or perplexity figures are claimed.

## When to use it

GEAR is a **quality-recovery** layer: use it to push the base bit-width lower
(e.g. 2-bit) while recovering accuracy through the residual low-rank + sparse
correction, when the KV residual is low-rank (coherent, long contexts) and a few
outliers dominate the remaining error. It is orthogonal to the bit-width and
cache-layout methods — it adds error feedback on top of a base quantizer. For a
pure storage/bandwidth win without quality recovery, prefer
[PALU](../algorithms/palu), [SVDq](../algorithms/svdq), or
[CacheGen](../algorithms/cachegen).

| Method | Reconstruction | Compresses / recovers via | Win |
|--------|----------------|---------------------------|-----|
| KIVI-2bit | group quant | fixed 2-bit packing | bandwidth |
| CacheGen | identical to group quant | entropy coding of deltas | storage |
| GEAR | base quant **+ error feedback** | low-rank residual + sparse outliers | quality at low bits |

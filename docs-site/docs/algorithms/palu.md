# PALU — True Low-Rank Latent Storage for Keys *and* Values

**Method id:** `palu` · **New in 0.15.0** · *Inspired by* [PALU (arXiv:2407.21118,
ICLR 2025)](https://arxiv.org/abs/2407.21118) — **PALU-adapted (VeloxQuant-MLX
implementation)**, not a faithful port.

PALU is the first method in VeloxQuant-MLX that keeps the KV cache itself in
**low-rank latent form** — it stores the projected codes `[S, r]` directly and
reconstructs full keys/values to fp16 *only at attend time*. Both keys **and**
values are compressed via shared per-group projections, layered with mixed-bit
quantization, for a full-KV effective rate well below 1 bit/element on low-rank
data.

## How it differs from SVDq

The repo already ships [SVDq](../algorithms/svdq), which also uses SVD. The
distinction is structural:

| | SVDq | **PALU** |
|---|---|---|
| Compresses | Keys only | **Keys and values** |
| Cache stores | Full fp16 keys (reconstructed) | **Latent `[S, r]` codes** |
| Win | Bandwidth accounting | **Storage + bandwidth** |
| Values | fp16 | Low-rank (+ optional mixed-bit) |
| Projection | One global `V` | **Group-head: heads share a projection** |

SVDq reconstructs full fp16 keys and hands them to the parent `mlx_lm` cache, so
its compression is a byte-accounting story. PALU bypasses the parent fp16 buffer
entirely — the cache genuinely holds `[S, r]`, not `[S, D]`.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="palu",
    head_dim=128,
    # Rank selection — explicit or via energy threshold:
    palu_rank=None,  # None → use energy threshold
    palu_energy_threshold=0.90,  # retain 90% of singular value energy
    # Group-head low-rank decomposition:
    palu_n_head_groups=4,  # heads per shared projection
    # Mixed-bit latent quantization:
    palu_hi_bit=4,  # top-25% latent channels (by singular value)
    palu_lo_bit=2,  # remaining 75%
    palu_hi_fraction=0.25,
    palu_group_size=32,
    palu_quantize_values=True,  # False → low-rank-only (fp16 latents)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

For a fixed rank and low-rank-only values:

```python
config = KVCacheConfig(
    method="palu",
    head_dim=128,
    palu_rank=32,  # explicit rank
    palu_quantize_values=False,  # values low-rank but fp16 latents
)
```

## How it works

**Prefill phase** (triggered once on the first batch, S > 1):

1. Partition the `H` attention heads into `palu_n_head_groups` contiguous
   groups (PALU's **group-head decomposition**, G-LRD).
2. For each group, stack its heads along the token axis and run truncated SVD →
   shared projection `V_g ∈ R^(D×r)` and mean `μ_g ∈ R^D`. Rank `r` is set by an
   explicit value or an energy threshold (≥90% of singular value energy).
3. To keep buffer shapes clean, a single rank `r` (the minimum retained across
   groups) is used for every group.
4. Project each head into its group's latent space: `L = (x − μ_g) @ V_g → [S, r]`.
5. **Mixed-bit quantize** the latents: top-25% of channels by singular value →
   4-bit, the rest → 2-bit (reusing the SVDq latent coder).
6. Store the quantized latents directly. Reconstruct fp16 K/V (`L @ V_gᵀ + μ_g`)
   only for the downstream attention call.

**Decode phase** (per new token):

1. Project the new key/value into the already-stored group projections.
2. Mixed-bit quantize and append to the latent buffers.
3. Reconstruct the full sequence to fp16 for attention.

**Why this works:** Within a head group, keys (and values) share a dominant
low-rank subspace — a few singular directions carry most of the attention-
relevant variance. Storing the compact latent and quantizing where channel
importance is explicitly ordered by singular value gives a much better
error-per-byte tradeoff than uniform quantization in the original space, on
*both* tensors.

## Effective bit-width

For default settings (r ≈ 0.25·D via energy threshold, hi_fraction = 0.25):

```
effective_bits ≈ (r/D) × (0.25 × 4 + 0.75 × 2) = 0.25 × 2.5 ≈ 0.6 bits/element
```

The exact rate depends on the rank chosen at prefill. The `assigned_avg_bits`
property reports the realised effective bit-width (the max of the key and value
latent rates). With `palu_quantize_values=False`, values store fp16 latents at a
rate of `16 × r/D` bits.

## Adaptation notes

**Fidelity to the paper:** This is a VeloxQuant-MLX adaptation of PALU, not a
faithful port. Adaptations:

- **Projection timing:** The paper fits projections offline on a calibration
  set; this implementation fits them from the prefill batch
  (`update_and_fetch` receives all prefill keys/values as a batch when S > 1),
  requiring no separate calibration step.
- **Uniform rank across groups:** Each group's SVD may retain a different rank
  under an energy threshold; we use the minimum so latent buffers share one `r`.
- **Mixed-bit compose:** The paper's latents are quantized; we reuse the
  already-tested SVDq mixed-bit latent coder rather than a bespoke one.
- **No fused kernel:** PALU's fused low-rank-reconstruction attention CUDA
  kernel is **not** ported. We reconstruct fp16 then call MLX SDPA. This means
  the *storage* is low-rank but the working set at attend time is briefly the
  reconstructed fp16 K/V — peak memory during attention is not reduced, only
  stored cache size. Documented as a known simplification.

**Known limitation:** PALU's quality advantage holds on low-rank structured data
(as real LLM K/V are). On uniformly random data the low-rank assumption fails and
it may not outperform naive quantization. No model-level benchmarks have been run
yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_palu_cache.py` (13 tests) and
`veloxquant_mlx/tests/quantizers/test_palu.py` (9 tests), on synthetic data:

- Group-head projections stored after prefill (`V_g` shape `[D, r]` per group)
- Output shape and dtype preserved (fp16, `[B, H, S, D]`) on prefill and decode
- **Storage is latent:** per-head buffers hold `[S, r]`, not `[S, D]`; the parent
  fp16 ring buffer is never populated (`cache.keys is None`)
- On synthetic rank-8 data (D=64), PALU r=8 achieves lower MSE than naive 2-bit
  for **both** keys and values
- Decode accumulation produces valid fp16 output with no NaNs; offset grows by
  exactly the decode steps
- Byte accounting: both `compressed_key_bytes < fp16_key_bytes` **and**
  `compressed_value_bytes < fp16_value_bytes`
- `palu_quantize_values=False` keeps fp16 latents and still compresses via rank
- `assigned_avg_bits < 2.0` at default settings
- Energy-threshold rank selection returns a rank in [1, D]
- Group SVD recovers a planted rank-r subspace to MSE < 1e-3
- Deterministic: two caches on the same data produce identical output

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_palu.py`
is the planned script; until its `results.json` is committed, no throughput or
perplexity figures are claimed. The offline reconstruction harness in that script
confirms PALU beats naive 2-bit on both K and V on synthetic low-rank data.

## When to use it

PALU targets the extreme full-KV low-memory regime — when you need to hold very
long contexts on Apple Silicon and want **both** keys and values stored compactly,
not just keys. It is complementary to [SVDq](../algorithms/svdq) (keys-only
low-rank, values fp16), [KIVI](../algorithms/kivi) (group quantization on both),
and [RaBitQ](../algorithms/rabitq) (1-bit vector quantization).

PALU's mixed-bit latent split is **fixed** (top-25%/75% by singular value, on
both K and V). [KVTC-adapted](../algorithms/kvtc) shares PALU's full-KV,
local-basis-at-prefill shape, but replaces the fixed split with a
**dynamic-programming-optimal** per-component bit allocation (may drop a
component to exactly 0 bits) and adds an entropy-coding stage on top of the
quantized codes.

| Method | Key bits | Value bits | Stores | Prefill cost |
|--------|----------|------------|--------|--------------|
| KIVI-2bit | 2 | 2 | full fp16 (dequant) | none |
| SVDq (default) | ~1.25 | 16 (fp16) | full fp16 keys | SVD once |
| **PALU (default)** | **~0.6** | **~0.6** | **latent `[S, r]`** | group SVD once |
| [KVTC-adapted](../algorithms/kvtc) | DP-optimal (0–8) | DP-optimal (0–8) | latent codes + entropy-coded payload | local PCA once |

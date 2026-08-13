# MiniCache — Cross-Layer Depth-Dimension Merging

**Method id:** `minicache` · **New in 0.16.0** · *Inspired by* [MiniCache (arXiv:2405.14366,
NeurIPS 2024)](https://arxiv.org/abs/2405.14366) — **MiniCache-adapted (VeloxQuant-MLX
implementation)**, faithful to the SLERP-merge + retention core, adapted at the
integration boundary via a shared coordinator.

MiniCache compresses the KV cache **across network depth**: adjacent layers in the
middle-to-deep portion of the model have nearly identical KV *directions*, so a
pair of layers is merged into **one shared direction** plus each layer's own
per-token magnitude. A pair of layers costs roughly one. High-divergence token
pairs are kept unmerged (the retention set).

## How it differs from XQuant

The repo already ships [XQuant](../algorithms/xquant), also cross-layer. The
distinction:

| | XQuant | **MiniCache** |
|---|---|---|
| Mechanism | reuses an anchor's quantized **codes** | merges the **tensors** via SLERP |
| Shared across layers | code assignment | direction vector |
| Per-layer kept | own scale/zero | own magnitude scalars |
| Quantizes | yes (low-bit) | no — operates in fp16 direction space |
| Unmergeable handling | residual correction | token retention set |

XQuant shares the *bin assignment*; MiniCache shares the *direction itself*.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="minicache",
    head_dim=128,
    minicache_start_frac=0.5,  # only merge layers past mid-depth
    minicache_group_size=2,  # merge adjacent pairs
    minicache_retention_threshold=0.9,  # cosine below which a token pair is kept
    minicache_slerp_t=0.5,  # SLERP midpoint
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

:::note[Requires `for_model`]
Cross-layer merging needs the shared `MiniCacheCoordinator` that
`KVCacheBuilder.for_model()` builds. Constructing a single cache via the factory
yields a degenerate (coordinator-less) **primary** that behaves as a lossless
fp16 passthrough — useful for unit testing the primary path in isolation.
:::

## How it works

**Role assignment** (`pair_layers_depth`, at build time): attention layers below
`minicache_start_frac` of depth are standalone **primary** layers (never merged —
early layers are not similar enough). Middle-to-deep layers are grouped; the first
of each group is **primary**, the rest are **merge**.

**Per forward pass**, for each merge group:

1. The **primary** layer stores its true KV to the coordinator and reconstructs
   itself losslessly (it is seen before its merge partner).
2. The **merge** layer fetches the primary's KV for the same token range and:
   - decomposes both layers' vectors into magnitude + unit direction,
   - **SLERP**-interpolates the two directions into one shared unit vector,
   - keeps each layer's own per-token magnitude,
   - reconstructs as `magnitude × shared_direction`.
3. **Token retention:** token pairs whose direction cosine is below
   `minicache_retention_threshold` are *not* merged — both layers' full vectors
   are kept. This caps the worst-case merge error.

Storage charged to the pair: one shared direction (fp16) + two magnitude scalars
per token + full vectors for the few retained tokens — roughly one layer's cost
for two layers.

## Adaptation notes

**Fidelity to the paper:** Faithful to the SLERP magnitude/direction
decomposition and the token-retention strategy. Adapted at the integration
boundary: rather than modifying the attention forward pass, all per-layer caches
share a `MiniCacheCoordinator` (the same pattern XQuant uses). The primary layer
stores its KV so the later-arriving merge layer can perform the merge — both
reconstructions then use the shared direction.

**Known limitations:**
- MiniCache merges *directions* in fp16; it does not additionally low-bit
  quantize. Compose with KIVI/PALU for further compression (future work).
- The merge happens at attend time on the reconstructed tensors; the working-set
  memory during attention is not reduced — the win is the stored cache size.
- No model-level benchmarks have been run yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_minicache_cache.py` (11 tests) and
`veloxquant_mlx/tests/quantizers/test_minicache.py` (11 tests):

- Role assignment: early layers all primary; middle-to-deep has merge layers
- SLERP endpoints (`t=0`/`t=1`) return the inputs; output is always unit-norm;
  collinear directions fall back to normalized lerp
- `magnitude × direction` recovers the original vector
- Similar layers (cosine ≈ 0.9995) merge with MSE < 0.0002 and 0% retention
- Opposite-direction tokens are 100% retained and reconstructed exactly
- Merge preserves each layer's own magnitude (a 3× magnitude ratio survives the
  shared direction)
- `n_retained + n_merged == total`; merge layer `compressed <= fp16`
- Degenerate (no-coordinator) primary is a lossless passthrough
- Coordinator `max_ctx` guard; determinism

The offline harness in `benchmark_scripts/benchmark_minicache.py` reports
adjacent-layer direction cosine 0.9995 (similar) → MSE 0.0002, and -0.01
(dissimilar) → 100% retention, MSE 0 — **synthetic, not model-level.**

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_minicache.py`
is the planned script; until its `results.json` is committed, no throughput or
perplexity figures are claimed.

## When to use it

MiniCache targets models deep enough to have a sizable middle-to-deep region of
similar layers. It composes the cross-layer axis with whatever per-layer scheme
you use on the unmerged early layers. It is the natural complement to
[XQuant](../algorithms/xquant) and [xKV](../algorithms/xkv): XQuant reuses
codes, MiniCache merges tensors, and xKV jointly factorizes a whole group into
one shared subspace — three different routes to inter-layer redundancy.

| Method | Cross-layer mechanism | Quantizes |
|--------|----------------------|-----------|
| XQuant | code reuse + residual | yes (low-bit) |
| MiniCache | SLERP direction merge + retention | no (fp16 directions) |
| xKV | joint SVD -> shared subspace across a group | yes (uniform-bit latents) |

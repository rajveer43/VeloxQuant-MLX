# SVDq — Sub-2-bit Key Cache via Offline SVD

**Method id:** `svdq` · **New in 0.10.0** · *Inspired by* [SVDq (arXiv:2502.15304,
Feb 2025)](https://arxiv.org/abs/2502.15304) — **SVDq-adapted (VeloxQuant-MLX
implementation)**, unreviewed preprint, not a faithful port.

SVDq is the first method in VeloxQuant-MLX that compresses keys via **linear
projection into a low-rank latent space**, achieving an effective key bit-width
of **~1.25 bits/element** — a 12.8× memory bandwidth reduction vs fp16 keys.
Values are left at fp16 throughout.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="svdq",
    head_dim=128,
    # Rank selection — either explicit or via energy threshold:
    svdq_rank=None,  # None → use energy threshold
    svdq_energy_threshold=0.95,  # retain 95% of singular value energy
    # Mixed-precision latent quantization:
    svdq_hi_bit=4,  # top-25% channels (by singular value)
    svdq_lo_bit=2,  # remaining 75%
    svdq_hi_fraction=0.25,
    svdq_group_size=32,
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

For a specific rank:

```python
config = KVCacheConfig(
    method="svdq",
    head_dim=128,
    svdq_rank=32,  # explicit rank
)
```

## How it works

**Prefill phase** (triggered once on the first batch of keys):

1. Compute the mean key vector K̄ and subtract it to center the key matrix.
2. Run truncated SVD: K − K̄ ≈ U · Σ_r · V^H, retaining rank r determined
   by either an explicit value or an energy threshold (≥95% of singular value
   energy by default).
3. Store V ∈ R^(D×r) (the right singular vectors) and K̄ as layer state.
   These are O(D²) and negligible in memory relative to long sequences.
4. Project keys into the latent space: L = (K − K̄) @ V → shape [S, r].
5. Apply **mixed-precision group quantization** to L:
   - Top-25% of latent channels by singular value magnitude → 4-bit
   - Remaining 75% → 2-bit
6. Reconstruct full fp16 keys for the downstream attention call.

**Decode phase** (per new token):

1. Project the new key: l = (k − K̄) @ V → [1, r].
2. Quantize with the same mixed-bit scheme.
3. Reconstruct fp16 and pass through to attention.

**Why this works:** Real LLM key caches are strongly low-rank — a few singular
directions carry most of the attention-relevant variance. SVDq exploits this by
quantizing in the compact latent space where each channel's importance is
explicitly ordered by singular value magnitude, enabling principled mixed-bit
allocation.

## Effective bit-width

For default settings (r = 0.5D via energy threshold, hi_fraction = 0.25):

```
effective_bits ≈ (r/D) × (0.25 × 4 + 0.75 × 2) = 0.5 × 2.5 = 1.25 bits/key element
```

The `assigned_avg_bits` property reports the actual effective bit-width for
the rank chosen at prefill time.

## Adaptation notes

**Fidelity to the paper:** This is a VeloxQuant-MLX adaptation of the SVDq
preprint, not a faithful port. Adaptations:

- **SVD timing:** The paper computes SVD offline over a calibration set. This
  implementation computes it from the prefill key batch (`update_and_fetch`
  receives all prefill keys as a batch when S > 1), requiring no separate
  calibration step.
- **Mixed-bit routing:** The paper uses importance-aware allocation; this
  implementation uses a fixed hi/lo split ordered by singular value magnitude,
  which is both deterministic and calibration-free.
- **Values:** Left at fp16; the paper's finding that values have weak low-rank
  structure is taken at face value.

**Known limitation:** SVDq's quality advantage over naive 2-bit quantization
holds on low-rank structured data (as real LLM keys are). On uniformly random
data it may not outperform naive quantization — the low-rank assumption is load-
bearing. No model-level benchmarks have been run yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_svdq_cache.py` (12 tests, synthetic data):

- SVD projection stored correctly after prefill (V shape [D, r], K̄ shape [D])
- Output shape and dtype preserved (fp16, [B, H, S, D])
- On synthetic rank-8 data (D=64), SVDq with r=8 achieves lower MSE than
  naive 2-bit quantization in the original key space
- Decode calls after prefill produce valid fp16 output with no NaNs
- `compressed_key_bytes < fp16_key_bytes`
- `assigned_avg_bits < 2.0` at default settings
- Energy-threshold rank selection returns a rank in [1, D]
- Deterministic: two caches on same data produce identical output

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_svdq.py`
is the planned script; until its `results.json` is committed, no throughput or
perplexity figures are claimed.

## When to use it

SVDq targets the extreme low-memory regime — when you need to hold very long
contexts on Apple Silicon and are willing to accept the SVD overhead at the
start of each sequence. It is complementary to [KIVI](../algorithms/kivi)
(which compresses both K and V with group quantization) and
[RaBitQ](../algorithms/rabitq) (which uses 1-bit vector quantization for both
tensors).

SVDq's mixed-bit latent split is a **fixed** top-25%/75% rule chosen by hand.
[KVTC-adapted](../algorithms/kvtc) uses the same local-PCA-then-latent-quantize
shape but replaces that fixed split with a **dynamic-programming-optimal**
per-component bit allocation (which can drop a component to exactly 0 bits)
plus an entropy-coding stage — and compresses **both** K and V, not keys-only.

| Method | Key bits | Value bits | Prefill cost |
|--------|----------|------------|--------------|
| KIVI-2bit | 2 | 2 | none |
| RaBitQ | ~1 (VQ) | ~4 (MSE) | k-means |
| SVDq (default) | ~1.25 | 16 (fp16) | SVD once |
| [KVTC-adapted](../algorithms/kvtc) | DP-optimal (0–8) | DP-optimal (0–8) | local PCA once |

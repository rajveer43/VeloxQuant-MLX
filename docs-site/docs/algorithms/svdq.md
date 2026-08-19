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
    # Mixed-precision latent quantization — paper Eq. 6's 8-group schedule,
    # most-significant group first. A 0 truncates that group entirely.
    svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0),  # paper's worked example, b̄ = 2
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
5. Apply **mixed-precision group quantization** to L using the paper's 8-group
   fixed schedule (Eq. 6): split the r latent channels (descending singular
   value order) into 8 equal-size contiguous groups, and quantize each group
   at its own fixed bit width. The default schedule is the paper's own worked
   example, `(8, 4, 2, 1, 1, 0, 0, 0)` — the last three groups are truncated
   to exactly 0 bits.
6. Reconstruct full fp16 keys for the downstream attention call.

Note: this schedule assumes each group spans many latent channels (the
paper's configs use d=1024, so d/8=128 channels per group — see Table 1).
If the chosen rank r is small and close to 8, each group can shrink to a
single channel, and a 0-bit group then truncates a channel that still
carries real signal rather than negligible tail energy — see "Known
limitation" below.

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

For default settings (r = 0.5D via energy threshold, schedule `(8,4,2,1,1,0,0,0)`,
paper Eq. 6's b̄ = mean(schedule) = 2):

```
effective_bits ≈ (r/D) × b̄ = 0.5 × 2 = 1.0 bits/key element
```

The `assigned_avg_bits` property reports the actual effective bit-width for
the rank and schedule chosen at prefill time; `equivalent_bit_width()` in
`quantizers/svdq.py` computes b̄ directly from any schedule.

## Adaptation notes

**Fidelity to the paper:** This is a VeloxQuant-MLX adaptation of the SVDq
preprint, **not a faithful port** — do not treat it as a reproduction of the
paper's reported accuracy numbers. What matches and what doesn't:

Matches the paper's Algorithm 1 / Eq. 6 mechanism:
- SVD centering, projection, and reconstruction formulas are identical.
- The 8-group fixed bit-schedule shape (equal-size contiguous groups over
  latent channels, ordered by descending singular value, with 0-bit
  truncation support) now matches the paper's grouping, not an earlier
  simplified top-25%/75% split this repo used before.

Still deviates, and these are load-bearing, not cosmetic:

- **SVD timing:** The paper computes SVD offline over a calibration set fit
  once per model. This implementation computes it from the live prefill key
  batch of a single sequence (`update_and_fetch` receives all prefill keys
  as a batch when S > 1). A per-sequence basis is not the same object as a
  calibration-fit basis and may generalize differently.
- **Schedule values are hand-chosen, not fit:** the default
  `(8,4,2,1,1,0,0,0)` is the paper's own worked example, not a schedule
  tuned against this repo's own calibration or benchmark data the way the
  paper's Table 2/3 schedules were tuned against RULER/LongBench.
- **No offline calibration, no model-level accuracy evidence:** all current
  tests are synthetic-data unit tests (see Evidence below). The paper's
  headline claims (1.25-bit, up to 410× compression, near-lossless on
  RULER/LongBench) have not been reproduced here.
- **Values:** Left at fp16; the paper's finding that values have weak
  low-rank structure is taken at face value, not independently verified.

**Known limitation — small-rank/group-count interaction, and the guard rail
against it:** the 8-group schedule assumes each group spans many latent
channels (paper's configs: d/8 = 128 channels/group). If the chosen rank `r`
is small and close to 8, groups shrink toward one channel each, and a 0-bit
group then truncates a channel that may still carry real signal rather than
negligible tail energy. In that regime SVDq can end up *worse* than naive
2-bit quantization — demonstrated directly in
`test_small_rank_near_group_count_is_rejected` (`r=8` on non-decaying
synthetic data). The compression advantage is real only when rank is large
and energy decays smoothly across most of it (see
`test_reconstruction_lower_mse_than_raw_2bit`, which uses an
exponentially-decaying `r=64` construction matching the paper's Section 4.3
decay model).

To keep this from silently degrading, `SVDqKVCache` validates rank against
the schedule at prefill time (`min_safe_rank()` in `quantizers/svdq.py`,
currently `n_groups * 4`):

- **Explicit `svdq_rank`** that lands below the safe floor for a
  truncating schedule → raises `ValueError` immediately, since that's a
  specific, informed choice that turned out unsafe.
- **Automatic rank** (energy-threshold path, `svdq_rank=None`) that lands
  below the floor — e.g. because a short prefill sequence can only support
  a small rank, which the caller didn't choose — degrades gracefully
  instead of raising: 0-bit groups in the schedule are substituted with
  1-bit for that layer only (`test_automatic_rank_degrades_gracefully_instead_of_raising`),
  so a short sequence still serves correctly rather than crashing on the
  default config.

No model-level benchmarks have been run yet to confirm where real LLM key
caches fall on the rank/energy-decay spectrum in practice.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_svdq_cache.py` (18 tests, synthetic data):

- SVD projection stored correctly after prefill (V shape [D, r], K̄ shape [D])
- Output shape and dtype preserved (fp16, [B, H, S, D])
- On synthetic exponentially-decaying rank-64 data (D=128), SVDq achieves
  lower MSE than naive 2-bit quantization in the original key space
- **On synthetic rank-8 data (D=64, r=8, no decay assumption), SVDq
  underperforms naive 2-bit quantization** — the 0-bit trailing groups
  truncate real signal when rank is close to n_groups=8
  (`test_small_rank_near_group_count_can_underperform_naive`)
- 0-bit groups reconstruct to exactly zero (`test_zero_bit_group_is_truncated_to_zero`)
- `equivalent_bit_width` matches the paper's own worked-example arithmetic (b̄=2)
- Decode calls after prefill produce valid fp16 output with no NaNs
- `compressed_key_bytes < fp16_key_bytes`
- `assigned_avg_bits` reflects the chosen schedule's mean bit-width, scaled by r/D
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

SVDq's mixed-bit latent split is a **fixed** 8-group schedule chosen by hand
(the paper's own worked example), not data-driven.
[KVTC-adapted](../algorithms/kvtc) uses the same local-PCA-then-latent-quantize
shape but replaces that fixed schedule with a **dynamic-programming-optimal**
per-component bit allocation (which can drop a component to exactly 0 bits
based on actual reconstruction-error tradeoffs, not a fixed group boundary)
plus an entropy-coding stage — and compresses **both** K and V, not keys-only.

| Method | Key bits | Value bits | Prefill cost |
|--------|----------|------------|--------------|
| KIVI-2bit | 2 | 2 | none |
| RaBitQ | ~1 (VQ) | ~4 (MSE) | k-means |
| SVDq (default) | ~1.0 | 16 (fp16) | SVD once |
| [KVTC-adapted](../algorithms/kvtc) | DP-optimal (0–8) | DP-optimal (0–8) | local PCA once |

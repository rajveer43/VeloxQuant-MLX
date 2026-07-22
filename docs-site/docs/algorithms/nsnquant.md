# NSNQuant — Calibration-Free Universal-Codebook VQ

**Method id:** `nsnquant` · **New in 0.28.0** · *Inspired by* [NSNQuant
(arXiv:2505.18231, NeurIPS 2025)](https://arxiv.org/abs/2505.18231) —
**NSNQuant-adapted (VeloxQuant-MLX implementation)**, faithful to the
Normalize-Shift-Normalize + Hadamard + universal-codebook core, adapted at the
integration boundary (post-RoPE keys, explicit value Hadamard) and simplified
on codebook training and metadata packing (see
[Adaptation notes](#adaptation-notes)).

NSNQuant inverts the usual vector-quantization relationship: instead of
fitting a codebook to the data (which needs calibration and breaks under
distribution shift), it **reshapes the data to match a fixed code**. A
Normalize-Shift-Normalize (NSN) transform plus a Hadamard rotation maps K/V
token vectors onto (approximately) the standard normal distribution — so one
codebook, built **offline from synthetic Gaussian samples and never from
model activations**, quantizes any model, layer, or dataset at 1–2 bits per
element. Calibration-free by construction.

## How it differs from the repo's other VQ methods

Every other VQ method here either adapts the codebook to the data or uses a
fixed *geometric* code:

| | RVQ / CommVQ | RaBitQ / VecInfer / [PolarQuant](../algorithms/polarquant) | **NSNQuant** |
|---|---|---|---|
| Codebook | fit to the sequence (k-means/EM) | data-independent geometry (signs, polar grids) | fixed, built offline from `randn` samples |
| Calibration | per-sequence fitting | none | none |
| Mechanism | adapt code to data | code ignores distribution | **adapt data to code** (NSN + Hadamard Gaussianization) |
| Fails when | distribution shifts mid-stream | distribution far from the geometry's sweet spot | Gaussianization is imperfect (e.g. post-RoPE structure) |

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="nsnquant",
    head_dim=128,
    nsn_bits=2,               # 2 = sign mask + index (~2 b/elem), 1 = index only
    nsn_residual_length=64,   # fp16 chunk buffer; paper suggests 128 for 1-bit
    nsn_codebook_size=256,    # centroids (256 -> uint8 indices)
    nsn_subvector_dim=8,      # VQ subvector dimension (paper: 8)
    nsn_seed=1234,            # codebook RNG seed (synthetic Gaussian)
    nsn_max_ctx=8192,         # per-layer token budget
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

No coordinator: NSNQuant is a **single-layer** wrapper (the simplest wrapper
shape in the repo). The universal codebook is deterministic per
`(nsn_codebook_size, nsn_subvector_dim, nsn_seed)` and cached process-wide, so
every layer shares one codebook at zero marginal cost.

## How it works

**Per chunk** of `nsn_residual_length` tokens (each chunk is self-contained —
its statistics are computed online from that chunk alone, never calibrated,
never frozen across chunks):

1. **Normalize** (token-wise): scale each token to norm `sqrt(d)`; keep the
   per-token scale `s1`. Suppresses outlier tokens.
2. **Shift** (channel-wise): subtract the chunk's channel mean `o`,
   zero-centering the distribution.
3. **Normalize** again (token-wise): rescale to norm `sqrt(d)`; keep `s2`.
   (This slightly perturbs the zero mean; the paper shows the deviation is
   negligible.)
4. **Hadamard transform** (`mx.hadamard_transform`, O(d log d),
   Metal-accelerated): decorrelates channels so the empirical distribution
   closely matches an isotropic standard normal — the distribution the
   universal codebook was built for.
5. **Vector quantization**: 8-dim subvectors matched by cosine against the
   universal codebook. `nsn_bits=2`: uint8 sign mask + uint8 index into a
   positive-orthant "magnitude" codebook (2 bits/element). `nsn_bits=1`:
   uint8 index into a "signed" codebook only (1 bit/element).
6. **Restoration** on fetch: codebook lookup (+ sign restore), renormalize to
   `sqrt(d)`, inverse Hadamard, then `x_hat = s1 * (s2 * x_nsn + o)`.

**Both keys and values** are quantized (mirroring the paper) — unlike the
keys-only [SVDq](../algorithms/svdq)/[xKV](../algorithms/xkv) precedent.

**Decode**: new tokens accumulate at fp16; every time `nsn_residual_length`
tokens age past the quantized frontier, that chunk is flushed through the
pipeline as one unit. Prefill and decode produce identical chunk boundaries
by construction, so the quantized state is path-independent — verified by
test.

## Byte accounting

Per chunk of `r` tokens at head_dim `D` (per tensor, per head):

- **payload** — `r * (D/8)` bytes of indices (+ `r * (D/8)` bytes of sign
  masks at 2-bit): exactly `nsn_bits` bits/element.
- **metadata** — fp16 `s1` + `s2` per token (4 bytes/token) and one fp16
  channel mean `o` per chunk (`2D` bytes, amortized over the chunk). All
  counted in `compressed_*_bytes` — the paper 4-bit double-quantizes these
  down to ~0.23 bits/element overhead; we store fp16 and report ~0.5
  bits/element honestly instead.
- `residual_fp16_bytes` — the un-flushed fp16 tail (a snapshot, not
  cumulative), reported separately so compression ratios aren't inflated.

Effective rate at defaults (D=128, r=64, 2-bit): ~2.5 bits/element including
all metadata — `assigned_avg_bits` reports the realized value.

## Adaptation notes

**Fidelity to the paper:** faithful to the core mechanism — NSN's three
steps, the Hadamard rotation, the positive-orthant magnitude codebook with a
separate sign mask (2-bit) and signed codebook (1-bit), online per-chunk
statistics, and the chunk-flush residual buffer all match the paper's design.

**What we do NOT implement:**
- **Pre-RoPE key handling.** The paper applies NSN to keys *before* RoPE and
  defers RoPE onto the stored mean inside a custom attention kernel. Our
  wrappers receive **post-RoPE** keys from `update_and_fetch`, so NSN +
  Hadamard run post-RoPE. This weakens the Gaussianization slightly (RoPE
  mixes channel pairs position-dependently) and is the central simplification
  of this adaptation.
- **Value-projection Hadamard fusion.** The paper folds the value-side
  Hadamard into the projection weights (model surgery). We apply it
  explicitly to cached values — extra FWHT compute, honest and auditable.
- **Gradient fine-tuning of the codebook.** The paper k-means-initializes on
  `randn` samples, then fine-tunes with gradient descent on a cosine
  objective. We keep the property that matters — model/data independence —
  via deterministic seeded **spherical k-means** on synthetic standard-normal
  samples, and skip the gradient fine-tune. Expect a slightly worse codebook
  than the paper's.
- **4-bit double quantization of metadata.** `s1`/`s2`/`o` are stored fp16
  and counted; the paper's ~0.23-bit overhead becomes ~0.5 bits here.
- **Fused CUDA/Triton kernels.** MLX ops only; the throughput story on Apple
  Silicon is *memory*, not speed, exactly as with KIVI.

**Known limitations:**
- Post-RoPE keys mean the paper's key-side quality numbers don't transfer;
  only the committed offline-synthetic numbers below are claimed.
- No model-level (perplexity/throughput) benchmark has been run yet.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_nsnquant.py` (16 tests) and
`veloxquant_mlx/tests/cache/test_nsnquant_cache.py` (19 tests):

- NSN round-trip is exact (to fp16 metadata precision) without VQ
- Post-NSN tokens have norm `sqrt(d)` and ~zero channel mean
- Hadamard forward/inverse round-trips at head_dim 64 and 128
- Codebook is deterministic per seed; magnitude variant is positive-orthant
- 2-bit / 1-bit round-trip cosine floors on Gaussian input; 2-bit > 1-bit
- **Mechanism validation:** on channel-biased input, the full NSN pipeline
  beats the identical Hadamard+VQ without NSN by a pinned margin
- Prefill vs token-by-token decode yield an identical quantized state
- Chunk *i*'s stored bytes never change after later pushes
- Byte accounting matches the closed form; ratio beats fp16 by >4x at long
  context
- Build-time validation (bits, divisibility, Hadamard compatibility) and the
  `nsn_max_ctx` guard raise with clear messages
- `for_model` wires every attention layer and leaves non-attention layers on
  the fallback cache

The offline harness in `benchmark_scripts/benchmark_nsn.py` (results in
`figures/nsnquant/results.json`) sweeps sequence length,
channel-bias strength, and bit-width against a no-NSN ablation and a KIVI
2-bit baseline at a matched residual window:

- **Ablation (the mechanism's whole claim):** NSN gains **+0.038 cosine at
  2-bit** and **+0.110 at 1-bit** over the same VQ without NSN when the
  synthetic channel bias is strong — and the gain honestly **collapses to
  ~+0.001–0.002 when the input is already centered** (NSN only helps when
  there is a bias to remove).
- **Reconstruction:** 0.96–0.98 mean cosine at 2-bit (~2.5 effective
  bits/element incl. metadata), 0.84–0.94 at 1-bit (~1.5 effective
  bits/element), across all bias levels tested.
- **vs KIVI-2bit** on the same synthetic inputs and residual window: NSNQuant
  2-bit reconstructs at higher cosine on every row of the sweep (KIVI:
  0.66–0.88).

**No model-level benchmark has been run.** These are offline-synthetic,
reconstruction-quality and byte-accounting numbers — not perplexity or
throughput on a real model.

## When to use it

NSNQuant is the repo's most aggressive *calibration-free* quantizer family
entry: pick it when you want VQ-level compression (1–2 bits/element payload)
without any per-model or per-dataset fitting, and when robustness to
distribution shift matters more than squeezing the last percent of
reconstruction quality out of a data-fit codebook. Compared to
[KIVI](../algorithms/kivi) (the other residual-window wrapper), it trades
scalar min/max simplicity for distribution-matched VQ; compared to the
geometric-VQ family (RaBitQ / VecInfer), it actively reshapes the input
rather than hoping the geometry fits.

| Method | Code | Calibration | Payload bits | Residual window |
|--------|------|-------------|--------------|-----------------|
| KIVI | scalar min/max groups | none | 2–4 | yes |
| RaBitQ / VecInfer | geometric (signs/binary) | none | 1–2 | no |
| **NSNQuant** | universal Gaussian codebook | none (by construction) | 1–2 | yes (chunk-flush) |
| [SKVQ](../algorithms/skvq) | scalar min/max + channel reorder + clip | none (first-chunk stats) | 2–4 | yes (chunk-flush, same idiom) |

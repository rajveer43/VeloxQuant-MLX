<div align="center">

<!-- Replace with your generated cover image -->
<img src="assets/veloxquant.png" alt="VeloxQuant-MLX" width="860" />

<h1>VeloxQuant-MLX</h1>

<p>
  <strong>Fast KV Cache Quantization for Apple Silicon</strong><br/>
  TurboQuant · RVQ · VecInfer · RateQuant · PolarQuant · QJL · SpectralQuant · CommVQ · RaBitQ — in MLX
</p>

<p>
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/pypi/v/VeloxQuant-MLX?style=flat-square&logo=pypi&logoColor=white&color=0078d4" alt="PyPI"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-0078d4?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon%20M1+-black?style=flat-square&logo=apple&logoColor=white" alt="Platform"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="License"/></a>
  <img src="https://img.shields.io/badge/tests-1484%20passing-22c55e?style=flat-square" alt="Tests"/>
  <a href="https://doi.org/10.5281/zenodo.20647294"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20647294-1f6feb?style=flat-square" alt="DOI"/></a>
</p>

<p>
  <a href="https://veloxquant-mlx.netlify.app/"><img src="https://img.shields.io/badge/landing%20page-veloxquant--mlx.netlify.app-7c3aed?style=flat-square" alt="Landing"/></a>
  <a href="https://veloxquant-mlx.netlify.app/playground.html"><img src="https://img.shields.io/badge/playground-try%20it%20live-00d4ff?style=flat-square" alt="Playground"/></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.42.1-64748b?style=flat-square" alt="Changelog"/></a>
  <a href="blogs/metal-kernels.md"><img src="https://img.shields.io/badge/blog-Metal%20kernels%20v1-f97316?style=flat-square" alt="Blog"/></a>
  <a href="blogs/turboquant-metal-kernels.md"><img src="https://img.shields.io/badge/blog-TurboQuant%20Metal%20kernels-f97316?style=flat-square" alt="Blog v2"/></a>
  <a href="https://ko-fi.com/rajveer43"><img src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?style=flat-square&logo=ko-fi&logoColor=white" alt="Ko-fi"/></a>
  <a href="https://buymeachai.in/rajveer43"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Chai-support-fbb034?style=flat-square&logo=buymeacoffee&logoColor=black" alt="Buy Me a Chai"/></a>
</p>

</div>

---

**VeloxQuant-MLX** compresses the KV cache of any `mlx_lm` model on Apple Silicon — up to **16× smaller** with near-lossless quality, in three lines of code. It ships **41 research-adapted compression methods**, from zero-calibration 1-bit quantizers to token-eviction caches to cross-layer merging, plus hand-written Metal kernels that make the hottest path **up to 14.7× faster**.

**Why VeloxQuant-MLX:**
- 41 methods behind one identical 3-line API — swap `method="..."` and go
- Metal-accelerated hot paths: 6.9–14.7× faster quantize, 98% less peak memory at the OOM-trigger shape
- Every "-adapted" method documents its honest deviation from the source paper — no silent approximations
- Validated end-to-end on 12 production models: Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon
- Vision-language models too: `patch_vlm_kv_cache` wires the same caches into [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) single-prompt generation (Qwen2-VL, LLaVA, …) — [docs](https://veloxquant-mlx.netlify.app/docs/guides/mlx-lm-integration)

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="Explain relativity simply.", max_tokens=200)
```

---

## Numbers that matter

| Metric | Value | Notes |
|---|---|---|
| Max key cache compression | **16×** | VecInfer-1bit, head_dim=128 |
| Metal kernel speedup | **13×** | `quantize_vq` at S=2048 (range 6.9–14.7× over S=128–8192) |
| Peak memory reduction | **98%** | 729 MB → 12 MB, Falcon3-7B shape |
| RVQ-1bit compression | **7.5×** | Near-zero throughput cost |
| FP16 throughput retained | **100%** | Qwen2.5-7B at 16× compression |
| SpectralQuant compression | **5.33×** | per-model measured (Qwen2.5-0.5B / Gemma-4-4B), same bit-width |
| SpectralQuant cosine sim | **+3pp** | over TurboQuant on Qwen2.5-0.5B |
| **RaBitQ full KV compression** | **6×** | 1-bit keys + MSE-b4 values, Falcon3-7B |
| **RaBitQ fused attend speedup** | **1.78×** | vs dequantize+SDPA at S_kv=8192, D=128 — single-dispatch 1-bit-key/4-bit-value attention, nibble-packed values |
| **RaBitQ fused encode speedup** | **6×** | vs numpy round-trip at N=32768, D=128 (2.9× vs pure MLX ops) |
| **RaBitQ context at 8 GB** | **~103k tokens** (est.) | KV-only linear extrapolation from measured memory rows; vs ~17k fp16 — 6× more context |
| **CommVQ key compression** | **64×** | RoPE-commutative VQ, D=128, n_cb=4 |
| **KIVI-2bit key compression** | **5.8×** | per-channel keys / per-token values; measured on Llama-3.2-3B, Qwen2.5-7B, Mistral-7B |
| **KIVI-2bit full-KV compression** | **~4×** | incl. fp16 residual window (32 tokens); 100–106% of fp16 throughput |
| Production models validated | **12** | Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon |

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Method library](#method-library) — all 41 methods at a glance
4. [Metal kernels](#metal-kernels--new-in-051)
5. [Benchmark results](#benchmark-results)
6. [What's inside](#whats-inside)
7. [Architecture](#architecture)
8. [CLI](#cli)
9. [Development](#development)
10. [Documentation & blog posts](#documentation--blog-posts)
11. [References](#references)
12. [Support](#support)

---

## Installation

```bash
pip install VeloxQuant-MLX
```

**Requirements:** Apple Silicon M1+, Python ≥ 3.11, MLX ≥ 0.18, NumPy ≥ 1.26.

Full install guide (source install, conda/miniforge, Metal troubleshooting,
verifying the install): [installation guide](https://veloxquant-mlx.netlify.app/docs/getting-started/installation).

---

## Quickstart

### RVQ 1-bit — 7.5× compression, no calibration (recommended default)

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Explain the theory of relativity in simple terms.",
    max_tokens=200,
)
```

More examples, walked through step by step:
- [5-minute quickstart](https://veloxquant-mlx.netlify.app/docs/getting-started/quickstart) — same example above, plus **VecInfer** (16×, Metal-accelerated) as a "stronger algorithm" follow-on
- [Mixed-precision guide](https://veloxquant-mlx.netlify.app/docs/guides/mixed-precision) — **RateQuant** automatic per-layer bit allocation via reverse-waterfilling
- [mlx_lm integration guide](https://veloxquant-mlx.netlify.app/docs/guides/mlx-lm-integration) — wiring compressed caches into any model

---

## Method library

All 41 methods share the same 3-line integration (`method="<id>"` in `KVCacheConfig`).
Each links to its full page — mechanism, config, evidence, and honest limitations — on
the [documentation site](https://veloxquant-mlx.netlify.app/docs/algorithms/overview).

**Quick decision:**
- No calibration, best default → **`turboquant_rvq` b=1** (7.5×, 0.92 cosine)
- Max compression, Qwen2.5/Gemma → **`vecinfer` 1-bit** (16×, Metal-accelerated)
- Best quality at moderate compression → **`spectral` b=3** (5.33×, ~5s calibration)
- Heterogeneous layers (sensitivity ratio >2×) → **RateQuant** on top of RVQ
- Max context length, fixed RAM → **`rabitq`** keys + MSE-b4 values (6× full KV)
- RoPE-compatible exact VQ → **`comm_vq`** (ICML 2025, 64× key compression)

### Quantization — compress every token

| Method | `method=` | What it does | Compression | New in |
|---|---|---|---|---|
| [TurboQuant RVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rvq) | `turboquant_rvq` | Residual VQ, zero calibration — **the default** | 7.5× @ 1-bit | — |
| [VecInfer](https://veloxquant-mlx.netlify.app/docs/algorithms/vecinfer) | `vecinfer` | Dual-transform product VQ, Metal-accelerated | 16× | 0.4.0 |
| [SpectralQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/spectral) | `spectral` | Rotate keys into eigenbasis — best quality-per-bit | 5.33× | 0.6.0 |
| [RateQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/ratequant) | *(allocator)* | Per-layer mixed precision via reverse-waterfilling | 5.2× @ 1.5 avg bit | — |
| [RaBitQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rabitq) | `rabitq` | 1-bit keys + MSE-b4 values | **6× full KV** | 0.7.0 |
| [QJL](https://veloxquant-mlx.netlify.app/docs/algorithms/qjl) | `qjl` | 1-bit JL sketch, simplest/fastest to set up | ~16× | — |
| [PolarQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/polarquant) | `polar` | Polar-coordinate quant for geometric key distributions | varies | — |
| [CommVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/commvq) | `comm_vq` | RoPE-commutative VQ, exact inner product (ICML 2025) | 64× keys | — |
| [KIVI](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi) | `kivi` | Tuning-free asymmetric 2-bit baseline | 5.8× | 0.8.0 |
| [KIVI-Sink](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi-sink) | `kivi_sink` | Sink-protected low-bit quantization | ~5.8× | 0.9.0 |
| [SKVQ-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/skvq) | `skvq` | Channel reordering + clipped dynamic quant behind a sliding fp16 window + sink filter (COLM 2024) | varies | 0.30.0 |
| [SVDq](https://veloxquant-mlx.netlify.app/docs/algorithms/svdq) | `svdq` | Sub-2-bit keys (~1.25 bit) via prefill SVD | ~10× | 0.10.0 |
| [Kitty](https://veloxquant-mlx.netlify.app/docs/algorithms/kitty) | `kitty` | Adaptive channel precision, zero calibration | varies | 0.11.0 |
| [KVQuant-NUQ](https://veloxquant-mlx.netlify.app/docs/algorithms/kvquant) | `kvquant` | Non-uniform datatype + outlier isolation | varies | 0.14.0 |
| [NSNQuant-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/nsnquant) | `nsnquant` | Calibration-free universal-codebook VQ — fixed Gaussian codebook (NeurIPS 2025) | 1–2 bit/elem | 0.28.0 |
| [ZipCache-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/zipcache) | `zipcache` | Per-token mixed bit-width by key-norm saliency | varies | 0.18.0 |
| [GEAR](https://veloxquant-mlx.netlify.app/docs/algorithms/gear) | `gear` | Error-feedback: low-rank + sparse residual correction | varies | 0.17.0 |
| [CacheGen](https://veloxquant-mlx.netlify.app/docs/algorithms/cachegen) | `cachegen` | Entropy-coded cache — storage win on correlated KV | varies | 0.16.0 |
| [AMC-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/amc) | `amc` | Saliency-driven tiered rank + precision — one L1-norm score drives both rank and bit-width per token, never evicts (**no verified venue** — second exception; hardware/RTL half of paper out of scope) | varies | 0.38.0 |
| [A2ATS-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/a2ats) | `a2ats` | Windowed RoPE + query-aware retrieval VQ — exact RoPE within a trailing window, shared approximate rotation outside it; query-aware codebook assignment for a retrieval-fraction subset (ACL 2025 Findings) | ~4× | 0.39.0 |

### Low-rank & cross-layer — compress across dimensions or depth

| Method | `method=` | What it does | Compression | New in |
|---|---|---|---|---|
| [PALU](https://veloxquant-mlx.netlify.app/docs/algorithms/palu) | `palu` | True low-rank latent storage of both K and V | varies | 0.15.0 |
| [XQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/xquant) | `xquant` | Cross-layer code reuse — adjacent layers share codes | varies | 0.12.0 |
| [MiniCache](https://veloxquant-mlx.netlify.app/docs/algorithms/minicache) | `minicache` | Cross-layer SLERP merge — deep layer pairs cost one | ~2× on deep layers | 0.16.0 |
| [xKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/xkv) | `xkv` | Cross-layer shared-subspace SVD — one basis jointly fit across a layer group | varies | 0.27.0 |
| [AdaKV-proxy](https://veloxquant-mlx.netlify.app/docs/algorithms/adakv) | `adakv` | Per-head adaptive bit budget, layered on KIVI | varies | 0.13.0 |
| [KVTC-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/kvtc) | `kvtc` | Local PCA + DP-optimal per-component bit allocation + entropy coding (ICLR 2026) — beats fixed-split mixed-precision at matched byte budget on skewed variance | varies | 0.35.0 |

### Token eviction & merging — drop or merge low-value tokens

| Method | `method=` | What it does | New in |
|---|---|---|---|
| [SnapKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/snapkv) | `snapkv` | Prefill observation-window eviction, once at prefill end | 0.19.0 |
| [StreamingLLM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/streaming_llm) | `streaming_llm` | Sink + recency window, constant memory | 0.20.0 |
| [H2O-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/h2o) | `h2o` | Cumulative attention-mass heavy-hitter eviction | 0.21.0 |
| [TOVA-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/tova) | `tova` | Memoryless current-step attention-weight eviction | 0.22.0 |
| [PyramidKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/pyramidkv) | `pyramidkv` | H2O eviction with a per-layer pyramid budget | 0.23.0 |
| [SqueezeAttention-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/squeeze) | `squeeze` | 2D layer×token data-driven budget eviction | 0.24.0 |
| [ChunkKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/chunkkv) | `chunkkv` | Chunk-level eviction (`chunk_size=1` == H2O) | 0.25.0 |
| [CaM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/cam) | `cam` | Cache **merging** — merge evicted tokens, don't drop (`cam_merge=drop` == H2O) | 0.26.0 |
| [L2Norm-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/knorm) | `knorm` | Intrinsic key-norm eviction — low norm ⇒ important (EMNLP 2024) | 0.29.0 |
| [Q-Filters-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/qfilters) | `qfilters` | Query-agnostic projection eviction — frozen per-head key-SVD direction | 0.31.0 |
| [Keyformer-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/keyformer) | `keyformer` | Gumbel-regularized heavy-hitter eviction (MLSys 2024); `keyformer_tau=0` == H2O | 0.32.0 |
| [MorphKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/morphkv) | `morphkv` | Recent-window correlation retention (ICML 2025); `morphkv_window=1` == TOVA | 0.33.0 |
| [KVzip-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/kvzip) | `kvzip` | Context-reconstruction reliance eviction (NeurIPS 2025); `kvzip_probe=latest` == TOVA | 0.34.0 |
| [CurDKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/curdkv) | `curdkv` | Value-aware leverage-score eviction via approximated CUR decomposition (NeurIPS 2025) — evicts key-similar but value-irrelevant tokens that key-only eviction (H2O) cannot distinguish | 0.36.0 |
| [NestedKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/nestedkv) | `nestedkv` | Multi-scale ensembled prefill eviction — stable + episodic + current key anomaly, combined by a head-adaptive blend and surprise-gated route (**no verified venue** — one-time exception) | 0.37.0 |

> Every "-adapted" method is an honest adaptation, not a faithful port — the cache
> wrapper sees per-layer K/V but not the model's true query/attention maps, so
> attention-based signals use a key-as-query proxy. Each method's docs page states its
> specific limitations plainly.

---

## Metal kernels — new in 0.5.1

The VecInfer `quantize_vq` hot path is now a 30-line Metal Shading Language shader, JIT-compiled by `mx.fast.metal_kernel` on first use. Same Python API — no changes required.

<div align="center">
  <img src="figures/metal/summary.png" alt="Metal kernel benchmark — quantize latency, speedup, and peak memory" width="820"/>
  <br/><sub>Benchmarked on Apple Silicon GPU. Left: quantize latency. Center: speedup factor. Right: peak memory.</sub>
</div>

<br/>

| Metric | Pure MLX | Metal kernel | Delta |
|---|---|---|---|
| Quantize latency (S=8192) | 228 ms | **15.6 ms** | **14.7×** faster |
| Peak memory (Falcon3-7B shape) | 729 MB | **12 MB** | **98%** reduction |
| API change required | — | None | `use_metal_kernels=None` auto-detects |

**Why the memory win:** the `[N, n_centroids, sub_dim]` diff tensor is never materialised — the argmin accumulator lives entirely in thread-local registers.

**Honest caveat:** the kernel pays a ~50–200 µs launch overhead per call. On tiny models (SmolLM2-135M, ~60 launches/token) that overhead can exceed the savings. Built for the regime that needs it: 7B+ models at realistic context lengths.

Full kernel source and how it was built: [blogs/metal-kernels.md](blogs/metal-kernels.md). Usage, fallback behaviour, and debugging: [docs — Metal GPU kernels](https://veloxquant-mlx.netlify.app/docs/guides/metal-kernels).

### Fused RaBitQ asymmetric pipeline

Two newer kernels form a fully GPU-resident pipeline for an **asymmetric-precision cache** — 1-bit packed keys scored via XOR+popcount, 4-bit codebook values — a K/V format combination fused attention kernels normally can't express:

- [`rabitq_encode`](veloxquant_mlx/metal/_rabitq_encode.py) — rotate + binarize + bit-pack + magnitude in one dispatch. Sign packing uses `simd_ballot`: each SIMD-group's 32 sign predicates land in a single vote mask, which is exactly 4 bytes of packed output.
- [`rabitq_fused_attend`](veloxquant_mlx/metal/_rabitq_attend.py) — scores packed keys, runs an online softmax split across 8 SIMD-groups (flash-decoding style), and accumulates codebook values — one dispatch, no dequantized K or V ever materialized.
- [`rabitq_pack_values`](veloxquant_mlx/metal/_rabitq_values.py) — two 4-bit value indices per byte; the attend kernel reads nibbles directly (auto-detected from the shape), halving value-cache memory and bandwidth with bit-identical outputs.

Measured (Apple M4, D=128 — `scripts/metal_rabitq_attend_bench.py`, `scripts/metal_rabitq_encode_bench.py`):

| Kernel | Config | Baseline | Fused | Speedup |
|---|---|---|---|---|
| attend, packed V | S_kv=8192, B=1 H=8 S_q=1 | 2.492 ms | **1.404 ms** | **1.78×** |
| attend, packed V | S_kv=2048 | 0.681 ms | **0.481 ms** | **1.42×** |
| attend, packed V | S_kv=512 | 0.309 ms | **0.281 ms** | **1.10×** |
| encode | N=32768 | 4.511 ms (numpy) | **0.752 ms** | **6.0×** |

**Honest caveat:** with unpacked (byte-per-index) values the fused attend *loses* at short contexts (0.65× at S_kv=512) — nibble-packing halves value bandwidth and flips that to a small win. Parity vs numpy references is covered by 63 dedicated tests ([`test_rabitq_attend.py`](veloxquant_mlx/tests/metal/test_rabitq_attend.py), [`test_rabitq_encode.py`](veloxquant_mlx/tests/metal/test_rabitq_encode.py), [`test_rabitq_values.py`](veloxquant_mlx/tests/metal/test_rabitq_values.py)), including an end-to-end encode→attend test and bit-exact packed-vs-unpacked equality.

For large `S_q` — the multi-turn VLM case, where a new turn attends over a long compressed image-token history — [`rabitq_prefill_attend`](veloxquant_mlx/metal/_rabitq_prefill.py) is the matmul-shaped companion: both `Q·K̂ᵀ` and `W·V̂` run on 8×8 `simdgroup_matrix` tiles, with keys sign-decoded and values nibble-decoded inside the tile loop. It scores exact dots rather than the Hamming estimate, and is cross-attention only (no causal mask).

### Fused group-affine (KIVI-style) attention — new in 0.42.0

[`scalar_fused_decode_attend`](veloxquant_mlx/metal/_scalar_attend.py) is the scalar/group-quant analogue of the codebook fused attends above — it serves the **KIVI / SKVQ / Kitty / group-quant family**, where K/V are `uint8` codes plus a per-group `(scale, zero)` pair instead of a codebook.

The pure-MLX path reconstructs `code * scale + zero` into a full fp16 tensor, then calls `scaled_dot_product_attention` — a `dequantize → DRAM → SDPA` round-trip paid every decode step. This kernel reconstructs `x_hat` in-register inside a FlashAttention-style online softmax, so no dequantized `K_hat`/`V_hat` ever reaches DRAM. The win compounds with context: the fp16 `K_hat` grows linearly with `S_kv` while the packed codes stay `16/b` times smaller.

Measured (Apple M4 10-core GPU, B=1 H=32 D=128 b=2 g=32 S_q=1) vs. dequantize → MLX SDPA:

| Config | Speedup |
|---|---|
| S_kv=512 | **6.4×** |
| S_kv=65536 | **12.2×** |

The kv axis is split flash-decoding style across `nsg` SIMD-groups so single-query decode shapes still fill the GPU (`nsg=8` tuned on M4), and one compiled kernel serves any `(S_kv, D, g)`. Parity max abs error is `1.2e-4` — the fp32 softmax accumulation makes it *more* accurate than the fp16 baseline it replaces ([`test_scalar_attend.py`](veloxquant_mlx/tests/metal/test_scalar_attend.py)).

---

## Benchmark results

### 10-model comparative study — VecInfer vs RVQ (v0.5.0)

<div align="center">
  <img src="figures/vecinfer/_summary/cross_model_comparison.png" alt="Cross-model comparison — VecInfer vs RVQ-1bit across 10 models" width="820"/>
  <br/><sub>End-to-end <code>mlx_lm.generate</code> · 200-token prompt · 120-token generation · Apple M-series unified memory</sub>
</div>

<br/>

**Compression ratio:**

| Model | RVQ-1bit | VecInfer-1bit |
|---|---|---|
| SmolLM2-135M | 7.1× | **16×** |
| Llama-3.2-1B | 7.1× | **16×** |
| Llama-3.2-3B | 7.5× | **16×** |
| Llama-3.1-8B | 7.5× | **16×** |
| Mistral-7B | 7.5× | **16×** |
| Qwen2.5-7B | 7.5× | **16×** |
| Qwen3-8B | 7.5× | **16×** |
| Phi-4 | 7.5× | **16×** |
| Falcon3-7B | 7.8× | **16×** |
| gemma-3-4b | 7.8× | **16×** |

**Throughput (tok/s):**

| Model | fp16 | RVQ-1bit | VecInfer-1bit |
|---|---|---|---|
| SmolLM2-135M | 250.4 | 188.5 | 175.8 |
| Llama-3.2-1B | 105.4 | **104.3** | 91.2 |
| Llama-3.2-3B | 47.6 | **46.2** | 40.2 |
| Llama-3.1-8B | 20.5 | **20.6** | 19.6 |
| Mistral-7B | 23.6 | **22.8** | 9.8 |
| Qwen2.5-7B | 21.0 | 20.7 | **21.5** ⬆ exceeds fp16 at 16× |
| Qwen3-8B | 20.3 | **19.6** | 2.4 |
| Phi-4 | 10.4 | 8.1 | 4.0 |
| Falcon3-7B | 17.3 | **21.7** | 17.0 |
| gemma-3-4b | 26.0 | 24.2 | **22.6** |

> **RVQ-1bit** is the safe default — within 5% of fp16 on most 7–8B models with zero calibration. **VecInfer-1bit** wins on memory (always 16×) and throughput on strong-GQA models (Qwen2.5, Gemma).

Historical benchmark snapshots (throughput optimisation journey, RateQuant V2, 8-model RVQ sweep) and full methodology: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

---

## What's inside

| Module | Purpose |
|---|---|
| [`veloxquant_mlx/quantizers/turboquant_rvq`](veloxquant_mlx/quantizers/turboquant_rvq.py) | Two-pass scalar RVQ — Gaussian + Laplacian codebooks, b=1/2/3+ |
| [`veloxquant_mlx/cache/vecinfer_cache`](veloxquant_mlx/cache/vecinfer_cache.py) | `VecInferKVCache` — smooth + Hadamard + product VQ |
| [`veloxquant_mlx/cache/turboquant_rvq_cache`](veloxquant_mlx/cache/turboquant_rvq_cache.py) | `TurboQuantRVQKVCache` — mlx_lm-compatible wrapper |
| [`veloxquant_mlx/allocators`](veloxquant_mlx/allocators/) | `allocate_bits_ratequant`, `calibrate_layer_sensitivities`, VecInfer calibration |
| [`veloxquant_mlx/metal`](veloxquant_mlx/metal/) | Hand-written Metal MSL kernels, JIT via `mx.fast.metal_kernel` |
| [`veloxquant_mlx/spectral`](veloxquant_mlx/spectral/) | `SpectralQuantizer`, rotation calibration, water-filling bit allocation |

Full module reference and API docs: [docs — API reference](https://veloxquant-mlx.netlify.app/docs/api/core-api).

---

## Architecture

VeloxQuant-MLX pipelines each quantizer as rotate → quantize (± residual) → pack, built via a Builder/Factory/Strategy layering so every method shares the same `KVCacheConfig` → `KVCacheBuilder` → `mlx_lm`-compatible cache path. Ten design patterns are used throughout (Abstract Base Classes, Factory, Chain of Responsibility, Builder, Strategy, Registry + Plugin, Composite, Observer, DAO, and custom data structures like RingBuffer/MaxHeap/BitPackBuffer/VoronoiTree).

Full pipeline diagrams (TurboQuantRVQ, VecInfer) and design-pattern breakdown: [docs — Core concepts](https://veloxquant-mlx.netlify.app/docs/getting-started/concepts).

---

## CLI

```bash
# Precompute rotation matrices, JL matrices, codebooks
python -m veloxquant_mlx precompute \
    --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --seed 42 \
    --output_dir ./artifacts/

# Synthetic benchmark — single config
python -m veloxquant_mlx benchmark \
    --method turboquant_rvq --head_dim 128 --bits 2 --seq_len 1000

# End-to-end model benchmarks
python benchmark_scripts/benchmark_vecinfer.py   # VecInfer 10-model sweep
python benchmark_scripts/run_outlier_ratequant.py # RateQuant mixed-precision

# Which method should I use on my Mac? (new in 0.42.0)
python -m veloxquant_mlx recommend \
    --chip M4 --ram-gb 16 --model-class 7B --goal everyday
```

The recommender is accounting-aware — it reports the key compression ratio *and* tells you when resident RAM savings are unlikely, rather than quoting a ratio that won't show up in RSS:

```
method=turboquant_rvq
knobs={'bit_width_inlier': 1, 'seed': 42}
key_accounting_ratio≈7.5x
resident_savings_likely=False
kv_fp16_mb≈512.0  kv_compressed_mb_est≈68.27
rationale: Zero-calibration default. Key accounting ~7.5x at head_dim=128.
           Default path dequantizes into parent fp16 cache.
warnings:
  - Tight RAM with a mid/large model: consider goal=max_context (rabitq)
    or goal=constant_memory (eviction) for long prompts.
```

Goals: `everyday`, `max_key_accounting`, `max_context`, `best_quality`, `constant_memory`. Add `--json` for machine-readable output, or `--seq-len` / `--n-layers` / `--n-kv-heads` / `--head-dim` to match a specific model. Also available in the browser via the [Compression Lab](https://veloxquant-mlx.netlify.app/playground.html).

Load precomputed artifacts to skip re-computation at runtime:

```python
from veloxquant_mlx.artifacts import NpyArtifactStore

cache = (
    KVCacheBuilder()
    .with_method("turboquant_rvq")
    .with_head_dim(128)
    .with_bit_width(inlier=2)
    .with_artifact_store(NpyArtifactStore("./artifacts/"))
    .build()
)
```

---

## Development

```bash
# Full test suite (includes Metal parity tests)
pytest veloxquant_mlx/tests/ -v

# 2-bit improvement validation — fast synthetic run
python test_2bit_improvements.py

# Generate optimization-journey figure
python scripts/plot_optimization_journey.py
```

Contributions welcome — please open an issue first for anything beyond a small bugfix. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Documentation & blog posts

Full docs, including per-method pages, guides, and API reference: **https://veloxquant-mlx.netlify.app/**

Deep-dive writeups live in [`blogs/`](blogs/) and are also published on the docs site:

| File | Description | Live |
|---|---|---|
| [`blogs/overview.md`](blogs/overview.md) | High-level overview of VeloxQuant-MLX and its goals | [↗](https://veloxquant-mlx.netlify.app/docs/blog/overview) |
| [`blogs/10-model-study.md`](blogs/10-model-study.md) | End-to-end benchmark study across 10 production models | [↗](https://veloxquant-mlx.netlify.app/docs/blog/10-model-study) |
| [`blogs/hands-on.md`](blogs/hands-on.md) | Hands-on tutorial: compressing your first model | [↗](https://veloxquant-mlx.netlify.app/docs/blog/hands-on) |
| [`blogs/kivi.md`](blogs/kivi.md) | Deep dive into the KIVI asymmetric quantization baseline | [↗](https://veloxquant-mlx.netlify.app/docs/blog/kivi) |
| [`blogs/metal-kernels.md`](blogs/metal-kernels.md) | How the Metal compute kernel cuts quantize latency 13× | [↗](https://veloxquant-mlx.netlify.app/docs/blog/metal-kernels) |
| [`blogs/results.md`](blogs/results.md) | Detailed benchmark results and analysis | [↗](https://veloxquant-mlx.netlify.app/docs/blog/results) |
| [`blogs/tensorops-research.md`](blogs/tensorops-research.md) | TensorOps research notes and findings | [↗](https://veloxquant-mlx.netlify.app/docs/blog/tensorops-research) |
| [`blogs/turboquant-metal-kernels.md`](blogs/turboquant-metal-kernels.md) | TurboQuant + Metal kernels: combined writeup | [↗](https://veloxquant-mlx.netlify.app/docs/blog/turboquant-metal-kernels) |

---

## References

41 methods, each adapted from a published paper with documented deviations
(39 from a verified peer-reviewed venue; 2, NestedKV-adapted and AMC-adapted,
from unpublished preprints as one-time, stated exceptions — see
[CITATIONS.md](CITATIONS.md)) — full bibliography (implemented methods,
related work, and survey papers): **[CITATIONS.md](CITATIONS.md)**.

Headline references: [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874), [VecInfer (2024)](https://arxiv.org/abs/2510.06175), [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2402.02855), [CommVQ (ICML 2025)](https://arxiv.org/abs/2506.18879), [KVzip (NeurIPS 2025)](https://arxiv.org/abs/2505.23416), [KVTC (ICLR 2026)](https://arxiv.org/abs/2511.01815), [CurDKV (NeurIPS 2025)](https://arxiv.org/abs/2509.15038), [NestedKV (preprint, arXiv:2605.26678)](https://arxiv.org/abs/2605.26678), [AMC (preprint, arXiv:2607.10109)](https://arxiv.org/abs/2607.10109), [A2ATS (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.644/). Built on [Apple MLX](https://github.com/ml-explore/mlx).

---

## Support

VeloxQuant-MLX is free, MIT-licensed,
and built nights-and-weekends — if it saves your Mac some memory (or you just
want to see the 42nd method land), you can
[**buy me a chai** ☕](https://buymeachai.in/rajveer43) or
[**tip on Ko-fi** 💜](https://ko-fi.com/rajveer43). Stars, issues, and
PRs are equally appreciated.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built for Apple Silicon · Engineered for speed · MIT License</sub>
  <br/>
  <sub>
    <a href="https://veloxquant-mlx.netlify.app/">Landing page</a> ·
    <a href="https://github.com/rajveer43/VeloxQuant-MLX/issues">Issues</a> ·
    <a href="blogs/10-model-study.md">Blog: 10-model study</a> ·
    <a href="blogs/metal-kernels.md">Blog: Metal kernels v1</a> ·
    <a href="blogs/turboquant-metal-kernels.md">Blog: TurboQuant Metal kernels</a>
  </sub>
</div>

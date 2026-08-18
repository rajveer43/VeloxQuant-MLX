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
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/pypi/dm/VeloxQuant-MLX?style=flat-square&logo=pypi&logoColor=white&color=0078d4" alt="PyPI downloads"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-0078d4?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon%20M1+-black?style=flat-square&logo=apple&logoColor=white" alt="Platform"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="License"/></a>
  <a href="https://doi.org/10.5281/zenodo.20647294"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20647294-1f6feb?style=flat-square" alt="DOI"/></a>
</p>

<p>
  <a href="https://github.com/rajveer43/VeloxQuant-MLX/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/rajveer43/VeloxQuant-MLX/release.yml?branch=master&style=flat-square&label=release%20%2B%20full%20test%20suite&logo=github" alt="Release build status"/></a>
  <a href="https://github.com/rajveer43/VeloxQuant-MLX/actions/workflows/non-metal-unit.yml"><img src="https://img.shields.io/github/actions/workflow/status/rajveer43/VeloxQuant-MLX/non-metal-unit.yml?branch=master&style=flat-square&label=unit&logo=github" alt="Non-Metal unit tests"/></a>
  <a href="https://github.com/rajveer43/VeloxQuant-MLX/actions/workflows/lint.yml"><img src="https://img.shields.io/github/actions/workflow/status/rajveer43/VeloxQuant-MLX/lint.yml?branch=master&style=flat-square&label=lint&logo=github" alt="Lint status"/></a>
  <img src="https://img.shields.io/badge/tests-2236%20passing-22c55e?style=flat-square" alt="Tests"/>
  <!-- The tests and changelog badges are rewritten on every release by
       scripts/sync_release_badges.py, which matches the literal
       "badge/tests-<n>%20passing-" and "badge/changelog-<version>-" patterns.
       Keep both in badge form — converting either to a text link silently
       disables that sync. -->
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.50.1-64748b?style=flat-square" alt="Changelog"/></a>
  <a href="SECURITY.md"><img src="https://img.shields.io/badge/security-policy-1f6feb?style=flat-square" alt="Security policy"/></a>
</p>

<!-- Text links rather than a third badge row. The project docs, governance and
     support links are all reachable, but they no longer compete with the
     status badges above for a first-time reader's attention. -->
<p>
  <a href="https://veloxquant-mlx.netlify.app/">Docs</a> ·
  <a href="https://veloxquant-mlx.netlify.app/playground.html">Playground</a> ·
  <a href="#project--governance">Governance</a> ·
  <a href="SECURITY.md">Security</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="#support">Support the project</a>
</p>

</div>

---

**VeloxQuant-MLX** shrinks the KV cache of any `mlx_lm` model on Apple Silicon so you can run longer contexts or bigger models in the same RAM — up to **16× smaller** with near-lossless quality, in three lines of code. Under the hood it's **41 compression methods** (each adapted from a published paper), from zero-calibration 1-bit quantizers to token-eviction caches to cross-layer merging, plus hand-written Metal kernels that speed up the hottest path by **up to 14.7×**.

If you're running `mlx_lm` locally and hitting a context-length or memory wall on
Apple Silicon, this swaps in a compressed cache with no model changes. Actively developed,
with a full test suite gating every release (see badges above).

> **Accounting vs. resident memory:** the compression ratios above are the theoretical
> byte count (bit-width accounting) — not what Activity Monitor will show you. Most
> quantization methods still store full fp16 tensors under the hood on the default
> `mlx_lm` serving path today, so process RSS won't drop by the same factor yet — see
> [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27) for the packed-storage
> roadmap. Methods that *do* shrink resident memory today (eviction/merging, which
> actually drop tokens) are marked 🔻RSS in the [method library](#method-library) below;
> the rest reduce accounting-only storage while staying fp16-sized in memory.

**Why VeloxQuant-MLX:**
- Try any of 41 compression strategies without rewriting your code — they all share one 3-line API, so switching is just changing `method="..."`
- The hot path runs on hand-written Metal kernels: 6.9–14.7× faster quantize, 98% less peak memory at the shape that used to OOM
- If a method had to cut a corner to work as a drop-in cache instead of a full model rewrite, we say so on its docs page — no silent approximations
- Battle-tested on 12 production models: Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon
- Works with vision-language models too — `patch_vlm_kv_cache` wires the same caches into [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) single-prompt generation (Qwen2-VL, LLaVA, …) — [docs](https://veloxquant-mlx.netlify.app/docs/guides/mlx-lm-integration)

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

> Compression ratios below are bit-width accounting, not measured RSS — see the
> accounting-vs-resident note above and [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27).
> "Peak memory reduction" and "context at 8 GB" rows are Metal-kernel working-set/estimate
> figures, not steady-state cache RSS under default `mlx_lm` serving.

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
3. [Method library](#method-library) — all 41 methods, grouped by family
4. [Metal kernels](#metal-kernels)
5. [Benchmark results](#benchmark-results)
6. [What's inside](#whats-inside)
7. [Architecture](#architecture)
8. [CLI](#cli)
9. [Development](#development)
10. [Project & governance](#project--governance) — security policy, maintainership, release process
11. [Documentation & blog posts](#documentation--blog-posts)
12. [References](#references)
13. [Support](#support)

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

### No Python? Start here — the control panel

```bash
veloxquant panel
```

Opens a local web UI at `http://127.0.0.1:7860`: pick a model and compression
method, press **Start Server**, and point any OpenAI-compatible client
(Claude Code, Cursor, the OpenAI SDK) at the URL it gives you.

Under the hood it runs `veloxquant serve`, which you can also use directly:

```bash
veloxquant methods --servable-only        # what can be served
veloxquant serve --model mlx-community/Llama-3.2-1B-Instruct-4bit \
                 --method turboquant_rvq --bits 2 --port 8000
```

> Compression is currently **accounting-only** — byte counters measure
> compression fidelity, not runtime memory saved. See
> [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27).

Full guide: [docs/control-panel.md](docs/control-panel.md)

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

All 41 methods drop in the same way — just set `method="<id>"` in `KVCacheConfig`.
For the full comparison table, a decision tree, and per-model recommendations —
mechanism, config, evidence, and honest limitations for every method — see the
[algorithm overview](https://veloxquant-mlx.netlify.app/docs/algorithms/overview).

**Quick decision:**
- No calibration, best default → **`turboquant_rvq` b=1** (7.5×, 0.92 cosine)
- Max compression, Qwen2.5/Gemma → **`vecinfer` 1-bit** (16×, Metal-accelerated)
- Best quality at moderate compression → **`spectral` b=3** (5.33×, ~5s calibration)
- Heterogeneous layers (sensitivity ratio >2×) → **RateQuant** on top of RVQ
- Max context length, fixed RAM → **`rabitq`** keys + MSE-b4 values (6× full KV)
- RoPE-compatible exact VQ → **`comm_vq`** (ICML 2025, 64× key compression)
- Hard cap on token count, fixed RAM → **`h2o`** or **`snapkv`** (eviction, reduces resident memory)

The 41 methods span three families — each links to its full docs page:

- **Quantization** (21 methods) — compress every token. Default: [TurboQuant RVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rvq) `turboquant_rvq`. Also: [VecInfer](https://veloxquant-mlx.netlify.app/docs/algorithms/vecinfer), [SpectralQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/spectral), [RateQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/ratequant), [RaBitQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rabitq), [QJL](https://veloxquant-mlx.netlify.app/docs/algorithms/qjl), [PolarQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/polarquant), [CommVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/commvq), [KIVI](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi) / [KIVI-Sink](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi-sink), [SKVQ-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/skvq), [SVDq](https://veloxquant-mlx.netlify.app/docs/algorithms/svdq), [Kitty](https://veloxquant-mlx.netlify.app/docs/algorithms/kitty), [KVQuant-NUQ](https://veloxquant-mlx.netlify.app/docs/algorithms/kvquant), [NSNQuant-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/nsnquant), [ZipCache-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/zipcache), [GEAR](https://veloxquant-mlx.netlify.app/docs/algorithms/gear), [CacheGen](https://veloxquant-mlx.netlify.app/docs/algorithms/cachegen), [AMC-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/amc), [A2ATS-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/a2ats).
- **Low-rank & cross-layer** (6 methods) — compress across dimensions or depth. [PALU](https://veloxquant-mlx.netlify.app/docs/algorithms/palu), [XQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/xquant), [MiniCache](https://veloxquant-mlx.netlify.app/docs/algorithms/minicache), [xKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/xkv), [AdaKV-proxy](https://veloxquant-mlx.netlify.app/docs/algorithms/adakv), [KVTC-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/kvtc).
- **Token eviction & merging** (14 methods) — drop or merge low-value tokens; these reduce **resident** memory today (🔻RSS), not just accounting. [SnapKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/snapkv), [StreamingLLM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/streaming_llm), [H2O-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/h2o), [TOVA-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/tova), [PyramidKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/pyramidkv), [SqueezeAttention-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/squeeze), [ChunkKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/chunkkv), [CaM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/cam), [L2Norm-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/knorm), [Q-Filters-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/qfilters), [Keyformer-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/keyformer), [MorphKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/morphkv), [KVzip-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/kvzip), [CurDKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/curdkv), [NestedKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/nestedkv).

> Category legend used on the docs site: 🧮 won't shrink your Mac's memory usage today, only
> the theoretical bit count (tagged `accounting_only` — still stores full fp16 under the hood;
> see [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)); 🔻RSS actually reduces
> memory you can see today (tagged `eviction`/`true_latent` — drops tokens or stores a
> genuinely smaller tensor); ⚙️ needs a bit more wiring to use (tagged `standalone` — doesn't
> subclass `mlx_lm`'s `KVCache`, so it isn't plugged into the default serving path the same
> way). Also worth knowing: every "-adapted" method is an honest adaptation, not a 1:1 port —
> the cache only sees per-layer K/V, never the model's real query/attention maps, so
> attention-based signals fall back to a key-as-query proxy. Full per-method compression
> ratios, categories, and release versions are on the
> [algorithm overview](https://veloxquant-mlx.netlify.app/docs/algorithms/overview).

---

## Metal kernels

VecInfer's `quantize_vq` — the slowest step in the pipeline — now runs on the GPU instead of in Python: a 30-line Metal shader, JIT-compiled by `mx.fast.metal_kernel` the first time you call it. Same Python API, no code changes required to benefit.

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

**Why the memory win:** nothing extra ever gets written out to memory — the `[N, n_centroids, sub_dim]` diff tensor that the pure-MLX version has to materialize is skipped entirely, since the argmin accumulator lives in thread-local GPU registers instead. That's the whole 98% peak-memory drop.

> **Caveat:** the kernel pays a ~50–200 µs launch overhead per call. On tiny models (SmolLM2-135M, ~60 launches/token) that overhead can exceed the savings. Built for the regime that needs it: 7B+ models at realistic context lengths.

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

> **Caveat:** with unpacked (byte-per-index) values the fused attend *loses* at short contexts (0.65× at S_kv=512) — nibble-packing halves value bandwidth and flips that to a small win.

Parity vs numpy references is covered by 63 dedicated tests ([`test_rabitq_attend.py`](veloxquant_mlx/tests/metal/test_rabitq_attend.py), [`test_rabitq_encode.py`](veloxquant_mlx/tests/metal/test_rabitq_encode.py), [`test_rabitq_values.py`](veloxquant_mlx/tests/metal/test_rabitq_values.py)), including an end-to-end encode→attend test and bit-exact packed-vs-unpacked equality.

For large `S_q` — the multi-turn VLM case, where a new turn attends over a long compressed image-token history — [`rabitq_prefill_attend`](veloxquant_mlx/metal/_rabitq_prefill.py) is the matmul-shaped companion: both `Q·K̂ᵀ` and `W·V̂` run on 8×8 `simdgroup_matrix` tiles, with keys sign-decoded and values nibble-decoded inside the tile loop. It scores exact dots rather than the Hamming estimate, and is cross-attention only (no causal mask).

### Fused group-affine (KIVI-style) attention — new in 0.42.0

[`scalar_fused_decode_attend`](veloxquant_mlx/metal/_scalar_attend.py) is the scalar/group-quant analogue of the codebook fused attends above — it serves the **KIVI / SKVQ / Kitty / group-quant family**, where K/V are `uint8` codes plus a per-group `(scale, zero)` pair instead of a codebook.

The pure-MLX path pays a real cost every decode step: it reconstructs `code * scale + zero` into a full fp16 tensor, writes that to memory, then reads it back for `scaled_dot_product_attention` — a `dequantize → DRAM → SDPA` round-trip. This kernel skips the memory round-trip entirely: it reconstructs `x_hat` directly in GPU registers inside a FlashAttention-style online softmax (a numerically stable way to compute softmax over a stream of values without holding them all in memory at once), so no dequantized `K_hat`/`V_hat` ever touches DRAM. The win grows with context length: the fp16 `K_hat` the old path builds grows linearly with `S_kv`, while the packed codes this kernel reads directly stay `16/b` times smaller.

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

Every method runs the same three-step pipeline: rotate the K/V tensors into a friendlier basis, quantize them (optionally with a residual pass for extra precision), then pack the bits. That's why swapping `method="..."` just works — every quantizer plugs into the same `KVCacheConfig` → `KVCacheBuilder` → `mlx_lm`-compatible cache path regardless of what it does internally.

If you're curious how that's wired up: it's built on standard design patterns (Factory, Strategy, Builder, and others) plus a few custom data structures for the lower-level bit-packing work. Full pipeline diagrams (TurboQuantRVQ, VecInfer) and the complete design-pattern breakdown: [docs — Core concepts](https://veloxquant-mlx.netlify.app/docs/getting-started/concepts).

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
rationale: The safe everyday pick: it works out of the box with no setup
           step, and shrinks the key half of the cache by about 7.5x. It
           unpacks each value back to full precision as it is read, so this
           is a size measurement rather than a drop in live memory use.
warnings:
  - RAM is tight for a model this size. For long prompts you will get more
    out of 'Fit the longest conversation' (rabitq), which compresses the
    whole cache, or 'Never grow past a fixed memory limit' (streaming_llm),
    which caps it outright.
  - This method measures smaller but may not free much actual RAM on short
    prompts, because its default path unpacks values back to full precision
    as it reads them. The size figure is real; treat it as a measure of how
    well the data compresses, not as RAM you get back.
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

## Project & governance

These policies already governed the project; this section makes them reachable
from the README rather than only from the file tree.

| | |
|---|---|
| **Security policy** | [SECURITY.md](SECURITY.md) — private disclosure by email, acknowledgement within 72 hours, confirmed issues resolved within 14 days, reporters credited in release notes |
| **Governance** | [GOVERNANCE.md](GOVERNANCE.md) — decision-making, contribution path, and how co-maintainers are added |
| **Code of conduct** | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) — open an issue before anything beyond a small bugfix |
| **Citations & provenance** | [CITATIONS.md](CITATIONS.md) — every method traced to its paper, with deviations documented |
| **Release process** | Automated via [python-semantic-release](https://github.com/python-semantic-release/python-semantic-release); every release is gated on the full test suite. See [CHANGELOG.md](CHANGELOG.md) |

**Maintainership and bus factor.** This project is maintained by one person
([rajveer43](https://github.com/rajveer43)) with a second contributor active on
the codebase. That is a real key-person risk and we would rather state it than
have you discover it: if you are evaluating VeloxQuant-MLX for production or
funding, weigh it accordingly. [GOVERNANCE.md](GOVERNANCE.md) describes how
co-maintainers are added, and the MIT license means the code remains usable and
forkable regardless. Continuity work — expanding the maintainer group and
documenting release-critical knowledge — is open and tracked in
[issues](https://github.com/rajveer43/VeloxQuant-MLX/issues).

**What is verified continuously.** Lint and the non-Metal unit suite run on
GitHub Actions (Ubuntu, Python 3.12) for every push; the full suite, including
Metal parity tests that need Apple Silicon, gates each release. Metal kernel
correctness is covered by dedicated numpy-parity tests rather than benchmarks
alone — see [Metal kernels](#metal-kernels).

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

## Beyond compression: cross-model KV transfer

One capability in this repo is **not** a compression method and is deliberately
not in the 41: [**cross-model KV cache transfer**](https://veloxquant-mlx.netlify.app/algorithms/cross-model-transfer)
(`veloxquant_mlx.transfer`). Instead of shrinking one model's cache, it maps a
*source* model's already-prefilled KV into a *target* model's format, so the
receiver can skip prefill when you swap between two models in the same family.
Cache size is unchanged; what you save is prefill compute.

It lives in its own subsystem rather than behind `method="..."` because it
needs two models, an offline per-pair fit, and a multi-GB artifact — none of
which the single-model cache contract can express. Adapted from
[Cross-Model KV Cache Transfer (NVIDIA, arXiv:2608.03893)](https://arxiv.org/abs/2608.03893);
the paper's retention and speedup figures are its own, measured on datacenter-scale
pairs, and are not reproduced here. See the
[docs page](https://veloxquant-mlx.netlify.app/algorithms/cross-model-transfer)
for the caveats before relying on it.

---

## References

41 methods, each adapted from a published paper with documented deviations
(39 from a verified peer-reviewed venue; 2, NestedKV-adapted and AMC-adapted,
from unpublished preprints as one-time, stated exceptions — see
[CITATIONS.md](CITATIONS.md)) — full bibliography (implemented methods,
related work, and survey papers): **[CITATIONS.md](CITATIONS.md)**. The
cross-model transfer subsystem above is counted separately, as it compresses
nothing.

Headline references: [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874), [VecInfer (2024)](https://arxiv.org/abs/2510.06175), [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2402.02855), [CommVQ (ICML 2025)](https://arxiv.org/abs/2506.18879), [KVzip (NeurIPS 2025)](https://arxiv.org/abs/2505.23416), [KVTC (ICLR 2026)](https://arxiv.org/abs/2511.01815), [CurDKV (NeurIPS 2025)](https://arxiv.org/abs/2509.15038), [NestedKV (preprint, arXiv:2605.26678)](https://arxiv.org/abs/2605.26678), [AMC (preprint, arXiv:2607.10109)](https://arxiv.org/abs/2607.10109), [A2ATS (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.644/). Built on [Apple MLX](https://github.com/ml-explore/mlx).

---

## Support

VeloxQuant-MLX is free and MIT-licensed, with no commercial offering and no
revenue behind it — development time is donated. Funding is not required to use
it and never gates a feature, but it is what buys maintenance time.

- **Using it?** Stars, issue reports, and PRs are the most useful contribution — they tell us what actually breaks on hardware we do not have.
- **Funding it?** [Buy me a chai ☕](https://buymeachai.in/rajveer43) or [tip on Ko-fi 💜](https://ko-fi.com/rajveer43).
- **Reporting a vulnerability?** Do not open a public issue — follow [SECURITY.md](SECURITY.md).
- **Citing it?** [DOI 10.5281/zenodo.20647294](https://doi.org/10.5281/zenodo.20647294); per-method attributions in [CITATIONS.md](CITATIONS.md).

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

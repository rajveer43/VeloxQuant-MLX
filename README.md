<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/veloxquant-logo-dark.svg" />
  <img src="assets/veloxquant-logo.svg" alt="VeloxQuant-MLX — KV-cache quantization and compression for Apple Silicon" width="560" />
</picture>

<p><strong>Run longer contexts in less memory on Apple Silicon.</strong></p>
<p>Research-based KV-cache quantization and compression for Apple MLX.</p>

<p>
  <a href="https://veloxquant.dev/"><img src="https://img.shields.io/badge/docs-veloxquant.dev-0078d4?style=flat-square&logo=readthedocs&logoColor=white" alt="Documentation"/></a>
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/pypi/v/VeloxQuant-MLX?style=flat-square&logo=pypi&logoColor=white&color=0078d4" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/pypi/dm/VeloxQuant-MLX?style=flat-square&logo=pypi&logoColor=white&color=0078d4" alt="Monthly PyPI downloads"/></a>
  <a href="https://pepy.tech/projects/veloxquant-mlx"><img src="https://static.pepy.tech/personalized-badge/veloxquant-mlx?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="Total downloads"/></a>
  <a href="https://github.com/rajveer43/VeloxQuant-MLX/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/rajveer43/VeloxQuant-MLX/release.yml?branch=master&style=flat-square&label=build&logo=github" alt="Release build status"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-0078d4?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-0078d4?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon"/>
  <!-- The tests and changelog badges are rewritten on every release by
       scripts/sync_release_badges.py. Keep their literal badge URL patterns. -->
  <img src="https://img.shields.io/badge/tests-5955%20passing-22c55e?style=flat-square" alt="Tests"/>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.96.0-64748b?style=flat-square" alt="Changelog"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="MIT License"/></a>
  <a href="https://doi.org/10.5281/zenodo.20647294"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20647294-1f6feb?style=flat-square" alt="DOI"/></a>
  <a href="SECURITY.md"><img src="https://img.shields.io/badge/security-policy-1f6feb?style=flat-square" alt="Security policy"/></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=rajveer43.VeloxQuant-MLX&style=flat-square&color=64748b" alt="Visitors"/>
</p>

<p>
  <a href="https://veloxquant.dev/docs/getting-started/quickstart">Quickstart</a> ·
  <a href="https://veloxquant.dev/docs/algorithms/overview">43 methods</a> ·
  <a href="https://veloxquant.dev/playground.html">Compression Lab</a> ·
  <a href="https://ai.miraheze.org/wiki/VeloxQuant-MLX">Wiki</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

</div>

---

## What is VeloxQuant-MLX?

A language model's KV cache grows with every generated token. On a Mac with 16–24 GB of unified memory, a long context can consume more memory than the model weights themselves. In practice, memory—not compute—can become the limiting factor.

**VeloxQuant-MLX replaces the KV cache used by an `mlx_lm` model with a configurable compressed cache.** It provides 43 methods adapted from published research behind one `KVCacheConfig` API: change `method="..."` to explore a different compression strategy. The performance-critical path includes Metal GPU kernels, and the project has been validated on 12 production model families/configurations, including Llama, Mistral, Qwen, Phi, Gemma 3/4, and Falcon. Vision-language models are supported through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

A key distinction: **a theoretical compression ratio is not necessarily a reduction in resident memory.** Many quantizers currently dequantize to fp16 for storage, so their bit-width ratios describe encoded representation rather than actual RSS. Token eviction and merging methods marked **RSS-reducing** do reduce resident memory today. See [the storage audit](https://github.com/rajveer43/VeloxQuant-MLX/issues/27) and [implementation findings](#engineering-findings).

## At a glance

| | |
|---|---|
| **Platform** | Apple Silicon M1 and newer |
| **Backend** | Apple MLX, with Metal kernels for selected hot paths |
| **Python** | 3.10+ |
| **Methods** | 43 research-based methods across quantization, eviction/merging, and hybrid approaches |
| **Integration** | `mlx_lm`, MLX-VLM, local OpenAI-compatible inference server, CLI, and control panel |
| **License** | MIT |

## Performance highlights

These are representative measurements, not universal guarantees. Read each caveat before comparing methods.

| Measurement | Result | Scope and caveat |
|---|---:|---|
| VecInfer key-cache compression | **Up to 16×** | 1-bit keys, `head_dim=128`; key bytes only; requires user-trained codebooks. Throughput can fall substantially on some models. |
| RVQ-1bit key compression | **7.5×** | Key ratio without calibration; values remain fp16. |
| VecInfer Metal quantization | **Up to 14.7× faster** | `quantize_vq`, measured at sequence length 2,048; speedup varies by workload. |
| `quantize_vq` peak scratch memory | **729 MB → 12 MB** | One operation at the Falcon3-7B shape; transient scratch reduction, not KV-cache memory reduction. |
| Qwen2.5-7B VecInfer throughput | **21.5 vs. 21.0 tok/s** | VecInfer-1bit vs. fp16 in one 200-token-prompt run; other models can lose throughput. |
| Models validated | **12** | The cross-model throughput table below reports 6 representative models from a 10-model study. |

The full metric table—including RaBitQ, CommVQ, KIVI, and SpectralQuant—plus measurement methodology is in [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

## Demo

<div align="center">

[![Demo video: VeloxQuant-MLX quantizing a KV cache and running mlx_lm.generate](https://img.youtube.com/vi/lwgsO0n_1VU/maxresdefault.jpg)](https://youtu.be/lwgsO0n_1VU)

</div>

## Installation

```bash
pip install VeloxQuant-MLX
```

**Requirements:** Apple Silicon M1+, Python ≥ 3.10, MLX ≥ 0.18, and NumPy ≥ 1.26.

For source installation, conda/miniforge, Metal troubleshooting, and installation verification, see the [installation guide](https://veloxquant.dev/docs/getting-started/installation).

## Quickstart

### Use a compressed cache from Python

The following example uses RVQ 1-bit, the recommended default for a smaller key cache without calibration.

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
    prompt="Explain relativity simply.",
    max_tokens=200,
)
```

The cache is selected through `KVCacheConfig`; the model continues to use the `mlx_lm` generation interface.

See the [5-minute quickstart](https://veloxquant.dev/docs/getting-started/quickstart), [mixed-precision guide](https://veloxquant.dev/docs/guides/mixed-precision), and [`mlx_lm` integration guide](https://veloxquant.dev/docs/guides/mlx-lm-integration).

### Use the local control panel

```bash
veloxquant panel
```

Open the local web UI at `http://127.0.0.1:7860`, choose a model and method, and select **Start Server**. The panel drives `veloxquant serve`; connect an OpenAI-compatible client (for example, Claude Code, Cursor, or the OpenAI SDK) to the URL it provides.

You can also run the server directly. See [`docs/control-panel.md`](docs/control-panel.md).

### Serve a Qwen3.5 vision-language model

Qwen3.5 is served through MLX-VLM rather than the text-only `veloxquant serve` launcher. Use `veloxquant vlm-serve` to apply VeloxQuant's cache implementation to the language-model KV cache.

```bash
uv sync --extra vlm

uv run --extra vlm veloxquant vlm-serve \
  --method turboquant_rvq \
  --bits 2 \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --host 127.0.0.1 \
  --port 1337 \
  --prefill-step-size 1024 \
  --max-tokens 512
```

Qwen3.5-9B natively supports a 262,144-token context. The model's context limit applies to prompt tokens plus generated tokens. In this command, `--bits 2` configures VeloxQuant's KV-cache compression; it does **not** set the model-weight quantization level.

#### OpenAI-style system messages

VeloxQuant accepts OpenAI-style leading `system` messages without an additional server flag. For Mistral, it installs a compatible Jinja chat template when the model loads; the HTTP request remains a separate `system` message followed by a `user` message. Qwen3.5's native template supports this role and is left unchanged.

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "Explain KV-cache compression."}
  ]
}
```

Mistral has no separate system token, so the compatibility template renders the system content in its first `[INST]` block without mutating the OpenAI message array. To confirm that VeloxQuant is active, look for `Wired … layer cache(s)` in the startup log.

---

<a id="method-library"></a>

## Choosing a method

All cache methods use the same configuration entry point: set `method="<id>"` in `KVCacheConfig`. The [algorithm overview](https://veloxquant.dev/docs/algorithms/overview) contains the complete comparison table, decision tree, per-model recommendations, configuration details, evidence, and limitations.

### Recommended starting points

You can ask the built-in recommender for a starting point:

```bash
python -m veloxquant_mlx recommend --goal everyday
```

Or use one of the methods currently marked `stable`:

| Goal | Method | Maturity |
|---|---|---|
| Smaller key cache without calibration | `turboquant_rvq` with `bit_width_inlier=1` | Stable |
| Quality-first quantization | `kivi` | Stable |
| Cap the prompt's cache and reduce resident memory (decode still grows) | `snapkv` | Stable |

Measured results for these three methods are tracked for the reproducible benchmark in [issue #786](https://github.com/rajveer43/VeloxQuant-MLX/issues/786).

### The 43-method library

The methods are grouped into three families:

| Family | Count | Examples | What they do |
|---|---:|---|---|
| **Quantization** | 20 | TurboQuant/RVQ, VecInfer, KIVI, KVQuant, SVDq, GEAR, PALU, XQuant, QJL, PolarQuant, SpectralQuant, KVTC | Reduce the representation precision of cached tokens. |
| **Token eviction and merging** | 16 | SnapKV, StreamingLLM, H2O, TOVA, PyramidKV, ChunkKV, Keyformer, KVzip, xKV | Drop or merge lower-priority tokens; these methods can reduce resident memory. |
| **Hybrid** | 7 | MiniCache, A2ATS, AnchorKV, KIVI-sink, RocketKV, SKVQ, ZipCache | Combine quantization with eviction or cross-layer merging. |

The maturity labels are exposed by `veloxquant methods` and `list_methods()`:

- **Stable:** suitable starting points and recommended defaults.
- **Beta:** tested, but not recommended as a default.
- **Experimental:** research ports, known deviations from the paper, or methods not yet servable.

For example, VecInfer requires user-trained codebooks and is classified as beta.

> **Research adaptation note:** A method labelled `-adapted` is not a 1:1 reproduction of its paper. The cache interface exposes per-layer K/V tensors, not the model's actual attention maps. Methods that rely on attention signals therefore use a key-as-query proxy.

See the [full method catalogue](https://veloxquant.dev/docs/algorithms/overview) for per-method details.

## Metal GPU kernels

Selected performance-critical operations use MLX's `mx.fast.metal_kernel`. VecInfer's `quantize_vq` was the slowest step in its pipeline, so it was moved to a JIT-compiled Metal kernel without changing the Python API.

| Metric | Pure MLX | Metal kernel | Change |
|---|---:|---:|---:|
| Quantization latency (`S=8192`) | 228 ms | **15.6 ms** | **14.7× faster** |
| Peak scratch memory (Falcon3-7B shape) | 729 MB | **12 MB** | **98% lower** |

The memory reduction comes from avoiding an intermediate `[N, n_centroids, sub_dim]` difference tensor. The kernel keeps the argmin accumulator in thread-local GPU registers instead.

A second set of kernels implements a GPU-resident **RaBitQ** pipeline: 1-bit packed keys scored using XOR and popcount, 4-bit codebook values, and a fused dispatch. It measured **1.78×** versus dequantize-plus-SDPA at `S_kv=8192`.

**Kernel caveat:** Metal kernels incur approximately 50–200 µs of launch overhead per call. On tiny models (for example, SmolLM2-135M, with roughly 60 launches per token), this overhead can exceed the savings. These kernels target 7B+ models at realistic context lengths.

The fused decode-and-attend kernel has different constraints: profiling found it was bound by thread-group count rather than memory bandwidth. Several possible optimizations were measured and rejected. See the [roofline findings](docs/KV_KERNEL_ROOFLINE_FINDINGS.md) and [optimization notes](blogs/metal-kernels.md).

More: [Metal GPU kernel guide](https://veloxquant.dev/docs/guides/metal-kernels).

## Benchmark results

<div align="center">
  <img src="figures/vecinfer/_summary/cross_model_comparison.png" alt="Cross-model comparison of VecInfer and RVQ-1bit across 10 models" width="820"/>
  <br/>
  <sub>End-to-end <code>mlx_lm.generate</code> · 200-token prompt · 120-token generation · Apple M-series unified memory</sub>
</div>

The following table is a representative subset of the v0.5.0 10-model VecInfer-versus-RVQ study. Throughput is measured end-to-end using `mlx_lm.generate`.

| Model | fp16 | RVQ-1bit (7.5× key ratio) | VecInfer-1bit (16× key ratio) |
|---|---:|---:|---:|
| Llama-3.2-3B | 47.6 | **46.2** | 40.2 |
| Llama-3.1-8B | 20.5 | **20.6** | 19.6 |
| Mistral-7B | 23.6 | **22.8** | 9.8 |
| Qwen2.5-7B | 21.0 | 20.7 | **21.5** |
| Qwen3-8B | 20.3 | **19.6** | 2.4 |
| Falcon3-7B | 17.3 | **21.7** | 17.0 |

*Throughput is in tokens per second.*

### Interpreting the results

- **RVQ-1bit** is the safer general starting point in this study: it stayed within 5% of fp16 on most tested 7–8B models and requires no calibration.
- **VecInfer-1bit** provides a higher key-cache compression ratio. It performed well on some strong-GQA models, including Qwen2.5, but throughput degraded substantially on others.
- The 16× and 7.5× figures describe **key-cache ratios**; values remain fp16. They should not be read as equivalent whole-cache or resident-RAM reductions.

All ten models, additional ratios, historical snapshots, and methodology are documented in [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

**Benchmark methodology:** throughput figures are end-to-end `mlx_lm.generate` measurements. Throughput can vary between processes on the same machine, so cross-method speed comparisons should use same-process interleaved runs. Quality is evaluated using paired negative log-likelihood (NLL) against fp16, not perplexity alone.

## Engineering findings

The project documents cases where an implementation or benchmark did not behave as expected. These findings are part of the performance record and are useful context when evaluating the methods.

### Compression must reduce stored memory

An early `turboquant_rvq` implementation encoded keys and immediately decoded them back to fp16. Its reported ratio therefore described the encoding, not the stored cache. An audit found the same pattern in several methods ([issue #27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)).

`turboquant_rvq` now stores keys bit-packed. In one single-shot prefill on Llama-3.2-1B, peak memory was 1,402 MB, compared with 1,608 MB for fp16 and 2,537 MB for `mlx_lm`'s `QuantizedKVCache(bits=4)`. This is one prompt on one machine, not a general performance claim. Values remain fp16, so the 7.5× figure is a key ratio. Details: [RVQ packed-storage findings](docs/RVQ_PACKED_STORAGE_FINDINGS.md).

### Fused kernels need workload-aware profiling

Profiling found the fused decode kernel used at most about 6% of memory bandwidth, even at 16k cached tokens. It was constrained by thread-group count, not bytes moved. GQA head-packing was 2.7–4.7× slower; cross-head sharing is not possible in Metal's SIMD model; and cross-layer batching does not work for single-request decode because each layer depends on the previous layer's output.

Autotuning `nsg` improved kernel-level performance by 1.2–4.2× and end-to-end performance by 1.07–1.85×, depending on context length. An earlier report used the wrong baseline and was corrected. Against plain fp16, the fused KIVI path remains at 0.48–0.85×.

Details: [KV kernel roofline findings](docs/KV_KERNEL_ROOFLINE_FINDINGS.md) · [NSG autotuning report](docs/NSG_AUTOTUNE_BENCHMARK_REPORT.md).

### Benchmark validation matters

An audit of benchmark scripts found two that produced misleading comparisons and one that did not run. A separate ground-truth check found the 8-bit Lloyd-Max codebook error was 2.8× higher than it should have been (1.05e-6 versus 3.71e-7). This did not measurably affect the common 2-bit and 4-bit settings.

### Hardware limits

Development and benchmarks were run on an M4 MacBook Air with 24 GB unified memory. Qwen2.5-32B-4bit leaves insufficient headroom for a quantizer on that machine, so larger-model results are not measured. See [memory constraint findings](docs/MEMORY_CONSTRAINT_FINDINGS.md).

## Architecture

The cache methods share a common configuration and integration path:

`KVCacheConfig` → `KVCacheBuilder` → `mlx_lm`-compatible cache.

At a high level, quantization methods can rotate K/V into a more suitable basis, quantize the representation (optionally with a residual pass), and pack the resulting bits. The exact operations depend on the selected method.

For pipeline diagrams and design patterns, see [Core concepts](https://veloxquant.dev/docs/getting-started/concepts).

## CLI

VeloxQuant includes commands for method selection, workload-aware configuration, benchmarking, and artifact precomputation.

```bash
# Get a method recommendation for your Mac
python -m veloxquant_mlx recommend \
    --chip M4 --ram-gb 16 --model-class 7B --goal everyday

# Generate a hardware-aware config for a workload
python -m veloxquant_mlx auto-config \
    --head-dim 128 --seq-len 32000 --n-layers 32 --batch-size 4 --json

# Run a synthetic benchmark for one configuration
python -m veloxquant_mlx benchmark \
    --method turboquant_prod --head_dim 128 --bits 2 --seq_len 1000

# Precompute rotation matrices, JL matrices, and codebooks
python -m veloxquant_mlx precompute \
    --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --seed 42 --output_dir ./artifacts/
```

The `recommend` command is accounting-aware: it reports the key compression ratio and flags cases where resident-RAM savings are unlikely. Available goals are `everyday`, `max_key_accounting`, `max_context`, `best_quality`, and `constant_memory`. Add `--json` for machine-readable output.

You can also use the browser-based [Compression Lab](https://veloxquant.dev/playground.html). For the full command reference—including loading precomputed artifacts to skip runtime computation—see the [CLI documentation](https://veloxquant.dev/docs/api/core-api).

## Beyond compression: cross-model KV transfer

VeloxQuant also includes **cross-model KV-cache transfer**, a separate capability that is not counted among the 43 compression methods. It is available through `veloxquant_mlx.transfer` and [documented separately](https://veloxquant.dev/docs/algorithms/cross-model-transfer).

Rather than compressing one model's cache, this subsystem maps a source model's already-prefilled KV cache into a target model's format. The target can then skip prefill when switching between two models in the same family. Cache size is unchanged; the intended saving is prefill compute.

It is separate from the `method="..."` cache API because it requires two models, an offline per-pair fit, and a multi-GB artifact. The implementation is adapted from [Cross-Model KV Cache Transfer (NVIDIA, arXiv:2608.03893)](https://arxiv.org/abs/2608.03893). The paper's retention and speedup figures were measured on datacenter-scale model pairs and have not been reproduced here. Review the [transfer documentation](https://veloxquant.dev/docs/algorithms/cross-model-transfer) before relying on this capability.

## Ecosystem and integrations

This repository contains the Python engine. Client SDKs in other languages connect to a running `veloxquant serve` instance for hardware detection, KV-cache estimation, and OpenAI-compatible local inference.

| Language | Package | Install | Source |
|---|---|---|---|
| TypeScript / JavaScript | [`@veloxquant/sdk`](https://www.npmjs.com/package/@veloxquant/sdk) | `npm i @veloxquant/sdk` | [veloxquant-sdk](https://github.com/rajveer43/veloxquant-sdk) |
| Rust | [`veloxquant`](https://crates.io/crates/veloxquant) | `cargo add veloxquant` | [veloxquant-rs](https://github.com/rajveer43/veloxquant-rs) |
| Go | [`veloxquant-go`](https://pkg.go.dev/github.com/rajveer43/veloxquant-go) | `go get github.com/rajveer43/veloxquant-go` | [veloxquant-go](https://github.com/rajveer43/veloxquant-go) |
| Kotlin / JVM | [JitPack](https://jitpack.io/#rajveer43/veloxquant-kotlin) | `com.github.rajveer43:veloxquant-kotlin` | [veloxquant-kotlin](https://github.com/rajveer43/veloxquant-kotlin) |
| Swift | [Swift Package Manager](https://github.com/rajveer43/veloxquant-swift) | `.package(url: "...veloxquant-swift", from: "0.1.0-alpha")` | [veloxquant-swift](https://github.com/rajveer43/veloxquant-swift) |

The Rust project also provides focused sub-crates—`veloxquant-core`, `veloxquant-memory`, `veloxquant-system`, `veloxquant-runtime`, `veloxquant-openai`, and `veloxquant-monitor`—for narrower dependencies. Kotlin and Swift are at `0.1.0-alpha`.

There is also a [VS Code extension](https://marketplace.visualstudio.com/items?itemName=veloxquant-mlx.veloxquant-vscode) ([source](https://github.com/rajveer43/veloxquant-vscode)) that recommends compression settings and opens the Compression Lab from the editor.

## Development

Run the test suite and selected validation scripts from the repository root:

```bash
# Full test suite, including Metal parity tests
pytest veloxquant_mlx/tests/ -v

# Fast synthetic validation for 2-bit improvements
python benchmark_scripts/test_2bit_improvements.py

# Regenerate the optimization-journey figure
python scripts/plot_optimization_journey.py
```

Contributions are welcome. For anything beyond a small bug fix, open an issue before starting work. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution and release guidelines.

Additional project records:

- [`RELEASE_NOTES.md`](RELEASE_NOTES.md) — concise per-version summaries.
- [`CHANGELOG.md`](CHANGELOG.md) — full release history.
- [Good first issues](https://www.goodfirstissue.org/rajveer43/VeloxQuant-MLX).

## Project policies and governance

| Area | Policy or resource |
|---|---|
| Security | [`SECURITY.md`](SECURITY.md) — private vulnerability disclosure by email, acknowledgement within 72 hours, confirmed issues resolved within 14 days, and reporter credit in release notes. |
| Governance | [`GOVERNANCE.md`](GOVERNANCE.md) — decision-making, contribution paths, and how co-maintainers are added. |
| Code of conduct | [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). |
| Contributions | [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow and release process. |
| Research provenance | [`CITATIONS.md`](CITATIONS.md) — paper references and documented deviations for the methods. |
| Release process | Weekly and automated via `python-semantic-release`; every release is gated on the full test suite. See [release guidelines](CONTRIBUTING.md#releases) and [`CHANGELOG.md`](CHANGELOG.md). |

**Maintainership:** The project is maintained by [rajveer43](https://github.com/rajveer43) with one other active contributor. This is a relevant key-person risk for production evaluations; the governance document explains how co-maintainers are added.

Linting, the non-Metal unit suite, and the full MLX/Metal suite run on CI for every PR and push to `master`. The release workflow runs the full suite again as its release gate. Metal tests use a GitHub-hosted macOS runner with a paravirtual Metal device to check correctness, not performance. Performance measurements come from real hardware.

Independent project listings:

- [Learn AI wiki (Miraheze)](https://ai.miraheze.org/wiki/VeloxQuant-MLX) — a community-editable, third-party encyclopedia article. This README and [`CITATIONS.md`](CITATIONS.md) remain the sources of truth for project figures.
- [Startuply profile](https://startuply.vc/startup/veloxquant-mlx-edq89q) — an independent, AI-generated directory profile. Its figures are not project-verified; use this README and the measured results as the source of truth.
- [INITE Atlas listing](https://inite.ai/en/atlas/veloxquant.dev) — an independent, automatically generated directory card. Its "up to 98 percent" peak-memory tagline is not a project claim and is not supported by this project's measurements; use this README and the measured results as the source of truth.
- [Hunted.Space launch dashboard](https://www.hunted.space/dashboard/veloxquant-mlx) — a third-party tracker for the project's Product Hunt launch. It reports launch activity (votes, comments, ranking), not project performance, and repeats the launch text as submitted.

## Documentation and research notes

- **Documentation:** [veloxquant.dev](https://veloxquant.dev/)
- **API reference:** [Read the Docs](https://veloxquant-mlx.readthedocs.io/en/latest/quickstart.html)
- **Algorithm catalogue:** [All 43 methods](https://veloxquant.dev/docs/algorithms/overview)
- **Playground:** [Compression Lab](https://veloxquant.dev/playground.html)

Selected technical writeups:

- [Project overview](https://veloxquant.dev/docs/blog/overview)
- [10-model study](https://veloxquant.dev/docs/blog/10-model-study)
- [Hands-on tutorial](https://veloxquant.dev/docs/blog/hands-on)
- [KIVI](https://veloxquant.dev/docs/blog/kivi)
- [Metal kernels](https://veloxquant.dev/docs/blog/metal-kernels)
- [Results](https://veloxquant.dev/docs/blog/results)
- [TensorOps research](https://veloxquant.dev/docs/blog/tensorops-research)

The source writeups are also in [`blogs/`](blogs/).

## References

The 43 methods are adapted from published papers, with deviations documented. The project README reports 39 peer-reviewed venue references and 2 preprint exceptions. The complete bibliography and per-method provenance are in [`CITATIONS.md`](CITATIONS.md).

Selected references:

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [VecInfer (2024)](https://arxiv.org/abs/2510.06175)
- [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2402.02855)
- [CommVQ (ICML 2025)](https://arxiv.org/abs/2506.18879)
- [KVzip (NeurIPS 2025)](https://arxiv.org/abs/2505.23416)
- [KVTC (ICLR 2026)](https://arxiv.org/abs/2511.01815)
- [CurDKV (NeurIPS 2025)](https://arxiv.org/abs/2509.15038)

Built on [Apple MLX](https://github.com/ml-explore/mlx).

## Support

VeloxQuant-MLX is free and MIT-licensed. There is no commercial offering or revenue behind the project. Funding is not required to use it and does not gate features; it supports maintenance time.

- **Contribute:** Stars, issue reports, and pull requests are useful—especially reports from hardware configurations the maintainers do not have.
- **Support development:** [Buy me a chai](https://buymeachai.in/rajveer43) or [tip on Ko-fi](https://ko-fi.com/rajveer43).
- **Report a vulnerability:** Do not open a public issue. Follow [`SECURITY.md`](SECURITY.md).
- **Ask questions or report bugs:** [GitHub Issues](https://github.com/rajveer43/VeloxQuant-MLX/issues).

## Star history

<div align="center">
  <a href="https://www.star-history.com/?repos=rajveer43%2FVeloxQuant-MLX&type=date&legend=top-left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=rajveer43/VeloxQuant-MLX&type=date&theme=dark&legend=top-left" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=rajveer43/VeloxQuant-MLX&type=date&legend=top-left" />
      <img alt="GitHub star history for rajveer43/VeloxQuant-MLX" src="https://api.star-history.com/chart?repos=rajveer43/VeloxQuant-MLX&type=date&legend=top-left" />
    </picture>
  </a>
</div>

## Citation

Using VeloxQuant-MLX in code, products, or infrastructure does not require permission or an academic citation under the MIT license. If you redistribute a copy or a substantial portion of the source, retain the MIT copyright and permission notice.

For academic and research use, citing the project is appreciated:

```bibtex
@software{veloxquant_mlx,
  author    = {Rathod, Rajveer},
  title     = {VeloxQuant-MLX: Fast KV-Cache Quantization for Apple Silicon},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20647294},
  url       = {https://doi.org/10.5281/zenodo.20647294}
}
```

Per-method attributions and the full bibliography are maintained in [`CITATIONS.md`](CITATIONS.md). For substantial contributions and potential acknowledgement or co-authorship, see the [authorship and acknowledgement policy](GOVERNANCE.md#authorship--acknowledgement-policy).

## License

VeloxQuant-MLX is released under the [MIT License](LICENSE). The license covers code in this repository; it does not grant rights to the research papers from which methods are adapted. See [`CITATIONS.md`](CITATIONS.md) for method provenance.

---

<div align="center">
  <sub>Apple Silicon M1+ · Python 3.10+ · 43 methods · MIT License</sub>
  <br/>
  <sub>
    <a href="https://veloxquant.dev/">Documentation</a> ·
    <a href="https://github.com/rajveer43/VeloxQuant-MLX/issues">Issues</a> ·
    <a href="CONTRIBUTING.md">Contributing</a>
  </sub>
</div>

---
id: changelog
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

---

## v0.42.0 — Latest

### Added
- **Fused group-affine (KIVI-style) decode + attention Metal kernel.** `scalar_fused_decode_attend` (`veloxquant_mlx/metal/_scalar_attend.py`) runs SDP attention directly over an asymmetric group-min/max quantized cache — the KIVI / SKVQ / Kitty / group-quant family, where K/V are `uint8` codes plus a per-group `(scale, zero)`. It reconstructs `k_hat`/`v_hat` in-register inside a FlashAttention-style online softmax, so no dequantized `K_hat`/`V_hat` reaches DRAM, killing the `dequantize -> DRAM -> SDPA` round-trip the pure-MLX path pays every decode step. Measured on Apple M4 (B=1 H=32 D=128 b=2 g=32 S_q=1): **6.4x at S_kv=512 rising to 12.2x at S_kv=65536**, parity max abs 1.2e-4 (fp32 softmax accumulation — more accurate than the fp16 baseline). See [Metal kernels guide](/docs/guides/metal-kernels) and [Metal API](/docs/api/metal-api).
- **Mac / RAM method recommender** — `veloxquant_mlx/tools/mac_recommender.py` and the `veloxquant recommend` CLI subcommand pick a method and bit budget from a Mac's unified memory and target model/context.
- **Interactive landing playground** — a browser-side Compression Lab and benchmark browser at `landing/playground.html`.

### Fixed
- **Release automation had been silently skipping every release since 0.41.0.** `build_command = false` is rejected by python-semantic-release >=10 (validated as a string), and the workflow discarded the resulting stderr — making a hard config error indistinguishable from "nothing to release". Four merged PRs never produced a tag. Also added `allow_zero_version = true`, without which the next release would have been `1.0.0` rather than `0.42.0` despite no breaking changes.
- Documented `scalar_fused_decode_attend` and `rabitq_prefill_attend` (the latter shipped in 0.40.x but was never added to the docs site).
- Install guide: broken `precompute` command corrected; install docs consolidated.
- Non-Metal CI no longer imports `mlx` through the package `conftest`.

---

## v0.41.0

### Added
- **mlx-vlm vision-language model support.** `patch_vlm_kv_cache(model, config)` (`veloxquant_mlx/integration/mlx_vlm_patch.py`) wires VeloxQuant caches into mlx-vlm models (Qwen2-VL, LLaVA, …). Verified against mlx-vlm 0.6.5: it overrides `model.language_model.make_cache()` exactly, leaving the batch/session path safe. Caches rebuild fresh on every `generate()`, so repeated generations never leak KV state. Token-eviction methods emit a `UserWarning` since image tokens sit in the prompt prefix. 8 integration tests.

### Fixed
- Integration guide "Pattern 2" documented a nonexistent `patch_mlx_lm` function with a nonexistent `bits=` kwarg; it now shows the real `patch_model_kv_cache` API.
- README badges: stale test count and changelog version.

---

## v0.40.0

### Added
- **Fused RaBitQ asymmetric Metal kernel pipeline** — a fully GPU-resident path for a 1-bit-key / 4-bit-value cache. `rabitq_fused_attend` scores keys from packed sign bits (XOR + popcount) with an online softmax split across 8 SIMD-groups, gathering values from a scalar codebook; measured 1.10–1.78x vs. dequantize+SDPA. `rabitq_encode` fuses rotate + binarize + bit-pack + L1 magnitude in one dispatch (`simd_ballot`), 6x vs. the numpy round-trip at N=32768. `rabitq_pack_values` packs two 4-bit indices per byte, halving value-cache memory with bit-identical outputs. `rabitq_prefill_attend` adds a `simdgroup_matrix`-tiled prefill/cross-attention companion for large `S_q` (multi-turn VLM). 63 new parity tests.

---

## v0.39.1

### Fixed
- **VecInfer's fused Metal encode+decode kernels silently dropped most/all tokens.** `vecinfer_encode_decode_metal` and `vecinfer_encode_decode_simple_metal` (`veloxquant_mlx/metal/_vecinfer.py`) dispatch one `D`-wide threadgroup per token, but passed `grid=(n_tokens, 1, 1)` — `mx.fast.metal_kernel`'s `grid` is in **threads**, not threadgroups, so this silently launched only `floor(n_tokens / D)` threadgroups (zero when `n_tokens < D`). Every token past that count kept uninitialized output-buffer contents instead of a real key/value reconstruction or codebook index — affecting every VecInfer Metal-accelerated encode/decode call. Fixed by dispatching `n_tokens * D` threads. `test_vecinfer_fused_sdpa.py` / `test_vecinfer_metal_parity.py` (5 failing tests) now pass.
- **Silent sink-token eviction when `n_sink >= budget`** in `pyramidkv`, `squeeze`, `chunkkv`, `curdkv` (`veloxquant_mlx/quantizers/{pyramidkv,squeeze,chunkkv,curdkv}.py`). These four `init_*_state` functions accepted degenerate sink/budget configs that leave no evictable room, silently evicting tokens documented as sink-protected — the same bug class `h2o`/`tova` were already guarded against (0.39.0, `a78cd7f`). Added the matching `n_sink < budget` check plus regression tests to all four.
- `benchmark_scripts/pyramidkv_benchmark_results.json` regenerated against the fixed kernels.

## v0.39.0
- **A2ATS-adapted** (`method="a2ats"`) — windowed RoPE + query-aware retrieval VQ. Every VQ method already in the repo applies RoPE uniformly (VecInfer: no RoPE-awareness at all; CommVQ-adapted: codebook-constrained, one exact rotation per position regardless of distance); A2ATS instead gates RoPE precision by each token's **distance** from the current decode position — exact within a trailing window, a single shared fixed-offset approximation outside it — combined with **query-aware** codebook assignment for a retrieval-fraction subset of tokens. This is a normal-track method: a live-verified peer-reviewed venue (ACL 2025 Findings), no exception needed. Inspired by "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu, Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings, aclanthology.org/2025.findings-acl.644) — documented as "A2ATS-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `a2ats_apply_exact_rope`/`a2ats_apply_windowed_rope` (`veloxquant_mlx/quantizers/a2ats_rope.py`) — distance-gated exact/approximate RoPE; `window<=0` degrades to always-approximate, `window` at or beyond sequence length degrades to always-exact.
  - `a2ats_query_aware_assignment`/`a2ats_select_retrieval_set` (`veloxquant_mlx/quantizers/a2ats.py`) — query-aware nearest-centroid assignment (`beta`-blended reconstruction-error/query-cosine-similarity) and heap-based retrieval-set top-k selection, reusing `dsa.MaxHeap` (the same pattern as `amc_assign_tiers`).
  - `A2ATSKVCache` (`veloxquant_mlx/cache/a2ats_cache.py`) — retrieval-set tokens get query-aware codebook assignment, bulk tokens get plain nearest-centroid assignment (reusing `VecInfer`'s `quantize_vq`/`dequantize_vq`); reconstruction is followed by windowed RoPE. Values follow a plain nearest-centroid VQ path (no RoPE, no retrieval preference).
  - Config: `a2ats_codebook_bits` (8), `a2ats_sub_dim` (8), `a2ats_window` (128), `a2ats_use_query_aware` (True), `a2ats_beta` (0.5), `a2ats_retrieval_fraction` (0.20), `a2ats_rope_base` (10000.0), `a2ats_codebook` (None).
  - 51 tests (13 RoPE + 13 quantizer + 25 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_a2ats.py`). Config-validation tests (`a2ats_beta`/`a2ats_retrieval_fraction` bounds) were written **first**, following a same-session bug-hunt finding that 5 sibling methods had shipped without this exact check.

### Honest scope
- **No query visible at cache level** — the incoming key vector is used as a proxy query for both the retrieval-set split and query-aware assignment, the same category of approximation as H2O-adapted/SnapKV-adapted/AMC-adapted's proxy-query methods.
- **Windowed RoPE has a real, measured cost in every geometry tested** — roughly 2.8x higher reconstruction MSE than always-exact RoPE on a positional-locality-favorable geometry, and roughly 4.4x higher on a long-range-dependent one. Not merely a long-range weakness — stated plainly in `benchmark_scripts/benchmark_a2ats.py`'s closing summary.
- **Query-aware assignment has higher reconstruction MSE than plain nearest-centroid VQ in every row measured** — mathematically expected (`beta=1.0` reduces exactly to plain VQ), not a bug; the intended downstream retrieval-quality payoff is not something an offline reconstruction-MSE benchmark can show.
- **Offline codebook calibration required** — same footgun class as VecInfer/CommVQ-adapted/Palu/SVDq/AMC-adapted; the default random-init codebook exists only for wiring/shape tests.
- No CUDA kernel fusion reproduced; no composition with CommVQ-adapted's RoPE-commuting codebook constraint attempted.

---

## v0.38.0

### Venue exception (read first)
**AMC-adapted is the second method in VeloxQuant-MLX (2 of 40) that does not
trace to a verified peer-reviewed venue** — the first was NestedKV-adapted
(v0.37.0). AMC (arXiv:2607.10109) is a bare single-revision preprint
(submitted 2026-07-11, no Comments/journal-ref field, live-verified 3 days
later on 2026-07-14). It ships anyway as a **second one-time, user-directed
exception**. The next method survey reverts to requiring a verified venue —
this is not a new standing precedent.

### Scope cut (read first)
**Roughly half of AMC's source paper (Sections IV-V: 45nm CMOS RTL, Verilog
clock-gating, the Precision-Gated Systolic Array, the Narrow-Width SRAM
write-back buffer, all pJ/µJ energy figures, the EDAP/Pareto silicon
comparisons) is entirely out of scope.** VeloxQuant-MLX is a pure-software
MLX library with no RTL/silicon layer to target. Only the software saliency
engine and rank/precision scaling math (Sections II-A, III) are ported.

### New
- **AMC-adapted** (`method="amc"`) — saliency-driven tiered rank + precision. Every rank-adaptive method already in the repo (Palu) and every bit-width-adaptive method (KIVI, SKVQ, RateQuant) picks **one** axis to adapt; AMC is the first to drive **both** rank and bit-width from a **single** per-token L1-norm saliency score, via three discrete tiers (High: rank 128/16-bit, Mid: rank 43/8-bit, Low: rank 8/4-bit at head_dim=128). Unlike every eviction method in the repo (H2O, SnapKV, CurDKV, NestedKV, ...), AMC never drops a token — compression-only. Inspired by "Adaptive Model Compression (AMC): Saliency-Driven Resource Allocation for Ultra-Low-Power Transformer Inference" (Hu, Yuan, Hu, Yin, Li, Suchter — Apple; arXiv:2607.10109) — documented as "AMC-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `amc_calibrate_channel_order`/`amc_permute_weights` (`veloxquant_mlx/quantizers/amc_calibration.py`) — offline, one-time SVD-based variance-descending channel permutation (Algorithm 1 Phase I), reusing the same `mx.linalg.svd` pattern as Palu/SVDq.
  - `amc_saliency`/`amc_query_aware_saliency`/`amc_assign_tiers`/`amc_adaptive_thresholds`/`amc_apply_rank_mask`/`amc_quantize_tier`/`amc_pack_low_tier` (`veloxquant_mlx/quantizers/amc.py`) — Eq. 1-7 faithfully ported; tier assignment uses `dsa.MaxHeap` top-k selection, closed-loop threshold state uses `dsa.RingBuffer`, Low-tier 4-bit codes use `dsa.BitPackBuffer` — all reused from the existing `veloxquant_mlx/dsa/` module rather than reimplemented.
  - `AMCKVCache` (`veloxquant_mlx/cache/amc_cache.py`) — every call (prefill or decode) scores, tiers, rank-masks, and quantizes every token; no eviction ever, stored sequence length always equals tokens seen.
  - Config: `amc_k_high` (0.20), `amc_k_mid` (0.30), `amc_use_query_saliency` (False), `amc_query_alpha` (0.5), `amc_adaptive_thresholds` (False), `amc_threshold_window` (64), `amc_gamma` (0.1), `amc_calib_variance` (None), `amc_group_size` (32).
  - 51 tests (9 calibration + 23 quantizer + 19 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_amc.py`).

### Honest scope
- **No verified peer-reviewed venue** and **hardware/RTL half out of scope** — see the two sections above.
- **Compression-only, never eviction** — a structurally different family from every other eviction method in the repo.
- **Query-aware saliency (Eq. 3) and closed-loop adaptive thresholds (Eq. 4-5) are opt-in, off by default.** The default path is pure magnitude-only scoring (Eq. 1-2).
- **Offline SVD/PCA channel-order calibration required** for the rank mask to be meaningful — `AMCKVCache` does not auto-invoke it; callers must run `amc_calibrate_channel_order` themselves before deployment.
- **A real, honestly-reported weakness found during benchmark construction**: on activation distributions with no genuine saliency signal (uniform magnitude), AMC's fixed percentile tiering comes out roughly 100x **worse** in reconstruction MSE than a matched-budget uniform baseline — not merely neutral. On the geometry the mechanism is designed for (sparse outliers), AMC beats the same baseline by roughly 8x. Both reported plainly in `benchmark_scripts/benchmark_amc.py`'s closing summary.
- The paper's own energy/throughput/accuracy numbers (59.2% energy reduction, 2.24x throughput, 3.6% accuracy trade-off) are hardware-measured on the paper's own 45nm RTL simulation and a specific 3-layer synthetic transformer setup — not reproduced here.

---

## v0.37.0

### Venue exception (read first)
**NestedKV-adapted is the first method in VeloxQuant-MLX (1 of 39 at the
time) that did not trace to a verified peer-reviewed venue.** Every one of
the prior 38 methods required a live-verified venue before implementation;
NestedKV (arXiv:2605.26678) is still a bare single-revision preprint
(submitted 2026-05-26, no Comments/journal-ref field) as of 2026-07-14. It
shipped anyway as a **one-time, user-directed exception**. See
[`NEW_METHOD_SURVEY_V21.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/paper/research/surveys/NEW_METHOD_SURVEY_V21.md).

### New
- **NestedKV-adapted** (`method="nestedkv"`) — multi-scale ensembled prefill eviction. Every eviction method already in the repo (H2O, SnapKV, CurDKV, PyramidKV, Keyformer, MorphKV, KVzip, ...) scores a token from **one** importance signal; NestedKV keeps **three** parallel key-only continuum-memory statistics (stable/global, episodic/block-local, current/recent-window), scores each token's anomaly against all three independently, and combines the rankings via a training-free **head-adaptive blend** (which scale is most discriminative on this head) and a per-token **surprise-gated route** (fall back to the single strongest scale when the three disagree). Inspired by "NestedKV: Nested Memory Routing for Long-Context KV Cache Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu; arXiv:2605.26678) — documented as "NestedKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `NestedKVState`/`nestedkv_score`/`nestedkv_allocate_head_budgets`/`nestedkv_compress_prefill`/`nestedkv_append_decode` (`veloxquant_mlx/quantizers/nestedkv.py`) — one-shot per-head scoring over the full prefill sequence, a cross-head budget-competition allocator (paper's component 5), and a plain unscored decode-append path.
  - `NestedKVKVCache` (`veloxquant_mlx/cache/nestedkv_cache.py`) — mirrors `SnapKVKVCache`'s prefill-once/decode-append phase split, not H2O's/CurDKV's per-step loop; zero-pads ragged per-head outputs (the first method here with legitimately unequal per-head token counts) purely for tensor-stacking, with byte accounting computed from each head's true unpadded state.
  - Config: `nestedkv_budget` (512), `nestedkv_n_sink` (4), `nestedkv_window` (64), `nestedkv_beta` (3.0), `nestedkv_tau` (0.60), `nestedkv_kappa` (10.0), `nestedkv_safeguard_alpha` (0.20) — all four gate/blend constants taken directly from the paper's Appendix A defaults.
  - 47 tests (30 quantizer + 17 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_nestedkv.py`).

### Honest scope
- **No verified peer-reviewed venue** — see the venue-exception section above.
- **One-shot prefill compression; the cache is NOT bounded during decode.** The paper's own design computes scores, blend weights, and surprise gates once at the end of prefill; decoded tokens are appended normally, never rescored or evicted. A faithful port of the paper's actual design, not a shortcut — but a real structural difference from every other eviction method in the repo (H2O, CurDKV, StreamingLLM all stay bounded through decode).
- **A structural finding from benchmark construction, not a bug**: at small synthetic scale, the head-adaptive blend's min-max normalization can make the stable scale's discriminative gap come out near-maximal almost by construction regardless of whether it's the actually-relevant signal for a given token, and the surprise gate's mean-centered threshold does not always fully compensate at that scale — the benchmark's `local_episodic_only` geometry shows 0% retention for both NestedKV and H2O, reported honestly rather than re-engineered until it matched the initial hypothesis. `global_outlier_only` and `recency_only` both show NestedKV at 100% retention vs H2O's 0%.
- Gate/blend constants (`beta=3.0`, `tau=0.60`, `kappa=10.0`, prior `(0.4,0.4,0.2)`, `safeguard_alpha=0.20`) are the paper's own Appendix A defaults, not guessed.
- The paper's own RULER/LongBench/LooGLE/InfiniteBench/MMLU-Pro numbers (Qwen3, Llama-3.2 family, NVIDIA L20 GPUs) are the paper's — not reproduced here.

---

## v0.36.0

### New
- **CurDKV-adapted** (`method="curdkv"`) — value-aware leverage-score eviction via approximated CUR decomposition. Every eviction method already in the repo (H2O, SnapKV, TOVA, PyramidKV, Keyformer, MorphKV, KVzip, ...) scores a token using only its **key** side (attention-mass, norm, key-SVD projection, reconstruction reliance); CurDKV derives a **leverage score** from the joint key-and-value structure of the proxy attention output, so a token's value contribution — not just its key/attention profile — decides whether it survives. Two tokens with identical keys but different values now provably receive different retention scores, something no existing eviction method in the repo can do. Inspired by "Value-Guided KV Compression for LLMs via Approximated CUR Decomposition" (Sengupta, Chaudhary, Chakraborty; **NeurIPS 2025**, confirmed poster, arXiv:2509.15038) — documented as "CurDKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `CurDKVState`/`curdkv_update`/`curdkv_get_kv` (`veloxquant_mlx/quantizers/curdkv.py`) — modeled on H2O's per-head sliding state and eviction loop; the new `_leverage_scores` helper estimates row-leverage via an **energy-weighted** sum over the proxy attention-weighted value block's leading singular vectors.
  - `CurDKVKVCache` (`veloxquant_mlx/cache/curdkv_cache.py`) — mirrors `H2OKVCache`'s structure line-for-line; both prefill and decode go through the same eviction loop.
  - Config: `curdkv_budget` (512), `curdkv_n_sink` (4), `curdkv_rank_cap` (16).
  - 39 tests (23 quantizer + 16 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_curdkv.py`).

### Honest scope
- **Key-as-query proxy, not the true query vector** — the same limitation H2O/SnapKV/Keyformer/MorphKV/KVzip already document.
- **An SVD-based, energy-weighted leverage-score estimator, not the paper's own CUR sampling algorithm.** Weighting each singular direction by its own energy (rather than a hard top-k/bottom-(n−k) split) is load-bearing: a hard rank cutoff degenerates to uniform leverage whenever the retained rank reaches the block size, since the left singular vectors of a full-rank block then form a complete orthogonal basis and every row trivially has unit norm — silently erasing the magnitude signal this estimator exists to capture.
- **Newly-appended tokens are seeded with their own leverage score, not a flat 0** — unlike H2O's softmax weights (never exactly 0), CurDKV's leverage scores can legitimately be exactly 0 for a genuinely negligible-value token, so a flat-0 seed would let a negligible-value newcomer tie forever with an already-negligible survivor and let arrival order, not value, decide the outcome.
- **Mechanism observable:** on a planted geometry (near-identical keys, sharply divergent values), CurDKV retains value-relevant tokens preferentially in 8/8 trials across seeds, while H2O — given the identical keys — cannot tell the classes apart and evicts near-uniformly. The benchmark also reports, honestly, that CurDKV retains fewer value-irrelevant tokens than H2O on a "correlated" control geometry too — not the initially expected null result — attributed to H2O's own tie-break dynamics in this small-N regime, not overclaimed as general CurDKV dominance.
- The paper's headline numbers (up to 9.6% higher accuracy than SOTA baselines, up to 40% latency reduction under aggressive compression) are the paper's, on trained models — not reproduced here.

---

## v0.35.0

### New
- **KVTC-adapted** (`method="kvtc"`) — local PCA + DP-optimal per-component bit allocation + order-0 entropy coding. Palu/SVDq/SpectralQuant all use a **fixed** mixed-bit split (a hand-chosen top-25%/75% tier, or a binary signal/noise cutoff via participation ratio); KVTC computes a **dynamic-programming-optimal** integer bit-width per principal component under a hard total-bit budget — including exactly **0 bits** (dropping a component entirely) — then entropy-codes the resulting quantized codes. Neither the DP-optimal discrete allocation nor the entropy-coding stage existed anywhere in the repo before this. Inspired by "KV Cache Transform Coding for Compact Storage in LLM Inference" (NVIDIA, **ICLR 2026**, accepted poster, arXiv:2511.01815) — documented as "KVTC-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `dp_allocate_bits` (`veloxquant_mlx/allocators/kvtc_dp.py`) — DP over (component, cumulative budget), reusing `ratequant.py`'s analytic `D(v,b) = v·β^(-b)` distortion curve rather than inventing a new one.
  - `entropy_encode`/`entropy_decode` (`veloxquant_mlx/quantizers/_entropy_coding.py`) — dependency-free order-0 Huffman coder (stdlib `heapq`), lossless round-trip, code-table cost counted in byte accounting.
  - `kvtc_compress`/`kvtc_decompress` (`veloxquant_mlx/quantizers/kvtc.py`) — local per-sequence PCA (reusing `_quant_utils.py::_truncated_svd`, the same helper SVDq/Palu/GEAR share), no fixed-energy truncation (the DP allocator decides survivors), byte-accounting helpers `kvtc_fp16_bytes` (realized entropy-coded payload) and `kvtc_pre_entropy_bytes` (pre-entropy-coding size).
  - `KVTCKVCache` (`veloxquant_mlx/cache/kvtc_cache.py`) — fits the PCA basis and DP allocation once at prefill, reuses them unchanged for every decode step; compresses **both K and V** (mirrors Palu's scope, not SVDq's keys-only scope).
  - Config: `kvtc_bit_budget` (512, i.e. 4 bits/component at head_dim=128), `kvtc_bit_choices` ((0,1,2,3,4,6,8)), `kvtc_beta` (3.5, shared with `ratequant.py`).
  - 73 tests (32 allocator + 15 entropy coder + 12 quantizer + 14 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_kvtc.py`).

### Honest scope
- **Local (per-sequence) PCA, not the paper's pre-calibrated global basis** — the same "fit-locally, no calibration set" limitation SVDq/Palu already document.
- **The DP allocator optimizes an analytic distortion proxy, not a real-activation-fit rate-distortion model.** The DP itself is exact; the objective it minimizes is the repo's existing Gaussian-quantization distortion curve, not a curve fit on real LLM activation statistics.
- **Entropy coding is a real, measured, lossless order-0 Huffman coder** — not the paper's (possibly more sophisticated) scheme, and never the theoretical Shannon-entropy bound. Realized post-entropy-coding bytes (including the code table) are always what's reported.
- **Uniform-variance collapse (pinned):** with equal per-component variance and a contiguous `bit_choices` range, the DP-optimal allocation is exactly `floor(budget/n)` per component (remainder to the first components) — the same allocation a naive uniform splitter would produce. No other collapse (e.g. to SVDq's fixed 25/75 split) is claimed — the DP should and does *beat* that split whenever variance is non-uniform.
- **Mechanism observable = reconstruction MSE/cosine at a matched total byte budget.** On planted skewed-variance geometry, KVTC's DP allocator reaches mean MSE ≈0.027 vs ≈87.6 (fixed-uniform) and ≈84.4 (SVDq-fixed-split); on a flat (isotropic) control it is roughly competitive with the fixed-uniform baseline, not a dramatic win. Entropy-coding's realized gain is modest (≈0.15–0.50 across the sweep), reported plainly.
- **Not path-dependent** (contrast with the eviction family H2O/TOVA/MorphKV/KVzip): the PCA basis and DP allocation are fixed once at prefill and reused for every subsequent token — pinned by a determinism test.
- The paper's headline numbers (up to 20×, up to 40× in some regimes, under 1pp accuracy loss on LLaMA 3/Mistral NeMo/R1-Qwen2.5 1.5B–70B across AIME25/GSM8K/LiveCodeBench/LongBench/MATH-500/MMLU/Qasper/RULER) are the paper's, on trained models — not reproduced here.

---

## v0.34.0

### New
- **KVzip-adapted** (`method="kvzip"`) — context-reconstruction reliance retention. Keeps a constant-size cache by ranking stored tokens according to how much the model relies on them to **reconstruct its own context** — a *query-agnostic* importance profile computed once and reused across all future queries — then evicting the least-relied-upon pairs. This is a new eviction axis: every other proxy scorer ranks a token by the attention it receives *from a query* (cumulative/latest/windowed); KVzip ranks by reconstruction reliance. Inspired by "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction" (Kim et al., **NeurIPS 2025 Oral**, arXiv:2505.23416) — documented as "KVzip-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `KVzipKVCache` (`veloxquant_mlx/cache/kvzip_cache.py`); primitives in `veloxquant_mlx/quantizers/kvzip.py`: `kvzip_update` (reconstruction-reliance ranking + protected-sink eviction), `kvzip_get_kv`, byte helpers, and `_reconstruction_importance` (max proxy-attention over the reconstruction probe).
  - Config: `kvzip_budget` (512), `kvzip_n_sink` (4), `kvzip_probe` ("context"; **"latest" = TOVA-adapted**).
  - 32 tests (19 quantizer + 13 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_kvzip.py`).

### Honest scope
- **`kvzip_probe="latest"` collapses onto TOVA-adapted, bit-for-bit** (pinned by a test). **No H2O collapse is claimed** — KVzip recomputes reconstruction reliance from the live keep set each step, it never accumulates.
- **Key-as-reconstruction-probe proxy** (a cache never runs the model to reconstruct text), same substitution family as H2O/TOVA/MorphKV-adapted.
- **Mechanism observable = reconstruction-critical retention under a reconstruction shift.** Cumulative H2O retains ~0.017 of the reconstruction-critical region while the context probe retains ~0.609, beating the `probe="latest"` (TOVA) reference (~0.248); a flat control shows no advantage. Downstream perturbation reported as-is.
- The paper's accuracy/memory numbers (3–4× reduction, ~2× decode, negligible loss up to 170K on LLaMA3.1/Qwen2.5/Gemma3) are the paper's, on trained models — not reproduced here.

### Meta
- Funding links updated: the dead Buy Me a Coffee handle is replaced with **Ko-fi** and **Buy Me a Chai** across the README, landing page, and `.github/FUNDING.yml`.
- JOSS paper (`paper/joss/paper.md`) refreshed to the current 37-method suite and the token-eviction family.

---

## v0.33.0

### New
- **MorphKV-adapted** (`method="morphkv"`) — recent-window correlation retention. Keeps a constant-size cache by ranking stored tokens against the attention pattern of a **sliding window of recent tokens**, eliminating the "early-token bias" of cumulative (H2O) scoring — where tokens that were heavy hitters early crowd out what the model is *currently* attending to. Inspired by "Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs" (Ghadia et al., **ICML 2025**, arXiv:2503.00979) — documented as "MorphKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `MorphKVKVCache` (`veloxquant_mlx/cache/morphkv_cache.py`); primitives in `veloxquant_mlx/quantizers/morphkv.py`: `morphkv_update` (recent-window relevance ranking + protected-sink/recent eviction), `morphkv_get_kv`, byte helpers, and `_recent_relevance` (mean proxy-attention over the recent window).
  - Config: `morphkv_budget` (512), `morphkv_n_sink` (4), `morphkv_window` (8; **1 = TOVA-adapted**).
  - 32 tests (19 quantizer + 13 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_morphkv.py`).

### Honest scope
- **`morphkv_window=1` collapses onto TOVA-adapted, bit-for-bit** — with a single recent token the recent-relevance is just the latest key's attention over the keep set (the latest-token ranking); a test asserts the `window=1` kept set equals TOVA's. **No H2O collapse is claimed** — MorphKV recomputes from the live window each step, it never becomes H2O's cumulative-forever rule.
- **Constant-size, recomputed — not accumulated.** No cumulative score array is stored; retention is recomputed each step from the live keep set and the last `morphkv_window` keys.
- **Key-as-query proxy** (same as H2O/TOVA/Keyformer-adapted): the incoming key stands in for the unseen query.
- **Mechanism evidence is the recent-relevant retention rate.** Under a constructed topic shift, cumulative H2O scoring retains ~0% of the recent-relevant region (captured by stale early heavy hitters) while MorphKV re-targets toward it; the recent signal is made deliberately noisy so a wider window materially beats the `window=1` (TOVA) reference. A "stable" control shows no advantage. Downstream probe perturbation is a noisier secondary effect, reported as-is. The paper's accuracy/memory numbers are the paper's, on trained models — not reproduced. No RoPE remapping. Uniform budget/window across heads. No model-level perplexity/throughput benchmark — offline-synthetic only.

---

## v0.32.0

### New
- **Keyformer-adapted** (`method="keyformer"`) — Gumbel-regularized heavy-hitter eviction. Structurally H2O's proxy-attention accumulator plus **Gumbel noise** on the eviction logits, so a "late riser" (a token that reads low early, before the queries that attend to it arrive) is not deterministically pruned before it can recover. Inspired by "Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference" (Adnan et al., **MLSys 2024**, arXiv:2403.09054) — documented as "Keyformer-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `KeyformerKVCache` (`veloxquant_mlx/cache/keyformer_cache.py`); primitives in `veloxquant_mlx/quantizers/keyformer.py`: `keyformer_update` (additive proxy-attention accumulation + `score + tau·gumbel` eviction ranking), `keyformer_get_kv`, byte helpers, and a deterministic per-position Gumbel draw.
  - Config: `keyformer_budget` (512), `keyformer_n_sink` (4), `keyformer_recent` (0, extension), `keyformer_tau` (1.0; **0 = H2O-adapted**), `keyformer_seed` (0).
  - 29 tests (17 quantizer + 12 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_keyformer.py`).

### Honest scope
- **`keyformer_tau=0` collapses onto H2O-adapted, bit-for-bit** — the only thing Keyformer adds over H2O is the Gumbel regularizer, and a test asserts the `tau=0` kept set equals H2O's; the benchmark prints an `h2o` cross-check column.
- **Frozen per-position Gumbel, not the paper's annealed schedule.** The paper redraws Gumbel noise and anneals a temperature across generation; a cache has no trustworthy global step, so we draw one deterministic Gumbel value per token position (seeded by `keyformer_seed` + a per-head running position) and freeze it. Preserves the "don't doom a borderline token on one low reading" intent; not claimed equivalent to the schedule.
- **Key-as-query proxy** (same as H2O/SnapKV-adapted): the incoming key stands in for the unseen query.
- **Mechanism evidence is the survival rate.** Under constructed late-riser geometry, greedy `tau=0` evicts the planted riser 100% of the time while `tau=6` rescues it ~75% of the time; the downstream probe perturbation is a noisier, regime-dependent secondary effect, reported as-is. No RoPE remapping. Uniform budget/tau across heads. No model-level perplexity/throughput benchmark — offline-synthetic survival-rate, output-perturbation and byte-accounting only.

---

## v0.31.0

### New
- **Q-Filters-adapted** (`method="qfilters"`) — query-agnostic projection eviction, the library's fourth eviction scorer class (after attention/proxy, structural, and intrinsic-norm). Each cached key is scored by its projection onto a single frozen per-head direction; over budget, the highest-scoring tokens are kept (sinks and an optional recent window protected). Inspired by "Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression" (arXiv:2503.02812, **preprint**) — documented as "Q-Filters-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `QFiltersKVCache` (`veloxquant_mlx/cache/qfilters_cache.py`); primitives in `veloxquant_mlx/quantizers/qfilters.py`: `estimate_filter_dir` (top singular vector of the observed keys, frozen after `qfilters_calib_tokens`), `qfilters_update`/`qfilters_get_kv`, byte helpers (K+V fp16 plus the float32 filter direction).
  - Config: `qfilters_budget` (512), `qfilters_n_sink` (4), `qfilters_recent` (0, extension), `qfilters_calib_tokens` (128), `qfilters_sign` (1; -1 = inverted ablation).
  - 27 tests (12 quantizer + 15 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_qfilters.py`).

### Honest scope
- **The filter is key-SVD-derived, not query-SVD-derived.** The paper estimates the direction offline from a sample of query vectors; a cache-side library never sees queries, so we substitute the SVD of the first observed *keys*. This recovers the dominant *axis* but not which *end* is important — the sign a query would disambiguate. The committed benchmark shows the key-SVD recovering the planted axis (`filter_cosine ≈ 0.97`) while which raw sign arm wins flips row to row, so `qfilters_sign` is a **genuine ablation**. Nothing here is claimed equivalent to the paper's filter.
- **Path-dependent** (unlike L2Norm): prefill-in-one-block and token-by-token decode can freeze different filters and diverge; there is deliberately no prefill/decode bit-for-bit equivalence guarantee.
- Preprint, no venue. No RoPE remapping after eviction. Uniform budget across heads. `qfilters_recent` is an extension, off by default. No model-level perplexity/throughput benchmark — offline-synthetic output-perturbation and byte-accounting only.

---

## v0.30.1

### Fixed
- **PyPI package metadata only — no code changes.** PyPI mirrors such as pepy.tech showed no summary/version/license/author because the published metadata was malformed for downstream consumers: the Summary was a ~700-character method list (now a one-line summary), the License field embedded the entire MIT license text via `license = { file = "LICENSE" }` (now a PEP 639 SPDX expression, `License-Expression: MIT`), and the `Author:` field was empty (now populated alongside `Author-email:`). Wheel/sdist contents are otherwise identical to 0.30.0.

---

## v0.30.0

### New
- **SKVQ-adapted** (`method="skvq"`) — sliding-window quantization with two mechanisms new to the library: **channel reordering** (permute head-dim channels so channels of similar dynamic range share a quantization group — per-head permutations sorted by range, frozen from the first flushed chunk) and **clipped dynamic quantization** (each group's min/max window shrunk by a per-group grid-searched clip factor α, saturating a few extremes to buy finer resolution everywhere else; α=1 is always in the grid so the search never loses under its own metric). Both K and V quantized with per-token channel groups behind a sliding fp16 window (the NSNQuant chunk-flush idiom) with the paper's attention-sink filter (first `skvq_n_sink` tokens stay fp16). Inspired by "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models" (Duanmu, Yuan, Li, Duan, Zhang, Lin, COLM 2024, arXiv:2405.06219) — documented as "SKVQ-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `SKVQKVCache` (`veloxquant_mlx/cache/skvq_cache.py`); primitives in `veloxquant_mlx/quantizers/skvq.py`: `channel_permutation`, `invert_permutation`, `apply_permutation`, `clipped_group_quant`/`clipped_group_dequant` (vectorized per-group α search, α folded into the stored lo/scale — nothing extra kept), `skvq_round_trip`, byte helpers.
  - Prefill and token-by-token decode produce **bit-for-bit identical caches** (chunk boundaries, first-chunk permutation statistics, clip search, and sink restore are all functions of the same chunk contents — pinned by test). Deterministic end to end: no RNG anywhere.
  - Config: `skvq_bits_key`/`skvq_bits_value` (default 2/2), `skvq_group_size` (32), `skvq_window` (128), `skvq_n_sink` (5), `skvq_reorder` (True; False = identity ablation), `skvq_clip_search` (True) / `skvq_clip_alpha`, `skvq_max_ctx`. No coordinator — the default `KVCacheBuilder.for_model()` path returns one `SKVQKVCache` per layer.
  - 13 quantizer tests + 18 cache tests; `benchmark_scripts/benchmark_skvq.py` + committed `skvq_benchmark_results.json` — under a heterogeneous-channel regime (2.5-decade smooth scale spread), reordering cuts key MSE a further **16.9%** on top of clip search and collapses per-channel normalized error ~450×; clip search adds **14.0%** on top of reordering; under the homogeneous control reordering buys **−0.3%** (nothing), reported in full. The repo's KIVI reference wins several heterogeneous rows outright (its per-channel key scheme is intrinsically immune to channel heterogeneity) — reported as measured.

### Honest scope
- The paper's offline calibration (KMeans channel clustering on WikiText-2 + attention-output-MSE clip search, permutation fused into projection weights) is replaced by **first-chunk statistics** (sort by dynamic range; per-group reconstruction-MSE grid search) with an explicit runtime permute/inverse-permute — a documented adaptation, not the paper's pipeline.
- No 1.5-bit value packing and no FP8(E4M3) metadata (both CUDA artifacts); integer bit-widths and fp16 metadata, all counted in the byte accounting.
- That real transformer K/V exhibit the heterogeneous-channel regime is the paper's premise (shared with KIVI/KVQuant), not something the offline-synthetic benchmark can validate — the homogeneous control shows reordering buys nothing without it.
- No model-level (perplexity/throughput) benchmark run.

---

## v0.29.0

### New
- **L2Norm-adapted** (`method="knorm"`) — the repo's first **intrinsic-signal** eviction cache: token importance is read directly off the stored key vector's L2 norm, with the counterintuitive sign the paper reports in trained decoder LMs — **low norm ⇒ high future attention** — so the cache keeps the lowest-norm tokens. No attention scores, no key-as-query proxy (the approximation H2O/SnapKV/TOVA need), no structure-only recency rule: the paper's actual signal is fully observable at the cache level, making this the cleanest adaptation in the eviction family. L2Norm-adapted ("A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression", Devoto, Zhao, Scardapane, Minervini, EMNLP 2024, arXiv:2406.11430) — documented as "L2Norm-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `L2NormKVCache` (`veloxquant_mlx/cache/knorm_cache.py`); primitives in `veloxquant_mlx/quantizers/knorm.py`: `KnormState`, `init_knorm_state`, `knorm_update` (vectorized — one protected top-k per incoming block, no per-token softmax-over-cache loop), `knorm_get_kv`, `knorm_fp16_bytes`, `full_knorm_fp16_bytes`.
  - Because the score is intrinsic (computed once at insertion, never updated): eviction is **~100–800× faster than H2O-adapted** at prefill on the committed harness (0.3 ms vs 240 ms at S=1024), and with `knorm_recent=0` the kept set is **path-independent** — prefill-in-one-block and token-by-token decode produce bit-for-bit identical caches (the "keep k best with a heap" invariant, pinned by test at both the primitive and wrapper level). No accumulating-score method has this property.
  - Config: `knorm_budget` (default 512), `knorm_n_sink` (default 4), `knorm_recent` (default 0 — trailing protected window, an extension beyond the paper; enabling it breaks path independence), `knorm_keep` (`"low"` = paper finding | `"high"` = inverted ablation arm). No coordinator — the default `KVCacheBuilder.for_model()` path returns one `L2NormKVCache` per layer.
  - 10 quantizer tests + 14 cache tests, including the bit-for-bit path-independence check and a mechanism test under paper-like geometry; `benchmark_scripts/benchmark_knorm.py` + committed `knorm_benchmark_results.json` — under geometry constructed to exhibit the paper's correlation, keep-low beats random eviction by **+0.17** mean output perturbation and the inverted scorer by **+0.21**; under the isotropic control the advantage **reverses** (keep-low ~0.07 *worse* than random — softmax favors high-norm keys on isotropic Gaussians), reported in full.

### Honest scope
- The low-norm ⇒ high-attention correlation is the paper's **empirical claim about trained models** — the offline-synthetic benchmark validates the machinery under constructed geometry, not the correlation itself, and the isotropic control shows the method can underperform random eviction when that geometry is absent.
- No RoPE position-ID remapping after eviction; uniform budget and n_sink across heads (same as the rest of the eviction family); `knorm_recent` and `knorm_keep="high"` are extensions beyond the paper, both off by default.
- No model-level (perplexity/throughput) benchmark run.

---

## v0.28.0

### New
- **NSNQuant-adapted** (`method="nsnquant"`) — the repo's first **calibration-free distribution-matching VQ**: instead of fitting a codebook to the data (per-sequence k-means, EM) or using a data-independent geometric code (signs, polar grids), NSNQuant **reshapes the data to match a fixed code**. A Normalize-Shift-Normalize transform (token-norm → channel-mean shift → token-norm) plus a Hadamard rotation maps K/V tokens onto the standard normal distribution, so one codebook built offline from synthetic Gaussian samples — never from model activations — quantizes any model at 1–2 bits/element. NSNQuant-adapted ("NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache", Son, Choi, Yoo, NeurIPS 2025, arXiv:2505.18231) — documented as "NSNQuant-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `NSNQuantKVCache` (`veloxquant_mlx/cache/nsnquant_cache.py`) — single-layer wrapper, no coordinator, with a chunk-flush fp16 residual buffer (KIVI's idiom): every `nsn_residual_length` tokens flush through the pipeline as one self-contained chunk with its own online channel mean; prefill and decode produce identical quantized state by construction. Primitives in `veloxquant_mlx/quantizers/nsnquant.py`: `nsn_transform`, `nsn_inverse`, `build_universal_codebook`, `vq_encode`, `vq_decode`, `hadamard_forward`/`hadamard_inverse` (reusing `mx.hadamard_transform` via the repo's existing Hadamard infrastructure).
  - Config: `nsn_bits` (default 2: uint8 sign mask + uint8 index per 8-dim subvector = 2 bits/element; 1: index only), `nsn_residual_length` (default 64; paper suggests 128 for 1-bit), `nsn_codebook_size` (default 256), `nsn_subvector_dim` (default 8), `nsn_seed` (default 1234), `nsn_max_ctx` (default 8192). Both keys **and** values quantized, mirroring the paper (unlike the keys-only SVDq/xKV precedent).
  - 16 quantizer tests + 19 cache tests, including a mechanism-validation ablation (on channel-biased input the full NSN pipeline must beat the identical Hadamard+VQ without NSN by a pinned margin) and a prefill-vs-decode path-independence check; `benchmark_scripts/benchmark_nsn.py` + committed `nsn_benchmark_results.json` — NSN gains +0.038 (2-bit) / +0.110 (1-bit) reconstruction cosine over the no-NSN ablation at strong channel bias, honestly collapsing to ~+0.001–0.002 on already-centered input; 0.96–0.98 cosine at ~2.5 effective bits/element (metadata included), beating a KIVI-2bit baseline on every row of the sweep.
  - Honest scope: post-RoPE keys (the paper applies NSN pre-RoPE with a custom kernel — the central simplification of this adaptation), explicit value Hadamard (no projection-layer fusion), spherical-k-means-only codebook (no gradient fine-tune), fp16 metadata (~0.5 bits/element overhead vs the paper's double-quantized ~0.23), no fused kernels, no model-level perplexity/throughput benchmark — offline reconstruction-quality and byte-accounting numbers only.

---

## v0.27.0

### New
- **xKV-adapted** (`method="xkv"`) — the repo's **third cross-layer** mechanism, alongside XQuant (code reuse) and MiniCache (SLERP direction merge). A fixed-size contiguous group of layers jointly factorizes its stacked key matrices into **one shared SVD basis** via a fan-in/fan-out coordinator; every group member then stores only its own latent codes in that shared basis, amortizing the basis storage cost across the whole group. xKV-adapted ("xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction", Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang, Li, Ceze, Wu, Abdelfattah, arXiv:2503.18893, preprint) — documented as "xKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `XKVCache` (`veloxquant_mlx/cache/xkv_cache.py`); `XKVCoordinator` (`veloxquant_mlx/cache/xkv_coordinator.py`) — a fan-in-then-fan-out coordinator, distinct from XQuant/MiniCache's single-publisher pattern since the joint SVD needs every group member's keys before any of them can compress; primitives in `veloxquant_mlx/quantizers/xkv.py`: `pair_layers_grouped`, `joint_svd_compress`, `project_into_shared_basis`, `reconstruct_from_shared_basis`, `quantize_latents_uniform`.
  - Config: `xkv_group_size` (default 2), `xkv_rank` (default `None` -> energy-threshold selection), `xkv_energy_threshold` (default 0.95), `xkv_latent_bits` (default 4), `xkv_group_quant_size` (default 32), `xkv_max_ctx` (default 8192). Keys only — values pass through fp16 unchanged, mirroring SVDq's precedent.
  - 9 quantizer tests + 14 cache tests, including a group-of-1 degeneracy check (`joint_svd_compress` on a single matrix matches SVDq's plain single-layer SVD) and a mechanism-validation test (shared structure across synthetic layers reconstructs better than independent per-layer SVD on unrelated noise at matched rank); `benchmark_scripts/benchmark_xkv.py` + committed `xkv_benchmark_results.json` — sweeps group size (2–4) and a synthetic shared-structure knob, showing near-parity reconstruction MSE (within ~1%) and 8–20% fewer bytes than independent per-layer SVD, improving with larger group sizes.
  - Honest scope: fixed contiguous grouping (no CKA-based layer-alignment validation), no "Selective Reconstruction" decode-time optimization, single-bit-width latent quantization (not SVDq-style mixed-bit routing), no model-level perplexity/throughput benchmark — offline reconstruction-quality and byte-accounting numbers only.

---

## v0.26.0

### New
- **CaM-adapted** (`method="cam"`) — the repo's **eighth eviction configuration** and the first on the **merge-vs-drop** axis. Every other eviction method permanently discards the tokens it evicts; CaM instead **merges** each evicted token into the surviving token it most resembles (a cosine-weighted blend of the value rows, and optionally the keys), then removes only the redundant slot — so the information is folded into a neighbour rather than lost. The eviction *choice* is H2O's; only the disposition differs. With `cam_merge="drop"` it reduces **bit-for-bit** to H2O-adapted. CaM-adapted ("CaM: Cache Merging for Memory-efficient LLMs Inference", Zhang et al., ICML 2024, PMLR 235:58840-58850) — documented as "CaM-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `CaMKVCache` (`veloxquant_mlx/cache/cam_cache.py`); primitives in `veloxquant_mlx/quantizers/cam.py`: `most_similar_survivor`, `merge_pair`, `CaMState`, `init_cam_state`, `cam_update`, `cam_get_kv`, `cam_fp16_bytes`, `full_cam_fp16_bytes`.
  - Config: `cam_budget` (default 512), `cam_n_sink` (default 4), `cam_merge` (`"sim_weighted"` | `"mean"` | `"drop"`, default `"sim_weighted"`), `cam_merge_keys` (default False). No coordinator — each layer merges independently; the default `KVCacheBuilder.for_model()` path returns one `CaMKVCache` per layer.
  - 18 quantizer tests + 14 cache tests, including a bit-for-bit `cam_merge="drop"` == H2O equivalence (identical kept keys *and* values vs `H2OKVCache`) at both the primitive and cache level; `benchmark_scripts/benchmark_cam.py` + committed `cam_benchmark_results.json`.

### Honest scope
- Cosine-similarity merge weight rather than the paper's attention-prominence weight (which is ~0 for a just-appended token that overflows before accumulating mass — the common streaming case); single nearest-survivor merge (no multi-target soft assignment / sampling); key-as-query proxy; no RoPE remapping; uniform budget across heads.
- No model-level (perplexity/throughput) benchmark run. The offline harness measures output **perturbation** (cosine distance of the compressed-cache attention output vs the full-cache output over probe queries) against the H2O `drop` baseline; the measured finding is that `sim_weighted` merging reduces perturbation and the gain grows with compression ratio (e.g. 0.955 → 0.708 at `seq=1024, budget=64`, 16×), shrinking to ~0 at low compression where dropping barely hurts. Not an end-to-end task-quality claim.

---

## v0.25.0

### New
- **ChunkKV-adapted** (`method="chunkkv"`) — the repo's **seventh eviction configuration** and the first to evict at **chunk** rather than **token** granularity. The sequence is partitioned into contiguous chunks of `chunk_size` tokens; each chunk is kept or dropped as a whole, ranked by a mean-pooled per-token importance proxy (H2O cumulative attention mass, or key L2 norm). Keeping whole contiguous spans preserves local structure that token-level eviction shreds. When `chunk_size=1` it reduces **bit-for-bit** to H2O-adapted. ChunkKV-adapted (arXiv:2502.00299, Liu et al., 2025) — documented as "ChunkKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `ChunkKVCache` (`veloxquant_mlx/cache/chunkkv_cache.py`); primitives in `veloxquant_mlx/quantizers/chunkkv.py`: `chunk_partition`, `chunk_scores`, `chunkkv_keep_mask`, `ChunkKVState`, `init_chunkkv_state`, `chunkkv_update`, `chunkkv_trim_to`, `chunkkv_get_kv`, `chunkkv_fp16_bytes`, `full_chunkkv_fp16_bytes`.
  - Config: `chunkkv_budget` (default 512), `chunkkv_chunk_size` (default 8), `chunkkv_n_sink` (default 4), `chunkkv_score` (`"attn_mass"` | `"key_norm"`, default `"attn_mass"`). No coordinator — each layer resolves its own chunks; the default `KVCacheBuilder.for_model()` path returns one `ChunkKVCache` per layer.
  - 19 quantizer tests + 14 cache tests, including a bit-for-bit `chunk_size=1` == H2O equivalence at both the primitive and cache level; `benchmark_scripts/benchmark_chunkkv.py` + committed `chunkkv_benchmark_results.json` (offline-synthetic).

### Honest scope
- Pooled per-token score as a proxy for the paper's attention-over-chunk importance; no layer-wise kept-index reuse (each layer resolves chunks independently).
- Key-as-query proxy for the `attn_mass` scorer (same as H2O-adapted); no RoPE position-ID remapping after eviction; uniform budget across heads within a layer.
- Whole-chunk retention lets heads settle at slightly different counts — the wrapper trims every head to the common minimum so the emitted tensor is rectangular.
- No model-level (perplexity/throughput) benchmark run yet. The committed harness measures compression, kept-token count, and eviction latency on synthetic data; larger chunks cut the pure-Python eviction pass sharply (~12.7× fewer passes at `C=16` vs `C=1` on the `seq=1024, budget=128` shape) while holding compression. ChunkKV's semantic-coherence advantage is a real-attention property and is not claimed from the synthetic harness.

---

## v0.20.0

### New
- **StreamingLLM-adapted** (`method="streaming_llm"`) — the repo's first **constant-memory** cache and first **structural positional eviction** method. Keeps only the first `stream_n_sink` token positions (frozen attention sinks) and the most recent `stream_window_size` positions (rolling FIFO). All other positions are permanently evicted. Both prefill and decode tokens go through the same logic — the cache never exceeds `stream_n_sink + stream_window_size` positions regardless of generation length. StreamingLLM-adapted (arXiv:2309.17453, ICLR 2024, Xiao et al.) — positional eviction (no scoring, no calibration); documented as "StreamingLLM-adapted (VeloxQuant-MLX implementation)."
  - `StreamingLLMKVCache` (`veloxquant_mlx/cache/streaming_llm_cache.py`); primitives in `veloxquant_mlx/quantizers/streaming_llm.py`: `StreamingWindow`, `init_streaming_window`, `stream_update`, `stream_get_kv`, `stream_fp16_bytes`, `full_stream_fp16_bytes`.
  - Config: `stream_n_sink` (default 4), `stream_window_size` (default 512). Single-layer; `KVCacheBuilder.for_model()` propagates all `stream_*` fields via `dataclasses.replace`.
  - 17 quantizer tests + 15 cache tests; `benchmark_scripts/benchmark_streaming_llm.py` (offline-synthetic, not run).

### Honest scope
- No attention mask adjustment: the model attends to all returned K/V positions; only the number of K/V rows is bounded.
- No RoPE position-ID remapping: original token positions are preserved in returned rows.
- Fixed `stream_n_sink` count — not adaptive.
- No model-level benchmark run yet; streaming_ratio and constant-memory property verified on synthetic data (32/32 tests passing).

---

## v0.19.0

### New
- **SnapKV-adapted** (`method="snapkv"`) — the repo's first **token eviction** method and the first where the paper's actual signal (observation-window attention scores) is computable at the cache level without model interception. During prefill, the last `snap_obs_window` key rows act as proxy queries; their softmax attention over all prefix positions scores each token. Only the top-`snap_budget` positions (plus `snap_n_sink` always-kept sink positions) are retained as fp16. Decode tokens are never evicted. SnapKV-adapted (arXiv:2404.14469, ICLR 2025, Yuan et al.) — key-as-query proxy and no max-pool smoothing; documented as "SnapKV-adapted (VeloxQuant-MLX implementation)."
  - `SnapKVKVCache` (`veloxquant_mlx/cache/snapkv_cache.py`); primitives in `veloxquant_mlx/quantizers/snapkv.py`: `obs_window_attention_scores`, `snap_select_indices`, `snapkv_compress`, `snapkv_fp16_bytes`, `full_fp16_bytes`.
  - Config: `snap_budget`, `snap_obs_window`, `snap_n_sink`. Single-layer; `KVCacheBuilder.for_model()` propagates all `snap_*` fields via `dataclasses.replace`.
  - 18 quantizer tests + 12 cache tests; `benchmark_scripts/benchmark_snapkv.py` (offline-synthetic, not run).
  - Single-layer (no coordinator); eviction is per-head, uniform budget.

### Honest scope
- The key-as-query proxy is weaker than true query vectors from the prompt (not observable at `update_and_fetch`). Still stronger than key-norm-only methods (computes the actual attention distribution from K).
- No max-pool smoothing (paper's `kernel_size > 1`).
- Uniform `snap_budget` across all heads.
- No model-level benchmark run yet; eviction ratio and attention-coverage lift verified on synthetic data.

---

## v0.18.0

### New
- **ZipCache-adapted** (`method="zipcache"`) — the repo's first **per-token mixed bit-width** cache. The top `hi_fraction` of tokens by key L2-norm (the saliency proxy) are quantized at `hi_bits`; the rest at `lo_bits`. Both groups remain quantized — not fp16. ZipCache-adapted (arXiv:2405.14256, NeurIPS 2024, He et al.): the paper's true signal is normalized attention scores, which are not observable by a cache wrapper; key L2-norm is the proxy (same as KIVI-Sink and AdaKV-proxy, but here the decision is bit-width routing rather than fp16 protection or head budgeting).
  - `ZipCacheKVCache` (`veloxquant_mlx/cache/zipcache_cache.py`); primitives in `veloxquant_mlx/quantizers/zipcache.py`: `token_key_norms`, `saliency_mask`, `channel_quant`, `channel_dequant`, `zipcache_compress`, `zipcache_reconstruct`, `zipcache_bytes`, `base_only_bytes`, `zipcache_quant_dequant`.
  - Config: `zipcache_hi_bits`, `zipcache_lo_bits`, `zipcache_hi_fraction`, `zipcache_group_size`, `zipcache_quantize_values`.
  - 16 quantizer tests + 11 cache tests; `benchmark_scripts/benchmark_zipcache.py` (offline-synthetic, not run).
  - Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all `zipcache_*` fields via `dataclasses.replace`.

### Honest scope
- The saliency proxy (key L2-norm) is weaker than true attention scores. This is the third use of the key-norm proxy in this repo; each prior use is on a different decision (KIVI-Sink: fp16 protection; AdaKV-proxy: head budget).
- The effective average key rate is `hi_frac×hi_bits + (1-hi_frac)×lo_bits` — between `lo_bits` and `hi_bits`, as expected.
- No model-level benchmark run yet; stored bytes and reconstruction MSE are test-verified on synthetic data.

---

## v0.17.0

### New
- **GEAR** (`method="gear"`) — the repo's first **error-feedback** KV cache. Every other method picks a bit-width or a cache layout and lives with the quantization error; GEAR makes *any* ultra-low-bit base quantizer near-lossless by reconstructing what it threw away, via the three-part decomposition `X ~= Quant_b(X) + L·R + S`: an ultra-low-bit base quant, a **low-rank** approximation of the quantization *residual* `E = X - dequant(Quant_b(X))`, and a **sparse** matrix correcting the top-magnitude outlier entries the low-rank term cannot absorb. Unlike CacheGen (reconstruction identical to group quant), GEAR's reconstruction genuinely **recovers quality** the base bit-width loses. GEAR-adapted (arXiv:2403.05527, Kang et al.): the residual SVD is computed per `update_and_fetch` call (reusing the SVDq/PALU prefill-SVD pattern) and GEAR's fused dequant CUDA kernel is not ported — we reconstruct fp16 then call MLX SDPA, so stored size shrinks but attend-time peak memory does not.
  - `GEARKVCache` (`veloxquant_mlx/cache/gear_cache.py`); primitives in `veloxquant_mlx/quantizers/gear.py`: `quantize_base`, `residual`, `lowrank_error`, `sparse_outliers`, `gear_compress`, `gear_reconstruct`, `gear_bytes`, `base_only_bytes`, `gear_quant_dequant`. The base quant is borrowed from CacheGen and the truncated-SVD helper (`_quant_utils._truncated_svd`) is shared with SVDq/PALU.
  - Config: `gear_bits`, `gear_rank`, `gear_energy_threshold`, `gear_sparse_fraction`, `gear_group_size`, `gear_quantize_values`
  - 10 cache tests + 13 quantizer tests; `benchmark_scripts/benchmark_gear.py` (offline-synthetic, not run)
  - Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates the `gear_*` fields automatically via `dataclasses.replace`.

### Honest scope
- GEAR's **stored** cache (base codes + low-rank factors + sparse triples) shrinks, but the working set *during* attention is the reconstructed fp16 K/V — attend-time peak memory is not reduced. The low-rank factors and sparse triples are overhead, so the rank must be genuinely low relative to the head dim (the GEAR premise); the overhead is reported honestly and never hidden.
- Quality evidence is unit-test level (synthetic low-rank-plus-outlier data); no model-level benchmark run yet.

---

## v0.16.0

### New
- **CacheGen** (`method="cachegen"`) — the repo's first **entropy-coded** KV cache. Every other method packs codes at a fixed bit-width; CacheGen exploits token-wise locality (adjacent tokens' KV are similar) by applying a reversible token-delta transform to the quantized codes and compressing the low-entropy residual stream toward its Shannon entropy. Reconstruction is identical to plain group quant (lossless over the codes); the contribution is the storage accounting. CacheGen-adapted (arXiv:2310.07240, SIGCOMM 2024): rather than ship a serial range codec that would bottleneck MLX decode, the entropy-coded byte size is modelled from the measured symbol entropy and capped at the fixed-width packed size, so savings are never negative (exactly 0% on incompressible iid data, ~10–17% on correlated data).
  - `CacheGenKVCache` (`veloxquant_mlx/cache/cachegen_cache.py`); primitives in `veloxquant_mlx/quantizers/cachegen.py`: `quantize_to_codes`, `dequant_codes`, `token_delta`, `symbol_entropy_bits`, `entropy_coded_bytes`, `fixed_width_bytes`, `cachegen_quant_dequant`
  - Config: `cachegen_bits`, `cachegen_group_size`, `cachegen_use_delta`
  - 12 cache tests + 9 quantizer tests; `benchmark_scripts/benchmark_cachegen.py` (not run)
- **MiniCache** (`method="minicache"`) — cross-layer compression in the **depth dimension**. Adjacent middle-to-deep layers have nearly identical KV directions, so a pair is merged into one shared SLERP-interpolated direction plus each layer's own per-token magnitude (a pair costs ~one layer). High-divergence token pairs are kept unmerged (the retention set). A different route to inter-layer redundancy than [XQuant](algorithms/xquant) — XQuant reuses quantized *codes*, MiniCache merges the *tensors* via spherical interpolation. MiniCache-adapted (arXiv:2405.14366, NeurIPS 2024): faithful to the magnitude/direction SLERP + token retention, integrated via a shared `MiniCacheCoordinator` (the XQuant pattern) rather than a modified attention forward.
  - `MiniCacheKVCache` (`veloxquant_mlx/cache/minicache_cache.py`), `MiniCacheCoordinator` (`veloxquant_mlx/cache/minicache_coordinator.py`); primitives in `veloxquant_mlx/quantizers/minicache.py`: `pair_layers_depth`, `to_mag_dir`, `slerp`, `merge_pair`, `reconstruct_layer`, `merge_similarity`
  - Config: `minicache_start_frac`, `minicache_group_size`, `minicache_retention_threshold`, `minicache_slerp_t`, `minicache_max_ctx`
  - 11 cache tests + 11 quantizer tests; `benchmark_scripts/benchmark_minicache.py` (not run)
  - Requires `KVCacheBuilder.for_model()` for the shared coordinator; a single factory-built cache is a degenerate lossless-passthrough primary.

### Honest scope
- Both are **storage**-compression methods: CacheGen's entropy coding and MiniCache's merge both reduce stored cache size but reconstruct fp16 for SDPA, so neither reduces working-set memory at attend time. On Apple Silicon's bandwidth-bound decode they are lower-leverage than the low-rank (PALU/SVDq) and quantization methods.
- Quality evidence is unit-test level (synthetic data); no model-level benchmark run yet.

---

## v0.15.0

### New
- **PALU** (`method="palu"`) — true low-rank latent storage for **both keys and values**, the repo's first method where the cache itself stays low-rank rather than reconstructing full fp16 for storage. At prefill it partitions heads into `palu_n_head_groups` groups, fits one shared projection per group via group-head SVD (G-LRD), and stores the projected codes `[S, r]` directly; full fp16 K/V is reconstructed only at attend time. Latents are mixed-bit quantized (top-25% of channels by singular value at 4-bit, the rest at 2-bit) for a full-KV effective rate below 1 bit/element on low-rank data. Unlike [SVDq](algorithms/svdq) — keys-only, reconstructs full fp16 and so wins on bandwidth accounting — PALU bypasses the parent fp16 ring buffer entirely (the storage win is real). Zero calibration. A PALU-adapted (arXiv:2407.21118, ICLR 2025) implementation: we fit projections from the prefill batch instead of an offline calibration set, and we do **not** port PALU's fused low-rank-reconstruction attention kernel (we reconstruct then call MLX SDPA), so peak memory during attention is not reduced — only stored cache size.
- `PALUKVCache` — new cache wrapper in `veloxquant_mlx/cache/palu_cache.py` (true latent storage; parent fp16 buffer bypassed, own offset bookkeeping)
- PALU primitives in `veloxquant_mlx/quantizers/palu.py`: `head_group_bounds()`, `group_head_svd()`, `project_to_latent()`, `reconstruct_from_latent()`, `quantize_latent()` (reuses the SVDq mixed-bit latent coder)
- New `KVCacheConfig` fields: `palu_rank`, `palu_energy_threshold`, `palu_n_head_groups`, `palu_hi_bit`, `palu_lo_bit`, `palu_hi_fraction`, `palu_group_size`, `palu_quantize_values`
- 13 tests in `tests/cache/test_palu_cache.py` + 9 in `tests/quantizers/test_palu.py`: factory dispatch, no-bits-leak, group projections stored, shape (prefill + decode), **latent-storage assertion** (buffers hold `[S, r]`, parent `keys is None`), PALU-beats-naive-2bit on both K and V, decode accumulation + offset growth, both-tensors-compressed accounting, low-rank-only values, sub-2-bit effective rate, energy-threshold rank, head-grouping, group SVD subspace recovery, determinism
- `benchmark_scripts/benchmark_palu.py` — throughput + memory sweep vs SVDq, KIVI, fp16, plus an offline full-KV reconstruction-MSE harness (PALU vs naive 2-bit on low-rank K and V)

### Fixed
- `KVCacheBuilder.for_model()` now propagates **all** method-specific config fields (`svdq_*`, `kitty_*`, `kvquant_*`, `palu_*`, …) to each per-layer cache via `dataclasses.replace`. Previously it rebuilt the per-layer config field by field and silently dropped method hyperparameters, so methods built through `for_model` fell back to defaults regardless of what the user passed.

---

## v0.14.0

### New
- **KVQuant-NUQ** (`method="kvquant"`) — non-uniform quantization datatype plus dense/sparse outlier isolation, the repo's first method that places quantization levels by the data distribution rather than uniformly. For each group it fits `2^bits` signpost levels via online 1-D Lloyd-Max (k-means), and carves the top-magnitude `outlier_fraction` of elements out to an fp16 sparse side-channel so a handful of outliers cannot stretch the level range. Keys are quantized per-channel (levels frozen after prefill), values per-token. At equal bit-width this strictly reduces reconstruction error on non-uniform K/V — measured ~73% lower MSE than uniform at 3-bit on Laplacian data. Zero calibration. A faithful adaptation of KVQuant (arXiv:2401.18079, NeurIPS 2024): we implement the two cache-observable pillars (NUQ + dense/sparse) and document the third (pre-RoPE key quantization, which needs a model-forward hook) as out of scope.
- `KVQuantKVCache` — new cache wrapper in `veloxquant_mlx/cache/kvquant_cache.py`
- NUQ utilities in `veloxquant_mlx/quantizers/kvquant.py`: `fit_nuq_levels()` (Lloyd-Max), `quantize_nuq()`, `dequant_nuq()`, `split_dense_sparse()` (outlier isolation), `nuq_quant_dequant()` (drop-in for `_group_quant_dequant`), `nuq_distortion()`
- New `KVCacheConfig` fields: `kvquant_bits`, `kvquant_outlier_fraction`, `kvquant_group_size`, `kvquant_lloyd_iters`, `kvquant_refit_interval`
- 15 new tests in `tests/cache/test_kvquant_cache.py`: factory dispatch, shape (prefill + decode), value reconstruction, NUQ-beats-uniform on non-uniform data, NUQ-not-worse on uniform data, Lloyd-Max monotone convergence, top-k outlier selection, outlier isolation lowers MSE, `outlier_fraction=0` pure-NUQ, level-table determinism, frozen-key-levels decode, byte accounting, effective-bits range, per-channel/per-token axis correctness, determinism
- `benchmark_scripts/benchmark_kvquant.py` — throughput + memory sweep over `bits ∈ {2,3}` and an outlier ablation vs KIVI (uniform), SVDq, fp16, plus offline NUQ-vs-uniform reconstruction MSE

---

## v0.13.0

### New
- **XQuant** (`method="xquant"`) — cross-layer KV cache reuse, the repo's first method to exploit *inter-layer* redundancy. Adjacent attention layers are grouped into anchor/reuse pairs: the anchor quantizes K/V with KIVI-style group quantization and publishes its integer codes through a shared coordinator; reuse layers borrow those codes and store only their own per-group scale/zero (+ optional low-bit residual), correcting the small cross-layer drift. Drives effective per-element key bits below 1.4 on correlated models (11–16× key bandwidth reduction across a group). Both keys and values compressed; zero calibration. A faithful adaptation of XQuant (arXiv:2510.11236, EMNLP 2025): the paper couples layers in a modified attention forward pass; we coordinate through a shared object so `mlx_lm.generate` stays untouched.
- `XQuantKVCache` — new cache wrapper in `veloxquant_mlx/cache/xquant_cache.py` with anchor/reuse role dispatch
- `XQuantCoordinator` — shared cross-layer code store in `veloxquant_mlx/cache/xquant_coordinator.py`, injected by `KVCacheBuilder.for_model()`
- XQuant utilities in `veloxquant_mlx/quantizers/xquant.py`: `pair_layers()`, `quantize_codes()`, `compute_reuse_params()`, `dequant_with_params()`, `quantize_residual()`, `cross_layer_similarity()`
- New `KVCacheConfig` fields: `xquant_group_size`, `xquant_base_bits`, `xquant_residual_bits`, `xquant_group_quant_size`, `xquant_max_ctx`
- `KVCacheBuilder.for_model()` now builds one shared coordinator and assigns anchor/reuse roles for `method="xquant"` (other methods unchanged)
- 16 new tests in `tests/cache/test_xquant_cache.py`: factory dispatch, `for_model` pairing, coordinator round-trip, anchor/reuse shape (prefill + decode), value reconstruction, residual-0 tolerance, residual lowers MSE, correlated near-self-quant, uncorrelated residual recovery (negative control), byte accounting, effective-bits, decode synchronization, token-budget guard, `group_size=3`, determinism
- `benchmark_scripts/benchmark_xquant.py` — throughput + memory sweep over `group_size ∈ {2,3}`, `residual_bits ∈ {0,1}` vs KIVI-2bit, SVDq-1.25bit, fp16, plus measured cross-layer key similarity

---

## v0.12.0

### New
- **AdaKV-proxy** (`method="adakv"`) — per-head adaptive bit allocation layered on KIVI-style group quantization. Ranks attention heads by online inter-token key-norm variance (an attention-free proxy for head importance), then solves a per-head bit budget so the average bits/element matches a configured target — high-importance heads get more bits, low-importance heads fewer. Zero calibration; values left at fp16. A *proxy* adaptation of Ada-KV (arXiv:2407.11550): true Ada-KV adapts the per-head *eviction* budget from softmax attention weights, which live outside the cache contract; we adapt the per-head *bit* budget instead.
- `AdaKVCache` — new cache wrapper in `veloxquant_mlx/cache/adakv_cache.py`
- AdaKV utilities in `veloxquant_mlx/quantizers/adakv.py`: `compute_head_norm_variance()`, `allocate_head_bits()` (budget allocator with greedy round-trip correction), `quantize_head()`
- New `KVCacheConfig` fields: `adakv_target_avg_bits`, `adakv_lo_bit`, `adakv_mid_bit`, `adakv_hi_bit`, `adakv_group_size`, `adakv_update_interval`
- 14 new tests in `tests/cache/test_adakv_cache.py`: factory dispatch, shape preservation (prefill + decode), values unchanged, high-importance heads get more bits, average bits matches target, equal-importance uniform degradation, lower MSE than lo_bit on the high-importance head, running norm-accumulator correctness, decode accumulation, byte accounting, avg_bits range, single-head trivial allocation, determinism
- `benchmark_scripts/benchmark_adakv.py` — throughput + memory sweep over `target_avg_bits ∈ {2.0, 2.5, 3.0}` vs KIVI-2bit, Kitty-2.5bit, fp16

---

## v0.11.0

### New
- **Kitty** (`method="kitty"`) — dynamic channel-wise mixed-precision key quantization. Ranks key channels by online per-channel variance at every step; top-25% channels get 4-bit, remaining 75% get 2-bit asymmetric group quantization. Achieves ~2.5-bit effective key precision (6.4× bandwidth reduction vs fp16). Zero calibration — no SVD, no codebook training, works on any model immediately. Values left at fp16. Inspired by Kitty (arXiv:2511.18643).
- `KittyKVCache` — new cache wrapper in `veloxquant_mlx/cache/kitty_cache.py`
- Kitty utilities in `veloxquant_mlx/quantizers/kitty.py`: `rank_channels_by_sensitivity()`, `quantize_mixed_channels()`, `compute_running_variance()`
- `veloxquant_mlx/quantizers/_quant_utils.py` — shared `_group_quant_dequant` helper extracted from `svdq.py` (no behavior change; both quantizers import from here)
- New `KVCacheConfig` fields: `kitty_hi_fraction`, `kitty_hi_bit`, `kitty_lo_bit`, `kitty_group_size`
- 14 new tests in `tests/cache/test_kitty_cache.py`: factory dispatch, shape preservation (prefill + decode), values unchanged, channel ranking correctness, hi-channel lower error than lo-channel, MSE vs uniform 2-bit on high-variance data, running variance accumulator, decode accumulation, byte accounting, avg_bits range, hi_fraction boundary cases, determinism
- `benchmark_scripts/benchmark_kitty.py` — throughput + memory sweep vs KIVI-2bit, SVDq-1.25bit, fp16

---

## v0.10.0

### New
- **SVDq** (`method="svdq"`) — sub-2-bit key compression via offline SVD + mixed-precision latent coding. Computes a truncated SVD of the prefill key matrix once, projects all keys into the low-rank latent space, and applies 4-bit / 2-bit mixed quantization ordered by singular value magnitude. Achieves ~1.25-bit effective key precision (12.8× bandwidth reduction vs fp16). Values left at fp16. Inspired by SVDq (arXiv:2502.15304).
- `SVDqKVCache` — new cache wrapper in `veloxquant_mlx/cache/svdq_cache.py`
- SVD utilities in `veloxquant_mlx/quantizers/svdq.py`: `svd_compress_keys()`, `quantize_latents_mixed()`, `reconstruct_keys()`
- New `KVCacheConfig` fields: `svdq_rank`, `svdq_energy_threshold`, `svdq_hi_bit`, `svdq_lo_bit`, `svdq_hi_fraction`, `svdq_group_size`
- 12 new tests in `tests/cache/test_svdq_cache.py`: SVD projection correctness, shape preservation, MSE vs naive 2-bit on low-rank data, decode accumulation, byte accounting, sub-2-bit effective bit-width, energy threshold rank selection, determinism

---

## v0.9.0

### New
- **KIVI-Sink** (`method="kivi_sink"`) — attention sink protection layered on KIVI group quantization. Tokens with anomalously high key L2-norm are kept in fp16 and excluded from quantization-parameter calibration, preventing sink outliers from inflating group scale and degrading neighboring tokens. Inspired by KVSink (Su & Yuan, COLM 2025).
- `SinkProtectedKVCache` — new cache wrapper in `veloxquant_mlx.cache.sink_cache`
- `KVCacheConfig.n_sink_tokens` — new field (default 5). Composes with KIVI's `residual_length`; byte accounting tracks `sink_fp16_bytes` separately with no double-counting. `n_sink_tokens=0` reproduces plain KIVI bit-for-bit.
- 9 new tests in `tests/cache/test_sink_cache.py`: sink detection, fp16 preservation, MSE improvement over plain KIVI, accounting partition, determinism. Full suite: 344/348 passing.

---

## v0.8.0

### New
- **KIVI** (`method="kivi"`) — tuning-free asymmetric 2-bit group quantization (Liu, Yuan et al., ICML 2024). Per-channel keys, per-token values; no codebook training, no rotation.
- `KIVIQuantizer` — registered as `"kivi"` in `QuantizerRegistry`
- `KIVIKVCache` — mlx_lm `update_and_fetch` wrapper with fp16 residual window (`residual_length`) and full byte-accounting
- `KVCacheConfig.kivi_group_size` — new field (default 32)
- Benchmark results on Llama-3.2-3B, Qwen2.5-7B, Mistral-7B (Apple M4): **KIVI-2bit ≈ 5.8× key / ≈ 4× full-KV at 100–106% of fp16 throughput**
- 25 new tests; 334/339 passing

---

## v0.7.0

### New
- **RaBitQ** — randomised Hadamard + 1-bit sign packing with IVF clustering for extreme key compression
- **SpectralQuant** — eigenvector-rotated quantization with signal/noise codebooks and water-filling bit allocation
- **CommVQ** — RoPE-commutative residual VQ for exact positional encoding compatibility
- `SpectralQuantKVCache`, `PolarQuantKVCache` — new cache wrappers
- `calibrate_spectral_rotation()`, `save_rotations()`, `load_cached_rotations()`
- `compute_participation_ratio()`, `compute_spectral_gap()`
- `water_fill_bits()` — per-dimension water-filling allocator
- `rabitq_hamming_score` — Metal XOR+popcount Hamming distance kernel
- `comm_vq_decode_metal` — fused centroid gather + RoPE Metal kernel
- 212+ passing tests

### Changed
- `KVCacheConfig` gains `signal_bits`, `noise_bits`, `rotations` fields for SpectralQuant
- `KVCacheFactory` and `KVCacheBuilder` updated for all new cache types

---

## v0.6.0

### New
- **PolarQuant** — recursive polar coordinate decomposition for spherical key distributions
- `PolarQuantizer`, `PolarQuantKVCache`
- `CommVQQuantizer` — first version (flat codebook, no Metal fusion yet)
- `TurboQuantProdAdaptive` — distortion-driven dynamic bit allocation

### Changed
- `CompositeQuantizer` — supports arbitrary-depth chains; cycle detection via `CyclicPipelineError`

---

## v0.5.1

### New
- **Metal GPU kernels for VecInfer** — hand-written Metal Shading Language shaders replacing pure-MLX hot paths
  - `vecinfer_quantize_metal` — fused nearest-centroid argmin, **13× speedup, 98% peak-memory reduction**
  - `vecinfer_dequant_metal` — bit-exact drop-in for `dequantize_vq`
  - `metal_available()` — capability probe
- `KVCacheConfig.use_metal_kernels` — three-state flag (`None` = auto-detect, `True` = require, `False` = force MLX)
- `VecInferKVCache` now dispatches to Metal kernels when available (zero API change)
- 7 new parity tests in `tests/cache/test_vecinfer_metal_parity.py`

---

## v0.5.0

### New
- **VecInfer** — product VQ with outlier-suppressing dual transform
  - `calibrate_smooth_factors()` — per-channel `λᵢ = √max|Kᵢ|`
  - `walsh_hadamard_matrix()`, `apply_dual_transform_keys/queries()`
  - `train_codebook()`, `quantize_vq()`, `dequantize_vq()`
  - `compute_query_lut()` — fused-score fast path
- `VecInferKVCache` — mlx_lm-compatible cache with `update_and_fetch`
- **Benchmarks**: 8× key compression at 2-bit, 16× at 1-bit on Llama-3.2-1B/3B

### Notes
- Throughput trades slightly vs fp16 (CUDA kernel fusion not available on Metal at this version)

---

## v0.3.6

### Breaking change
- **Package renamed**: `mlx_kv_quant` → `veloxquant_mlx`
- All imports must be updated: `from mlx_kv_quant import ...` → `from veloxquant_mlx import ...`
- No backward-compatibility shim

---

## v0.3.5

### New
- **RateQuant** becomes a first-class feature
  - `allocate_bits_ratequant()` — reverse-waterfilling allocator (arxiv:2605.06675)
  - `calibrate_layer_sensitivities()` — activation-norm sensitivity probe (1.6s)
  - `fit_distortion_curve()` — fits `D(b) = α·β^(-b)` per layer
- `TurboQuantRVQKVCache` — mlx_lm-compatible cache wrapper for RVQ
- `KeyNormObserver`, `KeyNormReport` — per-token key norm tracking
- `KVCacheConfig.bit_width_inlier` accepts `list[int]` for per-layer allocation
- 27 new tests (187 total passing)

### Results (M4 24 GB)

| Model | fp16 PPL | RVQ 1-bit | RateQuant 1.5-bit | Compression |
|---|---|---|---|---|
| Falcon3 7B | 22.9 | 23.1 | 22.8 | 5.22× |
| Gemma3 4B | 39.8 | 37.8 | 36.3 | 5.22× |

---

## v0.3.0

### New
- **QJL** — Johnson-Lindenstrauss 1-bit sign sketch cache
- `QJLQuantizer`, `QJLKVCache`
- `qjl_encode`, `qjl_inner_product` Metal kernels
- `DistortionObserver` — cosine similarity and IP error tracking
- `LatencyObserver` — encode/decode timing profiling
- `MemoryObserver` — peak memory and compression ratio

---

## v0.2.0

### New
- **TurboQuant RVQ** — two-pass residual VQ with Gaussian + Laplacian codebooks
- `TurboQuantRVQ` quantizer with Walsh-Hadamard preprocessing
- `turboquant_scalar_quantize`, `turboquant_hadamard_quantize` Metal kernels
- `turboquant_bit_pack`, `turboquant_bit_unpack` — sub-byte packing
- `KVCacheConfig`, `KVCacheFactory`, `KVCacheBuilder` — unified configuration API
- `NpyArtifactStore`, `MemoryArtifactStore` — artifact persistence
- `QuantizerRegistry` — plugin registration

---

## v0.1.0

### Initial release
- Core abstractions: `Quantizer`, `KVCache`, `Preconditioner`, `Codebook` ABCs
- `TurboQuantMSE` — MSE-optimal rotation + Lloyd-Max scalar quantization
- `ScalarCodebook`, `AdaptiveScalarCodebook`
- `RotationPreconditioner`, `JLSketchPreconditioner`
- `RingBuffer`, `AVLTree`, `BitPackBuffer` data structures
- Basic test suite (48 tests)

---

*Full commit-level history: [GitHub Commits](https://github.com/rajveer43/veloxquant-mlx/commits/master)*

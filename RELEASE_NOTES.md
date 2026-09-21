# Release Notes

Terse, per-version summary generated from `CHANGELOG.md` Conventional Commit subjects -- see `scripts/generate_release_notes.py`. For full detail (why a change was made, all commits/PRs), follow the link to the matching `CHANGELOG.md` section.

## v0.91.0 (2026-09-21)

- **planning**: Implement automatic KV cache strategy selection (RFC #469)
- **test**: Make hardware detection test resilient to paravirtual CI environment
- Add comprehensive docstring documentation for automatic strategy selection
- **readme**: Add total PyPI downloads badge
- **lint**: Apply ruff formatting to documentation docstrings

[Full changelog entry](CHANGELOG.md#v0910-2026-09-21)

## v0.90.8 (2026-09-20)

- **cache**: Don't drop window_size bound at N==1 in eviction mask
- **a2ats**: Skip redundant codebook float32 re-cast in decode hot path
- **adakv**: Skip redundant mx.eval() in per-token norm accumulator update
- **age-tiered**: Stop recomputing tier assignment twice per decode step
- **amc**: Hoist per-tier (rank, bits) config lookup out of the per-token loop
- **anchorkv**: Build ResidualCodec once instead of once per head at prefill

[Full changelog entry](CHANGELOG.md#v0908-2026-09-20)

## v0.90.7 (2026-09-20)

- **block-pool**: Maintain fragmentation count incrementally instead of O(n_blocks) rescan
- **quantizers**: Cache RoPE inv_freq in CommVQ instead of rebuilding per decode

[Full changelog entry](CHANGELOG.md#v0907-2026-09-20)

## v0.90.6 (2026-09-20)

- **quantized-linear**: Cache dequantized weight instead of rebuilding every forward call

[Full changelog entry](CHANGELOG.md#v0906-2026-09-20)

## v0.90.5 (2026-09-20)

- **spectral-quant**: Keep encode/decode/estimate_inner_product on-device

[Full changelog entry](CHANGELOG.md#v0905-2026-09-20)

## v0.90.4 (2026-09-20)

- **scalar-attend**: Cache tiny param buffers, clarify qjl_encode dot loop
- **scalar-attend**: Extend nsg autotune to batched + predecoded kernels

[Full changelog entry](CHANGELOG.md#v0904-2026-09-20)

## v0.90.3 (2026-09-20)

- **codebooks**: Use mx.searchsorted in ScalarCodebook.quantize, drop per-call import
- **dsa**: Vectorize 1/2-bit pack/unpack loops, drop heap copy in top_k

[Full changelog entry](CHANGELOG.md#v0903-2026-09-20)

## v0.90.2 (2026-09-20)

- **allocators**: Vectorize DP inner loop and k-means distance; fix quadratic concat
- **artifacts**: Move mlx imports to module level, fix temp file collisions
- **outlier**: Replace per-token heap top-k with vectorized argpartition
- **preconditioners**: Float32 accumulation and cache S in QJL/Hadamard
- **routing**: Drop RateEstimator's dead per-request history tracking
- **transfer**: Batch per-head ridge solves in cross-model KV mapper fit
- **ui**: Avoid full-buffer copies on the panel's 1s poll loop

[Full changelog entry](CHANGELOG.md#v0902-2026-09-20)

## v0.90.1 (2026-09-20)

- **profiling**: Remove per-token memory reads and batch mx.eval in KV cache profilers

[Full changelog entry](CHANGELOG.md#v0901-2026-09-20)

## v0.90.0 (2026-09-19)

- **testing**: Add pytest-cov with a measured 80% floor
- **testing**: Add pytest-cov with a measured 80% floor (#437)
- Document the coverage gate in CI_AND_TESTING.md
- Gate coverage on the MLX lane's 3.12 leg only

[Full changelog entry](CHANGELOG.md#v0900-2026-09-19)

## v0.89.0 (2026-09-18)

- **typing**: Add a per-module mypy strictness ladder
- **typing**: Add a per-module mypy strictness ladder (#438)
- **benchmark_scripts**: Hoist gemma4's mid-file imports instead of ignoring E402
- **cache**: Add strict=True to svdq's per-head zip()
- **cache**: Drop unused nestedkv_score import
- **packaging**: Exclude tests from the wheel, keep them in the sdist (#434)
- Correct stale 3.11 Python-floor references left by #432
- Correct stale 3.11 Python-floor references left by #432 (#439)
- **cache**: Backfill docstrings on CLI entry point, panel UI, and artifact store (#424)
- Make ruff check blocking for the shipped package
- Make ruff check blocking for the shipped package (#436)
- **quantizers**: Dedupe eviction-quantizer scaffolding into shared helper
- Pin documented Python floor to requires-python so #439 cannot recur
- Pin the mypy ladder's two module lists together
- Lower Python floor to 3.10, matching mlx's own requirement
- **deps**: Bump anyio from 4.13.0 to 4.14.2
- **deps**: Bump anyio from 4.13.0 to 4.14.2 in /scripts

[Full changelog entry](CHANGELOG.md#v0890-2026-09-18)

## v0.88.1 (2026-09-18)

- **cache**: Stop AgeTieredKVCache from re-quantizing settled tokens every step (#397)

[Full changelog entry](CHANGELOG.md#v0881-2026-09-18)

## v0.88.0 (2026-09-18)

- **bench**: Make prompt_len/nsg CLI-configurable in scalar_attend benchmark

[Full changelog entry](CHANGELOG.md#v0880-2026-09-18)

## v0.87.0 (2026-09-18)

- **cache**: Add per-method KVCacheConfig.options dataclasses (#420)

[Full changelog entry](CHANGELOG.md#v0870-2026-09-18)

## v0.86.3 (2026-09-18)

- **ci**: Run tests/non_metal/ as a whole, not test_mac_recommender.py by name
- **lint**: Clear F821/F841 ruff violations; wire ruff-check + mypy into CI
- **cache**: Extract _resolve_head_dim helper, dedupe 8 copies

[Full changelog entry](CHANGELOG.md#v0863-2026-09-18)

## v0.86.2 (2026-09-17)

- **metal**: Align rabitq_attend/streaming_prefill kernel names with source stems
- **metal**: Move streaming-prefill architecture note into docs/
- **cache**: Dispatch KVCacheFactory.create() via a registry table
- **metal**: Extract shared kernel-source-loading + caching helpers
- **quantizers**: Route zipcache channel_quant/dequant through shared group-quant helper

[Full changelog entry](CHANGELOG.md#v0862-2026-09-17)

## v0.86.1 (2026-09-17)

- **cache**: Raise QuantizerConfigError instead of bare ValueError for config validation
- **core**: Re-export BlockPoolExhaustedError and OwnerAlreadyActiveError
- **metal**: Key crosskv_rope's kernel cache on the real dtype, not a collapsed label
- **metal**: Round comm_vq grid to threadgroup multiple; remove invalid template_names kwarg
- **quantizers**: Replace CompositeQuantizer's NumPy round-trip with on-device MLX scatter
- **cache**: Extract shared reader-counting helper for xquant/minicache

[Full changelog entry](CHANGELOG.md#v0861-2026-09-17)

## v0.86.0 (2026-09-17)

- **cache**: Warn/raise on low QFilters retention before output degrades
- **blog**: Honest KIVI Metal kernel benchmark on Qwen3-8B
- **blog**: Honest QFilters benchmark on Qwen3-8B
- **blog**: Honest TurboQuantRVQ benchmark on Qwen3-8B
- **blog**: Honest VecInfer benchmark on Qwen3-8B
- **blog**: QFilters budget sweep finds a coherence cliff on Qwen3-8B
- **blog**: QFilters real-calibration follow-up on Qwen3-8B

[Full changelog entry](CHANGELOG.md#v0860-2026-09-17)

## v0.85.0 (2026-09-17)

- Declare and test Python 3.13/3.14 support

[Full changelog entry](CHANGELOG.md#v0850-2026-09-17)

## v0.84.1 (2026-09-17)

- **ci**: Honor GitHub's actual rate-limit cooldown in copyright-watch

[Full changelog entry](CHANGELOG.md#v0841-2026-09-17)

## v0.84.0 (2026-09-17)

- **ci**: Auto-generate copyright-watch queries from all 43 cache methods
- **ci**: Copyright-watch false positives and silent query failures
- Add throwaway macos runner Metal availability probe (issue #395)
- Formalize metal pytest marker + run MLX suite on every PR (issue #395)

[Full changelog entry](CHANGELOG.md#v0840-2026-09-17)

## v0.83.23 (2026-09-16)

- **cache**: Eviction caches no longer corrupt prefill/decode attention via mask="causal" (issue #370)
- **cache**: Skvq trim() silently corrupts flush-frontier state (issue #26)
- **cli**: Sync recommend argparse choices with mac_recommender.py (issue #391)
- **cache**: Ruff-format snapkv mask fix from #392

[Full changelog entry](CHANGELOG.md#v08323-2026-09-16)

## v0.83.22 (2026-09-16)

- **cache**: SlidingWindowKVCache now actually evicts (issue #274)

[Full changelog entry](CHANGELOG.md#v08322-2026-09-16)

## v0.83.21 (2026-09-16)

- **cache**: Raise xquant_residual_bits default from 0 to 4 (issue #380)
- Drop duplicate author entry and stale GitPython pin comment

[Full changelog entry](CHANGELOG.md#v08321-2026-09-16)

## v0.83.20 (2026-09-15)

- **cache**: Guard A2ATSKVCache.merge() against silent batching substitution

[Full changelog entry](CHANGELOG.md#v08320-2026-09-15)

## v0.83.19 (2026-09-15)

- **cache**: Guard skvq against silent merge() substitution (issue #26)
- **cache**: Warn about xquant_residual_bits=0's unsafe default in field help (issue #35)

[Full changelog entry](CHANGELOG.md#v08319-2026-09-15)

## v0.83.18 (2026-09-15)

- **cache**: Guard skvq against silent merge() substitution (issue #26)

[Full changelog entry](CHANGELOG.md#v08318-2026-09-15)

## v0.83.17 (2026-09-15)

- **cache**: Zipcache merge() guard silently substituted plain fp16 cache (issue #36)

[Full changelog entry](CHANGELOG.md#v08317-2026-09-15)

## v0.83.16 (2026-09-15)

- **cache**: Xquant merge() guard silently severed cross-layer basis sharing (issue #35)

[Full changelog entry](CHANGELOG.md#v08316-2026-09-15)

## v0.83.15 (2026-09-15)

- **cache**: Turboquant_rvq merge() guard silently disabled compression on every request (issue #32)
- **cache**: Vecinfer merge() guard silently disabled VQ compression (issue #33)
- **cache**: Xkv merge() guard silently severed cross-layer basis sharing (issue #34)

[Full changelog entry](CHANGELOG.md#v08315-2026-09-15)

## v0.83.14 (2026-09-15)

- **cache**: Tova merge() batching guard (issue #31)

[Full changelog entry](CHANGELOG.md#v08314-2026-09-15)

## v0.83.13 (2026-09-15)

- **cache**: Svdq per-head SVD basis, merge() guard, --set array parsing (issue #30)

[Full changelog entry](CHANGELOG.md#v08313-2026-09-15)

## v0.83.12 (2026-09-15)

- **cache**: Streaming_llm merge() batching guard and tokens_kept telemetry gap (issue #29)

[Full changelog entry](CHANGELOG.md#v08312-2026-09-15)

## v0.83.11 (2026-09-15)

- **cache**: Squeeze silent merge() substitution + squeeze_resolved_budget field leak (issue #28)

[Full changelog entry](CHANGELOG.md#v08311-2026-09-15)

## v0.83.10 (2026-09-15)

- **cache**: Snapkv silent merge() substitution + trim() corruption (issue #27)

[Full changelog entry](CHANGELOG.md#v08310-2026-09-15)

## v0.83.9 (2026-09-14)

- **cache**: Guard palu against silent merge() substitution; fix field_schema (issue #23)
- **cache**: Guard pyramidkv against silent merge() substitution; curate field_schema (issue #24)
- **cache**: Guard qfilters against silent merge() substitution (issue #25)
- **cache**: Pyramidkv silent merge() substitution + field_schema leak (issue #24)
- **cache**: Qfilters silent merge() substitution (issue #25)

[Full changelog entry](CHANGELOG.md#v0839-2026-09-14)

## v0.83.8 (2026-09-14)

- **cache**: Guard palu against silent merge() substitution; fix field_schema (issue #23)
- **cache**: Nestedkv's zero-padded ragged heads corrupt attention; also fix batching (issue #21)
- **cache**: NSNQuantKVCache is silently unbatchable-unsafe (issue #22)

[Full changelog entry](CHANGELOG.md#v0838-2026-09-14)

## v0.83.7 (2026-09-14)

- **cache**: Kvtc is invisible to /v1/kv/stats and loses compression under batched serving (issue #17)
- **cache**: Kvzip loses all eviction under batched serving (issue #18)
- **cache**: Minicache loses cross-layer merging under batched serving (issue #19)
- **cache**: Morphkv loses all eviction under batched serving (issue #20)

[Full changelog entry](CHANGELOG.md#v0837-2026-09-14)

## v0.83.6 (2026-09-14)

- **cache**: Kvquant loses n_sink config and its quantization under batched serving

[Full changelog entry](CHANGELOG.md#v0836-2026-09-14)

## v0.83.5 (2026-09-14)

- **cache**: Stop knorm from silently losing eviction under batched serving

[Full changelog entry](CHANGELOG.md#v0835-2026-09-14)

## v0.83.4 (2026-09-14)

- **registry**: Add missing n_sink_tokens to kivi_sink's config fields

[Full changelog entry](CHANGELOG.md#v0834-2026-09-14)

## v0.83.3 (2026-09-13)

- **registry**: Expose real config fields for methods missing from _CONFIG_FIELDS

[Full changelog entry](CHANGELOG.md#v0833-2026-09-13)

## v0.83.2 (2026-09-13)

- **landing**: Scope playground footer-hide rule to the playground page only
- **readme**: Use real Loom thumbnail URL from oEmbed
- **registry**: Update stale adakv paper-deviation caveat
- Note PEP 561 py.typed support in intro overview
- Sync getting-started pages and changelog with current source
- **algorithms**: Fix leftover KVCacheBuilder.build() call in spectral.md
- **algorithms**: Fix standalone-method examples and stale API calls
- **api**: Fix exceptions-api.md and memory-api.md against source
- **api**: Rewrite fabricated allocators/cache/core/spectral/metal API pages
- **control-panel**: Correct Optional-field count from 25 to 20
- **control-panel**: Document --set overrides and optional-field blanking
- **metal-kernels**: Add missing PyramidKV and TOVA fused-evict kernels
- **mlx-lm-integration**: Fix nonexistent bits/value_bits fields and API calls
- **readme**: Add demo video thumbnail linking to Loom recording
- **readme**: Add Star History chart
- **deps**: Move requirements.txt out of repo root

[Full changelog entry](CHANGELOG.md#v0832-2026-09-13)

## v0.83.1 (2026-09-12)

- **landing**: Use kivi 2-bit in quickstart example, not turboquant_rvq 1-bit
- **deps**: Bump urllib3 from 2.6.3 to 2.7.0

[Full changelog entry](CHANGELOG.md#v0831-2026-09-12)

## v0.83.0 (2026-09-12)

- **packaging**: Add PEP 561 py.typed marker
- Resolve remaining ruff violations flagged under current select
- **allocators**: Narrow Optional next/prev choice in ratequant rebalance
- **cache**: Fix chunkkv/composite/sliding-window type errors
- **cache**: Recognize PEP 604 `X | None` as Optional in describe_field
- **cache**: Type KVCacheFactory.create()'s dual cache hierarchy
- **dsa**: Make MaxHeap generic over its value type
- **metal**: Reword comment mypy misparses as a type comment
- **quantizers**: Fix rabitq.py encode() override and dead-code errors
- **quantizers**: Type CodebookFactory/PreconditionerFactory call sites
- **spectral**: Type CodebookFactory.create() calls and EncodedVector guards
- **tools**: Annotate mac_recommender's knobs dict as dict[str, Any]
- Add module docstrings for artifacts/cache/codebooks/core (PEP 257)
- Add module docstrings for core/dsa/handlers (PEP 257)
- Add module docstrings for math/memory/observers/outlier/preconditioners (PEP 257)
- Add module docstrings for quantizers/spectral/transforms/weight (PEP 257)
- **lint**: Record the rejected-rule-group and E501 audit trail
- Apply ruff format and autofix repo-wide baseline
- Enable ruff C4 (flake8-comprehensions) and fix its 66 violations
- Enable ruff PTH (flake8-use-pathlib) and fix its 13 violations
- Enable ruff RET (flake8-return) and fix its 14 violations
- Enable ruff SIM (flake8-simplify) and fix its 27 violations
- Standardize `from __future__ import annotations` across the package
- **deps**: Replace requirements.local.txt with a full venv freeze
- **deps**: Bump aiohttp from 3.14.1 to 3.14.3
- **deps**: Bump pillow from 12.2.0 to 12.3.0

[Full changelog entry](CHANGELOG.md#v0830-2026-09-12)

## v0.82.2 (2026-09-12)

- **cache**: Wire VecInfer fused_sdpa to actually reduce live memory

[Full changelog entry](CHANGELOG.md#v0822-2026-09-12)

## v0.82.1 (2026-09-11)

- **cache**: Warn instead of silently swallowing make_cache() failures

[Full changelog entry](CHANGELOG.md#v0821-2026-09-11)

## v0.82.0 (2026-09-11)

- **playground**: Add M5 to Apple Silicon chip selector
- **cli**: Add missing Dict import in serve.py type hints
- **cli**: Warn when --set targets a field irrelevant to the chosen method
- **landing**: Correct algorithm count from 33 to 43
- **landing**: Improve light-mode text contrast in several sections
- **playground**: Fix Compression Lab light theme leaking dark surfaces
- **playground**: Fix light theme leaking dark surfaces on recommendation card
- **playground**: Fix Metal Benchmarks loading errors and light theme
- **playground**: Fix remaining light-mode contrast issues
- **h2o**: Vectorize per-eviction row drop instead of Python index list
- **readme**: Replace Netlify domain with veloxquant.dev
- **landing**: Bump cache-busting query strings for styles.css/playground.js

[Full changelog entry](CHANGELOG.md#v0820-2026-09-11)

## v0.81.2 (2026-09-10)

- **snapkv**: Store storage dtype as name to fix deepcopy crash
- Add Node.js SDK developer blog
- **blog**: Polish Node.js SDK blog title and copy

[Full changelog entry](CHANGELOG.md#v0812-2026-09-10)

## v0.81.1 (2026-09-09)

- **landing**: Remove internal repo paths from visitor-facing copy
- Link npm package for Node.js users
- Make architecture guide reader friendly
- **deps**: Bump @swc/html from 1.15.40 to 1.16.2 in /docs-site
- **deps**: Bump colord from 2.9.3 to 2.10.0 in /docs-site
- **deps**: Bump js-yaml from 3.15.1 to 3.15.2 in /docs-site
- **deps**: Bump svgo from 3.3.4 to 3.3.5 in /docs-site

[Full changelog entry](CHANGELOG.md#v0811-2026-09-09)

## v0.81.0 (2026-09-09)

- Add MLX Metal worker protocol
- Add veloxquant_mlx/cli/worker.py
- Add veloxquant_mlx/tests/metal/test_worker_kernel_parity.py
- Expose worker command
- Add end-to-end architecture guide
- Fix architecture preview links
- Link architecture guide in sidebar

[Full changelog entry](CHANGELOG.md#v0810-2026-09-09)

## v0.80.1 (2026-09-08)

- **cache**: Add fused Metal kernels for TOVA eviction
- **cache**: Batch H2O eviction across heads, fix TOVA offset test, correct KIVI memory docs
- **cache**: Deferred-lineage and virtual-V routes for TOVA multi-token eviction
- **kernels**: Correct stale GQA head-packing finding, add eviction/quant benchmarks
- **kernels**: Correct stale GQA head-packing finding, add eviction/quant benchmarks to landing
- Fix ruff format on kv_kernel_gqa_packing_recheck.py
- Remove cache bookkeeping audit report from repo

[Full changelog entry](CHANGELOG.md#v0801-2026-09-08)

## v0.80.0 (2026-09-08)

- **landing**: Add VeloxQuant Studio waitlist modal
- **panel**: Add Hugging Face Hub search to model picker

[Full changelog entry](CHANGELOG.md#v0800-2026-09-08)

## v0.79.1 (2026-09-08)

- **panel**: Surface method-discovery failures and validate serve options

[Full changelog entry](CHANGELOG.md#v0791-2026-09-08)

## v0.79.0 (2026-09-07)

- **landing**: Add X, LinkedIn, and Discord social links to footer
- **landing**: Bump styles.css cache-busting version for social-link styles

[Full changelog entry](CHANGELOG.md#v0790-2026-09-07)

## v0.78.0 (2026-09-07)

- **landing**: Add Product Hunt badge to the hero
- **landing**: Add Swift SDK ecosystem card, fix nav CSS scoping
- **landing**: Promote Kotlin package to a live JitPack link, redesign ecosystem banner
- **landing**: Promote macOS app to a full-width featured banner
- **playground**: Search real Hugging Face models for the memory calculator
- **landing**: Bump asset cache-busting versions for the model search release
- **readme**: Add Ecosystem section linking the sibling SDKs
- **readme**: Drop VeloxQuant Studio from the Ecosystem section
- **readme**: Simplify — 557 -> 364 lines
- **readme**: Surface the website with a badge and a named link row
- Apply ruff format to base.py and test_scalar_attend.py
- **landing**: Add intent-search keywords and WebSite schema, additive only
- **seo**: Add PyPI keywords/classifiers and docs-site page metadata, additive only

[Full changelog entry](CHANGELOG.md#v0780-2026-09-07)

## v0.77.1 (2026-09-06)

- **cache**: Preserve native cache types on hybrid-attention models
- **readme**: Add visitor counter badge

[Full changelog entry](CHANGELOG.md#v0771-2026-09-06)

## v0.77.0 (2026-09-05)

- **landing**: Add interactive Studio showcase and kernel roofline benchmarks
- **landing**: Improve hero text contrast against background
- **landing**: Unify footer across pages, fix privacy.html SEO gaps, correct method count
- **metal**: Autotune nsg in scalar_fused_decode_attend (1.2-4.2x)
- **landing**: Refresh hero, nav, and add methods/why-it-matters sections
- **landing**: Unify pill nav across all pages, restyle playground/benchmarks

[Full changelog entry](CHANGELOG.md#v0770-2026-09-05)

## v0.76.0 (2026-09-05)

- **metal**: Cross-layer batched decode-attend kernel + real-model validation (#307 pt.1)
- **blog**: Add linked results-focused companion post for cross-layer batching
- Fix lint (E741, B023, UP045, F401) and apply ruff format

[Full changelog entry](CHANGELOG.md#v0760-2026-09-05)

## v0.75.1 (2026-09-04)

- **landing**: Fix CLS from announcement banner, lazy-load Zendesk, minify at build

[Full changelog entry](CHANGELOG.md#v0751-2026-09-04)

## v0.75.0 (2026-09-03)

- **cache**: Verify, tune, and extend MLX prefix-cache reuse
- Apply ruff format

[Full changelog entry](CHANGELOG.md#v0750-2026-09-03)

## v0.74.0 (2026-09-03)

- **landing**: Add Zendesk web widget for support chat
- **metal**: Add GQA head-packing to scalar_fused_decode_attend
- **metal**: GQA head-packing + SIMD-shuffle spike for scalar_fused_decode_attend (#307, #308)
- **metal**: Investigate SIMD-shuffle sharing for GQA decode (issue #308)
- **kernels**: Explain why no real-model benchmark was run for #307/#308
- **ci**: Stop landing/docs-only pushes from bumping the package version

[Full changelog entry](CHANGELOG.md#v0740-2026-09-03)

## v0.73.0 (2026-09-03)

- **landing**: Add privacy policy page, update Studio link to v0.1.1
- **landing**: Add privacy policy, update Studio download to v0.1.1
- **landing**: Don't name third-party vendors in privacy policy
- **landing**: Unlink privacy.html from site navigation
- **deps**: Bump fast-uri from 3.1.5 to 3.1.7 in /docs-site

[Full changelog entry](CHANGELOG.md#v0730-2026-09-03)

## v0.72.0 (2026-09-02)

- **landing**: Add npm/Go/Rust SDK links to install, footer, and nav
- **landing**: Add SDK keywords to meta/JSON-LD for search visibility
- **landing**: Bump styles.css cache-busting version

[Full changelog entry](CHANGELOG.md#v0720-2026-09-02)

## v0.71.1 (2026-09-02)

- **landing**: Bump styles.css cache-busting version
- **landing**: Cache-bust main.js/calc.js/playground.js

[Full changelog entry](CHANGELOG.md#v0711-2026-09-02)

## v0.71.0 (2026-09-02)

- **landing**: Add dismissible Studio release announcement banner
- **landing**: Replace Studio waitlist with direct download link
- Document `veloxquant auto-config` CLI (issue #253, PR #295)

[Full changelog entry](CHANGELOG.md#v0710-2026-09-02)

## v0.70.0 (2026-09-02)

- Add `veloxquant auto-config` CLI for select_kv_cache_config()
- **kernels**: Roofline analysis of KV-cache quantize/dequantize kernels (issue #259)
- Fix ruff-format drift on master

[Full changelog entry](CHANGELOG.md#v0700-2026-09-02)

## v0.69.1 (2026-09-02)

- **cache**: Investigate zero-copy KV-cache access (issue #255)
- **deps**: Bump browserslist from 4.28.2 to 4.28.8 in /docs-site
- **deps**: Bump postcss-selector-parser in /docs-site
- **deps**: Bump transformers from 5.7.0 to 5.10.1

[Full changelog entry](CHANGELOG.md#v0691-2026-09-02)

## v0.69.0 (2026-09-01)

- **cache**: Add AgeTieredKV (issue #256) — investigates position/age-gated multi-tier KV precision

[Full changelog entry](CHANGELOG.md#v0690-2026-09-01)

## v0.68.0 (2026-09-01)

- **profiling**: Add MLXCacheProfiler + `veloxquant profile` CLI

[Full changelog entry](CHANGELOG.md#v0680-2026-09-01)

## v0.67.3 (2026-09-01)

- **landing**: Cache-bust styles.css to unstick stale immutable cache
- Fix stale algorithm counts on intro/overview, add robots.txt

[Full changelog entry](CHANGELOG.md#v0673-2026-09-01)

## v0.67.2 (2026-09-01)

- **panel**: Make Copy buttons work reliably in control panel

[Full changelog entry](CHANGELOG.md#v0672-2026-09-01)

## v0.67.1 (2026-08-31)

- **docs-site**: Disable Algolia contextualSearch, index has no facets
- Add manual Algolia indexing script
- Document 8 missing Metal kernel modules
- Rename Algolia env vars to clearer ALGOLIA_* names
- Apply ruff format to metal-kernels.md code blocks

[Full changelog entry](CHANGELOG.md#v0671-2026-08-31)

## v0.67.0 (2026-08-30)

- Causal prefill attention + from-scratch flash kernel investigation
- Use MDX-safe truncate marker in prefill-roofline blog post
- Measurement-driven tuning pass on flash_prefill_attend
- Apply ruff format to CacheRoute files (pre-existing, unrelated to #277)
- Fix ruff-format violation in test_flash_prefill.py

[Full changelog entry](CHANGELOG.md#v0670-2026-08-30)

## v0.66.0 (2026-08-30)

- CacheRoute rate-aware session admission and shard placement (#278)

[Full changelog entry](CHANGELOG.md#v0660-2026-08-30)

## v0.65.0 (2026-08-28)

- Add KV-cache workload replay benchmark
- Add KV-cache workload replay benchmark (#258)
- Correct sliding-window test assumption after finding real eviction bug

[Full changelog entry](CHANGELOG.md#v0650-2026-08-28)

## v0.64.0 (2026-08-28)

- Add hardware-aware automatic KV-cache configuration
- Hardware-aware automatic KV quantization configuration
- Satisfy ruff-format on embedded code blocks in auto-config guide

[Full changelog entry](CHANGELOG.md#v0640-2026-08-28)

## v0.63.0 (2026-08-28)

- Add KV-cache kernel and memory profiler
- Satisfy ruff-format check on embedded code block
- Satisfy ruff-format on pre-existing blog post
- Add blog post documenting the fused RVQ quantize+pack kernel
- Add docs-site pages for KVCacheProfiler
- Make profiling guide lead with payoff and quickstart
- Publish KIVI multi-model benchmark results and validation scripts

[Full changelog entry](CHANGELOG.md#v0630-2026-08-28)

## v0.62.2 (2026-08-28)

- Fuse RVQ quantize and bit-pack into a single Metal kernel

[Full changelog entry](CHANGELOG.md#v0622-2026-08-28)

## v0.62.1 (2026-08-28)

- Warm KIVI Metal kernels at cache-build time (#250)

[Full changelog entry](CHANGELOG.md#v0621-2026-08-28)

## v0.62.0 (2026-08-27)

- Harden block pool allocator (exhaustion, concurrency, defrag)

[Full changelog entry](CHANGELOG.md#v0620-2026-08-27)

## v0.61.0 (2026-08-27)

- Add sitemap/robots.txt and fix missing meta tags for SEO

[Full changelog entry](CHANGELOG.md#v0610-2026-08-27)

## v0.60.0 (2026-08-27)

- Drive real mlx_lm.generate() traffic through the block pool allocator

[Full changelog entry](CHANGELOG.md#v0600-2026-08-27)

## v0.59.0 (2026-08-27)

- Add KV-cache-aware block pool allocator
- Satisfy ruff format on embedded code snippets in memory-api.md
- Add docs-site + changelog entries for the block pool allocator
- Email direct alert on copyright-watch hits via Resend

[Full changelog entry](CHANGELOG.md#v0590-2026-08-27)

## v0.58.0 (2026-08-26)

- **weight**: Mmap-backed reservoir for pre-quantized model weights
- **ci**: Pin GitPython <3.1.60 to unbreak semantic-release
- **metal**: Skip redundant fp32 cast in turboquant_scalar_quantize for fp16 input
- Fix stale method count in package/citation metadata (41 -> 43)
- Fix pre-existing ruff-format drift on master
- Fix ruff format violations in embedded code fences
- Fix ruff format/lint failures on weight-reservoir files

[Full changelog entry](CHANGELOG.md#v0580-2026-08-26)

## v0.57.1 (2026-08-25)

- **ui**: Harden static-file path containment check in control panel
- **landing**: Calmer, human-centric hero and dedicated benchmarks page
- **landing**: Add Plausible analytics

[Full changelog entry](CHANGELOG.md#v0571-2026-08-25)

## v0.57.0 (2026-08-22)

- **landing**: Link to VS Code extension
- **docs-site**: Patch npm vulnerabilities via dependency overrides
- Remove dangling links to deleted OPTIMIZATION_FINDINGS.md
- **readme**: Rewrite for human voice, no content changes
- **landing**: Remove Benchmarks section from landing page
- **landing**: Remove Comparison section from landing page
- **landing**: Remove confusing Quality Cost section from landing page
- **landing**: Remove How It Works explainer section from landing page
- **landing**: Remove llama.cpp-vs-VeloxQuant-MLX card comparison from landing page
- **landing**: Remove method picker section from landing page
- **landing**: Rewrite landing page copy for a non-technical audience
- **landing**: Tighten hero copy on landing page
- **readme**: Swap banner for app-icon-matching logo, trim stale copy

[Full changelog entry](CHANGELOG.md#v0570-2026-08-22)

## v0.56.0 (2026-08-21)

- **rocketkv**: Add RocketKV two-stage KV cache compression
- **readme**: Add RocketKV, fix stale method counts and AnchorKV link
- Fix ruff-format violation in mac_recommender.py

[Full changelog entry](CHANGELOG.md#v0560-2026-08-21)

## v0.55.0 (2026-08-21)

- **mac_recommender**: Extend model classes and RAM tiers to real-world scale

[Full changelog entry](CHANGELOG.md#v0550-2026-08-21)

## v0.54.1 (2026-08-21)

- **docs-site**: Bump joi and http-proxy-middleware to patched versions

[Full changelog entry](CHANGELOG.md#v0541-2026-08-21)

## v0.54.0 (2026-08-21)

- **landing**: Add VeloxQuant Studio waitlist section and fix calc URL leak

[Full changelog entry](CHANGELOG.md#v0540-2026-08-21)

## v0.53.0 (2026-08-20)

- **anchorkv**: Add AnchorKV anchor-residual KV cache compression
- **anchorkv**: Resolve ruff-format and link-check CI failures

[Full changelog entry](CHANGELOG.md#v0530-2026-08-20)

## v0.52.1 (2026-08-19)

- **svdq**: Implement paper's real 8-group bit schedule, guard against small-rank truncation
- Fix ruff formatting in test_svdq_cache.py

[Full changelog entry](CHANGELOG.md#v0521-2026-08-19)

## v0.52.0 (2026-08-19)

- **bench**: Add H2O and TOVA as Q-Filters comparison arms
- **qfilters**: Pin fallback chunk-dependence and budget safety

[Full changelog entry](CHANGELOG.md#v0520-2026-08-19)

## v0.51.1 (2026-08-19)

- **ci**: Stop the link checker failing on network flakiness
- **docs**: Correct two README links that 404ed, and check links in CI
- **site**: Redirect docs paths missing the /docs prefix instead of 404ing
- Drop oMLX from the comparison, keep llama.cpp and plain mlx_lm
- **deps**: Bump body-parser from 1.20.5 to 1.20.6 in /docs-site
- **deps**: Bump shell-quote from 1.8.4 to 1.10.0 in /docs-site
- **deps**: Bump svgo from 3.3.3 to 3.3.4 in /docs-site
- **deps**: Bump webpack-dev-server from 5.2.4 to 5.2.6 in /docs-site
- **deps**: Bump websocket-driver from 0.7.4 to 0.7.5 in /docs-site

[Full changelog entry](CHANGELOG.md#v0511-2026-08-19)

## v0.51.0 (2026-08-18)

- **bench**: RULER task categories beyond NIAH for Q-Filters
- **bench**: RULER task categories beyond NIAH for Q-Filters (#177)
- **blog**: The needle was the easy part -- RULER beyond NIAH
- **qfilters**: Report the non-NIAH RULER results and correct the gap claim
- **ruler**: Probe whether chunked prefill rescues Q-Filters on VT
- **ruler**: Raw results for the four non-NIAH RULER categories on Qwen2.5-7B
- **ruler**: Record actual prefilled token counts per task and context
- **ruler**: Verify the arms compress equally before reading the CWE spread
- **deps**: Bump brace-expansion from 1.1.15 to 1.1.18 in /docs-site
- **deps**: Bump postcss from 8.5.15 to 8.5.26 in /docs-site

[Full changelog entry](CHANGELOG.md#v0510-2026-08-18)

## v0.50.2 (2026-08-18)

- **qfilters**: Derive head_dim for Qwen and validate at 7B (#176)
- Rewrite README prose in a first-person engineering voice
- **release**: Collapse merge bursts into one release run
- **rotation**: Pin bare-m Hadamard behaviour to the installed MLX
- **deps**: Bump authlib from 1.7.0 to 1.7.1
- **deps**: Bump fast-uri from 3.1.2 to 3.1.5 in /docs-site
- **deps**: Bump gradio from 6.13.0 to 6.15.1
- **deps**: Bump idna from 3.13 to 3.15
- **deps**: Bump joserfc from 1.6.4 to 1.6.8
- **deps**: Bump js-yaml from 3.14.2 to 3.15.1 in /docs-site
- **deps**: Bump mcp from 1.27.0 to 1.28.1
- **deps**: Bump nanoid from 3.3.12 to 3.3.18 in /docs-site
- **deps**: Bump pydantic-settings from 2.14.0 to 2.14.2
- **deps**: Bump python-multipart from 0.0.26 to 0.0.31
- **deps**: Bump starlette from 1.0.0 to 1.3.1
- **deps**: Bump urllib3 from 2.6.3 to 2.7.0

[Full changelog entry](CHANGELOG.md#v0502-2026-08-18)

## v0.50.1 (2026-08-18)

- **landing**: Match the test-count format the release sync rewrites
- **readme**: Surface security and governance policies, trim badge wall

[Full changelog entry](CHANGELOG.md#v0501-2026-08-18)

## v0.50.0 (2026-08-17)

- **landing**: Lead with unqualified metrics, resequence hero caveats
- **cache**: Review SnapKV RoPE position semantics vs. paper (#188)
- **cache**: Review StreamingLLM RoPE position semantics vs. paper (#189)
- **knorm**: Measure real-model perplexity for L2Norm after RoPE fix (#190)
- **deps**: Bump cryptography from 47.0.0 to 50.0.0
- **deps**: Bump pillow from 12.2.0 to 12.3.0
- **deps**: Bump pyjwt from 2.12.1 to 2.13.0

[Full changelog entry](CHANGELOG.md#v0500-2026-08-17)

## v0.49.4 (2026-08-15)

- **qfilters**: Wire fused Metal eviction kernels into the cache hot path

[Full changelog entry](CHANGELOG.md#v0494-2026-08-15)

## v0.49.3 (2026-08-15)

- **cache**: Generalize RoPE offset fix to TOVA cache; audit H2O (#175)

[Full changelog entry](CHANGELOG.md#v0493-2026-08-15)

## v0.49.2 (2026-08-15)

- **cache**: Generalize RoPE offset fix to L2Norm cache (#174)

[Full changelog entry](CHANGELOG.md#v0492-2026-08-15)

## v0.49.1 (2026-08-14)

- **qfilters**: Vectorize the per-(B, H) eviction loop

[Full changelog entry](CHANGELOG.md#v0491-2026-08-14)

## v0.49.0 (2026-08-14)

- **qfilters**: Paper-faithful query-SVD calibration + fused Metal kernels
- **cache**: Correct RoPE offset after eviction in SnapKV and StreamingLLM
- **docs**: Use MDX comment syntax for the blog truncation marker
- **qfilters**: Correct RoPE offset after eviction, measure real perplexity
- **blog**: Add Q-Filters real-model investigation write-up
- **changelog**: Record the SnapKV/StreamingLLM RoPE offset fix
- Apply ruff format to Q-Filters markdown code blocks
- **qfilters**: Add TTFT, throughput and NIAH harness vs SnapKV/StreamingLLM
- **qfilters**: Validate query-SVD calibration on real trained models

[Full changelog entry](CHANGELOG.md#v0490-2026-08-14)

## v0.48.5 (2026-08-13)

- Apply ruff 0.16.3 formatting to CAM docs and tests
- Fix two release-workflow failures on master

[Full changelog entry](CHANGELOG.md#v0485-2026-08-13)

## v0.48.4 (2026-08-13)

- Apply ruff 0.16.3 formatting to cache/base.py

[Full changelog entry](CHANGELOG.md#v0484-2026-08-13)

## v0.48.3 (2026-08-13)


[Full changelog entry](CHANGELOG.md#v0483-2026-08-13)

## v0.48.2 (2026-08-13)

- Satisfy ruff 0.16.3 format across the AdaKV changes

[Full changelog entry](CHANGELOG.md#v0482-2026-08-13)

## v0.48.1 (2026-08-13)

- **kvquant**: Correct outlier selection, decode protection, and add sink-aware quantization

[Full changelog entry](CHANGELOG.md#v0481-2026-08-13)

## v0.48.0 (2026-08-12)

- **kivi**: Fused Metal kernel for asymmetric group quantization
- **kivi**: Split into layout-specific kernels, drop the transpose
- **kivi**: Correct stale register-caching note in channel kernel

[Full changelog entry](CHANGELOG.md#v0480-2026-08-12)

## v0.47.1 (2026-08-12)

- **kivi**: Buffer residual to group_size-aligned flushes
- **landing**: Rewrite the method picker for a non-expert audience
- **kivi**: Cover the documented model geometries; fill in the docs table

[Full changelog entry](CHANGELOG.md#v0471-2026-08-12)

## v0.47.0 (2026-08-11)

- **landing**: Add GoatCounter privacy-friendly analytics
- **landing**: Rewrite the playground for a non-expert audience

[Full changelog entry](CHANGELOG.md#v0470-2026-08-11)

## v0.46.0 (2026-08-11)

- **landing**: Add FAQ section, drop SmolLM2-135M from benchmark table
- **transfer**: Cross-model KV cache transfer via closed-form ridge mapper
- **pyproject**: Point Documentation URL at the Netlify docs site
- **registry**: Distinguish not-trimmable from crash-tier, unblocking releases
- **release**: Sync the landing hero badge to the real markup, not a dead main.js pattern
- **readme**: Condense method library, tighten caveats, add project context
- **readme**: Drop SmolLM2-135M from benchmark tables
- **readme**: Loosen research-paper-dense prose into plain developer language
- **transfer**: Correct the Metal kernel speedup, 12.9x was not reproducible
- Satisfy ruff format on the new docs page and a pre-existing test

[Full changelog entry](CHANGELOG.md#v0460-2026-08-11)

## v0.45.0 (2026-08-10)

- **keyformer**: Annealing, RoPE remap, fused Metal kernel, real-model validation

[Full changelog entry](CHANGELOG.md#v0450-2026-08-10)

## v0.44.4 (2026-08-10)

- **h2o**: Correct RoPE position desync after eviction
- **h2o**: Grace period fixes the early-token permanent-freeze bug
- **h2o**: Score decay fixes stale-token dominance after grace fixed the freeze
- **h2o**: Fused Metal kernel for the over-budget eviction step
- **h2o**: Vectorize below-budget prefill path, fixing a real crash

[Full changelog entry](CHANGELOG.md#v0444-2026-08-10)

## v0.44.3 (2026-08-10)

- **gear**: Wire KCVT base quantizer (per-channel keys, per-token values)
- **amc**: Make AMC-adapted algorithm page consumer-focused
- Apply ruff format to gear PR files
- Fix ruff format drift in RVQ docs and cache
- **amc**: Fix ruff format alignment in doc code blocks

[Full changelog entry](CHANGELOG.md#v0443-2026-08-10)

## v0.44.2 (2026-08-09)

- **cache**: Store turboquant_rvq keys genuinely packed, not dequantized fp16
- Correct turboquant_rvq API examples and update benchmark numbers
- Make TurboQuant RVQ page read as a normal feature page
- Record turboquant_rvq packed-storage investigation and update roadmap

[Full changelog entry](CHANGELOG.md#v0442-2026-08-09)

## v0.44.1 (2026-08-09)

- **cache**: Default KVCacheConfig method to turboquant_rvq, not turboquant_prod

[Full changelog entry](CHANGELOG.md#v0441-2026-08-09)

## v0.44.0 (2026-08-09)

- **release**: Publish to TestPyPI then PyPI via Trusted Publishing (OIDC)

[Full changelog entry](CHANGELOG.md#v0440-2026-08-09)

## v0.43.7 (2026-08-09)

- **ci**: Release.yml re-ran full release steps when nothing actually changed
- **readme**: Replace static status claims with live CI/PyPI badges

[Full changelog entry](CHANGELOG.md#v0437-2026-08-09)

## v0.43.6 (2026-08-09)

- **ci**: Release-notes extraction didn't match semantic-release's real heading format

[Full changelog entry](CHANGELOG.md#v0436-2026-08-09)

## v0.43.5 (2026-08-09)

- **release**: CHANGELOG.md was never auto-updated; add mode=update + backfill

[Full changelog entry](CHANGELOG.md#v0435-2026-08-09)

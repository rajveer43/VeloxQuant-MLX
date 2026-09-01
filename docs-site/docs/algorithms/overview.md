---
id: overview
title: Algorithm Overview
sidebar_label: Overview
slug: /algorithms/overview
---

# Algorithm Overview

VeloxQuant-MLX implements 43 KV cache compression algorithms. This page helps you pick the right one for your workload.

:::warning[Apple Silicon required]
All algorithms use Metal GPU kernels and require macOS on an M-series chip.
:::

:::tip[New here?]
If you haven't decided whether you need cache compression at all, read [VeloxQuant-MLX vs. llama.cpp / plain mlx_lm](../getting-started/comparison) first — it explains what problem these 43 methods solve and when you don't need them.
:::

## Comparison table

| Algorithm | Key bits | Val bits | Calibration | Compression | Quality | Best for |
|---|---|---|---|---|---|---|
| [TurboQuant RVQ](../algorithms/rvq) | 1–3 | 2–4 | None | 7.5× | ★★★★ | General purpose, zero setup |
| [VecInfer](../algorithms/vecinfer) | 1–4 | 2–4 | Codebook (2 min) | 16× | ★★★★ | Max throughput, Metal-accelerated |
| [RateQuant](../algorithms/ratequant) | mixed | mixed | Sensitivity (90 s) | 6–12× | ★★★★★ | Best accuracy per bit |
| [SpectralQuant](../algorithms/spectral) | 2–8 | 2–4 | SVD rotation (3 min) | 4–8× | ★★★★★ | Long context, high fidelity |
| [RaBitQ](../algorithms/rabitq) | 1 | fp16 | None | 6× total | ★★★ | Key-only extreme compression |
| [QJL](../algorithms/qjl) | 1 | fp16 | None | 8× key only | ★★★ | Simplest, fastest to set up |
| [PolarQuant](../algorithms/polarquant) | 1–2 | 2 | None | 8× | ★★★ | Geometric key distributions |
| [CommVQ](../algorithms/commvq) | 2–4 | fp16 | None | 4–8× | ★★★★ | RoPE-compatible models |
| [KIVI](../algorithms/kivi) | 2 | 2 | None | 4× total | ★★★ | Tuning-free asymmetric baseline |
| [KIVI-Sink](../algorithms/kivi-sink) | 2 | 2 | None | 4× total | ★★★★ | Sink-protected low-bit quantization |
| [SKVQ-adapted](../algorithms/skvq) | 2 | 2 | None (first-chunk stats) | ~5× incl. window | ★★★★ | Channel reordering + per-group clip search behind a sliding fp16 window — COLM 2024 |
| [SVDq](../algorithms/svdq) | ~1.25 | fp16 | SVD at prefill | 12.8× key | ★★★ | Sub-2-bit keys, long context |
| [Kitty](../algorithms/kitty) | ~2.5 | fp16 | None | 6.4× key | ★★★★ | Adaptive channel precision, zero calibration |
| [AdaKV-proxy](../algorithms/adakv) | adaptive (2–4) | fp16 | None | adaptive | ★★★★ | Per-head adaptive bits, layers on KIVI |
| [XQuant](../algorithms/xquant) | ~1.0–1.4 | yes | None | 11–16× | ★★★★ | First cross-layer reuse — adjacent layers share codes |
| [KVQuant-NUQ](../algorithms/kvquant) | 2–4 (non-uniform) | 2–4 | None | 5–8× | ★★★★★ | Non-uniform datatype + outlier isolation |
| [PALU](../algorithms/palu) | ~0.6 (low-rank) | ~0.6 (low-rank) | None | high (full-KV) | ★★★ | First true latent cache — both K and V stored low-rank |
| [CacheGen](../algorithms/cachegen) | 3–4 (entropy) | 3–4 (entropy) | None | +10–17% over packing | ★★★ | First entropy-coded cache — storage win on correlated KV |
| [MiniCache](../algorithms/minicache) | fp16 (merged) | fp16 (merged) | None | ~2× on merged layers | ★★★ | Cross-layer SLERP merge — pairs of deep layers cost one |
| [GEAR](../algorithms/gear) | 2–4 (+ feedback) | 2–4 (+ feedback) | None | quality at low bits | ★★★ | First error-feedback cache — residual low-rank + sparse outliers |
| [ZipCache-adapted](../algorithms/zipcache) | adaptive (2–4) | adaptive (2–4) | None | adaptive | ★★★★ | Per-token mixed bit-width — salient tokens get hi_bits, rest get lo_bits |
| [SnapKV-adapted](../algorithms/snapkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★★ | Token eviction — keeps only a budget of prefill positions by obs-window attention |
| [StreamingLLM-adapted](../algorithms/streaming_llm) | fp16 (kept tokens) | fp16 (kept tokens) | None | constant memory | ★★★★ | Structural eviction — first N sinks + last W recent tokens; constant-memory streaming |
| [H2O-adapted](../algorithms/h2o) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Cumulative attention-mass heavy-hitter eviction (ICLR 2024) — the library's original eviction baseline that SnapKV, KIVI-Sink, and several eviction scorers above reduce to at limiting parameter values |
| [TOVA-adapted](../algorithms/tova) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Memoryless current-step attention eviction (2024) — drops by lowest attention weight *this step only*, no running score; several eviction scorers above reduce to this at limiting parameter values |
| [PyramidKV-adapted](../algorithms/pyramidkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count, per-layer budget | ★★★★ | Layer-adaptive budget eviction (2024) — H2O-adapted's scoring with a pyramid of per-layer budgets (large early, small deep) at fixed average size; reduces to H2O-adapted when the pyramid is flat |
| [SqueezeAttention-adapted](../algorithms/squeeze) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count, data-driven per-layer budget | ★★★★ | 2D layer×token data-driven budget eviction (2024) — measures each layer's attention concentration during prefill and reallocates a fixed total budget accordingly; reduces to uniform H2O-adapted at zero strength |
| [ChunkKV-adapted](../algorithms/chunkkv) | fp16 (kept chunks) | fp16 (kept chunks) | None | constant memory | ★★★★ | Chunk-level eviction — keeps whole contiguous chunks by pooled importance; `chunk_size=1` == H2O |
| [CaM-adapted](../algorithms/cam) | fp16 (merged) | fp16 (merged) | None | constant memory | ★★★★ | Cache merging — merges evicted tokens into similar survivors instead of dropping; `cam_merge=drop` == H2O |
| [xKV-adapted](../algorithms/xkv) | uniform-bit (latent) | fp16 | None | 8–20% fewer bytes vs per-layer SVD | ★★★★ | Cross-layer shared-subspace — joint SVD basis amortized across a layer group |
| [NSNQuant-adapted](../algorithms/nsnquant) | 1–2 (VQ) | 1–2 (VQ) | None (by construction) | ~6.4× at 2-bit incl. metadata | ★★★★ | Calibration-free universal-codebook VQ — NSN + Hadamard reshape data to one fixed Gaussian codebook |
| [L2Norm-adapted](../algorithms/knorm) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Intrinsic key-norm eviction — low norm ⇒ important (EMNLP 2024 finding); zero per-step scoring cost, path-independent |
| [Q-Filters-adapted](../algorithms/qfilters) | fp16 (kept tokens) | fp16 (kept tokens) | None (first-chunk key-SVD) | token count | ★★★ | Query-agnostic projection eviction — score by projection onto a frozen per-head direction (preprint); key-SVD substitute for the paper's query-SVD, sign-ambiguous |
| [Keyformer-adapted](../algorithms/keyformer) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Gumbel-regularized heavy-hitter eviction (MLSys 2024) — H2O's accumulator plus frozen Gumbel noise that rescues late-rising tokens; `keyformer_tau=0` == H2O |
| [MorphKV-adapted](../algorithms/morphkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Recent-window correlation retention (ICML 2025) — ranks stored tokens by a sliding window of recent attention, dropping stale early heavy-hitters; `morphkv_window=1` == TOVA |
| [KVzip-adapted](../algorithms/kvzip) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Context-reconstruction reliance eviction (NeurIPS 2025) — keeps the KV pairs the model most relies on to reconstruct its own context (query-agnostic); `kvzip_probe="latest"` == TOVA |
| [KVTC-adapted](../algorithms/kvtc) | DP-optimal (0–8, may drop) | DP-optimal (0–8, may drop) | None (local PCA at prefill) | budget-matched | ★★★★ | Local PCA + DP-optimal per-component bit allocation + order-0 entropy coding (ICLR 2026) — beats fixed-split mixed-precision at matched byte budget on skewed variance |
| [CurDKV-adapted](../algorithms/curdkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★★ | Value-aware leverage-score eviction via approximated CUR decomposition (NeurIPS 2025) — evicts key-similar but value-irrelevant tokens that key-only eviction (H2O) cannot distinguish |
| [NestedKV-adapted](../algorithms/nestedkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Multi-scale ensembled prefill eviction — stable + episodic + current key anomaly, combined by a head-adaptive blend and surprise-gated route (no verified venue — one-time exception) |
| [AMC-adapted](../algorithms/amc) | adaptive (4/8/16) | adaptive (4/8/16) | SVD/PCA channel order (offline) | tiered, no eviction | ★★★ | Saliency-driven tiered rank + precision — one L1-norm score drives both rank and bit-width per token; compression-only, never evicts (no verified venue — second exception; hardware/RTL half of paper out of scope) |
| [A2ATS-adapted](../algorithms/a2ats) | codebook (2^8 default) | codebook (2^8 default) | Codebook training (offline) | ~4× at default settings | ★★★ | Windowed RoPE + query-aware retrieval VQ (ACL 2025 Findings) — exact position encoding for recent tokens, a cheap stand-in for distant ones; best when the tokens a query needs are usually nearby |
| [AnchorKV-adapted](../algorithms/anchorkv) | anchor+coeff (+residual) | anchor+coeff (+residual) | None | `anchorkv_theta`-driven, no eviction | ★★★★ | Anchor-residual compression — every token kept, most represented as a nearest-anchor projection, byte budget spent on residuals for the highest attention-output-impact tokens (no verified venue — second exception) |
| [RocketKV-adapted](../algorithms/rocketkv) | fp16 (kept tokens) + paged aux | fp16 (kept tokens) | None | `rocketkv_compression_ratio`-driven, adaptive split | ★★★★ | Two-stage compression (ICML 2025) — coarse SnapKV eviction, then per-step Hybrid Sparse Attention (paged max/min + head-dim top-k) dynamic selection over the survivors; closes the accuracy gap eviction-only methods hit at low token budgets |

*Compression ratios measured on Llama-3.1-8B at 4096 context. Source: [BENCHMARK_RESULTS.md](https://github.com/rajveer43/veloxquant-mlx/blob/master/BENCHMARK_RESULTS.md).*

## Decision guide

```
Do you want zero calibration?
├── Yes → TurboQuant RVQ (best quality), QJL (simplest), RaBitQ (1-bit keys)
└── No, I can spend 1–3 minutes calibrating →
    ├── Priority: max compression → VecInfer
    ├── Priority: max quality     → RateQuant or SpectralQuant
    └── Long sequences (8k+)     → SpectralQuant

Is RoPE positional encoding compatibility critical?
└── Yes → CommVQ

Do you have geometric/non-Gaussian key distributions?
└── Yes → PolarQuant

Do key channels have highly non-uniform variance?
└── Yes, want adaptive mixed-precision without calibration → Kitty

Are some attention heads far more quant-sensitive than others?
└── Yes, want a fixed average-bit target with per-head allocation → AdaKV-proxy

Are adjacent layers in your model highly correlated?
└── Yes, want the lowest effective bits by reusing codes across layers → XQuant

Are your K/V distributions heavy-tailed / non-uniform?
└── Yes, want best quality per bit without calibration → KVQuant-NUQ

Do a small fraction of your tokens have disproportionate attention weight?
└── Yes, want token-level bit allocation (not fp16 protection) → ZipCache-adapted

Do you need a hard cap on token count (very long context, fixed RAM budget)?
├── Yes, evict by importance score (score-based) → SnapKV-adapted
└── Yes, constant-memory streaming (positional) → StreamingLLM-adapted
```

## Method families

### Zero-calibration methods

These work immediately on any model with no setup beyond installation.

- **[TurboQuant RVQ](../algorithms/rvq)** — The recommended default. Uses analytical Gaussian + Laplacian codebooks precomputed from distribution theory. Two residual passes give excellent fidelity at 1 bit per pass.
- **[QJL](../algorithms/qjl)** — Johnson-Lindenstrauss 1-bit sign sketch. Provably preserves inner products in expectation. Extremely simple — great for prototyping.
- **[RaBitQ](../algorithms/rabitq)** — Randomised Hadamard transform + 1-bit sign packing with IVF clustering. Better than QJL for key-only compression.
- **[PolarQuant](../algorithms/polarquant)** — Recursive polar decomposition for models where keys form geometric clusters.
- **[CommVQ](../algorithms/commvq)** — RoPE-commutative residual VQ: quantization that commutes with rotary position embeddings, preserving exact positional information.
- **[Kitty](../algorithms/kitty)** — Dynamic channel-wise mixed-precision: ranks key channels by online variance and allocates 4-bit to high-variance channels, 2-bit to the rest. Zero calibration, 2.5-bit effective key precision.
- **[AdaKV-proxy](../algorithms/adakv)** — Per-head adaptive bit allocation layered on KIVI: ranks heads by importance and solves a per-head bit budget so the average bits/element hits a configured target. Two selectable signals: `norm_variance` (online key-norm variance — measures *quantization sensitivity*, and is anti-correlated with the paper's criterion) or `attention_entropy` (observation-window attention dispersion, which carries Ada-KV's own sign). Note the target must lie strictly inside `(lo_bit, hi_bit)` or the allocation degrades to uniform KIVI. Zero calibration; complements Kitty's per-channel axis.
- **[XQuant](../algorithms/xquant)** — Cross-layer reuse: adjacent layers are paired (anchor/reuse), the anchor publishes its quantized codes through a shared coordinator, and reuse layers store only their own scale/zero (+ optional residual). The first method to exploit *inter-layer* redundancy — sub-1.4-bit effective keys on correlated models, zero calibration.
- **[KVQuant-NUQ](../algorithms/kvquant)** — Non-uniform quantization datatype: places `2^bits` signpost levels where the data actually is via online Lloyd-Max fitting, plus dense/sparse outlier isolation that carves the top few extreme elements out to fp16. The first non-uniform-datatype method — strictly lower reconstruction error than uniform at the same bit-width, zero calibration.
- **[PALU](../algorithms/palu)** — True low-rank latent storage: fits one shared projection per head-group from the prefill batch and stores the cache as latent codes `[S, r]` for *both* keys and values, reconstructing fp16 only at attend time. Unlike SVDq (keys-only, reconstructs full fp16), the cache itself stays low-rank, so the storage win is real. Layered with mixed-bit latent quantization for a full-KV effective rate below 1 bit/element. Zero calibration.
- **[CacheGen](../algorithms/cachegen)** — Entropy coding: the first method to compress the *codes* themselves rather than just pick a bit-width. Exploits token-wise locality (adjacent tokens' KV are similar) by delta-coding the quantized codes and entropy-coding the low-entropy residual stream toward its Shannon entropy. Reconstruction is identical to group quant; the win is storage, capped to never exceed fixed-width packing. Zero calibration.
- **[MiniCache](../algorithms/minicache)** — Cross-layer depth merging: adjacent middle-to-deep layers share one SLERP-interpolated direction while each keeps its own per-token magnitude, so a pair of layers costs roughly one. High-divergence token pairs are retained unmerged. A different route to inter-layer redundancy than XQuant (which reuses codes); MiniCache merges the tensors. Zero calibration.
- **[xKV-adapted](../algorithms/xkv)** — Cross-layer shared-subspace compression: a fixed-size contiguous group of layers jointly factorizes its stacked key matrices into *one* shared SVD basis via a fan-in/fan-out coordinator, then each layer stores only its own latent codes in that basis. A third route to inter-layer redundancy alongside XQuant (code reuse) and MiniCache (direction merge) — xKV shares an entire subspace amortized across N layers rather than a pairwise code or direction. Keys only (values fp16). Zero calibration.
- **[NSNQuant-adapted](../algorithms/nsnquant)** — Calibration-free universal-codebook VQ: the first method to *reshape the data to match a fixed code* rather than fit a code to the data. A Normalize-Shift-Normalize transform plus a Hadamard rotation maps K/V tokens onto the standard normal distribution, so one codebook built offline from synthetic Gaussian samples — never from model activations — quantizes any model at 1–2 bits/element. Both keys and values quantized; chunk-flush fp16 residual buffer (KIVI's idiom). Zero calibration by construction.
- **[SKVQ-adapted](../algorithms/skvq)** — Channel reordering + clipped dynamic quantization: the first method to *regroup* head-dim channels by their statistics (sorted by dynamic range, frozen from the first flushed chunk) so per-token quantization groups stay tight, and the first to *clip* each group's range by a per-group grid-searched factor instead of covering outliers. Both K and V quantized behind a sliding fp16 window with an attention-sink filter (COLM 2024). Prefill and decode produce bit-for-bit identical caches. Zero calibration — the paper's offline KMeans/clip search is replaced by first-chunk statistics.
- **[GEAR](../algorithms/gear)** — Error feedback: the first method to reconstruct what an ultra-low-bit base quantizer threw away, rather than just pick a bit-width. It adds a low-rank approximation of the quantization *residual* plus a sparse correction for the few outlier entries the low-rank term cannot absorb — `X ~= Quant_b(X) + L·R + S`. The residual SVD reuses the same shared helper as SVDq/PALU, but applied to the error rather than the signal. Composes over any base quantizer to recover quality at low bits. Zero calibration.
- **[ZipCache-adapted](../algorithms/zipcache)** — Per-token mixed bit-width: the first method to allocate bit-width *per token* within the quantized space. Uses key L2-norm as a saliency proxy (the same proxy as KIVI-Sink and AdaKV-proxy, but with a different decision): the top `hi_fraction` tokens by norm get `hi_bits`; the rest get `lo_bits`. Both groups remain quantized — not fp16 protection. The effective average rate is `hi_frac×hi_bits + (1-hi_frac)×lo_bits`. Labeled "ZipCache-adapted" because the paper's true signal (attention scores) is not observable at the cache level. Zero calibration.
- **[L2Norm-adapted](../algorithms/knorm)** — Intrinsic-signal eviction: the first scorer read directly off the stored key itself. The EMNLP 2024 finding (Devoto et al.): keys with *low* L2 norm attract *high* attention in trained LMs — so keep the lowest-norm tokens. No attention, no proxy, no per-step scoring cost (norms are computed once at insertion), and the kept set is provably identical whether tokens arrive as one prefill block or one at a time. Note the sign inversion vs ChunkKV's key_norm option (which keeps high-norm). Zero calibration.
- **[Q-Filters-adapted](../algorithms/qfilters)** — Query-agnostic projection eviction: the repo's fourth eviction scorer class (after attention/proxy, structural, and intrinsic-norm). Scores each key by its projection onto a single frozen per-head direction — the paper's premise (arXiv:2503.02812, preprint) is that trained heads have an anisotropic QK geometry where one direction predicts attention. The honest crux: the paper derives that direction from *query* SVD offline, but a cache never sees queries, so we substitute the SVD of the first observed *keys* — which recovers the dominant axis but not which end is important (hence `qfilters_sign` is a genuine ablation, and the kept set is path-dependent). Zero calibration.
- **[Keyformer-adapted](../algorithms/keyformer)** — Gumbel-regularized heavy-hitter eviction (MLSys 2024, arXiv:2403.09054). Structurally H2O's proxy-attention accumulator plus one new ingredient: **Gumbel noise** on the eviction logits, so a "late riser" — a token that reads low early, before the queries that attend to it arrive — is not deterministically pruned before it can recover. The honest crux: the paper redraws and anneals the noise across generation, but a cache has no trustworthy global step, so we draw one deterministic per-position Gumbel value and freeze it (`keyformer_seed`); it preserves the intent, not the schedule. Setting `keyformer_tau=0` removes the noise and this cache **is** H2O-adapted, bit-for-bit — the honest ablation. Zero calibration.
- **[MorphKV-adapted](../algorithms/morphkv)** — Recent-window correlation retention (ICML 2025, arXiv:2503.00979). Where H2O ranks a stored token by **cumulative** attention (inertial — early heavy hitters dominate and crowd out the current topic) and TOVA ranks by the **single latest** query, MorphKV ranks by correlation with a **sliding window** of the most recent tokens, so a constant-size cache re-targets toward what the recent context actually reads — eliminating the "early-token bias." The honest crux: key-as-query proxy (a cache never sees the true query), and only the `morphkv_window=1` reduction is pinned — it **is** TOVA-adapted, bit-for-bit. No H2O collapse is claimed (MorphKV recomputes from the live window, never accumulates). Zero calibration.
- **[KVzip-adapted](../algorithms/kvzip)** — Context-reconstruction reliance retention (NeurIPS 2025 Oral, arXiv:2505.23416). Every other proxy scorer ranks a stored token by the attention it receives *from a query* (cumulative for H2O, latest for TOVA, windowed for MorphKV); KVzip ranks by **reconstruction reliance** — how much the model relies on a KV pair to *reconstruct its own context* — a **query-agnostic** importance profile computed once and reused across all future queries. The honest crux: key-as-reconstruction-probe proxy (a cache never runs the model to reconstruct), and only the `kvzip_probe="latest"` reduction is pinned — it **is** TOVA-adapted, bit-for-bit. No H2O collapse is claimed. Zero calibration.
- **[KVTC-adapted](../algorithms/kvtc)** — Local PCA + DP-optimal per-component bit allocation + order-0 entropy coding (ICLR 2026, arXiv:2511.01815). Palu/SVDq/SpectralQuant all use a **fixed** mixed-bit split (a hand-chosen top-25%/75% tier, or a binary signal/noise cutoff); KVTC computes a **dynamic-programming-optimal** bit-width per principal component under a hard total-bit budget — including exactly **0 bits** (dropping a component entirely) — then entropy-codes the resulting codes. The honest crux: local per-sequence PCA (not the paper's pre-calibrated global basis), an analytic Gaussian distortion proxy reused from `ratequant.py` (not a real-activation-fit rate-distortion model), and a plain order-0 Huffman coder (not the paper's possibly more sophisticated scheme, and never the Shannon-entropy bound). At a matched total byte budget, the DP allocator beats a fixed-uniform-bits baseline and SVDq's fixed top-25%/75% split on planted skewed-variance geometry. Zero calibration.
- **[CurDKV-adapted](../algorithms/curdkv)** — Value-aware leverage-score eviction via approximated CUR decomposition (NeurIPS 2025, arXiv:2509.15038). Every other eviction scorer above (H2O, KNorm, Q-Filters, Keyformer, MorphKV, KVzip) ranks a stored token using only its **key** side; CurDKV derives a **leverage score** from the joint key-and-value structure of the proxy attention output, so a token's value contribution — not just its key/attention profile — decides whether it survives. The honest crux: key-as-query proxy (same as H2O/SnapKV), an SVD-based energy-weighted leverage-score estimator rather than the paper's own CUR sampling routine, and newly-appended tokens are seeded with their own leverage score rather than a flat 0 (a flat-0 seed would let a negligible-value newcomer tie forever with an already-negligible survivor). Two tokens with **identical keys** but **different values** provably receive different CurDKV scores — a distinction no key-only method above can make. Zero calibration.
- **[NestedKV-adapted](../algorithms/nestedkv)** — Multi-scale ensembled prefill eviction (arXiv:2605.26678 — **no verified peer-reviewed venue, a one-time exception to this repo's standing rule**; see [`NEW_METHOD_SURVEY_V21.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/paper/research/surveys/NEW_METHOD_SURVEY_V21.md)). Every eviction method above commits to **one** importance signal; NestedKV keeps three parallel key-only continuum-memory statistics — stable/global, episodic/block-local, current/recent-window — and combines their per-token anomaly rankings via a head-adaptive blend plus a per-token surprise-gated route, all training-free. Runs **once at prefill**; unlike every other eviction method here, the cache is not bounded during decode (new tokens are simply appended, never rescored). Zero calibration.
- **[SnapKV-adapted](../algorithms/snapkv)** — Token eviction: the repo's first method to drop token positions entirely rather than compressing them. During prefill, the last `snap_obs_window` key rows act as proxy queries; their softmax attention over all prefix positions produces per-token importance scores. Only the top-`snap_budget` positions (plus `snap_n_sink` always-kept sink positions) are retained as fp16. Decode tokens are never evicted. The first method where the paper's actual signal (attention scores) is computable at the cache level — key-as-query proxy is stronger than key-norm-only methods. Zero calibration.
- **[StreamingLLM-adapted](../algorithms/streaming_llm)** — Structural positional eviction: the repo's first constant-memory cache. Keeps only the first `stream_n_sink` token positions (attention sinks, frozen forever) and the most recent `stream_window_size` positions (a rolling FIFO). All other positions are permanently dropped. Both prefill and decode tokens go through the same sink+window logic, so the cache never grows beyond `stream_n_sink + stream_window_size` positions regardless of generation length. Orthogonal to SnapKV-adapted (which evicts by score and grows during decode). Zero calibration.
- **[H2O-adapted](../algorithms/h2o)** — Cumulative attention-mass heavy-hitter eviction (ICLR 2024, arXiv:2306.14048). The library's original eviction baseline: the last `h2o_obs_window` key rows act as proxy queries, and their softmax attention accumulates into a running per-token score across the whole generation, not just one prefill pass. The lowest-scoring token is evicted whenever the cache is full. Several other eviction scorers above are pinned reductions of H2O at a limiting parameter (`chunk_size=1`, `cam_merge=drop`, `keyformer_tau=0`). Zero calibration.
- **[TOVA-adapted](../algorithms/tova)** — Memoryless current-step attention eviction (2024, arXiv:2401.06104). Where H2O accumulates a running score across every step, TOVA scores purely by the current step's attention weight and carries nothing forward — the token that mattered least *right now* is evicted, even if it mattered a lot earlier. Several methods above (`morphkv_window=1`, `kvzip_probe="latest"`) are pinned reductions of TOVA. Zero calibration.
- **[PyramidKV-adapted](../algorithms/pyramidkv)** — Layer-adaptive budget eviction (2024, arXiv:2406.02069). Takes H2O-adapted's cumulative-attention-mass scoring and replaces its single global budget with a pyramid of per-layer budgets — large in early layers, small in deep ones — while holding the average fixed so total cache size matches a uniform-budget baseline. Reduces exactly to H2O-adapted when the pyramid is flat (`pyramid_beta=1.0`). Zero calibration.
- **[SqueezeAttention-adapted](../algorithms/squeeze)** — 2D layer×token data-driven budget eviction (2024, arXiv:2404.04793). Extends PyramidKV's per-layer budget idea one step further: instead of assuming a pyramid shape, each layer's attention concentration is *measured* during prefill and a fixed total budget is reallocated toward broad (low-concentration) layers and away from concentrated ones. Reduces exactly to uniform H2O-adapted at `squeeze_strength=0.0`. Zero calibration.
- **[AnchorKV-adapted](../algorithms/anchorkv)** — Anchor-residual compression, no eviction (arXiv:2608.02901v1 — **no verified peer-reviewed venue, a second one-time exception to this repo's standing rule**; see [`NEW_METHOD_SURVEY_V22.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/paper/research/surveys/NEW_METHOD_SURVEY_V22.md)). Every eviction method above commits some tokens to being permanently discarded; AnchorKV keeps all of them, but at wildly non-uniform per-token cost — a handful of anchor tokens stored exactly, everything else reduced to one anchor index + one scalar coefficient, and a byte budget (set by `anchorkv_theta`) spent on quantized residuals for whichever tokens' approximation costs the most attention-output error (a first-order utility estimate, reusing SnapKV's key-as-query proxy for both anchor selection and residual scoring). Zero calibration.
- **[RocketKV-adapted](../algorithms/rocketkv)** — Two-stage compression (ICML 2025, arXiv:2502.14051): coarse permanent eviction (stage 1, directly reuses [SnapKV-adapted](../algorithms/snapkv)) followed by dynamic per-step top-k selection over the survivors (stage 2, Hybrid Sparse Attention — a two-dimensional approximate-attention-score reduction combining Quest-style paged sequence-dimension max/min summaries with SparQ-style head-dimension top-k channel selection). The paper's finding motivating this: dynamic top-k prediction is far more accurate once run over an already-filtered candidate set than over the full sequence, closing the accuracy gap either family hits alone at low token budgets. `rocketkv_compression_ratio` drives an adaptive formula (paper §3.6) that splits the target ratio across both stages automatically. Zero calibration. No RocketKV-MT (multi-turn) variant — see [issue #239](https://github.com/rajveer43/VeloxQuant-MLX/issues/239).
- **[AgeTieredKV](../algorithms/age-tiered)** — Position/age-gated 3-tier precision, no eviction (this library's own method, motivated by [issue #256](https://github.com/rajveer43/VeloxQuant-MLX/issues/256), not a paper port). KIVI's fp16 residual window is a two-level scheme (fp16 vs. one fixed bit-width) gated purely by recency; AMC's three discrete tiers are gated by per-token saliency instead of position. AgeTieredKV is the position-gated analogue of AMC's tier count: three quantization tiers (default 8-bit/4-bit/2-bit) chosen purely by `current_position - token_position` against two configurable age boundaries, reusing the same shared min/max group quantizer KIVI and AMC's lower tiers already use. Zero calibration.

### Calibration-required methods

These require a one-time calibration step, but deliver significantly better accuracy per bit.

- **[VecInfer](../algorithms/vecinfer)** — Product VQ with Metal-accelerated codebook lookup. Smooth scaling handles outlier dimensions. The fastest method at inference time due to fused SDPA kernels.
- **[RateQuant](../algorithms/ratequant)** — Mixed-precision allocation via reverse-waterfilling. Probes per-layer sensitivity and allocates more bits to layers that contribute most to output quality. Best accuracy per average bit.
- **[SpectralQuant](../algorithms/spectral)** — SVD rotation aligns key dimensions with high-variance directions. Separate signal/noise codebooks. Best for very long contexts (8k+).
- **[AMC-adapted](../algorithms/amc)** — Saliency-driven tiered rank + precision (arXiv:2607.10109 — **no verified peer-reviewed venue, a second one-time exception to this repo's standing rule**; the paper's hardware/RTL half, roughly Sections IV-V, is entirely out of scope for this software port). Every rank-adaptive method above (Palu) and every bit-width-adaptive method (KIVI, SKVQ, RateQuant) picks one axis; AMC is the first to drive **both** rank and bit-width from a **single** per-token L1-norm saliency score, via three discrete tiers (High: rank 128/16-bit, Mid: rank 43/8-bit, Low: rank 8/4-bit at head_dim=128). Requires an offline SVD/PCA channel-order calibration pass so the rank mask truncates the lowest-variance channels, not arbitrary ones. Unlike every eviction method above, AMC never drops a token — compression-only.
- **[A2ATS-adapted](../algorithms/a2ats)** — Windowed RoPE + query-aware retrieval VQ (He et al., **ACL 2025 Findings**, aclanthology.org/2025.findings-acl.644). Every VQ method above treats every position the same (VecInfer: no RoPE-awareness at all; CommVQ-adapted: codebook-constrained, one exact rotation per position regardless of distance); A2ATS instead gates position-encoding precision by each token's **distance** from the current decode step — exact within a trailing window, a single cheap stand-in outside it — plus **query-aware** codebook assignment for a subset of tokens. Requires an offline codebook calibration pass, same footgun class as VecInfer/CommVQ-adapted/Palu/SVDq/AMC. Best fit is long context with strong positional locality (chat, code, running summaries); a poor fit for genuine long-range retrieval, where the benchmark measures the approximation's cost as largest (8.3× vs 2.9× attention-score error). Tokens inside the window are handled exactly, so `a2ats_window` is a predictable accuracy/savings dial.

## Mixing methods

`CompositeQuantizer` is not a residual chain — it routes **outlier** and **inlier** channels of the same vector to two different quantizers (e.g. a high-bit quantizer for a few outlier channels, a low-bit one for the rest):

```python
import numpy as np
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
from veloxquant_mlx.quantizers.qjl import QJLQuantizer

total_dim = 128
outlier_idx = np.array([3, 17, 42, 88])  # channel indices treated as outliers

quantizer = CompositeQuantizer(
    outlier_quantizer=TurboQuantRVQ(d=len(outlier_idx), b=4, seed=42),
    inlier_quantizer=QJLQuantizer(d=total_dim - len(outlier_idx), m=64, seed=42),
    outlier_idx=outlier_idx,
    total_dim=total_dim,
)
```

## Per-model recommendations

| Model | Recommended algorithm | Notes |
|---|---|---|
| Llama 3.1/3.2 (7–8B) | TurboQuant RVQ 1-bit | Gaussian key distribution, zero setup |
| Mistral 7B / Mixtral | VecInfer 2-bit | Sliding window attention benefits from product VQ |
| Qwen 2.5 (7–14B) | SpectralQuant | Long-context optimised, benefits from SVD rotation |
| Phi-3 Mini | RaBitQ + CommVQ | Small head dim, CommVQ preserves RoPE exactly |
| Gemma 2B/7B | TurboQuant RVQ 2-bit | GQA benefits from slightly higher bit rate |
| Falcon 7B | RateQuant | Alibi positional bias; RateQuant adapts per-layer |

## Next steps

- Pick an algorithm and read its detailed page
- [mlx_lm integration guide](../guides/mlx-lm-integration)
- [Calibration guide](../guides/calibration)

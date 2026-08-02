# Implementation prompt — AMC-adapted (method #40, v0.38.0)

**DRAFT FOR REVIEW — do not execute until the user approves this prompt.**

## Phase 0 — venue gate (read this first)

Source: "Adaptive Model Compression (AMC): Saliency-Driven Resource Allocation
for Ultra-Low-Power Transformer Inference" (Hu, Yuan, Hu, Yin, Li, Suchter —
Apple), **arXiv:2607.10109v1 [cs.IR], submitted 2026-07-11**. Live-verified
today (2026-07-14): single version, no Comments field, no journal-ref — an
**unpublished preprint, 3 days old at verification time**.

This repo's standing rule (every method requires a live-verified
peer-reviewed venue before implementation) has been broken exactly once
before, for NestedKV, after nine surveys of deferral (V11–V19) and only at
the user's explicit direction (see `paper/research/surveys/NEW_METHOD_SURVEY_V21.md`).
The user has now directed a second exception for AMC. Per that precedent:

- State this plainly and prominently everywhere AMC is documented — module
  docstring, docs page banner, README, CHANGELOG, CITATIONS, EVIDENCE_TABLE.
  Do not soften or bury it.
- This is **method 2 of 40 without a verified venue**, not a new standing
  precedent. The next method survey reverts to requiring a verified venue.
- Also flag in the docs page: the paper is filed under `cs.IR` (Information
  Retrieval) despite being a hardware/systems paper (RTL, CMOS, systolic
  arrays) — an unusual category mismatch worth noting as a minor oddity, not
  a disqualifier.

## Phase 0.5 — scope cut (critical, read before Phase 1)

AMC's paper is a **hardware/software co-design**: roughly half the paper (all
of Sections IV–V, the RTL Verilog, the 45nm CMOS energy model, the
clock-gating systolic array, the SRAM narrow-write buffer, the EDAP/Pareto
silicon comparisons) targets a physical chip. VeloxQuant-MLX is a pure
software MLX library — there is no RTL/silicon layer here and no hardware
simulator in this repo. **None of that is portable and none of it should be
reimplemented.**

What ships is the **software saliency + tiered rank/precision assignment**
only — the part of the paper that is genuinely a KV-cache/activation
compression algorithm:

- Section II-A (Software Saliency Engine): L1-norm saliency score, the
  three-tier (High/Mid/Low) percentile partitioning, the query-aware semantic
  saliency blend (Eq. 3), the closed-loop threshold adaptation (Eq. 4-5).
- Section III (Adaptive Resource Scaling): the rank-masking mechanism (Eq. 6)
  and linear quantization mapping (Eq. 7), **and** the offline
  SVD/PCA post-training dimension alignment (Algorithm 1, Phase I) that makes
  truncating "dimensions r+1..D" safe.

Explicitly **not** implemented, and the docs must say so directly rather than
silently omitting them: the RTL/Verilog SAC controller, clock-gating,
systolic array column gating, SRAM bit-masked write buffer, all pJ/µJ energy
figures (Tables I/II, Eq. 8-17), the 45nm EDAP/Pareto hardware comparisons
(Fig. 4-5, Table V), and the LLM.int8/AWQ/H2O/StreamingLLM/Quest/RankDyna/DiP/
DynamicViT comparative numbers — those are the paper's own reported
figures on its own hardware/software baselines, never reproduced here. This
mirrors the existing convention (every "-adapted" method states its honest
deviation), just with a larger cut than usual because half the paper is a
different artifact class (silicon) entirely.

## Phase 1 — naming and scope

- Method id: `amc`
- Family: **calibration-required** (joins Palu/SVDq/RaBitQ — needs an offline
  SVD/PCA calibration pass before use), and also the first method that
  combines **per-token dynamic rank AND dynamic bit-width** driven by a
  single scalar saliency score (existing rank-only: Palu, CurDKV;
  bit-width-only-per-token: KIVI, SKVQ; per-layer bit allocation: RateQuant —
  none of those key on one saliency scalar to drive both axes at once).
- Framing: **"AMC-adapted (VeloxQuant-MLX implementation)"** — inspired by,
  not a faithful port of, "Adaptive Model Compression (AMC): Saliency-Driven
  Resource Allocation for Ultra-Low-Power Transformer Inference" (Hu, Yuan,
  Hu, Yin, Li, Suchter; arXiv:2607.10109, **no verified peer-reviewed venue as
  of 2026-07-14**), with the hardware/RTL half of the paper (Sections IV-V)
  explicitly out of scope (see Phase 0.5).
- Version target: **v0.38.0**. Method count after this ships: **40**.

## Phase 2 — offline calibration: `veloxquant_mlx/quantizers/amc_calibration.py`

Mirrors `group_head_svd` in `quantizers/palu.py` — reuse `mx.linalg.svd(x,
stream=mx.cpu)`, don't hand-roll SVD. This is a **one-time, pre-deployment**
pass, exactly matching the paper's Algorithm 1 "Phase I: Offline Post-Training
Structural Calibration."

```python
def amc_calibrate_channel_order(
    calib_activations: mx.array,  # [n_calib, D], representative activation sample for one layer
) -> mx.array:  # returns permutation indices [D], descending order of variance
    # SVD (or PCA via covariance eigendecomposition — SVD on centered
    # activations is equivalent and reuses the existing Palu pattern) over
    # calib_activations; sort columns of V by descending singular value.
    ...


def amc_permute_weights(weight: mx.array, perm: mx.array, axis: int) -> mx.array:
    # Static, offline reordering of a weight tensor's hidden-dim axis by perm.
    # Zero runtime cost — this permutation is baked into the stored weights
    # once, exactly as the paper states (Section III.4, "incurs strictly zero
    # computational, area, or energy overhead during real-time execution").
    ...
```

Because there is no RTL clock-gating here, "zero runtime overhead" for AMC in
this repo means: this function runs once during `KVCacheBuilder.for_model()`
setup (or is skippable via a `precomputed_perm` config field for
already-calibrated weights), never during the per-token hot path. Document
that explicitly — the paper's zero-overhead claim was about silicon, ours is
about it running outside the per-token loop.

## Phase 3 — `veloxquant_mlx/quantizers/amc.py`

**Saliency scoring** (paper Eq. 1-2, faithful port — pure vector math, no
adaptation needed):
```python
def amc_saliency(x: mx.array) -> mx.array:
    # x: [N, D] token activations (already channel-sorted by amc_calibrate_channel_order)
    # returns S: [N] = mean(|x|, axis=-1), clamped to [0, 1] per Eq. 2
    ...
```

**Query-aware semantic saliency** (Eq. 3, optional — expose as
`amc_use_query_saliency: bool = False` so the base magnitude-only path is the
default and matches the paper's primary reported numbers; the query-aware
variant is the paper's stated mitigation for "repetitive punctuation gets
high magnitude but low true importance"):
```python
def amc_query_aware_saliency(
    x: mx.array,  # [N, D] token activations
    keys: mx.array,  # [N, D] key projections (k_i)
    query: mx.array,  # [D] embedded query/prompt vector
    alpha: float = 0.5,
) -> mx.array:
    # S_i = alpha * mean(|x_i|) + (1 - alpha) * cosine_similarity(query, keys[i])
    # Use mx for the cosine term; guard zero-norm rows (degenerate all-zero
    # key edge case) same style as every other cosine-based method in this
    # repo (NestedKV's _cosine_anomaly, CurDKV's leverage-score guards).
    ...
```

**Tier assignment — this is the natural fit for `dsa.SortedChannelIndex`
(`veloxquant_mlx/dsa/heap.py`)**: the paper's percentile partition (top 20% /
next 30% / bottom 50%) is a **selection problem**, not a full sort. Use
`SortedChannelIndex.top_k(k)` (already backing an ordered-channel structure
elsewhere in this repo) to pull the top-`ceil(0.2N)` and next-`ceil(0.3N)`
saliency-ranked token indices in better-than-`O(N log N)` amortized fashion
where the existing structure supports incremental updates; if a per-call
one-shot partition is all that's needed (prefill-sized `N`, not a streaming
update), a single `mx.argpartition`-equivalent is acceptable too — but check
`SortedChannelIndex`'s `insert`/`top_k` API first and prefer it if it avoids
a full `O(N log N)` sort, consistent with the repo's existing DSA-first
convention (`ring_buffer.py` used by `polar_cache.py`/`qjl_cache.py` for
trailing-window state, `bit_pack.py` used by `turboquant_cache.py` for
sub-byte packing — reuse over reinvention is the pattern here).

```python
@dataclass
class AMCTierConfig:
    tier: int  # 0=High, 1=Mid, 2=Low
    rank: int  # 128 / 43 / 8
    bits: int  # 16 / 8 / 4


AMC_TIERS = (
    AMCTierConfig(tier=0, rank=128, bits=16),
    AMCTierConfig(tier=1, rank=43, bits=8),
    AMCTierConfig(tier=2, rank=8, bits=4),
)


def amc_assign_tiers(
    saliency: mx.array,  # [N]
    k_high: float = 0.20,
    k_mid: float = 0.30,
) -> mx.array:  # [N] int tier id (0/1/2), percentile-threshold assignment per token
    ...
```

**Sequence-adaptive closed-loop thresholds** (Eq. 4-5) — this is the natural
fit for `dsa.RingBuffer` (`veloxquant_mlx/dsa/ring_buffer.py`), same pattern
already used by `sliding_window_cache.py`/`polar_cache.py` for trailing-window
statistics:
```python
def amc_adaptive_thresholds(
    tau_high_base: float,
    tau_low_base: float,
    seq_variance: float,  # moving variance over the trailing activation window (RingBuffer-backed)
    calib_variance: float,  # nominal variance from the offline calibration set
    gamma: float = 0.1,
) -> Tuple[float, float]:
    # tau_H = tau_high_base * (1 - gamma * ln(seq_variance / calib_variance))
    # tau_L = tau_low_base  * (1 - gamma * ln(seq_variance / calib_variance))
    # Guard seq_variance/calib_variance <= 0 (degenerate all-zero activation
    # window) — clamp the ln() argument to a small epsilon floor, same defensive
    # style as every other log/div-by-zero guard in this repo.
    ...
```
Back this with a `RingBuffer` holding the trailing window of per-token
saliency values (window size configurable, default matching paper intuition
— reuse `RingBuffer`'s existing push/mean primitives rather than hand-rolling
a Welford update).

**Rank masking** (Eq. 6, faithful port — this is just Palu/CurDKV's existing
rank-truncation idea, but selected per-token via tier instead of per-layer or
per-head fixed):
```python
def amc_apply_rank_mask(x: mx.array, rank: int) -> mx.array:
    # x: [N, D] (channel-sorted). Zero out columns rank:D. Reuse the same
    # masking pattern as Palu's truncated projection — this is Hadamard
    # masking (Eq. 6), not a projection matrix, so it's simpler than Palu's
    # SVD projection: literal zeroing of the low-variance tail.
    ...
```

**Precision scaling / quantization** (Eq. 7) — reuse the repo's existing
quantization primitives, do not hand-roll a new fixed-point rounder. Check
`quantizers/svdq.py`'s `quantize_latents_mixed` and the core linear
quantizer used by RateQuant/KIVI for the established `round(x/S * 2^(b-1)) *
S/2^(b-1)` pattern already implemented in this repo — call the shared
function per-tier with `b ∈ {16, 8, 4}`, don't duplicate the rounding logic.

**Bit-packing for the 4-bit Low tier** — reuse `dsa.BitPackBuffer`
(`veloxquant_mlx/dsa/bit_pack.py`), exactly as `turboquant_cache.py` already
does for narrow-width storage. Mid tier (8-bit) can use a plain `mx.int8`-style
store (no packing needed, byte-aligned already); only the 4-bit Low tier
benefits from `BitPackBuffer`'s sub-byte packing.

Full function list, mirroring the repo's existing `__all__` convention:
```python
__all__ = [
    "AMCTierConfig",
    "AMC_TIERS",
    "amc_calibrate_channel_order",
    "amc_permute_weights",
    "amc_saliency",
    "amc_query_aware_saliency",
    "amc_assign_tiers",
    "amc_adaptive_thresholds",
    "amc_apply_rank_mask",
    "amc_quantize_tier",
    "amc_get_kv",
    "amc_fp16_bytes",
    "full_amc_fp16_bytes",
]
```

## Phase 4 — `veloxquant_mlx/cache/amc_cache.py`

Unlike NestedKV (one-shot prefill-only), AMC's paper describes **per-token
tiering applied continuously** — every token, prefill or decode, gets scored
and tiered (the paper's Fig. 3 timing diagram shows this happening every
cycle, not once). Mirror a **per-step scoring cache** (H2O/CurDKV's
`update_and_fetch` pattern), not SnapKV's/NestedKV's prefill-once pattern:
on every call (prefill batch of `S` tokens or decode's single token), compute
saliency, assign tier, apply rank mask + quantize, store. No eviction happens
in AMC — this is a **compression-only** method (all tokens retained, just at
different rank/precision), a genuinely new category compared to every
eviction method in this repo (H2O/SnapKV/PyramidKV/CurDKV/NestedKV all drop
tokens; AMC never does). State this plainly in the docs' mechanism-gap table:
AMC's family is "adaptive rank+precision," not "eviction."

Config fields (add to `KVCacheConfig` in `cache/base.py`):
```python
amc_k_high: float = 0.20  # top percentile -> High tier
amc_k_mid: float = 0.30  # next percentile -> Mid tier
amc_use_query_saliency: bool = False
amc_query_alpha: float = 0.5  # Eq. 3 balance coefficient
amc_adaptive_thresholds: bool = False  # Eq. 4-5 closed-loop adjustment
amc_threshold_window: int = 64  # RingBuffer window for seq_variance
amc_gamma: float = 0.1  # threshold attenuation factor
amc_calib_variance: float | None = (
    None  # from offline calibration; required if amc_adaptive_thresholds=True
)
```

Wire into `base.py`: add `"amc"` to the `Literal`, add the config block, add
`from veloxquant_mlx.cache.amc_cache import AMCKVCache` import, add factory
branch `elif config.method == "amc": cache = AMCKVCache(config)`, extend the
unknown-method error string.

Cache properties (mirror existing per-tier-accounting methods, e.g. RateQuant):
`amc_kept_bytes` (sum across the three tiers' actual bit-widths), `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, tier distribution counters
(`tokens_high`, `tokens_mid`, `tokens_low`) for observability/testing.

## Phase 5 — honesty crux (module docstring AND docs page)

1. **Unpublished preprint, 3 days old at verification, no venue** — state
   first, matching the venue-exception banner (Phase 0).
2. **Hardware/RTL half of the paper entirely out of scope** — no clock-gating,
   no systolic array, no 45nm silicon, no pJ/µJ energy numbers reproduced.
   State this is roughly half the source paper, not a minor omission.
3. **Compression-only, not eviction** — AMC never drops tokens, only reduces
   their rank/precision; this is a structurally different family from every
   eviction method in this repo.
4. **Query-aware saliency (Eq. 3) and closed-loop thresholds (Eq. 4-5) are
   opt-in, off by default** — state that the paper's headline 59.2%
   energy/2.24x throughput/3.6% accuracy numbers are the paper's own
   hardware-measured figures under its specific 3-layer synthetic setup
   (num-samples=4000, seq-len=32, vocab-size=16), not reproduced by this
   software port; VeloxQuant-MLX reports its own offline synthetic benchmark
   instead (Phase 6).
5. **Offline SVD/PCA calibration required** for the rank-masking to be safe
   (same category of requirement as Palu/SVDq/RaBitQ) — using `amc` without
   running `amc_calibrate_channel_order` first on representative data means
   truncating arbitrary, not lowest-variance, channels. Document this as a
   sharp footgun, same tone as Palu's calibration-required warnings.
6. **cs.IR category mismatch** noted as a minor oddity (Phase 0), not a
   disqualifier.
7. Nothing validated on real models/hardware here — synthetic offline
   benchmark only, same convention as every prior method.

## Phase 6 — tests

`veloxquant_mlx/tests/quantizers/test_amc.py` +
`veloxquant_mlx/tests/quantizers/test_amc_calibration.py` +
`veloxquant_mlx/tests/cache/test_amc_cache.py`. Aim for ~24-28 tests,
mirroring the CurDKV/NestedKV split. Critical mechanism tests:

- `test_saliency_matches_l1_norm_definition`: direct numeric check of Eq. 1-2
  on a hand-constructed activation matrix.
- `test_tier_assignment_respects_percentiles`: confirm ~20/30/50 split on a
  large-N synthetic saliency distribution (within reasonable tolerance for
  discrete boundaries).
- `test_high_tier_tokens_survive_full_precision`: construct tokens with
  known high/low saliency, confirm the high-saliency ones get
  `rank=128, bits=16` and low-saliency ones get `rank=8, bits=4`.
- `test_calibration_orders_channels_by_variance`: synthetic activation matrix
  with known high-variance and low-variance columns; confirm
  `amc_calibrate_channel_order` puts high-variance columns first.
- `test_rank_mask_zeros_low_variance_tail_after_calibration`: combine
  calibration + rank masking; confirm the truncated tail corresponds to the
  genuinely low-variance channels, not arbitrary ones (this is the direct
  proof of Phase 0.5/crux point 5 — the reason calibration is required at
  all).
- `test_query_aware_saliency_downweights_high_magnitude_irrelevant_tokens`:
  construct a token with high `|x|` but low cosine similarity to the query,
  and a token with moderate `|x|` but high query similarity; confirm the
  query-aware score (Eq. 3) reorders them relative to magnitude-only scoring
  — the direct proof this mode does something.
- `test_adaptive_thresholds_widen_on_high_variance_sequences`: feed a
  high-variance activation window, confirm `tau_H`/`tau_L` drop (widening the
  High/Mid allocation), and the inverse for a low-variance/repetitive window.
- `test_adaptive_thresholds_guard_degenerate_zero_variance`: `seq_variance`
  or `calib_variance` at/near zero — confirm no NaN/inf from the `ln()` term.
- `test_bitpack_roundtrip_low_tier`: confirm `BitPackBuffer`-backed 4-bit
  storage round-trips correctly through `amc_quantize_tier`.
- `test_no_eviction_all_tokens_retained`: run a full prefill+decode sequence,
  confirm token count in cache always equals tokens seen (the direct proof of
  crux point 3 — compression-only, never eviction).
- `test_determinism`: same input twice → identical tier assignment and
  quantized output.
- Standard suite: init, byte accounting across all three tiers,
  `for_model` config propagation (all 8 `amc_*` fields), factory dispatch,
  factory smoke test with `compression_ratio > 1.0`.

Run until fully green; fix root causes. Expect at least one non-obvious
mechanism bug around the calibration/rank-mask interaction (permutation
applied at calibration time must be threaded consistently through both the
weight tensors and the runtime activations, or the mask truncates the wrong
channels) — debug with a direct reproduction script outside the test suite
if a test fails non-obviously, same practice as prior methods.

## Phase 7 — benchmark (`benchmark_scripts/benchmark_amc.py`)

Offline synthetic benchmark, same shape as `benchmark_curdkv.py`/
`benchmark_nestedkv.py`: compare AMC vs the closest existing baseline
(RateQuant, as the other per-token/per-layer adaptive-bit-width method) and
vs a uniform-precision baseline (16-bit fixed, matching the paper's own
"Uniform Baseline" comparator in Fig. 4) on 2-3 synthetic geometries that
isolate the saliency-driven tiering signal — e.g. a "sparse-outlier" geometry
(few high-magnitude tokens among many low ones, where tiering should
concentrate rank/bits correctly) and a "uniform-magnitude" geometry (where
tiering has nothing to exploit, a stress/honesty case). Report
compression-ratio-vs-reconstruction-error, not accuracy-vs-energy (no energy
model in this port — see Phase 0.5). Commit deterministic results JSON
(`benchmark_scripts/amc_benchmark_results.json`), verify determinism by
diffing two runs. Write honest closing prose if the uniform-magnitude
geometry shows no benefit — expected and should be stated plainly, not hidden.

## Phase 8 — docs (`docs-site/docs/algorithms/amc.md`)

Mirror `curdkv.md`/`nestedkv.md` structure: title/method-id, **venue-status
banner leading with "unverified preprint, 3 days old"** (Phase 0), **scope-cut
banner leading with "hardware/RTL half of paper not implemented"** (Phase
0.5) — both banners prominent, not buried under the mechanism section.
Mechanism-gap table (contrast vs Palu/CurDKV [rank-only], KIVI/SKVQ
[bit-width-only], RateQuant [per-layer bit allocation] — AMC is the first
single-saliency-score-drives-both-rank-and-bits method). Honesty crux (7
points, Phase 5). Usage snippet with all 8 config fields, showing both the
default magnitude-only path and the opt-in query-aware + adaptive-threshold
path. How-it-works walkthrough (offline calibration → per-token saliency →
tier assignment → rank mask → quantize). Byte accounting. Benchmark section.
Evidence section (cite arXiv:2607.10109 with the preprint caveat repeated).
When-to-use table — emphasize "compression-only, never evicts" as the
differentiator from every eviction-family method.

Update `docs-site/sidebars.ts` (add `'algorithms/amc'` after
`'algorithms/nestedkv'`), `docs-site/docs/algorithms/overview.md`
("thirty-nine"→"forty"; add AMC row under **Calibration-required methods**;
add method-family bullet for the new "adaptive rank+precision" category),
`docs-site/docs/changelog.md` (new `## v0.38.0 — Latest` section, demote
v0.37.0), cross-link from `palu.md`'s or `ratequant.md`'s "See also" section.

## Phase 9 — README/CHANGELOG/CITATIONS/pyproject/EVIDENCE_TABLE

- `README.md`: 39→40 everywhere; add AMC row to the appropriate method table
  (verify placement with `grep -n "^###"` first — don't repeat the CurDKV
  mis-placement mistake noted in the NestedKV prompt); update "40th method"
  CTA.
- `CHANGELOG.md`: new `[0.38.0]` entry with an explicit **"Venue exception"**
  subsection (second one ever — cross-reference NestedKV's as the first) and
  a **"Scope cut"** subsection explaining the hardware/RTL half is omitted.
- `CITATIONS.md`: "39 compression methods"→"40"; add AMC bibliography entry
  marked as an unpublished arXiv preprint, no venue tag.
- `pyproject.toml`: version →`0.38.0`, description "...to NestedKV"→"...to
  AMC", "39"→"40".
- `paper/research/EVIDENCE_TABLE.md`: append rows under `## AMC-adapted
  saliency-driven tiered rank+precision (0.38.0) — added rows`, including the
  venue exception and the scope cut as documented, deliberate findings.

## Phase 10 — landing page

`landing/index.html`: meta description, hero pill/roll-call (40, "· AMC"
appended), what's-new list, filter-bar count, appropriate category
`cat-count`, remove `new-pill` from NestedKV card, add new `#algo-amc` card
(NEW pill), picker card + code-tab button + code panel.
`landing/assets/main.js`: `initBadgeTyping` text → "v0.38.0 — AMC-adapted
saliency-driven tiered rank+precision shipped".

## Phase 11 — full verification

Run new tests to green; re-run full existing suite (expect zero regressions
beyond documented pre-existing flakes); build wheel + `twine check`; build
docs site; grep sweep for stale "39"/"thirty-nine"; factory smoke test
(`KVCacheConfig(method="amc", ...)` → `KVCacheBuilder.for_model()` →
`compression_ratio > 1.0`).

## Phase 12 — release layer (CHAT TEXT ONLY — never execute)

Standing rule, unchanged: the user reviews and runs all git/publish commands
themselves. After implementation, give the v0.38.0 release sequence as chat
text — do NOT execute any git add/commit/tag/push/gh release/twine yourself.

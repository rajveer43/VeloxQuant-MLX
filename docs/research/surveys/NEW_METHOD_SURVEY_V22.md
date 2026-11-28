# New-method survey V22 — AnchorKV, a second explicit venue exception

## Standing discipline, and why this survey breaks it a second time

Every method shipped before NestedKV (38 total) required a live-verified
peer-reviewed venue before implementation. V21 broke that rule once, at the
user's explicit direction, for NestedKV (arXiv:2605.26678, still a
single-version preprint at the time). That was documented as a **one-time**
exception, not a change to the standing rule.

AnchorKV (arXiv:2608.02901v1, submitted 2026-08-03) is checked the same way:
**single-version preprint, "Preprint. Under review." stated on the paper's
own first page, no journal-ref, no Comments field.** No venue as of
2026-08-20 (17 days after submission — too soon for a venue decision either
way). At the user's explicit direction, this survey grants a **second**
one-time exception, for this method only, following the exact precedent set
by V21: ship it as an openly-labeled unpublished-preprint adaptation, state
that plainly everywhere (module docstrings, cache docstring, tests,
benchmark script, this survey), and leave the standing rule itself
unchanged for every future method.

## Why AnchorKV, mechanically

Source: "AnchorKV: Anchor-Residual KV Cache Compression" (Khalaf, Shamshoum,
Hodos, Sieradzki, Schuster — Technion), arXiv:2608.02901v1, 2026-08-03. Full
text read directly (all sections, including Appendix A's codec/layout
details) — not taken from the abstract alone.

**The paper's core idea:** every method already in the repo picks one of two
extremes. Eviction (H2O, SnapKV, PyramidKV, AdaKV, NestedKV, ...) drops
tokens outright — unbounded compression in principle, but a dropped token is
gone for every future query, not just the one that scored it low. Uniform
quantization (KIVI, GEAR, SVDq, KVQuant, ...) keeps every token but pushes
bit-width down directly, and accuracy falls off a cliff below ~2 bits,
capping practical compression well short of eviction's ratios. AnchorKV's
claim is that neither axis is the right one to push: instead of lowering
*every* token's precision uniformly, store a **small number of tokens
exactly** (anchors) and represent **every other token as a cheap projection
onto its nearest anchor** (one index + one scalar coefficient — a few bytes,
not a full vector), then spend whatever byte budget remains on **quantized
residuals for only the tokens whose approximation error costs the most
attention-output error**. No position ever leaves the softmax — the
compressed cache is a perturbation of the exact one, not a truncation of it
(paper Eq. 7).

### The exact mechanism (from the paper, Section 3)

**1. Anchor-residual representation (Section 3.1).** For vectors
`x_1, ..., x_S` (rows of K or V, one head): a subset `A` (anchors) is stored
exactly. Every other vector is assigned to the anchor maximizing absolute
cosine similarity (`Eq. 1`: `a(i) = argmax_a |<x_i, x_a>| / (||x_i|| ||x_a||)`),
then represented by its orthogonal projection onto that anchor's span
(`Eq. 2`: `γ_i = <x_i, x_a(i)> / ||x_a(i)||²`, `x̃_i = γ_i x_a(i)`,
`r_i = x_i - x̃_i`). Reconstruction (`Eq. 3`) adds a stored residual only for
positions in a selected subset `R`; everywhere else, the vector is
reconstructed purely from its anchor projection.

**2. Anchor selection (Section 3.2).** Per KV head: the trailing `W`
positions are always anchors (recency window + the source of the SnapKV-style
observation-window proxy queries + anchors that cost no extra selection
step). Of the remaining `k - W` slots, a fraction `ρ` goes to the
highest-scoring earlier positions under SnapKV's own observation-window
attention scoring (Li et al. 2024, already in this repo as
`quantizers/snapkv.py`), and the rest is sampled **uniformly at random** —
the paper's own stated rationale: attention-based selection finds tokens the
observation queries used directly, but an anchor also needs to be a good
*direction* for OTHER tokens to project onto, and uniform sampling improves
that directional coverage independent of what the observation window
happened to score highly. Keys are projected **pre-RoPE** (Section 3.2,
"Key projection is performed before RoPE") because RoPE rotation weakens the
cosine-similarity structure the assignment step relies on; RoPE is applied
during decode reconstruction since it distributes linearly over the
anchor+residual decomposition (`R_t K_t = R_t K̃_t + R_t r_t^K`).

**3. Attention-output-aware residual scoring (Section 3.3, Eq. 6).** A
first-order Taylor expansion of the softmax attention output under the
key/value reconstruction error yields separate key-channel and value-channel
utility terms per position, each estimated from the `m` observation-window
queries of that KV head:
`u_t^K = (1/m) Σ_w (α_t^(w))² (q^(w)·R_t r_t^K / √D)² ||V_t - y^(w)||²`,
`u_t^V = (1/m) Σ_w (α_t^(w))² ||r_t^V||²`. Larger utility means storing that
token's residual recovers more attention-output error; residuals go where
they buy the most, not simply on the largest-norm or most-attended tokens
(the paper's own ablation, Section 4.3, shows attention-only and
residual-norm-only rankings both underperform the combined utility, and the
gap widens as the byte budget tightens).

**4. Byte-budgeted allocation (Section 3.4, Eq. 9).** The user sets a single
knob `θ` — the fraction of the uncompressed fp16 cache to retain. Anchors
(stored exactly) and per-token metadata (index + coefficient, both sides)
are charged first; whatever bytes remain buy residual slots,
`N = max(0, floor((θ·M_full - M_base) / b_res))`, split roughly evenly
between K and V, with residual candidates pooled **across all KV heads in
the layer** so heads with larger estimated error get a larger share
(mirrors NestedKV's cross-head competition, Section 3.4 of that paper — the
same pooling shape, applied here to a fundamentally different
representation).

**5. Residual codec (Section 3.1, Appendix A.1).** Randomized Hadamard
rotation (spreads residual energy across coordinates before quantizing),
per-token absmax normalization, and a 4-level (2-bit) Lloyd-Max codebook fit
for a unit-Gaussian source — codes packed 4/byte.

## Genuinely new mechanism axis vs. the 40-method roster (pre-AnchorKV)

Every quantization method in the repo (KIVI, GEAR, SVDq, KVQuant, PALU,
CacheGen, MiniCache, ZipCache, ...) lowers bit-width or rank **uniformly or
near-uniformly across all tokens**. Every eviction method (H2O, SnapKV,
PyramidKV, AdaKV, NestedKV, CurDKV, ...) picks a subset of tokens to keep at
full precision and **discards the rest entirely**. AnchorKV does neither:
it keeps 100% of tokens (like quantization) but assigns them **wildly
non-uniform, data-dependent per-token byte cost** driven by two decoupled
decisions — which anchor a token projects onto (structural, geometric) and
whether it additionally gets a residual (utility-ranked, like eviction's
top-k selection, but the "losing" tokens are coarsened, not dropped). This
is the closest existing method to `nsnquant`/`comm_vq`'s codebook-style
coding (tokens coded against a shared reference), but the paper itself notes
the distinction (Section 2, "Shared Representations"): a shared codebook
entry is IDENTICAL for every token assigned to it, collapsing their
distinguishability inside the softmax, whereas AnchorKV's per-token
coefficient `γ_i` means no two non-anchor tokens ever collapse to the same
reconstructed vector even when they share an anchor.

## Cache-only feasibility

Confirmed from the paper directly (Section 3): "Compression runs once, at
the end of prefill, in three steps... All operations are per layer and per
KV head." No retraining, no model modification — the attention arithmetic
itself is unchanged (Section 1: "leaving the attention arithmetic
unchanged"). This matches every one-shot-prefill cache wrapper already in
this repo (SnapKV-adapted, NestedKV-adapted): the compression decision is
made once, from K/V alone, at the moment prefill ends.

## What we will NOT implement (stated up front, mirrors every honesty crux)

- **Key-as-query proxy**, same convention already used by `snapkv.py` /
  `nestedkv.py`. The paper's anchor scoring (Section 3.2) AND its utility
  estimate (Eq. 6) both use the prompt's true observation-window *query*
  vectors — not visible to a cache wrapper, which only sees K/V at
  `update_and_fetch`. We substitute the trailing `window` *key* rows as
  proxy queries throughout, for both purposes. This is a strictly larger
  reuse of the existing key-as-query approximation than any prior method in
  this repo (SnapKV uses it once for eviction scoring only; AnchorKV here
  uses it for BOTH anchor selection and residual utility).
- **No fused decode kernel.** The paper's steady-state memory story (Section
  4.4, "~19× reduction in decode peak memory") depends entirely on a
  FlashAttention-style tiled kernel that reconstructs each key/value tile
  only when consumed, never materializing the dense cache. This PR
  reconstructs a dense fp16 K/V tensor eagerly, in plain MLX ops, once per
  `update_and_fetch` call — correct (the compressed representation and its
  byte accounting are exact), but NOT the paper's memory-savings story
  during decode. A future PR could pursue a fused Metal kernel; explicitly
  out of scope here, same category of limitation as every non-fused-kernel
  method already in this repo (the fused Metal path is opt-in Phase-1/2
  infrastructure that most methods don't yet use).
- **One-shot prefill compression, not decode-loop rescoring.** Same
  convention as SnapKV-adapted / NestedKV-adapted: anchor selection, per-token
  assignment, and residual-budget allocation all run once at the end of
  prefill. Decode tokens are appended exactly, at fp16, never retroactively
  re-anchored. Unlike eviction methods, this is NOT a behavioral regression
  against the paper — the paper's own design (Section 3.2) is also a
  one-shot prefill compressor ("Compression runs once, at the end of
  prefill").
- **Anchor budget as a fraction of context length**, `k = ⌊S · anchorkv_anchor_frac⌋`
  with default `anchorkv_anchor_frac = 1/128`, matching the paper's own
  stated default (`k = S/128`, Section 4.1). At very small `S` (below ~a few
  hundred tokens with `head_dim` in the tens, as in this repo's synthetic
  tests and offline benchmark), anchors + per-token metadata alone can
  consume the entire `θ`-derived byte budget, leaving zero residual slots —
  confirmed directly via `anchorkv_budget_slots` and documented in
  `benchmark_scripts/benchmark_anchorkv.py`'s `THETAS` comment. This is a
  property of running the paper's byte accounting at a toy scale far below
  its own target regime (128K-token contexts, `head_dim=128`), not a
  deviation from the formula (Eq. 9 is implemented exactly).
- **No model-level validation.** The paper's RULER/LongBench/Needle-in-a-
  Haystack numbers (Llama-3.1-8B/70B-Instruct, Mistral-Small-3.1-24B-
  Instruct, NVIDIA A100-80GB) are the paper's own. This PR ships an
  offline-synthetic cache-primitive-level benchmark
  (`benchmark_scripts/benchmark_anchorkv.py`) comparing AnchorKV's
  reconstruction fidelity against H2O at matched byte budgets — the same
  "not a model benchmark, explicitly labeled as such" convention every
  prior method's benchmark script follows.
- **Residual codec reuses existing primitives rather than re-deriving new
  ones.** `HadamardPreconditioner` (already used by `turboquant_mse.py`,
  `turboquant_rvq.py`, `nsnquant.py`, etc.) and `CodebookFactory.create(
  "gaussian", ...)` (the same 4-level-Lloyd-Max-for-unit-Gaussian
  construction the paper itself specifies, Appendix A.1) are reused
  directly rather than hand-rolled — exact in spirit, not an approximation
  of the paper's own codec choice.

## Byte accounting

- `anchorkv_bytes` — true compressed storage: anchors (fp16, both K and V)
  + per-token metadata (int32 index + fp32 coefficient, per non-anchor
  token, per side) + packed residuals (2 bits/coordinate + fp16 scale),
  summed over every (batch, head).
- `full_seq_bytes` — hypothetical fp16 K + V cost for the same tokens.
- `compression_ratio` — `full_seq_bytes / anchorkv_bytes`; > 1 means
  savings.
- `tokens_kept` — always equals `tokens_total`. Unlike every eviction
  method's `tokens_kept <= tokens_total`, this is the direct, mechanical
  proof that AnchorKV never drops a token — enforced by
  `test_prefill_never_drops_tokens` and
  `test_tokens_kept_equals_tokens_total_always` in
  `tests/cache/test_anchorkv_cache.py`.

## Recommendation

Implement now, as a second explicit one-time venue exception (issue #237),
labeled openly everywhere per the V21 precedent. `MethodFamily.HYBRID`
(keeps every token like quantization, but does prefill-time budgeted
allocation like eviction) — the same family classification already used for
`a2ats`. Standing venue-verification rule is unaffected for every method
after this one.

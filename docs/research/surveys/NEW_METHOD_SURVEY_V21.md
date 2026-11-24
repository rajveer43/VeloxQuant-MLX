# New-method survey V21 (post-pause) — NestedKV, chosen with an explicit venue exception

## Standing discipline, and why this survey breaks it once

Every method shipped so far (38 total) required a live-verified peer-reviewed
venue before implementation — the discipline that correctly deferred MorphKV
across V13–V15 (implemented in V16 once ICML acceptance was confirmed) and
that has deferred NestedKV itself across nine straight surveys, V11 through
V19, and again in the V20 pause note.

Re-checked live today (2026-07-14): **NestedKV (arXiv:2605.26678) is still a
single-version preprint** — submitted 2026-05-26, no Comments field, no
journal-ref, no second revision. It has not acquired a venue in the ~7 weeks
since submission.

At the user's explicit direction, this survey breaks the venue-verification
rule **once, for this method only**, and ships NestedKV as an openly-labeled
unpublished-preprint adaptation. This is a deliberate, one-time exception, not
a change to the standing rule — every future method returns to requiring a
verified venue. The implementation, docs, changelog, and evidence table must
all state this plainly and not obscure it: this is the first of 39 methods
that does not trace to a peer-reviewed venue.

## Why NestedKV, mechanically

Source: "NestedKV: Nested Memory Routing for Long-Context KV Cache
Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu — HKUST Guangzhou /
Jimei University), arXiv:2605.26678v1, 2026-05-26. Full PDF read directly
(pages 1-8) for exact formulas — not taken from the abstract alone.

**The paper's core idea:** every eviction method already in the repo commits
to a *single* notion of token importance — cumulative attention (H2O),
prefill-window attention (SnapKV), layer-adaptive budget (PyramidKV), key-norm
proxy (KNorm/Keyformer/L2Norm-family), reconstruction-reliance (KVzip), or
CUR/leverage-score joint key-value structure (CurDKV). NestedKV's claim is
that a *single* anchor is structurally insufficient: a token can be important
because it's globally unusual (document-level), locally episodic
(segment/turn-level), or immediately relevant (recent-stream), and which one
matters shifts across documents, tasks, and compression ratios. It keeps
**three parallel key-only statistics at three time scales** and combines them
per-head and per-token rather than picking one.

### The exact mechanism (from the paper, Section 2)

All scores computed on **normalized keys** `k̂ᵢ = kᵢ/‖kᵢ‖₂` — directional
structure in key space only. No attention, no query, no value dependence for
scoring (values are retained/evicted alongside their key, never touched by
the score itself — same convention as every other eviction method here).

**1. Three continuum-memory statistics** (per layer, per KV head):
- Stable (global mean over the whole prefilled context):
  `μ_s = (1/N) Σⱼ k̂ⱼ`
- Episodic (mean over the local block containing token i, block size
  `b = clip(⌊N/32⌋, 128, 256)`):
  `μ_e(i) = (1/|B(i)|) Σ_{j∈B(i)} k̂ⱼ`
- Current (mean over a trailing causal window, `W = 64`):
  `μ_c(i) = (1/(i-ℓᵢ+1)) Σ_{j=ℓᵢ}^{i} k̂ⱼ`, `ℓᵢ = max(1, i-W+1)`

**2. Per-scale anomaly scores** (negative cosine similarity — high score =
more anomalous = more worth keeping):
`a_s(i) = -cos(k̂ᵢ, μ_s)`, `a_e(i) = -cos(k̂ᵢ, μ_e(i))`, `a_c(i) = -cos(k̂ᵢ, μ_c(i))`
Each is min-max normalized within its head to a common `[0,1]` scale
(`ã_s, ã_e, ã_c`). The first `n_sink=4` positions are pinned via a large
constant score (sink protection, same convention as StreamingLLM/H2O/etc.
already in the repo).

**3. Head-adaptive blend** (which scale is trustworthy on *this* head):
Per head, discriminative gap for scale k: `Δ_k = mean(top-10%(ãₖ)) - mean(bottom-10%(ãₖ))`.
Blend weight: `w_k = softmax(log(w_k⁰) + β·Δ_k)` over `k ∈ {s,e,c}`, with a
fixed log-prior `(w_s⁰, w_e⁰, w_c⁰) = (0.4, 0.4, 0.2)` and fixed temperature β
shared across the model (not trained/learned — a closed-form softmax over a
statistic computed from the current sequence). Blended score:
`a_blend(i) = w_s·ã_s(i) + w_e·ã_e(i) + w_c·ã_c(i)`

**4. Surprise-gated routing** (which combination rule applies to *this*
token): compression-induced surprise `s(i) = std(ã_s(i), ã_e(i), ã_c(i))` —
low surprise means the three scales agree (blend is safe); high surprise
means they disagree (any average risks being dragged down by whichever scale
finds the token typical). Routed/winner score: `a_win(i) = max(ã_s(i), ã_e(i), ã_c(i))`.
Sigmoid gate: `α(i) = σ(κ·(s(i) - τ))` with fixed threshold τ and sharpness κ
(shared constants, not trained). Final score:
`a*(i) = (1-α(i))·a_blend(i) + α(i)·a_win(i)`

**5. Head-wise memory competition** (adaptive per-head budget, replacing a
uniform per-head slice): pool `a*` residuals across all heads in a layer,
take the layer's global top-`B_ℓ` `(head, position)` pairs, subject to a small
per-head safeguard minimum so no head is starved to zero. This determines
each head's *effective* budget `B_h` for the layer, rather than a fixed
uniform `budget / n_heads` split.

**6. Eviction operator:** `TopB` on `a*` per head at its allocated `B_h`,
after sink pinning — same "keep top scores, drop the rest" shape as every
other eviction cache in this repo.

### Ablation (paper Table 2, Qwen3-4B RULER 4k, r=0.75)
Removing the head-adaptive allocation alone: -8.41 points. Removing the
three-scale continuum score alone (replaced with single-anchor key
distinctiveness): -7.99 points. Removing both: -19.10 (super-additive — the
two components compensate for each other when only one is removed). This is
directly useful evidence to carry into this repo's own honesty-crux
documentation: both components are load-bearing, not decorative.

## Genuinely new mechanism axis vs. the 38-method roster

Every eviction method already shipped commits to one importance signal.
NestedKV's distinguishing structure is **multiple simultaneous key-only
scales, each producing an independent per-token ranking, combined by a
closed-form (not learned) two-axis rule** — per-head scale-reliability
weighting, and per-token cross-scale-disagreement routing. Nothing in the
roster does multi-scale ensembling of independent importance signals; every
existing method computes one score per token from one signal. This is a
distinct axis from CurDKV's joint key+value leverage score (still a single
score per token from one decomposition), from KVzip's single reconstruction
probe, and from PyramidKV/SqueezeAttention's layer-budget adaptation (which
adapts total *budget* per layer, not per-token score combination).

## Cache-only feasibility

Confirmed from the paper directly (Section 2.1): "The model parameters,
attention function, and retained value vectors are unchanged; the method only
determines which cached positions remain." Scoring uses only normalized keys
— no true query/attention-score access at all (stronger than H2O/SnapKV/
CurDKV's key-as-query proxy — NestedKV needs no query proxy whatsoever, since
it never approximates attention in the first place). No multi-session state.
No retraining. Confirmed training-free: "no training or LLM modification,"
restated in the abstract, introduction, and conclusion.

## What we will NOT implement (stated up front, mirrors every honesty crux)

- Not validated on any hardware/model in this repo — the paper's numbers
  (Qwen3/Llama-3.2 family, RULER/LongBench/LooGLE/MMLU-Pro/InfiniteBench) are
  the paper's, on NVIDIA L20 GPUs, not reproduced here. Our benchmark will be
  a synthetic offline benchmark isolating the multi-scale-ensembling signal,
  same convention as CurDKV/KVTC/every prior method's benchmark script.
  Documented explicitly as such.
- The paper's `TopB` head-wise competition (component 5) operates jointly
  across all heads in a layer at once (a single global top-B_ℓ pool across
  `(head, position)` pairs, after a per-head guaranteed floor of
  `⌊α_s·(1-r)·N⌋` tokens, `α_s=0.20` — confirmed exact value from Appendix
  A). Adapted per-layer-cache-wrapper here exactly as specified — this
  repo's cache wrapper already iterates per-layer, so no simplification is
  needed for this part, unlike some prior methods. Documented as faithfully
  adapted, not simplified.
- **Full read of the paper's Appendix A ("Inference pipeline") reveals
  NestedKV is a one-shot prefill compressor, not a per-decode-step recurring
  eviction loop** — quoted directly: "NestedKV does not recompute scores,
  scale reliabilities, or routes for retained prompt tokens as new tokens are
  generated; newly decoded tokens are appended normally." This means, unlike
  every other eviction method in this repo (H2O, CurDKV, StreamingLLM), the
  cache does NOT stay bounded during decode — it is bounded only at the end
  of prefill. This is a faithful port of the paper's actual design, not a
  simplification, but it is a genuine behavioral difference worth flagging
  prominently in the docs (when-to-use table) so users aren't surprised by
  unbounded growth during a very long generation.
- The gate constants — `β=3.0` (blend temperature), `τ=0.60` (surprise gate
  threshold, applied to per-head min-max-normalized and mean-centered
  surprise scores), `κ=10.0` (gate sharpness), log-prior `(0.4, 0.4, 0.2)`,
  `α_s=0.20` (safeguard floor fraction) — all confirmed directly from the
  paper's Appendix A ("Hyperparameters"), not guessed. State this as a
  stronger-than-usual fidelity point in the docs.
- No PyTorch/CUDA reference kept; pure MLX from the start, per this repo's
  standing pattern.

## Recommendation

Implement NestedKV as method #39, explicitly flagged everywhere as
**"NestedKV-adapted (VeloxQuant-MLX implementation) — inspired by an
unpublished preprint, arXiv:2605.26678, no peer-reviewed venue as of
2026-07-14."** This is a one-time, user-directed exception to the
venue-verification rule that has governed all 38 prior methods; the next
method survey reverts to requiring a verified venue.

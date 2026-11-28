# New-method survey V23 — RocketKV, a normal-track method (verified venue)

## Standing discipline — no exception needed here

V21 (NestedKV) and V22 (AnchorKV) each granted a one-time exception to ship
an unpublished-preprint method. RocketKV needs no such exception:
**Behnam, Fu, Zhao, Tsai, Yu, Tumanov — "RocketKV: Accelerating Long-Context
LLM Inference via Two-Stage KV Cache Compression" — Proceedings of the 42nd
International Conference on Machine Learning (ICML 2025), PMLR 267, 2025**
(arXiv:2502.14051v3, last revised 2025-08-13). Live-verified peer-reviewed
venue, same standing as SnapKV/H2O/PyramidKV/Quest/SparQ/DuoAttention/A2ATS —
a normal-track method under this repo's existing rule, not a third exception.

## Why RocketKV, mechanically

Full text read directly (all sections through §5, plus the abstract and
related-work comparison in §2) — not taken from the abstract alone.

**The paper's core idea, motivated by a concrete measurement (§3.1, Figure
2):** for a random attention head on Mistral-7B-Instruct-v0.2 over 200
qasper questions, the CDF of maximum sequence length reaches 25,000 tokens,
but the CDF of the number of *unique* top-256 indices selected across *all*
decoding steps for that same head only reaches ~1200. In other words: an
oracle running exact top-k attention at every step only ever needs a
surprisingly small union of token positions across the WHOLE generation,
much smaller than the sequence length. This means a permanent-eviction
method with a budget around that union size should, in principle, match
oracle accuracy — but Figure 1 shows every existing eviction method (SnapKV,
DuoAttention) and every existing dynamic-selection method (Quest, SparQ)
falls well short of oracle accuracy once the token budget drops under
~1024, even though the oracle itself stays flat down to 256.

**RocketKV's resolution:** don't ask a single mechanism to do both jobs.
Run coarse, cheap PERMANENT eviction first (stage 1) to remove clearly
unimportant tokens — this doesn't need to be perfectly accurate, just
directionally right, because stage 2 gets to refine within whatever
survives. Then run DYNAMIC top-k selection (stage 2) over the much smaller
surviving set at every decode step — and because the candidate set is now
small, the same class of one-dimensional approximation (Quest's page
min/max, SparQ's head-magnitude selection) that failed over the full
sequence becomes accurate enough over the filtered set. The paper states
this directly (§3.1, closing paragraph): "an ideal solution would be to
perform permanent KV cache eviction with a larger token budget first and
then conduct dynamic KV token selection on the remaining KV tokens... This
fusion evicts unimportant tokens and also makes the dynamic selection more
accurate."

### The exact mechanism (from the paper, §3.2-3.6)

**1. Stage 1 — SnapKV, adopted directly (§3.3).** The paper does not propose
a new eviction mechanism; it uses SnapKV (Li et al., NeurIPS 2024,
arXiv:2404.14469) verbatim, with one tuning change: pooling kernel size is
set much larger than SnapKV's own LongBench default (63 vs. 7) because here
SnapKV is only doing coarse-grain filtering, not the final compression. For
GQA models, per-group (not per-head) token selection is adopted from Ada-KV
(Feng et al. 2024) to avoid redundant storage across heads sharing a KV
group.

**2. Stage 2 — Hybrid Sparse Attention, HSA (§3.4, Algorithm 1, Figure 4).**
Three steps, per decode step, per query:
  - **Step 1 (maintained continuously):** group the surviving keys into
    consecutive pages along the sequence dimension; store element-wise
    max (`K_max`) and min (`K_min`) per page, laid out aligned along the
    head dimension for efficient gathering in Step 2. Updated incrementally
    as new decode keys arrive.
  - **Step 2 (per query `q`):** select the top-`k1` head-dimension positions
    by `sum(|q|, dim=group)` (SparQ-style channel selection, computed per
    attention GROUP so every head in a GQA group makes the same channel
    choice). For those `k1` channels, `g <- sign(sum(q[i1], dim=group))`
    picks, PER CHANNEL, whether `K_max` or `K_min` participates — channel
    `c` uses `K_max[c]` when `q[c] >= 0`, `K_min[c]` when `q[c] < 0` — then
    dots the resulting per-channel-selected page vector with `q` to get an
    approximate per-page attention score. **This per-channel selection is
    the crux the paper's own pseudocode encodes and is easy to get wrong**:
    a naive `max(q . K_max, q . K_min)` (max of two WHOLE dot products) is
    NOT equivalent and is not even a valid upper bound when `q` has
    mixed-sign entries (verified numerically during this implementation —
    see Adaptation notes below). Top-`k2` pages by this approximate score
    are selected.
  - **Step 3:** fetch the exact key/value rows for the selected pages
    (`k2 * page_size` tokens) and run standard dense attention over just
    that subset.

**3. Adaptive compression decomposition (§3.6).** The user sets one number,
the overall target compression ratio `c = S / token_budget`. A split factor
`r = min(0.2 + 0.06 * log2(c), 0.8)` (floored at `0.2` in this
implementation, matching the paper's intent that `r` should not go
negative for very small `c`) divides `c` into a stage-1 ratio `c^r` and a
stage-2 ratio `c^(1-r)`. The paper's own worked example: `c=64` gives
`r=0.56`, so stage 1 gets `64^0.56 ≈ 10.3x` and stage 2 gets
`64^0.44 ≈ 6.2x`. The intuition (paper's own words): at small `c`, don't
evict too aggressively in stage 1 (information loss is unrecoverable);
as `c` grows, SnapKV's exact-attention-score selection is more reliable
than HSA's approximation, so shift more of the compression burden onto
stage 1. Within HSA, the stage-2 ratio is split evenly across sequence
(page size, rounded up to an integer) and head dimensions.

**4. RocketKV-MT, the multi-turn variant (§3.5).** For multi-turn
conversations, permanent eviction is a liability — a token dropped because
it looked unimportant to turn 1's query might be exactly what turn 5 needs
(the paper cites Li et al. 2025's SCBench finding that KV token importance
varies significantly across turns). RocketKV-MT never evicts: it keeps
every input token in memory across turns, but still restricts stage 2's
dynamic selection to a filtered subset (recomputed by re-running SnapKV
filtering, not eviction, on the full accumulated history each turn) so
decode-time compute stays bounded. This trades away RocketKV's storage
savings (Table 1: `1 + 2/c^((1+r)/2)` vs. RocketKV's `1/c^r + 2/c^((1+r)/2)`)
for accuracy nearly matching oracle top-k on SCBench (§4.2, Figure 6).

## Genuinely new mechanism axis vs. the 41-method roster (pre-RocketKV)

Every eviction method in this repo (H2O, SnapKV, PyramidKV, AdaKV, NestedKV,
CurDKV, ChunkKV, ...) makes its retention decision ONCE, at or shortly after
prefill, and never revisits it. Every dynamic-selection-flavored method
already here reasons along a SINGLE dimension: none exist yet in this
repo — Quest-style page-summary approximation and SparQ-style head-magnitude
approximation are both referenced in this repo's docs/citations as related
work but neither ships as a standalone `KVCacheConfig(method=...)` option.
RocketKV is the first method in this repo that (a) composes a permanent
eviction pass with a genuinely dynamic, per-decode-step re-selection pass
over the survivors, and (b) that dynamic pass itself reduces along BOTH the
sequence and head dimensions simultaneously (the two-dimensional HSA
reduction), rather than picking one axis the way Quest or SparQ alone would.
This is a structurally different shape from AnchorKV-adapted (§ V22): where
AnchorKV keeps every token forever at non-uniform cost, RocketKV evicts
permanently in stage 1 (same category as SnapKV) and then additionally
narrows WHICH surviving tokens get attended to at each individual decode
step — a per-step attention-time selection, not a per-token storage
decision.

## Cache-only feasibility

Confirmed directly from the paper (§3.2, §3.7): "RocketKV is fully
compatible with FlashAttention because it does not modify attention in the
prefill phase" and both stages "seamlessly integrate... because all
operations are symmetric across attention heads/groups." Stage 1 is exactly
the existing `SnapKVKVCache`'s one-shot-at-prefill design; stage 2 needs
only the incoming key/value (to maintain the paged summary) and a proxy for
the query (see Adaptation notes) at each `update_and_fetch` call — no model
modification, no retraining, matching every eviction/hybrid cache wrapper
already in this repo.

## What we will NOT implement (stated up front)

- **Key-as-query proxy**, inherited from SnapKV-adapted for stage 1
  (unchanged — this repo already ships that approximation) and REUSED for
  stage 2's per-decode-step HSA query, since a cache wrapper's
  `update_and_fetch` never receives the model's true query vector, only
  keys and values. This is not a new category of approximation — it is the
  same convention SnapKV-adapted, A2ATS-adapted, H2O-adapted, and
  AMC-adapted already use, applied at a second point (every decode step,
  not just prefill).
- **Page-granularity index selection, not token-granularity.** The paper's
  own Algorithm 1 selects whole pages (`k2` of them) as the sparse-attention
  set, so the retained token count after HSA is `k2 * page_size`, not
  exactly `k2`. This matches the paper's own design (Step 3 gathers "the
  original key and value vectors from the predicted k2 indices" where the
  indices are page indices throughout the algorithm), not a simplification
  introduced here.
- **No fused kernel.** The paper's reported 3.7x A100 / 3.3x H100 end-to-end
  decode speedups and 32.6% peak memory savings (§4.3) come from a
  `gpt-fast`-based low-latency Python implementation with FlashAttention
  underneath — not a custom fused CUDA/Metal kernel (the paper explicitly
  notes "it could be further improved with customized CUDA kernels"). This
  repo's implementation reconstructs/gathers the HSA-selected subset in
  eager MLX ops each decode step, the same limitation class as every other
  non-fused-kernel method here (AnchorKV-adapted, NestedKV-adapted).
- **No RocketKV-MT.** The multi-turn variant requires a multi-turn
  conversation harness (repeatedly re-filtering an ever-growing retained
  history across turns) that this repo has no test/benchmark scaffolding
  for yet. Tracked as follow-up work in
  [issue #239](https://github.com/rajveer43/VeloxQuant-MLX/issues/239);
  single-turn `RocketKVKVCache` is the unit shipped in this PR.
- **No model-level validation.** The paper's LongBench/Needle-in-a-Haystack/
  RULER/SCBench numbers (Llama3.1-8B-Instruct, Mistral-7B-Instruct-v0.2,
  LongChat-7B-v1.5-32k, NVIDIA A100/H100) are the paper's own. This PR ships
  an offline-synthetic cache-primitive-level benchmark
  (`benchmark_scripts/benchmark_rocketkv.py`) comparing RocketKV's
  reconstruction fidelity against SnapKV-only AT THE SAME FINAL TOKEN
  BUDGET — the fair comparison the paper's own claim is actually about
  (not a comparison against SnapKV run only to RocketKV's smaller stage-1
  sub-budget, which would understate SnapKV's achievable accuracy and was
  caught and corrected during this benchmark's own development).

## Byte accounting

- `stage1_bytes` — fp16 bytes for kept tokens after stage-1 eviction (same
  accounting convention as `SnapKVKVCache.evicted_key_bytes` /
  `evicted_value_bytes`, combined).
- `stage2_aux_bytes` — HSA's paged max/min auxiliary summary storage
  (fp16-packed, `n_pages * head_dim * 2 (max+min) * 2 bytes`).
- `full_fp16_bytes_total` — hypothetical fp16 K + V cost without any
  compression.
- `compression_ratio` — `full_fp16_bytes_total / (stage1_bytes +
  stage2_aux_bytes)`; > 1 means savings. Table 1's formula
  (`1/c^r + 2/c^((1+r)/2)`, relative to full fp16) is implemented exactly
  via the adaptive split.
- `tokens_kept` / `tokens_total` — diagnostic counters for stage-1 eviction
  only (stage 2 never drops a STORED token, only narrows what one decode
  step's attention touches — the eviction/selection distinction Table 1
  draws between RocketKV's storage savings and every method's shared
  traffic savings).

## Recommendation

Implement now, as a normal-track method (issue #239) — no venue exception
needed, unlike V21/V22. `MethodFamily.HYBRID` (two-stage: an eviction stage
plus a dynamic-selection stage, distinct in kind from AnchorKV's
"HYBRID-because-keeps-everything-at-non-uniform-cost" framing but sharing
the family label since neither is pure eviction nor pure quantization).

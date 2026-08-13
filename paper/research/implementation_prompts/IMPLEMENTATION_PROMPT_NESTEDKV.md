# Implementation prompt — NestedKV-adapted (method #39, v0.37.0)

**Read `paper/research/surveys/NEW_METHOD_SURVEY_V21.md` first — it has the
full mechanism derivation, exact formulas, and the explicit one-time
venue-exception rationale.** This prompt is the execute-cold spec; the survey
is the "why."

**This is a one-time deviation from standing practice**: NestedKV
(arXiv:2605.26678) is still an unpublished preprint (single v1, 2026-05-26,
no Comments/journal-ref field, re-verified live 2026-07-14). Every one of the
38 methods shipped so far required a verified peer-reviewed venue first; this
one ships anyway, at the user's explicit direction, and must say so plainly
and prominently everywhere it's documented — README, docs page, CHANGELOG,
CITATIONS, EVIDENCE_TABLE. Do not soften or bury this fact. The next method
survey reverts to requiring a verified venue; this is not a new precedent.

## Phase 1 — naming and scope (already decided, stated here for reference)

- Method id: `nestedkv`
- Family: token eviction/merging (14th method in that family after CurDKV)
- Framing: **"NestedKV-adapted (VeloxQuant-MLX implementation)"** — inspired
  by, not a faithful port of, "NestedKV: Nested Memory Routing for
  Long-Context KV Cache Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu;
  arXiv:2605.26678, **no verified peer-reviewed venue as of 2026-07-14**).
- Version target: **v0.37.0**. Method count after this ships: **39**.

## Phase 2 — `veloxquant_mlx/quantizers/nestedkv.py`

**Critical integration-pattern correction, confirmed from the paper's
Appendix A ("Inference pipeline"), quoted directly: "NestedKV is implemented
as a training-free prefill compressor: after the prompt is encoded, each
layer computes per-scale anomalies from cached keys, combines them with the
head-adaptive blend and surprise-gated route... allocates head-wise memory
budgets, and removes low-scoring entries before decoding. The head-adaptive
blend weights and surprise gates are computed once during this prefill-time
compression step and then kept fixed throughout autoregressive decoding...
NestedKV does not recompute scores, scale reliabilities, or routes for
retained prompt tokens as new tokens are generated; newly decoded tokens are
appended normally and remain available to subsequent decoding steps."**

This is structurally different from every eviction method already in this
repo (H2O, CurDKV, SnapKV, PyramidKV, etc.), all of which re-run their
eviction/scoring loop on every incoming token including during decode.
NestedKV is a **one-shot prefill-time compression** followed by an
**uncompressed decode phase** (new tokens are simply appended, never scored
or evicted). Do not build a per-step recurring eviction loop for this method
— that would be a real, avoidable deviation from the paper, not a forced
adaptation. Mirror SnapKV-adapted's structure instead of H2O's/CurDKV's for
the phase split (check `quantizers/snapkv.py` / `cache/snapkv_cache.py` for
the existing prefill-once / decode-append pattern already in this repo before
writing this module), but keep the three-scale scoring machinery below for
the one-time prefill compression step.

The per-token scoring needs THREE statistics computed once over the full
prefill key stream, mirrored in a dataclass state that, after prefill,
freezes into a simple append-only buffer for decode.

```python
@dataclass
class NestedKVState:
    keys: mx.array | None  # [n_kept, D] fp16
    values: mx.array | None  # [n_kept, D] fp16
    mu_stable: mx.array | None  # [D] float32 running mean of ALL normalized keys ever kept+seen
    n_seen_for_stable: int  # count backing mu_stable's running mean (Welford-style update)
    n_sink: int
    budget: int
    block_size: int  # b = clip(floor(N/32), 128, 256) — recomputed as N grows; see note below
    window_size: int  # W = 64, fixed
    n_sink_score: float  # large constant used to protect sink positions, e.g. 1e9
```

Key design decision (state carefully — this replaces an earlier draft of this
prompt that incorrectly assumed an incremental/streaming design before the
paper's Appendix A was read in full; corrected here):

**Good news: the paper's own design is a one-shot prefill compressor (see the
integration-pattern note above), which matches this library's `N`-fixed
prefill call exactly — no streaming/Welford-mean adaptation is needed at
all.** At the moment `nestedkv` compression runs, the full prefill key
sequence `[N, D]` is already sitting in memory (arrives as one `[B,H,S,D]`
call with `S=N` at prefill, same as `SnapKVKVCache`'s prefill-detection
pattern). Compute `μ_s`, `μ_e(i)` for every `i`, and `μ_c(i)` for every `i`
directly and exactly as the paper specifies, over that one fixed array — a
faithful, non-approximated port of Section 2.2's formulas, computed once.

- `μ_s = mean over all N prefill keys` — plain batch mean, exactly as
  written, no adaptation needed.
- `μ_e(i) = mean over block B(i)`, `b = clip(⌊N/32⌋, 128, 256)`, contiguous
  blocks over the **fixed prefill-time token indices** — exactly as written,
  no adaptation needed (unlike a hypothetical incremental version, there is
  no eviction-collapses-indices problem here, because the block means are all
  computed before any eviction happens).
- `μ_c(i) = trailing window mean, W=64` — exactly as written.
- After scoring and the one-shot eviction (Phase 2/3's `TopB` + head-wise
  competition), the retained tokens are stored in whatever order they had
  (or re-sorted by original position, matching this repo's existing
  no-RoPE-remapping convention — retained tokens keep their relative order,
  same as every other eviction method here); newly generated decode tokens
  are then simply **appended, unscored, never evicted** — the state after
  prefill becomes a fixed-size buffer plus a growing decode tail, exactly
  mirroring `SnapKVKVCache`'s post-prefill behavior. Check
  `cache/snapkv_cache.py` for the exact append-only decode-phase pattern to
  reuse here.

**Per-token anomaly scores** (all on normalized keys `k̂ = k / ‖k‖₂`):
```python
def _cosine_anomaly(k_hat: mx.array, mu: mx.array) -> mx.array:
    # k_hat: [n, D], mu: [D] (broadcast) or [n, D] (per-token, e.g. mu_e/mu_c)
    # returns [n] = -cosine_similarity, i.e. high score = anomalous = keep-worthy
    ...
```
Compute `a_s, a_e, a_c` once for all `N` prefill tokens, then min-max
normalize each of the three vectors independently across the full prefill set
(`ã_s, ã_e, ã_c` on `[0,1]`) — a single whole-set pass, exactly as the paper
specifies (no incremental/per-step approximation needed, since this is a
one-shot prefill computation, not a recurring per-decode-step update).

**Head-adaptive blend** (per head — but note the quantizer module here is
single-head; the cache wrapper loops heads, same as H2O/CurDKV):
```python
def _head_adaptive_blend(a_s_hat, a_e_hat, a_c_hat, beta, prior=(0.4, 0.4, 0.2)) -> mx.array:
    # Delta_k = mean(top-10%(a_k_hat)) - mean(bottom-10%(a_k_hat)) for k in {s,e,c}
    # w_k = softmax(log(prior_k) + beta * Delta_k)
    # returns a_blend[i] = w_s*a_s_hat[i] + w_e*a_e_hat[i] + w_c*a_c_hat[i]
    ...
```
Guard the top-10%/bottom-10% split for small `n` (same guard style as
CurDKV's `rank_cap = min(rank_cap, n, d)`): when `n` is too small for a
non-degenerate 10% split, fall back to `top_p = bot_p = max(1, n // 10)`
element(s), and if `n == 1`, all three deltas are 0 and the blend reduces to
the fixed prior — document this as the small-n floor, same category of edge
case as CurDKV's degenerate-collapse fix.

**Surprise-gated routing:**
```python
def _surprise_gate(a_s_hat, a_e_hat, a_c_hat, a_blend, tau, kappa) -> mx.array:
    # s(i) = std(a_s_hat[i], a_e_hat[i], a_c_hat[i])  (population std over the 3 values, per token)
    # a_win(i) = max(a_s_hat[i], a_e_hat[i], a_c_hat[i])
    # alpha(i) = sigmoid(kappa * (s(i) - tau))
    # returns a_star(i) = (1 - alpha(i)) * a_blend(i) + alpha(i) * a_win(i)
    ...
```
Fixed constants exposed as config, **now confirmed from the paper's Appendix A
("Hyperparameters") — use these exact values, not placeholders**:
`nestedkv_beta = 3.0` (head-adaptive blend temperature), `nestedkv_tau = 0.60`
(surprise gate threshold — note: the paper applies this to surprise scores
that have been **min-max normalized within each head and mean-centered**
first, not raw `std(a_s,a_e,a_c)` — implement that normalization step, don't
skip it), `nestedkv_kappa = 10.0` (gate sharpness), log-prior
`(w_s⁰, w_e⁰, w_c⁰) = (0.4, 0.4, 0.2)` (already used above), block size
schedule `clip(⌊N/32⌋, 128, 256)` (already specified), window `W=64`
(already specified), 4 pinned sink tokens (matches this repo's existing
`n_sink=4` convention across every other method — no change needed).

**Eviction:** one-shot, per head, at the end of the prefill compression step:
after sink pinning (first `n_sink` positions get `a_star = float("inf")` —
excluded from blend/gate math, injected only at the final `TopB` compare
step, same convention as H2O/CurDKV), keep the top-`B_h` scoring tokens
(sinks always included), drop the rest. This happens exactly once per
sequence, not repeatedly per decode step — see the integration-pattern note
above.

**Head-wise memory competition (component 5)** — this is the one part that
is NOT single-head-independent like H2O/CurDKV. It must happen in the CACHE
WRAPPER (Phase 3), not this quantizer module, since it needs visibility across
all `H` heads for a batch element simultaneously. This quantizer module
(`nestedkv.py`) should expose a **pure per-head scoring function** that
returns `a_star` for all currently-held tokens in one head (no eviction
decision baked in), and a separate **`nestedkv_allocate_head_budgets`**
free function taking the per-head `a_star` arrays (as a list of length H) and
the layer's total budget `B_layer`, returning a list of H integer budgets
`B_h` such that `sum(B_h) == B_layer`, via the paper's exact two-step rule
(Appendix A, confirmed): **first**, for a sequence of current length `n` and
eviction ratio `r` (`r = 1 - B_layer/(n*H)` at the head-average level — derive
`r` from the configured budget, don't require the caller to pass it
separately), each head is guaranteed to keep its own top
`⌊α_s · (1-r) · n⌋` tokens by score, where `α_s = 0.20` (the paper's stated
safeguard fraction — expose as `nestedkv_safeguard_alpha: float = 0.20`,
replacing the placeholder `nestedkv_min_per_head_budget` int field). **Then**,
the remaining budget (`B_layer` minus the sum of all per-head guaranteed
floors) is allocated by pooling all NOT-yet-guaranteed `(head, position,
a_star)` triples across heads and taking the global top-K by `a_star` until
the remaining budget is exhausted. This is a cleaner and more faithful
adaptation of the paper's `TopB_{B_ℓ}{(h,i): a_{h,i}}` global-pool rule than
an ad hoc "top-up from surplus" scheme — implement this two-step version, not
the earlier surplus-donation sketch.

Full function list to implement, mirroring CurDKV's `__all__` pattern (note:
`nestedkv_update` is now a one-shot prefill call plus a plain append path for
decode, not a recurring per-step eviction update like H2O/CurDKV's):
```python
__all__ = [
    "NestedKVState",
    "init_nestedkv_state",
    "nestedkv_score",  # per-head, one-shot over the full prefill: returns a_star for all N tokens
    "nestedkv_allocate_head_budgets",  # cross-head budget competition (component 5), one-shot
    "nestedkv_compress_prefill",  # per-head: score + evict once, called only at prefill
    "nestedkv_append_decode",  # per-head: plain unscored append for decode tokens
    "nestedkv_get_kv",
    "nestedkv_fp16_bytes",
    "full_nestedkv_fp16_bytes",
]
```

## Phase 3 — `veloxquant_mlx/cache/nestedkv_cache.py`

Mirror `cache/snapkv_cache.py`'s prefill/decode phase split (not H2O's/
CurDKV's per-step loop). `update_and_fetch` must detect prefill (`S > 1`,
first call) vs decode (`S == 1`, subsequent calls) — check exactly how
`SnapKVKVCache` distinguishes these before writing this, and reuse the same
detection convention. On prefill, per batch element `b`:
1. Run `nestedkv_score` for every head `h` to get each head's `a_star` vector
   over all `N` prefill tokens.
2. Call `nestedkv_allocate_head_budgets` once across all `H` heads for this
   batch element to get `[B_0, ..., B_{H-1}]` summing to the layer's total
   budget (`nestedkv_budget * H` — keep `nestedkv_budget` as a
   per-head-equivalent default, so the total layer budget passed to the
   allocator is `nestedkv_budget * H`, preserving apples-to-apples comparison
   with H2O/CurDKV's per-head `budget` config field and this repo's existing
   `compression_ratio` accounting convention).
3. Call `nestedkv_compress_prefill` per head with its allocated `B_h` — this
   runs once, ever, per sequence.

On decode (every subsequent call), call `nestedkv_append_decode` per head —
a plain unscored append, no scoring, no eviction, exactly matching the
paper's "newly decoded tokens are appended normally" behavior. If the
sequence somehow exceeds the original prefill budget purely from decode
growth (a long decode run), that's expected and matches the paper's design
(the paper does not re-evict during decode) — document this plainly as the
one deliberate quirk of the one-shot design: **NestedKV's total cache size
can grow unboundedly during a very long decode phase**, unlike every
per-step method (H2O/CurDKV/StreamingLLM) which stays bounded throughout.
State this clearly in the honesty crux (Phase 4) and the docs page's
when-to-use table — NestedKV is a prefill-compression method, not a
constant-memory streaming method, and that's a real, paper-faithful
trade-off, not an implementation gap.

Config fields (add to `KVCacheConfig` in `cache/base.py`):
```python
nestedkv_budget: int = 512  # per-head-equivalent budget (total layer budget = this * n_heads)
nestedkv_n_sink: int = 4
nestedkv_window: int = 64  # W, current-memory trailing window
nestedkv_beta: float = 3.0  # head-adaptive blend temperature (paper's stated default)
nestedkv_tau: float = 0.60  # surprise gate threshold (paper's stated default)
nestedkv_kappa: float = 10.0  # surprise gate sharpness (paper's stated default)
nestedkv_safeguard_alpha: float = (
    0.20  # per-head guaranteed-floor fraction (paper's stated default, Appendix A)
)
```

Wire into `base.py`: add `"nestedkv"` to the `Literal`, add the config block
(after `curdkv_*`), add `from veloxquant_mlx.cache.nestedkv_cache import
NestedKVKVCache` import, add factory branch `elif config.method ==
"nestedkv": cache = NestedKVKVCache(config)`, extend the unknown-method error
string to include `nestedkv`.

Cache properties (mirror CurDKV exactly): `nestedkv_kept_bytes`,
`full_seq_bytes`, `compression_ratio`, `tokens_seen`, `tokens_kept`.

## Phase 4 — honesty crux (put this in the module docstring AND the docs page)

State these plainly, mirroring CurDKV's 6-point crux:
1. **Unpublished preprint, no verified venue** — the headline exception for
   this method. State this first, not buried.
2. **One-shot prefill compression, unbounded during decode** — faithfully
   matches the paper's own design (Appendix A), but is a real structural
   difference from every other eviction method in this repo (H2O, CurDKV,
   StreamingLLM, etc.), which stay bounded throughout decode. State plainly
   that NestedKV's cache can grow during a very long decode run — this is a
   property of the paper's method, not a shortcut taken in this port.
3. Episodic block computed over fixed prefill-time positions (exactly as the
   paper specifies — no adaptation needed here, since there's no
   eviction-collapses-indices problem in a one-shot compressor). State that
   this part is a faithful, non-approximated port, in contrast to point 2.
4. Gate/blend constants (`beta=3.0`, `tau=0.60`, `kappa=10.0`,
   `safeguard_alpha=0.20`) — all taken directly from the paper's Appendix A;
   state that these are the paper's own defaults, not this implementation's
   guesses (a rarer, stronger form of fidelity than most adapted methods in
   this repo get to claim — say so).
5. Key-only, no query/attention access at all (stronger than H2O/SnapKV/
   CurDKV's key-as-query proxy — NestedKV needs no proxy since it never
   approximates attention).
6. Nothing validated on real models/hardware here — paper's RULER/LongBench/
   LooGLE/MMLU-Pro/InfiniteBench numbers are the paper's (NVIDIA L20,
   Qwen3/Llama-3.2 family), not reproduced. Synthetic offline benchmark only,
   same convention as every prior method.

## Phase 5 — tests (`veloxquant_mlx/tests/quantizers/test_nestedkv.py`,
`veloxquant_mlx/tests/cache/test_nestedkv_cache.py`)

Aim for ~24 tests total, mirroring CurDKV's split (quantizer-level mechanism
proofs + cache-level integration/factory tests). Critical mechanism tests to
include (don't skip these — they're the actual proof the method does
something, same bar as CurDKV's `test_h2o_blind_spot_on_same_planted_geometry`):

- `test_three_scales_diverge_on_planted_geometry`: construct a synthetic key
  stream with a clear global outlier (anomalous vs `μ_s` only), a clear local
  episodic anomaly (anomalous vs `μ_e` only, blends into the recent window),
  and a clear recency anomaly (anomalous vs `μ_c` only) — confirm each
  survives eviction preferentially under a budget tight enough to force
  choices, using randomized (not strictly alternating) arrival order, per the
  CurDKV lesson on recency-tie confounds (see survey V19 / CurDKV docs).
- `test_single_anchor_blind_spot`: prove a single-scale-only scorer (e.g. just
  `μ_s`) fails to retain a token that's only locally/recently anomalous —
  the direct analogue of CurDKV's H2O-blind-spot test, proving the
  multi-scale ensembling does something a single anchor can't.
- `test_surprise_gate_routes_on_disagreement`: construct a token where the
  three per-scale scores strongly disagree and confirm `a_star` tracks
  `a_win` (not `a_blend`) in that regime, and a token where they agree and
  confirm `a_star` tracks `a_blend`.
- `test_head_adaptive_blend_upweights_discriminative_scale`: construct a head
  where one scale cleanly separates tokens and the other two don't; confirm
  the blend weight favors the discriminative scale.
- `test_head_wise_budget_competition_reallocates`: construct two heads with
  very different score distributions (one concentrated/high-residual, one
  flat/low-residual) and confirm the competition step gives more of the total
  budget to the concentrated head, subject to the safeguard floor.
- `test_safeguard_floor_respected`: confirm no head ever drops below its
  `⌊nestedkv_safeguard_alpha · (1-r) · n⌋` guaranteed floor even under
  extreme cross-head score imbalance.
- `test_degenerate_all_identical_keys_no_nan`: all-identical key rows (cosine
  undefined direction edge case, zero-variance) — confirm no NaN/crash.
- `test_small_n_blend_falls_back_to_prior`: `n=1` or very small `n` — confirm
  the top-10%/bottom-10% guard doesn't crash and blend reduces to the prior.
- `test_decode_tokens_appended_unscored`: after prefill compression, append
  several decode-phase tokens and confirm they are never dropped/rescored —
  cache size after decode equals `B_h (post-prefill) + n_decode_tokens`, not
  clamped back down to `B_h`. This is the direct proof of the one-shot
  design (point 2 in the honesty crux) — don't skip it.
- Standard suite: init, sink protection, prefill budget enforcement, byte
  accounting (`nestedkv_fp16_bytes`/`full_nestedkv_fp16_bytes`), determinism
  (same prefill input twice → identical output), prefill-then-decode-appended
  loop, `for_model` config propagation (all 7 `nestedkv_*` fields), factory
  dispatch, factory smoke test with `compression_ratio > 1.0` on both K and V
  measured immediately after prefill (before any decode growth).

Run until fully green; fix root causes (mirror CurDKV's process: expect at
least one non-obvious mechanism bug from the head-wise-competition
reallocation logic or the prefill/decode phase-detection boundary — don't be
surprised if the first implementation degenerates in some edge case the way
CurDKV's hard-rank-cutoff did; debug with direct reproduction scripts outside
the test suite when a test fails non-obviously, same as done for CurDKV).

## Phase 6 — benchmark (`benchmark_scripts/benchmark_nestedkv.py`)

Offline synthetic benchmark, same shape as `benchmark_curdkv.py`: compare
NestedKV vs the single-best-fitting existing baseline (H2O, as the other
cumulative/adaptive-scoring method) on 2-3 synthetic geometries that isolate
the multi-scale-ensembling signal (a "global-outlier-only" geometry, a
"local-episodic-only" geometry, a "recency-only" geometry, and ideally a
"mixed" geometry combining all three so no single existing method's signal
dominates). Report retention rates per class. Commit deterministic results
JSON (`benchmark_scripts/nestedkv_benchmark_results.json`), verify determinism
by diffing two runs. Write the honest closing prose — if results don't cleanly
match the design target for some geometry, say so plainly and explain why,
same discipline as the CurDKV benchmark's `correlated`-geometry honesty note.

## Phase 7 — docs (`docs-site/docs/algorithms/nestedkv.md`)

Mirror `curdkv.md`'s structure: title/method-id/venue-status banner (lead with
the "no verified venue" fact, don't bury it), mechanism-gap table (contrast
vs every other eviction method — H2O/SnapKV/PyramidKV/CaM/Keyformer/MorphKV/
KVzip/CurDKV, all single-signal scorers), honesty crux (6 points from Phase
4), usage snippet with all 7 config fields, how-it-works walkthrough (three
scales → per-scale anomaly → head-adaptive blend → surprise-gated routing →
head-wise competition → eviction), byte accounting, benchmark section, mixed
adaptation notes, evidence section, when-to-use table.

Update `docs-site/sidebars.ts` (add `'algorithms/nestedkv'` after
`'algorithms/curdkv'`), `docs-site/docs/algorithms/overview.md` ("thirty-eight"
→"thirty-nine"; add NestedKV row + method-family bullet), `docs-site/docs/
changelog.md` (new `## v0.37.0 — Latest` section, demote v0.36.0), and add a
cross-link from `curdkv.md`'s or `h2o.md`'s "See also" section mentioning
NestedKV as the multi-scale-ensembling contrast.

## Phase 8 — README/CHANGELOG/CITATIONS/pyproject/EVIDENCE_TABLE

- `README.md`: 38→39 everywhere (headline, bullet, TOC, integration line,
  References line), changelog badge 0.36.0→0.37.0, add NestedKV row to Token
  Eviction & merging table (verify correct table placement immediately —
  don't repeat the CurDKV mis-placement mistake; `grep -n "^###"` first to
  confirm section boundaries before inserting), fix "39th method" CTA →"40th".
- `CHANGELOG.md`: new `[0.37.0]` entry mirroring CurDKV's, with an explicit
  **"Venue exception" subsection** stating NestedKV is unpublished, why it
  shipped anyway (user-directed one-time exception), and that the next method
  reverts to requiring a verified venue.
- `CITATIONS.md`: "38 compression methods"→"39"; add NestedKV bibliography
  entry, marked clearly as an unpublished arXiv preprint (not "NeurIPS 2025"
  or similar — no venue tag at all, just arXiv).
- `pyproject.toml`: version →`0.37.0`, description "...to CurDKV"→"...to
  NestedKV", "38"→"39".
- `paper/research/EVIDENCE_TABLE.md`: append new rows under a
  `## NestedKV-adapted multi-scale ensembled eviction (0.37.0) — added rows`
  section, covering all pinned claims/tests including the venue-exception
  itself as a documented, deliberate finding (not silently different from
  every other row).
- `paper/joss/paper.md`: check for any stale "thirty-eight"→"thirty-nine" (the
  JOSS paper text, not just a build artifact — grep for "thirty-eight" first).

## Phase 9 — landing page

`landing/index.html`: meta description, hero pill/roll-call (39, "· NestedKV"
appended), what's-new list (new 0.37.0 `<li>` above 0.36.0's), filter-bar
count, Token Eviction `cat-count` (14→15), remove `new-pill` from CurDKV card,
add new `#algo-nestedkv` card (with NEW pill) in Token Eviction `cat-group`
after the CurDKV card, add picker card + code-tab button + code panel.
`landing/assets/main.js`: `initBadgeTyping` text → "v0.37.0 — NestedKV-adapted
multi-scale ensembled eviction shipped".

## Phase 10 — full verification

Run new tests to green; re-run full existing suite (expect zero regressions
beyond the documented pre-existing VecInfer Metal flakes); build wheel +
`twine check`; build docs site; grep sweep for stale "38"/"thirty-eight";
factory smoke test (`KVCacheConfig(method="nestedkv", ...)` →
`KVCacheBuilder.for_model()` → `compression_ratio > 1.0` on both K and V).

## Phase 11 — release layer (provide as CHAT TEXT ONLY — never execute)

Standing rule, unchanged: the user reviews and runs all git/publish commands
themselves. After implementation, give the v0.37.0 release sequence as chat
text for them to run — do NOT execute any git add/commit/tag/push/gh
release/twine yourself.

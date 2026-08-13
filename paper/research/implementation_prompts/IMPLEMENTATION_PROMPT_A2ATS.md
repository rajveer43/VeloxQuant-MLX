# Implementation prompt — A2ATS-adapted (method #41, v0.39.0)

**DRAFT FOR REVIEW — do not execute until the user approves this prompt.**

## Phase 0 — venue gate

Source: "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He et al.),
**ACL 2025 Findings**, aclanthology.org/2025.findings-acl.644,
arXiv preprint mirrors the same content. Live-verify at implementation time
(re-run the check even though this was verified during scouting on
2026-07-15 — venue status and the "not yet implemented" fact can both go
stale) via:
- ACL Anthology page resolves and shows Findings of ACL 2025 as the venue.
- Confirm no existing `a2ats` method id anywhere in this repo
  (`grep -rn "a2ats\|A2ATS" --include=*.py --include=*.md .`) — it does not
  exist as of this prompt's writing (verified: only VecInfer/CommVQ occupy
  the query-aware / RoPE-aware vector-quantization niche today).

This is a **normal-track method** — a live-verified peer-reviewed venue,
no exception needed. Do not carry over AMC/NestedKV's "venue exception"
framing; this method does not require one.

## Phase 1 — naming and scope

- Method id: `a2ats`
- Family: **vector quantization** (joins VecInfer/CommVQ/RaBitQ/NSNQuant),
  but the first method in this family whose primary contribution is
  **RoPE-position-aware windowing of the compression scheme itself**, not
  the codebook/rotation construction. Contrast explicitly against the two
  existing RoPE-touching methods already in this repo:
  - **CommVQ-adapted** (`veloxquant_mlx/quantizers/comm_vq.py`, arXiv:2506.18879)
    solves RoPE by constraining the *codebook centroids* to a
    RoPE-commuting subspace (`_project_rope_commuting` — quantize
    pre-RoPE, reconstruct, then apply RoPE once at decode). One codebook,
    uniform treatment of every position.
  - **A2ATS** instead **decouples RoPE handling by token distance**: tokens
    within a trailing window of the current decode position get *exact*
    RoPE applied post-dequantization (cheap — the window is small); tokens
    outside the window get a *fixed-offset approximation* (the relative
    rotation between two far-apart positions is approximated by a single
    representative offset, avoiding the need to store or recompute a
    distinct rotation per absolute position). This is a genuinely different
    axis: CommVQ changes what the codebook can represent, A2ATS changes
    *when* exact-vs-approximate RoPE is paid for, based on retrieval
    relevance (distant tokens are the ones actually being vector-quantized
    for compression; nearby tokens are cheap enough to keep exact).
  - Query-aware vector quantization (Eq. in the paper's retrieval-selection
    section) reuses the same *shape* of idea as `amc_query_aware_saliency`
    (blend a magnitude/reconstruction term with a query-similarity term)
    but applied to *codebook assignment* rather than *tier assignment* —
    check `veloxquant_mlx/quantizers/amc.py`'s `amc_query_aware_saliency`
    for the guarded-cosine-similarity pattern to reuse (zero-norm key
    guard, same style as NestedKV's `_cosine_anomaly` and CurDKV's
    leverage-score guards) rather than re-deriving it.
- Framing: **"A2ATS-adapted (VeloxQuant-MLX implementation)"** — inspired
  by, not a faithful port of, "A2ATS: Retrieval-Based KV Cache Reduction
  via Windowed Rotary Position Embedding and Query-Aware Vector
  Quantization" (He et al., ACL 2025 Findings, aclanthology.org/2025.findings-acl.644).
- Version target: **v0.39.0**. Method count after this ships: **41**.
- Calibration: needs an offline codebook (same requirement class as
  VecInfer/CommVQ — a k-means/EM-trained codebook over representative
  key activations). Mirror `veloxquant_mlx/allocators/vecinfer.py`'s
  codebook-construction utilities rather than writing a new EM loop from
  scratch; check whether `comm_vq.py`'s EM trainer can be reused/parameterized
  before writing a third one.

## Phase 2 — windowed RoPE primitives: `veloxquant_mlx/quantizers/a2ats_rope.py`

The paper's core mechanism split into two regimes by trailing distance
`w` from the current decode position:

```python
def a2ats_apply_exact_rope(x: mx.array, positions: mx.array, base: float = 10000.0) -> mx.array:
    # x: [N, D] token vectors (already dequantized fp16), positions: [N] absolute
    # positions. Standard RoPE application — reuse comm_vq.py's
    # _rope_cos_sin_np / _apply_rope_np pattern (port to mx, don't
    # hand-roll a second RoPE implementation with different numerics).
    ...


def a2ats_apply_windowed_rope(
    x: mx.array,  # [N, D] token vectors
    positions: mx.array,  # [N] absolute positions
    query_position: int,  # current decode step's absolute position
    window: int = 128,  # trailing exact-RoPE window size
    base: float = 10000.0,
) -> mx.array:
    # For tokens with (query_position - positions) < window: apply exact
    # RoPE (delegate to a2ats_apply_exact_rope).
    # For tokens outside the window: apply a single fixed-offset
    # approximate rotation representing the "far" relative distance class
    # (paper's approximation — precompute once, not per-token).
    # Split via mx.where / boolean masking, not a Python loop over N.
    ...
```

Guard: `window <= 0` should degrade to "always approximate" (documented,
not an error — a valid, if extreme, config point), and `window >= sequence
length seen so far` should degrade to "always exact" (equivalent to
CommVQ's uniform treatment) — test both boundary behaviors explicitly.

## Phase 3 — query-aware VQ assignment: `veloxquant_mlx/quantizers/a2ats.py`

Reuse `VecInferKVCache`'s codebook encode/decode primitives
(`quantize_vq`/`dequantize_vq` in `veloxquant_mlx/allocators/vecinfer.py`)
for the actual vector quantization step — do not reimplement product/additive
VQ a third time in this repo. A2ATS's distinguishing contribution on top of
that machinery is *which* codebook entry gets selected:

```python
def a2ats_query_aware_assignment(
    x: mx.array,  # [N, D] keys to quantize
    codebook: mx.array,  # [K, D] candidate centroids
    query: mx.array,  # [D] current query vector
    beta: float = 0.5,  # blend coefficient, mirrors amc's alpha naming choice — confirm paper's actual symbol before finalizing
) -> mx.array:  # [N] assigned centroid indices
    # score = beta * (-reconstruction_error) + (1 - beta) * cosine_similarity(query, centroid)
    # i.e. prefer centroids that are both low-error AND query-relevant,
    # not pure nearest-centroid (that's what plain quantize_vq already does).
    # Reuse the zero-norm-guard cosine pattern from amc_query_aware_saliency.
    ...
```

Retrieval/selection component (paper's KV-reduction half — decide *which*
tokens are even worth retrieving at full fidelity before quantizing the
rest): expose as a `Tuple[mx.array, mx.array]` (retrieved_idx, compressed_idx)
split, mirroring the shape of NestedKV's or SnapKV's selection-mask
primitives — check those before inventing a new selection-mask convention.

Full `__all__` list, mirroring repo convention — finalize exact names
against what actually gets written, but expect roughly:
```python
__all__ = [
    "a2ats_apply_exact_rope",
    "a2ats_apply_windowed_rope",
    "a2ats_query_aware_assignment",
    "a2ats_select_retrieval_set",
    "a2ats_get_kv",
    "a2ats_fp16_bytes",
    "full_a2ats_fp16_bytes",
]
```

## Phase 4 — `veloxquant_mlx/cache/a2ats_cache.py`

Mirror `VecInferKVCache`'s `update_and_fetch` structure (quantize on the
way in, dequantize + apply windowed RoPE on the way out so SDPA sees plain
fp16), not an eviction-style cache — A2ATS retrieves-and-compresses, it
does not evict outright (confirm against the paper whether tokens outside
the retrieval set are dropped or merely quantized harder; state whichever
is true plainly, don't assume before checking — this is the single most
important mechanism-fidelity check for this method, get it right before
writing tests around it).

Config fields (add to `KVCacheConfig` in `cache/base.py`):
```python
a2ats_codebook_bits: int = 8
a2ats_sub_dim: int = 8
a2ats_window: int = 128  # trailing exact-RoPE window
a2ats_use_query_aware: bool = True  # paper's primary reported path — default ON, unlike AMC's off-by-default query path, since this is the paper's main contribution not an ablation extra; confirm against the paper's own default before finalizing
a2ats_beta: float = 0.5  # query/reconstruction blend
a2ats_retrieval_fraction: float = 0.20  # fraction of tokens kept at high fidelity
a2ats_rope_base: float = 10000.0
```

Add the same class of bounds validation this session just retrofitted onto
5 sibling methods — **do not repeat the bug just fixed**: `a2ats_beta` and
`a2ats_retrieval_fraction` must be validated to `[0, 1]` in `__init__`
directly (see `kitty_cache.py`/`svdq_cache.py`/`amc_cache.py` for the exact
pattern to copy), not left to silently clamp downstream.

Wire into `base.py`: add `"a2ats"` to the `Literal`, add the config block,
import `A2ATSKVCache`, add factory branch, extend the unknown-method error
string — identical mechanical steps to every prior method's Phase 4.

## Phase 5 — honesty crux (module docstring AND docs page)

1. **Retrieval-selection signal is approximated.** The paper's retrieval
   set is chosen using real attention-relevance signals unavailable inside
   a cache wrapper (same category of gap as every other saliency/attention
   proxy method in this repo — H2O's key-as-query proxy, SnapKV's prefill
   window, ZipCache's key-norm proxy). State exactly what proxy is
   substituted and why, in the same directness as those methods' existing
   docs.
2. **Windowed RoPE approximation is a genuine accuracy/compute tradeoff,**
   not free — the fixed-offset approximation for distant tokens has
   nonzero error vs. exact RoPE; state the tradeoff plainly rather than
   implying "windowed" is strictly better.
3. **No CUDA kernel fusion reproduced** — same MLX/Metal disclaimer as
   VecInfer/CommVQ: the paper's throughput numbers assume a fused kernel;
   this port's benefit on Apple Silicon is memory footprint, not speed.
4. **Query-aware assignment reuses AMC's guarded-cosine pattern** — cite
   that reuse explicitly in the docstring rather than presenting it as new
   machinery.
5. **Offline codebook calibration required**, same footgun class as
   VecInfer/CommVQ/Palu/SVDq/AMC — using `a2ats` without a properly trained
   codebook on representative data degrades silently to near-random
   quantization. State this as sharply as those methods do.
6. Nothing validated on real models/hardware — synthetic offline benchmark
   only, same convention as every prior method.

## Phase 6 — tests

`veloxquant_mlx/tests/quantizers/test_a2ats_rope.py` +
`veloxquant_mlx/tests/quantizers/test_a2ats.py` +
`veloxquant_mlx/tests/cache/test_a2ats_cache.py`. Target ~25-30 tests,
mirroring the AMC/VecInfer split. Critical mechanism tests:

- `test_windowed_rope_within_window_matches_exact_rope`: tokens inside the
  window get bit-identical (or near-identical, fp16 tolerance) results to
  `a2ats_apply_exact_rope` directly — the direct proof the window boundary
  behaves as documented.
- `test_windowed_rope_outside_window_uses_fixed_offset`: tokens outside the
  window do NOT match exact RoPE (confirm the approximation actually
  differs — an accidentally-exact "approximation" would silently hide a
  bug, same class of mistake this session just found and fixed in AMC's
  own test suite with the saturated-clamp bug).
- `test_window_zero_always_approximate` / `test_window_exceeds_seqlen_always_exact`:
  the two documented boundary degradations from Phase 2.
- `test_query_aware_assignment_prefers_relevant_centroid_over_nearest`:
  construct a case where the nearest centroid by reconstruction error is
  NOT the most query-relevant one; confirm `beta < 1.0` shifts the
  assignment toward the query-relevant centroid — the direct proof this
  mode does something (mirrors AMC's
  `test_query_aware_saliency_downweights_high_magnitude_irrelevant_tokens`,
  including the same magnitude-clamping trap: use values that don't
  saturate whatever bound applies here before asserting a gap).
- `test_beta_and_retrieval_fraction_bounds_validated`: **write this test
  FIRST, before the cache wrapper's `__init__`, TDD-style** — this session
  just spent real time confirming and fixing 5 methods that shipped without
  this exact check; don't ship method #41 with the same gap on day one.
- `test_retrieval_set_disjoint_from_compressed_set` (or documents the
  actual overlap semantics, whichever the paper specifies — see the Phase
  4 fidelity note).
- `test_determinism`: identical input twice → identical output.
- Standard suite: init, byte accounting, `for_model` config propagation
  (all 7 `a2ats_*` fields), factory dispatch, factory smoke test with
  `compression_ratio > 1.0`.

Run until fully green; fix root causes, same practice as every prior
method.

## Phase 7 — benchmark (`benchmark_scripts/benchmark_a2ats.py`)

Offline synthetic benchmark, same shape as `benchmark_vecinfer.py`: compare
A2ATS vs VecInfer (closest existing sibling — both are query-touching VQ
methods) and vs a plain fixed-window RoPE baseline (window=full-sequence,
i.e. no approximation — isolates what the windowing itself buys or costs)
on 2-3 synthetic geometries, at minimum one with strong positional locality
(where windowing should help) and one with long-range-dependent structure
(where the fixed-offset approximation should show its cost — an honesty
case, not hidden). Commit deterministic results JSON
(`benchmark_scripts/a2ats_benchmark_results.json`), verify determinism by
diffing two runs, same as every prior method.

## Phase 8 — docs (`docs-site/docs/algorithms/a2ats.md`)

Mirror `vecinfer.md`/`amc.md` structure: title/method-id, mechanism-gap
table contrasting against CommVQ (codebook-constraint RoPE handling) and
VecInfer (no RoPE-awareness at all — smooth+Hadamard only). Honesty crux
(6 points, Phase 5). Usage snippet with all 7 config fields. How-it-works
walkthrough (offline codebook calibration → per-token windowed RoPE regime
selection → query-aware centroid assignment → quantize). Byte accounting.
Benchmark section. Evidence section (cite aclanthology.org/2025.findings-acl.644).
When-to-use table — emphasize "long-context retrieval workloads with strong
positional locality" as the differentiator.

Update `docs-site/sidebars.ts` (add `'algorithms/a2ats'` after
`'algorithms/vecinfer'` or alongside the other VQ methods), `overview.md`
("forty"→"forty-one"; add row under vector-quantization family),
`changelog.md` (new `## v0.39.0 — Latest` section, demote v0.38.0).

## Phase 9 — README/CHANGELOG/CITATIONS/pyproject/EVIDENCE_TABLE

- `README.md`: 40→41 everywhere; add A2ATS row to the vector-quantization
  method table (verify exact table placement with `grep -n "^###"` first).
- `CHANGELOG.md`: new `[0.39.0]` entry — **no venue-exception subsection
  needed** (this is a normal-track, verified-venue method; don't carry over
  AMC's exception framing where it doesn't apply).
- `CITATIONS.md`: "40 compression methods"→"41"; add A2ATS bibliography
  entry with the ACL Anthology venue link.
- `pyproject.toml`: version →`0.39.0`, description "...to AMC"→"...to
  A2ATS", "40"→"41".
- `paper/research/EVIDENCE_TABLE.md`: append rows under `## A2ATS-adapted
  windowed-RoPE query-aware VQ (0.39.0) — added rows`.

## Phase 10 — landing page

`landing/index.html`: meta description, hero pill/roll-call (41, "· A2ATS"
appended), what's-new list, filter-bar count, vector-quantization
`cat-count`, remove `new-pill` from AMC's card, add new `#algo-a2ats` card
(NEW pill) in the Vector Quantization category group, picker card +
code-tab button + code panel. Verify div-tag balance via the same
regex-count script used for AMC (792 opens == 792 closes was the AMC
checkpoint — recount after this edit, don't assume the same number holds).
`landing/assets/main.js`: `initBadgeTyping` text → "v0.39.0 — A2ATS-adapted
windowed-RoPE query-aware VQ shipped".

## Phase 11 — full verification

Run new tests to green; re-run full existing suite (expect zero
regressions beyond the pre-existing VecInfer/Metal-kernel flakes — 5-7
failures in `test_vecinfer_fused_sdpa.py`/`test_vecinfer_metal_parity.py`/
`test_vecinfer_cache.py::test_reconstruction_error_bounded` is the known
baseline as of this session, confirmed unrelated to any of this session's
changes); build wheel + `twine check`; build docs site; grep sweep for
stale "40"/"forty" references; factory smoke test
(`KVCacheConfig(method="a2ats", ...)` → `KVCacheBuilder.for_model()` →
`compression_ratio > 1.0`).

## Phase 12 — release layer (CHAT TEXT ONLY — never execute)

Standing rule, unchanged: the user reviews and runs all git/publish
commands themselves. After implementation, give the v0.39.0 release
sequence as chat text — do NOT execute any git add/commit/tag/push/gh
release/twine yourself.

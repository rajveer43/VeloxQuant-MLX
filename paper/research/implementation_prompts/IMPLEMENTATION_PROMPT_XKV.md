# Full Autonomous Implementation Prompt — xKV-adapted (v0.27.0)

**Purpose of this document:** a self-contained, execute-without-supervision
spec for shipping the next VeloxQuant-MLX method end-to-end: survey (done,
see `paper/NEW_METHOD_SURVEY_V10.md`) → quantizer primitives → coordinator →
cache wrapper → config/builder wiring → tests → benchmark → docs → changelog →
README → landing page → version bump. Written so an agent picking this up
cold, with no other context, can complete every step and leave the repo in
the same finished state as every prior release (0.19.0 through 0.26.0).

**Do not deviate from the "adapted, not faithful port" discipline** that
governs this entire repo: label everything "xKV-adapted (VeloxQuant-MLX
implementation)", report only numbers from a committed `results.json` you
generate yourself, never repeat the paper's headline numbers as if they were
measured here, and document every simplification plainly (a "What we do NOT
implement" section, exactly like every other algorithm doc in this repo).

Read `paper/NEW_METHOD_SURVEY_V10.md` first — it contains the full rationale,
the exact mechanism, and the adaptation decisions already made. This document
is the *execution checklist* for that survey's "Planned artifacts" section.
Do not re-derive the design; the survey already settled it. If something in
this checklist conflicts with the survey, the survey wins — fix this document,
don't silently diverge.

---

## 0. Ground rules (apply to every phase)

- Work in small, buildable commits, one per phase below, each passing tests
  before moving to the next phase — this mirrors every prior release's git
  history (`git log --oneline` shows one commit per method-phase pattern:
  survey → primitives+tests → cache+tests → bench → docs/release).
- Every new Python file gets a module docstring citing the arXiv ID, stating
  "adapted, not faithful port", and listing what's NOT implemented.
- No new third-party dependencies. Use `mx.linalg.svd(x, stream=mx.cpu)`
  exactly as `svdq.py` does (MLX SVD currently only runs correctly on CPU
  stream — do not attempt GPU stream for SVD).
- Never invent benchmark numbers. If you cannot run models on real hardware
  in this environment, build the offline-synthetic harness (same pattern as
  `benchmark_scripts/benchmark_cam.py`, `benchmark_chunkkv.py`) and commit
  whatever `xkv_benchmark_results.json` that harness actually produces when
  run. If MLX/Metal is unavailable in the execution sandbox, state that
  explicitly in the CHANGELOG entry ("NOT YET RUN on hardware") rather than
  fabricating numbers — this repo has done that honestly before (see the
  StreamingLLM 8f1259c commit message: "offline-synthetic harness ... NOT YET
  RUN on hardware").
- Run `pytest veloxquant_mlx/tests/ -x -q` after each phase; do not proceed
  past a red test suite.
- Follow existing naming: files `xkv.py` (quantizer primitives),
  `xkv_coordinator.py`, `xkv_cache.py`, `test_xkv.py`, `test_xkv_cache.py`,
  `benchmark_xkv.py`, `xkv.md`, `xkv_benchmark_results.json`.

---

## Phase 1 — Quantizer primitives

**File:** `veloxquant_mlx/quantizers/xkv.py`

Module docstring: cite arXiv:2503.18893 (Chang, Lin, Lin, Chiang, Akhauri, Dai,
Jiang, Li, Ceze, Wu, Abdelfattah — preprint, code at
https://github.com/abdelfattah-lab/xKV), state "xKV-adapted", and summarize the
mechanism (joint cross-layer SVD into a shared basis, contiguous fixed-size
layer groups, no CKA-based grouping, no Selective Reconstruction, keys only —
mirrors `svdq.py`'s docstring structure).

Implement, modeled directly on `svdq.py`'s existing functions (reuse
`_group_quant_dequant` from `veloxquant_mlx.quantizers._quant_utils` for the
latent quantization step — do not reimplement group quant):

```python
def joint_svd_compress(
    key_stack: list[
        mx.array
    ],  # N arrays, each [S, D] fp16/fp32 — one per group member, same token range
    rank: Optional[int] = None,
    energy_threshold: float = 0.95,
) -> tuple[mx.array, mx.array, mx.array]:
    """Jointly factorize N layers' key matrices into one shared basis.

    Stacks the N centered [S, D] matrices along the token axis into one
    [N*S, D] matrix, computes truncated SVD once, returns the shared basis.

    Returns:
        (V_g, K_mean_g, singular_values) — V_g: [D, r], K_mean_g: [D], s: [r].
        K_mean_g is the mean over ALL stacked rows (shared across the group),
        not per-layer — this is the key difference from SVDq's per-layer mean.
    """
```

```python
def project_into_shared_basis(
    keys: mx.array,  # [S, D]
    V_g: mx.array,  # [D, r]
    K_mean_g: mx.array,  # [D]
) -> mx.array:
    """Project one layer's own keys into an already-computed shared basis. Returns latent [S, r]."""
```

```python
def reconstruct_from_shared_basis(
    L_q: mx.array,  # [S, r] quantized latents
    V_g: mx.array,
    K_mean_g: mx.array,
) -> mx.array:
    """Inverse of project_into_shared_basis. Returns [S, D] fp16."""
```

```python
def quantize_latents_uniform(
    L: mx.array,  # [S, r]
    bits: int = 4,
    group_size: int = 32,
) -> mx.array:
    """Single-bit-width latent quantization (default path) — thin wrapper over
    _group_quant_dequant. xkv_latent_bits controls `bits`; mixed-bit routing
    (SVDq-style hi/lo split) is NOT the default here — the shared basis is
    xKV's distinguishing feature, not novel bit allocation. Optional mixed-bit
    extension may reuse quantize_latents_mixed from svdq.py directly if a
    caller wants it (do not duplicate that logic)."""
```

Reuse `veloxquant_mlx.quantizers.svdq.quantize_latents_mixed` directly (import,
don't copy) if you wire up the optional mixed-bit path — do not duplicate that
function.

**Tests:** `veloxquant_mlx/tests/quantizers/test_xkv.py`
- `joint_svd_compress` on N=1 degenerates to (numerically close to)
  `svd_compress_keys` from `svdq.py` on the same single matrix — this is the
  group-of-1 equivalence check the survey calls for.
- `joint_svd_compress` on N=3 synthetic matrices sharing a true common
  low-rank structure (construct via `shared_basis @ per_layer_coeffs +
  per_layer_noise`) reconstructs all 3 layers within tolerance ε, and the
  reconstruction error is *lower* than each layer independently SVD'd at the
  same rank on *noisy, non-shared* structure (sanity: shared basis helps when
  structure is actually shared — this validates the mechanism, not just the
  plumbing).
- `project_into_shared_basis` + `reconstruct_from_shared_basis` round-trip:
  project then reconstruct without quantization noise ⇒ near-exact recovery
  (float32 precision floor).
- `quantize_latents_uniform` byte math sanity (same style as `svdq.py`'s
  `_group_quant_dequant` tests already in `test_svdq.py` if that quantizer
  test file exists — check `veloxquant_mlx/tests/quantizers/` for a
  `test_svdq.py`; if the repo instead only tests SVDq at the cache level,
  match that convention and only test `xkv.py` at the primitive level here,
  covering it again through the cache-level tests in Phase 3).

Run `pytest veloxquant_mlx/tests/quantizers/test_xkv.py -v` before continuing.

---

## Phase 2 — Cross-layer coordinator

**File:** `veloxquant_mlx/cache/xkv_coordinator.py`

Model this **directly** on `veloxquant_mlx/cache/minicache_coordinator.py`
(read that file in full before writing this one — it is the closest existing
pattern: a shared object injected at build time, keyed state, single-threaded
sequential-forward-pass assumption). Key differences from MiniCache's
coordinator:

- MiniCache's coordinator stores one primary's *raw KV* awaiting exactly one
  merge partner (pairwise). xKV's coordinator must collect raw prefill keys
  from **all N members of a group** before the joint SVD can run, then
  broadcast the resulting shared basis back to all N members (including the
  leader) — a fan-in-then-fan-out pattern, not fan-in-then-single-consumer.
- Model the API as:

```python
class XKVCoordinator:
    def __init__(self, max_ctx: int = 8192) -> None: ...

    def reset(self) -> None: ...

    def publish_member_keys(
        self,
        group_id: int,
        member_idx: int,
        token_start: int,
        keys: mx.array,  # this member's own [S, D] centered-later keys (raw, per-layer)
    ) -> None:
        """Any group member (leader or follower) publishes its raw prefill keys
        for this token range. Once ALL expected members for (group_id, token_start)
        have published, the basis is computed automatically and cached."""

    def get_shared_basis(
        self,
        group_id: int,
        token_start: int,
        expected_members: int,
    ) -> Optional[tuple[mx.array, mx.array, mx.array]]:
        """Returns (V_g, K_mean_g, singular_values) once all `expected_members`
        have published for this (group_id, token_start); None if still waiting.
        Computes joint_svd_compress lazily on the Nth publish and memoizes."""

    @property
    def max_ctx(self) -> int: ...
```

Important design note carried over from the survey: **the joint SVD only runs
once, at prefill, when the group's token range first fills.** After that, the
basis is frozen and stored on the coordinator (or copied onto each member
cache after first fetch) for the rest of generation — decode-time calls do NOT
re-publish through the coordinator; they project directly into the
already-fetched frozen basis (each `XKVCache` should cache its own copy of
`V_g`/`K_mean_g` locally after the first successful `get_shared_basis` call, so
it never needs to hit the coordinator again — this matches how `SVDqKVCache`
stores `self._V` once and never recomputes it).

Raise `RuntimeError` with a message pointing at `xkv_max_ctx` if publishing
would exceed `max_ctx`, matching `MiniCacheCoordinator.publish_primary`'s
existing guard exactly (same wording style).

**No dedicated coordinator test file needed** — its correctness is exercised
indirectly by the cache-level tests in Phase 3 (this matches the repo's
existing convention: there is no standalone `test_minicache_coordinator.py` or
`test_xquant_coordinator.py`; coordinators are tested through the cache).
Verify this convention by checking `find veloxquant_mlx/tests -iname
"*coordinator*"` returns nothing before skipping a coordinator test file — if
it turns out one does exist for a prior method, follow that pattern instead.

---

## Phase 3 — Cache wrapper

**File:** `veloxquant_mlx/cache/xkv_cache.py`

Model on `veloxquant_mlx/cache/xquant_cache.py` (read in full — it's the
cleanest role-based coordinator-consumer pattern in the repo) crossed with
`svdq_cache.py`'s SVD/quantize/reconstruct mechanics.

```python
class XKVCache(_MLXKVCache):
    """KV cache implementing xKV cross-layer shared-subspace key compression.

    Args:
        config: KVCacheConfig. Fields consumed:
            head_dim, xkv_rank, xkv_energy_threshold, xkv_latent_bits,
            xkv_group_quant_size.
        member_idx: This layer's index within its group (0 = leader by convention).
        group_id: Cross-layer group this layer belongs to.
        n_members: Number of layers in this group (coordinator waits for all of them).
        coordinator: Shared XKVCoordinator (None → degenerate: standalone per-layer
            SVD compression, equivalent to group_size=1 / SVDq-lite path).
    """
```

Behavior:
- **First call (prefill, `S > 1`, or first call generally):** publish this
  layer's own raw keys to the coordinator via `publish_member_keys`. Call
  `get_shared_basis`. If it returns `None` (not all members have published
  yet within *this* forward pass — should not normally happen since mlx_lm
  iterates layers sequentially within one forward call and each layer gets
  exactly one `update_and_fetch` per step, so by the time the *last* group
  member calls, all members have published for that step) — handle
  gracefully: if still `None` after this layer's own publish (i.e., this
  layer published but somehow isn't last), the layer should NOT block;
  instead it falls back to a private per-layer SVD compression for *this
  step only* (same fallback-on-missing-data discipline as
  `XQuantKVCache._reconstruct_reuse`'s "anchor hasn't published — fall back to
  self-quantization" branch) and retries fetching the shared basis on the
  next call. Once a shared basis is obtained (from the coordinator or a prior
  cached copy), store it locally (`self._V_g`, `self._K_mean_g`,
  `self._singular_values`) and never call the coordinator again for this
  layer instance.
- **Subsequent calls (decode):** project directly into the locally cached
  `self._V_g` (no coordinator interaction) — quantize the latent with
  `quantize_latents_uniform`, reconstruct, pass through.
- **Degenerate `coordinator=None` path:** behaves as standalone per-layer SVD
  (do NOT require a group — this makes `xkv_group_size=1` testable in
  isolation exactly like `XQuantKVCache`'s degenerate anchor-only path).

Byte accounting properties (mirror `svdq_cache.py`'s naming so downstream
benchmark tooling and any shared reporting code that greps for these names
keeps working):
- `compressed_key_bytes` — this layer's own latent codes only (NOT the shared
  basis — the basis is charged separately, once, amortized).
- `shared_basis_bytes` — new property: the `V_g`/`K_mean_g` storage cost,
  charged once per group (not once per layer) — expose it on every member but
  document that summing it across all members of a group double-counts; the
  correct group cost is `shared_basis_bytes / group_size` per layer, OR
  (cleaner) only the leader (`member_idx == 0`) reports nonzero
  `shared_basis_bytes`; followers report 0 there and only report their own
  `compressed_key_bytes`. **Choose the leader-only-reports convention** — it
  avoids double-counting entirely in any naive sum-across-layers benchmark
  code, which is the safer default given this repo's benchmarks tend to sum
  per-layer bytes.
- `fp16_key_bytes` — always the uncompressed cost, as in every other wrapper.
- `value_fp16_bytes` — values pass through unchanged (keys-only method, like SVDq).
- `assigned_avg_bits` — effective per-element key bit-width, same formula
  style as `SVDqKVCache.assigned_avg_bits` but accounting for the amortized
  basis cost only on the leader.

**Tests:** `veloxquant_mlx/tests/cache/test_xkv_cache.py` — follow
`test_svdq_cache.py`'s structure (factory dispatch, no-bits-leak,
SVD-projection correctness, prefill-only, decode-accumulation, byte-accounting,
rank-selection, values-passthrough) PLUS the group-specific cases the survey
calls for:
- Group-of-1 (`xkv_group_size=1`, standalone/degenerate) reduces to
  numerically-close standalone SVD compression.
- Multi-member group (e.g. 3 layers): all 3 `XKVCache` instances sharing one
  coordinator receive the *identical* `V_g`/`K_mean_g` arrays after the group's
  prefill step.
- Byte accounting: only `member_idx == 0` reports nonzero `shared_basis_bytes`;
  others report 0; each layer's `compressed_key_bytes` reflects only its own
  latent codes.
- Decode-time calls do not raise even after the coordinator's per-group budget
  (`xkv_max_ctx`) — verify the same `RuntimeError`-on-overflow guard as
  MiniCache/XQuant when exceeded.
- Non-attention / fallback-cache layers are unaffected (mirror the analogous
  XQuant/MiniCache builder test if one exists — check
  `veloxquant_mlx/tests/cache/test_base.py` or similar for a
  `_build_xquant`/`_build_minicache` builder-level test to mirror for
  `_build_xkv`).

Run `pytest veloxquant_mlx/tests/cache/test_xkv_cache.py -v` before continuing.

---

## Phase 4 — Config + builder wiring

**File:** `veloxquant_mlx/cache/base.py`

1. Add `"xkv"` to the `method` literal/allowed-values list at line ~37-38
   (the `"polar", "qjl", ... "cam", ...` tuple) — append it, keep alphabetical-
   ish grouping consistent with how `"cam"` and `"chunkkv"` were added
   (look at their addition diffs via `git log -p --follow -- veloxquant_mlx/cache/base.py`
   if you want the exact insertion convention).
2. Add a new config field block (mirror the `svdq_*` / `cam_*` blocks around
   lines 65-70 / 156-159):
   ```python
   xkv_group_size: int = 2  # layers per shared-subspace group (2 = pairs)
   xkv_rank: Optional[int] = None  # explicit rank; None → energy threshold
   xkv_energy_threshold: float = 0.95
   xkv_latent_bits: int = 4  # single-bit-width latent quantization
   xkv_group_quant_size: int = 32
   xkv_max_ctx: int = 8192  # coordinator per-group token budget
   ```
3. Add `elif config.method == "xkv":` dispatch around line ~268-274 (next to
   the `svdq`/`xquant` branches) — but note xKV needs a coordinator like
   XQuant/MiniCache, so the *single-layer* `KVCacheFactory.create` dispatch
   path should raise or route to the degenerate standalone case only; the real
   multi-layer wiring happens in `KVCacheBuilder`, not the factory (check how
   `xquant`/`minicache` are excluded from — or specially handled in — the
   plain factory dispatch vs. requiring `KVCacheBuilder.for_model`; mirror
   whichever convention those two already use exactly).
4. In `KVCacheBuilder.for_model` (~line 527+), add a branch:
   ```python
   if config.method == "xkv":
       return KVCacheBuilder._build_xkv(layers, args, config, _FallbackCache)
   ```
   next to the existing `if config.method == "xquant":` branch (~578).
5. Add `KVCacheBuilder._build_xkv` as a new staticmethod, modeled **line for
   line** on `_build_xquant` (~626-673): assign `(member_idx, group_id)` over
   attention-bearing layers only via a new `pair_layers_grouped(n_layers,
   group_size)` helper in `quantizers/xkv.py` (returns `list[tuple[int, int,
   int]]` = `(member_idx_within_group, group_id, group_size_actual)` — the
   `group_size_actual` matters for the trailing partial group, exactly as
   `pair_layers` in `quantizers/xquant.py` handles a trailing partial group).
   Build one shared `XKVCoordinator(max_ctx=config.xkv_max_ctx)`, instantiate
   one `XKVCache` per attention layer with its role tuple, and a fallback
   cache for non-attention layers — copy `_build_xquant`'s non-attention
   fallback logic verbatim in structure.
6. Update the error message listing valid methods (~line 325-326) to include
   `xkv`.

Run the full test suite: `pytest veloxquant_mlx/tests/ -x -q`.

---

## Phase 5 — Benchmark

**File:** `benchmark_scripts/benchmark_xkv.py`

Model on `benchmark_scripts/benchmark_cam.py` or `benchmark_chunkkv.py`
(offline-synthetic harness, no live model download required, consistent with
this repo's existing benchmark convention — check whichever of those two is
more recent/canonical before copying structure). Measure:
- Output-perturbation proxy (cosine distance of compressed-cache attention
  output vs. full-cache attention output over probe queries) — same metric
  family as CaM's benchmark — comparing xKV against XQuant and MiniCache at
  matched effective compression ratios, sweeping `seq_len` and `xkv_group_size`
  (2, 3, 4).
- Reconstruction MSE of the shared-basis path vs. an independent per-layer SVD
  at the same rank, to surface the actual claim ("shared basis helps when
  cross-layer structure exists") honestly — including the case where it does
  *not* help (synthetic layers with no shared structure), so the writeup can
  be honest about when xKV underperforms per-layer SVDq.
- Byte-accounting sanity: total bytes across a group (leader + followers) vs.
  naive per-layer SVDq bytes at the same rank, showing the amortization win.

Run it (`python benchmark_scripts/benchmark_xkv.py`) and commit whatever
`xkv_benchmark_results.json` it actually produces. If MLX SVD or the sandbox
can't execute (e.g. no Metal device), say so plainly in the CHANGELOG rather
than fabricating results — follow the StreamingLLM precedent (`8f1259c`: "NOT
YET RUN on hardware").

---

## Phase 6 — Docs site

**File:** `docs-site/docs/algorithms/xkv.md`

Model structure on `docs-site/docs/algorithms/cam.md` or `minicache.md`
(cross-layer precedent is more relevant than CaM here — prefer `minicache.md`
as the template since it's also a cross-layer coordinator-based method).
Sections: what it is, the paper's mechanism, the adaptation decisions (copy
faithfully from `NEW_METHOD_SURVEY_V10.md`'s "honest adaptation problem"
section — do not re-derive), config reference table, code example, what we do
NOT implement, benchmark results table (from the committed
`xkv_benchmark_results.json`), links to XQuant/MiniCache/SVDq docs pages as
"related cross-layer methods."

Update:
- `docs-site/sidebars.ts` — add `xkv` entry to the algorithms sidebar category,
  alphabetically/logically placed near `xquant`.
- `docs-site/docs/algorithms/overview.md` — bump "twenty-nine" → "thirty",
  add xKV to whatever comparison table/list already itemizes all methods.
- `docs-site/docs/getting-started/intro.md` — bump "twenty-nine" → "thirty" (2
  occurrences per the earlier grep), add xKV to the algorithm list.
- `docs-site/docs/changelog.md` — append the new release entry (see Phase 7
  for exact content; this file gets the same entry as the root `CHANGELOG.md`).
- Cross-link from `docs-site/docs/algorithms/xquant.md` and
  `docs-site/docs/algorithms/minicache.md` — add "See also: xKV" pointing at
  the new page, matching how CaM cross-links from H2O/ChunkKV per the 0.26.0
  changelog entry.

---

## Phase 7 — CHANGELOG, README, version bump

1. **`pyproject.toml`**: bump `version = "0.26.0"` → `"0.27.0"`.

2. **`CHANGELOG.md`**: insert a new `## [0.27.0] — <today's date>` section
   above `## [0.26.0]`, following the exact structural template of the
   existing 0.26.0/0.25.0 entries (### Added — heading naming the method and
   its axis; bullet list of files/classes added; config fields; "Honest
   scope"-style limitations subsection if the top-level entries here don't
   already have one — check the 0.26.0 entry's own trailing "### Honest scope"
   block and replicate that subsection heading exactly). Content to include:
   - `veloxquant_mlx.cache.xkv_cache.XKVCache` — the library's **thirtieth**
     configuration and the **third cross-layer** mechanism (after XQuant's
     rematerialization and MiniCache's SLERP merge) — joint shared-subspace
     SVD across a fixed-size contiguous layer group.
   - `veloxquant_mlx.quantizers.xkv` — pure primitives listed.
   - `veloxquant_mlx.cache.xkv_coordinator.XKVCoordinator` — fan-in/fan-out
     shared-basis broadcast, described briefly.
   - Config fields listed.
   - Tests: N quantizer tests + M cache tests (fill in actual counts once
     written), all passing, including the group-of-1 degeneracy check.
   - Benchmark: `benchmark_scripts/benchmark_xkv.py` + committed
     `xkv_benchmark_results.json` — describe what it measures (perturbation
     proxy vs XQuant/MiniCache, reconstruction MSE with/without true shared
     structure) and the honest headline number once you have it.
   - "### Honest scope" subsection: fixed contiguous grouping (no CKA
     validation), no Selective Reconstruction, keys-only (values fp16), no
     model-level perplexity/throughput benchmark (offline perturbation proxy
     only) if that's what Phase 5 actually produced.
   - Docs: new `docs-site/docs/algorithms/xkv.md`, sidebar + overview + intro +
     changelog entries, cross-links from XQuant and MiniCache pages.
     README/landing counts: twenty-nine → **thirty** strategies; version bump
     0.26.0 → 0.27.0.

3. **`README.md`**:
   - Line ~25: changelog badge `0.26.0` → `0.27.0`.
   - Line ~34: "twenty-nine" → "thirty" in the summary paragraph; if the
     paragraph enumerates cross-layer methods by name anywhere, add xKV.
   - Line ~62: "all 29 methods" → "all 30 methods".
   - Line ~172: "All 29 methods share..." → "All 30 methods share...".
   - Method table (near line 219, where the CaM row lives): add a new row,
     e.g. `| [xKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/xkv) | \`xkv\` | Cross-layer **shared-subspace** SVD — joint basis across a layer group | 0.27.0 |`
   - Sources section (near the ChunkKV citation at line ~576): add an xKV
     citation line in the same format: `- [xKV (arXiv:2503.18893)](https://arxiv.org/abs/2503.18893) — Chang et al., "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction" — joint cross-layer shared-subspace SVD compression via CKA-aligned singular vectors (adapted: fixed contiguous grouping not CKA-validated, no Selective Reconstruction, keys only)`
   - `paper/EVIDENCE_TABLE.md`: add a row for xKV following the existing
     table's column format (open the file, match columns exactly).

4. **Landing page** — `landing/index.html` (now split into HTML/CSS/JS per the
   most recent commit `e596de7`; check `landing/assets/` for where the JS/CSS
   actually live post-split before editing):
   - Meta description (~line 10): add "xKV" to the method list, update the
     "New in 0.27.0: ..." sentence to describe xKV instead of CaM.
   - Method name strip (~line 67): add `· xKV` to the inline list.
   - Changelog/what's-new list (~line 204 area): add a new `<li>` entry for
     0.27.0 above the 0.26.0 one, same markup style.
   - Method Library picker grid (~line 287 area): add a new `picker-card`
     for xKV, same markup as the CaM card, pick an unused accent color.
   - Full algo card section (~line 304 area): add a new `algo-card` block
     `id="algo-xkv"` with the same structure as `algo-cam` (title, headline
     stat, full description, honest-scope callout).
   - Code tabs (~line 1517 / 1715 area): add a new `tab-btn` and
     `code-panel` for `xkv` with a working `KVCacheConfig(method="xkv", ...)`
     example mirroring the CaM tab's structure.
   - Verify the page still renders: there's a `landing/__probe.html` file
     already open in this session's IDE — check what it's used for (likely a
     manual visual QA scratch file) and use the same probing approach if it
     helps confirm markup validity, but don't leave stray temp files behind
     if you create new ones — clean up after yourself.

---

## Phase 8 — Final verification

1. `pytest veloxquant_mlx/ -x -q` — full suite green.
2. Grep the whole repo for stray "twenty-nine"/"29 method"/"29 strateg" that
   Phase 6/7 might have missed: `grep -rn "twenty-nine\|29 method\|29 compression\|29 strateg" --include="*.md" --include="*.html" --include="*.ts" .`
   — every hit must be bumped to thirty, except historical CHANGELOG entries
   dated before 0.27.0 (those describe the state *at that time* and must NOT
   be edited — only the current/latest-state references change).
3. Confirm `git status` shows a clean, reviewable diff: new files (`xkv.py`,
   `xkv_coordinator.py`, `xkv_cache.py`, test files, benchmark file, results
   json, docs page) + modified files (`base.py`, `CHANGELOG.md`, `README.md`,
   `pyproject.toml`, landing page, docs-site sidebar/overview/intro/changelog).
4. Do not `git commit` or `git push` anything without the user's explicit
   go-ahead when they return — prepare the working tree in a complete,
   reviewable state and stop there. Summarize what was built and point at the
   diff; let the user decide on commit message and whether to ship it.

---

## Appendix — file manifest (new files this release)

```
paper/NEW_METHOD_SURVEY_V10.md                       (already written)
paper/IMPLEMENTATION_PROMPT_XKV.md                   (this file)
veloxquant_mlx/quantizers/xkv.py
veloxquant_mlx/tests/quantizers/test_xkv.py
veloxquant_mlx/cache/xkv_coordinator.py
veloxquant_mlx/cache/xkv_cache.py
veloxquant_mlx/tests/cache/test_xkv_cache.py
benchmark_scripts/benchmark_xkv.py
benchmark_scripts/xkv_benchmark_results.json
docs-site/docs/algorithms/xkv.md
```

## Appendix — files modified this release

```
veloxquant_mlx/cache/base.py          (config fields, method dispatch, builder)
pyproject.toml                        (version bump)
CHANGELOG.md                          (new release entry)
README.md                             (counts, method table, sources, badge)
paper/EVIDENCE_TABLE.md               (new row)
docs-site/sidebars.ts                 (sidebar entry)
docs-site/docs/algorithms/overview.md (count bump, table row)
docs-site/docs/algorithms/xquant.md   (cross-link)
docs-site/docs/algorithms/minicache.md (cross-link)
docs-site/docs/getting-started/intro.md (count bump)
docs-site/docs/changelog.md           (new release entry, mirrors root CHANGELOG.md)
landing/index.html (or split assets)  (meta, method strip, changelog list, picker card, algo card, code tab)
```

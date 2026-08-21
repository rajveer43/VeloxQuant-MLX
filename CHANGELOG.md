# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

## [Unreleased]

### Performance

**Wired the existing Q-Filters Metal eviction kernels into the cache**
([#179](https://github.com/rajveer43/VeloxQuant-MLX/issues/179)) — the
per-`(B, H)` Python loop on the calibrated path was already removed in
#173 (`qfilters_update_batched`), but the fused Metal kernels written and
tested alongside it (`qfilters_fused_evict`,
`veloxquant_mlx/metal/_qfilters_evict.py`) were never called from
`QFiltersKVCache` — the batched path always ran the pure-MLX
`mx.argsort` / `mx.take_along_axis` selection even when Metal was
available. `QFiltersKVCache` now resolves a three-state
`use_metal_kernels` flag (`None` auto-detect, `True` require, `False`
force pure-MLX — the same convention `VecInferKVCache` already uses) and
dispatches the over-budget branch of the calibrated/batched path to the
fused kernel when eligible. Verified bit-for-bit interchangeable with the
pure-MLX path it replaces via a new parametrized parity test (Metal
hardware only); the key-SVD fallback's per-head loop and the under-budget
passthrough are untouched. **Not yet benchmarked** — no before/after
TTFT or decode-throughput numbers exist for this path; that measurement
needs real Apple Silicon and remains open.

### Documentation

**Measured real-model perplexity for L2Norm after the RoPE offset fix**
([#190](https://github.com/rajveer43/VeloxQuant-MLX/issues/190),
validating [#174](https://github.com/rajveer43/VeloxQuant-MLX/issues/174))
— #174 generalized the #171 offset fix to `L2NormKVCache` and re-enabled
L2Norm as an arm in the Q-Filters generation-perplexity benchmark, but was
authored in a sandbox without MLX, so the fix shipped with unit-test
coverage and **no end-to-end numbers**. Ran
`benchmark_scripts/qfilters_real_model_perplexity.py` on
Llama-3.2-1B-Instruct-4bit (1024 tokens, budgets 128/256) on Apple
Silicon: against an fp16 baseline of ppl 4.050, L2Norm scores 20.469 at
budget 128 and 9.529 at budget 256, landing between calibrated Q-Filters
(16.307 / 8.476) and the key-SVD fallback (23.645 / 13.358) at both
points. The budget-responsive curve confirms the offset fix end-to-end —
pre-#174 position drift grows without bound with sequence length and would
flatten every arm regardless of budget. Documented in
`docs-site/docs/algorithms/knorm.md`, replacing its "no model-level
benchmark has been run" note. **Scope**: one model, two budgets — larger
models and a wider sweep remain open under
[#181](https://github.com/rajveer43/VeloxQuant-MLX/issues/181) /
[#180](https://github.com/rajveer43/VeloxQuant-MLX/issues/180); L2Norm
TTFT/throughput is still unmeasured. Docs only, no code changes.

**Reviewed StreamingLLM-adapted's RoPE position semantics against the paper**
([#189](https://github.com/rajveer43/VeloxQuant-MLX/issues/189)) — the
implementation already preserved original absolute token positions after
eviction rather than the paper's cache-slot renumbering (a deliberate
divergence noted briefly in the docs since the #171 offset fix), but this
had never been reviewed against the paper explicitly, and unlike every
other eviction cache in the library, `StreamingLLMKVCache` had **no
regression test** guarding the `_true_offset`/`#171` fix. Added three
offset-tracking tests to `test_streaming_llm_cache.py` (mirroring the
KNorm/SnapKV/TOVA/Q-Filters coverage: true position through sustained
eviction, block-size advance on prefill, and correctness across a
prefill-then-decode mix), and expanded `docs-site/docs/algorithms/streaming_llm.md`
with a full review section explaining why absolute-position preservation
was kept over the paper's renumbering (RoPE relativity makes it exact;
consistency with every sibling cache; avoids a per-step re-rotation cost;
the cache-wrapper boundary can't implement the paper's scheme cleanly
anyway). **Not measured**: whether the paper's renumbering scheme would
produce different generation quality — that needs real model inference
and remains open, same as [#187](https://github.com/rajveer43/VeloxQuant-MLX/issues/187).

**Reviewed SnapKV-adapted's RoPE position semantics against the paper**
([#188](https://github.com/rajveer43/VeloxQuant-MLX/issues/188)) — unlike
StreamingLLM, the SnapKV paper doesn't define a position-renumbering scheme
at all (it compresses the cache once, at the end of prefill, before
generation starts); common HF-`DynamicCache`-based reference
implementations instead fall back to the retained-row-count convention this
repo's `#171` fix moved away from. Reviewed why absolute-position
preservation isn't a stylistic choice here but the behavior mathematically
required by how the eviction mechanism works: SnapKV selects from K rows
the model already rotated with RoPE during its own prefill forward pass, so
survivors' rotations are permanently baked in before eviction happens —
reporting anything but their true position would break RoPE's
relative-distance identity. Unlike StreamingLLM (continuous per-step
eviction), a hypothetical renumbering scheme here would only cost a
one-time `O(budget)` re-rotation right after prefill compression — still not
worth implementing, since exact positions already require none. Test
coverage already existed
(`test_offset_tracks_true_position_not_retained_rows` plus the
chunked-prefill offset assertions) — no code or test changes needed, this
was purely a documentation review. Added to
`docs-site/docs/algorithms/snapkv.md`.

### Fixed

**RoPE positions after eviction in TOVA-adapted**
([#175](https://github.com/rajveer43/VeloxQuant-MLX/issues/175)) — audited
H2O-adapted and TOVA-adapted for the same class of defect fixed for
Q-Filters, L2Norm, SnapKV, and StreamingLLM. Findings:

- **H2O-adapted already fixed** (shipped in v0.44.4, predating this audit):
  `self.offset` already tracks the true absolute step count directly, and
  because H2O renumbers positions to close gaps when an interior eviction
  happens (e.g. with `h2o_n_sink > 0`), it additionally re-rotates the
  shifted survivors — a stronger fix than the other caches need, since they
  never renumber. No change required.
- **TOVA-adapted carried the defect**: `self.offset` reported the retained
  row count rather than the true absolute token position once eviction
  pinned the kept set at `tova_budget`. `tova_update` drops exactly the
  evicted row and keeps the rest in temporal order (never renumbers), so
  the same `_true_offset` counter that sufficed for Q-Filters/L2Norm applies
  directly — no `offset` property split like SnapKV's was needed, since
  `TOVAKVCache.update_and_fetch` fully resets `self.keys`/`self.values`/
  `self.offset` on every call, prefill and decode alike.

Regression tests added to `test_tova_cache.py` mirroring the Q-Filters/L2Norm
coverage: offset tracks true position through sustained eviction, advances
by block size on prefill, and stays correct across a prefill-then-decode mix.

**RoPE positions after eviction in L2Norm-adapted**
([#174](https://github.com/rajveer43/VeloxQuant-MLX/issues/174)) — carried
the same defect fixed earlier for Q-Filters, SnapKV, and StreamingLLM:
`self.offset` reported the **retained row count** rather than the true
absolute token position once eviction pinned the kept set at
`knorm_budget`. `mlx_lm` rotates both the query and the incoming key at
`offset=cache.offset` *before* `update_and_fetch` runs, so subsequent
tokens were rotated at a stale, non-advancing position — the offset drift
that excluded L2Norm as a fair baseline in Q-Filters benchmarks.

L2Norm's `knorm_update` already restores kept rows to temporal order after
top-k selection (never renumbers survivors), so the same fix that sufficed
for Q-Filters applies directly: a `_true_offset` counter incremented by the
incoming block size `S`, reported as `self.offset` after every
`update_and_fetch` call. Unlike SnapKV, no `offset` property split was
needed — `L2NormKVCache.update_and_fetch` fully resets
`self.keys`/`self.values`/`self.offset` on *every* call (prefill and decode
alike), so the base class's cursor arithmetic never observes the true
position as a stale row count between calls.

Regression tests added mirroring the Q-Filters/SnapKV coverage: offset
tracks true position through sustained eviction, advances by block size
(not retained rows) on prefill, and stays correct across a prefill-then-decode
mix. L2Norm can now be re-enabled as a fair comparison arm in Q-Filters
benchmarks.

**RoPE positions after eviction in SnapKV-adapted and StreamingLLM-adapted**
([#171](https://github.com/rajveer43/VeloxQuant-MLX/issues/171)) — both
carried the same defect fixed earlier for Q-Filters: `self.offset` reported
the **retained row count** rather than the true absolute token position.
`mlx_lm` rotates both the query and the incoming key at
`offset=cache.offset` *before* `update_and_fetch` runs, so once eviction
started every subsequent token was rotated at the wrong position. Measured
over 200 decode steps after a 64-token prefill at budget 32:

- **StreamingLLM** — offset froze at **32** while the true position reached
  **263** (drift **+231**, growing without bound once the window saturated).
- **SnapKV** — offset advanced but stayed **exactly 32 behind**, the constant
  deficit being the tokens dropped during prefill compression.

Both preserve original positions (`snap_select_indices` returns kept indices
sorted ascending; StreamingLLM's window drops rows without renumbering) and
RoPE is relative, so reporting the true position is sufficient — survivors
need no re-rotation, unlike H2O/Keyformer which renumber.

SnapKV needed more than the `_true_offset` counter that sufficed for
StreamingLLM: its decode path appends deltas, so the base class's cursor
arithmetic and return slice (`self.keys[..., :self.offset, :]`) genuinely
require the row count. `offset` is now a property yielding the row count
while the base class is on the stack and the true position outside it.

Note this diverges from the StreamingLLM paper, which assigns positions by
index *within* the cache; matching that would require re-rotating every
survivor each step. Recorded in the cache docstring.

Two existing tests asserted `offset == retained rows` — the old meaning —
and were updated, with a dedicated regression test added.

**A2ATS-adapted paper-fidelity fixes** ([#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)) —
four deviations from the source paper (He et al., ACL 2025 Findings), three
of which contradicted its equations rather than knowingly adapting them:

- **Far keys are no longer rotated.** `a2ats_apply_windowed_rope` applied a
  fixed `R_window` rotation to out-of-window keys. Paper Eq. (12) leaves them
  **unrotated** (`k̃_i = k_i`), with the constant `R_b` belonging on the
  *query* (Eq. 11, `u_ij = q_i R_b k_j^T`). The old form computed
  `q_i R_w^T k_j^T` — wrong operand, wrong direction — and left far keys in a
  rotated frame, defeating the position-decoupling that makes a shared
  codebook viable (§3.1, Observation 2). Adds `a2ats_apply_far_query_rope`
  for the query-side half, exposed as `A2ATSKVCache.far_query_rope`.
- **`b` is now independent of `w`.** The far-token offset was derived from
  `a2ats_window`, hardcoding `b == w`; the paper's §5.1 uses `w=64`,
  `b=2048`. New `a2ats_b` config field (default `2048`).
- **Distance gating tracks the decode position.** Rotated keys were written
  into the parent `KVCache`, which never revisits them — freezing each
  token's near/far class at write time, so a token classified "near" during
  prefill kept exact RoPE forever. The cache now stores *pre-RoPE*
  reconstructions and re-applies windowed RoPE to the accumulated cache each
  step, against the current query position. Costs an `O(total_tokens)` pass
  per step, inherent to honest gating under this protocol.
- **The paper's Eq. (14) assignment is now implemented.** Adds
  `a2ats_query_second_moment`, `a2ats_cholesky_factor`, and
  `a2ats_h_weighted_assignment` — the `H`-weighted quadratic form, computed
  exactly via the Eq. (15)–(18) Cholesky identity. Enabled by the new
  `a2ats_query_h` config field. The previous cosine blend is retained as a
  documented *substitute* for the decode path (no calibrated `H` available),
  not relabeled as the paper's estimator.

Benchmark now measures **attention-score** error rather than key-vector
error — under Eq. (12) far keys are intentionally unrotated, so the old
target scored the method's design as error. Windowed RoPE still loses to
always-exact in both geometries (2.9x / 8.3x), and this **survived** the
fixes: the near bucket is now numerically identical to always-exact (max gap
`5e-07`), so the entire penalty is far tokens (~92–96% of the sequence).
Sweeping `b` does not remove it. Tests: 51 → 67, including the strengthened
Eq. (12) assertion that the old `R_window` bug slipped past.

**AdaKV-proxy: the default configuration was not adaptive** ([#31](https://github.com/rajveer43/VeloxQuant-MLX/issues/31)) —
`adakv_target_avg_bits` defaulted to `2.0` while `adakv_lo_bit` defaulted to
`2`. Because per-head adaptation requires headroom on both sides of the target
(raising one head must be payable by lowering another), a target sitting exactly
on the floor forced *every* head to `lo_bit` for every possible importance
vector. The shipped default was therefore bit-identical to plain KIVI while the
docs advertised per-head adaptation. The default is now `2.5`, and
`allocate_head_bits` emits a `UserWarning` when the target sits at or outside an
endpoint of the allowed set rather than silently flattening.

**AdaKV-proxy: clamp-before-normalize saturated the allocation** — the
real-valued per-head budget was computed as
`clip(importance_share × H × target, lo, hi)`. Since raw key-norm variances span
orders of magnitude, this pinned nearly every head to `lo` or `hi` and discarded
the interior ordering that the snap and greedy-correction passes exist to act
on: importance vectors differing by 10 000× produced byte-identical allocations.
Replaced with rank-normalisation to a bounded, mean-centred spread placed around
the target, so the mean lands on the budget by construction and no single
extreme head can flatten the result.

**AdaKV-proxy: allocation depended on head ordering** — greedy correction scanned
heads in index order and kept the first strict minimum, so `[10,1,1,1]` and
`[1,1,1,10]` starved different heads. Ties are now broken on importance, making
the allocation permutation-equivariant for distinct importances. (With exact ties
and an indivisible budget some tied head must still lose a bit — an
integer-allocation fact, now documented rather than incidental.)

**AdaKV-proxy: corrected an inverted claim about the importance signal** — the
docstring and docs described inter-token key-norm variance as "a proxy for high
attention entropy". It is the opposite: high norm-variance indicates a few
outlier-magnitude tokens dominating the logits, i.e. an attention-*sparse* head,
which Ada-KV (§3.3, Fig. 1b) would give *less* budget. The signal is sound for
allocating bits — it measures quantization sensitivity — but it is a *different*
criterion from the paper's, not an approximation of it. Now stated plainly and
pinned by a test.

### Added

**`compute_head_attention_entropy`** — an AdaKV-proxy importance signal that
carries Ada-KV's own sign: dispersed heads score higher and receive more budget.
Estimates per-head attention entropy over an observation window using the
keys-as-proxy-queries substitution already established by SnapKV-adapted,
normalised by `ln(S)`. Select via `adakv_importance="attention_entropy"`;
window size via `adakv_obs_window` (default 32).

**`KVCacheConfig`** — new fields `adakv_importance`
(default `"norm_variance"`) and `adakv_obs_window` (default `32`).

### Changed

- `adakv_target_avg_bits` default `2.0` → `2.5` (see above).
- `benchmark_scripts/benchmark_adakv.py` now sweeps targets inside the adaptive
  range and adds an `attention_entropy` arm at matched budget.

<!-- version list -->

## v0.54.0 (2026-08-21)

### Features

- **landing**: Add VeloxQuant Studio waitlist section and fix calc URL leak
  ([`d18556b`](https://github.com/rajveer43/VeloxQuant-MLX/commit/d18556bd55fc7f5f2a5da5ca01bc8b918481e453))


## v0.53.0 (2026-08-20)

### Bug Fixes

- **anchorkv**: Resolve ruff-format and link-check CI failures
  ([#238](https://github.com/rajveer43/VeloxQuant-MLX/pull/238),
  [`51b45ac`](https://github.com/rajveer43/VeloxQuant-MLX/commit/51b45ace5f74c2bce227ddd69f66340f0948d684))

### Features

- **anchorkv**: Add AnchorKV anchor-residual KV cache compression
  ([#238](https://github.com/rajveer43/VeloxQuant-MLX/pull/238),
  [`51b45ac`](https://github.com/rajveer43/VeloxQuant-MLX/commit/51b45ace5f74c2bce227ddd69f66340f0948d684))


## v0.52.1 (2026-08-19)

### Bug Fixes

- **svdq**: Implement paper's real 8-group bit schedule, guard against small-rank truncation
  ([`579fb8f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/579fb8fc0397a2d26875a6ad5f68d7713e73db75))

### Code Style

- Fix ruff formatting in test_svdq_cache.py
  ([`1e32922`](https://github.com/rajveer43/VeloxQuant-MLX/commit/1e3292257c09cd24379a78b61d370d4d801c16e1))


## v0.52.0 (2026-08-19)

### Features

- **bench**: Add H2O and TOVA as Q-Filters comparison arms
  ([#235](https://github.com/rajveer43/VeloxQuant-MLX/pull/235),
  [`bd816e2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/bd816e2efaf8e1863aee3effaa5453991a0aebc7))

### Testing

- **qfilters**: Pin fallback chunk-dependence and budget safety
  ([#235](https://github.com/rajveer43/VeloxQuant-MLX/pull/235),
  [`bd816e2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/bd816e2efaf8e1863aee3effaa5453991a0aebc7))


## v0.51.1 (2026-08-19)

### Bug Fixes

- **ci**: Stop the link checker failing on network flakiness
  ([#234](https://github.com/rajveer43/VeloxQuant-MLX/pull/234),
  [`806e2a6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/806e2a6971e576570dac13875e081235d12de5c3))

- **docs**: Correct two README links that 404ed, and check links in CI
  ([#234](https://github.com/rajveer43/VeloxQuant-MLX/pull/234),
  [`806e2a6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/806e2a6971e576570dac13875e081235d12de5c3))

- **site**: Redirect docs paths missing the /docs prefix instead of 404ing
  ([#234](https://github.com/rajveer43/VeloxQuant-MLX/pull/234),
  [`806e2a6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/806e2a6971e576570dac13875e081235d12de5c3))

### Build System

- **deps**: Bump body-parser from 1.20.5 to 1.20.6 in /docs-site
  ([#225](https://github.com/rajveer43/VeloxQuant-MLX/pull/225),
  [`58881a5`](https://github.com/rajveer43/VeloxQuant-MLX/commit/58881a5e00cbf21db081df74a47142c350013c9a))

- **deps**: Bump shell-quote from 1.8.4 to 1.10.0 in /docs-site
  ([#227](https://github.com/rajveer43/VeloxQuant-MLX/pull/227),
  [`7bfbbe0`](https://github.com/rajveer43/VeloxQuant-MLX/commit/7bfbbe0e7baed58b377421eeb7b92c03c1742b94))

- **deps**: Bump svgo from 3.3.3 to 3.3.4 in /docs-site
  ([#224](https://github.com/rajveer43/VeloxQuant-MLX/pull/224),
  [`d15a3b5`](https://github.com/rajveer43/VeloxQuant-MLX/commit/d15a3b50ecca64bdf39a6c801b0f3859aefaa1c4))

- **deps**: Bump webpack-dev-server from 5.2.4 to 5.2.6 in /docs-site
  ([#226](https://github.com/rajveer43/VeloxQuant-MLX/pull/226),
  [`546ba0c`](https://github.com/rajveer43/VeloxQuant-MLX/commit/546ba0cbdc64a08ab9f745c38ec9260ce8115329))

- **deps**: Bump websocket-driver from 0.7.4 to 0.7.5 in /docs-site
  ([#233](https://github.com/rajveer43/VeloxQuant-MLX/pull/233),
  [`ee13ec5`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ee13ec5eff32818f5f8bed30ff8aceb28fc10f4c))

### Documentation

- Drop oMLX from the comparison, keep llama.cpp and plain mlx_lm
  ([#234](https://github.com/rajveer43/VeloxQuant-MLX/pull/234),
  [`806e2a6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/806e2a6971e576570dac13875e081235d12de5c3))


## v0.51.0 (2026-08-18)

### Build System

- **deps**: Bump brace-expansion from 1.1.15 to 1.1.18 in /docs-site
  ([#222](https://github.com/rajveer43/VeloxQuant-MLX/pull/222),
  [`59a208c`](https://github.com/rajveer43/VeloxQuant-MLX/commit/59a208cd3c6af6eb6f78e1a295490a0cdfb7119b))

- **deps**: Bump postcss from 8.5.15 to 8.5.26 in /docs-site
  ([#223](https://github.com/rajveer43/VeloxQuant-MLX/pull/223),
  [`f4dc708`](https://github.com/rajveer43/VeloxQuant-MLX/commit/f4dc708dedbd353bc64f0122197d2396b331ac3c))

### Documentation

- **blog**: The needle was the easy part -- RULER beyond NIAH
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

- **qfilters**: Report the non-NIAH RULER results and correct the gap claim
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

### Features

- **bench**: RULER task categories beyond NIAH for Q-Filters
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

- **bench**: RULER task categories beyond NIAH for Q-Filters (#177)
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

### Testing

- **ruler**: Probe whether chunked prefill rescues Q-Filters on VT
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

- **ruler**: Raw results for the four non-NIAH RULER categories on Qwen2.5-7B
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

- **ruler**: Record actual prefilled token counts per task and context
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))

- **ruler**: Verify the arms compress equally before reading the CWE spread
  ([#232](https://github.com/rajveer43/VeloxQuant-MLX/pull/232),
  [`b7eb1ee`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b7eb1ee49acf3d08e13b685e11a35cd4ab04bbc8))


## v0.50.2 (2026-08-18)

### Bug Fixes

- **qfilters**: Derive head_dim for Qwen and validate at 7B (#176)
  ([#230](https://github.com/rajveer43/VeloxQuant-MLX/pull/230),
  [`bc9cb4f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/bc9cb4fb0c492dea9d1fe92baf4305d057542513))

### Build System

- **deps**: Bump authlib from 1.7.0 to 1.7.1
  ([#216](https://github.com/rajveer43/VeloxQuant-MLX/pull/216),
  [`2e5f7d4`](https://github.com/rajveer43/VeloxQuant-MLX/commit/2e5f7d4eccc5d68adbf5749af7abeadb57c9d611))

- **deps**: Bump fast-uri from 3.1.2 to 3.1.5 in /docs-site
  ([#219](https://github.com/rajveer43/VeloxQuant-MLX/pull/219),
  [`e112723`](https://github.com/rajveer43/VeloxQuant-MLX/commit/e1127230d50a44be28cc8a8d0a90defbb30344ab))

- **deps**: Bump gradio from 6.13.0 to 6.15.1
  ([#208](https://github.com/rajveer43/VeloxQuant-MLX/pull/208),
  [`4abd88c`](https://github.com/rajveer43/VeloxQuant-MLX/commit/4abd88ca0ce76af3ae6cec399262a0a75c502ea4))

- **deps**: Bump idna from 3.13 to 3.15
  ([#215](https://github.com/rajveer43/VeloxQuant-MLX/pull/215),
  [`21ea5c1`](https://github.com/rajveer43/VeloxQuant-MLX/commit/21ea5c1b808fae1a9a9d29de2cf43a7f421894f5))

- **deps**: Bump joserfc from 1.6.4 to 1.6.8
  ([#211](https://github.com/rajveer43/VeloxQuant-MLX/pull/211),
  [`4ea2cca`](https://github.com/rajveer43/VeloxQuant-MLX/commit/4ea2cca0ec8911662c16fc15170d72a1a5d25dcf))

- **deps**: Bump js-yaml from 3.14.2 to 3.15.1 in /docs-site
  ([#220](https://github.com/rajveer43/VeloxQuant-MLX/pull/220),
  [`396634d`](https://github.com/rajveer43/VeloxQuant-MLX/commit/396634daa2c050ca9633dac3862703665791cc5a))

- **deps**: Bump mcp from 1.27.0 to 1.28.1
  ([#209](https://github.com/rajveer43/VeloxQuant-MLX/pull/209),
  [`e542da7`](https://github.com/rajveer43/VeloxQuant-MLX/commit/e542da75dc8ecda81d1cdd0da470f6d15aa5ddd1))

- **deps**: Bump nanoid from 3.3.12 to 3.3.18 in /docs-site
  ([#218](https://github.com/rajveer43/VeloxQuant-MLX/pull/218),
  [`b4dfde3`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b4dfde3ec6ffe16c609b3e826c7c29eb923799fd))

- **deps**: Bump pydantic-settings from 2.14.0 to 2.14.2
  ([#210](https://github.com/rajveer43/VeloxQuant-MLX/pull/210),
  [`a9491c1`](https://github.com/rajveer43/VeloxQuant-MLX/commit/a9491c1d3630a67c665d865a5308491914f90d63))

- **deps**: Bump python-multipart from 0.0.26 to 0.0.31
  ([#214](https://github.com/rajveer43/VeloxQuant-MLX/pull/214),
  [`288c50b`](https://github.com/rajveer43/VeloxQuant-MLX/commit/288c50ba1be0a5c1cd84dae57489efd21c059f5f))

- **deps**: Bump starlette from 1.0.0 to 1.3.1
  ([#212](https://github.com/rajveer43/VeloxQuant-MLX/pull/212),
  [`47135f6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/47135f68714b5e5631d94eb874fe0a7a2b26ffff))

- **deps**: Bump urllib3 from 2.6.3 to 2.7.0
  ([#217](https://github.com/rajveer43/VeloxQuant-MLX/pull/217),
  [`3d2ecbb`](https://github.com/rajveer43/VeloxQuant-MLX/commit/3d2ecbb9acac0c91140b1ad8ae9c74b18035fb68))

### Continuous Integration

- **release**: Collapse merge bursts into one release run
  ([#213](https://github.com/rajveer43/VeloxQuant-MLX/pull/213),
  [`99c04db`](https://github.com/rajveer43/VeloxQuant-MLX/commit/99c04dbe6d6a044514d36672f2eba38340c29eeb))

### Documentation

- Rewrite README prose in a first-person engineering voice
  ([#221](https://github.com/rajveer43/VeloxQuant-MLX/pull/221),
  [`ecbe1e0`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ecbe1e0db012e2ee53c3575988cfc0c12ccd16c0))

### Testing

- **rotation**: Pin bare-m Hadamard behaviour to the installed MLX
  ([#231](https://github.com/rajveer43/VeloxQuant-MLX/pull/231),
  [`9726875`](https://github.com/rajveer43/VeloxQuant-MLX/commit/9726875649955d5d3c8a2c9f6038190a82303d13))


## v0.50.1 (2026-08-18)

### Bug Fixes

- **landing**: Match the test-count format the release sync rewrites
  ([#207](https://github.com/rajveer43/VeloxQuant-MLX/pull/207),
  [`90cc805`](https://github.com/rajveer43/VeloxQuant-MLX/commit/90cc8054f2c89a1082e9b277a48151346f5b82e5))

### Documentation

- **readme**: Surface security and governance policies, trim badge wall
  ([#207](https://github.com/rajveer43/VeloxQuant-MLX/pull/207),
  [`90cc805`](https://github.com/rajveer43/VeloxQuant-MLX/commit/90cc8054f2c89a1082e9b277a48151346f5b82e5))


## v0.50.0 (2026-08-17)

### Build System

- **deps**: Bump cryptography from 47.0.0 to 50.0.0
  ([#204](https://github.com/rajveer43/VeloxQuant-MLX/pull/204),
  [`346b54d`](https://github.com/rajveer43/VeloxQuant-MLX/commit/346b54dd19de373c04cc7ecd5ea9799737a5e1fc))

- **deps**: Bump pillow from 12.2.0 to 12.3.0
  ([#203](https://github.com/rajveer43/VeloxQuant-MLX/pull/203),
  [`88f633a`](https://github.com/rajveer43/VeloxQuant-MLX/commit/88f633a6c1695f5849189e1d2b6e1891d27da6c4))

- **deps**: Bump pyjwt from 2.12.1 to 2.13.0
  ([#202](https://github.com/rajveer43/VeloxQuant-MLX/pull/202),
  [`0dbd9db`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0dbd9dbdb1a70674dfa73ae533b89686c87a6cf6))

### Documentation

- **cache**: Review SnapKV RoPE position semantics vs. paper (#188)
  ([#200](https://github.com/rajveer43/VeloxQuant-MLX/pull/200),
  [`14fb58b`](https://github.com/rajveer43/VeloxQuant-MLX/commit/14fb58bcf1169c67420a52787976249b65cd2a17))

- **cache**: Review StreamingLLM RoPE position semantics vs. paper (#189)
  ([#199](https://github.com/rajveer43/VeloxQuant-MLX/pull/199),
  [`ae38052`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ae38052873ea1810f5dbbbda2ca77df775f8a94b))

- **knorm**: Measure real-model perplexity for L2Norm after RoPE fix (#190)
  ([#205](https://github.com/rajveer43/VeloxQuant-MLX/pull/205),
  [`3c6de43`](https://github.com/rajveer43/VeloxQuant-MLX/commit/3c6de43d578b6a5e8bac8257db1a853dfb9ae97f))

### Features

- **landing**: Lead with unqualified metrics, resequence hero caveats
  ([#206](https://github.com/rajveer43/VeloxQuant-MLX/pull/206),
  [`ba085b2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ba085b2db9a60b0c61f6f13e13b898d913f0f362))


## v0.49.4 (2026-08-15)

### Performance Improvements

- **qfilters**: Wire fused Metal eviction kernels into the cache hot path
  ([#197](https://github.com/rajveer43/VeloxQuant-MLX/pull/197),
  [`297ab04`](https://github.com/rajveer43/VeloxQuant-MLX/commit/297ab04cca776396eff2a2a87d51662896805a4a))


## v0.49.3 (2026-08-15)

### Bug Fixes

- **cache**: Generalize RoPE offset fix to TOVA cache; audit H2O (#175)
  ([#196](https://github.com/rajveer43/VeloxQuant-MLX/pull/196),
  [`6cef57b`](https://github.com/rajveer43/VeloxQuant-MLX/commit/6cef57bbd75f08dadd4263733798ba8ac46ad00b))


## v0.49.2 (2026-08-15)

### Bug Fixes

- **cache**: Generalize RoPE offset fix to L2Norm cache (#174)
  ([#195](https://github.com/rajveer43/VeloxQuant-MLX/pull/195),
  [`0241a5d`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0241a5d4eb1f6be894604aaeca97f2b80415fc62))


## v0.49.1 (2026-08-14)

### Performance Improvements

- **qfilters**: Vectorize the per-(B, H) eviction loop
  ([`701388c`](https://github.com/rajveer43/VeloxQuant-MLX/commit/701388ca36b221795d4c876bcf74c72ddae7311e))


## v0.49.0 (2026-08-14)

### Bug Fixes

- **cache**: Correct RoPE offset after eviction in SnapKV and StreamingLLM
  ([`e4186f7`](https://github.com/rajveer43/VeloxQuant-MLX/commit/e4186f7f6a9c0429bfb169ca21a7b84ca06517f5))

- **docs**: Use MDX comment syntax for the blog truncation marker
  ([`8669596`](https://github.com/rajveer43/VeloxQuant-MLX/commit/8669596cbed18325407c95931ea05420f7c9ec63))

- **qfilters**: Correct RoPE offset after eviction, measure real perplexity
  ([`cf9a255`](https://github.com/rajveer43/VeloxQuant-MLX/commit/cf9a2552024d0c24c579f05ba98ee0581e6c52f7))

### Code Style

- Apply ruff format to Q-Filters markdown code blocks
  ([`0b023d2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0b023d2c17a80855be93fbe955e2a787ff452c88))

### Documentation

- **blog**: Add Q-Filters real-model investigation write-up
  ([`570b7e2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/570b7e27731e3afca4272656918a9cb3fdd783bb))

- **changelog**: Record the SnapKV/StreamingLLM RoPE offset fix
  ([`2d93a41`](https://github.com/rajveer43/VeloxQuant-MLX/commit/2d93a411c4b4317746dc912dbd98c8401725b099))

### Features

- **qfilters**: Paper-faithful query-SVD calibration + fused Metal kernels
  ([`3cc4142`](https://github.com/rajveer43/VeloxQuant-MLX/commit/3cc41429c4d9f84fca7fb7a0dff33e0888e845c5))

### Testing

- **qfilters**: Add TTFT, throughput and NIAH harness vs SnapKV/StreamingLLM
  ([`a600927`](https://github.com/rajveer43/VeloxQuant-MLX/commit/a60092748b3058d1d72799268035b3abed1b8551))

- **qfilters**: Validate query-SVD calibration on real trained models
  ([`3fa0c60`](https://github.com/rajveer43/VeloxQuant-MLX/commit/3fa0c60d539b21524643fd1ab223c2ce41ccf940))


## v0.48.5 (2026-08-13)

### Code Style

- Apply ruff 0.16.3 formatting to CAM docs and tests
  ([`445d0c0`](https://github.com/rajveer43/VeloxQuant-MLX/commit/445d0c0b22a96085b564d8729295ce22bbb7a77b))

### Testing

- Fix two release-workflow failures on master
  ([`cd5aa6f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/cd5aa6f119b26bd86408cbfcc888eb87c8ae7fe3))


## v0.48.4 (2026-08-13)

### Code Style

- Apply ruff 0.16.3 formatting to cache/base.py
  ([`088334a`](https://github.com/rajveer43/VeloxQuant-MLX/commit/088334a1a0b9724f3b796a68a4de74f22775793b))


## v0.48.3 (2026-08-13)


## v0.48.2 (2026-08-13)

### Code Style

- Satisfy ruff 0.16.3 format across the AdaKV changes
  ([`7000e98`](https://github.com/rajveer43/VeloxQuant-MLX/commit/7000e9801e5b22d32b0c38f1be3b1b814f3ff558))


## v0.48.1 (2026-08-13)

### Bug Fixes

- **kvquant**: Correct outlier selection, decode protection, and add sink-aware quantization
  ([`24fad52`](https://github.com/rajveer43/VeloxQuant-MLX/commit/24fad5281ab1e6ad39db16b1ab7ea139720b2c3f))


## v0.48.0 (2026-08-12)

### Documentation

- **kivi**: Correct stale register-caching note in channel kernel
  ([`1a23cd3`](https://github.com/rajveer43/VeloxQuant-MLX/commit/1a23cd36d7c715864b46555092139aa708c70d84))

### Features

- **kivi**: Fused Metal kernel for asymmetric group quantization
  ([`0aeeead`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0aeeead8717d4ff5175800ea70bc32d362f0a8cb))

### Performance Improvements

- **kivi**: Split into layout-specific kernels, drop the transpose
  ([`3b3eca6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/3b3eca67cc3ecf328a1e678d480b03e8c3e1bc8d))


## v0.47.1 (2026-08-12)

### Bug Fixes

- **kivi**: Buffer residual to group_size-aligned flushes
  ([#162](https://github.com/rajveer43/VeloxQuant-MLX/pull/162),
  [`d8922a8`](https://github.com/rajveer43/VeloxQuant-MLX/commit/d8922a84652146e9c8ff81c62c9cb99dc210693b))

### Documentation

- **landing**: Rewrite the method picker for a non-expert audience
  ([`efb1ca3`](https://github.com/rajveer43/VeloxQuant-MLX/commit/efb1ca3bcc832cf702180ee7e988903d52315ff9))

### Testing

- **kivi**: Cover the documented model geometries; fill in the docs table
  ([`cbaef08`](https://github.com/rajveer43/VeloxQuant-MLX/commit/cbaef08abfca3780fce987350c40a266199eb241))


## v0.47.0 (2026-08-11)

### Documentation

- **landing**: Rewrite the playground for a non-expert audience
  ([`052df5a`](https://github.com/rajveer43/VeloxQuant-MLX/commit/052df5adcb6d7c3da9940a95273b1f9fb4771a06))

### Features

- **landing**: Add GoatCounter privacy-friendly analytics
  ([`8977926`](https://github.com/rajveer43/VeloxQuant-MLX/commit/897792686d260ec43c1d777ad92a2a18ebb38b44))


## v0.46.0 (2026-08-11)

### Bug Fixes

- **pyproject**: Point Documentation URL at the Netlify docs site
  ([`06300d9`](https://github.com/rajveer43/VeloxQuant-MLX/commit/06300d968cc241fe67d6355b1b7edced58db4664))

- **registry**: Distinguish not-trimmable from crash-tier, unblocking releases
  ([`c90c3de`](https://github.com/rajveer43/VeloxQuant-MLX/commit/c90c3de2e81d7ab87f5a857dea7b2b2aef9098d2))

- **release**: Sync the landing hero badge to the real markup, not a dead main.js pattern
  ([`eef8a32`](https://github.com/rajveer43/VeloxQuant-MLX/commit/eef8a3274e2ac53f800826b5b3ee9e6bccd0e068))

### Code Style

- Satisfy ruff format on the new docs page and a pre-existing test
  ([`1a9f16b`](https://github.com/rajveer43/VeloxQuant-MLX/commit/1a9f16b096ab8aebb84bfcde7aa7d81eef689fbe))

### Documentation

- **readme**: Condense method library, tighten caveats, add project context
  ([`b2efe24`](https://github.com/rajveer43/VeloxQuant-MLX/commit/b2efe24c3cb0f0843050c26ecb3431a2712dadb5))

- **readme**: Drop SmolLM2-135M from benchmark tables
  ([`5f3d37f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/5f3d37f06c10a5ac5705a3d4e53e65581b4be9de))

- **readme**: Loosen research-paper-dense prose into plain developer language
  ([`515db67`](https://github.com/rajveer43/VeloxQuant-MLX/commit/515db6793b3186bd412f015ab906991e0c68f9de))

- **transfer**: Correct the Metal kernel speedup, 12.9x was not reproducible
  ([`1ff8e42`](https://github.com/rajveer43/VeloxQuant-MLX/commit/1ff8e42451ec866a39704d15bba3652124100af1))

### Features

- **landing**: Add FAQ section, drop SmolLM2-135M from benchmark table
  ([`2693639`](https://github.com/rajveer43/VeloxQuant-MLX/commit/26936395ce7146e89bbde836c41f9dc89ab54157))

- **transfer**: Cross-model KV cache transfer via closed-form ridge mapper
  ([`f3fdf3f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/f3fdf3ff37d2e71b3e4c8b86786de0fcb1ca99a2))


## v0.45.0 (2026-08-10)

### Features

- **keyformer**: Annealing, RoPE remap, fused Metal kernel, real-model validation
  ([`ab82606`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ab826068c56dc4d726c0f4fa0eff8ea4c82071d7))


## v0.44.4 (2026-08-10)

### Bug Fixes

- **h2o**: Correct RoPE position desync after eviction
  ([`470ab0f`](https://github.com/rajveer43/VeloxQuant-MLX/commit/470ab0fe9f07049082a9191404086a0cda274577))

- **h2o**: Grace period fixes the early-token permanent-freeze bug
  ([`85eb8e5`](https://github.com/rajveer43/VeloxQuant-MLX/commit/85eb8e5e2d5ec98bc50e76a2fbe37d443105b05d))

- **h2o**: Score decay fixes stale-token dominance after grace fixed the freeze
  ([`62f3a72`](https://github.com/rajveer43/VeloxQuant-MLX/commit/62f3a72b52317cab472dbfdd649aa556cbcecd78))

### Performance Improvements

- **h2o**: Fused Metal kernel for the over-budget eviction step
  ([`6f7db91`](https://github.com/rajveer43/VeloxQuant-MLX/commit/6f7db918495a0f1f09732ff9cc6e9e4c7b42332f))

- **h2o**: Vectorize below-budget prefill path, fixing a real crash
  ([`dd3e0b1`](https://github.com/rajveer43/VeloxQuant-MLX/commit/dd3e0b1a15e55dfd31b67548d4a8d54826145550))


## v0.44.3 (2026-08-10)

### Bug Fixes

- **gear**: Wire KCVT base quantizer (per-channel keys, per-token values)
  ([`1aaa312`](https://github.com/rajveer43/VeloxQuant-MLX/commit/1aaa312e3c5b82033562527ae5d2a479634d91e6))

### Code Style

- Apply ruff format to gear PR files
  ([`0d6de55`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0d6de55d653f43962530b31af12b55e84a5952cd))

- Fix ruff format drift in RVQ docs and cache
  ([`9aad459`](https://github.com/rajveer43/VeloxQuant-MLX/commit/9aad4593b9149fe3932e43e3288908cbc3f884b6))

- **amc**: Fix ruff format alignment in doc code blocks
  ([`0ccac92`](https://github.com/rajveer43/VeloxQuant-MLX/commit/0ccac92e3365c297bab0e079ccd780682eb155d4))

### Documentation

- **amc**: Make AMC-adapted algorithm page consumer-focused
  ([`65728d2`](https://github.com/rajveer43/VeloxQuant-MLX/commit/65728d2a690540b1b9d75eedc680d09a3d5f54b0))


## v0.44.2 (2026-08-09)

### Bug Fixes

- **cache**: Store turboquant_rvq keys genuinely packed, not dequantized fp16
  ([`ae04c26`](https://github.com/rajveer43/VeloxQuant-MLX/commit/ae04c26f5ebd48d0bdeb0e709a52233b10441967))

### Documentation

- Correct turboquant_rvq API examples and update benchmark numbers
  ([`4bf0cf3`](https://github.com/rajveer43/VeloxQuant-MLX/commit/4bf0cf3b67fe256f0898510783aa3698ac0255c0))

- Make TurboQuant RVQ page read as a normal feature page
  ([`53069d6`](https://github.com/rajveer43/VeloxQuant-MLX/commit/53069d657fbd589fd1b4c5a43cf1218856ac9772))

- Record turboquant_rvq packed-storage investigation and update roadmap
  ([`884bbd8`](https://github.com/rajveer43/VeloxQuant-MLX/commit/884bbd831c409e24f848a273debb9967d358d170))


## v0.44.1 (2026-08-09)

### Bug Fixes

- **cache**: Default KVCacheConfig method to turboquant_rvq, not turboquant_prod
  ([`f6e9434`](https://github.com/rajveer43/VeloxQuant-MLX/commit/f6e94348b4eb5ffd71b12a13019f0a52a0a8c6d2))


## v0.44.0 (2026-08-09)

### Features

- **release**: Publish to TestPyPI then PyPI via Trusted Publishing (OIDC)
  ([`61d49f9`](https://github.com/rajveer43/VeloxQuant-MLX/commit/61d49f99695fa7ea5db984bf50e2a03c8b20f960))


## v0.43.7 (2026-08-09)

### Bug Fixes

- **ci**: Release.yml re-ran full release steps when nothing actually changed
  ([`828c627`](https://github.com/rajveer43/VeloxQuant-MLX/commit/828c6277f3b181d957ba1e76787e11f0300c35a4))

### Documentation

- **readme**: Replace static status claims with live CI/PyPI badges
  ([`38a4752`](https://github.com/rajveer43/VeloxQuant-MLX/commit/38a47527b07baac38f4c8020fe7f54204935fb24))


## v0.43.6 (2026-08-09)

### Bug Fixes

- **ci**: Release-notes extraction didn't match semantic-release's real heading format
  ([`4344666`](https://github.com/rajveer43/VeloxQuant-MLX/commit/4344666bff7db1707ef17b33c48dda5869e93bf0))


## v0.43.5 (2026-08-09)

### Bug Fixes

- **release**: CHANGELOG.md was never auto-updated; add mode=update + backfill
  ([`cd27c1e`](https://github.com/rajveer43/VeloxQuant-MLX/commit/cd27c1eecd8d09b4b5632f31bf08d3c17103dad9))


## [0.43.4] — 2026-08-09

### Fixed

**Release pipeline had been silently broken since PR #109 (2026-08-08).** `test_mlx_lm_patch.py::test_patch_model_kv_cache_refuses_standalone_method` called `pytest.raises(...)` with no `import pytest` in the file, so it failed with a bare `NameError` rather than a real assertion. This test runs inside `release.yml`'s test-suite gate, so every release run since PR #109 failed at that step -- `pyproject.toml`/`__init__.py`/git tags reached v0.43.2/v0.43.3 from earlier partial runs, but no GitHub Release or CHANGELOG.md entry had been produced for any of it. Fixed by adding the import (PR #117).

Fixing the import surfaced a real bug underneath: `patch_model_kv_cache()` (`veloxquant_mlx/integration/mlx_lm_patch.py`) assigned `model.make_cache` *before* validating the config, so `KVCacheBuilder.for_model()`'s `QuantizerConfigError` for standalone methods (e.g. `"spectral"`) left the model permanently patched with a broken cache hook -- contradicting the function's own documented contract. Fixed by validating/building the cache before assigning the hook (PR #117).

`semantic-release publish` does not create a GitHub Release from scratch -- it only uploads build artifacts to one that already exists. With `--no-vcs-release` on the bump step (this workflow tags and pushes manually), nothing ever created the release object, so `publish` logged a warning and exited 0 without doing anything -- a silent no-op that looked like success. Replaced with `gh release create` (PR #118).

`CHANGELOG.md` had never been auto-updated by any release, ever: `python-semantic-release`'s default changelog mode (`"init"`) only generates a changelog if the target file doesn't already exist, and this file was hand-authored before the first automated release ran. Every release since silently no-op'd on it. Fixed by setting `mode = "update"` and adding the `<!-- version list -->` insertion marker this file was missing.

## [0.43.0] – [0.43.3] — 2026-08-08

Twenty-four bug-fix PRs (#88–#116) merged in one day, none of which reached a GitHub Release or this file until the pipeline fix above (v0.43.4) restored automated changelog generation. Backfilled by hand from the merged PRs' own descriptions.

### Fixed — KV-cache classes

- **`KVQuantKVCache` crashed on first attention call** (#88). Its public `.bits` property was treated by mlx_lm's SDPA dispatcher as a sentinel for the quantized-matmul path, which then failed looking for a nonexistent `.group_size`. Renamed to `.nuq_bits`.
- **`KIVIKVCache` never quantized decode tokens** (#89). `update_and_fetch` compared only the current call's row count against `residual_length`; every decode call has `S == 1`, so the fp16 residual window grew unboundedly instead of staying capped. Fixed by tracking true cumulative offset. `SinkProtectedKVCache` (subclasses `KIVIKVCache`) shared and inherited the fix.
- **`SinkProtectedKVCache` selected sink positions from batch element 0 only** (#90), misprotecting outlier tokens in every other batch element. Sink tracking is now per-batch-element.
- **`SnapKVKVCache` exceeded its budget under mlx_lm's standard chunked prefill** (#91). Prompts longer than one `prefill_step_size` chunk let each chunk compress independently and append, growing retained tokens past `snap_budget`. Later chunks now recompress the full retained-so-far set and replace, not append.
- **17 of 40 eviction/compression cache classes crashed on ordinary generation** (#92) — `AttributeError` on `.state`, hit on the first prefill chunk past `prefill_step_size` — because they never called `super().update_and_fetch(...)`, leaving mlx_lm's base `.keys`/`.values`/`.offset` permanently `None`.
- **`QJLKVCache`/`PolarQuantKVCache` crashed once tokens exceeded configured `capacity`** (#93). `attend()` used an unbounded lifetime counter instead of the buffer's actual capacity-capped live size.
- **`sliding_window` silently produced a fully broken cache for 35+ of 40 methods** (#94). `KVCacheFactory.create()` wrapped any method regardless of whether it implemented the wrapper's expected interface. Now raises `QuantizerConfigError` at creation for incompatible methods.
- **`XQuantCoordinator` and `MiniCacheCoordinator` crashed generation at their configured `max_ctx`** (#95, #96) even in routine use — published segments were never reclaimed after being consumed. Segments now track `reads_remaining` and are reclaimed once every expected reader has fetched them.

### Fixed — Quantizers, allocators, transforms

- **`zipcache_reconstruct` hardcoded `group_size=32`** (#97), silently desyncing dequant grouping whenever `zipcache_group_size` was set to anything else.
- **KVTC entropy coding had no fixed-width fallback** (#98), so `entropy_coding_gain` regularly fell below 1.0 on near-uniform quantized codes.
- **`pyramid_budgets()` exceeded `avg_budget` by up to 17.5% for `beta > 2.0`** (#99). The floor clamp raised sub-floor layers without compensating elsewhere; now uses water-filling renormalization.
- **`water_fill_bits`'s zero-eigenvalue early-exit branch silently dropped `total_bit_budget % d` bits** (#100).
- **`apply_dual_transform_keys`/`queries` silently averaged smooth factors across heads for genuine 3D per-head input** (#102) instead of true per-head math.
- **`allocate_bits_ratequant()` could return bit-widths not in `bit_choices`** for non-contiguous choice sets (#106).
- **`PolarQuantizer` level-1 angles caused severe silent reconstruction corruption** (#109). `arctan2`'s native `(-pi, pi]` range didn't match the level-1 codebook's assumed `[0, 2*pi)` support — roughly half of all level-1 angles were quantized against an all-positive codebook. Affected every `PolarQuantizer` use. Reconstruction MSE in testing dropped from 1.28 to 0.055 at `b=3, d=64` after the fix.
- **`AdaptiveScalarCodebook.observe()` silently dropped calibration rows beyond `n_calib`** within a single call (#110).
- **`is_hadamard_compatible(d)` returned `True` for `d` in `{12, 20, 28}`** (#111), which crash `mx.hadamard_transform` with an uncatchable Metal shader compile failure.

### Fixed — Core, observers, artifacts

- **`NpyArtifactStore` save methods were not atomic** (#107), risking partial-file reads under concurrent `exists()`-then-`save()` construction. Now write-temp-then-`os.replace()`.
- **`DistortionObserver.report()` computed an unweighted mean-of-batch-means MSE** rather than a sample-weighted mean (#108), skewing `mse_ratio` under varying batch sizes.
- **`EncodedVector.memory_bytes()` undercounted usage for outlier-split (composite) quantizers** (#112) by omitting `outlier_idx`.
- **`calibrate_layer_sensitivities()` permanently corrupted `model.make_cache` if the calibration forward pass raised** (#105) — no `try/finally` around the monkey-patch/restore. Same bug class independently found and fixed again in the release-pipeline itself, above (`patch_model_kv_cache`).
- **`allocate_bits_ratequant` (RateQuant)'s bit_choices membership fix** — see Quantizers section above.

### Fixed — DSA

- **`SortedChannelIndex.top_k()` returned duplicate channel indices** (#66/#114) when a channel was lazily re-inserted with a magnitude coincidentally matching its current live value. Fixed by tagging each insert with a monotonically increasing version and comparing `(magnitude, version)` tuples; also fixes unbounded heap growth from the same root cause.

### Changed

- **Inline MSL kernel strings extracted to standalone `.metal` files** (#64/#113). All 19 kernel sources moved from inline Python strings to `veloxquant_mlx/metal/src/*.metal`, read at import time. Pure refactor — `mx.fast.metal_kernel` still JIT-compiles the same source string at call time.

### Docs

- **Documented the `veloxquant_mlx/tests/` vs `tests/non_metal/` split** (#63/#115) — why the two directories exist and can't be merged (`veloxquant_mlx/__init__.py` imports `mlx`, which only installs on Apple Silicon).

### Landing page

- **Removed the "Verifiable, not asserted" (proof-of-maintenance) section** (#101/#103).
- **Aligned landing-page section order and cross-links with the three visitor personas defined in `#scenarios`** (#104/#116): reordered Methods before Quickstart, added persona-entry nav link, method-picker → quickstart tab handoff, compare → benchmarks cross-link, quality-cost mention in the calculator, install micro-CTAs at each persona endpoint, and synced `playground.html`'s independently-drifted nav bar to match `index.html`'s.

## [0.42.0] — 2026-07-27

### Added

**Fused group-affine (KIVI-style) decode + attention Metal kernel** —
`scalar_fused_decode_attend`
(`veloxquant_mlx/metal/_scalar_attend.py`) runs scaled-dot-product
attention directly over an asymmetric group-min/max ("affine")
quantized KV cache: the KIVI / SKVQ / Kitty / group-quant family, where
keys and values are `uint8` codes plus a per-group `(scale, zero)`
pair. It reconstructs `k_hat = code*scale + zero` (per-channel groups)
and `v_hat = code*scale + zero` (per-token groups) in-register inside a
FlashAttention-style online softmax, so no dequantized `K_hat`/`V_hat`
is ever materialized in DRAM.

This is the scalar/group-quant analogue of the existing codebook fused
attends (`_rvq_attend`, `fused_sdpa`, `_rabitq_attend`). It kills the
`dequantize -> DRAM -> SDPA` round-trip the pure-MLX path pays every
decode step; the win compounds with context because the fp16 `K_hat`
grows linearly with `S_kv` while the packed codes stay `16/b` times
smaller. The kv axis is split flash-decoding style across `nsg`
SIMD-groups so single-query decode shapes still fill the GPU (`nsg=8`
tuned on M4), and one compiled kernel serves any `(S_kv, D, g)`.

Measured on Apple M4 (10-core GPU), B=1 H=32 D=128 b=2 g=32 S_q=1, vs.
dequantize -> MLX SDPA: **6.4x at S_kv=512 rising to 12.2x at
S_kv=65536**. Parity max abs error 1.2e-4 — the fp32 softmax
accumulation makes it more accurate than the fp16 baseline it replaces.
Covered by `veloxquant_mlx/tests/metal/test_scalar_attend.py`.

**Mac / RAM method recommender** — `veloxquant_mlx/tools/mac_recommender.py`
plus a `veloxquant recommend` CLI subcommand
(`veloxquant_mlx/cli/recommend.py`) pick a compression method and bit
budget from a given Mac's unified memory and target model/context.

**Interactive landing playground** (`landing/playground.html`) — a
browser-side compression explorer with a Compression Lab and a
benchmark browser, plus a landing page reworked for a general audience.

### Fixed

- **Release automation had been silently skipping every release since
  0.41.0.** `build_command = false` in `pyproject.toml` is rejected by
  python-semantic-release >=10 (it validates as a string), and the
  release workflow discarded the resulting stderr via
  `2>/dev/null || echo ""` — so a hard config error was indistinguishable
  from "nothing to release" and four merged PRs never produced a tag.
  Fixed the value to `""`, and the workflow now fails loudly on any
  version-computation error it cannot identify as a genuine no-release.
- **`major_on_zero = false` alone would have released 1.0.0.** On
  python-semantic-release >=10 staying on `0.x` also requires
  `allow_zero_version = true`; without it the next release computed
  `1.0.0` rather than `0.42.0`, despite no breaking changes.
- Documented `scalar_fused_decode_attend` and `rabitq_prefill_attend`
  (the latter shipped in 0.40.x but was never added to the docs site) in
  the Metal kernels guide and the Metal API reference.
- Install guide: broken `precompute` command corrected; install docs
  consolidated.
- Non-Metal CI no longer imports `mlx` through the package `conftest`;
  recommender tests moved out of the package tree.
- Release workflow installs `mlx-lm`, without which the test gate could
  not import the package it was gating.

## [0.41.0] — 2026-07-20

### Added

**mlx-vlm vision-language model support** —
`patch_vlm_kv_cache(model, config)`
(`veloxquant_mlx/integration/mlx_vlm_patch.py`) wires VeloxQuant caches
into mlx-vlm models (Qwen2-VL, LLaVA, …). Verified against mlx-vlm
0.6.5: the single-prompt generate path builds its cache through
`model.language_model.make_cache()`, and the patch overrides exactly
that hook — the top-level model is left unpatched so mlx-vlm's
batch/session path (whose `to_batch_cache()` rejects foreign cache
types) stays safe. Unlike the text patch, caches are rebuilt fresh on
every `generate()` call, so repeated generations never leak KV state.
Token-eviction methods (`snapkv`, `h2o`, `pyramidkv`, …) emit a
`UserWarning` since image tokens sit in the prompt prefix and can be
evicted. 8 integration tests, including one driving mlx-vlm's real
`make_prompt_cache`.

### Fixed

- Integration guide "Pattern 2" documented a nonexistent
  `patch_mlx_lm` function with a nonexistent `bits=` kwarg; it now
  shows the real `patch_model_kv_cache` API.
- README badges: stale test count (now 1417 passing, verified by a
  full-suite run) and stale changelog version; removed the downloads
  badge and download-count claim.

## [0.40.0] — 2026-07-19

### Added

**Fused RaBitQ asymmetric Metal kernel pipeline** — three new kernels
form a fully GPU-resident path for a 1-bit-key / 4-bit-value cache:

- `rabitq_fused_attend` (`veloxquant_mlx/metal/_rabitq_attend.py`) —
  single-dispatch attention: keys scored from packed sign bits via
  XOR + popcount, online softmax split across 8 SIMD-groups
  (flash-decoding style), values gathered from a scalar codebook. No
  dequantized K or V is ever materialized. Measured 1.10–1.78× vs the
  dequantize+SDPA baseline (D=128, 8 heads, S_kv=512–8192, Apple M4).
- `rabitq_encode` (`_rabitq_encode.py`) — fused rotate + binarize +
  bit-pack + L1 magnitude in one dispatch; `simd_ballot` turns each
  SIMD-group's 32 sign predicates into 4 packed bytes in a single
  instruction. 6× vs the numpy round-trip at N=32768.
- `rabitq_pack_values` (`_rabitq_values.py`) — two 4-bit value indices
  per byte; the attend kernel reads nibbles natively (auto-detected
  from the v_idx shape) with bit-identical outputs, halving value-cache
  memory and bandwidth.

63 new parity tests (`tests/metal/test_rabitq_{attend,encode,values}.py`),
including an end-to-end encode→attend test and packed-vs-unpacked
bit-exactness. Benchmarks: `scripts/metal_rabitq_attend_bench.py`,
`scripts/metal_rabitq_encode_bench.py`.

## [0.39.1] — 2026-07-17

### Fixed

**Metal dispatch bug — VecInfer's fused encode+decode kernels silently
dropped most/all tokens.** `vecinfer_encode_decode_metal` and
`vecinfer_encode_decode_simple_metal` (`veloxquant_mlx/metal/_vecinfer.py`)
launch one `D`-wide threadgroup per token, but their Metal dispatch passed
`grid=(n_tokens, 1, 1)` — `mx.fast.metal_kernel`'s `grid` argument is
specified in **threads**, not threadgroups, so this silently truncated to
`floor(n_tokens / D)` threadgroups (zero whenever `n_tokens < D`). Every
token beyond that count kept its uninitialized output-buffer contents
instead of a real reconstruction or codebook index. This affected every
VecInfer Metal-accelerated encode/decode call, not only the specific
cross-kernel-reuse case the regression tests happened to catch. Fixed by
dispatching `n_tokens * D` threads, matching every other kernel in
`veloxquant_mlx/metal/`. `test_vecinfer_fused_sdpa.py` and
`test_vecinfer_metal_parity.py` (5 failing tests) now pass.

**Silent sink-token eviction when `n_sink >= budget`.** `init_pyramid_state`,
`init_squeeze_state`, `init_chunkkv_state`, and `init_curdkv_state`
(`veloxquant_mlx/quantizers/{pyramidkv,squeeze,chunkkv,curdkv}.py`) accepted
degenerate sink/budget configurations that leave no evictable room, silently
evicting tokens documented as sink-protected. `h2o`/`tova` already guarded
this (see 0.39.0's `a78cd7f`); the same `n_sink < budget` check is now
applied to these four siblings, with matching regression tests.

## [0.38.0] — 2026-07-14

### Venue exception (read first)

**AMC-adapted is the second method in VeloxQuant-MLX (2 of 40) that does not
trace to a verified peer-reviewed venue** — the first was NestedKV-adapted
(v0.37.0, below). AMC (arXiv:2607.10109) is a bare single-revision preprint
(submitted 2026-07-11, no Comments/journal-ref field), live-verified 3 days
later on 2026-07-14. It ships anyway as a **second one-time, user-directed
exception** to this repo's standing venue-verification rule. The next method
survey reverts to requiring a verified venue — this is not a new standing
precedent. The paper is also filed under `cs.IR` (Information Retrieval), an
unusual category for what is fundamentally a hardware architecture paper —
noted as a minor oddity, not a disqualifier.

### Scope cut (read first)

**Roughly half of AMC's source paper (Sections IV-V: 45nm CMOS RTL, Verilog
clock-gating, the Precision-Gated Systolic Array, the Narrow-Width SRAM
write-back buffer, all pJ/µJ energy figures, the EDAP/Pareto silicon
comparisons) is entirely out of scope for this software port.**
VeloxQuant-MLX is a pure-software MLX library with no RTL/silicon layer to
target — none of that half is implemented. Only Section II-A (the software
saliency engine) and Section III (rank/precision scaling math), plus
Algorithm 1 Phase I's offline SVD/PCA channel-order calibration, are ported.

### Added — AMC-adapted saliency-driven tiered rank + precision (`method="amc"`)

The library's 40th method, and the first whose family is "adaptive
rank+precision" rather than "eviction" — no token is ever dropped. Every
rank-adaptive method already in the repo (Palu) and every bit-width-adaptive
method (KIVI, SKVQ, RateQuant) adapts **one** axis; AMC is the first to drive
**both** rank and bit-width from a **single** per-token L1-norm saliency
score, via three discrete tiers (High: rank 128/16-bit, Mid: rank 43/8-bit,
Low: rank 8/4-bit at head_dim=128). Inspired by "Adaptive Model Compression
(AMC): Saliency-Driven Resource Allocation for Ultra-Low-Power Transformer
Inference" (Hu, Yuan, Hu, Yin, Li, Suchter — Apple; arXiv:2607.10109) —
shipped as "AMC-adapted (VeloxQuant-MLX implementation)," **not a faithful
port**.

- `veloxquant_mlx/quantizers/amc_calibration.py` — `amc_calibrate_channel_order`
  (offline, one-time SVD-based variance-descending channel permutation,
  Algorithm 1 Phase I, reusing the same `mx.linalg.svd` pattern as
  Palu/SVDq), `amc_permute_weights`.
- `veloxquant_mlx/quantizers/amc.py` — `amc_saliency`/`amc_query_aware_saliency`
  (Eq. 1-3), `amc_assign_tiers` (percentile tiering via `dsa.MaxHeap` top-k
  selection), `amc_adaptive_thresholds` (Eq. 4-5, `dsa.RingBuffer`-backed
  trailing-window variance tracking), `amc_apply_rank_mask` (Eq. 6),
  `amc_quantize_tier` (Eq. 7, reuses the shared group quantizer),
  `amc_pack_low_tier` (`dsa.BitPackBuffer`-backed dense 4-bit packing), byte
  helpers.
- `veloxquant_mlx/cache/amc_cache.py` — `AMCKVCache`; every call (prefill or
  decode) scores, tiers, rank-masks, and quantizes every token — no
  eviction ever, stored sequence length always equals tokens seen.
- `veloxquant_mlx/cache/base.py` — `"amc"` added to the method `Literal`;
  `amc_k_high` (0.20), `amc_k_mid` (0.30), `amc_use_query_saliency` (False),
  `amc_query_alpha` (0.5), `amc_adaptive_thresholds` (False),
  `amc_threshold_window` (64), `amc_gamma` (0.1), `amc_calib_variance`
  (None), `amc_group_size` (32) config fields; factory branch;
  unknown-method error string extended.
- 51 tests (9 calibration + 23 quantizer + 19 cache,
  `veloxquant_mlx/tests/quantizers/test_amc_calibration.py` +
  `veloxquant_mlx/tests/quantizers/test_amc.py` +
  `veloxquant_mlx/tests/cache/test_amc_cache.py`) and a deterministic
  offline benchmark (`benchmark_scripts/benchmark_amc.py` +
  `amc_benchmark_results.json`).
- Docs: `docs-site/docs/algorithms/amc.md`, sidebar entry, overview
  table/bullet, changelog entry.

### Honest scope

- **No verified peer-reviewed venue** and **hardware/RTL half of the paper
  out of scope** — see the two sections above.
- **Compression-only, never eviction** — a structurally different family
  from every other eviction method in the repo (H2O, SnapKV, CurDKV,
  NestedKV, and more all drop tokens; AMC never does).
- **Query-aware saliency (Eq. 3) and closed-loop adaptive thresholds (Eq.
  4-5) are opt-in, off by default.** The default path is pure
  magnitude-only scoring (Eq. 1-2), matching the paper's primary reported
  configuration.
- **Offline SVD/PCA channel-order calibration required** for the rank mask
  to be meaningful — `AMCKVCache` does not auto-invoke it; callers must run
  `amc_calibrate_channel_order` themselves before deployment, the same
  category of requirement as Palu/SVDq's calibration step.
- **A real, honestly-reported weakness found during benchmark
  construction**: on activation distributions with no genuine saliency
  signal (uniform magnitude), AMC's fixed percentile tiering comes out
  roughly 100x **worse** in reconstruction MSE than a matched-budget
  uniform baseline — not merely neutral. On the geometry the mechanism is
  designed for (sparse outliers), AMC beats the same baseline by roughly
  8x. Both reported plainly in `benchmark_scripts/benchmark_amc.py`'s
  closing summary, not hidden.
- The paper's own energy/throughput/accuracy numbers (59.2% energy
  reduction, 2.24x throughput, 3.6% accuracy trade-off) are hardware-measured
  on the paper's own 45nm RTL simulation and a specific 3-layer synthetic
  transformer setup (`num-samples=4000, seq-len=32, vocab-size=16`) — not
  reproduced here.

## [0.37.0] — 2026-07-14

### Venue exception (read first)

**NestedKV-adapted is the first method in VeloxQuant-MLX (1 of 39 at the
time) that did not trace to a verified peer-reviewed venue.** Every one of
the prior 38
methods required a live-verified venue before implementation; NestedKV
(arXiv:2605.26678) is still a bare single-revision preprint (submitted
2026-05-26, no Comments/journal-ref field) as of 2026-07-14. It ships anyway
as a **one-time, user-directed exception** to this repo's standing
venue-verification rule. The next method survey reverts to requiring a
verified venue — this is not a new precedent. See
`paper/research/surveys/NEW_METHOD_SURVEY_V21.md` for the full rationale,
including why two prior candidates (KVP/Apple/ICML 2026, KQ-SVD) were
rejected first — one for requiring shipped pretrained weights, one for a
venue claim that failed independent re-verification.

### Added — NestedKV-adapted multi-scale ensembled prefill eviction (`method="nestedkv"`)

The library's 39th method, joining the token-eviction family (H2O, SnapKV,
CurDKV, PyramidKV, Keyformer, MorphKV, KVzip, and more). Every existing
eviction method scores a token from **one** importance signal. NestedKV's
axis: keep **three** parallel key-only continuum-memory statistics —
stable/global, episodic/block-local, current/recent-window — score every
token's anomaly against each independently, and combine the three rankings
via a training-free **head-adaptive blend** (which scale is most
discriminative on this head) and a per-token **surprise-gated route** (route
to the single strongest scale when the three disagree, instead of
averaging). Inspired by "NestedKV: Nested Memory Routing for Long-Context KV
Cache Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu; arXiv:2605.26678)
— shipped as "NestedKV-adapted (VeloxQuant-MLX implementation)," **not a
faithful port**.

- `veloxquant_mlx/quantizers/nestedkv.py` — `NestedKVState`, `nestedkv_score`
  (one-shot per-head anomaly scoring over the full prefill sequence),
  `nestedkv_allocate_head_budgets` (cross-head budget competition, the
  paper's component 5), `nestedkv_compress_prefill`, `nestedkv_append_decode`,
  byte helpers.
- `veloxquant_mlx/cache/nestedkv_cache.py` — `NestedKVKVCache`, mirroring
  `SnapKVKVCache`'s prefill-once/decode-append phase split (not H2O's/
  CurDKV's per-step loop); zero-pads ragged per-head outputs (the first
  method here with legitimately unequal per-head token counts, found while
  fixing an `mx.stack` shape-mismatch during test-writing) purely for
  tensor-stacking — byte accounting is computed from each head's true,
  unpadded state.
- `veloxquant_mlx/cache/base.py` — `"nestedkv"` added to the method
  `Literal`; `nestedkv_budget` (512), `nestedkv_n_sink` (4),
  `nestedkv_window` (64), `nestedkv_beta` (3.0), `nestedkv_tau` (0.60),
  `nestedkv_kappa` (10.0), `nestedkv_safeguard_alpha` (0.20) config fields;
  factory branch; unknown-method error string extended.
- 47 tests (30 quantizer + 17 cache,
  `veloxquant_mlx/tests/quantizers/test_nestedkv.py` +
  `veloxquant_mlx/tests/cache/test_nestedkv_cache.py`) and a deterministic
  offline benchmark (`benchmark_scripts/benchmark_nestedkv.py` +
  `nestedkv_benchmark_results.json`).
- Docs: `docs-site/docs/algorithms/nestedkv.md`, sidebar entry, overview
  table/bullet, changelog entry, cross-link from `curdkv.md`.

### Honest scope

- **No verified peer-reviewed venue** — see the venue exception above.
- **One-shot prefill compression; the cache is NOT bounded during decode.**
  The paper's own design (Appendix A) computes scores, blend weights, and
  surprise gates once at the end of prefill; decoded tokens are appended
  normally, never rescored or evicted. A faithful port of the paper's actual
  design, not a shortcut — but a real structural difference from every
  other eviction method in the repo (H2O, CurDKV, StreamingLLM all stay
  bounded through decode).
- **Gate/blend constants are the paper's own Appendix A defaults**
  (`beta=3.0`, `tau=0.60`, `kappa=10.0`, log-prior `(0.4,0.4,0.2)`,
  `safeguard_alpha=0.20`), not guessed — a stronger fidelity point than most
  adapted methods in this repo get to claim.
- **A structural finding from benchmark construction, not a bug**: at small
  synthetic scale, the head-adaptive blend's min-max normalization can make
  the stable scale's discriminative gap come out near-maximal almost by
  construction regardless of whether it's the actually-relevant signal for
  a given token, and the surprise gate's mean-centered threshold does not
  always fully compensate at that scale. The benchmark's
  `local_episodic_only` geometry shows 0% retention for both NestedKV and
  H2O — reported honestly rather than re-engineered until it matched the
  initial hypothesis. `global_outlier_only` and `recency_only` both show
  NestedKV at 100% retention vs H2O's 0%.
- The paper's own RULER/LongBench/LooGLE/InfiniteBench/MMLU-Pro numbers
  (Qwen3, Llama-3.2 family, NVIDIA L20 GPUs) are the paper's — not
  reproduced here.

## [0.36.0] — 2026-07-13

### Added — CurDKV-adapted value-aware leverage-score eviction (`method="curdkv"`)

The library's 38th method, joining the token-eviction family (H2O, SnapKV,
TOVA, PyramidKV, Keyformer, MorphKV, KVzip, and more). Every existing
eviction method scores a token using only its **key** side (attention-mass,
norm, key-SVD projection, reconstruction reliance) — none of them fold the
**value** vector's own contribution into the retention decision. CurDKV's
axis: build the proxy attention-weighted value block and derive a
**leverage score** from its dominant singular directions, energy-weighted
by singular value, so a token's value contribution — not just its key or
attention-mass profile — decides whether it survives. Inspired by
"Value-Guided KV Compression for LLMs via Approximated CUR Decomposition"
(Sengupta, Chaudhary, Chakraborty; **NeurIPS 2025**, confirmed poster,
arXiv:2509.15038) — shipped as "CurDKV-adapted (VeloxQuant-MLX
implementation)," **not a faithful port**.

- `veloxquant_mlx/quantizers/curdkv.py` — `CurDKVState`, `curdkv_update`,
  `curdkv_get_kv`, byte helpers, and the internal `_leverage_scores`
  estimator, modeled on `quantizers/h2o.py`'s per-head sliding state and
  eviction loop.
- `veloxquant_mlx/cache/curdkv_cache.py` — `CurDKVKVCache`, mirroring
  `H2OKVCache`'s structure line-for-line; prefill and decode go through the
  same eviction loop.
- `veloxquant_mlx/cache/base.py` — `"curdkv"` added to the method
  `Literal`; `curdkv_budget` (512), `curdkv_n_sink` (4), `curdkv_rank_cap`
  (16) config fields; factory branch; unknown-method error string extended.
- 39 tests (23 quantizer + 16 cache,
  `veloxquant_mlx/tests/quantizers/test_curdkv.py` +
  `veloxquant_mlx/tests/cache/test_curdkv_cache.py`) and a deterministic
  offline benchmark (`benchmark_scripts/benchmark_curdkv.py` +
  `curdkv_benchmark_results.json`).
- Docs: `docs-site/docs/algorithms/curdkv.md`, sidebar entry, overview
  table/bullet, changelog entry, cross-link from `h2o.md`.

### Honest scope

- **Key-as-query proxy, not the true query vector** — the same limitation
  H2O/SnapKV/Keyformer/MorphKV/KVzip already document; the cache wrapper
  never sees the model's real query, so the incoming key vector stands in
  for it.
- **An SVD-based, energy-weighted leverage-score estimator, not the
  paper's own CUR sampling algorithm.** Energy-weighting (`l_i = Σⱼ (sⱼ² /
  Σs²) · U[i,j]²`) rather than a hard top-k/bottom-(n−k) split is
  load-bearing: a hard rank cutoff degenerates to uniform leverage whenever
  the retained rank reaches the block size `n` — the left singular vectors
  of a full-rank `[n, k]` block with `k ≥ n` form a complete orthogonal
  basis, and every row of an orthogonal matrix has unit norm by
  construction, erasing the magnitude signal regardless of how small the
  tail singular values actually are.
- **Newly-appended tokens are seeded with their own leverage score, not a
  flat 0.** Unlike H2O's softmax weights (never exactly 0), CurDKV's
  leverage scores can legitimately be exactly 0 for a genuinely
  negligible-value token; a flat-0 seed would let such a token tie forever
  with an already-negligible survivor and let arrival order, not value,
  decide the outcome.
- **Mechanism observable:** on a planted geometry (near-identical keys,
  sharply divergent values), CurDKV retains value-relevant tokens
  preferentially in 8/8 trials across seeds; H2O, given the identical keys,
  cannot tell the classes apart and evicts near-uniformly. Two tokens with
  identical keys but different values provably receive different CurDKV
  scores (`test_identical_keys_different_values_diverge`).
- The benchmark also reports, honestly, that CurDKV retains fewer
  value-irrelevant tokens than H2O on a "correlated" control geometry too —
  not the initially expected null result. This is attributed to H2O's own
  tie-break dynamics in this small-N synthetic regime, not overclaimed as
  general CurDKV dominance; the always-true claim stays scoped to
  planted_value_divergence and the direct same-key/divergent-value test.
- The paper's headline numbers (up to 9.6% higher accuracy than SOTA
  baselines, up to 40% latency reduction under aggressive compression) are
  the paper's, on trained models — not reproduced here.

---

## [0.35.0] — 2026-07-12

### Added — KVTC-adapted local PCA + DP-optimal bit allocation + entropy coding (`method="kvtc"`)

The library's 37th method, joining the low-rank / spectral family (Palu,
SVDq, SpectralQuant). All three existing low-rank methods use a **fixed**
mixed-bit split (a hand-chosen top-25%/75% tier, or a binary signal/noise
cutoff via participation ratio) — none compute a **provably optimal**
allocation for a given total-bit budget, and none can assign **zero bits**
to an individual low-variance component while another component gets more
than the "high" tier. KVTC's axis: given a vector of per-component
variances from a local PCA and a total bit budget, use **dynamic
programming** to choose an integer bit-width per component (including
**0**, i.e. drop the component entirely) that minimizes total expected
distortion — then **entropy-code** the resulting codes for a further
lossless size reduction. Inspired by "KV Cache Transform Coding for
Compact Storage in LLM Inference" (NVIDIA, **ICLR 2026**, accepted poster,
arXiv:2511.01815) — shipped as "KVTC-adapted (VeloxQuant-MLX
implementation)," **not a faithful port**.

- `veloxquant_mlx/allocators/kvtc_dp.py` — `dp_allocate_bits`: DP over
  (component, cumulative budget), reusing `ratequant.py`'s analytic
  `D(v,b) = v·β^(-b)` distortion curve instead of inventing a new one.
- `veloxquant_mlx/quantizers/_entropy_coding.py` — `entropy_encode`/
  `entropy_decode`: dependency-free order-0 Huffman coder (stdlib `heapq`),
  lossless round-trip, code-table cost counted in byte accounting.
- `veloxquant_mlx/quantizers/kvtc.py` — `kvtc_compress`/`kvtc_decompress`:
  local per-sequence PCA (reusing `_quant_utils.py::_truncated_svd`, the
  same helper SVDq/Palu/GEAR share), no fixed-energy truncation (the DP
  allocator decides survivors), `kvtc_fp16_bytes` (realized entropy-coded
  payload) / `kvtc_pre_entropy_bytes` (pre-entropy-coding size).
- `veloxquant_mlx/cache/kvtc_cache.py` — `KVTCKVCache`, fits the PCA basis
  and DP allocation once at prefill and reuses them unchanged for every
  decode step; compresses **both K and V** (mirrors Palu's scope, not
  SVDq's keys-only scope).
- `veloxquant_mlx/cache/base.py` — `"kvtc"` method, `kvtc_bit_budget` (512),
  `kvtc_bit_choices` ((0,1,2,3,4,6,8)), `kvtc_beta` (3.5) config, factory
  branch.

**Honest scope:**
- Local (per-sequence) PCA, not the paper's pre-calibrated global basis —
  the same "fit-locally, no calibration set" limitation SVDq/Palu already
  document.
- The DP allocator optimizes an analytic distortion proxy, not a
  real-activation-fit rate-distortion model — the DP itself is exact; the
  objective it minimizes is the repo's existing Gaussian-quantization
  distortion curve, not one fit on real LLM activation statistics.
- Entropy coding is a real, measured, lossless order-0 Huffman coder — not
  the paper's (possibly more sophisticated) scheme, and never the
  theoretical Shannon-entropy bound.
- **Uniform-variance collapse (pinned):** with equal per-component variance
  and a contiguous `bit_choices` range, the DP-optimal allocation is exactly
  `floor(budget/n)` per component (remainder to the first components) — the
  same allocation a naive uniform splitter would produce. No other collapse
  is claimed — the DP should and does *beat* SVDq's fixed 25/75 split
  whenever variance is non-uniform.
- Mechanism observable = reconstruction MSE/cosine at a matched total byte
  budget: on planted skewed-variance geometry, KVTC's DP allocator reaches
  mean MSE ≈0.027 vs ≈87.6 (fixed-uniform) and ≈84.4 (SVDq-fixed-split); on
  a flat (isotropic) control it is roughly competitive with the
  fixed-uniform baseline, not a dramatic win. Entropy-coding's realized gain
  is modest (≈0.15–0.50 across the sweep), reported plainly.
- Not path-dependent (contrast with the eviction family
  H2O/TOVA/MorphKV/KVzip): the PCA basis and DP allocation are fixed once
  at prefill and reused for every subsequent token — pinned by a
  determinism test.
- The paper's numbers (up to 20×, up to 40× in some regimes, under 1pp
  accuracy loss on LLaMA 3/Mistral NeMo/R1-Qwen2.5 1.5B–70B across
  AIME25/GSM8K/LiveCodeBench/LongBench/MATH-500/MMLU/Qasper/RULER) are the
  paper's, on trained models — not reproduced.

73 new tests (32 allocator + 15 entropy coder + 12 quantizer + 14 cache) and
a deterministic offline benchmark (`benchmark_scripts/benchmark_kvtc.py`).

## [0.34.0] — 2026-07-11

### Added — KVzip-adapted context-reconstruction reliance retention (`method="kvzip"`)

The library's 37th method, joining the proxy-attention eviction family
(SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM, Keyformer,
MorphKV). It shares the H2O/TOVA/MorphKV scaffolding but introduces a **new
ranking axis**: every other proxy scorer ranks a stored token by the attention it
receives *from a query* (cumulative for H2O, latest for TOVA, windowed for
MorphKV); KVzip ranks by **reconstruction reliance** — how much the model relies
on a KV pair to *reconstruct its own context* — a **query-agnostic** importance
profile computed once and reused across all future queries. Inspired by "KVzip:
Query-Agnostic KV Cache Compression with Context Reconstruction" (Kim, Kim, Kwon,
Lee, Yun & Song, **NeurIPS 2025 Oral**, arXiv:2505.23416,
github.com/snu-mllab/KVzip) — shipped as "KVzip-adapted (VeloxQuant-MLX
implementation)," **not a faithful port**.

- `veloxquant_mlx/quantizers/kvzip.py` — `KVzipState`, `init_kvzip_state`,
  `kvzip_update` (reconstruction-reliance ranking + protected-sink eviction),
  `kvzip_get_kv`, byte helpers, `_reconstruction_importance` (max proxy-attention
  over the reconstruction probe).
- `veloxquant_mlx/cache/kvzip_cache.py` — `KVzipKVCache`, single-layer, no
  coordinator, no `.bits`, fp16, lazy per-head state, byte-accounting properties.
- `veloxquant_mlx/cache/base.py` — `"kvzip"` method, `kvzip_budget` (512) /
  `kvzip_n_sink` (4) / `kvzip_probe` ("context") config, factory branch.

**Honest scope:**
- `kvzip_probe="latest"` collapses onto TOVA-adapted **bit-for-bit** (pinned by a
  test); **no H2O collapse is claimed** — KVzip recomputes reconstruction reliance
  from the live keep set each step, it never accumulates.
- Key-as-reconstruction-probe proxy (a cache never runs the model to reconstruct
  text), same substitution family as H2O/TOVA/MorphKV-adapted.
- Mechanism observable = reconstruction-critical retention under a reconstruction
  shift: cumulative H2O retains ~0.017 of the reconstruction-critical region while
  the context probe retains ~0.609, beating the `probe="latest"` (TOVA) reference
  (~0.248); a flat control shows no advantage. Downstream perturbation reported
  as-is.
- The paper's numbers (3–4× reduction, ~2× decode, negligible loss up to 170K on
  LLaMA3.1/Qwen2.5/Gemma3) are the paper's, on trained models — not reproduced.

32 new tests (19 quantizer + 13 cache) and a deterministic offline benchmark
(`benchmark_scripts/benchmark_kvzip.py`).

### Changed — meta
- Replaced the dead Buy Me a Coffee handle with working **Ko-fi** and **Buy Me a
  Chai** links across the README, landing page, and `.github/FUNDING.yml`.
- Refreshed the JOSS paper (`paper/joss/paper.md`) to the current 37-method suite
  and the token-eviction family.

## [0.33.0] — 2026-07-10

### Added — MorphKV-adapted recent-window correlation retention (`method="morphkv"`)

The library's 36th method, joining the proxy-attention eviction family
(SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM, Keyformer). It
shares the H2O/TOVA scaffolding but introduces a **new ranking axis**: instead
of cumulative attention (H2O — inertial, early-token bias) or the single latest
query (TOVA — memoryless), it keeps a **constant-size** cache by ranking stored
tokens against the attention pattern of a **sliding window of recent tokens**,
so retention re-targets toward what the recent context actually reads. Inspired
by "Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in
LLMs" (Ghadia et al., **ICML 2025**, arXiv:2503.00979) — documented as
"MorphKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.

- **`veloxquant_mlx/quantizers/morphkv.py`** — `MorphKVState`,
  `init_morphkv_state` (validates budget/window/sink bounds), `morphkv_update`
  (recent-window relevance ranking + protected sinks/recent-window eviction),
  `morphkv_get_kv`, `morphkv_fp16_bytes`, `full_morphkv_fp16_bytes`, and
  `_recent_relevance` (mean key-as-query proxy-attention over the recent window).
- **`veloxquant_mlx/cache/morphkv_cache.py`** — `MorphKVKVCache`, a
  single-layer, no-coordinator, no-`.bits`, fp16 cache with byte-accounting
  properties (`morphkv_kept_bytes`, `full_seq_bytes`, `compression_ratio`,
  `tokens_seen`, `tokens_kept`).
- **Config** (`cache/base.py`): `morphkv_budget` (512), `morphkv_n_sink` (4),
  `morphkv_window` (8; **1 = TOVA-adapted**).
- **32 tests** (19 quantizer + 13 cache) and a deterministic offline benchmark
  (`benchmark_scripts/benchmark_morphkv.py` + committed results JSON).

### Honest scope

- **`morphkv_window=1` collapses onto TOVA-adapted, bit-for-bit** — the single
  recent key's attention over the keep set is exactly TOVA's latest-token
  ranking; a test asserts the kept set equals TOVA's. **No H2O collapse is
  claimed** — MorphKV recomputes from the live window each step and never
  becomes H2O's cumulative-forever rule.
- **Constant-size, recomputed — not accumulated.** No cumulative score array is
  stored; retention is recomputed each step from the live keep set and the last
  `morphkv_window` keys.
- **Key-as-query proxy** (same as H2O/TOVA/Keyformer-adapted): the incoming key
  stands in for the unseen query.
- **Mechanism evidence is the recent-relevant retention rate.** Under a
  constructed topic shift, cumulative H2O scoring retains ~0% of the
  recent-relevant region (captured by stale early heavy hitters) while MorphKV
  re-targets toward it; the recent signal is made deliberately weak/noisy so a
  wider window materially beats the `window=1` (TOVA) reference. A "stable"
  control shows no advantage. Downstream probe perturbation is a noisier
  secondary effect, reported as-is. The paper's accuracy/memory numbers are the
  paper's, on trained models — not reproduced. No RoPE remapping; uniform
  budget/window across heads; offline-synthetic only (no model-level
  perplexity/throughput benchmark).

---

## [0.32.0] — 2026-07-10

### Added — Keyformer-adapted Gumbel-regularized heavy-hitter eviction (`method="keyformer"`)

The library's 35th method, joining the proxy-attention eviction family
(SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM). Structurally it
**is** H2O-adapted — additive key-as-query proxy-attention accumulation with a
protected-sink top-budget eviction — plus **one** new ingredient: **Gumbel
noise** on the eviction logits. The noise stops a "late riser" (a token that
reads low early, before the queries that attend to it arrive) from being
greedily pruned before it can recover. Inspired by "Keyformer: KV Cache
Reduction through Key Tokens Selection for Efficient Generative Inference"
(Adnan et al., **MLSys 2024**, arXiv:2403.09054) — documented as
"Keyformer-adapted (VeloxQuant-MLX implementation)," not a faithful port.

- **`veloxquant_mlx/quantizers/keyformer.py`** — `KeyformerState`,
  `init_keyformer_state` (validates `tau >= 0` and evictable room),
  `keyformer_update` (proxy-attention accumulation + `score + tau·gumbel`
  eviction ranking), `keyformer_get_kv`, `keyformer_fp16_bytes`,
  `full_keyformer_fp16_bytes`, and `_gumbel_at` (a deterministic Gumbel(0,1)
  draw keyed by `(seed, position)` via inverse-CDF).
- **`veloxquant_mlx/cache/keyformer_cache.py`** — `KeyformerKVCache`, a
  single-layer wrapper (no coordinator) modeled on `H2OKVCache`; per-head
  states seeded with a per-head offset so heads' frozen noise is independent.
- **`cache/base.py`** — `method="keyformer"`, config fields, factory branch.
- Config: `keyformer_budget` (512), `keyformer_n_sink` (4), `keyformer_recent`
  (0, extension), `keyformer_tau` (1.0; **0 = H2O-adapted**), `keyformer_seed`
  (0).
- **Tests (29):** `tests/quantizers/test_keyformer.py` (17) and
  `tests/cache/test_keyformer_cache.py` (12) — incl. the `tau=0`==H2O collapse,
  `tau=0` seed-invariance, Gumbel determinism, and the late-riser survival
  mechanism.
- **Benchmark:** `benchmark_scripts/benchmark_keyformer.py` +
  `keyformer_benchmark_results.json` (deterministic in all non-timing fields).

### Honest scope

- **`keyformer_tau=0` collapses onto H2O-adapted, bit-for-bit.** The only thing
  Keyformer adds over H2O is the Gumbel regularizer; a test asserts the `tau=0`
  kept set equals H2O's over an identical stream, and the benchmark prints an
  `h2o` cross-check column.
- **Frozen per-position Gumbel, not the paper's annealed schedule.** The paper
  redraws Gumbel noise and anneals a temperature across generation; a cache has
  no trustworthy global step, so we draw one deterministic Gumbel value per
  token position (seeded by `keyformer_seed` + a per-head running position) and
  freeze it. Preserves the "don't doom a borderline token on one low reading"
  intent; **not** claimed equivalent to the schedule.
- **Key-as-query proxy** (same as H2O/SnapKV-adapted): the incoming key stands
  in for the unseen query, not the model's real attention logits.
- **Mechanism evidence is the survival rate.** Under constructed late-riser
  geometry, greedy `tau=0` evicts the planted riser 100% of the time while
  `tau=6` rescues it ~75% of the time. The downstream probe-attention
  perturbation is a noisier, regime-dependent secondary effect, reported as-is
  rather than cherry-picked. No RoPE remapping after eviction. Uniform
  budget/tau across heads. No model-level perplexity/throughput benchmark —
  offline-synthetic survival-rate, output-perturbation and byte-accounting only.

---

## [0.31.0] — 2026-07-09

### Added — Q-Filters-adapted query-agnostic projection eviction (`method="qfilters"`)

The library's 34th method and its **fourth eviction scorer class** — after
attention/proxy (SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV,
CaM), structural (StreamingLLM, sink), and intrinsic-norm (L2Norm). Each
cached key is scored by its projection onto a single frozen per-head
direction (the *Q-Filter*); over budget, the highest-scoring tokens are kept,
with sinks and an optional recent window protected. Inspired by "Q-Filters:
Leveraging QK Geometry for Efficient KV Cache Compression" (arXiv:2503.02812,
**preprint**) — documented as "Q-Filters-adapted (VeloxQuant-MLX
implementation)," not a faithful port.

- `veloxquant_mlx/quantizers/qfilters.py` — `QFiltersState`,
  `estimate_filter_dir` (top singular vector of the observed keys, frozen
  after `qfilters_calib_tokens`), `qfilters_update`, `qfilters_get_kv`, byte
  helpers (K+V fp16 plus the float32 filter direction).
- `veloxquant_mlx/cache/qfilters_cache.py` — `QFiltersKVCache`, single-layer,
  no coordinator, modeled on `L2NormKVCache`.
- `veloxquant_mlx/cache/base.py` — `method="qfilters"` config fields
  (`qfilters_budget` 512, `qfilters_n_sink` 4, `qfilters_recent` 0,
  `qfilters_calib_tokens` 128, `qfilters_sign` 1), factory dispatch.
- 27 tests (12 quantizer + 15 cache), all passing.
- `benchmark_scripts/benchmark_qfilters.py` + committed
  `qfilters_benchmark_results.json` (deterministic; sign±1, best-of-sign,
  KNorm/H2O/random arms, two geometries, `filter_cosine` field).
- Docs: `docs-site/docs/algorithms/qfilters.md`, sidebar/overview entries,
  cross-links from `knorm.md` and `h2o.md`.

**Honest scope.**

- **The filter is key-SVD-derived, not query-SVD-derived.** The paper
  estimates the direction offline from a sample of *query* vectors; a
  cache-side library never sees queries, so we substitute the SVD of the
  first observed *keys*. This recovers the dominant *axis* but not which
  *end* is important — the sign a query would disambiguate. The committed
  benchmark shows the key-SVD recovering the planted axis
  (`filter_cosine ≈ 0.97`) while which raw sign arm wins flips row to row, so
  `qfilters_sign` is a **genuine ablation**. Nothing here is claimed
  equivalent to the paper's query-derived filter.
- **Path-dependent** (unlike L2Norm): prefill-in-one-block and
  token-by-token decode can freeze different filters and diverge — no
  prefill/decode bit-for-bit equivalence guarantee (deliberately not tested).
- Preprint, no venue. No RoPE remapping after eviction. Uniform budget across
  heads. `qfilters_recent` is an extension, off by default. No model-level
  perplexity/throughput benchmark — offline-synthetic output-perturbation and
  byte-accounting only.

---

## [0.30.1] — 2026-07-08

### Fixed — PyPI package metadata (no code changes)

Metadata-only patch release. PyPI mirrors such as pepy.tech showed no
summary/version/license/author for the package because the published
metadata was malformed in ways downstream consumers reject:

- **Summary** was a ~700-character 33-method list — replaced with a proper
  one-line summary (the full method list lives in the README, which is the
  PyPI long description).
- **License** field contained the entire MIT license text
  (`license = { file = "LICENSE" }` embeds the file verbatim) — now a PEP 639
  SPDX expression (`License-Expression: MIT`, `License-File: LICENSE`);
  the deprecated `License ::` classifier was dropped per PEP 639.
- **Author** was empty (name+email pairs emit only `Author-email:`) — now
  also emits `Author: Rajveer Rathod`.

Wheel/sdist contents are otherwise identical to 0.30.0.

## [0.30.0] — 2026-07-08

### Added — SKVQ: sliding-window reorder + clip quantization (`method="skvq"`)

Sliding-window quantization with two mechanisms new to the library, inspired
by "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language
Models" (Duanmu, Yuan, Li, Duan, Zhang, Lin — **COLM 2024**,
arXiv:2405.06219). Documented as **"SKVQ-adapted (VeloxQuant-MLX
implementation)"** — not a faithful port.

- **Channel reordering** — per-head permutations sort head-dim channels by
  dynamic range so channels of similar range share a quantization group
  (one wide channel no longer stretches the scale for its whole group).
  Computed from the **first flushed chunk** of live traffic, then frozen.
- **Clipped dynamic quantization** — per-token, per-group asymmetric
  min/max quantization whose window is shrunk by a clip factor α found by
  **per-group grid search** against reconstruction MSE; α=1 (plain min/max)
  is always in the grid, so the search never loses under its own metric.
  The chosen α is folded into the stored (lo, scale) — nothing extra kept.
- **Sliding fp16 window + sink filter** — the NSNQuant chunk-flush idiom:
  tokens aging past `skvq_window` are quantized once and frozen; the first
  `skvq_n_sink` tokens stay fp16 (the paper's attention-sink filter). Both
  K and V quantized, per-token channel groups (reordering is what makes
  that viable for keys).
- **Path independence, pinned by test:** prefill-in-one-block and
  token-by-token decode produce **bit-for-bit identical caches**. Fully
  deterministic — no RNG anywhere.
- Code: `veloxquant_mlx/quantizers/skvq.py` (`channel_permutation`,
  `invert_permutation`, `apply_permutation`, `clipped_group_quant`,
  `clipped_group_dequant`, `skvq_round_trip`, byte helpers),
  `veloxquant_mlx/cache/skvq_cache.py` (`SKVQKVCache`), config fields
  `skvq_bits_key`/`skvq_bits_value` (2/2), `skvq_group_size` (32),
  `skvq_window` (128), `skvq_n_sink` (5), `skvq_reorder`,
  `skvq_clip_search`/`skvq_clip_alpha`, `skvq_max_ctx`. No coordinator —
  single-layer factory branch.
- Tests: 13 quantizer + 18 cache (31 new), incl. α=1 ≡ plain min/max
  against a numpy reference, never-worse clip search, frozen permutations,
  sink-row fp16 exactness, closed-form byte accounting, `for_model` wiring.
- Benchmark (`benchmark_scripts/benchmark_skvq.py`, committed
  `skvq_benchmark_results.json`, offline-synthetic): under a
  heterogeneous-channel regime, reordering cuts key MSE a further **16.9%**
  on top of clip search and collapses per-channel normalized error ~450×;
  clip search adds **14.0%** on top of reordering; under the homogeneous
  control reordering buys **−0.3%** (nothing) — both regimes reported. The
  repo's KIVI reference wins several heterogeneous rows outright (its
  per-channel key scheme is intrinsically immune to channel heterogeneity)
  — reported as measured.

### Honest scope
- The paper's offline calibration (KMeans channel clustering on WikiText-2,
  attention-output-MSE clip search, permutation fused into projection
  weights) is replaced by first-chunk statistics with an explicit runtime
  permute/inverse-permute — a documented adaptation, not the paper's
  pipeline.
- No 1.5-bit value packing, no FP8(E4M3) metadata (CUDA packing artifacts);
  integer bit-widths and fp16 metadata, all counted in byte accounting.
- That real transformer K/V exhibit the heterogeneous-channel regime is the
  paper's premise (shared with KIVI/KVQuant) — the offline-synthetic
  benchmark cannot validate it, and the homogeneous control shows
  reordering buys nothing without it.
- No model-level (perplexity/throughput) benchmark run.

## [0.29.0] — 2026-07-07

### Added — L2Norm: intrinsic key-norm eviction (`method="knorm"`)

- **`veloxquant_mlx.cache.knorm_cache.L2NormKVCache`** — the library's
  **thirty-second configuration** and its first **intrinsic-signal** eviction
  cache. *Inspired by, not a faithful port of,* "A Simple and Effective L2
  Norm-Based Strategy for KV Cache Compression" (Devoto, Zhao, Scardapane &
  Minervini, EMNLP 2024, arXiv:2406.11430). Every eviction method shipped so
  far scores tokens with attention / a key-as-query proxy (SnapKV, H2O,
  TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM) or pure structure
  (StreamingLLM, sink, sliding-window); L2Norm reads importance **directly
  off the stored key** — the paper's trained-LM finding is that *low* key
  norm predicts *high* attention, so the lowest-norm tokens are kept. Note
  the sign inversion vs ChunkKV's `key_norm` option and ZipCache's saliency
  proxy (which treat high norm as important) — the inversion is the paper's
  empirical content.
- **`veloxquant_mlx.quantizers.knorm`** — `KnormState`, `init_knorm_state`,
  `knorm_update` (vectorized: norms are computed once at insertion and never
  updated, so eviction is a single protected top-k per incoming block — no
  per-token softmax-over-cache loop like H2O), `knorm_get_kv`,
  `knorm_fp16_bytes`, `full_knorm_fp16_bytes`.
- **Two properties fall out of the intrinsic score** (both measured/pinned):
  - **Speed:** 0.3–1.2 ms per prefill block vs H2O-adapted's 37–275 ms on
    identical inputs in the committed harness (~100–800×).
  - **Path independence** (`knorm_recent=0`): the kept set equals the global
    budget-best over all tokens seen regardless of arrival grouping (the
    "keep k best with a heap" invariant) — prefill and token-by-token decode
    produce bit-for-bit identical caches, pinned by test at the primitive
    and wrapper level. No accumulating-score method has this property.
- **Config** — `knorm_budget` (default 512), `knorm_n_sink` (default 4),
  `knorm_recent` (default 0; extension, breaks path independence when on),
  `knorm_keep` (`"low"` paper default | `"high"` inverted ablation).
  Build-time validation (keep mode, sinks+recent < budget). No coordinator.
- **Tests** — 10 quantizer + 14 cache tests (903 total collected).
- **Benchmark** — `benchmark_scripts/benchmark_knorm.py` + committed
  `knorm_benchmark_results.json` (offline-synthetic): under geometry
  constructed to exhibit the paper's correlation, keep-low beats random
  eviction by **+0.17** mean output perturbation and the inverted scorer by
  **+0.21**, and beats H2O-adapted on most rows at matched budget; under
  the isotropic control the advantage **reverses** (keep-low ~0.07 worse
  than random), reported in full. **Explicitly NOT a model-level
  perplexity/throughput benchmark.**

### Honest scope

- The low-norm ⇒ high-attention correlation is the **paper's empirical
  claim about trained models**. Synthetic data cannot validate it — the
  benchmark validates the machinery under constructed geometry and shows
  the method underperforming random eviction when that geometry is absent.
- No RoPE position-ID remapping after eviction; uniform budget/n_sink
  across heads (same as the rest of the eviction family).
- `knorm_recent` and `knorm_keep="high"` are extensions beyond the paper,
  both off by default.
- No model-level benchmark run.

## [0.28.0] — 2026-07-06

### Added — NSNQuant: calibration-free universal-codebook VQ (`method="nsnquant"`)

- **`veloxquant_mlx.cache.nsnquant_cache.NSNQuantKVCache`** — the library's
  **thirty-first configuration** and its first **calibration-free
  distribution-matching VQ**. *Inspired by, not a faithful port of,*
  "NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit
  Vector Quantization of KV Cache" (Son, Choi & Yoo, NeurIPS 2025,
  arXiv:2505.18231). Every other VQ method in the repo either fits its
  codebook to the data (RVQ's per-sequence k-means, CommVQ) or uses a
  data-independent geometric code (RaBitQ signs, VecInfer binary, PolarQuant
  grids, QJL sketches); NSNQuant inverts the relationship — a
  **Normalize-Shift-Normalize transform + Hadamard rotation reshapes K/V
  tokens onto the standard normal distribution**, so one codebook built
  offline from synthetic Gaussian samples (never model activations)
  quantizes any model at 1–2 bits/element.
- **`veloxquant_mlx.quantizers.nsnquant`** — pure primitives: `nsn_transform`
  / `nsn_inverse` (token-norm → channel-mean shift → token-norm, exact
  restoration `x̂ = s1·(s2·x_nsn + o)`), `build_universal_codebook`
  (deterministic seeded spherical k-means on synthetic standard-normal
  samples; "magnitude" positive-orthant variant for 2-bit + sign mask,
  "signed" variant for 1-bit), `vq_encode`/`vq_decode` (8-dim subvector
  cosine matching, uint8 indices), `hadamard_forward`/`hadamard_inverse`
  (reusing `mx.hadamard_transform` — norm-preserving, so it composes with
  NSN's stored scales).
- **Chunk-flush residual buffer** — KIVI's fp16-residual idiom, upgraded:
  every `nsn_residual_length` tokens age past the quantized frontier as one
  self-contained chunk with its own online channel mean (no frozen
  statistics, no coordinator, chunk *i* forever independent of later
  arrivals). Prefill and decode produce identical quantized state by
  construction — verified bit-for-bit by test. Unlike KIVI's
  incoming-block-only simplification, decode tokens *do* get quantized once
  they age out.
- **Config** — `nsn_bits` (default 2: uint8 sign mask + uint8 codebook index
  per 8-dim subvector = 2 bits/element; 1: index only), `nsn_residual_length`
  (default 64; the paper recommends 128 for 1-bit), `nsn_codebook_size`
  (default 256), `nsn_subvector_dim` (default 8), `nsn_seed` (default 1234),
  `nsn_max_ctx` (default 8192). Both keys **and** values quantized (mirrors
  the paper; unlike the keys-only SVDq/xKV precedent). Build-time validation
  with clear messages (bits ∈ {1,2}, head_dim divisibility, Hadamard
  compatibility).
- **Byte accounting** — payload at exactly `nsn_bits` bits/element plus fp16
  metadata counted honestly (`s1`+`s2` per token, `o` per chunk ≈ 0.5
  bits/element at defaults — the paper double-quantizes these to ~0.23; we
  don't, and say so); `residual_fp16_bytes` reported separately as a
  snapshot so ratios aren't inflated. ~2.5 effective bits/element at 2-bit
  defaults → ~6.4× vs fp16.
- **Tests** — 16 quantizer + 19 cache tests (871 total passing), including a
  mechanism-validation ablation (on channel-biased input the full NSN
  pipeline must beat the identical Hadamard+VQ without NSN by a pinned
  margin) and the prefill-vs-decode path-independence check.
- **Benchmark** — `benchmark_scripts/benchmark_nsn.py` + committed
  `nsn_benchmark_results.json` (offline-synthetic, no model download):
  NSN gains **+0.038 (2-bit) / +0.110 (1-bit)** reconstruction cosine over
  the no-NSN ablation at strong synthetic channel bias, and the gain
  honestly **collapses to ~+0.001–0.002 when the input is already centered**
  (NSN only helps when there is a bias to remove); 0.96–0.98 cosine at ~2.5
  effective bits/element, above a KIVI-2bit baseline (0.66–0.88) on every
  row of the sweep. **Explicitly NOT a model-level perplexity/throughput
  benchmark.**

### Honest scope

- **Post-RoPE keys** — the paper applies NSN to keys *before* RoPE and defers
  RoPE onto the stored mean inside a custom attention kernel; our cache
  wrappers receive post-RoPE keys from `update_and_fetch`, so NSN + Hadamard
  run post-RoPE. This is the central simplification of the adaptation.
- **No value-projection Hadamard fusion** (model surgery) — the value-side
  Hadamard is applied explicitly to cached tensors instead.
- **No gradient fine-tuning of the codebook** — seeded spherical k-means
  only; expect a slightly worse codebook than the paper's.
- **No 4-bit double quantization of metadata** — fp16, counted.
- **No fused kernels** — MLX ops; on Apple Silicon the win is memory, not
  throughput, exactly as with KIVI.
- **No model-level benchmark run** — offline-synthetic reconstruction and
  byte-accounting numbers only.

## [0.27.0] — 2026-07-06

### Added — xKV: cross-layer shared-subspace key compression (`method="xkv"`)

- **`veloxquant_mlx.cache.xkv_cache.XKVCache`** — the library's **thirtieth
  configuration** and the **third cross-layer** mechanism, alongside XQuant
  (code reuse) and MiniCache (SLERP direction merge). *Inspired by, not a
  faithful port of,* "xKV: Cross-Layer KV-Cache Compression via Aligned
  Singular Vector Extraction" (Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang,
  Li, Ceze, Wu & Abdelfattah, arXiv:2503.18893, preprint). Every other
  cross-layer method either reuses one anchor's codes (XQuant) or merges a
  *pair* of layers' directions (MiniCache); xKV instead **jointly factorizes
  a whole group of layers** into one shared low-rank SVD basis, amortizing
  the basis storage cost across every member of the group.
- **`veloxquant_mlx.cache.xkv_coordinator.XKVCoordinator`** — a
  fan-in-then-fan-out coordinator: every group member publishes its own raw
  prefill keys; once all members of a group have published for the same
  token range, the joint SVD runs once and the resulting shared basis is
  broadcast back to every member (including whichever one triggered the
  computation). This is a different coordination shape than XQuant/MiniCache,
  which have a single publisher and one or more readers.
- **`veloxquant_mlx.quantizers.xkv`** — pure primitives: `pair_layers_grouped`
  (fixed-size contiguous grouping, including a trailing partial group),
  `joint_svd_compress` (stack N layers' centered keys, single truncated SVD),
  `project_into_shared_basis`, `reconstruct_from_shared_basis`,
  `quantize_latents_uniform`.
- **Grouping** — `xkv_group_size` (default 2) chunks attention-bearing layers
  into fixed contiguous groups; layer 0 of each group is the conventional
  "leader" (the only member reporting the amortized `shared_basis_bytes`
  cost, avoiding double-counting when bytes are summed across layers).
- **Config** — `xkv_group_size` (default 2), `xkv_rank` (default `None` →
  energy-threshold selection), `xkv_energy_threshold` (default 0.95),
  `xkv_latent_bits` (default 4 — single-bit-width latent quantization, not
  SVDq-style mixed-bit routing), `xkv_group_quant_size` (default 32),
  `xkv_max_ctx` (default 8192). Keys only — values pass through fp16
  unchanged, mirroring SVDq's existing precedent in this repo.
- **Tests** — 9 quantizer tests + 14 cache tests (all passing), including a
  group-of-1 degeneracy check (`joint_svd_compress` on a single matrix
  matches SVDq's plain single-layer SVD at the same rank) and a
  mechanism-validation test (a shared basis fit jointly across synthetic
  layers with genuinely shared low-rank structure reconstructs better than
  independent per-layer SVD on unrelated noise at matched rank).
- **Benchmark** — `benchmark_scripts/benchmark_xkv.py` + committed
  `xkv_benchmark_results.json` (offline-synthetic). Sweeps group size (2–4)
  and a synthetic shared-structure fraction against an independent-per-layer
  -SVD baseline at matched rank: reconstruction MSE lands within ~1% of
  independent SVD across every configuration tested (near-parity, not a
  quality regression), while byte cost is **8–20% lower** than independent
  SVD, improving with larger group sizes — the amortization win the
  shared-basis mechanism is designed to deliver.

### Honest scope

- Fixed contiguous layer grouping — no CKA-based (Centered Kernel Alignment)
  validation that the grouped layers actually share a subspace, unlike the
  paper's empirical per-architecture grouping.
- No "Selective Reconstruction" — the paper's decode-time latency
  optimization (exactly reconstruct a subset of group layers, derive the
  rest) is not implemented; every layer is fully reconstructed on every
  fetch, like every other wrapper in this repo.
- Single-bit-width latent quantization, not SVDq's importance-ranked
  mixed-bit routing — xKV's distinguishing feature is the shared basis, not a
  novel bit-allocation scheme.
- **No model-level (perplexity/throughput) benchmark run.** The harness
  measures reconstruction-MSE parity and byte-accounting savings against an
  independent-SVD baseline, and an output-perturbation proxy — not end-to-end
  task quality on a real model.
- Docs: new `docs-site/docs/algorithms/xkv.md`, sidebar + overview + intro +
  changelog entries, cross-links from XQuant and MiniCache pages. README/
  landing counts: twenty-nine → thirty strategies; version bump 0.26.0 →
  0.27.0.

## [0.26.0] — 2026-07-04

### Added — CaM: cache merging (merge evicted tokens instead of dropping) (`method="cam"`)

- **`veloxquant_mlx.cache.cam_cache.CaMKVCache`** — the library's **eighth
  eviction configuration** and the first on the **merge-vs-drop** axis. *Inspired
  by, not a faithful port of,* "CaM: Cache Merging for Memory-efficient LLMs
  Inference" (Zhang, Du, Luo, Zhong, Zhang, Liu & Ji, ICML 2024, PMLR
  235:58840-58850). Every other eviction method permanently discards the tokens it
  evicts; CaM instead **merges** each evicted token into the surviving token it
  most resembles (a cosine-weighted blend of the value rows, and optionally the
  keys), then removes only the redundant slot. The eviction *choice* is H2O's;
  only the disposition differs. With `cam_merge="drop"` it reduces **bit-for-bit**
  to H2O-adapted.
- **`veloxquant_mlx.quantizers.cam`** — pure primitives: `most_similar_survivor`
  (nearest retained non-sink key by cosine), `merge_pair` (the weighted blend),
  `CaMState` + `init_cam_state` / `cam_update` / `cam_get_kv` / `cam_fp16_bytes` /
  `full_cam_fp16_bytes`.
- **Merge modes** — `cam_merge="sim_weighted"` (default) blends by
  `w = clip(cos(k_evicted, k_survivor), 0, 1)`; `"mean"` is an unweighted average;
  `"drop"` skips the blend (== H2O). Values are always merged; keys only when
  `cam_merge_keys=True`.
- **Config** — `cam_budget` (default 512), `cam_n_sink` (default 4), `cam_merge`
  (default `"sim_weighted"`), `cam_merge_keys` (default False). No coordinator;
  the default `KVCacheBuilder.for_model()` path returns one `CaMKVCache` per layer.
- **Tests** — 18 quantizer tests + 14 cache tests (all passing), including a
  bit-for-bit `cam_merge="drop"` == H2O equivalence (identical kept keys *and*
  values vs `H2OKVCache`) at both the primitive and cache level.
- **Benchmark** — `benchmark_scripts/benchmark_cam.py` + committed
  `cam_benchmark_results.json` (offline-synthetic, Apple Silicon). Measures output
  **perturbation** (cosine distance of the compressed-cache attention output vs the
  full cache over probe queries) against the H2O `drop` baseline; `sim_weighted`
  merging reduces perturbation and the gain grows with compression ratio
  (0.955 → 0.708 at `seq=1024, budget=64`, 16×), shrinking to ~0 at low compression.

### Honest scope

- Cosine-similarity merge weight rather than the paper's attention-prominence
  weight (which is ~0 for a just-appended token that overflows before it
  accumulates mass — the common streaming case); single nearest-survivor merge (no
  multi-target soft assignment / sampling); key-as-query proxy; no RoPE remapping;
  uniform budget across heads.
- **No model-level (perplexity/throughput) benchmark run.** The harness measures
  the output-perturbation proxy CaM targets, not end-to-end task quality.
- Docs: new `docs-site/docs/algorithms/cam.md`, sidebar + overview + intro +
  changelog entries, cross-links from H2O and ChunkKV. README/landing counts:
  twenty-eight → twenty-nine strategies; version bump 0.25.0 → 0.26.0.

## [0.25.0] — 2026-07-04

### Added — ChunkKV: chunk-level (semantic-block) eviction (`method="chunkkv"`)

- **`veloxquant_mlx.cache.chunkkv_cache.ChunkKVCache`** — the library's **seventh
  eviction configuration** and the first to evict at **chunk** rather than **token**
  granularity. *Inspired by, not a faithful port of,* "ChunkKV: Semantic-Preserving
  KV Cache Compression for Efficient Long-Context LLM Inference" (Liu et al., 2025,
  arXiv:2502.00299). Every other eviction method scores and drops individual tokens;
  ChunkKV partitions the sequence into contiguous chunks of `chunk_size` tokens and
  keeps or drops each chunk *as a whole*, preserving local coherence that token-level
  eviction shreds. When `chunk_size=1` it reduces **bit-for-bit** to H2O-adapted.
- **`veloxquant_mlx.quantizers.chunkkv`** — pure primitives: `chunk_partition`
  (split into sink + body chunks), `chunk_scores` (mean-pool a per-token score into
  per-chunk scores), `chunkkv_keep_mask` (chunk-aligned keep-mask for a budget),
  `ChunkKVState` + `init_chunkkv_state` / `chunkkv_update` / `chunkkv_trim_to` /
  `chunkkv_get_kv` / `chunkkv_fp16_bytes` / `full_chunkkv_fp16_bytes`.
- **Chunk-importance proxy** — `chunkkv_score="attn_mass"` (default) mean-pools H2O's
  cumulative attention mass; `chunkkv_score="key_norm"` mean-pools the key L2 norm
  (calibration-free, coarser). Sinks (`chunkkv_n_sink`) are always kept and never
  grouped into an evictable chunk.
- **Config** — `chunkkv_budget` (default 512), `chunkkv_chunk_size` (default 8),
  `chunkkv_n_sink` (default 4), `chunkkv_score` (`"attn_mass"` | `"key_norm"`).
  No coordinator: each layer resolves its own chunks, so the default
  `KVCacheBuilder.for_model()` path returns one `ChunkKVCache` per layer. Whole-chunk
  retention lets heads settle at slightly different counts, so the wrapper trims every
  head to the common minimum (`chunkkv_trim_to`) to emit a rectangular tensor.
- **Tests** — 19 quantizer tests + 14 cache tests (all passing), including a
  bit-for-bit `chunk_size=1` == H2O equivalence (identical kept keys *and* values vs
  `H2OKVCache`) at both the primitive and cache level. Survivors verified to be whole
  chunks; sinks always preserved; both score modes exercised; deterministic.
- **Benchmark** — `benchmark_scripts/benchmark_chunkkv.py` + committed
  `chunkkv_benchmark_results.json` (offline-synthetic, Apple Silicon). Confirms
  `chunk_size=1` reproduces H2O and that larger chunks cut the pure-Python eviction
  pass sharply (~12.7× fewer/faster passes at `C=16` vs `C=1` on the
  `seq=1024, budget=128` shape) while holding compression.

### Honest scope

- Mean-pooled per-token score as a proxy for the paper's attention-over-chunk
  importance; no layer-wise kept-index reuse (each layer resolves chunks independently).
- Key-as-query proxy for the `attn_mass` scorer (same as H2O-adapted); no RoPE
  position-ID remapping after eviction; uniform budget across heads within a layer.
- **No model-level (perplexity/throughput) benchmark run.** The harness measures
  compression, kept-token count, and eviction latency on synthetic data. ChunkKV's
  semantic-coherence advantage is a real-attention property and is not claimed from
  the synthetic harness.
- Docs: new `docs-site/docs/algorithms/chunkkv.md`, sidebar + overview + changelog
  entries, cross-links from SnapKV and SqueezeAttention. README intro now reads
  "twenty-eight compression strategies" (seven token-eviction caches). Landing page
  updated with a ChunkKV card, picker entry, quickstart tab, and what's-new item.

## [0.24.1] — 2026-07-04

### Changed — documentation & landing page

- **README** — dynamic shields.io PyPI version badge (auto-reads the live release),
  new pepy.tech total-downloads badge, tests updated to 750/756, changelog badge to
  0.24.1, and the intro now reads "twenty-seven compression strategies" (six of them
  token-eviction caches). No code or API changes.
- **Landing page** — "Method Library" redesign: uniform card grid grouped by category
  (Eviction / Quantization / Low-rank / Cross-layer), quiet version metadata, a single
  NEW pill on the three latest methods, and progressive-disclosure `<details>`
  expanders. De-duplicated the install/quickstart sections and added a SqueezeAttention
  quickstart tab. Fixed an invisible footer tagline and stale test/version counts.

## [0.24.0] — 2026-07-03

### Added — SqueezeAttention: 2D layer×token data-driven budget eviction (`method="squeeze"`)

- **`veloxquant_mlx.cache.squeeze_cache.SqueezeAttentionCache`** — the library's
  first **2D (layer × token)** budget eviction method and the first with a
  **data-driven** per-layer budget. *Inspired by, not a faithful port of,*
  "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise
  Optimal Budget" (Wang et al., 2024, arXiv:2404.04793). SqueezeAttention is
  H2O's cumulative-attention-mass eviction with a per-layer budget that is
  *measured*, not assumed: each layer reports its attention **concentration**
  during prefill and a fixed total budget is reallocated toward broad
  (low-concentration) layers and away from concentrated ones. When
  `squeeze_strength=0.0` it reduces exactly to uniform H2O.
- **`concentration_score(keys)`** — an attention-free concentration proxy: mean
  pairwise cosine similarity of a layer's key set. High → keys cluster →
  attention concentrated → the layer needs *less* budget.
- **The allocator — `squeeze_budgets(concentrations, avg_budget, n_sink, strength)`** —
  reallocates a fixed total by inverse-concentration (mean held ≈ `avg_budget`,
  floored at `n_sink + 1`); `strength` interpolates linearly between uniform
  (`0.0`) and the full split (`1.0`).
- **`SqueezeCoordinator`** — the first eviction method with a **runtime
  re-budgeting** step. A single shared coordinator (injected at
  `KVCacheBuilder.for_model()` build time) collects per-layer concentration
  during prefill, computes the schedule **once at the prefill boundary**, and
  publishes each layer's resolved budget; over-budget layers are then trimmed by
  H2O score. Unlike XQuant / MiniCache it exchanges only per-layer scalars and
  runs its allocation exactly once — decode steps use the frozen schedule.
- **Sixth distinct eviction configuration in VeloxQuant-MLX** — completing the
  budget-axis matrix: SnapKV (prefill-only), StreamingLLM (positional), H2O
  (uniform), TOVA (memoryless), PyramidKV (fixed per-layer pyramid),
  SqueezeAttention (data-driven per-layer budget).
- **Registered** as `method="squeeze"` in `KVCacheFactory`; new config fields
  `squeeze_budget` (avg, default 512), `squeeze_n_sink` (4), `squeeze_strength`
  (1.0), `squeeze_resolved_budget` (override, None).
- **28 quantizer + 19 cache tests — all 47 passing.** A synthetic benchmark
  (`benchmark_scripts/benchmark_squeeze.py`) sweeps
  `(n_layers, seq_len, avg_budget, strength)` and was run on Apple Silicon;
  results committed in `squeeze_benchmark_results.json`. Confirms the design:
  `strength=0.0` gives uniform budgets (== H2O); `strength>0` reallocates so the
  broad early layer keeps more than the concentrated deep layer; schedule mean
  ≈ `avg_budget`.

#### Adaptation limitations (documented, not a faithful port)

- Key-as-query proxy for both concentration measurement and within-layer
  eviction (same as H2O-adapted / PyramidKV-adapted).
- Cosine-dispersion proxy for attention entropy (paper reads actual attention
  maps, not visible at cache level).
- One-shot re-budget at the prefill boundary, frozen for decode.
- No RoPE position-ID remapping; uniform budget across heads within a layer.
- Benchmark is synthetic (schedule / kept-token / compression only); no
  model-level perplexity or throughput figure is claimed.

## [0.23.1] — 2026-07-03

### Changed

- **License** — extended the copyright notice to `2025-2026` to reflect ongoing
  active development. No code or API changes; this is a metadata-only release so
  the corrected copyright year is rendered on the PyPI project page.

## [0.23.0] — 2026-07-02

### Added — PyramidKV: layer-adaptive budget attention-mass eviction (`method="pyramidkv"`)

- **`veloxquant_mlx.cache.pyramidkv_cache.PyramidKVCache`** — the library's first
  **layer-adaptive budget** eviction method. *Inspired by, not a faithful port of,*
  "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"
  (Cai et al., 2024, arXiv:2406.02069). PyramidKV is H2O's cumulative-attention-mass
  eviction with a **per-layer budget** instead of a single global one: early layers
  (broad attention) get a large budget, deep layers (concentrated attention) get a
  small one, holding the *average* budget fixed so total memory matches a uniform
  baseline. When the pyramid is flat (`pyramid_beta=1.0`) it reduces exactly to
  H2O-adapted.
- **The allocator — `pyramid_budgets(n_layers, avg_budget, n_sink, beta)`** — returns
  the per-layer budget schedule (monotonically decreasing, mean ≈ `avg_budget`,
  floored at `n_sink + 1`). Resolved once at `KVCacheBuilder.for_model()` build time
  and baked into each layer's config as `pyramid_resolved_budget`. **No runtime
  coordinator** is needed (unlike XQuant / MiniCache) — layers never exchange data
  during generation; the only cross-layer signal is each layer's index, consumed at
  build time.
- **Fifth distinct eviction configuration in VeloxQuant-MLX:**
  - SnapKV-adapted — score-based, once at prefill end.
  - StreamingLLM-adapted — positional (recency + sink), constant-memory.
  - H2O-adapted — cumulative attention mass, **uniform** budget, every step.
  - TOVA-adapted — current-step attention weight (memoryless), every step.
  - PyramidKV-adapted — H2O scoring with a **per-layer pyramid** budget.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy (same as H2O-adapted / SnapKV-adapted).
  - Fixed monotone (linear) budget schedule rather than the paper's
    prefill-entropy-derived allocation — funneling shape preserved, exact per-layer
    values not data-driven.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer (the pyramid is across layers).
- Primitives in `veloxquant_mlx/quantizers/pyramidkv.py`: `pyramid_budgets`,
  `PyramidState`, `init_pyramid_state`, `pyramid_update`, `pyramid_get_kv`,
  `pyramid_fp16_bytes`, `full_pyramid_fp16_bytes`.
- Config: `pyramid_budget` (int, default 512, the average/fallback), `pyramid_n_sink`
  (int, default 4), `pyramid_beta` (float, default 2.0 — pyramid steepness; 1.0 = flat).
  Single-cache `KVCacheFactory.create` (no layer context) falls back to
  `pyramid_budget` and behaves as one uniform-budget H2O layer.
- **Tests** — `tests/quantizers/test_pyramidkv.py` (24 tests) +
  `tests/cache/test_pyramidkv_cache.py` (19 tests): allocator shape/monotonicity/
  mean-preservation/flat==uniform/sink-floor/edge-cases, budget enforcement, sink
  protection, byte accounting, determinism, and `for_model` producing a decreasing
  pyramid of per-layer budgets (early layers keep more tokens than deep layers).
- Offline-synthetic harness in `benchmark_scripts/benchmark_pyramidkv.py` sweeping
  `(n_layers, seq_len, avg_budget, beta)` on synthetic fp16 data — **run on Apple
  Silicon**; results committed in `benchmark_scripts/pyramidkv_benchmark_results.json`
  (24 configs). They confirm the design end-to-end: `beta=1.0` gives a flat schedule
  (== uniform H2O), `beta>1.0` gives strictly decreasing budgets with early layers
  retaining more tokens than deep layers, and schedule mean == `avg_budget`
  everywhere. No model-level perplexity/throughput figures are claimed.

---

## [0.22.0] — 2026-07-01

### Added — TOVA: current-step attention-weight eviction, memoryless (`method="tova"`)

- **`veloxquant_mlx.cache.tova_cache.TOVAKVCache`** — the library's first
  **memoryless** eviction method. *Inspired by, not a faithful port of,*
  "Transformers are Multi-State RNNs" (Oren et al., 2024, arXiv:2401.06104), whose
  TOVA (Token Omission Via Attention) policy keeps a fixed-size cache by dropping,
  at each step, the single token receiving the **lowest attention weight in the
  current step**. On every step (prefill and decode alike), the approximate
  current-step attention distribution over the post-append cache is computed using
  the **new key vector as a proxy query** (true queries are not visible at
  cache-wrapper level — same approximation as SnapKV-adapted and H2O-adapted).
  When the cache exceeds `tova_budget`, the **lowest current-step-weight non-sink
  token** is permanently evicted. The cache is bounded at all times to
  `tova_budget` positions.
- **Fourth distinct eviction axis in VeloxQuant-MLX — and the key contrast with H2O:**
  - SnapKV-adapted — score-based, fires once at prefill end; grows during decode.
  - StreamingLLM-adapted — positional (recency + sinks), constant-memory throughout.
  - H2O-adapted — **cumulative** attention mass (inertial: past heavy hitters resist eviction).
  - TOVA-adapted — **current-step** attention weight (memoryless: a token that stops
    being attended to is evicted even if it dominated earlier). TOVA is the more
    reactive policy; H2O is the more conservative one.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy: approximates the paper's true query attention signal.
  - No RoPE position-ID remapping after eviction.
  - Uniform `tova_budget` and `tova_n_sink` across all heads.
- Primitives in `veloxquant_mlx/quantizers/tova.py`: `TovaState`,
  `init_tova_state`, `tova_update`, `tova_get_kv`, `tova_fp16_bytes`,
  `full_tova_fp16_bytes`. No `scores` field — state carries no cross-step history.
- Config: `tova_budget` (int, default 512), `tova_n_sink` (int, default 4).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all
  `tova_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_tova.py` (19 tests) +
  `tests/cache/test_tova_cache.py` (15 tests): init state, no-scores-field assertion,
  single-token bootstrap, multi-token absorption, budget enforcement (never exceeded
  across 30 decode steps), sink protection (sinks always present after evictions),
  n_sink=0 edge case, memorylessness (no scores carried across steps), current-step
  eviction correctness (a token orthogonal to the current key is dropped over a
  similar one), byte accounting formula, compression_ratio, tokens_seen, factory
  dispatch, for_model propagation, determinism.
- Offline-synthetic harness in `benchmark_scripts/benchmark_tova.py` sweeping
  `(seq_len, budget, n_sink)` on synthetic fp16 data — **run on Apple Silicon**;
  results committed in `benchmark_scripts/tova_benchmark_results.json` (28 configs).
  Measured compression ratio equals `seq_len / budget` exactly across every config
  (e.g. 2048 tokens at budget 64 → 32×). No model-level perplexity/throughput
  figures are claimed.

---

## [0.21.0] — 2026-07-01

### Added — H2O: cumulative attention-mass heavy-hitter oracle eviction (`method="h2o"`)

- **`veloxquant_mlx.cache.h2o_cache.H2OKVCache`** — the library's first
  **continuous-decode cumulative-score eviction** method. *Inspired by, not a
  faithful port of,* "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
  of Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048). On every
  step (prefill and decode alike), each incoming token's approximate attention
  distribution over the existing cache is computed using the **new key vector as a
  proxy query** (true queries are not visible at cache-wrapper level — same
  approximation as SnapKV-adapted). The resulting softmax weights are accumulated
  into a per-token cumulative importance score. When the cache exceeds
  `h2o_budget`, the **lowest-score non-sink token** is permanently evicted.
  The cache is thus bounded at all times to `h2o_budget` positions.
- **Third distinct eviction axis in VeloxQuant-MLX:**
  - SnapKV-adapted — score-based, fires once at prefill end; grows during decode.
  - StreamingLLM-adapted — positional (recency + sinks), constant-memory throughout.
  - H2O-adapted — cumulative attention mass, budget-bounded at every step.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy: approximates the paper's true query attention signal.
  - No RoPE position-ID remapping after eviction.
  - Uniform `h2o_budget` and `h2o_n_sink` across all heads.
  - Scores accumulate as a running sum of softmax weights; some paper variants
    accumulate unnormalised logits — may diverge at very low budgets.
- Primitives in `veloxquant_mlx/quantizers/h2o.py`: `H2OState`,
  `init_h2o_state`, `h2o_update`, `h2o_get_kv`, `h2o_fp16_bytes`,
  `full_h2o_fp16_bytes`.
- Config: `h2o_budget` (int, default 512), `h2o_n_sink` (int, default 4).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all
  `h2o_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_h2o.py` (18 tests) +
  `tests/cache/test_h2o_cache.py` (15 tests): init state, single-token bootstrap,
  multi-token absorption, budget enforcement (never exceeded across 30 decode steps),
  sink protection (sinks always present after evictions), n_sink=0 edge case,
  score non-negativity, score accumulation across steps, byte accounting formula,
  compression_ratio, tokens_seen, factory dispatch, for_model propagation,
  determinism.
- Offline-synthetic harness in `benchmark_scripts/benchmark_h2o.py` sweeping
  `(seq_len, budget, n_sink)` on synthetic fp16 data. Not yet run on Apple Silicon
  hardware.

---

## [0.20.0] — 2026-07-01

### Added — StreamingLLM: sink + recency-window structural eviction (`method="streaming_llm"`)

- **`veloxquant_mlx.cache.streaming_llm_cache.StreamingLLMKVCache`** — the repo's
  first **constant-memory** cache and first **structural positional eviction** method.
  *Inspired by, not a faithful port of,* "Efficient Streaming Language Models with
  Attention Sinks" (Xiao et al., ICLR 2024, arXiv:2309.17453). Keeps only the first
  `stream_n_sink` token positions (frozen as attention sinks) and the most recent
  `stream_window_size` positions (rolling FIFO). All other positions are permanently
  evicted. Both prefill (`S > 1`) and decode (`S == 1`) tokens are processed
  identically through the same sink+window logic — the cache **never** grows beyond
  `stream_n_sink + stream_window_size` positions regardless of how many tokens are
  generated. The `streaming_ratio` and `tokens_in_window` properties report storage
  accounting.
- **Orthogonal to SnapKV-adapted**: SnapKV evicts by importance score at prefill and
  then grows during decode; StreamingLLM-adapted evicts continuously by position and
  stays constant-memory throughout generation.
- **Adaptation limitations (documented, not hidden):**
  - No attention mask adjustment — the model attends to all returned K/V positions; only
    the number of K/V rows is bounded.
  - No RoPE position-ID remapping — original token positions preserved in returned rows;
    remapping requires model-level patching.
  - Fixed `stream_n_sink` count — not adaptive.
- Primitives in `veloxquant_mlx/quantizers/streaming_llm.py`: `StreamingWindow`,
  `init_streaming_window`, `stream_update`, `stream_get_kv`, `stream_fp16_bytes`,
  `full_stream_fp16_bytes`.
- Config: `stream_n_sink` (int, default 4), `stream_window_size` (int, default 512).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all `stream_*`
  fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_streaming_llm.py` (17 tests) +
  `tests/cache/test_streaming_llm_cache.py` (15 tests): init shapes, sink absorption,
  FIFO trimming, constant-memory guarantee (30-step stress), stream_get_kv shape/dtype/
  sink-first ordering, byte accounting, streaming_ratio, large-prefill trim, n_sink=0
  edge, determinism, for_model config propagation. **32/32 passing.**
- Offline-synthetic harness in `benchmark_scripts/benchmark_streaming_llm.py` sweeping
  `(seq_len, window_size)` on synthetic data. Not yet run on Apple Silicon hardware.

---

## [0.19.0] — 2026-07-01

### Added — SnapKV: prefill observation-window token eviction (`method="snapkv"`)

- **`veloxquant_mlx.cache.snapkv_cache.SnapKVKVCache`** — the repo's first
  **token eviction** cache and the first where the paper's actual attention
  signal is computable at cache level without model surgery. *Inspired by, not
  a faithful port of,* "SnapKV: LLM Knows What You are Looking for Before
  Generation" (Yuan et al., ICLR 2025, arXiv:2404.14469). During prefill
  (`S > 1`), the last `snap_obs_window` key rows act as proxy queries; scaled
  dot-product softmax over all `S` prefix key positions gives per-token
  importance scores. The top-`snap_budget` tokens (plus `snap_n_sink`
  always-kept sink positions) are retained as fp16. All evicted positions are
  permanently dropped. Decode tokens (`S == 1`) are always appended — never
  evicted. The `eviction_ratio` and `keep_rate` properties report the storage
  accounting.
- **Adaptation:** the paper uses the final prompt *query* vectors for the
  observation window (not visible to a cache wrapper). We substitute the last
  `snap_obs_window` *key* vectors as proxy queries — stronger than key-norm
  alone (computes the actual attention distribution from K) but still an
  approximation. No max-pool smoothing (paper's `kernel_size > 1`). Uniform
  budget across all heads. Documented as "SnapKV-adapted (key-as-query proxy)"
  throughout; never claimed as a faithful port.
- Primitives in `veloxquant_mlx/quantizers/snapkv.py`: `obs_window_attention_scores`,
  `snap_select_indices`, `snapkv_compress`, `snapkv_fp16_bytes`, `full_fp16_bytes`
  (+ `SnapKVState`).
- Config: `snap_budget` (int, default 512), `snap_obs_window` (int, default 32),
  `snap_n_sink` (int, default 4). Single-layer (no coordinator); `KVCacheBuilder.for_model()`
  propagates all `snap_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_snapkv.py` (18 tests) +
  `tests/cache/test_snapkv_cache.py` (13 tests): obs-window scores shape, dtype,
  value range; `obs_window` clamp; `snap_select_indices` exact count, sorted order,
  sink guarantee, high-score preference; `snapkv_compress` output shape/dtype;
  budget≥S no-eviction edge case; byte accounting; eviction ratio > 1; keep rate
  in range; decode accumulation; decode-only no-eviction; determinism;
  `for_model` propagation.
- **Benchmark** — `benchmark_scripts/benchmark_snapkv.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** key-as-query proxy; no max-pool smoothing; no per-head budget;
  no model-level benchmark yet.

## [0.18.0] — 2026-06-30

### Added — ZipCache: saliency-adaptive per-token mixed-precision (`method="zipcache"`)

- **`veloxquant_mlx.cache.zipcache_cache.ZipCacheKVCache`** — the repo's first
  **per-token mixed bit-width** cache. *Inspired by, not a faithful port of,*
  "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token
  Identification" (He et al., NeurIPS 2024, arXiv:2405.14256). The top
  `zipcache_hi_fraction` of tokens by key L2-norm are quantized at `zipcache_hi_bits`;
  the rest at `zipcache_lo_bits`. Both groups remain quantized — this is not fp16
  protection (KIVI-Sink) nor head budgeting (AdaKV-proxy). Effective average key rate:
  `hi_frac × hi_bits + (1-hi_frac) × lo_bits`.
- **Adaptation:** the paper's true saliency signal is normalized attention scores,
  which are not observable by a cache wrapper. Key L2-norm is the proxy (same signal
  used by KIVI-Sink and AdaKV-proxy, but with a different decision — bit-width routing
  rather than fp16 protection or head budgeting). This is the third use of the key-norm
  proxy; the proxy weakness is documented plainly.
- Primitives in `veloxquant_mlx/quantizers/zipcache.py`: `token_key_norms`,
  `saliency_mask`, `channel_quant`, `channel_dequant`, `zipcache_compress`,
  `zipcache_reconstruct`, `zipcache_bytes`, `base_only_bytes`, `zipcache_quant_dequant`
  (+ `ZipCacheState`).
- Config: `zipcache_hi_bits`, `zipcache_lo_bits`, `zipcache_hi_fraction`,
  `zipcache_group_size`, `zipcache_quantize_values`. Single-layer (no coordinator);
  `KVCacheBuilder.for_model()` propagates all `zipcache_*` fields automatically via
  `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_zipcache.py` (16 tests) +
  `tests/cache/test_zipcache_cache.py` (11 tests): saliency mask selects exact
  top-fraction by key-norm; 4-bit channel quant cosine > 0.995; 2-bit cosine > 0.8;
  compress/reconstruct shape and dtype; `hi_fraction=0` and `=1` edge cases;
  byte ordering `compressed ≤ fp16`, mixed-bit ≥ all-lo-bit baseline; effective avg
  bits in `[lo_bits, hi_bits]`; values-off passthrough; decode accumulation;
  determinism; build via both `create` and `for_model`.
- **Benchmark** — `benchmark_scripts/benchmark_zipcache.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** proxy weakness (key-norm, not true attention scores) is stated in
  all docs; no model-level benchmark run yet.

## [0.17.0] — 2026-06-29

### Added — GEAR: error-feedback KV cache (`method="gear"`)

- **`veloxquant_mlx.cache.gear_cache.GEARKVCache`** — the repo's first
  **error-feedback** cache. *Inspired by, not a faithful port of,* "GEAR: An
  Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of
  LLM" (Kang et al., arXiv:2403.05527). Every other method picks a bit-width (or
  a cache layout) and lives with the quantization error; GEAR makes *any*
  ultra-low-bit base quantizer near-lossless by reconstructing what it threw away
  via the three-part decomposition `X ≈ Quant_b(X) + L·R + S`: an ultra-low-bit
  base group quant, a **low-rank** approximation of the quantization residual
  `E = X − dequant(Quant_b(X))`, and a **sparse** matrix correcting the
  top-magnitude outlier entries the low-rank term cannot absorb. Unlike CacheGen
  (reconstruction identical to group quant), GEAR's reconstruction genuinely
  **recovers quality** the base bit-width loses.
- **Adaptation:** the residual SVD is computed per `update_and_fetch` call on the
  tensor the cache holds (reusing the SVDq/PALU prefill-SVD pattern), and GEAR's
  fused streaming-dequant CUDA kernel is **not** ported — we reconstruct fp16
  then call MLX SDPA, so the *stored* cache shrinks but attend-time peak memory
  does not. The base quant is borrowed from CacheGen and the truncated-SVD helper
  (`_quant_utils._truncated_svd`) is shared with SVDq/PALU.
- Primitives in `veloxquant_mlx/quantizers/gear.py`: `quantize_base`, `residual`,
  `lowrank_error`, `sparse_outliers`, `gear_compress`, `gear_reconstruct`,
  `gear_bytes`, `base_only_bytes`, `gear_quant_dequant` (+ `GEARState`).
- Config: `gear_bits`, `gear_rank`, `gear_energy_threshold`,
  `gear_sparse_fraction`, `gear_group_size`, `gear_quantize_values`. Single-layer
  (no coordinator); `KVCacheBuilder.for_model()` propagates the `gear_*` fields
  automatically via `dataclasses.replace`.
- **Tests** — `tests/cache/test_gear_cache.py` (10) +
  `tests/quantizers/test_gear.py` (13): GEAR reconstruction MSE strictly below
  base-quant-alone on low-rank+outlier data; low-rank-alone and sparse-alone each
  help; `rank=0, sparse=0` collapses exactly to base group quant; rank-`r`
  residual recovered to `< eps`; sparse selection picks true top-magnitude
  entries; byte-accounting ordering `base_only ≤ compressed ≤ fp16`;
  `error_recovery_ratio` in `(0,1]`; values-off path; decode accumulation;
  determinism; build via both `create` and `for_model`.
- **Benchmark** — `benchmark_scripts/benchmark_gear.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** the stored cache shrinks but reconstruction is fp16 for SDPA,
  so attend-time peak memory is not reduced; the low-rank/sparse factors are
  overhead, so the rank must be low relative to the head dim (the GEAR premise) —
  reported honestly, never hidden.

## [0.16.0] — 2026-06-26

### Added — CacheGen: entropy-coded KV cache (`method="cachegen"`)

- **`veloxquant_mlx.cache.cachegen_cache.CacheGenKVCache`** — the repo's first
  **entropy-coded** cache. *Inspired by, not a faithful port of,* "CacheGen: KV
  Cache Compression and Streaming for Fast LLM Serving" (Liu et al., **SIGCOMM
  2024**, arXiv:2310.07240). Every other method packs codes at a fixed
  bit-width; CacheGen exploits token-wise locality (adjacent tokens' KV are
  similar) by applying a reversible token-delta transform to the quantized codes
  and compressing the low-entropy residual stream toward its Shannon entropy.
  Reconstruction is identical to plain group quant (lossless over the codes).
- **Adaptation:** rather than ship a serial range codec (which would bottleneck
  MLX's parallel decode), the entropy-coded byte size is modelled from the
  measured symbol entropy of the delta stream and **capped at the fixed-width
  packed size** — a real coder falls back to raw packing when the stream is
  incompressible, so savings are never negative (exactly 0% on iid data, ~10–17%
  on token-correlated data).
- Primitives in `veloxquant_mlx/quantizers/cachegen.py`: `quantize_to_codes`,
  `dequant_codes`, `token_delta`, `symbol_entropy_bits`, `entropy_coded_bytes`,
  `fixed_width_bytes`, `cachegen_quant_dequant`.
- Config: `cachegen_bits`, `cachegen_group_size`, `cachegen_use_delta`.
- **Tests** — `tests/cache/test_cachegen_cache.py` (12) +
  `tests/quantizers/test_cachegen.py` (9): lossless reconstruction vs group
  quant, reversible token-delta, delta-entropy < raw-entropy on correlated data,
  positive savings on correlated / never-negative on iid, entropy primitives
  (0 for constants, 1 bit for 50/50, bounded by log2-alphabet), byte-accounting
  ordering, decode, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_cachegen.py` (offline entropy
  harness + throughput vs KIVI/fp16). **Not yet run.**

### Added — MiniCache: cross-layer depth-dimension merge (`method="minicache"`)

- **`veloxquant_mlx.cache.minicache_cache.MiniCacheKVCache`** +
  **`MiniCacheCoordinator`** — cross-layer compression in the **depth
  dimension**. *Inspired by* "MiniCache: KV Cache Compression in Depth Dimension
  for Large Language Models" (Liu et al., **NeurIPS 2024**, arXiv:2405.14366).
  Adjacent middle-to-deep layers have nearly identical KV directions, so a pair
  is merged into one shared **SLERP**-interpolated direction plus each layer's
  own per-token magnitude (a pair costs ~one layer). High-divergence token pairs
  are kept unmerged (the retention set). A different route to inter-layer
  redundancy than XQuant — XQuant reuses quantized *codes*, MiniCache merges the
  *tensors*.
- **Adaptation:** faithful to the magnitude/direction SLERP + token retention;
  integrated via a shared coordinator (the XQuant pattern) rather than a modified
  attention forward. The primary layer publishes its KV so the later-arriving
  merge layer can perform the merge — both then reconstruct from the shared
  direction.
- Primitives in `veloxquant_mlx/quantizers/minicache.py`: `pair_layers_depth`,
  `to_mag_dir`, `slerp`, `merge_pair`, `reconstruct_layer`, `merge_similarity`.
- Config: `minicache_start_frac`, `minicache_group_size`,
  `minicache_retention_threshold`, `minicache_slerp_t`, `minicache_max_ctx`.
- **Tests** — `tests/cache/test_minicache_cache.py` (11) +
  `tests/quantizers/test_minicache.py` (11): role assignment (early all primary,
  deep has merge), SLERP endpoints/unit-norm/collinear-fallback, similar layers
  merge MSE < 2e-4 with 0% retention, opposite directions 100% retained and
  reconstructed exactly, magnitude preservation, `n_retained+n_merged==total`,
  degenerate lossless passthrough, coordinator `max_ctx` guard, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_minicache.py` (offline merge-
  quality harness + throughput vs XQuant/KIVI/fp16). **Not yet run.**

### Honest scope

- Both are **storage**-compression methods: CacheGen's entropy coding and
  MiniCache's merge reduce stored cache size but reconstruct fp16 for SDPA, so
  neither reduces working-set memory at attend time. On Apple Silicon's
  bandwidth-bound decode they are lower-leverage than the low-rank (PALU/SVDq)
  and quantization methods.
- Quality evidence is unit-test level (synthetic data); no model-level benchmark
  or downstream-task evaluation has been run.

## [0.15.0] — 2026-06-26

### Added — PALU: true low-rank latent storage for keys *and* values (`method="palu"`)

- **`veloxquant_mlx.cache.palu_cache.PALUKVCache`** — the first method in the
  suite where the KV cache *itself* stays low-rank. *Inspired by, not a faithful
  port of,* "PALU: Compressing KV-Cache with Low-Rank Projection" (Chang et al.,
  **ICLR 2025**, arXiv:2407.21118). At prefill it partitions the attention heads
  into `palu_n_head_groups` contiguous groups and fits one shared projection per
  group via group-head SVD (PALU's G-LRD), then stores the projected codes
  `[S, r]` **directly** — full fp16 keys/values are reconstructed only at attend
  time. The latents are mixed-bit quantized (top-25% of channels by singular
  value at 4-bit, the rest at 2-bit, reusing the SVDq latent coder) for a
  full-KV effective rate below 1 bit/element on low-rank data. Unlike SVDq
  (keys-only, reconstructs full fp16 and so wins on byte-accounting/bandwidth),
  PALU bypasses the parent `mlx_lm` fp16 ring buffer entirely and tracks its own
  offset — the stored-cache win is real.
- **`veloxquant_mlx.quantizers.palu`** — pure primitives `head_group_bounds`,
  `group_head_svd`, `project_to_latent`, `reconstruct_from_latent`,
  `quantize_latent`.
- **`KVCacheConfig`** — new fields `palu_rank`, `palu_energy_threshold`
  (default 0.90), `palu_n_head_groups` (default 4), `palu_hi_bit`, `palu_lo_bit`,
  `palu_hi_fraction`, `palu_group_size`, `palu_quantize_values` (default True;
  `False` → low-rank-only with fp16 latents).
- **Tests** — `tests/cache/test_palu_cache.py` (13) + `tests/quantizers/test_palu.py`
  (9): factory dispatch, no-`.bits`-leak, group projections stored,
  prefill/decode shape, the **latent-storage assertion** (buffers hold `[S, r]`,
  parent `keys is None`), PALU-beats-naive-2bit on **both** K and V, decode
  accumulation + offset growth, both-tensors-compressed accounting,
  low-rank-only values, sub-2-bit effective rate, energy-threshold rank,
  head-grouping, group-SVD subspace recovery, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_palu.py` (fp16 / KIVI-2bit /
  SVDq / PALU-LR-only / PALU-LR+mixed / PALU-aggressive) plus an offline
  full-KV reconstruction-MSE harness. **Not yet run** — no throughput or
  compression figures are claimed for this method until its `results.json` is
  committed.

### Fixed

- `KVCacheBuilder.for_model()` now propagates **all** method-specific config
  fields (`svdq_*`, `kitty_*`, `kvquant_*`, `palu_*`, …) to each per-layer cache
  via `dataclasses.replace`. Previously it rebuilt the per-layer config field by
  field and silently dropped method hyperparameters, so any method built through
  `for_model` ran with default hyperparameters regardless of the user's config.

### Honest scope

- PALU's fused low-rank-reconstruction attention kernel is **not** ported — we
  reconstruct fp16 then call MLX SDPA. The storage is low-rank, but the working
  set during attention is briefly the reconstructed fp16 K/V, so peak memory at
  attend time is not reduced — only the stored cache size. Documented as a known
  simplification.
- Quality evidence is unit-test level (synthetic low-rank data); no model-level
  benchmark or downstream-task evaluation has been run.

## [0.14.0] — 2026-06-25

### Added — KVQuant-NUQ: non-uniform quantization + outlier isolation (`method="kvquant"`)

- **`veloxquant_mlx.cache.kvquant_cache.KVQuantKVCache`** — *Inspired by, not
  a faithful port of,* "KVQuant: Towards 10 Million Context Length LLM
  Inference with KV Cache Quantization" (arXiv:2401.18079, NeurIPS 2024).
  Implements the two cache-observable pillars: a non-uniform quantization
  (NUQ) datatype fit online via Lloyd-Max iterations, plus dense/sparse
  outlier isolation that carves the top-magnitude elements out to an fp16
  side-channel. Matches KVQuant's per-channel-keys / per-token-values
  quantization axis asymmetry (the same axes KIVI uses). Pre-RoPE key
  quantization (the paper's third pillar) is out of scope — a cache wrapper
  only sees post-RoPE keys.
- **`veloxquant_mlx.quantizers.kvquant`** — the NUQ level-fitting and
  outlier-isolation primitives.
- **`KVCacheConfig`** — new fields `kvquant_bits` (default 3),
  `kvquant_outlier_fraction` (default 0.01), `kvquant_group_size` (default
  32), `kvquant_lloyd_iters` (default 8), `kvquant_refit_interval` (default
  0 — levels are fit once at prefill and frozen; a positive value re-fits
  every N decode steps).
- **Tests** — `veloxquant_mlx/tests/cache/test_kvquant_cache.py` (15 tests
  at introduction).
- **Benchmark script** — `benchmark_scripts/benchmark_kvquant.py`.

### Honest scope

- Pre-RoPE key quantization is not implemented (needs a model-forward hook,
  outside the cache contract).
- Level fitting is online/zero-calibration, not offline calibration-set
  fitting as in the paper.
- Attention-aware sensitivity weighting is not implemented (needs attention
  scores, which a cache wrapper does not see).

## [0.13.0] — 2026-06-25

### Added — XQuant: cross-layer KV cache reuse (`method="xquant"`)

- **`veloxquant_mlx.cache.xquant_cache.XQuantKVCache`** +
  **`veloxquant_mlx.cache.xquant_coordinator.XQuantCoordinator`** — *Inspired
  by* "XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer
  Compression" (arXiv:2510.11236, EMNLP 2025); faithful to the cross-layer
  reuse core, adapted at the integration boundary via a shared coordinator
  rather than a modified attention forward pass. Adjacent layers are paired
  at build time (`pair_layers`) into an anchor and a reuse role: the anchor
  quantizes K/V with asymmetric group quantization and publishes its integer
  codes to the coordinator; the reuse layer fetches the paired anchor's codes
  for the same token range and fits only its own per-group scale/zero to
  correct cross-layer drift — never storing a full code tensor of its own.
  The first method in the suite to exploit *inter-layer* redundancy. Both
  keys and values are compressed.
- **`veloxquant_mlx.quantizers.xquant`** — the anchor/reuse quantization
  primitives.
- **`KVCacheConfig`** — new fields `xquant_group_size` (default 2, layers per
  anchor/reuse pair), `xquant_base_bits` (default 2), `xquant_residual_bits`
  (default 0 — pure reuse), `xquant_group_quant_size` (default 32),
  `xquant_max_ctx` (default 8192, coordinator per-group token budget).
- **Tests** — `veloxquant_mlx/tests/cache/test_xquant_cache.py` (16 tests at
  introduction).
- **Benchmark script** — `benchmark_scripts/benchmark_xquant.py`.

### Honest scope

- With no coordinator supplied (a single isolated layer), the cache
  degenerates to a plain anchor — documented as a unit-testing convenience,
  not the intended deployment shape.

## [0.12.0] — 2026-06-24

### Added — AdaKV-proxy: per-head adaptive bit allocation (`method="adakv"`)

- **`veloxquant_mlx.cache.adakv_cache.AdaKVCache`** — a proxy adaptation of
  "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for
  Efficient LLM Inference" (arXiv:2407.11550, 2024), documented as
  "AdaKV-proxy" rather than a faithful port. Layers on top of KIVI-style
  group quantization: ranks attention heads by running per-head key-norm
  variance and solves a per-head bit assignment so the average bits/element
  across heads hits a configured target, recomputed every step by default.
  Complements Kitty's per-channel axis with a per-head axis. Key-only;
  values stay fp16.
- **`veloxquant_mlx.quantizers.adakv`** — the per-head bit-assignment
  primitives.
- **`KVCacheConfig`** — new fields `adakv_target_avg_bits` (default 2.0),
  `adakv_lo_bit` (default 2), `adakv_mid_bit` (default 3), `adakv_hi_bit`
  (default 4), `adakv_group_size` (default 32), `adakv_update_interval`
  (default 1 — the field is wired but the bit assignment is recomputed every
  step regardless of this value; documented as a future optimisation, not a
  current behavior).
- **Tests** — `veloxquant_mlx/tests/cache/test_adakv_cache.py` (14 tests at
  introduction).
- **Benchmark script** — `benchmark_scripts/benchmark_adakv.py`.

### Honest scope

- True Ada-KV head-adaptive *eviction* budget (the paper's actual mechanism)
  is not implemented — it needs softmax attention scores a cache wrapper
  doesn't see. Cross-layer budget sharing is likewise not implemented.

## [0.11.0] — 2026-06-23

### Added — Kitty: dynamic channel-wise mixed-precision key cache (`method="kitty"`)

- **`veloxquant_mlx.cache.kitty_cache.KittyKVCache`** — *Inspired by, not a
  faithful port of,* "Kitty: Plug-and-Play Continuous Batching with Dynamic
  Token Selection" (arXiv:2511.18643, Nov 2025, unreviewed preprint). Ranks
  key channels by running variance (updated incrementally from prefill
  accumulators) and routes the top `kitty_hi_fraction` of channels to
  `kitty_hi_bit`, the rest to `kitty_lo_bit`, via asymmetric group
  quantization. Zero calibration. Key-only; values stay fp16. Default
  configuration gives an effective 2.5 bits/element (6.4× key bandwidth
  reduction).
- **`veloxquant_mlx.quantizers.kitty`** — the channel-ranking and
  mixed-precision quantization primitives.
- **`KVCacheConfig`** — new fields `kitty_hi_fraction` (default 0.25),
  `kitty_hi_bit` (default 4), `kitty_lo_bit` (default 2), `kitty_group_size`
  (default 32).
- **Tests** — `veloxquant_mlx/tests/cache/test_kitty_cache.py` (12 tests at
  introduction).
- **Benchmark script** — `benchmark_scripts/benchmark_kitty.py`.

## [0.10.0] — 2026-06-21

### Added — SVDq: sub-2-bit key cache via offline SVD (`method="svdq"`)

- **`veloxquant_mlx.cache.svdq_cache.SVDqKVCache`** — *Inspired by, not a
  faithful port of,* "SVDq: Singular Value Decomposition-based KV Cache
  Quantization" (arXiv:2502.15304, Feb 2025, unreviewed preprint). At
  prefill, computes a truncated SVD of the incoming key batch, stores the
  right singular vectors and mean key as layer state, and projects keys into
  that latent space. Latents are mixed-precision group quantized (top-25%
  of channels by singular value at 4-bit, rest at 2-bit). Decode tokens
  project into the already-fitted latent space. Key-only — values stay fp16
  throughout (the source material notes values have weaker low-rank
  structure). Default configuration gives an effective ~1.25 bits/element.
- **`veloxquant_mlx.quantizers.svdq`** — the SVD-fitting and latent
  quantization primitives.
- **`KVCacheConfig`** — new fields `svdq_rank` (default `None` → energy
  threshold), `svdq_energy_threshold` (default 0.95), `svdq_hi_bit` (default
  4), `svdq_lo_bit` (default 2), `svdq_hi_fraction` (default 0.25),
  `svdq_group_size` (default 32).
- **Tests** — `veloxquant_mlx/tests/cache/test_svdq_cache.py` (12 tests at
  introduction).
- **Benchmark script** — `benchmark_scripts/benchmark_sink.py` (this release's
  commit also carried the sink-protection work described under
  [0.9.0](#090--2026-06-12) below).

## [0.9.0] — 2026-06-12

### Added — KVSink-adapted sink protection (`method="kivi_sink"`)

- **`veloxquant_mlx.cache.sink_cache.SinkProtectedKVCache`** — dynamic
  attention-sink protection layered on KIVI group quantization. *Inspired
  by, not a faithful port of,* "KVSink: Understanding and Enhancing the
  Preservation of Attention Sinks in KV Cache Quantization for LLMs"
  (Su & Yuan, **COLM 2025**, arXiv:2508.04257): the paper detects sinks via
  hidden-state outlier channels at a model-specific emergence layer, which
  cache wrappers cannot see; this implementation uses the cache-observable
  proxy of **anomalously high key L2-norm** (mean over KV heads, running
  top-k of absolute positions). Selected tokens are kept fp16 and —
  critically, per the paper — **excluded from quantization-parameter
  calibration** (sink rows are replaced by the nearest non-sink row before
  group min/max is computed; without this, a large-magnitude sink inflates
  its group's scale and ruins every neighbor even though the sink itself is
  restored — our tests reproduce that failure when calibration exclusion is
  omitted).
- **`KVCacheConfig.n_sink_tokens`** — new field (default 5, the paper's k).
  Composes with KIVI's `residual_length` window; byte accounting tracks
  `sink_fp16_bytes` separately from `residual_fp16_bytes` with no double
  counting. `n_sink_tokens=0` reproduces plain KIVI bit-for-bit (tested).
- **Tests** — `tests/cache/test_sink_cache.py` (9 tests): planted-sink
  detection + bit-exact fp16 preservation; sink-protected MSE < plain KIVI
  at equal bit-width; **dynamic selection MSE < Preserve-First-N at equal
  fp16 budget** (the KVSink paper's central claim, reproduced at cache
  level on synthetic planted-sink data); accounting partition; determinism.
  Full suite: **344 passed / 348 collected** (4 pre-existing flaky VecInfer
  parity tests, unrelated).
- **Benchmark script** — `benchmark_scripts/benchmark_sink.py` (fp16 /
  KIVI-2bit / +sink k=5 / +sink k=20, long-prompt protocol). **Not yet
  run** — no throughput or compression figures are claimed for this method
  until its `results.json` is committed.

### Honest scope

- Known v1 limitation: sink selection is **prefill-dominant** — tokens
  quantized in earlier calls are not retroactively restored if they later
  qualify as sinks. Sinks emerge among early tokens in practice, which
  arrive in the prefill block where protection is fully effective.
- Quality evidence is unit-test level (synthetic planted sinks); no
  model-level benchmark or downstream-task evaluation has been run.

## [0.8.0] — 2026-06-10

### Added — KIVI: tuning-free asymmetric group quantization (baseline)

- **`veloxquant_mlx.quantizers.kivi.KIVIQuantizer`** — re-implementation of
  "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (Liu, Yuan
  et al., **ICML 2024**, arXiv:2402.02750). Deterministic asymmetric min/max
  group quantization: **per-channel keys** (group along the token axis) and
  **per-token values** (group along the channel axis). No codebook training,
  no rotation, no RNG. Registered as `"kivi"` in `QuantizerRegistry`.
- **`veloxquant_mlx.cache.kivi_cache.KIVIKVCache`** — mlx_lm
  `update_and_fetch` wrapper. Keeps the most-recent `residual_length` tokens
  in fp16 (KIVI's residual window) and quantizes only tokens that age out.
  Full byte-accounting (`compressed_key_bytes`, `fp16_key_bytes`,
  `residual_fp16_bytes`); never exposes `.bits`. Selectable via
  `KVCacheConfig(method="kivi", bit_width_inlier=2, kivi_group_size=32,
  residual_length=32)`.
- **`KVCacheConfig.kivi_group_size`** — new field (default 32).
- **Benchmarks** — `benchmark_scripts/benchmark_kivi.py` records throughput,
  peak memory, and realized key / full-KV compression with a **real fp16
  baseline timing** and a `hardware` block, under
  `figures/kivi/<model>/results.json`. Measured on Llama-3.2-3B, Qwen2.5-7B,
  Mistral-7B (Apple M4): **KIVI-2bit ≈ 5.8× key / ≈ 4× full-KV at 100–106%
  of fp16 throughput**.
- **Figures** — `scripts/plot_kivi.py` emits four figures (compression vs
  quality, throughput, analytic memory-at-scale, KIVI-vs-VecInfer) +
  `figures/kivi/results_summary.json`, all read from committed JSONs.
- **Tests** — `tests/quantizers/test_kivi.py` and
  `tests/cache/test_kivi_cache.py`: shape/dtype, deterministic seeded
  reconstruction cosine/SNR per bit-width, monotone-quality-in-bits,
  residual-window correctness, byte-accounting, no-`.bits`-leak. **+25 tests
  (334/339 pass; the 5 failures are the pre-existing flaky VecInfer parity
  tests documented in `paper/EVIDENCE_TABLE.md`, unrelated to KIVI).**

### Honest scope

- KIVI's published *speedup* is a CUDA kernel that does not port to Metal; on
  Apple Silicon the win is **memory**, not raw speed.
- Compression only manifests once context exceeds the residual window; at
  short prompts the whole prefill stays fp16 (realized ratio 1.0×).
- Peak runtime memory is **not** reduced (keys dequantize to fp16 before SDPA).
- KIVI-2bit is genuinely lossy on raw keys (synthetic cosine ~0.93); VecInfer
  compresses harder. KIVI's role is the recognized, calibration-free baseline.

## [0.5.1] — 2026-05-25

### Added — Metal compute kernels for VecInfer (Phase 1)

- **`veloxquant_mlx.metal`** — new subpackage with hand-written Metal
  Shading Language shaders that replace pure-MLX hot paths in
  `VecInferKVCache`. JIT-compiled on first use via `mx.fast.metal_kernel`.
  - `vecinfer_quantize_metal` — fused nearest-centroid argmin. Squared
    distance is accumulated in thread-local registers so the kernel never
    materializes the `[chunk, n_centroids, sub_dim]` diff tensor that
    OOMed Falcon3-7B-style configurations on the pure-MLX path.
    **Measured: 6.9–13× speedup, 98% peak-memory reduction at the OOM
    trigger shape (head_dim=256, n_centroids=256, sub_dim=4).**
  - `vecinfer_dequant_metal` — bit-exact drop-in for `dequantize_vq`.
    Ships at MLX `mx.take` parity (no speedup); included as a building
    block for the Phase-2 fused dequant+SDPA kernel.
  - `metal_available()` capability probe.
- **`KVCacheConfig.use_metal_kernels`** — three-state opt-in flag.
  `None` (default) auto-detects, `True` requires Metal, `False` forces
  the pure-MLX path for debugging/parity testing.
- **`VecInferKVCache`** now dispatches to the Metal kernels when
  available — zero public-API change. Existing benchmark scripts pick
  up the speedup automatically.
- **Tests**: `veloxquant_mlx/tests/cache/test_vecinfer_metal_parity.py`
  — 7 new tests covering flag resolution, shape/dtype preservation,
  reconstruction-MSE parity vs pure-MLX, no `.bits` leak, byte-account
  consistency, head_dim=256 sanity. **All 212 tests pass.**
- **Scripts** (`scripts/`):
  - `metal_quantize_proof.py` — correctness + speedup + memory benchmark.
  - `metal_dequant_proof.py` — same for the dequant kernel.
  - `metal_end_to_end_smoke.py` — `mlx_lm.generate` parity smoke test.
  - `metal_falcon3_unblock.py` — Falcon3-7B-shape sanity check.

### Notes

- Phase 2 (fused dequant+SDPA so fp16 keys are never materialized) is
  scoped but not yet implemented.
- The dequant kernel is at-parity with MLX's tuned `mx.take`; the win
  here is the quantize kernel.

## [0.5.0] — 2026-05-23

### Added — VecInfer (vector quantization with outlier-suppressing dual transform)

- **`veloxquant_mlx.allocators.vecinfer`** — algorithmic primitives for
  VecInfer (arxiv:2510.06175, Yao et al. 2025):
  - `calibrate_smooth_factors(keys)` → per-(head, channel) `lambda_i = sqrt(max|K_i|)`.
  - `walsh_hadamard_matrix(d)` → orthonormal rotation; `d` must be power-of-2.
  - `apply_dual_transform_keys / queries` → preserve `q @ K.T` under
    smooth + Hadamard (Eq. 7), with GQA fallback when smooth was
    calibrated on more heads than the cache stores.
  - `train_codebook`, `quantize_vq`, `dequantize_vq` → product VQ with a
    pure-numpy Lloyd's k-means.
  - `compute_query_lut` → optional fused-score fast path.
- **`veloxquant_mlx.cache.vecinfer_cache.VecInferKVCache`** — mlx_lm
  `update_and_fetch` wrapper that quantizes and immediately dequantizes
  keys/values so downstream SDPA sees standard fp16 tensors. Tracks
  `compressed_key_bytes`, `fp16_key_bytes`, `codebook_bytes`,
  `assigned_avg_bits`. Selectable via `KVCacheConfig(method="vecinfer", ...)`.
- **Benchmarks**: 8× key compression at 2-bit, 16× at 1-bit on
  Llama-3.2-1B/3B-Instruct-4bit. Plots and `results.json` under
  `figures/vecinfer/<model>/`. Run:
  `PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py --model <hf-id>`
- **Tradeoff**: throughput drops vs fp16 (the paper's CUDA kernel fusion
  is not portable to Metal). The win on Apple Silicon is memory
  compression, not raw speed.
- 18 new tests (`tests/allocators/test_vecinfer.py`,
  `tests/cache/test_vecinfer_cache.py`).

---

## [0.3.6] — 2026-05-17

### Breaking Change — Package namespace renamed

- **`mlx_kv_quant` → `veloxquant_mlx`**: The Python import namespace now
  matches the PyPI distribution name `VeloxQuant-MLX`. All imports must be
  updated: `from mlx_kv_quant import ...` → `from veloxquant_mlx import ...`.
  No backward-compatibility shim is provided; this is a clean break at pre-1.0.

---

## [0.3.5] — 2026-05-16

### Added — RateQuant becomes a first-class library feature

- **`veloxquant_mlx.allocators.allocate_bits_ratequant`** — RateQuant Theorem 2
  closed-form reverse-waterfilling allocator (arxiv:2605.06675). Given a list
  of per-layer sensitivities and a fractional `target_avg_bits`, returns an
  integer-valued list of bit-widths whose mean exactly matches the target.
  Defaults match the paper's RVQ-fitted β=3.5; configurable per quantizer.
- **`veloxquant_mlx.allocators.calibrate_layer_sensitivities`** — one-pass
  activation-norm probe. Runs 8 default calibration prompts (overridable),
  collects per-token squared key L2 norm via a transparent KV-cache subclass.
  Returns one float per attention layer; ratios above ~2× indicate
  RateQuant will deliver measurable gains.
- **`veloxquant_mlx.allocators.fit_distortion_curve`** — least-squares fit of
  `D(b) = α·β^(-b)` on synthetic unit-norm Gaussian keys. Use this if
  adapting the allocator to a different quantizer family (paper reports
  β≈5.0 for KIVI/QuaRot vs 3.5 for TurboQuant).
- **`KVCacheConfig.bit_width_inlier`** now accepts `int | list[int]`.
  When a list is supplied, `KVCacheBuilder.for_model(model, config)` consumes
  element `i` for layer `i`. Length mismatch raises `QuantizerConfigError`.
  `KVCacheFactory.create()` continues to require an int (the list path
  dispatches through `for_model` to per-layer factory calls).
- **`veloxquant_mlx.cache.turboquant_rvq_cache.TurboQuantRVQKVCache`** —
  library-grade mlx_lm-compatible cache wrapper around `TurboQuantRVQ`.
  Exposes `compressed_key_bytes`, `fp16_key_bytes`, and `assigned_bits`
  (never `bits` — that name collides with mlx_lm's quantized-SDPA dispatch).
- **`veloxquant_mlx.observers.KeyNormObserver`** and `KeyNormReport` —
  event-driven observer that accumulates per-token key L2 norm² and reports
  mean / min / max plus a `heterogeneity_ratio` property (predicts RateQuant
  benefit).
- **`turboquant_rvq` registered** in `KVCacheFactory.create()` — users can
  now configure RVQ via `method="turboquant_rvq"` in `KVCacheConfig` without
  manually constructing the cache class.
- **27 new tests** across `tests/allocators/`, `tests/observers/`, and
  `tests/cache/test_turboquant_rvq_cache.py`. Full suite: 187 passing.

### Changed
- `KVCacheBuilder.with_bit_width(inlier=...)` now accepts a list for
  per-layer RateQuant allocations.
- Top-level package re-exports `allocate_bits_ratequant`,
  `calibrate_layer_sensitivities`, `fit_distortion_curve`,
  `KeyNormObserver`, and `KeyNormReport`.
- `pyproject.toml`: version 0.3.5; added `maintainers`, `Author`, `Changelog`,
  `Documentation` URLs so PyPI displays attribution cleanly.

### Results (RateQuant V2 trial — 2 models on Apple M4 24 GB)

| Model | fp16 | RVQ 1-bit | **RVQ + RateQuant V2** (b̄=1.5) | sensitivity ratio |
|---|---|---|---|---|
| Falcon3 7B | 22.9 | 23.1 | **22.8 (100%)** at 5.22× | 6.48× |
| Gemma3 4B | 39.8 | 37.8 | **36.3 (91%)** at 5.22× | 14.39× |

> Per-layer bit allocations from 1.6s real-activation calibration:
> Falcon3 = 14/14 (b=2/b=1); Gemma3 = 3/11/20 (b=3/b=2/b=1).
> Source figures: [`figures/2026-05-16/`](figures/2026-05-16/).

### Known limitations vs paper
- **Per-head granularity** not implemented (paper: L×H groups, ours: L).
  mlx_lm's cache is per-layer; adding per-head requires splitting the cache
  layout. Estimated gain left on the table: ~30% of the paper's headline
  improvement.
- **Gradient-based sensitivity** not implemented (paper uses gradient,
  notes activation is ~1 PPL worse but both beat uniform). Gradient requires
  backprop through `mlx_lm.generate`, which is not currently practical.
- **K/V separate budgets** not implemented (paper's biggest single fix on
  KIVI). Our cache currently only quantizes keys; values pass through fp16.

## [0.3.4] — 2026-05-15

### Added
- **`OutlierTokenRVQMLXKVCache`** (arxiv:2505.10938, ACL 2025) — RVQ 1-bit
  cache that routes high-L2-norm "sink" tokens through an fp16 side buffer
  at prefill. Vectorized mask-blend implementation (no scatter) keeps decode
  S=1 overhead-free. Catches 0.05–0.09% of tokens on Phi-4, Qwen3, Llama,
  Gemma3 — exactly the sink-token pattern the paper predicts.
- **`RateQuantRVQMLXKVCache`** (arxiv:2605.06675) — per-layer integer bit
  allocation via reverse-waterfilling on a fitted distortion curve
  D(b) = α·β^(-b). Computed once at construction, zero inference overhead.
  Uses `.assigned_bits` (not `.bits`) to avoid triggering mlx_lm's quantized
  SDPA path that expects a different cache layout.
- **`benchmark_scripts/outlier_ratequant_core.py`** — 4-config figure
  pipeline (fp16, RVQ 1-bit, RVQ 1-bit + Outlier, RVQ + RateQuant) with
  a dedicated palette and the same 6-PNG layout as `_generate_figures_v3`.
- **`benchmark_scripts/run_outlier_ratequant.py`** — 8-model × 4-config
  benchmark runner with subprocess isolation. Outputs to
  `figures/outlier_token_ratequant/<model>/`.
- **`docs/MEMORY_CONSTRAINT_FINDINGS.md`** — documents the Qwen2.5-32B
  memory-headroom constraint on 24 GB Apple M4 and the watchdog mechanism
  added to protect the GPU from OOM-driven kernel events.
- **`.github/workflows/copyright-watch.yml`** — weekly GitHub Actions job
  that searches the public code index for distinctive class names
  (TurboQuantRVQMLXKVCache, OutlierTokenRVQMLXKVCache, etc.) and fails
  the workflow on any hit, triggering an email per GitHub notification
  settings.
- **`NOTICE`** — explicit attribution-requirements notice that strengthens
  the MIT license terms for DMCA purposes.

### Results (OTRQ sweep, 7 of 8 models, Apple M4 24 GB)

Outlier-Token RVQ matches or **beats fp16 throughput** on 5 of 7 models at
7.5× compression:

| Model | fp16 | RVQ 1-bit | RVQ 1-bit + Outlier | vs fp16 |
|---|---|---|---|---|
| Mistral 7B | 21.4 | 21.9 | **22.2** | **104%** |
| Phi-4 | 10.3 | 9.1 | **11.3** | **110%** |
| Qwen3 4B | 38.9 | 34.7 (187 tok) | **35.7 (196 tok)** | 92% + better completeness |
| Qwen3 8B | 19.6 | 17.1 | **20.3** | **104%** |
| Gemma3 4B | 35.9 | 34.7 | **36.5** | **102%** |
| Llama 3.1 8B | 18.8 | 17.5 | 17.9 | 95% |
| Falcon3 7B | 23.4 | 22.5 | 21.8 | 93% |

Qwen2.5-32B-Instruct-4bit could not complete any non-fp16 OTRQ config on
24 GB unified memory — see `docs/MEMORY_CONSTRAINT_FINDINGS.md`.

### Engineering note
- **Watchdog for large-model runs**: a memory-pressure poller
  (`/tmp/memory_watchdog.sh`) terminates the benchmark process tree if
  free + inactive memory drops below 1 GB. Validated: the watchdog caught
  the Qwen2.5-32B run at 891 MB free and killed cleanly before MLX could
  fault the Metal heap.

## [0.3.3] — 2026-05-12

### Added
- **RVQ 1-bit quantizer** — `TurboQuantRVQ(b=1)` is now fully supported.
  Stage 1 is a 2-level sign quantizer ({−0.798, +0.798} Gaussian Lloyd-Max);
  stage 2 applies a 2-level Laplacian correction to the sign-quantization error.
  Achieves **cosine 0.917 / SNR +7.6 dB** at d=128 on synthetic data, and
  **201 coherent tokens at 97–98% of fp16 throughput** on Mistral 7B and Qwen3 8B.
  Per-vector storage: `ceil(d / 4) + 2` bytes → **7.5× key compression** at d=128.
  Docstring updated with supported bit-widths (b=1, 2, 3+) and expected quality.
- **`benchmark_scripts/run_full_reports.py`** — model-agnostic 8-model × 6-config
  sweep orchestrator. Spawns one fresh Python subprocess per (model, config) to
  guarantee clean MLX graph state. Outputs `figures/2026-05-12/<model>/` with the
  full 6-figure v3 report. Idempotent: skips completed models/configs unless `--force`.
- **`_generate_figures_v3` + `run_benchmark_v3_from_results`** in `benchmark_core.py`
  — v3 figure pipeline extended to 6 configs (fp16 / TQ 2-3-4-bit / RVQ 2-bit ★ /
  RVQ 1-bit ★). New RVQ-1bit ★ traces appear in all 6 figures. Original v2 functions
  left untouched.
- **`benchmark_scripts/run_text_sweep.py`** — lightweight sweep runner used for
  fp16/RVQ-1/RVQ-2/TQ-4 comparison across models; results go to `figures/updated_tests/text_sweep/`.
- **`benchmark_scripts/diagnose_vlm_key_stats.py`** — VLM key-distribution diagnostic.
  Hooks into each layer's `update_and_fetch` to capture real key tensors, then reports
  per-layer L2 norm (image vs text tokens), post-rotation kurtosis, and RVQ-2bit cosine.
  Saves histograms to `figures/updated_tests/qwen2_vl/key_stats/`.
- **`benchmark_scripts/benchmark_qwen2_vl.py`** rewritten with `--run-config` subprocess
  isolation mode. Fixes the MLX graph-reuse bug that caused 2nd+ configs to produce
  0 tokens in the same process.

### Changed
- **`_read_model_cfg()` in `benchmark_core.py`** — new helper that robustly reads
  `(head_dim, n_kv_heads, n_layers)` from any mlx_lm model, handling:
  - Standard text models (Mistral, Qwen3, Llama, Phi) via `model.args`.
  - VLM-style wrappers where `model.args.text_config` is a plain `dict` (Gemma3, Qwen2-VL).
  - GQA models (Gemma3) where `hidden_size // n_heads` gives the wrong `head_dim` —
    always uses direct `attn.head_dim` from layer inspection instead of derived formula.
- **`TurboQuantMLXKVCache` and `TurboQuantRVQMLXKVCache` `update_and_fetch`** —
  dtype-aware norm handling. Safe-norm threshold and scale factor now use `keys.dtype`
  (bfloat16 for Qwen2-VL-7B-bf16, float16 for most text models) instead of always
  casting to float16. Eliminates a redundant cast and preserves the wider exponent
  range of bfloat16 for large-norm image-patch keys.
- **`test_2bit_improvements.py`** — added RVQ b=1 synthetic check (`Extra TQ-RVQ (b=1 x2)`,
  cosine 0.9165) with assert `cosine > 0.80`.

### Fixed
- **Gemma3 `head_dim` detection** — `_read_model_cfg` previously derived `head_dim`
  as `hidden_size // num_attention_heads = 2560 // 8 = 320`, but Gemma3's actual
  per-head dimension is 256. Now reads `attn.head_dim` directly from the layer.
- **VLM benchmark prompt** — `benchmark_qwen2_vl.py` previously rejected the
  Qwen2-VL chat template (which ends with `<|im_start|>assistant\n`) and fell back
  to raw text, degrading quantized output quality. Now always uses the full chat
  template unconditionally.

### Results (v3 sweep, Apple M4 16GB, figures/2026-05-12/)

Full 6-config benchmark across 8 models (Apple M4 16GB):

| Model | fp16 tok/s | RVQ 1-bit ★ | RVQ 2-bit ★ | TQ 4-bit | RVQ 1-bit compr. | vs fp16 |
|---|---|---|---|---|---|---|
| Mistral 7B v0.3 | 23.3 | **22.2** (201 tok) | 22.5 (201) | 21.4 (201) | 7.53× | **95%** |
| Falcon3 7B | 24.0 | **23.1** (200 tok) | 22.7 (200) | 22.1 (200) | 7.76× | **96%** |
| Phi-4 | 11.9 | **11.8** (200 tok) | 11.7 (200) | 11.4 (200) | 7.53× | **99%** |
| Qwen3 4B | 40.2 | **34.3** (187 tok) | 35.0 (197) | 33.5 (199) | 7.53× | **85%** |
| Qwen3 8B | 20.5 | **21.1** (200 tok) | 20.7 (200) | 19.8 (200) | 7.53× | **103%** |
| Llama 3.1 8B | 22.0 | **21.5** (201 tok) | 20.9 (201) | 20.3 (201) | 7.53× | **98%** |
| Gemma3 4B | 32.5 | **30.5** (201 tok) | 29.2 (201) | 27.7 (201) | 7.76× | **94%** |
| Qwen2.5 32B | 3.7 | **3.9** (200 tok) | 4.2 (200) | 3.9 (200) | 7.53× | **107%** |

Notable: on Qwen3-8B, Phi-4, and Qwen2.5-32B, RVQ configs **match or exceed fp16 throughput** (all memory-bandwidth bound). At 32B scale, RVQ 2-bit achieves 4.2 tok/s vs fp16's 3.7 tok/s (114%) — the KV-cache compression benefit grows with model size. TQ single-pass 2-bit degrades severely on Qwen2.5-32B (5 tokens) and is not suitable for this model; RVQ consistently delivers full outputs across all models and bit-widths.

## [0.3.2] — 2026-05-12

### Added
- VLM support for **Qwen2-VL-7B-Instruct-bf16** via `build_vlm_caches()` and
  `KVCacheBuilder.for_model()`.
- `benchmark_scripts/benchmark_qwen2_vl.py` — VLM benchmark with image+text prompt
  capability (text-only path validated; image path requires mlx-vlm).

## [0.3.1] — 2026-05-10

### Changed
- README restructured with TOC, algorithm picker table, per-model benchmark tables,
  and throughput optimization journey. All emojis removed for plain-text rendering.
- Distribution metadata now reflects the new structure.

## [0.3.0] — 2026-05-10

### Added
- **`TurboQuantRVQ`** — two-pass Residual Vector Quantization quantizer that lifts
  2-bit cosine similarity from 0.69 → **0.98** and SNR from −0.5 dB → **13.2 dB**.
  Stage 1 uses N(0, 1/d) Lloyd-Max; stage 2 fits a Laplacian PDF on the per-coordinate
  residual. Total storage 2·b bits/dim. Registered as `turboquant_rvq` in the registry.
- **`AdaptiveScalarCodebook`** — wrapper that refits codebook centroids from observed
  post-rotation distribution after a calibration phase. Plumbed via
  `TurboQuantProd(use_adaptive_codebook=True)` and `TurboQuantProdAdaptive`.
- **Adaptive JL sketch dimension** — `TurboQuantProd.m_default(d, b)` now returns
  `d` at b ≤ 2 and `min(d, 64)` at b ≥ 3, doubling the QJL correction budget at 2-bit.
- **Optimization journey figure** — [`figures/updated_tests/optimization_journey.png`](figures/updated_tests/optimization_journey.png).
- **`OPTIMIZATION_FINDINGS.md`** — full writeup of bottleneck analysis and four-stage
  speedup attribution.
- **`benchmark_mistral7b_v2.py`** and **`benchmark_qwen3_4b_v2.py`** — 5-config v2
  benchmark scripts that include `TurboQuantRVQMLXKVCache` alongside the existing
  fp16/2/3/4-bit configurations.
- **`test_2bit_improvements.py`** — synthetic validation script with asserts for all
  three 2-bit accuracy improvements.

### Changed
- **Throughput parity with fp16** for quantized configs on memory-bound models:
  Mistral 7B RVQ 2-bit at 22.3 tok/s vs fp16 22.1 tok/s. Qwen3 4B RVQ 2-bit at
  36.0 tok/s vs fp16 39.2 tok/s (92% of fp16). Achieved via four sequential changes:
  1. Single shared quantizer with `(B·H·S, D)` flat batching (eliminates per-head Python loop).
  2. Hadamard rotation by default in benchmark wrappers (`use_hadamard=True`).
  3. Boundary-sum `quantize()` in `ScalarCodebook` (replaces broadcast-argmin).
  4. Dropped redundant fp32 ↔ fp16 casts in `update_and_fetch`.
- `ScalarCodebook.__init__` now sorts centroids and precomputes Voronoi boundaries
  in `self._boundaries_mx`. `quantize()` returns 100% index-match output vs the prior
  argmin path.
- `TurboQuantMLXKVCache` and `TurboQuantRVQMLXKVCache` in `benchmark_core.py` use a
  single shared quantizer instance instead of `n_kv_heads` separate ones.

### Performance
- Mistral 7B RVQ 2-bit: **17.7 → 22.3 tok/s** (+26%).
- Qwen3 4B RVQ 2-bit: **24.8 → 36.0 tok/s** (+45%).
- Boundary-sum quantize verified bitwise-identical to broadcast-argmin (100.00% index match on synthetic test).

### Quality
- RVQ 2-bit synthetic cosine **0.9766** preserved through every optimization step.
- Real-model output completeness preserved at every step:
  - Mistral 7B: 201/201 tokens across all 5 configs.
  - Qwen3 4B `<think>` mode: 199/200 tokens for RVQ 2-bit (vs 50/200 for single-pass 4-bit).

## [0.2.0] — 2025-05-07

### Added
- Published to PyPI as `VeloxQuant-MLX`
- `veloxquant` CLI entry point (alias for `mlx-kv-quant`)
- 2-bit quantization support in benchmark suite (11.6× compression ratio)
- Per-model benchmark scripts: Falcon3-7B, Mistral-7B, Qwen3-4B, Qwen3-8B, Qwen2.5-32B, Gemma-4, Phi-4
- `benchmark_core.py` unified benchmark runner with 6-figure report generation
- Validated across 7 models: near-lossless at 3-bit and 4-bit; 2-bit degrades gracefully

### Changed
- Package distribution name renamed from `mlx-kv-quant` → `VeloxQuant-MLX`
- Status classifier updated from Alpha → Beta

## [0.1.0] — 2025-04-01

### Added
- Initial implementation of TurboQuant KV cache quantization for Apple Silicon MLX
- PolarQuant and QJL algorithms
- Chain-of-Responsibility quantization pipeline
- Lloyd-Max scalar codebooks
- Random orthogonal rotation preconditioner
- Builder pattern (`KVCacheBuilder`) for fluent cache construction
- Observer framework (latency, memory, distortion)
- Precompute CLI for offline codebook generation
- Full test suite

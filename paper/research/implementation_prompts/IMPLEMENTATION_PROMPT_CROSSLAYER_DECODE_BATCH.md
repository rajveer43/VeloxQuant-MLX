# Implementation Prompt — Cross-Layer Batched Decode-Attend Dispatch (issue #307 part 1)

Execute-cold spec for a **kernel/perf** change, not a new quantization
method: no new algorithm, no README method-count bump, no landing-page
card. This closes the one lever
`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` names as unblocked and *not yet
attempted* — read that document and its two addenda in full before writing
any code. Do not re-derive its conclusions from scratch; cite them.

## The one-sentence goal, and the one-sentence non-goal

**Goal:** raise the threadgroup count `scalar_fused_decode_attend` dispatches
per Metal kernel launch, at real single-token decode shapes, by batching the
*independent* per-layer decode-attend calls a transformer forward pass
already makes into one dispatch — without changing the math, the memory
layout, or the per-(b, h_kv, sq) work each threadgroup does.

**Non-goal:** do not attempt to beat `mx.fast.scaled_dot_product_attention`
on prefill, do not attempt GQA head-packing again (already measured
2.7-4.7x slower, `docs/KV_KERNEL_ROOFLINE_FINDINGS.md` "Addendum: GQA
head-packing"), do not attempt `simd_shuffle` cross-head sharing (already
ruled out architecturally, same doc, "Addendum: SIMD-shuffle spike"). Both
are closed questions in this repo, re-litigating them without new evidence
is out of scope.

---

## Why this, precisely (repeat the honest framing on every surface)

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md`'s own occupancy sweep is the entire
justification:

| B | H | S_q | threadgroups | latency | GB/s | % peak |
|---|---|---|---|---|---|---|
| 1 | 8 | 1 | 8 | 6.95 ms | 6.0 | 6.1% |
| 1 | 32 | 1 | 32 | 7.74 ms | 21.7 | 21.8% |
| 4 | 32 | 1 | 128 | 18.4 ms | 36.5 | 36.7% |

Bandwidth utilization scales with **threadgroup count**, not bytes moved —
confirmed directly, not inferred. A real decode step for one request
dispatches `scalar_fused_decode_attend` once **per transformer layer**
(typically 28-80 layers in current open-weight models), each call
independently launching only `B*H_kv*S_q` threadgroups (8-32 in the table
above). Those per-layer calls are mutually independent — layer *L*'s
attention output isn't needed to *compute* layer *L+1*'s decode-attend
inputs at the Metal-dispatch level within one already-materialized set of
per-layer K/V caches and post-attention Q projections — so nothing about
attention's data dependencies forces them to be 28-80 separate kernel
launches. Batching them into one dispatch multiplies the threadgroup count
by the layer count for free, which is exactly the axis the occupancy sweep
shows moves the needle.

**State this precisely, because it is easy to overclaim:** this does NOT
make any single layer's attention compute cheaper, and it does NOT reduce
total bytes moved (same K/V bytes, same FLOPs, summed over layers either
way). It only changes *how many independently-schedulable units of that
already-fixed work* one kernel launch hands the GPU scheduler at once. If
the occupancy hypothesis in the roofline doc is right, wall-clock time for
the same total work should drop because the GPU's cores are no longer
sitting idle between 28-80 tiny sequential launches; if the hypothesis is
wrong, this experiment will show a null or negative result, and **that
must be reported as plainly as issue #307 part 2 and #308 were** — this
repo's standing convention (see both addenda) is to ship a negative,
well-instrumented result rather than hide or discard it.

---

## What "batched across layers" means concretely

Today, `scalar_fused_decode_attend` (`veloxquant_mlx/metal/_scalar_attend.py:207`)
takes single-layer tensors: `q [B,H,S_q,D]`, `k_codes/v_codes [B,H_kv,S_kv,D]`
+ their per-group scale/zero arrays, and dispatches
`grid=(B*H_kv*S_q*32, nsg, 1)` (`_scalar_attend.py:295-316`), with the Metal
kernel (`veloxquant_mlx/metal/src/scalar_affine_attend.metal:30-34`) decoding
`(b_idx, hkv_idx, sq_idx)` out of `threadgroup_position_in_grid.x` alone.

The batched version adds a **new, outermost grid axis for layer index**
`L`, so one dispatch covers all layers of one decode step:

- New Python entry point `scalar_fused_decode_attend_batched` in
  `veloxquant_mlx/metal/_scalar_attend.py`, alongside (not replacing) the
  existing single-layer function — the single-layer path stays for callers
  that don't have layer-stacked tensors (tests, the two-pass spike, any
  caller processing one layer at a time).
- Inputs gain a leading `num_layers` (`NL`) dimension: `q [NL,B,H,S_q,D]`,
  `k_codes/v_codes [NL,B,H_kv,S_kv,D]`, `k_scale/k_zero
  [NL,B,H_kv,GK,D]`, `v_scale/v_zero [NL,B,H_kv,S_kv,GV]`. This assumes the
  caller has already stacked (`mx.stack`) each layer's per-layer arrays
  along a new axis 0 — **this kernel does not itself gather scattered
  per-layer Python objects into one buffer; that stacking is the caller's
  responsibility and its cost must be measured, not assumed free (see
  Phase 3).**
- `S_kv` may legitimately differ per layer in this repo (e.g. per-layer
  eviction budgets, `PyramidKV`-style pyramid schedules) — **decide and
  document explicitly** whether v1 requires uniform `S_kv` across the
  batched layers (simplest: pad the ragged case at the call site before
  stacking, document the padding waste) or supports a per-layer `S_kv`
  array read inside the kernel (`k_codes_shape` becomes per-layer, more
  kernel complexity, no padding waste). Recommend starting with the
  uniform-`S_kv` case as v1 and stating the ragged case as explicit future
  work — do not silently assume one without saying so.
- New grid: `grid=(NL * B * H_kv * S_q * 32, nsg, 1)`,
  `threadgroup=(32, nsg, 1)` — threadgroup count becomes `NL * B * H_kv *
  S_q`, e.g. `32 * 1 * 8 * 1 = 256` for a 32-layer model at the `B=1,H_kv=8`
  decode shape the roofline doc measured at 8 threadgroups today.
- Kernel indexing (`scalar_affine_attend.metal:30-34`) gains one more
  division/modulo to peel off the layer index from `tg`, in the same style
  as the existing `sq_idx`/`hkv_idx`/`b_idx` derivation:
  ```
  uint sq_idx  = tg % S_q;
  uint hkv_idx = (tg / S_q) % H_kv;
  uint b_idx   = (tg / (S_q * H_kv)) % B;
  uint l_idx   = tg / (S_q * H_kv * B);
  ```
  and every buffer offset that currently uses `bh_kv = b_idx * H_kv +
  hkv_idx` (lines 34, 57-58, 88-89) and every `q_off`/`out_off` computation
  (lines 63, 134) needs the added `l_idx * (B * H_kv * ...)` /
  `l_idx * (B * H_q * ...)` leading stride term for the new outer axis.
  **Read the full existing kernel file
  (`veloxquant_mlx/metal/src/scalar_affine_attend.metal`) before editing —
  every offset expression that touches `bh_kv`, `q`, or `out` needs the new
  term, not just the ones listed above; do not assume this list is
  exhaustive without re-checking against the current file.**
- Output shape: `[NL, B, H_q, S_q, D]` fp16.
- Threadgroup-memory budget check (`_scalar_attend.py:283-293`,
  `_TG_MEM_BUDGET_BYTES = 32768`) is unaffected — batching happens on the
  grid axis, not inside one threadgroup's per-lane state — but re-verify
  this explicitly with a test rather than asserting it from the diff alone.
- Put the new kernel source in a **new file**
  `veloxquant_mlx/metal/src/scalar_affine_attend_batched.metal` rather than
  branching the existing file on a compile-time flag — this repo's
  convention (see `scalar_affine_decode_once.metal` /
  `scalar_predecoded_attend.metal` as siblings of
  `scalar_affine_attend.metal`) is one file per dispatch shape, selected by
  a distinct Python kernel-factory function
  (`_scalar_affine_attend_batched_kernel`, mirroring
  `_scalar_affine_attend_kernel` at `_scalar_attend.py:139-160`), not a
  runtime/`#ifdef` branch inside one mega-kernel.

---

## Phase 1 — Kernel + Python wrapper

`veloxquant_mlx/metal/src/scalar_affine_attend_batched.metal` +
`veloxquant_mlx/metal/_scalar_attend.py`:

- `scalar_fused_decode_attend_batched(q, k_codes, k_scale, k_zero, v_codes,
  v_scale, v_zero, group_size, scale, nsg=4) -> mx.array`, full docstring
  matching the discipline of the existing `scalar_fused_decode_attend`
  docstring (`_scalar_attend.py:219-246`) — document the layer-stacking
  precondition explicitly (caller must pre-stack; kernel does not gather).
- All the same shape/value guards as the single-layer function
  (`_scalar_attend.py:247-293`) plus: `NL >= 1`, every per-layer array's
  leading dimension equals `NL` and matches across `q`/`k_codes`/`v_codes`
  and their scale/zero siblings, `D <= 256`, `1 <= nsg <= 32`,
  `heads_per_kv <= _MAX_HEADS_PER_KV`, threadgroup-memory budget — reuse
  `_MAX_HEADS_PER_KV` and `_TG_MEM_BUDGET_BYTES` as already defined
  (`_scalar_attend.py:194-199`), do not redefine.
- GQA support carries through unchanged (H_q/H_kv ratio logic is orthogonal
  to the new layer axis) — do not special-case MHA-only for v1 just
  because it's simpler; the existing kernel already handles both uniformly
  and the batched one should too.
- Export from `veloxquant_mlx/metal/kernels.py` alongside the existing
  `scalar_fused_decode_attend` (kernels.py:83,108).

## Phase 2 — Correctness tests

`veloxquant_mlx/tests/metal/test_scalar_attend.py` (extend, matching the
existing GQA parity test style in this file, ~15 new tests):

- **Parity against the single-layer kernel called in a Python loop**: for
  `NL in {1, 4, 32}`, stack `NL` independently-random layers' inputs, run
  `scalar_fused_decode_attend_batched` once vs. calling
  `scalar_fused_decode_attend` `NL` times in a loop and stacking the
  outputs — assert `allclose` (same tolerance the existing GQA parity
  tests use). This is the load-bearing correctness test: it proves the new
  grid axis is pure batching, not a semantic change.
  - "existing" parity (see #307 addendum: `H_q == H_kv` produces
    bit-identical output vs. non-GQA path) must still hold with `NL > 1`
    layered on top.
- Parity against a numpy reference across `NL in {1, 8}` and the same
  `H_q/H_kv` ratios the existing suite already covers
  (`(8,2),(32,4),(32,8)`, plus plain MHA).
- Shape/divisibility/threadgroup-memory-budget guard tests (mirror the
  existing guard tests for the single-layer kernel one-for-one, with the
  new `NL` dimension added).
- Threadgroup-memory-budget test explicitly confirming the budget check is
  unaffected by `NL` (construct a case that would fail the budget at fixed
  `nsg`/`heads_per_kv` regardless of `NL`, and one with large `NL` that
  still passes — proves the budget scales with grid axis count, not
  per-threadgroup memory).
- `NL=1` degenerates to calling the single-layer path with a size-1 leading
  axis — bit-identical output (same style as the existing "H_q==H_kv
  degenerates to non-GQA" pin).
- Batched `B > 1` case (exercise the new `l_idx`/`b_idx` combined indexing
  — a deliberately adversarial test that would catch a swapped stride
  order between the two, e.g. `NL=3, B=2` with per-(layer,batch) distinct
  random data, checked position-by-position against the loop reference).
- Non-uniform-content-but-uniform-shape sanity: two layers with
  deliberately different `S_kv` *content* (not shape, if v1 requires
  uniform `S_kv` per the earlier decision) still each get their own
  layer's K/V, not layer 0's broadcast across all layers (a copy-paste
  stride bug would silently pass a same-content test but fail this one).

## Phase 3 — Benchmark (this is the actual point of the exercise)

New `benchmark_scripts/benchmark_crosslayer_decode_batch.py` +
`scripts/kv_kernel_roofline_bench.py` extension (this repo's existing
roofline harness, per `docs/KV_KERNEL_ROOFLINE_FINDINGS.md`'s
"Reproduction" section — reuse its self-calibrating bandwidth-peak
methodology, do not build a separate ad hoc timing script):

- **Primary comparison**: `NL` sequential single-layer
  `scalar_fused_decode_attend` calls (today's actual behavior — this is
  the real baseline, not a strawman) vs. one
  `scalar_fused_decode_attend_batched` call over the same `NL` layers'
  data, at realistic decode shapes: `B=1, H_kv in {2,4,8}, H_q/H_kv in
  {1,4,8}, S_kv in {128, 2048, 16384}, NL in {28, 32, 48, 80}` (span
  current open-weight model depths — cite actual layer counts, don't
  invent round numbers).
- **Report bandwidth utilization against the same calibrated peak this
  repo already measured** (~97-99 GB/s on the M4 used throughout
  `KV_KERNEL_ROOFLINE_FINDINGS.md` — recalibrate live on whatever machine
  runs this, per that doc's own "never trust an unverified number"
  principle; do not hardcode the M4 figure as if it applies to the actual
  test machine).
- **Measure and report the stacking cost separately, not folded into the
  kernel's own number**: `mx.stack` over `NL` per-layer arrays has a real
  cost (a gather/copy) that a real integration would pay somewhere — time
  it standalone and report it as its own line item, so the batched
  kernel's win/loss isn't accidentally flattering itself by ignoring the
  cost of producing its own input layout. If in a real `mlx_lm` decode
  loop the per-layer K/V/Q arrays are never naturally contiguous in a
  layer-stacked buffer to begin with (verify this against how `mlx_lm`'s
  `KVCache` actually stores per-layer state before assuming either way),
  say so plainly — this determines whether the stacking cost is a one-time
  layout change or a per-step tax, which materially changes whether this
  technique is a net win end-to-end.
- Determinism: non-timing fields identical across two runs (repo
  convention, see CurDKV/H2O benchmark determinism checks).
- **This benchmark's output IS the deliverable finding, not a side
  artifact.** Whatever it shows — a clean win, a null result, a win that's
  erased by stacking overhead, a partial win only past some `NL` threshold
  — write it up with the same rigor as the #307/#308 addenda: numbers,
  the mechanism explanation for *why*, and an explicit statement of what
  regime (if any) it's recommended for.

## Phase 4 — Documentation (the honest-findings surface, not README/landing)

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` — add a **third addendum**, same
format and rigor as the GQA head-packing and SIMD-shuffle addenda already
there:

- State the hypothesis being tested (threadgroup count is the lever;
  cross-layer batching raises it "for free" with no compute/bandwidth
  cost) up front, citing Recommendation #2 from this same document as the
  origin of the hypothesis.
- Report the Phase 3 measurements in full, including the stacking-cost
  line item and the "does mlx_lm's KVCache naturally produce a
  layer-stacked buffer" finding.
- If positive: state the regime it wins in (which `NL`/`S_kv`/`H_kv`
  combinations), and explicitly flag what integration work remains
  unaddressed by this prompt (wiring this into a real `mlx_lm` decode loop
  end-to-end is a separate, larger follow-up — mirror how the #307/#308
  addenda flagged real-model integration as future work rather than
  silently implying it's done).
- If null or negative: state why as precisely as the GQA head-packing
  addendum did (e.g. "stacking cost exceeds the occupancy win at these
  shapes because X", not just "didn't help") — a well-instrumented null
  result is this repo's explicit standing bar (see both existing
  addenda), not a failure to hide.
- Do NOT claim this "outperforms" any other kernel in this repo or
  elsewhere in absolute terms — the only valid claim is a same-total-work,
  before/after threadgroup-count comparison against this repo's own
  existing single-layer dispatch, exactly as rigorous and exactly as
  scoped as the two existing addenda.
- Cross-reference from `blogs/prefill-roofline.md`'s "What decode kernels
  do that prefill can't reuse" framing if relevant (that post already
  distinguishes decode's bandwidth story from prefill's compute story —
  this addendum refines the decode side further, it doesn't change that
  distinction).

## Phase 5 — Verify

- Full pytest: confirm the new tests pass and zero regressions in the
  existing `test_scalar_attend.py` suite (43+10 tests documented in the
  #307/#308 commit message, now plus Phase 2's ~15).
- Benchmark determinism (two runs, non-`_ms` fields identical).
- Re-run the existing occupancy sweep from
  `docs/KV_KERNEL_ROOFLINE_FINDINGS.md` alongside the new batched numbers
  on the same run, so the "before" and "after" rows are directly
  comparable (same machine, same session, not numbers from different
  historical runs with their own variance).

---

## What we do NOT implement (state plainly, mirroring this repo's convention)

- GQA head-packing combined with cross-layer batching as a *combined*
  optimization — the roofline doc's own conclusion is that packing "should
  be revisited only after occupancy is fixed" by something like this
  change; that revisit is future work, not scoped here. Ship cross-layer
  batching alone first and measure it alone.
- Any change to `scalar_decode_once` / `scalar_predecoded_attend` (the
  #308 two-pass spike) — orthogonal, already-shipped, already-documented
  as not adopted for long-context targets.
- Any real `mlx_lm` end-to-end integration (monkeypatch dispatcher, actual
  decode-loop wiring) — Phase 3 measures the kernel and the stacking cost
  in isolation and Phase 4 explicitly flags integration as unaddressed
  future work, matching how `fused_sdpa`'s real-model integration was
  measured and separately documented rather than assumed.
- Prefill. This entire prompt is decode-shape-only, per the roofline doc's
  own regime split (decode: bandwidth/occupancy story; prefill: compute
  story, already separately investigated and closed against in
  `blogs/prefill-roofline.md`).
- Ragged per-layer `S_kv` support — v1 requires uniform `S_kv` across
  batched layers (state this limitation explicitly per Phase 1); a
  per-layer `S_kv` array is named as explicit future work, not implemented
  here.
- Any claim, anywhere, that this kernel "outperforms" MLX's SDPA, any
  other library's kernel, or any other kernel in this repo — the only
  claim in scope is a same-work, before/after threadgroup-count comparison
  against this repo's own existing single-layer dispatch shape.

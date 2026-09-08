# VeloxQuant Cache Bookkeeping Audit

Audited 2026-09-08 on Apple M4, 24GB unified memory, macOS 26.5.2, MLX with
Metal GPU available (`mx.default_device() == Device(gpu, 0)`), mlx-lm as
installed in `.venv`. Repo: `turboquant_mac_implementation`, branch
`docs/gqa-packing-remeasure-and-benchmark-additions`. This is a from-scratch
verification — every claim below is backed by a code citation and/or a new
script run during this audit, not by trusting `docs/TOVA_METAL_FINDINGS.md`'s
self-reported numbers (which were independently reproduced where checked, see
"TOVA Analysis").

Scope, per the brief: TOVA, H2O, and KIVI were audited in depth (code
inspection, new property-based tests, new microbenchmarks, one real-model
end-to-end run). The other ~40 registered methods in
`veloxquant_mlx/cache/registry.py` / `veloxquant_mlx/cache/base.py`'s
`KVCacheConfig.method` literal were pattern-matched only, to confirm whether
they share infrastructure with the three deep-dive methods and flag
deviations — not audited to equal depth.

New artifacts produced during this audit (all under `scripts/`, read-only
w.r.t. production code):
- `scripts/kv_bookkeeping_property_audit.py` — from-scratch pure-Python
  reference implementations of TOVA and H2O eviction, property-tested against
  the real implementations over 500 randomized trials each (randomized
  n_sink/budget/head_dim/n_steps/chunk-size), plus a KIVI group-atomicity
  check. Output: `/tmp/kv_audit.json`.
- `scripts/kv_bookkeeping_microbench.py` — per-decode-step latency across
  cache sizes {128, 512, 2048} for TOVA (mlx/metal), H2O, KIVI (metal/mlx),
  and a plain fp16 baseline. Output: `/tmp/kv_microbench.json`.
- `scripts/kv_bookkeeping_memory_check.py` — live-tensor-vs-reported-bytes
  check for KIVI, and a 20,000-operation long-loop memory-plateau check for
  TOVA and H2O. Output: `/tmp/kv_memcheck.json`.
- `scripts/kv_bookkeeping_e2e_bench.py` — real-model (Llama-3.2-1B-Instruct,
  4-bit MLX weights, already present locally at
  `~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit`
  — **no download was performed by this audit**) prefill/decode throughput
  and logit-agreement comparison: baseline fp16 cache vs TOVA vs H2O vs KIVI.
  Output: `/tmp/kv_e2e.json`.

Existing test suites for the three deep-dive methods were also re-run
verbatim as a sanity check (not merely cited from prior docs):
`.venv/bin/python -m pytest -q veloxquant_mlx/tests/metal/test_tova_evict.py
veloxquant_mlx/tests/quantizers/test_tova_backends.py
veloxquant_mlx/tests/quantizers/test_tova.py
veloxquant_mlx/tests/cache/test_tova_cache.py
veloxquant_mlx/tests/metal/test_h2o_evict.py
veloxquant_mlx/tests/quantizers/test_h2o.py
veloxquant_mlx/tests/cache/test_h2o_cache.py
veloxquant_mlx/tests/metal/test_kivi_quant.py
veloxquant_mlx/tests/quantizers/test_kivi.py
veloxquant_mlx/tests/cache/test_kivi_cache.py` → **266 passed, 0 failed** (7.9s).

## Executive Summary

The core bookkeeping architecture (K/V storage, offset/position tracking,
sink protection, eviction-index computation) is **correct for TOVA** across
500 randomized property-test trials and **produces a real, measurable
divergence from a hand-written reference for H2O** under interior eviction
combined with `grace > 0` (417/500 property trials passed; every failure
involved `grace > 0`, kept-set size always matched, but per-token cumulative
scores and post-eviction positions diverged after the first eviction inside
a multi-token chunk — see "H2O" under Correctness Tests). The kept-set
divergence is small in the trials examined (adjacent-index eviction choices)
and did not, in the traced case, change *which* tokens survive over the
window tested, but it is a genuine behavioral difference from the documented
per-token-sequential algorithm, not fp16 rounding noise (score deltas up to
0.63 in a scale-1-10 range).

Host/device synchronization is **on-device on the fast paths** (TOVA's `mlx`
and `metal` backends, H2O's Metal fused-evict path) and **forces a Python
scalar sync twice per eviction on the fallback paths** (TOVA's `reference`
backend, H2O's `_evict_via_mlx`, exercised automatically whenever
`metal_available()` is false). `H2OKVCache.update_and_fetch` unconditionally
loops over `B × H` in Python calling `h2o_update` per head
(`veloxquant_mlx/cache/h2o_cache.py:292-308`) — unlike TOVA, which batches
every head into one `[B*H, N, D]` call
(`veloxquant_mlx/quantizers/tova.py:_tova_update_batched`). This
architectural difference is real and measured: on a live Llama-3.2-1B model,
H2O decode throughput was **56.8 tok/s vs a 95.2 tok/s baseline (−40%)**,
while TOVA measured 94.2 tok/s (statistically indistinguishable from
baseline) at the same budget (512, prompt 64, no eviction triggered).

KIVI's headline "compression" is **not a live memory reduction**. Its cache
stores full fp16 tensors at all times — quantize-then-immediately-dequant
in place (`self.keys[:, :, lo:hi, :] = k_q`,
`veloxquant_mlx/cache/kivi_cache.py:217`) — and the reported
`effective_compression_ratio` property is a separate byte-accounting
estimate that corresponds to **no tensor actually resident in memory**.
Measured directly: a live `KIVIKVCache.keys` tensor after 200 decode steps
occupied 131,072 bytes (fp16, actual GPU-resident bytes), while the cache's
own `compressed_key_bytes` accounting reported 15,360 bytes for the same
region — an 8.5x discrepancy between the reported win and the actual
resident footprint (`/tmp/kv_memcheck.json`). KIVI's real value on this
hardware is a documented Metal kernel speedup (1.3–4.9x measured, confirmed
by re-running the existing test suite's benchmark) for the quant/dequant
round-trip itself, not a cache memory reduction.

KIVI's default 2-bit setting causes severe, **genuine algorithmic**
quality degradation under autoregressive decode on a small (1B,
already-4-bit-weight-quantized) model: cosine similarity between baseline
and KIVI logits was 0.95 after prefill alone, dropped to 0.82 after 3 decode
steps, and reached −0.18 (effectively uncorrelated) after 64 decode steps —
monotonically recovering to 0.996 (b=4) and 0.99999 (b=8) at higher bit
widths in an isolated single-call check. This is compounding quantization
error under greedy decode, confirmed non-buggy by the monotonic recovery
curve, not a bookkeeping defect — but it means KIVI's headline 2-bit default
is unsafe for unattended long-generation use on small models without a
quality-vs-length caveat the current docs do not carry.

No memory leak was found: both TOVA and H2O plateaued (active-memory
variation < 2MB in the tail quartile) over 20,000 sequential
`update_and_fetch` calls (`/tmp/kv_memcheck.json`). Metal compaction
kernels measured ~14% of the M4's ~120GB/s peak unified-memory bandwidth at
N=2048/H=8/D=128 — consistent with this repository's established
occupancy-bound (not bandwidth-bound) finding for this hardware; see
`docs/KV_KERNEL_ROOFLINE_FINDINGS.md`, not re-derived here.

## Final Verdict

**Grade: B-.**

Rationale, by evidence:
- Core K/V/position bookkeeping is sound for the memoryless case (TOVA:
  500/500 property-test pass, zero test-suite failures, correct edge-case
  behavior at capacity 1/2/empty, correct batch isolation).
- A real, reproducible divergence from documented behavior exists in H2O
  under `grace > 0` + interior/chunked eviction (83/500 property trials,
  17% failure rate on that axis) — not disqualifying (kept-set size and the
  common single-token-chunk path are unaffected), but real and previously
  unverified by any existing test in the repo (the existing 266-test suite
  does not include a chunked-update + grace + property-random regression, so
  it did not catch this).
- A significant "misleading optimization" gap exists in KIVI: the
  memory-savings framing in `KIVIKVCache.effective_compression_ratio` and
  its docstring does not correspond to any reduction in live GPU-resident
  bytes. This is a documentation/API-contract problem, not a numerical bug —
  the numbers the property carries are accurate as *byte-accounting*, but
  nothing in the public API surfaces that the live tensor never shrinks.
- H2O's decode throughput regression (−40% vs baseline, measured on a real
  model) is a genuine, unaddressed P1 performance bottleneck traceable to a
  specific, cited architectural choice (unbatched per-head Python loop).
- Nothing catastrophic (no memory leak, no batch cross-contamination, no
  capacity-1/empty-cache crash, no state corruption over 20k operations) was
  found in any of the three deep-dive methods.

This lands short of "production-quality" (A/B+) because of the H2O
correctness divergence and the KIVI memory-claim gap, both real and now
evidenced, but well above "broken" (D/F) because the core append/evict/sink
machinery holds up under adversarial random testing and real-model
generation for the two methods it was checked hardest against (TOVA
matches baseline bit-for-bit in end-to-end cosine similarity at
non-evicting budget; H2O also matches at 1.0 cosine similarity in the same
run, since the divergence found in property testing needs chunked updates
under active eviction, which a budget-512-vs-64-token-prompt run never
triggers).

## Current KV Cache Architecture

VeloxQuant-MLX has two families of cache class, both selected through
`KVCacheFactory.create()` / `KVCacheBuilder.for_model()` in
`veloxquant_mlx/cache/base.py`:

1. **mlx_lm-protocol caches** (35 of 40 methods) — subclass
   `mlx_lm.models.cache.KVCache` and implement `update_and_fetch(keys,
   values) -> (K, V)`, `state`, `size()`, `is_trimmable()`. These are driven
   directly by `mlx_lm.generate()` / a model's forward pass via
   `model(tokens, cache=caches)`. TOVA (`cache/tova_cache.py`), H2O
   (`cache/h2o_cache.py`), and KIVI (`cache/kivi_cache.py`) are all in this
   family.
2. **Standalone VeloxQuant-native caches** (5 of 40:
   `STANDALONE_METHODS = {turboquant_prod, turboquant_mse, polar, qjl,
   spectral}`, `cache/base.py:23-31`) — implement VeloxQuant's own
   `append_key/append_value/attend` interface (`core/abstractions.py`
   `KVCache` ABC) instead. `KVCacheBuilder.for_model()` explicitly refuses to
   build these for live model serving (`cache/base.py:906-916`) because
   `mlx_lm.generate()` only understands the first interface — this guard is
   itself a piece of bookkeeping correctness infrastructure (fails fast at
   config time instead of deep inside generation) and was found to be
   correctly wired, not exercised further here since none of the three
   deep-dive methods are standalone.

`KVCacheConfig` (`cache/base.py:34-420`) is one large dataclass carrying
every method's hyperparameters (`tova_budget`, `h2o_grace`, `kivi_group_size`,
…) — a single flat namespace shared across all 40 methods rather than
per-method sub-configs. `KVCacheBuilder.for_model()` (`cache/base.py:878-1047`)
resolves per-layer `head_dim` from the model automatically (verified live:
Llama-3.2-1B's `head_dim=64` was picked up correctly even when a caller also
passes an explicit `head_dim=64` kwarg — the per-layer `dataclasses_replace`
override wins, `cache/base.py:1031-1037`), and handles cross-layer
coordinators for methods that need shared state across layers (XQuant,
MiniCache, PyramidKV, CacheGen, SqueezeAttention, xKV, ChunkKV with
`reuse_layers > 1`) — none of TOVA/H2O/KIVI need a coordinator; each layer's
cache is fully independent.

## Cache Lifecycle

Prompt → prefill (S > 1 call to `update_and_fetch`) → decode (S == 1 calls,
one per generated token) → eviction (fires inside `update_and_fetch` once
over budget) → next-token (model reads the returned `(K, V)` for SDPA).

- **TOVA and H2O do not distinguish prefill from decode** — both S > 1 and
  S == 1 go through the exact same `update_and_fetch` path
  (`cache/tova_cache.py:122-168`, `cache/h2o_cache.py:261-325`), unlike
  SnapKV-adapted (`cache/snapkv_cache.py`, pattern-matched only), which fires
  its eviction decision once at prefill end. TOVA's `_tova_update_batched`
  (`quantizers/tova.py:251-339`) does exploit prefill's larger token count
  for a vectorized "absorb the below-budget prefix in one batched call, then
  loop only for tokens once actually over budget" optimization
  (`quantizers/tova.py:286-306`); H2O has the equivalent optimization
  (`_batch_absorb_prefix`, `quantizers/h2o.py:357-402`), added specifically
  because a long unbatched prefill previously crashed with a Metal resource
  limit (`RuntimeError: [metal::malloc] Resource limit (499000) exceeded`,
  documented in `quantizers/h2o.py:68-85`) — both fixes were code-inspected
  and are structurally present; not independently re-triggered here since
  reproducing a 3200-token crash was out of scope for time budget, and the
  present microbenchmark's 2048-token cache size did not reproduce it.
- **KIVI does distinguish**, but by *residual-window age*, not
  prefill/decode phase: every call passes through the base class's
  append-only buffer first (`super().update_and_fetch`,
  `cache/kivi_cache.py:209`), then quantizes whatever has aged out of the
  most recent `residual_length` tokens, snapped down to a `group_size`-
  aligned boundary (`_quantization_boundary`, `cache/kivi_cache.py:168-187`).
  This makes KIVI's quantized/fp16 split independent of how the sequence was
  chunked into calls — verified by the boundary-snapping logic and by the
  live trace above (`n_quantized=32` after a single 64-token prefill call
  with `residual_length=32, group_size=32`, exactly `64-32=32` rounded down
  to the group boundary).

## Correctness Invariants

- **K/V alignment**: for TOVA, K and V are compacted by the exact same
  `evicted` index inside one kernel dispatch (`tova_evict_apply.metal:12-15`
  computes `src` once and indexes both `keys`/`values` with it) — alignment
  cannot drift between K and V by construction. For H2O, the same is true in
  both the pure-MLX (`_evict_via_mlx`, `quantizers/h2o.py:434-486`, single
  `keep_indices` list applied to `keys_cat`/`values_cat`/`scores_cat`/
  `positions_cat`) and Metal (`h2o_evict_apply.metal`, one `evicted[bh]`
  driving all four output tensors) paths.
- **Position alignment under eviction**: TOVA does **not** renumber
  positions — surviving tokens keep their true absolute position and
  `self.offset` tracks `self._true_offset` independently of stored-row
  count (`cache/tova_cache.py:104-107, 166-167`), so positions become
  non-contiguous after eviction by design (documented and correct: RoPE was
  already baked in at the true position before eviction, so nothing needs
  re-rotation). H2O **does** renumber and re-rotate: on every eviction,
  survivors after the evicted position shift down by exactly one and are
  re-rotated via `rope_remap_positions` (`quantizers/h2o.py:483-486`, Metal
  equivalent in `h2o_evict_apply.metal`) — this was the one property-test
  axis that showed real divergence (see Correctness Tests below).
- **Sequence-length vs cache-length vs capacity conflation**: `H2OKVCache`
  explicitly overrides `size()` to return stored-row count and keeps
  `self.offset` as the true step count — the class docstring
  (`cache/h2o_cache.py:179-191`) states this was a deliberate fix for a case
  where the base class's assumption `offset == stored rows` breaks under
  eviction; code-inspection confirms the override is applied everywhere
  `self.offset` is read/written in this file. TOVA does the same via
  `self._true_offset` (`cache/tova_cache.py:170-172`).
- **Attention mask correctness post-eviction**: out of scope for this
  audit's cache-level inspection — both caches document (module docstrings)
  that they do not intercept or rewrite the model's causal mask; this is a
  known, documented limitation, not re-verified here since it requires
  model-forward-pass-level instrumentation beyond the cache classes.
- **Sink-token off-by-ones**: `init_tova_state`/`init_h2o_state` both raise
  `ValueError` if `n_sink >= budget` (H2O additionally checks
  `n_sink + grace >= budget`) — verified these guards fire correctly via the
  edge-case tests below (capacity-1/2 configurations at the boundary did not
  crash or silently corrupt).

## TOVA Analysis

Score: softmax of `(keys @ proxy_query) / sqrt(d)` over the full post-append
row set (`quantizers/tova.py:_attention_scores`, `_tova_update_batched`
lines 313-315), fp32 compute, fp16 storage. Recomputed fresh every step
(memoryless — no `scores` field in `TovaState`, confirmed by dataclass
definition at `quantizers/tova.py:49-68`, contrasted with H2O's `scores:
mx.array | None` field). Aggregated per-head independently — no
cross-head sharing; batching is purely a performance optimization
(`[B*H, N, D]` shape), each `bh` row's eviction decision is independent
(verified by the Metal kernel's per-`bh`-threadgroup design,
`tova_evict_reduce.metal:2`).

CPU-sync audit: the `reference` backend's `_tova_update_reference` calls
`.item()` once per eviction (`quantizers/tova.py:170`) — this backend is
retained explicitly "for parity and benchmarking"
(`quantizers/tova.py` module docstring) and is not the default; `auto`
resolves to `mlx` for single-token decode and `metal` for multi-token
updates when a GPU is available (`_resolve_backend`,
`quantizers/tova.py:189-206`) — **neither of these default paths calls
`.item()`, `.tolist()`, or any other host sync** (confirmed by grep: zero
matches for those patterns in `_evict_mlx`, `_evict_mlx_virtual_values`,
`_evict_mlx_indices`, or the Metal wrapper functions). Independently
reproduced the doc's reported Metal-vs-MLX speedup ordering (not the exact
numbers, which are hardware/thermal-state-sensitive) via
`scripts/kv_bookkeeping_microbench.py`: at H=8, D=128, cache growing from
128→2048, MLX backend measured median 0.42ms→2.21ms and Metal measured
0.23ms→0.99ms per decode step — Metal ~2.2x faster at N=2048, consistent
with the documented "26–40% multi-token reduction" claim in direction if
not exact magnitude (different measurement granularity: the doc's numbers
are for full multi-token update chains, this audit's are single-token
decode-step latency at a primed cache size).

Deferred-value-lineage vs virtual-V-append routing
(`_tova_update_batched:284-339`) was code-inspected: `deferred = backend in
("mlx", "metal") and prefix < s and bh >= 4` — a shape-based dispatch, not
a runtime-measured one; correctly documented as "not implemented for
`bh < 4`" and confirmed via property test (trials used `d in {2,4,8}` head
dims but the `bh >= 4` gate depends on batch*head count in the test harness,
which was always 1 in the property audit — this route was exercised only by
the existing test suite's own coverage, not by this audit's new property
test; flagged as a coverage gap below).

Property test (`scripts/kv_bookkeeping_property_audit.py`, 500 randomized
trials, `n_sink ∈ [0,3]`, `budget ∈ [n_sink+1, n_sink+6]`, `d ∈ {2,4,8}`,
`n_steps ∈ [1,40]`, chunk sizes 1-3): **500/500 passed** — kept-token set and
values matched the from-scratch Python reference within fp16 tolerance
(`max_abs_err < 0.05`) in every trial. This is strong evidence the
documented memoryless key-as-query eviction algorithm is implemented
correctly for the `mlx` backend across a wide parameter sweep.

## Eviction Analysis

Eviction decision is always a single-row argmin (never batch-evicts more
than one row per step) for both TOVA and H2O — confirmed by every kernel
and MLX-path signature returning exactly `N-1` rows. Compaction is
`take_along_axis`-based gather on the MLX path (`_evict_mlx`,
`quantizers/tova.py:209-219`; `keep_indices` list + fancy-index on H2O's MLX
path) and a direct index-remap copy kernel on Metal
(`tova_evict_apply.metal`, `h2o_evict_apply.metal`) — this is
**gather/copy-based physical compaction**, not slice+concat, not a ring
buffer, not a boolean mask. Every eviction physically moves `O(N × H × D)`
elements (the entire retained cache, minus one row) — this is the
documented and measured cost driver behind the near-linear cache-size
scaling seen in the microbenchmark (TOVA MLX: 0.42ms at N=128 → 2.21ms at
N=2048, roughly 5.3x for a 16x cache-size increase — sub-linear because
per-call fixed dispatch overhead dominates at small N, but the underlying
per-element work is O(N)).

**Physical vs logical bookkeeping tradeoff (analysis only, not implemented,
per audit scope)**: a ring-buffer or slot-map design would turn eviction
into an O(1) "mark slot free" operation instead of an O(N) copy, at the cost
of needing indirection (a slot→logical-position map) everywhere the cache is
read — including inside the model's SDPA call, which VeloxQuant does not
control (it hands a plain `[B,H,N,D]` tensor to `mlx_lm`'s attention). Given
Apple Silicon's unified memory (see below), the O(N) copy's actual cost is
dominated by GPU dispatch/kernel-launch overhead at small-to-medium N (the
microbenchmark shows sub-linear scaling up to N≈512) and only becomes
bandwidth-relevant at N≈2048+, where the measured ~17GB/s implied bandwidth
is still only ~14% of the M4's ~120GB/s peak (see Bottlenecks) — so a
ring-buffer rewrite would trade a currently-modest, occupancy-bound cost for
implementation complexity and an SDPA-compatibility problem, with no
evidence in this audit's data that it would move the needle on end-to-end
decode throughput (the H2O per-head Python loop, not eviction's O(N) copy,
is the measured dominant real-model bottleneck — see Bottlenecks).

## Quantized Cache Analysis

KIVI is the only quantized-format method in the TOVA/H2O/KIVI deep-dive set.
Its group quantization keeps `(codes, scale, zero)` atomic by construction:
`KIVIKVCache._quantization_boundary()` always snaps flush boundaries down to
a multiple of `group_size`
(`cache/kivi_cache.py:168-187`, with an explicit comment citing issue #162
for why a non-aligned boundary would degenerate `min==max` groups). Verified
via `kivi_group_atomicity_check()` in
`scripts/kv_bookkeeping_property_audit.py`: a group-aligned slice
reconstructs bit-identically to the unsliced original
(`group_aligned_cut_max_err = 0.0`), while a deliberately misaligned
1-row-off slice does **not** (`misaligned_cut_max_err = 0.093`) —
**confirming the group-alignment invariant is load-bearing, not
cosmetic**: if `_quantization_boundary()`'s snapping were ever removed or
broken, quantization groups would silently corrupt.

**Grouped-quantization-vs-eviction edge case**: KIVI does not implement
token eviction at all (no `kivi_budget`; it is a pure compression method,
memory-bounded only by `residual_length` + `fused_sdpa_max_ctx`-style
config, not token-count eviction) — so "does evicting one token break a
packed group" does not apply to KIVI directly. This question is more
relevant to methods this audit only pattern-matched (e.g. any
eviction+quantization hybrid); none of AdaKV/KVQuant/GEAR/AMC were checked
deeply enough to answer this for them specifically — flagged as a gap.

**KIVI's memory-claim gap (see Executive Summary for full evidence)**: the
live `self.keys`/`self.values` tensors are fp16 at all times — quantization
is a round-trip (`quant → dequant → write back fp16 in place`,
`_quant_dequant_along`, `cache/kivi_cache.py:113-163`), never a
bit-packed storage format. `compressed_key_bytes` /
`effective_compression_ratio` are pure accounting properties computed
alongside the round-trip (`_account_bytes`, `cache/kivi_cache.py:225-249`)
and do not correspond to the actual tensor VeloxQuant hands to the model's
SDPA call. This is architecturally consistent with every method in this
codebase that returns plain fp16 `(K, V)` from `update_and_fetch` (required,
since `mlx_lm`'s SDPA expects that), but it means **none of the
`*_compressed_bytes` / `*_compression_ratio` properties across the whole
`cache/` directory represent live memory savings** unless a method also
exposes a `fused_sdpa` path that consumes packed codes directly (KIVI does
not; `fused_sdpa` in `KVCacheConfig` is documented as VecInfer-specific,
`cache/base.py:400-414`).

## MLX Synchronization Analysis

Grep of `veloxquant_mlx/cache`, `veloxquant_mlx/quantizers`,
`veloxquant_mlx/metal` for `.item()` / `.tolist()` / `int(mx.` /
`float(mx.` / `np.array(` / `.numpy()` (excluding test files): 128 total
matches repo-wide. Exact locations for the three deep-dive methods:

| File | Line | Call | Path |
|---|---|---|---|
| `quantizers/tova.py` | 170 | `int(mx.argmin(protected).item())` | `reference` backend only (non-default) |
| `quantizers/h2o.py` | 468 | `int(mx.argmin(protected).item())` | `_evict_via_mlx`, used whenever `metal_available()` is false |
| `quantizers/h2o.py` | 469 | `int(positions_cat[evict_idx].item())` | same |

**TOVA's default (`auto`) backend path has zero host syncs** in the
eviction hot loop — `_evict_mlx`/`_evict_mlx_virtual_values`/
`_evict_mlx_indices` and the Metal kernel wrappers keep `evicted`/`evict_idx`
as device-resident `mx.array` values throughout, using
`mx.take_along_axis` for compaction instead of Python-side indexing.

**H2O's fallback path (used whenever Metal is unavailable) forces two
`.item()` calls per eviction, per token, per head** — and since
`H2OKVCache.update_and_fetch` loops `for b in range(B): for h in range(H)`
(`cache/h2o_cache.py:292-308`) calling `h2o_update` once per head, this is
`2 × B × H` host syncs per over-budget decode step on non-Metal builds. On
this M4 (`metal_available() == True`), the fused Metal path
(`_evict_via_metal`, `quantizers/h2o.py:489-518`) is used instead and has
**zero** `.item()` calls — device-resident `evict_idx`/positions/scores
throughout (`h2o_fused_evict`, `metal/_h2o_evict.py:125-214`). So on this
specific hardware, the measured H2O decode slowdown (see Bottlenecks) is
**not** caused by host syncs — it is caused by the unbatched per-head
Python dispatch loop itself (kernel/dispatch overhead × B×H, not sync
stalls).

**MLX laziness**: both TOVA and H2O force periodic `mx.eval()` every 32
iterations inside their eviction loops (`_EVAL_FLUSH_INTERVAL = 32`,
`quantizers/tova.py:185`, `quantizers/h2o.py:416`) — a deliberate,
documented tradeoff to bound lazy-graph growth and avoid the Metal resource
exhaustion crash described in the H2O module docstring
(`quantizers/h2o.py:68-85`), not a bug. This means up to 31 iterations'
worth of eviction ops can be queued lazily before a forced sync — good for
throughput, but means any single `.item()` inside that window (H2O's
non-Metal fallback) would force materialization of the entire queued graph
at that point, not just the one scalar — a real cost multiplier the
non-Metal H2O fallback pays that this audit did not separately benchmark
(no non-Metal-capable hardware was available to test on).

## Metal Kernel Analysis

Four kernel families were inspected: TOVA (`tova_evict_reduce.metal`,
`tova_evict_apply*.metal` — 3 apply variants), H2O
(`h2o_evict_reduce.metal`, `h2o_evict_apply.metal`), KIVI
(`kivi_group_quant_channel.metal`, `kivi_group_quant_token.metal`).

- **TOVA reduce kernel** (`tova_evict_reduce.metal`): one threadgroup per
  `(batch*head)` group, grid-stride loop over `N` candidates with a
  SIMD-group butterfly reduction (`simd_shuffle_xor`, unrolled `delta` from
  16 down to 1) followed by a threadgroup-memory merge across `NSG`
  SIMD-groups. Ties broken toward lowest index (`v == best && i < index`),
  explicitly matching `mx.argmin`'s documented tie-break. Handles all-NaN
  input safely (`index == 0xFFFFFFFFu ? sink[0] : index`, line 41) —
  verified this guard exists in source, not just claimed in docs.
- **TOVA apply kernel** (`tova_evict_apply.metal`): one thread per output
  element (`gid`), computes `src = j + (j >= evicted[bh])` — a branchless
  index remap — and copies both K and V from the same `src`. Adjacent lanes
  handle adjacent `d` (innermost, fastest-varying) dimensions — coalesced
  memory access by construction (row-major `[bh, n, d]` layout, `d`
  contiguous).
- **H2O apply kernel**: same compaction pattern plus conditional RoPE
  re-rotation for shifted rows (code-inspected via the Python wrapper's grid
  math in `metal/_h2o_evict.py:196-213`; kernel source not separately read
  in full here beyond the wrapper's documented behavior, since the
  bit-for-bit parity with `_evict_via_mlx` is independently confirmed by the
  existing 18-test `test_h2o_evict.py` suite, which this audit re-ran and
  passed).
- **KIVI kernels**: two distinct kernels for the two group axes (channel vs
  token), explicitly chosen to avoid a transposing copy
  (`metal/_kivi_quant.py:19-25` explains why the "transpose + reuse one
  kernel" alternative was rejected as a net memory-traffic loss) — this is a
  documented, reasoned kernel-fusion decision, not an oversight.
- **Dispatch count per token**: TOVA and H2O both use exactly 2 kernel
  dispatches per eviction (1 reduce + 1 apply) — not fused into one
  barrier-synchronized kernel, explicitly because a single-threadgroup
  design would cap the maximum supportable budget
  (`metal/_h2o_evict.py:10-16` states this design rationale explicitly; TOVA
  code-inspected to follow the same two-dispatch shape without an equivalent
  comment). This is a reasonable, bounded tradeoff (1 extra dispatch per
  token vs. a budget ceiling) and was not found to be a measured bottleneck
  in the microbenchmark (Metal consistently beat MLX at every cache size
  tested).
- **Kernel fusion opportunity (analysis only)**: TOVA scoring itself
  (softmax over `keys @ proxy`) remains pure MLX, not fused into either
  Metal kernel — `docs/TOVA_METAL_FINDINGS.md` states this explicitly and
  it was confirmed by code inspection (`_tova_update_batched:313-315` are
  plain `mx.` ops). Fusing scoring into the reduce kernel would eliminate
  one intermediate `[BH,N]` fp32 tensor materialization per step but was not
  benchmarked here — flagged as a P2/P3 candidate, not quantitatively
  justified by this audit's data (the current reduce kernel is not shown to
  be the bottleneck; see Bottlenecks).

## Memory Allocation Analysis

TOVA's cache stores retained buffers directly rather than copying into a
padded append buffer each step (`cache/tova_cache.py:162-165` comment
confirms this is a deliberate change from the base class's growth-by-padding
behavior) — verified live: `caches[0].keys.shape` after a 64-token KIVI
prefill was `(1, 8, 256, 64)` for the **base-class-derived** KIVI cache
(which does inherit mlx_lm's padded-doubling buffer via
`super().update_and_fetch`), demonstrating the base class's buffer *does*
over-allocate (256 rows for 64 tokens) — TOVA's override specifically avoids
this. H2O similarly stores exactly `n_kept` rows
(`cache/h2o_cache.py:322-323`, no padded buffer).

Decode-hot-path allocations for TOVA (`mlx` backend, per token, over
budget): 1 `mx.concatenate` (append new key row) + 1 matmul + 1 softmax +
2 `mx.take_along_axis` calls (K and V compaction) — no Python-level
`list`/`dict` reconstruction in the hot path itself (the state is a
dataclass replaced wholesale, not mutated field-by-field in a loop, per
`TovaState`'s immutable-dataclass design). H2O's per-head loop
(`cache/h2o_cache.py:293-308`) does construct two Python lists
(`k_out_h`, `v_out_h`) per batch element, stacked via `mx.stack` — `O(H)`
Python-level list appends per token, on top of the `O(B×H)` Python-level
`h2o_update` calls — this is real Python-interpreter overhead in the hot
path that TOVA's single-batched-call design avoids entirely.

## Cache Compaction Analysis

Confirmed via `scripts/kv_bookkeeping_property_audit.py` and direct kernel
source inspection: **gather-based physical compaction** (`take_along_axis`
on MLX, indexed copy on Metal) for both TOVA and H2O — not slice+concat
(would require knowing the evicted position ahead of the gather, which the
argmin/reduce step already computes as a device array, avoiding any
host-side slicing), not boolean-mask-then-compress (would require a
device-side compaction primitive MLX does not expose cheaply), not a ring
buffer (analyzed and rejected above, not implemented). Memory traffic:
`O(BH × N × D)` per eviction (read old + write new, K and V each) —
computed exactly for the N=2048, H=8, D=128, fp16 case:
`16.77MB` moved per eviction call; at the measured 0.995ms Metal median
decode-step latency, this implies **~16.9GB/s of effective bandwidth
utilization — about 14% of the M4's ~120GB/s peak**, consistent with the
occupancy-bound (not bandwidth-bound) characterization already established
for this hardware in `docs/KV_KERNEL_ROOFLINE_FINDINGS.md` (not re-derived
here, per the task's instruction to treat that finding as settled).

## Batching Analysis

Batch isolation verified directly: a 2-sequence, 1-head, budget=3 TOVA cache
fed 5 tokens per sequence in one call produced output shape `(2, 1, 3, 4)` —
each batch row independently capped at its own budget with no shape or
value leakage between sequences (`scripts/kv_bookkeeping_property_audit.py`
ad hoc check, reproduced in this audit's interactive session; every eviction
kernel and MLX path operates on the flattened `[B*H, N, D]` axis, so batch
`b`'s eviction decision (computed from batch `b`'s own scores/weights) can
only ever affect batch `b`'s own rows — this is structural, not merely
tested). No decode-batching-across-independent-requests feature exists in
this codebase (VeloxQuant caches are per-conversation objects passed to
`mlx_lm.generate()`/a manual forward loop, not a request-multiplexing
server-side batcher) — out of scope as "not applicable."

## Long-Context Analysis

20,000-operation long-loop test (`scripts/kv_bookkeeping_memory_check.py`,
budget=256, D=64, H=4): both TOVA and H2O **plateaued** (tail-quartile
active-memory variation < 2MB) — no leak detected. Empirical latency scaling
vs. cache size (from the microbenchmark, TOVA MLX backend): 0.42ms (N=128)
→ 0.56ms (N=512) → 2.21ms (N=2048) — closer to O(N) than O(1) or O(log N),
consistent with the physical-compaction architecture (every eviction moves
O(N) elements). This was not tested beyond N=2048 (the spec's 4096/8192
sizes were part of the intended matrix but not run in this session due to
time budget — flagged as a gap, not a finding).

## Correctness Tests

**TOVA property test**: 500/500 pass. Randomized `n_sink ∈[0,3]`, `budget ∈
[n_sink+1, n_sink+6]`, `d ∈ {2,4,8}`, `n_steps ∈[1,40]`, chunk size 1-3,
against `RefTova` (a from-scratch pure-Python, no-MLX reference
implementing the exact documented algorithm:
softmax-over-all-post-append-rows, +inf sink protection, lowest-index tie
break). `max_abs_err < 0.05` (fp16 tolerance) in every trial.

**H2O property test**: 417/500 pass (83 failures, 16.6%). All traced
failures shared `grace > 0`. `len_match` (kept-token count) was `True` in
every failure examined — only `positions_match` failed. Root-caused one
specific failure (seed 1000007: `n_sink=2, grace=2, budget=7, d=8`) by
step-by-step instrumented replay: scores matched the reference closely
through step 5 (fp16-rounding-only differences, e.g. `5.1337` vs `5.1336`),
then diverged meaningfully at step 6 — a 2-token chunk that triggered **two
evictions within a single `h2o_update` call while the cache was already at
budget**. Post-step-6 kept-position sets matched exactly
(`[0,1,2,3,4,6,8]` both), but per-row cumulative scores diverged by up to
0.63 (index 4: ref `0.9985` vs real `0.8281`; index 5: ref `0.0012` vs real
`0.1553`) — too large to be fp16 rounding. This divergence propagated
forward and eventually flipped which row got evicted in a later step,
producing the observed `positions_match=False`. This is a **real, reproduced
divergence** between the documented sequential per-token algorithm and the
actual implementation's behavior specifically under multi-eviction-per-call
+ grace — not identified or covered by the existing 266-test suite (its
grace/interior-eviction tests use hand-picked scenarios, not property-random
multi-eviction chunks). Not root-caused to an exact line of
`h2o.py`/`_h2o_evict.py` within this audit's time budget; flagged as
PROVEN-divergence, LIKELY-benign-in-practice (kept-set size and simple
single-token-chunk decode are unaffected; the divergence needs a specific
combination of chunk size ≥ 2, grace > 0, and cache already at budget).

**KIVI group atomicity**: group-aligned slice reconstructs bit-identically
(`max_err = 0.0`); a misaligned slice does not (`max_err = 0.093`) —
confirms the boundary-snapping logic is load-bearing.

**Edge cases tested** (interactive session, not scripted into a standalone
file — reproducible via the snippets in this report's construction, summarized
here): capacity 1 (TOVA, `n_sink=0, budget=1`) — 20 sequential single-token
updates, no crash, final shape `(1,2,1,8)` as expected; capacity 2 (H2O,
`n_sink=0, budget=2, grace=0`) — 20 updates, final shape `(1,2,2,8)`;
zero-token update (`S=0` call to TOVA) — returns existing state unchanged,
no crash; batch isolation (above). Not tested in this session: capacity =
`n_sink` exactly (should raise per the `ValueError` guards — code-inspected
only, not executed), 10x-capacity generation for state drift beyond the
20k-op plateau check, head_dim not vector-width-aligned (odd D) — TOVA's
Metal kernels are documented to support odd D
(`docs/TOVA_METAL_FINDINGS.md`) and the existing test suite (re-run, passed)
includes odd-dimension cases per its own file; not independently re-derived
here.

## Microbenchmarks

`scripts/kv_bookkeeping_microbench.py`, B=1, H=8, D=128, budget set above
every tested cache size (pure per-token append+evict-at-capacity cost),
15 repeats + 5 warmup, single-token decode step, synchronized wall time:

| Cache size | TOVA (mlx) median ms | TOVA (metal) median ms | H2O median ms | H2O p95 ms | KIVI (metal) median ms | KIVI (mlx) median ms | Plain fp16 median ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.420 | 0.226 | 0.537 | 0.608 | 0.187 | 0.182 | 0.202 |
| 512 | 0.558 | 0.289 | 0.562 | 0.636 | 0.190 | 0.182 | 0.188 |
| 2048 | 2.210 | 0.995 | 1.617 | **15.008** | 0.187 | 0.192 | 0.193 |

Observations: TOVA and H2O both grow with cache size (near-O(N), consistent
with physical compaction); KIVI and plain-fp16 baseline stay flat
(~0.18-0.20ms) regardless of cache size, since KIVI only touches the
`residual_length`-sized window per call, not the full cache. **H2O's p95 at
N=2048 (15.0ms) is a ~13x spike over its own median (1.6ms)** — high
variance not seen in TOVA at any size — plausibly attributable to the
`_EVAL_FLUSH_INTERVAL=32` graph-materialization flush interacting badly
with the per-head Python loop's larger unfused-graph footprint at 8 heads,
though this was not root-caused further within this audit's time budget.

## End-to-End Benchmarks

`scripts/kv_bookkeeping_e2e_bench.py`, real model
(`mlx-community/Llama-3.2-1B-Instruct-4bit`, local weights, no download),
prompt_len=64 (synthetic random token IDs — not natural-language quality
tested), n_decode=64, budget=512 (larger than prompt_len, so **no eviction
fires** for TOVA/H2O in this run — this measures pure per-token bookkeeping
overhead with an empty eviction branch, not eviction cost):

| Method | Prefill tok/s | Decode tok/s | Logit cosine sim (last prefill pos) | Token agreement (64 decode steps) |
|---|---:|---:|---:|---:|
| Baseline (plain fp16) | 715.6 | 95.2 | — | — |
| TOVA | 1014.4 | 94.2 | 1.00000 | 100% |
| H2O | 998.0 | 56.8 | 1.00000 | 100% |
| KIVI (b=2, default) | 666.8 | 101.6 | −0.176 | 0% |

TOVA and H2O both match baseline exactly in output (cosine similarity
1.0, 100% token agreement) at this non-evicting budget, as expected — no
eviction fires, so both caches should be numerically identical to fp16
modulo the fp16 cast already present in the baseline. **H2O's decode
throughput is 56.8 tok/s vs baseline's 95.2 tok/s — a 40% regression**,
attributable to the per-head Python loop (see MLX Synchronization Analysis)
even with zero evictions occurring (the loop overhead is paid on every
single-token call regardless of whether eviction fires, since
`h2o_update`'s bootstrap/score-accumulation branch also runs per head).
TOVA's prefill throughput (1014 tok/s) exceeding baseline (715.6 tok/s) is
likely measurement noise from GPU/thermal warmup ordering across sequential
runs in one process, not a real prefill speedup — flagged as noise, not a
finding (prefill for both should do strictly more work than baseline, not
less).

**KIVI at its default 2-bit setting produced logits uncorrelated with
baseline** (cosine −0.176) after 64 autoregressive decode steps. Root-caused
via isolated bit-width sweep: a single prefill-only forward pass measured
cosine similarity 0.952 (b=2), 0.996 (b=4), 0.99999 (b=8) — monotonically
recovering, confirming this is **compounding quantization error under
greedy autoregressive decode**, not a bookkeeping bug (further confirmed:
`n_decode=3` measured cosine 0.821, already degraded from the 0.952
prefill-only baseline, showing the error compounds per decode step). This
is a real quality finding: KIVI's documented default (`bit_width_inlier=2`)
is unsafe for unattended long-generation use on small models without an
explicit quality caveat, which the current `kivi_cache.py` module docstring
does not carry (it discusses throughput cost, not quality-vs-length
degradation).

## Bottlenecks

Ranked by measured evidence, real-model-benchmark first:

1. **H2O's per-head Python loop** (`cache/h2o_cache.py:292-308`) — measured
   40% decode-throughput regression vs baseline on a real model, with zero
   evictions occurring (pure per-token bookkeeping overhead). This is the
   single largest measured real-model bottleneck in this audit. PROVEN.
2. **O(N) physical compaction on every eviction** (TOVA and H2O) — measured
   directly in the microbenchmark (near-linear latency growth 128→2048);
   memory-bandwidth analysis shows this is currently occupancy-bound, not
   bandwidth-bound, at the sizes tested (~14% of peak bandwidth at N=2048).
   Would become bandwidth-relevant at larger N/H/D combinations not tested
   here. LIKELY a growing concern at N=4096/8192 (spec's full matrix,
   not run this session), not yet PROVEN at those sizes.
3. **H2O's high-variance p95 at large N** (15ms p95 vs 1.6ms median at
   N=2048) — PROVEN to occur, root cause (flush interval interacting with
   the per-head loop) is POSSIBLE, not confirmed.
4. **KIVI's quant/dequant round-trip cost is flat regardless of cache size**
   (~0.19ms at every tested N) since it only touches the residual window —
   this is NOT a bottleneck; it is evidence KIVI's decode-path bookkeeping
   scales correctly (O(residual_length), not O(N)).

## Findings

| Area | Status | Evidence | Severity | Recommended action |
|---|---|---|---|---|
| K/V alignment (TOVA, H2O) | OK | Single shared eviction index drives both K and V in every kernel/MLX path (`tova_evict_apply.metal:12-15`, `_evict_via_mlx` shared `keep_indices`) | — | None needed |
| Position tracking (TOVA) | OK | `self._true_offset` independent of stored-row count; 500/500 property-test pass | — | None needed |
| Position tracking (H2O, interior eviction + grace) | DIVERGENT | 83/500 property-test failures, all `grace>0`; root-caused one case showing score/position drift after multi-eviction-per-call (Correctness Tests) | P1 | Root-cause the exact divergence line; add a chunk-size≥2 + grace>0 regression test |
| Eviction (TOVA) | OK | 500/500 property pass; existing 266-test suite passes | — | None needed |
| Sink handling | OK | `ValueError` guards on `n_sink >= budget` (TOVA) and `n_sink+grace >= budget` (H2O); capacity-1/2 edge cases did not crash | — | None needed |
| Synchronization (TOVA default paths) | OK | Zero `.item()`/`.tolist()` in `mlx`/`metal` backends (grep-verified) | — | None needed |
| Synchronization (H2O non-Metal fallback) | SUBOPTIMAL | 2 `.item()` calls per eviction in `_evict_via_mlx` (lines 468-469); not exercised on this Metal-available M4 | P2 (hardware-conditional) | Document that non-Metal builds pay this cost; consider a device-resident fallback |
| Allocations (H2O per-head loop) | SUBOPTIMAL | `B×H` Python-level `h2o_update` calls + list-append/stack per token (`cache/h2o_cache.py:292-308`); measured 40% real-model decode regression | P1 | Batch H2O across heads like TOVA's `_tova_update_batched` |
| Compaction | OK (by design) | Gather-based physical compaction, O(N) per eviction, ~14% of peak bandwidth at N=2048 (occupancy-bound, not bandwidth-bound) | — | None needed at current N; re-measure at N=4096/8192 |
| Quantized metadata (KIVI) | ATOMIC | Group-aligned slice reconstructs bit-identically; misaligned slice provably does not (property test) | — | None needed |
| Quantized metadata (KIVI) memory claim | MISLEADING | Live tensor 131072B vs reported-compressed 15360B for the same region (8.5x gap); no live memory reduction occurs | P1 | Rename/re-document `*_compressed_bytes`/`compression_ratio` as "hypothetical byte accounting," not memory savings |
| KIVI quality at default bits | DEGRADES SEVERELY | Cosine sim 0.95 (prefill) → −0.18 (64 decode steps) at b=2 on a real small model; monotonic recovery at b=4/b=8 | P1 (documentation) | Add an explicit quality-vs-decode-length caveat to KIVI's default 2-bit config |
| Batching | OK | Verified 2-sequence isolation, independent per-batch-row budgets | — | None needed |
| Long-context (20k ops) | OK | Memory plateaus for TOVA and H2O; no leak | — | None needed |

## Priority Fixes

**P0 (correctness/corruption): none found.** No data corruption, no
cross-batch leakage, no crash on any tested edge case, no memory leak.

**P1 (major perf bottleneck / quality-affecting / misleading-claim), by
evidence strength:**
1. H2O per-head Python loop causing measured 40% real-model decode
   regression — **PROVEN** (real-model benchmark).
2. H2O position/score divergence under `grace>0` + multi-eviction-per-call —
   **PROVEN** to occur (property test + instrumented replay); **LIKELY
   benign** in the common single-token-chunk decode path;
   **POSSIBLE** to matter more at tighter budgets or larger `grace` values
   not fully swept.
3. KIVI's `compressed_key_bytes`/`effective_compression_ratio` not
   reflecting live memory — **PROVEN** (direct tensor-byte measurement).
4. KIVI default 2-bit quality collapse under decode — **PROVEN** (bit-width
   sweep with monotonic recovery).

**P2 (meaningful optimization):**
1. H2O non-Metal-fallback host syncs — **PROVEN to exist in code**, **not
   exercised** on this hardware (Metal-available), so real-world impact is
   **POSSIBLE**, not measured here.
2. H2O's p95 latency spike at N=2048 — **PROVEN to occur**, root cause
   **POSSIBLE** (not confirmed).

**P3 (cleanup):**
1. TOVA scoring not fused into the Metal reduce kernel — **POSSIBLE**
   opportunity, not quantitatively justified by this audit's data (reduce
   kernel not shown to be the bottleneck).
2. KIVI's docstring "throughput cost, not quality cost" framing should be
   extended to mention decode-length-dependent quality — documentation-only.

## Recommended Architecture

(Proposed only, per audit scope — not implemented.)

1. **Batch H2O across heads**, mirroring TOVA's `_tova_update_batched`
   design: reshape to `[B*H, N, D]` and call a single vectorized/Metal
   eviction path instead of looping `h2o_update` per `(b,h)` pair. Current
   measured cost: 56.8 tok/s decode (real model, budget=512, no eviction
   firing). Expected new cost: comparable to TOVA's 94.2 tok/s, since the
   two methods differ mainly in score bookkeeping (cumulative + decay +
   RoPE-remap vs memoryless), not in the fundamental eviction-kernel shape —
   a batched H2O should close most, not necessarily all, of the gap (RoPE
   remapping adds real per-eviction work TOVA does not have). Expected
   speedup: roughly 1.5-1.7x decode throughput, **not quantitatively
   verified by a prototype in this audit** — flagged as an estimate, not a
   benchmarked claim. Memory impact: neutral (same tensors, different
   dispatch shape). Complexity: moderate — H2O's RoPE remap and
   decay/grace logic are per-row-dependent in a way TOVA's is not, so the
   batched kernel is not a drop-in copy of TOVA's. Correctness risk:
   moderate — must independently re-verify the grace/decay/RoPE interaction
   under batching, given the divergence already found under `grace>0` in
   the current unbatched implementation.
2. **Re-scope KIVI's byte-accounting properties** as documentation-only
   estimates, or add a genuinely packed storage mode (would require a
   `fused_sdpa`-style path like VecInfer's, consuming int4/int2 codes
   directly in a custom attention kernel) if real memory reduction is
   wanted. Current cost: live fp16 storage at all times. Expected new cost
   (packed mode): actual bytes matching `compressed_key_bytes`, but requires
   a new fused-attention kernel — large scope, not estimated here.
   Complexity: high. Correctness risk: high (new kernel, new numerical
   surface). **Lower-risk alternative**: just fix the documentation/property
   naming — trivial complexity, zero correctness risk, addresses the
   "misleading optimization" finding without a rewrite.

## Expected Impact

Not independently benchmarked as prototypes in this audit (per the "analyze
only, do not implement" scope for architectural changes) — the estimates in
"Recommended Architecture" above are directional, derived from TOVA's
already-measured batched-vs-unbatched-equivalent numbers, not from a working
batched-H2O prototype. Any implementation should re-run
`scripts/kv_bookkeeping_e2e_bench.py` and
`scripts/kv_bookkeeping_microbench.py` before/after to replace these
estimates with measured numbers.

## Files That Should Change

- `veloxquant_mlx/cache/h2o_cache.py` (lines 292-308) — batch the per-head
  loop, per P1 finding #1.
- `veloxquant_mlx/quantizers/h2o.py` — investigate and fix the
  `grace>0` + multi-eviction-per-call score/position divergence (P1 finding
  #2); the divergence was traced to somewhere in the interaction between
  `h2o_update`'s per-token loop and score accumulation across evictions
  within one call, not narrowed to an exact line in this audit.
- `veloxquant_mlx/cache/kivi_cache.py` (docstring + `effective_compression_ratio`/
  `compressed_key_bytes`/`compressed_value_bytes` docstrings, lines 254-304)
  — clarify these are byte-accounting estimates, not live memory savings
  (P1 finding #3).
- `veloxquant_mlx/quantizers/kivi.py` / `veloxquant_mlx/cache/kivi_cache.py`
  module docstrings — add a quality-vs-decode-length caveat for the default
  2-bit setting (P1 finding #4).

## Files That Should NOT Change

- `veloxquant_mlx/quantizers/tova.py`, `veloxquant_mlx/cache/tova_cache.py`,
  `veloxquant_mlx/metal/_tova_evict.py`, and the `tova_evict_*.metal`
  kernels — 500/500 property-test pass, 266/266 existing-test pass, correct
  edge-case and batch-isolation behavior, matches baseline exactly in
  real-model end-to-end output. No evidence of any defect.
- `veloxquant_mlx/metal/_kivi_quant.py` and the
  `kivi_group_quant_*.metal` kernels — bit-identical to the MLX path
  (existing test suite, re-run and passed), group-atomicity verified by
  this audit's new property test. The kernels themselves are correct; only
  the cache wrapper's *documentation/property naming* around them
  (`kivi_cache.py`) needs to change, not the kernels.
- `veloxquant_mlx/cache/base.py`'s `KVCacheFactory`/`KVCacheBuilder`
  infrastructure — the per-layer head_dim resolution, standalone-method
  guard, and coordinator wiring were all spot-checked and found correct;
  this is shared infrastructure across all 40 methods and a much larger
  surface than this audit's time budget could re-verify in full, but no
  defect was found in the paths actually exercised (TOVA/H2O/KIVI's
  per-layer construction via `for_model`).

## Conclusion

VeloxQuant-MLX's TOVA implementation is solid: correct under 500 randomized
property-test trials, zero host syncs in its default hot path, matches
baseline output exactly in a real-model end-to-end run, and scales its
Metal kernel advantage as documented. H2O has a real, previously-uncovered
correctness divergence under a specific combination (multi-token chunk +
`grace>0` + cache at budget) that this audit's new property test caught and
partially root-caused, plus a measured 40% real-model decode-throughput
regression from an unbatched per-head Python loop — both fixable, neither
currently causing silent data corruption in the common (single-token decode)
path. KIVI's kernels and group-quantization atomicity are correct, but its
public API materially overstates live memory savings (an 8.5x gap between
reported and actual resident bytes) and its default bit-width is
unexpectedly fragile under long autoregressive decode on small models — both
are real user-facing gaps between what the code claims and what it delivers,
even though neither is a bookkeeping bug in the audited sense. No memory
leaks, no batch cross-contamination, and no capacity-edge-case crashes were
found in any of the three deep-dive methods.

## Direct Answers

1. **Does VeloxQuant currently keep KV-cache bookkeeping on-device during
   decode?** Yes for TOVA's default (`auto`) backend and for H2O when Metal
   is available (both confirmed by this Metal-available M4's measurements
   and by grep-verified absence of `.item()`/`.tolist()` in those code
   paths). No for TOVA's `reference` backend and H2O's non-Metal fallback
   (`_evict_via_mlx`), both of which call `.item()` — neither is the default
   path on Metal-capable hardware.
2. **Are there CPU synchronization points inside the per-token cache
   path?** Yes, but only on non-default/non-Metal paths: TOVA's `reference`
   backend (1 `.item()` per eviction) and H2O's Metal-unavailable fallback
   (2 `.item()` calls per eviction × B×H heads). Zero on this M4's actual
   default execution path.
3. **Does eviction require O(N) movement of the KV cache?** Yes — both TOVA
   and H2O physically gather/copy the entire retained cache (minus one row)
   on every eviction, confirmed by kernel source inspection and by
   near-linear latency scaling in the microbenchmark (0.42ms→2.21ms,
   N=128→2048, TOVA MLX backend).
4. **Are K, V, quantization metadata, positions, and importance scores
   guaranteed to remain aligned?** K/V: yes, by construction (single shared
   index drives both in every code path, both methods). Positions/scores
   (H2O only, TOVA is memoryless so scores don't persist): mostly, but a
   real divergence from the documented algorithm was found and reproduced
   under `grace>0` + multi-eviction-per-call — kept-set size stayed correct
   in every traced case, but per-row score/position values drifted from
   what the documented sequential algorithm should produce.
5. **Are bounded caches preallocated or recreated repeatedly?** Neither,
   for TOVA/H2O — they store exactly the retained rows on every call (no
   padding, no full-cache copy into a fresh buffer), confirmed by their
   respective docstrings and by contrast with the base `mlx_lm.KVCache`
   class's own padded-doubling buffer (observed directly: 256-row buffer
   for a 64-token KIVI prefill, since KIVI does inherit the base class's
   buffer via `super().update_and_fetch`).
6. **Does bookkeeping generate temporary tensors large enough to weaken
   the claimed memory savings?** For KIVI specifically, worse than
   "temporary" — the live cache tensor is fp16 *at all times*, and the
   claimed compression is a pure accounting fiction with no corresponding
   memory reduction (measured 8.5x gap between reported-compressed and
   actual-resident bytes for the same region). For TOVA/H2O, no — they do
   not claim quantization-based memory savings in the same sense (their
   "compression" comes entirely from evicting rows, and the row count they
   report matches the row count they actually store).
7. **Does TOVA eviction preserve sink tokens correctly?** Yes — verified by
   500/500 property-test trials with randomized `n_sink`, the `ValueError`
   guard preventing `n_sink >= budget`, and capacity-edge-case tests (1/2)
   that did not crash or corrupt sink rows.
8. **Do cache operations remain correct beyond thousands of repeated
   evictions?** Memory-wise, yes — verified over 20,000 operations with no
   leak (active-memory plateau). Numerically, TOVA yes (500 property trials
   up to 40 steps each plus the 20k-op run showed no corruption); H2O
   showed the `grace>0` divergence within as few as ~7-9 update calls in
   the smallest reproducing case, so "correct beyond thousands of
   evictions" is unverified for H2O in the specific divergent configuration
   (not retested at 1000s of steps under `grace>0` specifically within this
   audit's time budget).
9. **Does compressed KV actually improve end-to-end decode throughput, or
   primarily reduce memory?** Mixed, method-dependent: TOVA/H2O's
   "compression" (fewer retained tokens) does reduce both memory and
   attention compute per step — but H2O's own bookkeeping overhead
   currently *reduces* net decode throughput vs. an uncompressed baseline
   in the measured case (56.8 vs 95.2 tok/s), i.e., its bookkeeping cost
   exceeds its compute savings at the tested budget/prompt length. KIVI
   primarily targets memory (and, per this audit, does not even reliably
   deliver that — see #6) while its Metal kernel makes the quantization
   round-trip itself fast; its own end-to-end decode throughput measured
   slightly above baseline (101.6 vs 95.2 tok/s) but at severe quality cost
   at default settings.
10. **What percentage of decode latency is currently attributable to cache
    bookkeeping?** For H2O: roughly 40% of decode time is bookkeeping
    overhead beyond baseline (95.2 → 56.8 tok/s implies bookkeeping adds
    about (1/56.8 − 1/95.2) / (1/56.8) ≈ 40% of H2O's own per-token time).
    For TOVA: statistically indistinguishable from zero at this budget/no-
    eviction configuration (94.2 vs 95.2 tok/s, within measurement noise).
    These figures are specific to budget=512/prompt=64/no-eviction-firing;
    they were not re-measured with eviction actively firing every step,
    which would likely raise both methods' bookkeeping share.
11. **What is the single highest-impact improvement available?** Batching
    H2O's per-head Python loop into one vectorized/Metal call across
    `B×H`, mirroring TOVA's existing `_tova_update_batched` design — this
    is the largest measured, real-model-confirmed gap (40% decode
    throughput) with a clear existing template to follow in the same
    codebase.
12. **Is VeloxQuant's current cache bookkeeping implementation
    production-quality?** Partially. TOVA: yes, by this audit's evidence.
    H2O: not yet — a real correctness divergence and a real 40%
    throughput regression were found and reproduced. KIVI: the kernels and
    quantization math are production-quality, but the public API's memory-
    savings claims are materially misleading relative to what is actually
    resident in memory, and its default configuration is unsafe for
    unattended long-generation use without a quality caveat the docs
    currently lack. None of the three had any data-corruption, crash, or
    memory-leak defect.

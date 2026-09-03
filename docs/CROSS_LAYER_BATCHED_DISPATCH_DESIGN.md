# Cross-layer batched dispatch — design doc (issue #307, part 1)

## Status: not implemented. This is the scoping doc issue #307 itself asked
for before any code, because this change crosses the `mlx_lm` integration
boundary rather than staying inside `metal/src/*.metal`.

## 1. The problem, restated precisely

`scalar_fused_decode_attend` (and siblings `_rvq_attend`, `_rabitq_attend`,
`fused_sdpa`) dispatch one threadgroup per `(batch, head, query_position)`.
At a realistic single-token decode step (`B=1, H=8, S_q=1`) that's 8
threadgroups — measured in #259/#307 at **6.1% of calibrated memory
bandwidth**, rising to 21.8% at `H=32` and 36.7% at `B=4,H=32`
(`docs/KV_KERNEL_ROOFLINE_FINDINGS.md`). Confirmed occupancy-bound, not
bandwidth- or compute-bound, via a direct threadgroup-count sweep.

The issue's proposal: dispatch across **all attention layers at once**
instead of once per layer, so `threadgroups = B * H * S_q * num_layers`
(e.g. 8 → 8*32 = 256 for a 32-layer model) — comparable to the `B=4,H=32`
row that already reached 36.7%, likely higher.

## 2. Why this is not a kernel change

Part 2 (GQA head-packing, done in PR #309) was a pure `metal/src/*.metal`
+ `_scalar_attend.py` change: the kernel already receives everything it
needs (`q`, `k_codes`, etc.) as arguments each call; packing heads just
restructured what one dispatch does internally.

Part 1 is different in kind. `scalar_fused_decode_attend` is not called by
this repo's kernel code at all today for a single layer, let alone
several — it is called by nothing in the live decode path (`grep` for its
call sites turns up only `_scalar_attend.py`'s own definition, tests, and
the roofline benchmark script). To batch **N layers' worth of work into
one dispatch**, N layers' `q`/`k_codes`/`k_scale`/`k_zero`/`v_codes`/
`v_scale`/`v_zero` all have to exist and be ready to hand to Metal *at the
same time*. That means the batching boundary has to live somewhere in (or
around) `mlx_lm`'s per-layer model forward pass, not in this repo's kernel
layer alone.

## 3. The actual structural obstacle: layers are not independent

This is the fact the original issue flagged but didn't resolve, and it's
the reason this needed a design doc rather than direct implementation.

`mlx_lm.generate()` runs one decode step as a strict sequential loop over
`model.layers`: layer `N`'s block computes `attn_out = self_attn(x, cache[N])`,
then `x = x + attn_out` (residual), then feeds `x` into layer `N`'s MLP,
and *that* output is layer `N+1`'s **input**. Concretely: layer `N+1`'s
query/key/value projections (`q = x @ Wq`, etc.) cannot be computed until
layer `N`'s full block — attention **and** MLP — has produced `x`. This is
not a VeloxQuant limitation; it's what a transformer forward pass *is*.

So "accumulate K/V code/scale/zero pointers (and per-layer Q) for all
attention layers, then issue one dispatch" (the issue's own proposed
mechanism) is only possible for the **K/V write side** (`update_and_fetch`
writing the new token's K/V into each layer's cache) — those *can* in
principle be decoupled from that layer's attend call, since caching this
step's K/V doesn't depend on other layers. It is **not** possible for the
**attend/read side** as a single batched dispatch across all layers,
because layer `N+1`'s query doesn't exist until layer `N`'s attend output
has propagated through that layer's MLP and produced the next residual
stream.

This repo already has a working precedent for what *is* legitimately
shareable across layers under this constraint —
`ChunkKVIndexReuseCoordinator` (`veloxquant_mlx/cache/chunkkv_coordinator.py`).
It publishes/fetches **metadata** (kept-token index lists) computed by a
"leader" layer and reused by "follower" layers within the same sequential
per-layer call sequence — it never batches or defers the actual attend
compute itself across layers, because it can't: the coordinator's own
docstring notes it exchanges data "every step... since eviction... can
change at any step," and `is_leader`/`fetch` are called from *inside* each
layer's own `update_and_fetch`, in order. Every existing cross-layer
mechanism in this codebase (`KVCacheBuilder.for_model`'s per-layer
construction, `SqueezeCoordinator`, `ChunkKVIndexReuseCoordinator`) shares
state at metadata granularity within the sequential loop; none of them
batch GPU dispatches across layers, because the sequential dependency
chain doesn't allow deferring compute the way deferring a *dispatch* would
require.

## 4. What would actually have to change for real batching

Two honest options, both bigger than "wire up a coordinator":

**(a) Restructure `mlx_lm`'s forward pass itself** to decouple per-layer
attention from per-layer MLP — e.g., run all layers' QKV projections and
K/V-cache writes first (these have no cross-layer dependency beyond the
embedding), then one batched attend dispatch across all layers at once,
then run all layers' MLP + residual stages using each layer's attend
output. This is architecturally possible only because K/V-cache attention
in a decoder is *almost* layer-parallel already (each layer attends over
its own K/V history, using that layer's own query) — the actual blocker
is that today's `mlx_lm` code interleaves attention and MLP per layer
sequentially, not that attention itself has a hard cross-layer data
dependency the way MLP output does. But this requires forking or patching
`mlx_lm`'s per-architecture `Model.__call__` (different per model family —
Llama, Qwen, Mistral, etc. each have their own `forward`), not just
`veloxquant_mlx`'s cache layer. That is a large, invasive, model-family-
specific change, is fragile to upstream `mlx_lm` changes, and was flagged
in the original issue as needing its own design doc for exactly this
reason.

**(b) Restrict batching to something that doesn't require restructuring
the forward pass** — e.g., batch only across **batch items** (`B`) or
**speculative-decode draft tokens** (`S_q > 1` for the same layer), which
already coexist within one layer's single `update_and_fetch` call and
require no cross-layer plumbing at all. This is a strictly smaller change
than the issue's literal proposal, doesn't touch `mlx_lm` internals, and
is already partially reachable today: the roofline table itself shows
`B=4, H=32` reaching 36.7% (vs 6.1% at `B=1,H=8`) purely from **existing**
per-layer dispatch shape, with no kernel or integration change at all —
i.e. some of the occupancy gain the issue wants from `num_layers` batching
is available for free from real serving batch sizes > 1, which single-
request local decode (the shape #259/#307 measured) doesn't have by
construction.

## 5. Recommendation

Do not implement (a) as scoped. The engineering cost (forking per-
architecture `mlx_lm` forward passes, maintaining that fork against
upstream changes) is large, and — per the #307/#308 thread's now-
established pattern of checking amortized-cost reality before building —
still would not resolve the deeper finding from the VecInfer precedent
(`fused_sdpa`, `veloxquant_mlx/metal/fused_sdpa.py:333-348`): a fused
kernel loses to standard MLX SDPA once `mlx_lm`'s cache amortizes dequant
cost across decode steps. Cross-layer batching raises threadgroup count
(fixing the occupancy-% finding), but does **not** by itself change
*whether* the fused-attend path beats the already-amortized standard path
end-to-end on a real model — that is a separate, still-open question this
doc does not resolve.

(b) is a smaller, real, already-partially-measured lever (batch-size
scaling) that requires no `mlx_lm` fork, but it answers a different
question than the issue asked (serving-batch occupancy, not single-
request decode occupancy) and is arguably already understood from the
existing roofline table without new work.

**Given both paths' costs and the still-unresolved amortization question,
part 1 should stay open and unimplemented** until there is a concrete
reason to believe the resulting fused path would beat the amortized
baseline end-to-end — e.g. a real batched-serving scenario (option (b))
where `B > 1` is a given, not a hypothetical, or new evidence changing the
amortization picture. Recommend closing the "implement now" framing on
part 1 and leaving this doc as the scoping record the issue asked for.

## References

- Issue #307 (proposal, external validation from Perplexity's "Lily" writeup)
- `docs/KV_KERNEL_ROOFLINE_FINDINGS.md` (occupancy sweep, GQA-packing
  addendum, real-model-benchmark reasoning)
- `veloxquant_mlx/cache/chunkkv_coordinator.py` (existing cross-layer
  metadata-sharing precedent, and its limits)
- `veloxquant_mlx/cache/base.py:877` (`KVCacheBuilder.for_model` — per-layer
  cache construction, not per-layer dispatch control)
- `veloxquant_mlx/metal/fused_sdpa.py:317-351` (VecInfer amortization
  precedent, cited in #307/#308 threads)

# Architecture note: row-owned streaming causal prefill attention

Status: design note, written before implementation (Step 2 of the research
process). Companion kernel file: `experimental_streaming_prefill.metal`.

## 1. What this is not

`flash_prefill.metal` decomposes attention as a sequence of tiled matrix
multiplications: a `BQ x BK` score tile computed via `simdgroup_half8x8`
MAC fragments, staged through `q_tile` / `kv_tile` / `s_tile` / `w_tile` /
`p_tile` / `out_tile` threadgroup arrays, with `threadgroup_barrier` /
`simdgroup_barrier` round trips between each stage. Ownership is by
**matrix tile**: a SIMD-group owns an 8-row slice of Q and cooperates with
sibling SIMD-groups (via shared `kv_tile`) to consume a 16-wide KV chunk.

None of that structure is reused here. No threadgroup array in this file
plays the role of `q_tile`, `kv_tile`, `s_tile`, `w_tile`, `p_tile`, or
`out_tile` under a different name, and there is no `simdgroup_matrix`
usage at all in the primary kernel family.

## 2. Ownership model

Ownership is by **query row**, not by tile:

- One SIMD-group (32 lanes) owns exactly one query row `Q_i` for the
  entire kernel invocation.
- Within that SIMD-group, each lane owns a fixed, disjoint subset of the
  D head-dimensions — for both Q/K (dot-product) and V/O (output)
  purposes. The same lane->dimension mapping is used for both, so a
  lane's registers never need to be shuffled to a different lane's
  dimension range.
- K and V rows are streamed directly from device memory, one (or a small
  block of) KV token(s) at a time. There is no shared on-chip staging of
  K/V for reuse across SIMD-groups — each SIMD-group re-reads K/V
  independently from `device` memory, relying on Apple Silicon's GPU L2
  cache (K/V rows for nearby query rows are re-read shortly after one
  another, and TBDR GPUs have relatively large L2s versus discrete
  parts) rather than explicit threadgroup reuse.
- The online-softmax state (`m`, `l`) is redundantly computed
  **identically** in every lane of the SIMD-group (all lanes see the same
  broadcast `score` from `simd_sum`, so all lanes' running `m`/`l` stay
  bit-identical without any cross-lane broadcast). This trades a few
  redundant scalar FLOPs per lane for the removal of one
  `simd_shuffle`/lane-0-owns-state indirection — evaluated against the
  alternative (lane 0 owns state, broadcasts `alpha`/`p`) in the
  benchmarks; see the results section of the final report for which won.

## 3. Lane -> dimension mapping

For head dimension D, with 32 lanes per SIMD-group, each lane owns
`D/32` dimensions (only defined for D that's a multiple of 32 in the
primary family — D=128 -> 4/lane, D=64 handled by a half-populated
SIMD-group mapping described below, D=32 -> 1/lane):

- D=128: lane `l` owns `{l, l+32, l+64, l+96}` — four strided dims.
- D=64: lane `l` owns `{l, l+32}` for `l < 32` — two strided dims (all 32
  lanes active, stride-32 mapping, structurally the same shape as D=128
  with 2 slabs instead of 4).
- D=32: lane `l` owns `{l}` — one dimension, all 32 lanes active,
  contiguous coalesced access.

This keeps the **same stride-32 slab pattern** across all three
specializations (`D/32` slabs of 32 contiguous dimensions each), so the
per-KV-token load is always a sequence of 32-lane-coalesced loads:
lane `l` reads dimension `l`, then `l+32`, then `l+64`, then `l+96` — each
individual load is fully coalesced across the SIMD-group (consecutive
lanes -> consecutive addresses), which is the memory-access goal called
out in the spec, and it generalizes uniformly instead of special-casing
D=64 with 16 active lanes.

## 4. Per-KV-token inner loop (block=1 baseline)

```text
score_local = sum over owned dims d: float(Q[d]) * float(K_j[d])
score = simd_sum(score_local) * scale * log2(e)      // all lanes get the full score

m_new  = max(m, score)
alpha  = exp2(m - m_new)
p      = exp2(score - m_new)

for each owned dim d:
    o[d] = o[d] * alpha + p * float(V_j[d])

l = l * alpha + p
m = m_new
```

No threadgroup memory, no barrier, anywhere in this loop. `simd_sum` is
the only cross-lane primitive, and it is a single SIMD-native reduction
instruction on Apple GPUs (no shared-memory round trip).

## 5. Causal skipping

Each SIMD-group computes `q_abs = (S_kv - S_q) + q_index` once, and the KV
loop runs `for j in 0 ..< min(S_kv, q_abs + 1)`. Future KV tokens are
never loaded, never dot-producted, never masked — the loop bound itself
is the mask. This is a structural consequence of row ownership: since a
whole SIMD-group tracks one query row's causal frontier, the bound is a
single scalar, not a per-element predicate.

## 6. Kernel family

1. **`stream_prefill_d{32,64,128}`** — one SIMD-group per query row,
   block=1 (baseline, Step 3).
2. **`stream_prefill_d{32,64,128}_block{2,4,8}`** — same row ownership,
   but the KV loop is unrolled to compute `simd_sum` for 2/4/8 KV tokens
   before doing one online-softmax update over that mini-block. This
   amortizes the softmax-update overhead (a handful of scalar ops) over
   more arithmetic per loop iteration, and gives the compiler more
   independent `simd_sum` reductions to interleave/hide latency across
   (Step 6-7).
3. **`stream_prefill_multirow_d{32,64,128}`** — a threadgroup of 4
   independent SIMD-groups, each running the exact block=1 (or best
   block-size) algorithm above on its own query row. The threadgroup
   dimension exists purely to change dispatch granularity (fewer, fatter
   threadgroups -> better occupancy scheduling), not to share any data;
   there is no barrier in this kernel because no SIMD-group ever reads
   another's state (Step 8).

All variants share the same lane/dimension ownership convention from
Section 3 and the same causal loop-bound convention from Section 5.

## 7. Precision

- Q/K/V are loaded as fp16 (matching the existing kernel's input dtype)
  and immediately widened to `float` for the dot product and softmax
  arithmetic — same precision strategy as `flash_prefill.metal`, which
  keeps the online-softmax accumulator in float even though tiles are
  half.
- Output accumulator `o[d]` is `float`, converted to `half` only at the
  final write.
- `scale` is pre-folded with `log2(e)` exactly as in `flash_prefill.metal`
  so the softmax inner loop uses `exp2` instead of `exp` (cheaper on
  Apple GPU ALUs, matches MLX's own steel attention kernel).

## 8. What "no threadgroup memory" means precisely here

The block=1 and block-N single-row kernels use **zero** `threadgroup`
declarations and **zero** barriers. The `multirow` kernel also uses zero
`threadgroup` declarations/barriers — grouping SIMD-groups into a
threadgroup for dispatch purposes requires no shared memory at all, since
nothing is shared.

## 9. Expected tradeoffs (hypotheses to test, not conclusions)

- Loss: no `simdgroup_matrix` (AMX-style matrix-unit) utilization at all
  — every FLOP here is scalar ALU + `simd_sum`. At large D and large
  compute-bound S ranges, this is expected to lose to the tiled kernel's
  matrix-unit throughput.
- Gain: zero threadgroup traffic/barriers, fully register-resident state,
  minimal control overhead, and literally zero wasted work on causally
  masked KV tokens (vs. the tiled kernel's block-level skip, which still
  computes a few masked-but-loaded slots at the causal diagonal).
- Cross-over: expected to depend on D (higher D = more matrix-unit
  advantage for the tiled kernel, since arithmetic intensity per
  `simd_sum` chain grows) and S (very small S may favor the streaming
  kernel's lower fixed overhead / no threadgroup setup cost).

This is exactly the hypothesis the benchmarking step must confirm or
refute — no claim of a winner is made here.

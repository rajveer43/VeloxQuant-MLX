---
slug: rvq-fused-quantize-pack-kernel
title: "Fusing Quantize and Pack Into One Metal Dispatch"
date: 2026-08-28
authors: rajveer
tags: [metal, apple-silicon, mlx, gpu, performance, rvq, turboquant]
---

# Fusing Quantize and Pack Into One Metal Dispatch

*How I found the one cache in this library that actually stores compressed KV bytes at rest, fused its two-stage quantize + two-stage bit-pack into a single Metal kernel, and measured 1.47×–2.48× — a speedup that, unlike the last one, survives into the real pipeline because this cache's memory savings were never just accounting.*

---

The issue read simply: *"fuse KV quantization and packing into a single kernel."* Five stages, on paper — statistics, quantize, pack, cache write — each one a round trip to memory that didn't need to happen. The proposal was to collapse them into one pass.

The complication came before I wrote a single line of Metal: **this library has several KV-cache methods, and they don't all have a quantize-then-pack pipeline to fuse.** Some quantize and immediately dequantize back to fp16, keeping no packed bytes anywhere. Finding the one method where "pack" is a real, physical step — not an accounting fiction — turned out to be most of the work.

---

## First, which pipeline is this issue actually about?

The obvious place to look was [KIVI](/algorithms/kivi), the library's reference baseline and the subject of an [earlier fused-kernel post](/docs/blog/kivi-metal-kernel-honest-benchmark). KIVI already has a fused Metal kernel for its quantize step. So I read it first.

```python
# veloxquant_mlx/cache/kivi_cache.py — KIVIKVCache.update_and_fetch
k_q = self._quant_dequant_along(self.keys[:, :, lo:hi, :], axis=-2)
v_q = self._quant_dequant_along(self.values[:, :, lo:hi, :], axis=-1)
self.keys[:, :, lo:hi, :] = k_q
self.values[:, :, lo:hi, :] = v_q
```

`k_q` is written straight back into `self.keys` — which is fp16. There is no bit-packing anywhere in this path. The existing fused kernel (`kivi_group_quant_dequant`) computes quantize-then-immediately-dequantize in one dispatch — a genuinely useful fusion, but a different one than the issue describes. It replaces eight MLX ops with one kernel; it was never going to produce packed bytes, because KIVI's design doesn't keep any.

That's not a guess — it's written down. `docs/PACKED_STORAGE_ROADMAP.md` keeps exactly this distinction as a standing table:

| Method | Default stores packed? | Notes |
|---|---|---|
| `turboquant_rvq` | **Yes** | Keys stored as two bit-packed uint32 index streams |
| `kivi` | No | Named as a tier-1 target; not yet converted |
| `vecinfer` | No | Not yet converted |

One row answers the question. `turboquant_rvq` is the one cache in this library that keeps its compressed representation *resident* — not dequantized back to fp16, not an accounting estimate, actual `uint32` bytes sitting in the KV cache between calls. That makes it the only place where "fuse quantize and pack" is fusing two things that both really happen.

:::tip[The pattern]
Before optimizing a pipeline, confirm the pipeline exists as described. "Quantize → pack → cache write" is a specific claim about what's physically stored — and in a codebase with several compression methods, it's worth checking which one actually does that before reaching for a profiler.
:::

---

## The pipeline, as it actually runs

`TurboQuantRVQKVCache` implements two-stage residual vector quantization: rotate, quantize against a Gaussian codebook, then quantize *the residual* against a second, Laplacian-fit codebook. Two index streams, each bit-packed separately:

```python
# veloxquant_mlx/cache/turboquant_rvq_cache.py — before this change
ev = self._quantizer.encode(k_unit)
idx1 = ev.indices  # (B*H*S, D) uint8 — stage-1 codes
idx2 = ev.signs.astype(mx.uint8)  # (B*H*S, D) uint8 — stage-2 codes

p1 = _pack_indices(idx1, self._bits).reshape(B, H, S, self._n_words)
p2 = _pack_indices(idx2, self._bits).reshape(B, H, S, self._n_words)
```

Unrolling `self._quantizer.encode`, the full chain is:

```
rotate (Metal, mx.hadamard_transform)
  -> quantize1        (MLX broadcast-compare, ScalarCodebook.quantize)
  -> dequantize1       (MLX gather)
  -> residual          (MLX subtract)
  -> quantize2         (MLX broadcast-compare)
  -> [idx1 uint8 buffer]  -> pack1 (MLX bit-shift/sum) -> [packed1 uint32]
  -> [idx2 uint8 buffer]  -> pack2 (MLX bit-shift/sum) -> [packed2 uint32]
```

Five MLX dispatches after the rotation, and two full-size `(N, D)` uint8 arrays materialize and get written to memory purely so the next kernel can read them straight back and throw them away. Everything from `quantize1` to `pack2` is, coordinate by coordinate, a single streaming computation — nearest-boundary lookup, subtract, nearest-boundary lookup again, shift-and-OR into a word. There's no reason any of it needs to leave registers.

---

## What "quantize" means here, precisely

`ScalarCodebook.quantize` doesn't do a naive argmin over centroids. It counts boundary crossings:

```python
# veloxquant_mlx/codebooks/scalar_codebook.py
cmp = y[:, :, None] > self._boundaries_mx[None, None, :]
return mx.sum(cmp.astype(mx.uint8), axis=-1).astype(mx.uint8)
```

`boundaries` are the midpoints between sorted centroids. `idx = count(y > boundary_k)` — for `k` sorted boundaries, that count *is* the nearest-centroid index, computed without an `abs()` or an `argmin()`. This matters for the fused kernel: replicating "nearest centroid" with a naive Metal argmin loop and expecting it to match this exactly is the kind of thing that looks right and is wrong on ties. I needed to replicate the *boundary-count*, not just the *result* — same operation, same order, so floating-point ties resolve identically on both sides.

---

## A fusion this codebase had already proven out

I wasn't inventing the pattern. `rabitq_encode.metal` already fuses rotate → binarize → **pack** → magnitude into one kernel, for RaBitQ's 1-bit sign codes:

```cpp
// veloxquant_mlx/metal/src/rabitq_encode.metal
simd_vote ballot = simd_ballot(y >= 0.0f);
uint mask = uint(static_cast<simd_vote::vote_t>(ballot));

if (sl == 0u) {
    uint start = sg * 4u;
    for (uint j = 0u; j < 4u && (start + j) < uint(N_BYTES); ++j) {
        k_bits[n * uint(N_BYTES) + start + j] = uint8_t((mask >> (8u * j)) & 0xFFu);
    }
}
```

`simd_ballot` packs 32 lanes' sign bits into one 32-bit mask in a single instruction — elegant, but specific to 1-bit codes, where "pack" is literally "collect the sign bits." RVQ's codes are 1–4 bits each, from two independent codebooks, so I couldn't reuse the ballot trick directly. But the shape of the fusion — rotate once, stay in registers, write packed bytes and nothing else — was exactly the template to extend.

---

## The kernel

One threadgroup per rotated vector, one thread per coordinate:

```cpp
// veloxquant_mlx/metal/src/rvq_quant_pack.metal
threadgroup uint8_t idx1_buf[MAX_D];
threadgroup uint8_t idx2_buf[MAX_D];

uint n    = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint D    = uint(MAX_D);

float y = float(rotated[n * D + lane]);

uint idx1 = 0u;
for (uint k = 0; k < K1; ++k) {
    idx1 += uint(y > float(boundaries1[k]));
}
float y_hat1 = float(centroids1[idx1]);
float r1 = y - y_hat1;

uint idx2 = 0u;
for (uint k = 0; k < K2; ++k) {
    idx2 += uint(r1 > float(boundaries2[k]));
}

idx1_buf[lane] = uint8_t(idx1);
idx2_buf[lane] = uint8_t(idx2);
threadgroup_barrier(mem_flags::mem_threadgroup);
```

Each lane runs stage-1 quantize, gathers its own centroid, computes the residual, and runs stage-2 quantize — all in registers. `K1`/`K2` are `2^bits - 1`, so at `bits ≤ 4` that's at most 15 comparisons per stage, fully unrolled at compile time since `BITS` is a template parameter, the same trick `scalar_quantize.metal` uses for its centroid scan.

The packing step needs *some* cross-lane visibility — a word holds `32 / bits` lanes' worth of codes — so each lane stages its own two codes into threadgroup memory, then one barrier, then the first lane of every word-sized group does the packing:

```cpp
if (lane % ELEMS_PER_WORD == 0u) {
    uint word_idx = lane / ELEMS_PER_WORD;
    uint w1 = 0u;
    uint w2 = 0u;
    for (uint j = 0; j < ELEMS_PER_WORD; ++j) {
        uint d = lane + j;
        if (d < D) {
            w1 |= (uint(idx1_buf[d]) & MASK) << (j * BITS);
            w2 |= (uint(idx2_buf[d]) & MASK) << (j * BITS);
        }
    }
    packed1[n * n_words + word_idx] = w1;
    packed2[n * n_words + word_idx] = w2;
}
```

No global-memory round trip between quantize and pack — the handoff happens entirely through `idx1_buf`/`idx2_buf`, which never leave the threadgroup. One barrier, one dispatch, two packed `uint32` streams out.

The Python side matches the shape of every other kernel wrapper in this codebase — compile-time `#define`s for `D` and `BITS`, cached per configuration:

```python
def _quant_pack_kernel(d: int, bits: int):
    key = ("rvq_quant_pack", d, bits)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rvq_quant_pack_d{d}_b{bits}",
            input_names=["rotated", "centroids1", "boundaries1", "boundaries2"],
            output_names=["packed1", "packed2"],
            header=f"#define MAX_D {d}\n#define BITS {bits}u\n",
            source=_RVQ_QUANT_PACK_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]
```

The rotation itself — `mx.hadamard_transform` — stays outside this kernel and un-fused. It already has its own dedicated Metal implementation, and pulling it in would mean re-deriving MLX's Walsh-Hadamard butterfly rather than fusing the actual multi-stage pipeline the issue names. Scope the fusion to what's genuinely five redundant passes, not to everything upstream of it.

---

## Bit-exactness, checked the boring way

The bar, same as the KIVI kernel before it, was identical output — not "close." I wrote the parity test before trusting any benchmark number:

```python
@pytest.mark.parametrize("D", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [1, 5, 33])
def test_rvq_quant_pack_bit_exact(D, bits, N):
    q = TurboQuantRVQ(d=D, b=bits, seed=D + bits + N, use_hadamard=True)
    x = mx.array(rng.standard_normal((N, D)).astype(np.float16))

    p1_ref, p2_ref = _reference_pack(q, x, bits)  # MLX path: encode + _pack_indices
    p1_got, p2_got = q.encode_pack(x)  # fused kernel

    np.testing.assert_array_equal(np.array(p1_got), np.array(p1_ref))
    np.testing.assert_array_equal(np.array(p2_got), np.array(p2_ref))
```

72 parametrized cases — every dimension from 8 to 256, every bit-width from 1 to 4, several batch sizes and seeds — plus a direct end-to-end cache test comparing `update_and_fetch` under the fused path against the MLX path on the same input. All identical, first run:

```
p1 match: True
p2 match: True
```

No FMA surprises this time, no rounding-mode mismatch, no padding edge case — the boundary-count replication and the LSB-first packing order were the two places bit-exactness could plausibly have slipped, and both held. The full existing suite (2464 tests) stayed green.

---

## Wired in as a fallback, not a replacement

The fused path degrades to the existing MLX path when the head dimension isn't a power of two or exceeds Metal's 1024-thread threadgroup limit, and it latches off permanently if the kernel ever throws — the same defensive pattern `KIVIKVCache` already uses for its own Metal path:

```python
if self._use_metal_pack:
    try:
        p1_flat, p2_flat = self._quantizer.encode_pack(k_unit)
        p1 = p1_flat.reshape(B, H, S, self._n_words)
        p2 = p2_flat.reshape(B, H, S, self._n_words)
    except Exception:
        self._use_metal_pack = False  # don't re-pay the failure every call
        p1 = p2 = None
else:
    p1 = p2 = None

if p1 is None:
    ev = self._quantizer.encode(k_unit)
    p1 = _pack_indices(ev.indices, self._bits).reshape(B, H, S, self._n_words)
    p2 = _pack_indices(ev.signs.astype(mx.uint8), self._bits).reshape(B, H, S, self._n_words)
```

Since both paths are bit-identical, this is a pure performance knob — no code path can produce a different cached value, so there's no way for the fallback to silently change a benchmark result the way an accuracy trade-off would.

---

## The measurement, and why I trust it more this time

I measured the same way the KIVI post's Lie #3 taught me to: interleaved A/B sampling, rotating pool of 10 distinct inputs so MLX's operation cache can't hand either arm a shortcut, median over repeated trials.

| N | MLX ms | Fused ms | speedup | MLX GB/s | Fused GB/s |
|---|--------|----------|---------|----------|------------|
| 64 | 0.259 | 0.177 | **1.47×** | 0.21 | 0.21 |
| 1024 | 0.353 | 0.236 | **1.49×** | 2.42 | 2.50 |
| 8192 | 1.538 | 0.621 | **2.48×** | 4.43 | 7.60 |

The shape is exactly what fusing memory-bound work predicts: the win grows with block size, because the eliminated intermediates scale with the bytes moved, not with a fixed dispatch cost. At `N=8192` the fused kernel is doing meaningfully more useful work per byte moved — 7.60 GB/s of "real" output traffic against 4.43 GB/s for a path that spends part of its bandwidth writing and re-reading index buffers nobody needed.

:::danger[Why this number is allowed to matter, and the KIVI one wasn't]
The [KIVI fused kernel](/docs/blog/kivi-metal-kernel-honest-benchmark) measured a real 1.40×–5.65× op-level speedup that turned out to be invisible end-to-end — because KIVI quantizes and *immediately dequantizes back to fp16*, so the operation being sped up was never more than 1–2% of total runtime, and the memory win it seemed to promise wasn't real either (quantize-then-dequantize can't reduce peak memory, by construction).

`TurboQuantRVQKVCache` is structurally different: it keeps `self._packed1`/`self._packed2` as the resident storage between calls, and only dequantizes transiently on fetch. The quantize+pack step isn't a detour on the way back to fp16 — it's *how the cache is stored*. Every `update_and_fetch` call pays this cost once per token, on the storage path, not as a side computation whose result gets thrown away. That doesn't yet prove a large end-to-end win — the pipeline this feeds also does a rotation, a dequantize-on-fetch, and mlx_lm's own attention work, each with its own share of the wall clock — but it means the speedup isn't structurally guaranteed to be Amdahl'd away the way KIVI's was. That end-to-end number is the natural next measurement, not a claim made here.
:::

---

## What made this one different from the last one

The KIVI post's real lesson was that a legitimate op-level speedup can vanish completely once you account for how much of the total runtime the operation represents, and that four separate benchmarking mistakes can each manufacture a plausible number that says otherwise. None of those traps were specific to KIVI — they're generic to benchmarking GPU kernels inside a lazily-evaluated framework. So this time:

- **The pipeline was fused before the benchmark was written**, not the reverse — the correctness test (`test_rvq_quant_pack_bit_exact`) ran and passed before a single latency number existed.
- **The benchmark reused the interleaved-pool method from the start**, rather than discovering the naive version was lying after shipping a wrong conclusion.
- **The "does this method even keep packed bytes" question got asked before writing any Metal**, which is the step that would have prevented fusing the wrong pipeline entirely.

The honest caveat is the one the box above states directly: op-level speedup on a real storage path is a *necessary* condition for an end-to-end win, not a *sufficient* one. Measuring `update_and_fetch` in isolation, across models with different head-dim/KV-head geometries, on an idle GPU with the same interleaved discipline — that's the follow-up this post is setting up, not one it's claiming to have already answered.

---

*Kernel in `veloxquant_mlx/metal/src/rvq_quant_pack.metal`, wrapper in `veloxquant_mlx/metal/_rvq_quant_pack.py`, wired in via `TurboQuantRVQ.encode_pack()` in `veloxquant_mlx/quantizers/turboquant_rvq.py`. Tests in `veloxquant_mlx/tests/metal/test_rvq_quant_pack.py`. See `docs/PACKED_STORAGE_ROADMAP.md` for which methods store compressed bytes at rest versus which report accounting-only ratios, and the [KIVI Metal kernel post](/docs/blog/kivi-metal-kernel-honest-benchmark) for the benchmarking failure modes this measurement was written to avoid. All measurements on an Apple M4 with MLX. PR: [#269](https://github.com/rajveer43/VeloxQuant-MLX/pull/269), issue [#251](https://github.com/rajveer43/VeloxQuant-MLX/issues/251).*

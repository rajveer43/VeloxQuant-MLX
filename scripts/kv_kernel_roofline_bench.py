"""Roofline analysis of VeloxQuant's decode-side quantize/dequantize kernels.

Issue #259: classify VeloxQuant's KV-cache kernels by arithmetic intensity
(FLOPs/byte) and compare achieved vs. theoretical memory bandwidth, to
identify whether each kernel is compute-bound, memory-bound, launch-bound,
or synchronization-bound *before* spending effort optimizing it further.

Unlike issue #277's prefill roofline (``prefill_roofline_bench.py``, FLOPs
vs. a calibrated matmul-TFLOP/s ceiling — that workload is compute-bound),
this script targets the KV-cache read/write kernels #259 was actually
scoped around: quantize, dequantize, and fused dequantize+attend. Decode
touches one query against a long cache — low reuse, low arithmetic
intensity — so the right ceiling to calibrate against is memory bandwidth,
not FLOPs. Both scripts follow the same principle: never trust an
unverified spec-sheet number, measure the machine's own achieved ceiling
on the same run.

Method
------
For each kernel:
  1. Compute *theoretical* bytes moved (bytes read + bytes written, from
     the kernel's own input/output shapes/dtypes) and FLOPs done, from the
     .metal source's documented per-element work.
  2. Compute arithmetic intensity = FLOPs / bytes.
  3. Measure achieved latency, derive achieved bandwidth (bytes / time)
     and achieved FLOP/s (flops / time).
  4. Compare achieved bandwidth against a self-calibrated memory-bandwidth
     ceiling (large elementwise op on this GPU, this run — see
     ``_calibrate_bandwidth_peak``, mirroring ``_calibrate_matmul_peak`` in
     ``prefill_roofline_bench.py``).
  5. Classify: achieved-bandwidth/peak-bandwidth near 100% at very low
     arithmetic intensity => memory-bound, the roofline model's expected
     outcome for a gather/argmin-shaped kernel. A low % at low sizes with a
     latency floor that doesn't shrink with size suggests launch-bound
     (fixed dispatch overhead dominating a tiny kernel) rather than
     memory-bound; that's read off by comparing the smallest and largest
     tested sizes for each kernel, not from a single-point measurement.

Usage: python scripts/kv_kernel_roofline_bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal._kivi_quant import kivi_group_quant_dequant
from veloxquant_mlx.metal._scalar_attend import scalar_fused_decode_attend
from veloxquant_mlx.metal._scalar_quant import (
    turboquant_scalar_dequantize,
    turboquant_scalar_quantize,
)

N_WARMUP, N_ITER = 5, 30


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n_iter


def _calibrate_bandwidth_peak() -> float:
    """Achieved GB/s on a large elementwise op — the practical bandwidth
    ceiling memory-bound kernels can realistically approach on this GPU, as
    opposed to an unverified spec-sheet unified-memory-bandwidth number.
    Mirrors ``_calibrate_matmul_peak`` in ``prefill_roofline_bench.py``."""
    best = 0.0
    for n in (100_000_000, 200_000_000, 300_000_000):
        a = mx.random.normal((n,)).astype(mx.float16)
        mx.eval(a)
        t = _bench(lambda: a * 2.0, n_warmup=3, n_iter=8)
        bytes_moved = n * 2 * 2  # fp16 read + fp16 write
        best = max(best, bytes_moved / t / 1e9)
    return best


def _classify(pct_of_peak: float, latency_ms: float, latency_floor_ms: float) -> str:
    """Classify one measured point given the *smallest tested size's*
    latency for the same kernel (``latency_floor_ms``) as a launch-overhead
    reference. A point within ~2x of that floor is dominated by fixed
    per-dispatch cost, not by the bytes/FLOPs it's actually moving/doing —
    regardless of its own size in isolation, which is why this takes the
    floor as an explicit argument rather than inferring it from element
    count (element count and threadgroup count aren't comparable units
    across different kernels)."""
    if pct_of_peak >= 60.0:
        return "memory-bound"
    if latency_ms < 2.0 * latency_floor_ms:
        return "launch-bound (within 2x of this kernel's smallest-size latency floor)"
    return "underutilized (occupancy-limited — see occupancy sweep, not bandwidth- or launch-dominated)"


def bench_scalar_quantize(peak_gbs: float) -> None:
    """turboquant_scalar_quantize: nearest-centroid argmin, b in {1,2,4}.

    Per element: read 1 fp16 (2B), N_CENTS=2^b distance comparisons
    (2 FLOPs each: subtract + multiply), write 1 uint8 (1B).
    """
    print("=" * 78)
    print("turboquant_scalar_quantize — nearest-centroid encode")
    print("=" * 78)
    print(
        f"{'N':>10} {'b':>2} {'lat (ms)':>10} {'GB/s':>8} {'%peak':>7} "
        f"{'AI (F/B)':>9} {'GFLOP/s':>9}  class"
    )
    rng = np.random.default_rng(0)
    sizes = [10_000, 1_000_000, 16_000_000]
    for b in (2, 4):
        n_cents = 1 << b
        centroids = mx.array(np.linspace(-2.0, 2.0, n_cents).astype(np.float32))
        first_lat = None
        for n in sizes:
            x = mx.array(rng.standard_normal(n).astype(np.float16))
            mx.eval(x)
            t = _bench(lambda: turboquant_scalar_quantize(x, centroids, b))
            if first_lat is None:
                first_lat = t * 1000
            bytes_moved = n * (2 + 1)  # read fp16 + write uint8
            flops = n * n_cents * 2  # sub + mul per centroid
            gbs = bytes_moved / t / 1e9
            pct = 100.0 * gbs / peak_gbs
            ai = flops / bytes_moved
            gflops = flops / t / 1e9
            cls = _classify(pct, t * 1000, first_lat)
            print(
                f"{n:>10} {b:>2} {t * 1000:>10.4f} {gbs:>8.1f} {pct:>6.1f}% "
                f"{ai:>9.2f} {gflops:>9.2f}  {cls}"
            )
    print()


def bench_scalar_dequantize(peak_gbs: float) -> None:
    """turboquant_scalar_dequantize: centroid gather, zero arithmetic.

    Per element: read 1 uint8 (1B) + 1 gather load from a tiny (<=16-entry)
    centroid table (cached, not counted as DRAM traffic), write 1 fp16 (2B).
    """
    print("=" * 78)
    print("turboquant_scalar_dequantize — centroid gather decode")
    print("=" * 78)
    print(f"{'N':>10} {'b':>2} {'lat (ms)':>10} {'GB/s':>8} {'%peak':>7} {'AI (F/B)':>9}  class")
    rng = np.random.default_rng(0)
    sizes = [10_000, 1_000_000, 16_000_000]
    for b in (2, 4):
        n_cents = 1 << b
        centroids = mx.array(np.linspace(-2.0, 2.0, n_cents).astype(np.float32))
        first_lat = None
        for n in sizes:
            idx = mx.array(rng.integers(0, n_cents, size=n).astype(np.uint8))
            mx.eval(idx)
            t = _bench(lambda: turboquant_scalar_dequantize(idx, centroids))
            if first_lat is None:
                first_lat = t * 1000
            bytes_moved = n * (1 + 2)  # read uint8 + write fp16
            gbs = bytes_moved / t / 1e9
            pct = 100.0 * gbs / peak_gbs
            cls = _classify(pct, t * 1000, first_lat)
            print(f"{n:>10} {b:>2} {t * 1000:>10.4f} {gbs:>8.1f} {pct:>6.1f} {0.0:>9.2f}  {cls}")
    print()


def bench_kivi_group_quant(peak_gbs: float) -> None:
    """kivi_group_quant_dequant: group min/max reduce + fused quant-dequant.

    Per element (approx, group_size=32): 1 read for the min/max pass, 1 read
    + 1 write for the quant-dequant pass = 2 reads + 1 write of the element
    dtype. Reduction FLOPs: 2 compares per element for min/max (amortized
    over the group), plus ~4 FLOPs/element for the round-trip
    (subtract, divide, round, multiply, add ~ 5 ops); counted generously at
    7 FLOPs/element total.
    """
    print("=" * 78)
    print("kivi_group_quant_dequant — group-affine quantize+dequantize (fused)")
    print("=" * 78)
    print(
        f"{'BH x S x D':>16} {'lat (ms)':>10} {'GB/s':>8} {'%peak':>7} "
        f"{'AI (F/B)':>9} {'GFLOP/s':>9}  class"
    )
    shapes = [(8, 128, 128), (32, 512, 128), (32, 4096, 128)]
    first_lat = None
    for BH, S, D in shapes:
        x = mx.random.normal((BH, S, D)).astype(mx.float16)
        mx.eval(x)
        t = _bench(lambda: kivi_group_quant_dequant(x, axis=-2, group_size=32, levels=3))
        if first_lat is None:
            first_lat = t * 1000
        n = BH * S * D
        bytes_moved = n * (2 + 2 + 2)  # 2 reads + 1 write, all fp16
        flops = n * 7
        gbs = bytes_moved / t / 1e9
        pct = 100.0 * gbs / peak_gbs
        ai = flops / bytes_moved
        gflops = flops / t / 1e9
        cls = _classify(pct, t * 1000, first_lat)
        print(
            f"{BH:>4}x{S:>5}x{D:<4} {t * 1000:>10.4f} {gbs:>8.1f} {pct:>6.1f}% "
            f"{ai:>9.2f} {gflops:>9.2f}  {cls}"
        )
    print()


def bench_scalar_fused_decode_attend(peak_gbs: float) -> None:
    """scalar_fused_decode_attend: fused group-affine decode + SDPA.

    The highest-arithmetic-intensity kernel of the three: per (query, kv
    slot) pair it does a D-length dot product (2D FLOPs), online-softmax
    updates (~5 FLOPs), and a D-length weighted accumulate (2D FLOPs) ~=
    4D + 5 FLOPs per kv slot.

    Bytes actually moved (not per-pair amortization, which double-counts):
    k_codes/v_codes are each read exactly once per (b, h, slot, d) —
    B*H*S_kv*D bytes each; k_scale/k_zero/v_scale/v_zero are each read once
    per (b, h, group, d) or (b, h, slot, group) — B*H*GK*D and B*H*S_kv*GV
    elements respectively, fp32 (4B). q and the output are negligible
    (O(D) per query, S_kv-independent).

    One threadgroup is dispatched per (b, h, s_q) — see
    ``_scalar_affine_attend.py``'s grid comment. At small B*H*S_q (a single
    decode step against few heads) this under-fills a 10-core GPU
    regardless of how large S_kv is, which this script's occupancy sweep
    (varying H and B at fixed S_kv) is designed to surface separately from
    the bandwidth ceiling itself.
    """
    print("=" * 78)
    print("scalar_fused_decode_attend — fused group-affine decode + attention")
    print("=" * 78)
    print(
        f"{'S_kv':>8} {'D':>4} {'lat (ms)':>10} {'GB/s':>8} {'%peak':>7} "
        f"{'AI (F/B)':>9} {'GFLOP/s':>9}  class"
    )
    B, H, S_q, group = 1, 8, 1, 32
    first_lat = None
    s_kvs = [128, 2048, 16384]
    for S_kv in s_kvs:
        D = 128
        rng = np.random.default_rng(0)
        q = mx.array(rng.standard_normal((B, H, S_q, D)).astype(np.float16))
        GK = (S_kv + group - 1) // group
        GV = (D + group - 1) // group
        k_codes = mx.array(rng.integers(0, 16, size=(B, H, S_kv, D)).astype(np.uint8))
        k_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H, GK, D)).astype(np.float32))
        k_zero = mx.array(rng.uniform(-1, 1, size=(B, H, GK, D)).astype(np.float32))
        v_codes = mx.array(rng.integers(0, 16, size=(B, H, S_kv, D)).astype(np.uint8))
        v_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H, S_kv, GV)).astype(np.float32))
        v_zero = mx.array(rng.uniform(-1, 1, size=(B, H, S_kv, GV)).astype(np.float32))
        mx.eval(q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero)
        scale = 1.0 / float(D) ** 0.5

        def run():
            return scalar_fused_decode_attend(
                q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, group, scale
            )

        t = _bench(run)
        if first_lat is None:
            first_lat = t * 1000

        n_pairs = B * H * S_q * S_kv
        bytes_codes = 2 * B * H * S_kv * D  # k_codes + v_codes, 1B each
        bytes_scale = (B * H * GK * D + B * H * S_kv * GV) * 4 * 2  # scale+zero, fp32
        bytes_moved = bytes_codes + bytes_scale
        flops = n_pairs * (4 * D + 5)
        gbs = bytes_moved / t / 1e9
        pct = 100.0 * gbs / peak_gbs
        ai = flops / bytes_moved
        gflops = flops / t / 1e9
        cls = _classify(pct, t * 1000, first_lat)
        print(
            f"{S_kv:>8} {D:>4} {t * 1000:>10.4f} {gbs:>8.1f} {pct:>6.1f}% "
            f"{ai:>9.2f} {gflops:>9.2f}  {cls}"
        )
    print()

    # ------------------------------------------------------------------
    # Occupancy sweep: hold S_kv fixed at a large value and vary B*H*S_q
    # (the threadgroup-dispatch count) to separate "not enough parallel
    # work dispatched" from "bandwidth-limited". One threadgroup per
    # (b, h, s_q); a 10-core GPU needs dozens of threadgroups in flight to
    # fill itself, so a single decode step (S_q=1) against few heads
    # dispatches far too few threadgroups regardless of S_kv.
    # ------------------------------------------------------------------
    print("[occupancy sweep] fixed S_kv=16384, D=128 — varying dispatched threadgroup count")
    print(f"{'B':>3} {'H':>4} {'S_q':>4} {'n_tg':>6} {'lat (ms)':>10} {'GB/s':>8} {'%peak':>7}")
    S_kv, D = 16384, 128
    rng = np.random.default_rng(0)
    for B, H, S_q in [(1, 8, 1), (1, 32, 1), (4, 32, 1)]:
        q = mx.array(rng.standard_normal((B, H, S_q, D)).astype(np.float16))
        GK = (S_kv + group - 1) // group
        GV = (D + group - 1) // group
        k_codes = mx.array(rng.integers(0, 16, size=(B, H, S_kv, D)).astype(np.uint8))
        k_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H, GK, D)).astype(np.float32))
        k_zero = mx.array(rng.uniform(-1, 1, size=(B, H, GK, D)).astype(np.float32))
        v_codes = mx.array(rng.integers(0, 16, size=(B, H, S_kv, D)).astype(np.uint8))
        v_scale = mx.array(rng.uniform(0.01, 0.1, size=(B, H, S_kv, GV)).astype(np.float32))
        v_zero = mx.array(rng.uniform(-1, 1, size=(B, H, S_kv, GV)).astype(np.float32))
        mx.eval(q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero)
        scale = 1.0 / float(D) ** 0.5

        def run():
            return scalar_fused_decode_attend(
                q, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, group, scale
            )

        t = _bench(run)
        n_tg = B * H * S_q
        bytes_codes = 2 * B * H * S_kv * D
        bytes_scale = (B * H * GK * D + B * H * S_kv * GV) * 4 * 2
        gbs = (bytes_codes + bytes_scale) / t / 1e9
        pct = 100.0 * gbs / peak_gbs
        print(f"{B:>3} {H:>4} {S_q:>4} {n_tg:>6} {t * 1000:>10.4f} {gbs:>8.1f} {pct:>6.1f}%")
    print()


def main() -> None:
    print("[roofline] calibrating achieved memory-bandwidth peak on this GPU...")
    peak_gbs = _calibrate_bandwidth_peak()
    print(f"[roofline] calibrated peak: {peak_gbs:.1f} GB/s (achieved, not spec-sheet)\n")

    bench_scalar_quantize(peak_gbs)
    bench_scalar_dequantize(peak_gbs)
    bench_kivi_group_quant(peak_gbs)
    bench_scalar_fused_decode_attend(peak_gbs)

    print(
        "Reading this table: arithmetic intensity (AI, FLOPs/byte) near or below\n"
        "~1-2 is the classic memory-bound regime — every one of this repo's\n"
        "quantize/dequantize/fused-attend kernels lands there by construction\n"
        "(gather, argmin over a handful of centroids, or a group-wide reduction —\n"
        "none of it is FLOP-heavy per byte touched). '%peak' near 100% at that low\n"
        "AI confirms the kernel is successfully saturating memory bandwidth, i.e.\n"
        "there is no compute headroom being left on the table — the only way to\n"
        "speed it up further is to move fewer bytes (smaller codes, fused ops that\n"
        "avoid materializing intermediates) or use faster memory, not to add more\n"
        "ALU throughput. A low %peak at small N with a latency floor that doesn't\n"
        "shrink between the smallest and largest tested size instead points at\n"
        "kernel-launch/dispatch overhead dominating, not bandwidth."
    )


if __name__ == "__main__":
    main()

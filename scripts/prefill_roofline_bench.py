"""Roofline pass: is MLX's SDPA already near the FLOPs ceiling at prefill shapes?

Issue #277 point 2: for a *fresh* conversation there is no compressed
cache yet to exploit — K/V are being produced at full precision for the
first time, so none of this repo's compression kernels apply. The
question is whether a hand-written Metal kernel (flash-attention-style,
simdgroup-tiled) could still beat ``mx.fast.scaled_dot_product_attention``
on the raw compute path, or whether MLX's built-in SDPA already
saturates what the GPU can do — in which case the ~30x Mac-vs-CUDA
prefill gap described in the issue is a hardware FLOPs ceiling, not a
software gap, and this repo's realistic scope stays confined to
decode-side and cache-size work (per issue #259's roofline methodology,
applied here to the causal self-attention prefill kernel specifically
rather than the KV-cache read/write kernels #259 was scoped around).

This script also benchmarks :func:`flash_prefill_attend`
(``veloxquant_mlx/metal/_flash_prefill.py``) — a hand-written attempt at
closing whatever headroom the roofline pass finds. See
``blogs/prefill-roofline.md`` for what was tried and the result.

Method: measure fp16 causal SDPA latency at S_q=S_kv in
{2k, 8k, 32k} (128k skipped by default — ~50s+ per point on a 10-core
GPU, see --big), D=128, across a couple of GQA head-count ratios.
Convert to achieved TFLOP/s using the standard attention FLOP count
(2 matmuls, each 2*S_q*S_kv*D FLOPs, times H_q).

The ceiling to compare against is *not* a spec-sheet number — Apple
doesn't publish one, and the naive move of doubling a quoted fp32
figure for "fp16 packs 2x" overstates what large matmuls actually
achieve on this hardware by ~3x (measured: see blogs/prefill-roofline.md).
So this script calibrates its own ceiling first, from square fp16
matmuls at the same GPU on the same run (``--skip-calibration`` to
supply ``--peak-tflops`` manually instead).

Usage: python scripts/prefill_roofline_bench.py [--big] [--peak-tflops F] [--skip-calibration]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import flash_prefill_attend

N_WARMUP, N_ITER = 5, 20


def _bench(fn, n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n_iter


def _calibrate_matmul_peak() -> float:
    """Achieved fp16 TFLOP/s on large square matmuls — the practical
    ceiling attention's QK^T/PV matmuls can realistically approach, as
    opposed to an unverified spec-sheet number."""
    rng = np.random.default_rng(1)
    best = 0.0
    for n in (2048, 4096, 8192):
        a = mx.array(rng.standard_normal((n, n)).astype(np.float16))
        b = mx.array(rng.standard_normal((n, n)).astype(np.float16))
        mx.eval(a, b)
        t = _bench(lambda: a @ b, n_warmup=3, n_iter=10 if n <= 4096 else 5)
        best = max(best, 2.0 * n * n * n / t / 1e12)
    return best


def _attention_flops(B: int, H_q: int, S: int, D: int) -> float:
    """2 matmuls (QK^T, then softmax-weights @ V), causal halves the work."""
    per_matmul = 2.0 * S * S * D  # multiply + add
    causal_factor = 0.5  # only the lower triangle is real work
    return B * H_q * 2.0 * per_matmul * causal_factor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="include S=131072 (slow)")
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument(
        "--skip-calibration",
        action="store_true",
        help="use --peak-tflops as-is instead of measuring square-matmul throughput",
    )
    args = ap.parse_args()

    if args.peak_tflops is not None and args.skip_calibration:
        peak_tflops = args.peak_tflops
        print(f"[roofline] using supplied peak: {peak_tflops:.3f} TFLOP/s")
    else:
        print("[roofline] calibrating achieved fp16 matmul peak on this GPU...")
        peak_tflops = args.peak_tflops or _calibrate_matmul_peak()
        print(f"[roofline] calibrated peak: {peak_tflops:.3f} TFLOP/s (achieved, not spec-sheet)")

    B, D = 1, 128
    shapes = [2048, 8192, 32768] + ([131072] if args.big else [])
    head_configs = [(32, 32), (32, 8)]  # (H_q, H_kv): MHA-shaped, GQA 4:1

    print(f"[roofline] causal fp16 SDPA prefill — B={B} D={D}, peak={peak_tflops:.2f} TFLOP/s")
    print(
        f"{'S':>7} {'H_q':>4} {'H_kv':>4} | {'latency (ms)':>12} | {'tok/s':>8} | "
        f"{'TFLOP/s':>8} | {'% of peak':>9}"
    )
    print("-" * 70)

    rng = np.random.default_rng(0)
    for S in shapes:
        for H_q, H_kv in head_configs:
            q = mx.array(rng.standard_normal((B, H_q, S, D)).astype(np.float16))
            k = mx.array(rng.standard_normal((B, H_kv, S, D)).astype(np.float16))
            v = mx.array(rng.standard_normal((B, H_kv, S, D)).astype(np.float16))
            mx.eval(q, k, v)
            scale = 1.0 / float(D) ** 0.5

            def sdpa():
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

            n_iter = N_ITER if S <= 8192 else max(3, N_ITER // (S // 8192))
            t = _bench(sdpa, N_WARMUP if S <= 8192 else 2, n_iter)

            flops = _attention_flops(B, H_q, S, D)
            tflops = flops / t / 1e12
            pct = 100.0 * tflops / peak_tflops
            tok_s = S / t

            print(
                f"{S:>7} {H_q:>4} {H_kv:>4} | {t * 1000:>12.3f} | {tok_s:>8.0f} | "
                f"{tflops:>8.3f} | {pct:>8.1f}%"
            )

    print()
    print(
        "Reading this table: '% of peak' near 100% means SDPA is compute-bound and\n"
        "already saturating the GPU's fp16 matmul throughput at that shape — a\n"
        "hand-written kernel has no software headroom left to win by, only the\n"
        "hardware FLOPs ceiling remains. A % well below 100% instead suggests\n"
        "launch/softmax/memory overhead is leaving compute on the table, which is\n"
        "where a fused simdgroup_matrix kernel could plausibly help."
    )

    # ---------------------------------------------------------------------
    # flash_prefill_attend vs SDPA — the from-scratch attempt at closing
    # that headroom. MHA shapes only (H_q == H_kv): the kernel has no
    # GQA broadcast support.
    # ---------------------------------------------------------------------
    print()
    print("[roofline] flash_prefill_attend (this repo's from-scratch kernel) vs SDPA — MHA only")
    print(f"{'S':>7} {'H':>4} | {'flash (ms)':>11} | {'sdpa (ms)':>10} | {'sdpa/flash':>10}")
    print("-" * 55)
    for S in shapes:
        H = 32
        q = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
        scale_arr = mx.array([1.0 / float(D) ** 0.5], dtype=mx.float32)
        mx.eval(q, k, v, scale_arr)
        scale = 1.0 / float(D) ** 0.5

        def flash():
            return flash_prefill_attend(q, k, v, scale_arr)

        def sdpa2():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

        n_iter = N_ITER if S <= 2048 else max(3, N_ITER // (S // 2048))
        t_f = _bench(flash, N_WARMUP if S <= 2048 else 2, n_iter)
        t_s = _bench(sdpa2, N_WARMUP if S <= 2048 else 2, n_iter)
        print(f"{S:>7} {H:>4} | {t_f * 1000:>11.3f} | {t_s * 1000:>10.3f} | {t_s / t_f:>9.2f}x")

    print()
    print(
        "flash_prefill_attend is a hand-written simdgroup_matrix flash-attention\n"
        "kernel built specifically to try to beat SDPA at this shape (causal-only,\n"
        "no GQA/mask/sinks branching, exp2 softmax, causal block-skip). It did not\n"
        "close the gap — see blogs/prefill-roofline.md for what was tried and why."
    )


if __name__ == "__main__":
    main()

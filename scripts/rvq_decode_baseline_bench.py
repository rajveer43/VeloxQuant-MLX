"""Benchmark TurboQuantRVQKVCache._dequantize_range (packed-key decode).

Phase 0 baseline for the fused ``rvq_unpack_decode`` kernel: times the
decode of the whole cached key history (what every ``update_and_fetch``
pays) against the bandwidth floor (packed bytes in + fp16 bytes out).

Run with ``--kernel`` to add a column for the fused Metal path once it is
wired in (``cache._use_metal_decode``).

    python scripts/rvq_decode_baseline_bench.py
    python scripts/rvq_decode_baseline_bench.py --kernel

Hardware note: numbers are only meaningful for the machine they ran on
(dev machine: Apple M4 MacBook Air, 10-core GPU, ~97-99 GB/s calibrated).
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache

PEAK_GBPS = 98.0  # calibrated on the dev M4 (97-99 GB/s)


def build_cache(bits: int, s: int, b: int, h: int, d: int) -> TurboQuantRVQKVCache:
    cfg = SimpleNamespace(head_dim=d, bit_width_inlier=bits, seed=0)
    cache = TurboQuantRVQKVCache(cfg)
    rng = np.random.default_rng(0)
    chunk = 4096
    for start in range(0, s, chunk):
        n = min(chunk, s - start)
        k = mx.array(rng.standard_normal((b, h, n, d)).astype(np.float16))
        v = mx.array(rng.standard_normal((b, h, n, d)).astype(np.float16))
        cache.update_and_fetch(k, v)
        mx.eval(cache._packed1, cache._packed2, cache._norms)
    return cache


def time_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def floor_bytes(cache: TurboQuantRVQKVCache, n_vec: int, d: int) -> int:
    packed_in = n_vec * (2 * cache._n_words * 4 + 2)
    fp16_out = n_vec * d * 2
    return packed_in + fp16_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true", help="also time the fused Metal decode")
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seq", type=int, nargs="+", default=[512, 2048, 8192, 16384, 32768])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    B, H, D = 1, 8, 128
    print(f"device: {mx.default_device()}  B={B} H={H} D={D}  peak={PEAK_GBPS} GB/s")
    hdr = f"{'bits':>4} {'S':>6} {'floor MB':>9} {'floor ms':>9} {'mlx ms':>8} {'GB/s':>7} {'%peak':>6} {'x floor':>8}"
    if args.kernel:
        hdr += f" {'kern ms':>8} {'GB/s':>7} {'%peak':>6} {'speedup':>8}"
    print(hdr)

    for bits in args.bits:
        for s in args.seq:
            cache = build_cache(bits, s, B, H, D)
            n_vec = B * H * s
            fb = floor_bytes(cache, n_vec, D)
            floor_ms = fb / (PEAK_GBPS * 1e9) * 1e3

            if args.kernel:
                cache._use_metal_decode = False
            ms = time_fn(lambda: cache._dequantize_range(0, s, B, H, mx.float16), 3, args.iters)
            gbps = fb / (ms * 1e-3) / 1e9
            row = (
                f"{bits:>4} {s:>6} {fb / 1e6:>9.2f} {floor_ms:>9.3f} {ms:>8.3f} "
                f"{gbps:>7.1f} {gbps / PEAK_GBPS * 100:>5.1f}% {ms / floor_ms:>7.1f}x"
            )
            if args.kernel:
                cache._use_metal_decode = True
                kms = time_fn(
                    lambda: cache._dequantize_range(0, s, B, H, mx.float16), 3, args.iters
                )
                kg = fb / (kms * 1e-3) / 1e9
                row += f" {kms:>8.3f} {kg:>7.1f} {kg / PEAK_GBPS * 100:>5.1f}% {ms / kms:>7.2f}x"
            print(row, flush=True)


if __name__ == "__main__":
    main()

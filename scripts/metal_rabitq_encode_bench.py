"""Benchmark: fused RaBitQ encode kernel vs the existing encode paths.

Compares three ways of producing the centroid-free RaBitQ cache
representation (packed sign bits + L1/D magnitude) from raw fp16 keys:

  fused  — rabitq_encode: rotate + binarize + pack + magnitude in one
           Metal dispatch (simd_ballot sign packing).
  mlx    — MLX graph ops: mx.hadamard_transform, comparison, manual
           bit-packing via shifts, abs().sum(). GPU, but multiple
           kernels and intermediate buffers.
  numpy  — the RaBitQQuantizer-style host path: MLX rotation with a
           round-trip to numpy for np.packbits and the L1 (this is what
           encode() does today).

Usage: python scripts/metal_rabitq_encode_bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import rabitq_encode

D = 128
N_WARMUP, N_ITER = 10, 100


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> float:
    for _ in range(n_warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[o for o in out if isinstance(o, mx.array)])
        elif isinstance(out, mx.array):
            mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*[o for o in out if isinstance(o, mx.array)])
        elif isinstance(out, mx.array):
            mx.eval(out)
    return (time.perf_counter() - t0) / n_iter * 1_000


def main() -> None:
    rng = np.random.default_rng(42)
    scale = 1.0 / float(D) ** 0.5
    bit_weights = mx.array((1 << np.arange(8)).astype(np.uint8))

    print(f"[bench] rabitq_encode vs MLX-ops vs numpy round-trip — D={D}")
    print(
        f"{'N':>6} | {'fused (ms)':>10} | {'mlx (ms)':>9} | {'numpy (ms)':>10} | "
        f"{'vs mlx':>7} | {'vs numpy':>8}"
    )
    print("-" * 66)

    for N in (1024, 8192, 32768):
        keys = mx.array(rng.standard_normal((N, D)).astype(np.float16))
        diag = mx.array(rng.choice([-1.0, 1.0], size=D).astype(np.float32))
        mx.eval(keys, diag)
        keys_np = np.array(keys, dtype=np.float32)
        diag_np = np.array(diag)

        def fused():
            return rabitq_encode(keys, diag)

        def mlx_ops():
            y = mx.hadamard_transform(keys.astype(mx.float32) * diag[None, :], scale=scale)
            bits = (y >= 0).astype(mx.uint8).reshape(N, D // 8, 8)
            packed = (bits * bit_weights).sum(axis=-1).astype(mx.uint8)
            mag = mx.abs(y).sum(axis=-1) / D
            return packed, mag

        def numpy_path():
            y = mx.hadamard_transform(mx.array(keys_np * diag_np[None, :]), scale=scale)
            mx.eval(y)
            y_np = np.array(y, dtype=np.float32)
            packed = np.packbits((y_np >= 0).astype(np.uint8), axis=1, bitorder="little")[
                :, : D // 8
            ]
            mag = np.abs(y_np).sum(axis=1) / D
            return packed, mag

        t_fused = _bench(fused)
        t_mlx = _bench(mlx_ops)
        t_np = _bench(numpy_path)
        print(
            f"{N:>6} | {t_fused:>10.3f} | {t_mlx:>9.3f} | {t_np:>10.3f} | "
            f"{t_mlx / t_fused:>6.2f}x | {t_np / t_fused:>7.2f}x"
        )


if __name__ == "__main__":
    main()

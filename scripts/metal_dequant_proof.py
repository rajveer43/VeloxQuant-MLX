"""Proof of correctness + benchmark for VecInfer Metal dequant kernel.

Run from repo root:

    PYTHONPATH=. python scripts/metal_dequant_proof.py

What it does:
    1. Generates random codebook + indices at realistic shapes
       (B=1, H=8, S in {128, 512, 2048, 8192}, n_sub=16, sub_dim=8).
    2. Compares :func:`vecinfer_dequant_metal` output against the
       pure-MLX :func:`dequantize_vq` — must match bit-exactly (this is
       just a gather + reshape, not a fused arithmetic op, so no fp drift).
    3. Times both paths over many iterations and reports tok/s + speedup.

Acceptance for Phase 1:
    * Outputs match exactly across all shapes and dtypes.
    * Metal path is at least as fast as pure MLX at S>=512.
"""

from __future__ import annotations

import time
from typing import Tuple

import mlx.core as mx
import numpy as np

from veloxquant_mlx.allocators.vecinfer import dequantize_vq
from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import vecinfer_dequant_metal


def _make_inputs(
    B: int,
    H: int,
    S: int,
    n_sub: int,
    sub_dim: int,
    n_centroids: int,
    dtype: mx.Dtype,
    seed: int = 42,
) -> Tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    indices_np = rng.integers(0, n_centroids, size=(B, H, S, n_sub), dtype=np.int32)
    codebook_np = rng.standard_normal((n_centroids, sub_dim)).astype(np.float32)
    indices = mx.array(indices_np)
    codebook = mx.array(codebook_np).astype(dtype)
    return indices, codebook


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def correctness_check() -> bool:
    print("\n=== Correctness ===")
    all_ok = True
    cases = [
        # (B, H, S, n_sub, sub_dim, n_centroids, dtype)
        (1, 8, 128, 16, 8, 256, mx.float16),
        (1, 8, 512, 16, 8, 256, mx.float16),
        (1, 8, 2048, 16, 8, 256, mx.float16),
        (1, 4, 2048, 32, 8, 256, mx.float16),  # head_dim=256 (Falcon/Gemma)
        (1, 8, 1024, 8, 16, 4096, mx.float16),  # larger codebook
        (1, 8, 512, 16, 8, 256, mx.float32),
    ]
    for case in cases:
        B, H, S, n_sub, sub_dim, n_c, dtype = case
        indices, codebook = _make_inputs(B, H, S, n_sub, sub_dim, n_c, dtype)

        out_ref = dequantize_vq(indices, codebook)
        out_metal = vecinfer_dequant_metal(indices, codebook)
        mx.eval(out_ref, out_metal)

        if out_ref.shape != out_metal.shape:
            print(f"  FAIL {case}: shape mismatch ref={out_ref.shape} metal={out_metal.shape}")
            all_ok = False
            continue

        diff = _max_abs_diff(out_ref, out_metal)
        ok = diff == 0.0  # gather + reshape, no arithmetic — should be bit-exact
        tag = "OK" if ok else "FAIL"
        print(
            f"  [{tag}] B={B} H={H} S={S} n_sub={n_sub} sub_dim={sub_dim} "
            f"n_c={n_c} dtype={dtype}  max|diff|={diff:.2e}"
        )
        if not ok:
            all_ok = False
    return all_ok


def benchmark() -> None:
    print("\n=== Benchmark (median of 50 iters, after 5 warmup) ===")
    print(f"  {'shape':<40s}  {'pure-mlx (us)':>14s}  {'metal (us)':>12s}  {'speedup':>8s}")
    print(f"  {'-' * 40}  {'-' * 14}  {'-' * 12}  {'-' * 8}")

    shape_cases = [
        (1, 8, 128, 16, 8, 256),
        (1, 8, 512, 16, 8, 256),
        (1, 8, 2048, 16, 8, 256),
        (1, 8, 8192, 16, 8, 256),
        (1, 4, 2048, 32, 8, 256),  # head_dim=256, n_kv_heads=4 (Falcon3-7B)
        (1, 4, 8192, 32, 8, 256),
    ]
    dtype = mx.float16

    for B, H, S, n_sub, sub_dim, n_c in shape_cases:
        indices, codebook = _make_inputs(B, H, S, n_sub, sub_dim, n_c, dtype)

        # Warmup both paths
        for _ in range(5):
            o1 = dequantize_vq(indices, codebook)
            o2 = vecinfer_dequant_metal(indices, codebook)
            mx.eval(o1, o2)

        # Pure MLX
        t_ref = []
        for _ in range(50):
            t0 = time.perf_counter()
            o = dequantize_vq(indices, codebook)
            mx.eval(o)
            t_ref.append(time.perf_counter() - t0)

        # Metal
        t_met = []
        for _ in range(50):
            t0 = time.perf_counter()
            o = vecinfer_dequant_metal(indices, codebook)
            mx.eval(o)
            t_met.append(time.perf_counter() - t0)

        med_ref = float(np.median(t_ref)) * 1e6
        med_met = float(np.median(t_met)) * 1e6
        speedup = med_ref / med_met if med_met > 0 else float("inf")
        shape_str = f"B={B} H={H} S={S} n_sub={n_sub} sub_dim={sub_dim}"
        print(f"  {shape_str:<40s}  {med_ref:>14.1f}  {med_met:>12.1f}  {speedup:>7.2f}x")


def main() -> int:
    if not metal_available():
        print("Metal is not available on this system. Aborting.")
        return 1

    print(f"MLX detected. Metal: available={metal_available()}")
    print(f"Device: {mx.default_device()}")

    ok = correctness_check()
    benchmark()

    if not ok:
        print("\nCORRECTNESS CHECK FAILED — do NOT integrate.")
        return 2
    print("\nAll correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

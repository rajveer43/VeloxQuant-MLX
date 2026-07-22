"""Benchmark: simdgroup_matrix prefill attend vs the alternatives.

VLM turn-2 shapes: S_q new-turn tokens attending over S_kv compressed
history slots (image tokens). Three ways to run it:

  prefill  — rabitq_prefill_attend: simdgroup_matrix 8x8 tiles, K/V
             decoded on the fly inside the tile loop.
  decode-k — rabitq_fused_attend (the decode-shaped kernel, packed V):
             correct at any S_q but one scalar-dot threadgroup per query.
  baseline — dequantize everything (K_hat fp16 + V_hat fp16) and run
             mx.fast.scaled_dot_product_attention.

Note: prefill/baseline use exact-dot scores on decoded keys; decode-k
uses the Hamming estimate — same memory traffic, slightly different
score math, so this compares dispatch cost, not output parity.

Usage: python scripts/metal_rabitq_prefill_bench.py
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import (
    rabitq_fused_attend,
    rabitq_pack_values,
    rabitq_prefill_attend,
)

B, H, D = 1, 8, 128
N_WARMUP, N_ITER = 10, 50


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n_iter * 1_000


def main() -> None:
    rng = np.random.default_rng(42)
    print(f"[bench] rabitq_prefill_attend — VLM turn-2 shapes, B={B} H={H} D={D}")
    print(f"{'S_q':>5} {'S_kv':>6} | {'prefill (ms)':>12} | {'decode-k (ms)':>13} | "
          f"{'baseline (ms)':>13} | {'vs base':>7}")
    print("-" * 66)

    for S_q, S_kv in ((256, 2048), (256, 8192), (1024, 8192)):
        q = mx.array(rng.standard_normal((B, H, S_q, D)).astype(np.float16))
        scale = mx.array([1.0 / np.sqrt(D)], dtype=mx.float32)
        k_bits = mx.array(rng.integers(0, 256, (B, H, S_kv, D // 8), dtype=np.uint8))
        k_mag = mx.array(rng.uniform(0.05, 0.15, (B, H, S_kv)).astype(np.float32))
        k_const = mx.array(np.zeros((B, H, S_kv), dtype=np.float32))
        v_idx = mx.array(rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8))
        v_cents = mx.array(np.sort(rng.standard_normal(16)).astype(np.float32))
        v_packed = rabitq_pack_values(v_idx)
        # decode-kernel inputs (per-query scale in place of the scalar)
        q_scale = mx.array(
            np.full((B, H, S_q), 1.0 / np.sqrt(D), dtype=np.float32)
        )
        shifts = mx.arange(8, dtype=mx.uint8)
        mx.eval(q, scale, k_bits, k_mag, k_const, v_idx, v_cents, v_packed, q_scale)

        def prefill():
            return rabitq_prefill_attend(
                q, scale, k_bits, k_mag, k_const, v_packed, v_cents
            )

        def decode_k():
            return rabitq_fused_attend(
                q, q_scale, k_bits, k_mag, k_const, v_packed, v_cents
            )

        def baseline():
            bits = (k_bits[..., None] >> shifts) & 1
            signs = bits.reshape(B, H, S_kv, D).astype(mx.float16) * 2 - 1
            k_hat = signs * k_mag[..., None].astype(mx.float16)
            lo = (v_packed & 15).astype(mx.uint32)
            hi = (v_packed >> 4).astype(mx.uint32)
            idx = mx.stack([lo, hi], axis=-1).reshape(B, H, S_kv, D)
            v_hat = v_cents.astype(mx.float16)[idx]
            return mx.fast.scaled_dot_product_attention(
                q, k_hat, v_hat, scale=1.0 / float(D) ** 0.5
            )

        t_p = _bench(prefill)
        t_d = _bench(decode_k)
        t_b = _bench(baseline)
        print(f"{S_q:>5} {S_kv:>6} | {t_p:>12.3f} | {t_d:>13.3f} | "
              f"{t_b:>13.3f} | {t_b / t_p:>6.2f}x")


if __name__ == "__main__":
    main()

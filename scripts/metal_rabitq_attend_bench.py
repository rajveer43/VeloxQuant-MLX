"""Benchmark: fused RaBitQ asymmetric attend vs. dequantize-then-SDPA.

Compares two ways of running attention over an asymmetric RaBitQ cache
(1-bit packed keys + 4-bit codebook values):

  fused    — rabitq_fused_attend: one dispatch, keys scored from packed
             bits via XOR+popcount, values gathered from the codebook.
  baseline — the materializing path a cache would otherwise use: unpack
             bits -> K_hat [S_kv, D] fp16, gather V_hat [S_kv, D] fp16,
             then mx.fast.scaled_dot_product_attention.

Note: the two paths use different score math (Hamming inner-product
estimate vs exact fp16 dot on the dequantized keys), so this measures
the dispatch/materialization cost of each route, not output parity.

Usage: python scripts/metal_rabitq_attend_bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal.kernels import rabitq_fused_attend, rabitq_pack_values

B, H, S_q, D = 1, 8, 1, 128
N_WARMUP, N_ITER = 10, 100


def _bench(fn, n_warmup: int = N_WARMUP, n_iter: int = N_ITER) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n_iter * 1_000


def main() -> None:
    rng = np.random.default_rng(42)
    print(f"[bench] rabitq_fused_attend vs dequant+SDPA — B={B} H={H} S_q={S_q} D={D}")
    print(
        f"{'S_kv':>6} | {'fused (ms)':>10} | {'packed-V (ms)':>13} | "
        f"{'baseline (ms)':>13} | {'speedup':>7} | {'pk spd':>6}"
    )
    print("-" * 66)

    for S_kv in (512, 2048, 8192):
        q = mx.array(rng.standard_normal((B, H, S_q, D)).astype(np.float16))
        q_scale = mx.array((rng.uniform(0.05, 0.15, (B, H, S_q)) / np.sqrt(D)).astype(np.float32))
        k_bits = mx.array(rng.integers(0, 256, (B, H, S_kv, D // 8), dtype=np.uint8))
        k_mag = mx.array(rng.uniform(0.5, 1.5, (B, H, S_kv)).astype(np.float32))
        k_const = mx.array(np.zeros((B, H, S_kv), dtype=np.float32))
        v_idx = mx.array(rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8))
        v_cents = mx.array(np.sort(rng.standard_normal(16)).astype(np.float32))
        mx.eval(q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents)

        v_idx_packed = rabitq_pack_values(v_idx)
        mx.eval(v_idx_packed)

        def fused():
            return rabitq_fused_attend(q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents)

        def fused_packed():
            return rabitq_fused_attend(q, q_scale, k_bits, k_mag, k_const, v_idx_packed, v_cents)

        shifts = mx.arange(8, dtype=mx.uint8)

        def baseline():
            # Unpack 1-bit keys -> +-1 signs, scale by per-key magnitude.
            bits = (k_bits[..., None] >> shifts) & 1  # [B,H,S_kv,D/8,8]
            signs = bits.reshape(B, H, S_kv, D).astype(mx.float16) * 2 - 1
            k_hat = signs * k_mag[..., None].astype(mx.float16)
            v_hat = v_cents.astype(mx.float16)[v_idx]  # [B,H,S_kv,D]
            return mx.fast.scaled_dot_product_attention(
                q, k_hat, v_hat, scale=1.0 / float(D) ** 0.5
            )

        t_fused = _bench(fused)
        t_packed = _bench(fused_packed)
        t_base = _bench(baseline)
        print(
            f"{S_kv:>6} | {t_fused:>10.3f} | {t_packed:>13.3f} | {t_base:>13.3f} | "
            f"{t_base / t_fused:>6.2f}x | {t_base / t_packed:>5.2f}x"
        )


if __name__ == "__main__":
    main()

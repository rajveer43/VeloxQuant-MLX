"""Benchmark: simdgroup_matrix prefill attend vs the alternatives.

Two shape regimes:

  * VLM turn-2 (cross-attention, no mask): S_q new-turn tokens attending
    over S_kv compressed history slots (image tokens).
  * Causal self-attention prefill (S_q == S_kv): the from-scratch-prefill
    regime from issue #277 — no pre-existing cache, standard
    autoregressive masking.

Three ways to run each:

  prefill  — rabitq_prefill_attend: simdgroup_matrix 8x8 tiles, K/V
             decoded on the fly inside the tile loop.
  decode-k — rabitq_fused_attend (the decode-shaped kernel, packed V):
             correct at any S_q but one scalar-dot threadgroup per query
             (cross-attention shapes only — no causal masking support).
  baseline — dequantize everything (K_hat fp16 + V_hat fp16) and run
             mx.fast.scaled_dot_product_attention (causal=True for the
             self-attention rows).

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


def _run_shapes(shapes, *, causal: bool) -> None:
    rng = np.random.default_rng(42)
    label = "causal self-attention" if causal else "VLM turn-2 (cross-attention)"
    print(f"[bench] rabitq_prefill_attend — {label} shapes, B={B} H={H} D={D}")
    print(
        f"{'S_q':>5} {'S_kv':>6} | {'prefill (ms)':>12} | {'decode-k (ms)':>13} | "
        f"{'baseline (ms)':>13} | {'vs base':>7}"
    )
    print("-" * 66)

    for S_q, S_kv in shapes:
        q = mx.array(rng.standard_normal((B, H, S_q, D)).astype(np.float16))
        scale = mx.array([1.0 / np.sqrt(D)], dtype=mx.float32)
        k_bits = mx.array(rng.integers(0, 256, (B, H, S_kv, D // 8), dtype=np.uint8))
        k_mag = mx.array(rng.uniform(0.05, 0.15, (B, H, S_kv)).astype(np.float32))
        k_const = mx.array(np.zeros((B, H, S_kv), dtype=np.float32))
        v_idx = mx.array(rng.integers(0, 16, (B, H, S_kv, D), dtype=np.uint8))
        v_cents = mx.array(np.sort(rng.standard_normal(16)).astype(np.float32))
        v_packed = rabitq_pack_values(v_idx)
        # decode-kernel inputs (per-query scale in place of the scalar)
        q_scale = mx.array(np.full((B, H, S_q), 1.0 / np.sqrt(D), dtype=np.float32))
        shifts = mx.arange(8, dtype=mx.uint8)
        mx.eval(q, scale, k_bits, k_mag, k_const, v_idx, v_cents, v_packed, q_scale)

        def prefill():
            return rabitq_prefill_attend(
                q, scale, k_bits, k_mag, k_const, v_packed, v_cents, causal=causal
            )

        def decode_k():
            # No causal-masking support — cross-attention shapes only.
            return rabitq_fused_attend(q, q_scale, k_bits, k_mag, k_const, v_packed, v_cents)

        def baseline():
            bits = (k_bits[..., None] >> shifts) & 1
            signs = bits.reshape(B, H, S_kv, D).astype(mx.float16) * 2 - 1
            k_hat = signs * k_mag[..., None].astype(mx.float16)
            lo = (v_packed & 15).astype(mx.uint32)
            hi = (v_packed >> 4).astype(mx.uint32)
            idx = mx.stack([lo, hi], axis=-1).reshape(B, H, S_kv, D)
            v_hat = v_cents.astype(mx.float16)[idx]
            return mx.fast.scaled_dot_product_attention(
                q, k_hat, v_hat, scale=1.0 / float(D) ** 0.5, mask="causal" if causal else None
            )

        t_p = _bench(prefill)
        t_d = None if causal else _bench(decode_k)
        t_b = _bench(baseline)
        t_d_str = f"{t_d:>13.3f}" if t_d is not None else f"{'n/a':>13}"
        print(f"{S_q:>5} {S_kv:>6} | {t_p:>12.3f} | {t_d_str} | {t_b:>13.3f} | {t_b / t_p:>6.2f}x")


def main() -> None:
    _run_shapes([(256, 2048), (256, 8192), (1024, 8192)], causal=False)
    print()
    _run_shapes([(2048, 2048), (8192, 8192)], causal=True)


if __name__ == "__main__":
    main()

"""Before/after benchmark for turboquant_scalar_quantize's forced fp32 cast.

_scalar_quant.py:187 unconditionally does
    flat_x = x.reshape(-1).astype(mx.float32)
even when x is already fp16, adding one extra MLX dispatch (a cast kernel +
memory pass) before the Metal quantize kernel runs at all. The Metal kernel
body already does float(x[elem]) internally (_SCALAR_QUANTIZE_SRC line 1),
so accepting half* directly and letting Metal upcast in-register is safe.

This script isolates that specific overhead: fp16 input end-to-end through
turboquant_scalar_quantize, at realistic KV-cache shapes, before and after
the change. Run once before editing _scalar_quant.py to capture baseline,
then again after, to compute the delta.

Usage:
    PYTHONPATH=. python scripts/bench_scalar_quantize_dtype_cast.py
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

from veloxquant_mlx.metal._scalar_quant import turboquant_scalar_quantize


def _bench(fn, n_warmup: int = 10, n_iter: int = 100) -> float:
    for _ in range(n_warmup):
        mx.eval(fn())
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3  # ms


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"Device: {mx.default_device()}")
    print(f"{'shape (B,d)':<16s} {'b':>3s} {'fp16-in ms':>12s}")
    print(f"{'-' * 16} {'-' * 3} {'-' * 12}")

    # Realistic KV-cache quantize shapes: (batch*heads*seq, head_dim)
    cases = [
        (64, 128),  # small decode step
        (512, 128),  # medium seq
        (4096, 128),  # long context, head_dim=128
        (4096, 256),  # long context, head_dim=256 (Falcon3/Gemma-style)
        (16384, 128),  # very long context
    ]
    bits = [2, 4]

    results = {}
    for B, d in cases:
        for b in bits:
            n_cents = 1 << b
            cents = mx.array(np.linspace(-2.0, 2.0, n_cents).astype(np.float32))
            x = mx.array(rng.standard_normal((B, d)).astype(np.float16))

            ms = _bench(lambda _x=x, _c=cents, _b=b: turboquant_scalar_quantize(_x, _c, _b))
            results[(B, d, b)] = ms
            print(f"{f'{B}x{d}':<16s} {b:>3d} {ms:>12.4f}")

    return results


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

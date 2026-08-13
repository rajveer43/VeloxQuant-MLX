"""Entry point: python -m veloxquant_mlx benchmark"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _build_cache(args, vectorized: bool, fused: bool, outlier: bool):
    from veloxquant_mlx.cache.base import KVCacheBuilder

    builder = (
        KVCacheBuilder()
        .with_method(args.method)
        .with_head_dim(args.head_dim)
        .with_bit_width(inlier=args.bits)
        .with_jl_dim(args.jl_dim)
        .with_seed(args.seed)
        .with_vectorized_attend(vectorized)
        .with_fused_query_dot(fused)
        .with_outlier_two_stream(outlier)
        .with_n_outlier_channels(args.n_outlier_channels)
        .with_n_calib_tokens(args.n_calib_tokens)
    )
    return builder.build()


def _time_attend(cache, q, n_calls: int = 10) -> float:
    import mlx.core as mx

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = cache.attend(q)
        mx.eval(out)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_calls


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="veloxquant_mlx benchmark",
        description="Benchmark KV cache encode/decode latency and memory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="turboquant_prod",
        choices=["turboquant_prod", "turboquant_mse", "qjl", "polar"],
    )
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--jl_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--seq_lens", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_optimized", action="store_true")
    parser.add_argument("--n_outlier_channels", type=int, default=4)
    parser.add_argument("--n_calib_tokens", type=int, default=200)
    args = parser.parse_args()

    import mlx.core as mx

    d = args.head_dim
    seq_lens = args.seq_lens or [args.seq_len]
    rng = np.random.default_rng(args.seed)

    print("\n=== veloxquant_mlx benchmark ===")
    print(f"Method: {args.method}, head_dim={d}, bits={args.bits}, jl_dim={args.jl_dim}")
    print("seq_len | baseline_attend_ms | optimized_attend_ms | speedup")

    for seq_len in seq_lens:
        keys = mx.array(rng.standard_normal((seq_len, d)).astype(np.float16))
        vals = mx.array(rng.standard_normal((seq_len, d)).astype(np.float16))
        q = mx.array(rng.standard_normal(d).astype(np.float16))
        cache_base = _build_cache(args, vectorized=False, fused=False, outlier=False)
        for i in range(seq_len):
            cache_base.append(keys[i], vals[i])
        base_ms = _time_attend(cache_base, q)

        if args.compare_optimized:
            cache_opt = _build_cache(args, vectorized=True, fused=True, outlier=True)
            for i in range(seq_len):
                cache_opt.append(keys[i], vals[i])
            opt_ms = _time_attend(cache_opt, q)
            speed = base_ms / max(opt_ms, 1e-8)
            print(f"{seq_len:7d} | {base_ms:18.3f} | {opt_ms:19.3f} | {speed:7.3f}x")
        else:
            print(f"{seq_len:7d} | {base_ms:18.3f}")


if __name__ == "__main__":
    main()

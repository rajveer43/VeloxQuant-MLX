"""Offline-synthetic benchmark for StreamingLLM-adapted sink + recency-window eviction.

Sweeps (seq_len, window_size) combinations on synthetic K/V data shaped like
realistic attention heads, measuring streaming_ratio, tokens_in_window,
output shape stability, and ms/head throughput. No model loading.

NOTE: Results are from SYNTHETIC data — no real model has been run.
Until ``results_streaming_llm.json`` is committed with Apple Silicon hardware
numbers from an actual model forward pass, no throughput or perplexity figures
are claimed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.streaming_llm import (
    full_stream_fp16_bytes,
    init_streaming_window,
    stream_fp16_bytes,
    stream_update,
    stream_get_kv,
)


def _high_attention_outlier_kv(S: int, D: int, seed: int = 42) -> tuple[mx.array, mx.array]:
    """Synthetic K/V with a few high-norm tokens (realistic sink-like structure)."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((S, D)).astype(np.float32)
    # Make first 4 tokens high-magnitude (attention sinks)
    k[:4] *= 10.0
    v = rng.standard_normal((S, D)).astype(np.float32)
    return mx.array(k), mx.array(v)


def _benchmark_single(
    seq_len: int,
    D: int,
    n_sink: int,
    window_size: int,
    n_heads: int = 32,
    seed: int = 0,
) -> dict:
    k, v = _high_attention_outlier_kv(seq_len, D, seed=seed)

    results_per_head = []
    t0 = time.perf_counter()
    for h in range(n_heads):
        w = init_streaming_window(n_sink=n_sink, D=D)
        w = stream_update(w, k, v, n_sink=n_sink, window_size=window_size)
        ko, vo = stream_get_kv(w)
        kept_bytes = stream_fp16_bytes(w)
        full_bytes = full_stream_fp16_bytes(seq_len, D)
        results_per_head.append(
            {
                "n_in_window": w.n_sink + w.n_recent,
                "n_sink": w.n_sink,
                "n_recent": w.n_recent,
                "tokens_seen": w.tokens_seen,
                "kept_bytes": kept_bytes,
                "full_bytes": full_bytes,
                "ratio": full_bytes / max(kept_bytes, 1),
                "out_shape": list(ko.shape),
            }
        )
    dt_ms = (time.perf_counter() - t0) * 1000 / n_heads

    avg = lambda key: sum(r[key] for r in results_per_head) / n_heads
    return {
        "seq_len": seq_len,
        "D": D,
        "n_sink": n_sink,
        "window_size": window_size,
        "n_heads": n_heads,
        "avg_in_window": avg("n_in_window"),
        "avg_ratio": avg("ratio"),
        "avg_kept_bytes": avg("kept_bytes"),
        "avg_full_bytes": avg("full_bytes"),
        "ms_per_head": dt_ms,
        "out_shape_h0": results_per_head[0]["out_shape"],
        "note": "SYNTHETIC data — NOT YET RUN on dedicated Apple Silicon hardware",
    }


def main() -> None:
    D = 128
    n_sink = 4
    n_heads = 32

    seq_lens = [256, 512, 1024, 2048, 4096]
    window_sizes = [64, 128, 256, 512]

    rows = []
    for seq_len in seq_lens:
        for window_size in window_sizes:
            r = _benchmark_single(
                seq_len=seq_len,
                D=D,
                n_sink=n_sink,
                window_size=window_size,
                n_heads=n_heads,
            )
            rows.append(r)
            print(
                f"seq={seq_len:5d} win={window_size:4d} | "
                f"in_window={r['avg_in_window']:6.1f} | "
                f"ratio={r['avg_ratio']:.2f}x | "
                f"{r['ms_per_head']:.3f} ms/head"
            )

    out = Path(__file__).parent / "results_streaming_llm.json"
    out.write_text(
        json.dumps(
            {
                "note": "SYNTHETIC offline benchmark — NOT YET RUN on dedicated Apple Silicon hardware",
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

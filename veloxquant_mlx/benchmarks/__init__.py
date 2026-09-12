"""Benchmark scripts for measuring VeloxQuant-MLX performance and accuracy.

Groups standalone, runnable benchmarks (attention throughput, Metal kernel
microbenchmarks, RaBitQ/CacheRoute/comm-VQ comparisons, end-to-end model KV
benchmarks, and workload-replay traces) that are executed directly as
scripts rather than imported as a library API; this package holds no
re-exports of its own.
"""

from __future__ import annotations

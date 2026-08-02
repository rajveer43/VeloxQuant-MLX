"""Offline-synthetic benchmark for ChunkKV-adapted KV cache.

Sweeps (seq_len, budget, chunk_size, score_mode) on synthetic fp16 K/V data,
running each config through a single-layer ChunkKVCache and comparing it to a
token-level H2O-adapted cache at the *same* budget. Reports compression ratio,
kept tokens, a locality-coherence metric (how contiguous the surviving tokens
are), and an importance-retention proxy (share of high-attention-mass tokens
kept). No real model required.

Honest scope: on this synthetic data the *measured* facts are compression ratio,
kept-token count, and eviction latency. The ``coherence`` columns (contiguity of
survivors) are reported as a diagnostic — on a proxy scorer the token-level H2O
baseline already tends to keep contiguous survivors, so the coherence *gain* is
near zero here. ChunkKV's real semantic-coherence advantage is a property of true
attention on real prompts, which this model-free harness cannot exhibit; do not
read the coherence columns as an end-to-end quality win. The load-bearing result
is that larger chunks cut eviction passes (and wall-clock) sharply while holding
compression, and that chunk_size=1 reproduces H2O exactly.

Usage
-----
    python benchmark_scripts/benchmark_chunkkv.py

Prints a table and saves a JSON summary. Wall-clock is dominated by the O(S^2)
pure-Python eviction loop (a prefill worst case), not a per-decode-step cost.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
from veloxquant_mlx.cache.h2o_cache import H2OKVCache

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [256, 512, 1024]
BUDGETS = [64, 128]
CHUNK_SIZES = [1, 4, 8, 16]
SCORE_MODES = ["attn_mass", "key_norm"]
N_HEADS = 8
HEAD_DIM = 128
N_SINK = 4


def _synthetic_kv(S: int, seed: int = 0):
    """One layer's K/V: a few salient directions embedded in a broad background.

    A handful of contiguous spans share a strong direction (salient chunks); the
    rest is near-random. This gives eviction a real signal to preserve and lets
    the coherence metric distinguish chunk- from token-level survival.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32)
    # Inject 3 salient contiguous spans with a shared strong direction.
    direction = rng.standard_normal((HEAD_DIM,)).astype(np.float32)
    direction /= np.linalg.norm(direction)
    for start in (S // 6, S // 2, (4 * S) // 5):
        span = slice(start, min(start + S // 12, S))
        base[:, :, span, :] += 4.0 * direction
    K = mx.array(base.astype(np.float16))
    V = mx.array(rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float16))
    mx.eval(K, V)
    return K, V


def _coherence(kept_keys: mx.array, ref_keys: mx.array) -> float:
    """Fraction of contiguity retained among survivors (1 = one solid run).

    Recovers each survivor's original position by matching rows against the full
    key set, then counts the runs of consecutive positions. coherence =
    1 - (runs - 1) / (n_kept - 1); a single contiguous run scores 1.0, maximally
    fragmented (every survivor isolated) scores 0.0.
    """
    n_kept = int(kept_keys.shape[0])
    if n_kept <= 1:
        return 1.0
    ref = np.asarray(ref_keys.astype(mx.float32), dtype=np.float32)  # [S, D]
    kept = np.asarray(kept_keys.astype(mx.float32), dtype=np.float32)  # [n, D]
    # Match each kept row to its original index (exact fp16 rows → nearest).
    pos = []
    for r in kept:
        pos.append(int(np.argmin(np.sum((ref - r) ** 2, axis=1))))
    pos = sorted(set(pos))
    runs = 1
    for a, b in zip(pos, pos[1:]):
        if b != a + 1:
            runs += 1
    return 1.0 - (runs - 1) / max(len(pos) - 1, 1)


def _run_once(seq_len, budget, chunk_size, score_mode) -> dict:
    K, V = _synthetic_kv(seq_len, seed=chunk_size + budget)
    ref_keys = K[0, 0]  # [S, D] for position recovery (head 0)

    cfg = KVCacheConfig(
        method="chunkkv",
        head_dim=HEAD_DIM,
        chunkkv_budget=budget,
        chunkkv_chunk_size=chunk_size,
        chunkkv_n_sink=N_SINK,
        chunkkv_score=score_mode,
    )
    cache = ChunkKVCache(cfg)
    t0 = time.perf_counter()
    Ko, Vo = cache.update_and_fetch(K, V)
    mx.eval(Ko, Vo)
    latency_ms = (time.perf_counter() - t0) * 1_000

    coh = _coherence(cache._states[0].keys, ref_keys)

    # Token-level H2O baseline at the same budget (attn_mass only — H2O has no
    # key_norm mode; we compare against it for both to show the chunk delta).
    h2o = H2OKVCache(
        KVCacheConfig(method="h2o", head_dim=HEAD_DIM, h2o_budget=budget, h2o_n_sink=N_SINK)
    )
    Kh, Vh = h2o.update_and_fetch(K, V)
    mx.eval(Kh, Vh)
    coh_h2o = _coherence(h2o._states[0].keys, ref_keys)

    return {
        "seq_len": seq_len,
        "budget": budget,
        "chunk_size": chunk_size,
        "score_mode": score_mode,
        "tokens_kept": cache.tokens_kept,
        "compression_ratio": round(cache.compression_ratio, 3),
        "coherence": round(coh, 3),
        "coherence_h2o": round(coh_h2o, 3),
        "coherence_gain": round(coh - coh_h2o, 3),
        "latency_ms": round(latency_ms, 2),
    }


def main() -> None:
    print("ChunkKV-adapted KV Cache — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  n_sink={N_SINK}")
    print("  (coherence = contiguity of survivors; H2O = token-level baseline)")
    print("  (chunk_size=1 + attn_mass == H2O by construction)")
    print()
    header = (
        f"{'seq':>5}  {'budget':>6}  {'chunk':>5}  {'score':>10}  "
        f"{'kept':>5}  {'ratio':>6}  {'coher':>6}  {'coh_h2o':>7}  "
        f"{'gain':>6}  {'ms':>7}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for seq_len, budget, chunk_size, score_mode in product(
        SEQ_LENS, BUDGETS, CHUNK_SIZES, SCORE_MODES
    ):
        if budget >= seq_len:
            continue
        row = _run_once(seq_len, budget, chunk_size, score_mode)
        results.append(row)
        print(
            f"{row['seq_len']:>5}  {row['budget']:>6}  {row['chunk_size']:>5}  "
            f"{row['score_mode']:>10}  {row['tokens_kept']:>5}  "
            f"{row['compression_ratio']:>5.2f}x  {row['coherence']:>6.3f}  "
            f"{row['coherence_h2o']:>7.3f}  {row['coherence_gain']:>+6.3f}  "
            f"{row['latency_ms']:>7.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "chunkkv" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

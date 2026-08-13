"""Benchmark: RaBitQ 1-bit ANN search (Metal Hamming kernel) vs fp16 exact.

Sweeps S_kv = [64, 128, 256, 512, 1024, 2048, 4096] for H=8, D=128.
Saves 4 figures to figures/metal/rabitq/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer

FIGURES_DIR = Path(__file__).parents[2] / "figures" / "metal" / "rabitq"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

D = 128
H = 8
NLIST = 64
NPROBE = 8
RERANK = 32
TOP_K = 10
S_KVS = [64, 128, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _bench_mlx(fn, n_warmup: int = 5, n_iter: int = 20) -> float:
    for _ in range(n_warmup):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
    return (time.perf_counter() - t0) / n_iter * 1e3


def _bench_np(fn, n_warmup: int = 5, n_iter: int = 20) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter * 1e3


# ---------------------------------------------------------------------------
# NumPy baseline: exact fp32 dot over all candidates
# ---------------------------------------------------------------------------


def _np_exact_search(corpus_np: np.ndarray, query_np: np.ndarray, top_k: int) -> np.ndarray:
    scores = corpus_np @ query_np
    return np.argsort(-scores)[:top_k]


# ---------------------------------------------------------------------------
# Recall helper
# ---------------------------------------------------------------------------


def _recall_at_k(result: np.ndarray, true_top: np.ndarray, k: int) -> float:
    return len(set(result[:k].tolist()) & set(true_top[:k].tolist())) / k


# ---------------------------------------------------------------------------
# Build quantizer once
# ---------------------------------------------------------------------------


def _build_quantizer(calib_np: np.ndarray) -> RaBitQQuantizer:
    q = RaBitQQuantizer(d=D, nlist=NLIST, nprobe=NPROBE, rerank=RERANK, seed=42)
    q.fit(mx.array(calib_np), max_samples=min(4096, len(calib_np)))
    return q


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(n_iter: int = 20) -> dict:
    rng = np.random.default_rng(0)
    calib = rng.standard_normal((4096, D)).astype(np.float16)
    print(f"Building RaBitQ quantizer (D={D}, nlist={NLIST}, nprobe={NPROBE})...")
    q = _build_quantizer(calib)

    results = {
        "config": {"D": D, "H": H, "nlist": NLIST, "nprobe": NPROBE, "rerank": RERANK},
        "S_kvs": S_KVS,
        "rabitq_ms": [],
        "fp16_exact_ms": [],
        "numpy_exact_ms": [],
        "speedup_vs_numpy": [],
        "recall_at_10": [],
    }

    print(
        f"\n{'S_kv':>6}  {'RaBitQ(ms)':>11}  {'fp16(ms)':>9}  {'NumPy(ms)':>10}  {'Speedup':>8}  {'Recall@10':>10}"
    )
    print("-" * 66)

    for S_kv in S_KVS:
        N = H * S_kv
        rng2 = np.random.default_rng(S_kv + 1)
        corpus_np = rng2.standard_normal((N, D)).astype(np.float16)
        query_np = rng2.standard_normal(D).astype(np.float16)
        corpus_mx = mx.array(corpus_np)
        query_mx = mx.array(query_np)

        # Encode corpus
        ev = q.encode(corpus_mx)
        mx.eval(ev.indices, ev.norm)

        # --- RaBitQ search (Metal) ---
        def _rabitq():
            return q.search(query_mx, ev, top_k=TOP_K)

        # --- fp16 exact dot (MLX) ---
        def _fp16_exact():
            scores = corpus_mx.astype(mx.float32) @ query_mx.astype(mx.float32)
            return mx.argsort(-scores)[:TOP_K]

        # --- NumPy exact ---
        corpus_f32 = corpus_np.astype(np.float32)
        query_f32 = query_np.astype(np.float32)

        def _np_exact():
            return _np_exact_search(corpus_f32, query_f32, TOP_K)

        t_rabitq = _bench_mlx(_rabitq, n_iter=n_iter)
        t_fp16 = _bench_mlx(_fp16_exact, n_iter=n_iter)
        t_np = _bench_np(_np_exact, n_iter=n_iter)
        speedup = t_np / t_rabitq

        # Recall — RaBitQ approximates L2, so use L2 ground truth
        rabitq_result = np.array(q.search(query_mx, ev, top_k=TOP_K))
        l2_dists = np.sum((corpus_f32 - query_f32[None, :]) ** 2, axis=1)
        true_top = np.argsort(l2_dists)
        recall = _recall_at_k(rabitq_result, true_top, TOP_K)

        results["rabitq_ms"].append(round(t_rabitq, 3))
        results["fp16_exact_ms"].append(round(t_fp16, 3))
        results["numpy_exact_ms"].append(round(t_np, 3))
        results["speedup_vs_numpy"].append(round(speedup, 2))
        results["recall_at_10"].append(round(recall, 3))

        print(
            f"{S_kv:>6}  {t_rabitq:>11.3f}  {t_fp16:>9.3f}  {t_np:>10.3f}  {speedup:>7.2f}x  {recall:>10.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(results: dict) -> None:
    S_kvs = results["S_kvs"]
    t_rb = results["rabitq_ms"]
    t_fp16 = results["fp16_exact_ms"]
    t_np = results["numpy_exact_ms"]
    speedup = results["speedup_vs_numpy"]
    recall = results["recall_at_10"]

    # Fig 1: Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_kvs, t_rb, "o-", label="RaBitQ (Metal)", color="steelblue")
    ax.plot(S_kvs, t_fp16, "s--", label="fp16 exact (MLX)", color="darkorange")
    ax.plot(S_kvs, t_np, "^:", label="NumPy exact (CPU)", color="green")
    ax.set_xlabel("KV sequence length (S_kv × H=8 heads)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"RaBitQ Search Latency vs Exact\n(D={D}, nlist={NLIST}, nprobe={NPROBE})")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig1_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 2: Speedup vs NumPy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in S_kvs], speedup, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", label="1× breakeven")
    ax.set_xlabel("S_kv")
    ax.set_ylabel("Speedup (NumPy exact / RaBitQ Metal)")
    ax.set_title("RaBitQ Metal Speedup over NumPy Exact Search")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig2_speedup.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 3: Recall@10 vs S_kv
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(S_kvs, recall, "o-", color="darkgreen")
    ax.axhline(0.5, color="red", linestyle="--", label="0.5 target")
    ax.set_xlabel("S_kv")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"RaBitQ Recall@10 vs Corpus Size\n(nprobe={NPROBE}, rerank={RERANK})")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig3_recall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # Fig 4: Memory comparison bar
    d = results["config"]["D"]
    fp16_bytes = d * 2
    rbitq_bytes = d // 8
    cr = fp16_bytes / rbitq_bytes
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["fp16 keys", "RaBitQ 1-bit"]
    values = [fp16_bytes, rbitq_bytes]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%d bytes/token", padding=3)
    ax.set_ylabel("Bytes per token")
    ax.set_title(f"RaBitQ Memory: {cr:.0f}× compression\n(D={d})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig4_memory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RaBitQ benchmark")
    parser.add_argument("--n_iter", type=int, default=20)
    args = parser.parse_args()

    results = run_benchmark(n_iter=args.n_iter)
    save_figures(results)

    out_path = FIGURES_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

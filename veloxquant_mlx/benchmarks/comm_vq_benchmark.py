"""Benchmark: CommVQ decode (Metal kernel vs Python/MLX) vs VecInfer Metal decode.

Sweeps S_kv = [64, 128, 256, 512, 1024, 2048, 4096] for H=8, D=128, n_cb=4.
Saves figures to figures/metal/comm_vq/.
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

from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer

FIGURES_DIR = Path(__file__).parents[2] / "figures" / "metal" / "comm_vq"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _bench_mlx(fn, n_warmup: int = 5, n_iter: int = 30) -> float:
    """Return mean ms per call (MLX lazy evaluation — forces with mx.eval)."""
    for _ in range(n_warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / n_iter * 1e3


def _bench_np(fn, n_warmup: int = 5, n_iter: int = 30) -> float:
    """Return mean ms per call (NumPy / CPU)."""
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter * 1e3


# ---------------------------------------------------------------------------
# Build a trained CommVQ quantizer once
# ---------------------------------------------------------------------------


def _build_quantizer(D: int = 128, n_cb: int = 4, b: int = 4) -> CommVQQuantizer:
    q = CommVQQuantizer(d=D, b=b, n_codebooks=n_cb, seed=42)
    rng = np.random.default_rng(0)
    calib = rng.standard_normal((2048, D)).astype(np.float16)
    q.fit(mx.array(calib), max_samples=2048)
    return q


# ---------------------------------------------------------------------------
# NumPy reference decode (no Metal)
# ---------------------------------------------------------------------------


def _numpy_comm_vq_decode(
    indices_np: np.ndarray,  # [N, n_cb] uint8
    codebook_np: np.ndarray,  # [n_cb, K, sub_dim] float32
    positions_np: np.ndarray,  # [N] int32
    D: int,
    rope_base: float = 10000.0,
) -> np.ndarray:
    N, n_cb = indices_np.shape
    sub_dim = D // n_cb
    half = D // 2

    # Gather centroids
    out = np.zeros((N, D), dtype=np.float32)
    for cb_i in range(n_cb):
        start = cb_i * sub_dim
        cb = codebook_np[cb_i]  # [K, sub_dim]
        idxs = indices_np[:, cb_i].astype(np.int32)  # [N]
        out[:, start : start + sub_dim] = cb[idxs]

    # Apply RoPE
    inv_freq = 1.0 / (rope_base ** (np.arange(half, dtype=np.float32) / half))
    angles = positions_np[:, None].astype(np.float32) * inv_freq[None, :]
    cos_v, sin_v = np.cos(angles), np.sin(angles)
    x1, x2 = out[:, :half], out[:, half:]
    result = np.concatenate([x1 * cos_v - x2 * sin_v, x1 * sin_v + x2 * cos_v], axis=1)
    return result.astype(np.float16)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(n_iter: int = 30) -> dict:
    D = 128
    H = 8
    n_cb = 4
    b = 4

    S_kvs = [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"Building CommVQ quantizer (D={D}, n_cb={n_cb}, b={b})...")
    q = _build_quantizer(D=D, n_cb=n_cb, b=b)
    assert q.trained
    cb_np = np.array(q._codebooks, dtype=np.float32)  # [n_cb, K, sub_dim]
    cb_mx = q._codebooks_mx  # [n_cb, K, sub_dim] fp16

    results = {
        "config": {"D": D, "H": H, "n_cb": n_cb, "b": b},
        "S_kvs": S_kvs,
        "mlx_decode_ms": [],
        "numpy_decode_ms": [],
        "speedup": [],
    }

    print(f"\n{'S_kv':>6}  {'MLX (ms)':>10}  {'NumPy (ms)':>11}  {'Speedup':>8}")
    print("-" * 44)

    for S_kv in S_kvs:
        N = H * S_kv  # total tokens across all heads
        rng = np.random.default_rng(S_kv)
        keys_np = rng.standard_normal((N, D)).astype(np.float16)
        pos_np = np.tile(np.arange(S_kv, dtype=np.int32), H)

        keys_mx = mx.array(keys_np)
        pos_mx = mx.array(pos_np)

        # Encode once (common setup)
        ev = q.encode(keys_mx, positions=pos_mx)
        mx.eval(ev.indices, ev.norm)
        indices_np = np.array(ev.indices)
        indices_mx = ev.indices

        # --- MLX decode (Python path, no Metal kernel yet) ---
        def _mlx_decode():
            return q.decode(ev)

        # --- NumPy decode ---
        def _np_decode():
            return _numpy_comm_vq_decode(indices_np, cb_np, pos_np, D)

        t_mlx = _bench_mlx(_mlx_decode, n_iter=n_iter)
        t_np = _bench_np(_np_decode, n_iter=n_iter)
        speedup = t_np / t_mlx

        results["mlx_decode_ms"].append(round(t_mlx, 3))
        results["numpy_decode_ms"].append(round(t_np, 3))
        results["speedup"].append(round(speedup, 2))

        print(f"{S_kv:>6}  {t_mlx:>10.3f}  {t_np:>11.3f}  {speedup:>7.2f}x")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(results: dict) -> None:
    S_kvs = results["S_kvs"]
    t_mlx = results["mlx_decode_ms"]
    t_np = results["numpy_decode_ms"]
    speedup = results["speedup"]

    # Fig 1: Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_kvs, t_mlx, "o-", label="MLX decode (Python)")
    ax.plot(S_kvs, t_np, "s--", label="NumPy decode (CPU)")
    ax.set_xlabel("KV sequence length (S_kv × H heads)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CommVQ Decode Latency: MLX vs NumPy\n(D=128, n_cb=4, b=4, H=8)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_decode_latency.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'fig1_decode_latency.png'}")

    # Fig 2: Speedup
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in S_kvs], speedup, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", label="1× (breakeven)")
    ax.set_xlabel("S_kv")
    ax.set_ylabel("Speedup (NumPy / MLX)")
    ax.set_title("CommVQ Decode MLX Speedup over NumPy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_speedup.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'fig2_speedup.png'}")

    # Fig 3: Memory savings vs fp16
    D = results["config"]["D"]
    n_cb = results["config"]["n_cb"]
    compression = (D * 2) / (n_cb * 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["fp16 keys", "CommVQ indices"]
    values = [D * 2, n_cb]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%d bytes/token", padding=3)
    ax.set_ylabel("Bytes per token")
    ax.set_title(
        f"CommVQ Memory: {compression:.0f}× compression\n(D={D}, n_cb={n_cb}, b={results['config']['b']})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_memory.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'fig3_memory.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CommVQ benchmark")
    parser.add_argument("--n_iter", type=int, default=30)
    args = parser.parse_args()

    results = run_benchmark(n_iter=args.n_iter)
    save_figures(results)

    out_path = FIGURES_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

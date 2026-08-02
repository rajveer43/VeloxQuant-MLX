"""Benchmark the VecInfer Metal kernels and save the result figures.

Produces three figures under ``figures/metal/``:

  1. ``quantize_throughput.png`` — pure-MLX vs Metal latency across
     realistic shapes (B=1, H in {4,8}, S in {128..8192}).
  2. ``quantize_memory.png`` — peak memory at the Falcon3-7B OOM
     trigger shape (head_dim=256, n_centroids=256, sub_dim=4).
  3. ``summary.png`` — combined 2-panel figure for the README/blog.

Plus a ``results.json`` rollup of the raw numbers.

Run from repo root:

    PYTHONPATH=. python scripts/plot_metal_benchmarks.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from veloxquant_mlx.allocators.vecinfer import dequantize_vq, quantize_vq
from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import (
    vecinfer_dequant_metal,
    vecinfer_quantize_metal,
)

OUT_DIR = Path("figures/metal")

# Match the proof-script shape set so the README/blog numbers line up.
QUANT_SHAPES = [
    # (B, H, S, D, sub_dim, n_centroids)
    (1, 8, 128, 128, 8, 256),
    (1, 8, 512, 128, 8, 256),
    (1, 8, 2048, 128, 8, 256),
    (1, 8, 8192, 128, 8, 256),
    (1, 4, 1024, 256, 8, 256),
    (1, 4, 4096, 256, 8, 256),
]

DEQUANT_SHAPES = [
    # (B, H, S, n_sub, sub_dim, n_centroids)
    (1, 8, 128, 16, 8, 256),
    (1, 8, 512, 16, 8, 256),
    (1, 8, 2048, 16, 8, 256),
    (1, 8, 8192, 16, 8, 256),
    (1, 4, 2048, 32, 8, 256),
    (1, 4, 8192, 32, 8, 256),
]

MEM_SHAPE = (1, 4, 4096, 256, 4, 256)  # the OOM trigger

# Color palette — match the existing landing page / VecInfer summary.
C_PURE = "#4C72B0"  # pure-MLX (blue)
C_METAL = "#7c3aed"  # metal (purple, matches landing page accent)
C_OK = "#4ade80"  # green for memory savings
GRID_A = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _peak_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def _shape_label(B, H, S, D_or_sub, *_rest) -> str:
    return f"S={S}, D={D_or_sub}"


def _bench(fn, *args, iters: int = 30, warmup: int = 3) -> float:
    """Return median wall time in seconds."""
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Quantize: throughput + memory
# ---------------------------------------------------------------------------
def measure_quantize() -> dict:
    print("\n=== Quantize: pure-MLX vs Metal ===")
    rng = np.random.default_rng(42)

    results = []
    for B, H, S, D, sub_dim, n_c in QUANT_SHAPES:
        x_np = rng.standard_normal((B, H, S, D)).astype(np.float32)
        cb_np = rng.standard_normal((n_c, sub_dim)).astype(np.float32)
        x = mx.array(x_np).astype(mx.float16)
        cb = mx.array(cb_np).astype(mx.float16)

        t_pure = _bench(quantize_vq, x, cb, sub_dim)
        t_metal = _bench(vecinfer_quantize_metal, x, cb, sub_dim)
        speedup = t_pure / t_metal if t_metal > 0 else float("inf")

        row = {
            "B": B,
            "H": H,
            "S": S,
            "D": D,
            "sub_dim": sub_dim,
            "n_centroids": n_c,
            "pure_ms": t_pure * 1e3,
            "metal_ms": t_metal * 1e3,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"  S={S:>4d} D={D:>3d}: pure={t_pure * 1e3:7.2f} ms  "
            f"metal={t_metal * 1e3:6.2f} ms  speedup={speedup:5.2f}x"
        )

    # Memory at the OOM shape
    print("\n=== Quantize: peak memory at Falcon3-7B OOM shape ===")
    B, H, S, D, sub_dim, n_c = MEM_SHAPE
    x = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)
    cb = mx.array(rng.standard_normal((n_c, sub_dim)).astype(np.float32)).astype(mx.float16)
    mx.eval(x, cb)

    _reset_peak()
    mx.clear_cache()
    out_p = quantize_vq(x, cb, sub_dim)
    mx.eval(out_p)
    peak_pure = _peak_mb()
    del out_p
    mx.clear_cache()

    _reset_peak()
    out_m = vecinfer_quantize_metal(x, cb, sub_dim)
    mx.eval(out_m)
    peak_metal = _peak_mb()
    del out_m
    mx.clear_cache()

    print(f"  pure-MLX peak: {peak_pure:.1f} MB")
    print(f"  metal peak:    {peak_metal:.1f} MB")

    return {
        "shapes": results,
        "memory": {
            "shape": dict(zip(("B", "H", "S", "D", "sub_dim", "n_centroids"), MEM_SHAPE)),
            "pure_mb": peak_pure,
            "metal_mb": peak_metal,
            "reduction_pct": 100.0 * (peak_pure - peak_metal) / max(peak_pure, 1e-9),
        },
    }


# ---------------------------------------------------------------------------
# Dequantize: throughput (memory parity — same gather)
# ---------------------------------------------------------------------------
def measure_dequantize() -> dict:
    print("\n=== Dequantize: pure-MLX vs Metal ===")
    rng = np.random.default_rng(42)
    results = []
    for B, H, S, n_sub, sub_dim, n_c in DEQUANT_SHAPES:
        indices = mx.array(rng.integers(0, n_c, size=(B, H, S, n_sub), dtype=np.int32))
        cb = mx.array(rng.standard_normal((n_c, sub_dim)).astype(np.float32)).astype(mx.float16)

        t_pure = _bench(dequantize_vq, indices, cb)
        t_metal = _bench(vecinfer_dequant_metal, indices, cb)
        speedup = t_pure / t_metal if t_metal > 0 else float("inf")

        row = {
            "B": B,
            "H": H,
            "S": S,
            "n_sub": n_sub,
            "sub_dim": sub_dim,
            "n_centroids": n_c,
            "pure_ms": t_pure * 1e3,
            "metal_ms": t_metal * 1e3,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"  S={S:>4d} n_sub={n_sub:>2d}: pure={t_pure * 1e3:6.2f} ms  "
            f"metal={t_metal * 1e3:6.2f} ms  speedup={speedup:5.2f}x"
        )
    return {"shapes": results}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _grouped_bar(
    ax, labels, pure_vals, metal_vals, ylabel, title, fmt: str = ".2f", log: bool = False
):
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(
        x - w / 2, pure_vals, w, label="pure-MLX", color=C_PURE, edgecolor="white", linewidth=0.6
    )
    b2 = ax.bar(
        x + w / 2,
        metal_vals,
        w,
        label="Metal kernel",
        color=C_METAL,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_A)
    if log:
        ax.set_yscale("log")
    # value labels
    for bars, vals in ((b1, pure_vals), (b2, metal_vals)):
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.05 if not log else v * 1.15,
                f"{v:{fmt}}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    ax.legend(fontsize=9, loc="upper left")


def plot_quantize_throughput(data: dict) -> Path:
    rows = data["shapes"]
    labels = [f"S={r['S']}\nD={r['D']}" for r in rows]
    pure = [r["pure_ms"] for r in rows]
    metal = [r["metal_ms"] for r in rows]
    speedups = [r["speedup"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "VecInfer Metal kernel — quantize throughput\n"
        "Lower is better. Speedup labels on right panel.",
        fontsize=13,
        fontweight="bold",
    )

    _grouped_bar(
        axes[0],
        labels,
        pure,
        metal,
        "Median latency (ms, log)",
        "Per-call latency",
        fmt=".2f",
        log=True,
    )

    # Speedup panel
    x = np.arange(len(labels))
    bars = axes[1].bar(x, speedups, color=C_METAL, edgecolor="white", linewidth=0.8)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="parity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("Speedup (×)", fontsize=11)
    axes[1].set_title("Metal speedup over pure-MLX", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=GRID_A)
    axes[1].legend(fontsize=9)
    for b, v in zip(bars, speedups):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            v + 0.3,
            f"{v:.1f}x",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(speedups) * 1.18)

    plt.tight_layout()
    out = OUT_DIR / "quantize_throughput.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def plot_quantize_memory(data: dict) -> Path:
    mem = data["memory"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["pure-MLX", "Metal kernel"],
        [mem["pure_mb"], mem["metal_mb"]],
        color=[C_PURE, C_METAL],
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    ax.set_ylabel("Peak memory (MB)", fontsize=11)
    ax.set_title(
        "Peak memory at Falcon3-7B OOM trigger shape\n"
        f"head_dim={mem['shape']['D']}, n_centroids={mem['shape']['n_centroids']}, "
        f"sub_dim={mem['shape']['sub_dim']}, seq_len={mem['shape']['S']}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=GRID_A)
    for b, v in zip(bars, [mem["pure_mb"], mem["metal_mb"]]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(mem["pure_mb"], 1) * 0.02,
            f"{v:.1f} MB",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.text(
        0.5,
        0.92,
        f"{mem['reduction_pct']:.1f}% reduction ({mem['pure_mb'] - mem['metal_mb']:.0f} MB saved)",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        color=C_OK,
        fontweight="bold",
        bbox=dict(facecolor="#0f1a12", edgecolor=C_OK, boxstyle="round,pad=0.4"),
    )
    out = OUT_DIR / "quantize_memory.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def plot_dequantize_throughput(data: dict) -> Path:
    rows = data["shapes"]
    labels = [f"S={r['S']}\nn_sub={r['n_sub']}" for r in rows]
    pure = [r["pure_ms"] for r in rows]
    metal = [r["metal_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "VecInfer Metal kernel — dequantize (gather) — parity check", fontsize=13, fontweight="bold"
    )
    _grouped_bar(
        ax, labels, pure, metal, "Median latency (ms)", "Per-call latency", fmt=".2f", log=False
    )
    out = OUT_DIR / "dequantize_throughput.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


def plot_summary(quant: dict, dequant: dict) -> Path:
    """The headline figure for README / blog / landing page."""
    rows = quant["shapes"]
    labels = [f"S={r['S']}, D={r['D']}" for r in rows]
    pure_ms = [r["pure_ms"] for r in rows]
    metal_ms = [r["metal_ms"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    mem = quant["memory"]

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])
    fig.suptitle(
        "VecInfer Metal kernel — Phase 1 results (v0.5.1)\n"
        "Apple Silicon · pure-MLX vs hand-written Metal",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1 — throughput (log y)
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(labels))
    w = 0.38
    ax1.bar(x - w / 2, pure_ms, w, label="pure-MLX", color=C_PURE, edgecolor="white", linewidth=0.6)
    ax1.bar(
        x + w / 2,
        metal_ms,
        w,
        label="Metal kernel",
        color=C_METAL,
        edgecolor="white",
        linewidth=0.6,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax1.set_ylabel("Latency (ms, log)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_title("Quantize — per-call latency", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=GRID_A, which="both")
    ax1.legend(fontsize=9, loc="upper left")

    # Panel 2 — speedup
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(x, speedups, color=C_METAL, edgecolor="white", linewidth=0.8)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Speedup (×)", fontsize=11)
    ax2.set_title("Quantize — speedup over pure-MLX", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=GRID_A)
    for b, v in zip(bars, speedups):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + max(speedups) * 0.02,
            f"{v:.1f}x",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(speedups) * 1.18)

    # Panel 3 — memory at OOM shape
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(
        ["pure-MLX", "Metal"],
        [mem["pure_mb"], mem["metal_mb"]],
        color=[C_PURE, C_METAL],
        edgecolor="white",
        linewidth=0.8,
        width=0.5,
    )
    ax3.set_ylabel("Peak memory (MB)", fontsize=11)
    ax3.set_title(
        f"Memory at OOM shape\nhead_dim={mem['shape']['D']}, sub_dim={mem['shape']['sub_dim']}",
        fontsize=11,
        fontweight="bold",
    )
    ax3.grid(axis="y", alpha=GRID_A)
    for b, v in zip(bars3, [mem["pure_mb"], mem["metal_mb"]]):
        ax3.text(
            b.get_x() + b.get_width() / 2,
            v + max(mem["pure_mb"], 1) * 0.02,
            f"{v:.0f} MB",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax3.text(
        0.5,
        0.78,
        f"−{mem['reduction_pct']:.1f}%",
        transform=ax3.transAxes,
        ha="center",
        fontsize=18,
        color=C_OK,
        fontweight="bold",
    )

    plt.tight_layout()
    out = OUT_DIR / "summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not metal_available():
        print("Metal not available — aborting.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {mx.default_device()}")
    print(f"Output dir: {OUT_DIR.resolve()}")

    quant = measure_quantize()
    dequant = measure_dequantize()

    print("\n=== Saving figures ===")
    paths = [
        plot_quantize_throughput(quant),
        plot_quantize_memory(quant),
        plot_dequantize_throughput(dequant),
        plot_summary(quant, dequant),
    ]
    for p in paths:
        print(f"  {p}")

    rollup = OUT_DIR / "results.json"
    with open(rollup, "w") as f:
        json.dump({"quantize": quant, "dequantize": dequant}, f, indent=2)
    print(f"  {rollup}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

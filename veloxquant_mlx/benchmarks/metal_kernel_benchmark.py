"""Benchmark all TurboQuant Metal kernels vs pure-MLX/NumPy baselines.

Produces figures saved to figures/metal/turboquant_kernels/:

  fig1_bit_pack_throughput.png      — bit_pack / bit_unpack GB/s vs N, b
  fig2_scalar_quant_throughput.png  — scalar_quantize / dequantize GB/s vs N, b
  fig3_hadamard_quantize.png        — fused Hadamard quant: ms vs D, B × D comparison
  fig4_qjl_encode_throughput.png    — qjl_encode: Metal vs reference ms vs B
  fig5_qjl_ip_throughput.png        — qjl_inner_product: Metal ms vs S_kv
  fig6_rvq_attend_throughput.png    — fused RVQ attend: Metal ms vs S_kv
  fig7_memory_savings.png           — memory footprint: fp16 vs b-bit compressed
  fig8_summary_speedup.png          — unified speedup bar chart across all kernels

Usage::

    python -m veloxquant_mlx.benchmarks.metal_kernel_benchmark
    python -m veloxquant_mlx.benchmarks.metal_kernel_benchmark --n_iter 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import mlx.core as mx

from veloxquant_mlx.metal._bit_packing import turboquant_bit_pack, turboquant_bit_unpack
from veloxquant_mlx.metal._scalar_quant import (
    turboquant_scalar_quantize,
    turboquant_scalar_dequantize,
    turboquant_hadamard_quantize,
)
from veloxquant_mlx.metal._qjl import qjl_encode, qjl_inner_product
from veloxquant_mlx.metal._rvq_attend import turboquant_fused_rvq_decode_attend

OUT_DIR = Path(__file__).parents[2] / "figures" / "metal" / "turboquant_kernels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "metal": dict(color="#2196F3", linewidth=2.0, marker="o", markersize=5),
    "numpy": dict(color="#FF5722", linewidth=2.0, marker="s", markersize=5, linestyle="--"),
    "mlx": dict(color="#4CAF50", linewidth=2.0, marker="^", markersize=5, linestyle="--"),
}
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _bench_mlx(fn: Callable, n_warmup: int = 5, n_iter: int = 50) -> float:
    """Return ms/iter for an MLX kernel call (evaluates graph each iter)."""
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000.0 / n_iter


def _bench_np(fn: Callable, n_warmup: int = 5, n_iter: int = 50) -> float:
    """Return ms/iter for a NumPy reference function."""
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n_iter


# ---------------------------------------------------------------------------
# 1. Bit-pack / unpack
# ---------------------------------------------------------------------------


def bench_bit_pack(n_iter: int):
    Ns = [256, 512, 1024, 4096, 16384, 65536]
    bits = [1, 2, 4]
    rng = np.random.default_rng(0)

    pack_results: dict = {b: {} for b in bits}
    unpack_results: dict = {b: {} for b in bits}

    for b in bits:
        for N in Ns:
            indices = mx.array((rng.integers(0, 1 << b, size=N)).astype(np.uint8))
            packed = turboquant_bit_pack(indices, b)
            mx.eval(packed)

            pack_ms = _bench_mlx(lambda i=indices, _b=b: turboquant_bit_pack(i, _b), n_iter=n_iter)
            unpack_ms = _bench_mlx(
                lambda p=packed, _N=N, _b=b: turboquant_bit_unpack(p, _N, _b), n_iter=n_iter
            )

            # Numpy reference: simple bit-packing loop
            idx_np = np.array(indices, dtype=np.uint8)
            pack_ms_np = _bench_np(lambda i=idx_np, _b=b: _np_pack(i, _b), n_iter=n_iter)

            pack_results[b][N] = (pack_ms, pack_ms_np)
            unpack_results[b][N] = (unpack_ms, pack_ms_np)

    return Ns, pack_results, unpack_results


def _np_pack(indices: np.ndarray, b: int) -> np.ndarray:
    n = len(indices)
    elems = 8 // b
    mask = (1 << b) - 1
    n_bytes = n * b // 8
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        byte_val = 0
        for j in range(elems):
            byte_val |= (int(indices[i * elems + j]) & mask) << (j * b)
        out[i] = byte_val
    return out


def plot_bit_pack(Ns, pack_results, unpack_results, n_iter: int):
    bits = [1, 2, 4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Bit-pack / Bit-unpack: Metal vs NumPy", fontsize=14, fontweight="bold")

    for ax, (results, title) in zip(
        axes, [(pack_results, "bit_pack"), (unpack_results, "bit_unpack")]
    ):
        for i, b in enumerate(bits):
            metal_ms = [results[b][N][0] for N in Ns]
            np_ms = [results[b][N][1] for N in Ns]
            # throughput in GB/s: N bytes processed
            metal_gb = [N / 1e9 / (ms * 1e-3) for N, ms in zip(Ns, metal_ms)]
            np_gb = [N / 1e9 / (ms * 1e-3) for N, ms in zip(Ns, np_ms)]
            c = PALETTE[i]
            ax.plot(Ns, metal_gb, color=c, marker="o", linewidth=2, label=f"Metal b={b}")
            ax.plot(
                Ns,
                np_gb,
                color=c,
                marker="s",
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
                label=f"NumPy b={b}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("N (elements)", fontsize=11)
        ax.set_ylabel("Throughput (GB/s)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns], rotation=30, fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "fig1_bit_pack_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Scalar quantize / dequantize
# ---------------------------------------------------------------------------


def _lloyd_max_centroids(b: int) -> np.ndarray:
    """Approximate Lloyd-Max centroids for N(0,1) with 2^b bins."""
    n = 1 << b
    # uniform quantiles as an approximation
    from scipy import stats  # type: ignore

    edges = np.linspace(1e-6, 1 - 1e-6, n + 1)
    qedges = stats.norm.ppf(edges)
    return ((qedges[:-1] + qedges[1:]) / 2).astype(np.float32)


def bench_scalar_quant(n_iter: int):
    Ns = [256, 1024, 4096, 16384, 65536, 262144]
    bits = [1, 2, 4]
    rng = np.random.default_rng(1)

    sq_results: dict = {b: {} for b in bits}
    sdq_results: dict = {b: {} for b in bits}

    for b in bits:
        try:
            cents = mx.array(_lloyd_max_centroids(b))
        except ImportError:
            cents = mx.linspace(-1.0, 1.0, 1 << b).astype(mx.float32)

        for N in Ns:
            x = mx.array(rng.standard_normal(N).astype(np.float16))
            indices = turboquant_scalar_quantize(x, cents, b)
            mx.eval(indices)

            sq_ms = _bench_mlx(
                lambda _x=x, _c=cents, _b=b: turboquant_scalar_quantize(_x, _c, _b), n_iter=n_iter
            )
            sdq_ms = _bench_mlx(
                lambda _i=indices, _c=cents: turboquant_scalar_dequantize(_i, _c), n_iter=n_iter
            )

            # NumPy reference (argmin-based)
            x_np = np.array(x, dtype=np.float32)
            c_np = np.array(cents, dtype=np.float32)
            sq_ms_np = _bench_np(
                lambda _x=x_np, _c=c_np: _np_scalar_quantize(_x, _c), n_iter=n_iter
            )

            sq_results[b][N] = (sq_ms, sq_ms_np)
            sdq_results[b][N] = (sdq_ms, sq_ms_np)

    return Ns, sq_results, sdq_results


def _np_scalar_quantize(x: np.ndarray, cents: np.ndarray) -> np.ndarray:
    diffs = np.abs(x[:, None] - cents[None, :])
    return np.argmin(diffs, axis=1).astype(np.uint8)


def plot_scalar_quant(Ns, sq_results, sdq_results, n_iter: int):
    bits = [1, 2, 4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scalar Quantize / Dequantize: Metal vs NumPy", fontsize=14, fontweight="bold")

    for ax, (results, title) in zip(
        axes, [(sq_results, "scalar_quantize"), (sdq_results, "scalar_dequantize")]
    ):
        for i, b in enumerate(bits):
            metal_ms = [results[b][N][0] for N in Ns]
            np_ms = [results[b][N][1] for N in Ns]
            metal_gb = [N * 2 / 1e9 / (ms * 1e-3) for N, ms in zip(Ns, metal_ms)]
            np_gb = [N * 2 / 1e9 / (ms * 1e-3) for N, ms in zip(Ns, np_ms)]
            c = PALETTE[i]
            ax.plot(Ns, metal_gb, color=c, marker="o", linewidth=2, label=f"Metal b={b}")
            ax.plot(
                Ns,
                np_gb,
                color=c,
                marker="s",
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
                label=f"NumPy b={b}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("N (elements)", fontsize=11)
        ax.set_ylabel("Throughput (GB/s, fp16 input)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "fig2_scalar_quant_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Fused Hadamard + quantize
# ---------------------------------------------------------------------------


def bench_hadamard_quantize(n_iter: int):
    Ds = [64, 128, 256, 512, 1024]
    Bs = [1, 8, 32, 64, 128]
    rng = np.random.default_rng(2)

    try:
        cents = mx.array(_lloyd_max_centroids(4))
    except ImportError:
        cents = mx.linspace(-1.0, 1.0, 16).astype(mx.float32)

    # Sweep D at fixed B=64
    d_results: dict = {}
    B_fixed = 64
    for D in Ds:
        diag = mx.array(rng.choice([-1.0, 1.0], size=D).astype(np.float32))
        x = mx.array(rng.standard_normal((B_fixed, D)).astype(np.float16))
        ms = _bench_mlx(
            lambda _x=x, _d=diag, _c=cents: turboquant_hadamard_quantize(_x, _d, _c, 4),
            n_iter=n_iter,
        )
        d_results[D] = ms

    # Sweep B at fixed D=128
    D_fixed = 128
    diag_fixed = mx.array(rng.choice([-1.0, 1.0], size=D_fixed).astype(np.float32))
    b_results: dict = {}
    for B in Bs:
        x = mx.array(rng.standard_normal((B, D_fixed)).astype(np.float16))
        ms = _bench_mlx(
            lambda _x=x, _d=diag_fixed, _c=cents: turboquant_hadamard_quantize(_x, _d, _c, 4),
            n_iter=n_iter,
        )
        b_results[B] = ms

    return Ds, d_results, Bs, b_results, B_fixed, D_fixed


def plot_hadamard_quantize(Ds, d_results, Bs, b_results, B_fixed, D_fixed, n_iter: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fused Hadamard + Quantize (Metal, b=4)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ms_vals = [d_results[D] for D in Ds]
    ax.plot(Ds, ms_vals, **STYLE["metal"])
    ax.set_xlabel("D (head dimension)", fontsize=11)
    ax.set_ylabel("Latency (ms/iter)", fontsize=11)
    ax.set_title(f"Latency vs D  [B={B_fixed}]", fontsize=12)
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)
    for D, ms in zip(Ds, ms_vals):
        ax.annotate(
            f"{ms:.2f}", (D, ms), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center"
        )

    ax = axes[1]
    ms_vals = [b_results[B] for B in Bs]
    throughput = [B * D_fixed * 2 / 1e6 / (ms * 1e-3) for B, ms in zip(Bs, ms_vals)]
    ax.plot(Bs, throughput, **STYLE["metal"])
    ax.set_xlabel("Batch size B", fontsize=11)
    ax.set_ylabel("Throughput (MB/s, fp16 input)", fontsize=11)
    ax.set_title(f"Throughput vs B  [D={D_fixed}]", fontsize=12)
    ax.set_xticks(Bs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "fig3_hadamard_quantize.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 4. QJL encode
# ---------------------------------------------------------------------------


def bench_qjl_encode(n_iter: int):
    Bs = [1, 8, 32, 64, 128, 256]
    d = 128
    m = 128
    rng = np.random.default_rng(3)
    S = mx.array(rng.standard_normal((m, d)).astype(np.float16) / np.sqrt(d))

    metal_results: dict = {}
    np_results: dict = {}
    for B in Bs:
        x = mx.array(rng.standard_normal((B, d)).astype(np.float16))
        x_np = np.array(x, dtype=np.float32)
        S_np = np.array(S, dtype=np.float32)

        metal_ms = _bench_mlx(lambda _x=x, _S=S: qjl_encode(_x, _S), n_iter=n_iter)
        np_ms = _bench_np(lambda _x=x_np, _S=S_np: _np_qjl_encode(_x, _S), n_iter=n_iter)

        metal_results[B] = metal_ms
        np_results[B] = np_ms

    return Bs, metal_results, np_results, d, m


def _np_qjl_encode(x: np.ndarray, S: np.ndarray):
    proj = S @ x.T  # (m, B)
    signs = (proj >= 0).astype(np.uint8)
    packed = np.packbits(signs.T, axis=1)
    norms = np.linalg.norm(x, axis=1).astype(np.float16)
    return packed, norms


def plot_qjl_encode(Bs, metal_results, np_results, d, m, n_iter: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"QJL Encode (Metal vs NumPy)  [d={d}, m={m}]", fontsize=14, fontweight="bold")

    metal_ms = [metal_results[B] for B in Bs]
    np_ms = [np_results[B] for B in Bs]

    ax = axes[0]
    ax.plot(Bs, metal_ms, label="Metal", **STYLE["metal"])
    ax.plot(Bs, np_ms, label="NumPy", **STYLE["numpy"])
    ax.set_xlabel("Batch size B", fontsize=11)
    ax.set_ylabel("Latency (ms/iter)", fontsize=11)
    ax.set_title("Latency vs B", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    speedup = [np_ms[i] / metal_ms[i] for i in range(len(Bs))]
    ax.bar(range(len(Bs)), speedup, color=PALETTE[0], alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(Bs)))
    ax.set_xticklabels([str(B) for B in Bs])
    ax.set_xlabel("Batch size B", fontsize=11)
    ax.set_ylabel("Speedup vs NumPy", fontsize=11)
    ax.set_title("Metal Speedup", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for i, s in enumerate(speedup):
        ax.text(i, s + 0.05, f"{s:.1f}×", ha="center", fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "fig4_qjl_encode_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 5. QJL inner product
# ---------------------------------------------------------------------------


def bench_qjl_ip(n_iter: int):
    S_kvs = [64, 128, 256, 512, 1024, 2048, 4096]
    H = 8
    m = 128
    d = 128
    rng = np.random.default_rng(4)
    S_mat = mx.array(rng.standard_normal((m, d)).astype(np.float16) / np.sqrt(d))
    q_proj = mx.array(rng.standard_normal((H, m)).astype(np.float16))

    metal_results: dict = {}
    for S_kv in S_kvs:
        x_keys = mx.array(rng.standard_normal((S_kv * H, d)).astype(np.float16))
        packed_signs_flat, norms_flat = qjl_encode(x_keys, S_mat)
        mx.eval(packed_signs_flat, norms_flat)
        packed_signs = packed_signs_flat.reshape(S_kv, H, m // 8)
        norms = norms_flat.reshape(S_kv, H)

        ms = _bench_mlx(
            lambda _q=q_proj, _ps=packed_signs, _n=norms: qjl_inner_product(_q, _ps, _n),
            n_iter=n_iter,
        )
        metal_results[S_kv] = ms

    return S_kvs, metal_results, H, m


def plot_qjl_ip(S_kvs, metal_results, H, m, n_iter: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"QJL Inner Product (Metal)  [H={H}, m={m}]", fontsize=14, fontweight="bold")

    ms_vals = [metal_results[S] for S in S_kvs]

    ax = axes[0]
    ax.plot(S_kvs, ms_vals, **STYLE["metal"])
    ax.set_xlabel("S_kv (KV sequence length)", fontsize=11)
    ax.set_ylabel("Latency (ms/iter)", fontsize=11)
    ax.set_title("Latency vs S_kv", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Throughput: H × S_kv inner products per call
    ops = [H * S * m * 2 / 1e9 for S in S_kvs]  # GFLOPs (m mults + m adds per pair)
    gflops = [op / (ms * 1e-3) for op, ms in zip(ops, ms_vals)]
    ax.plot(S_kvs, gflops, color=PALETTE[1], marker="^", linewidth=2, markersize=5)
    ax.set_xlabel("S_kv", fontsize=11)
    ax.set_ylabel("GFLOP/s", fontsize=11)
    ax.set_title("Effective GFLOP/s", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "fig5_qjl_ip_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Fused RVQ decode + attend
# ---------------------------------------------------------------------------


def bench_rvq_attend(n_iter: int):
    S_kvs = [64, 128, 256, 512, 1024, 2048]
    B = 1
    H = 8
    D = 128
    b1, b2, bv = 4, 4, 4
    rng = np.random.default_rng(5)

    n_cents1 = 1 << b1
    n_cents2 = 1 << b2
    n_cents_v = 1 << bv
    V_SUB = 16
    n_sub_v = D // V_SUB

    centroids1 = mx.array(rng.standard_normal(n_cents1).astype(np.float32))
    centroids2 = mx.array(rng.standard_normal(n_cents2).astype(np.float32) * 0.5)
    v_codebook = mx.array(rng.standard_normal((n_cents_v, V_SUB)).astype(np.float16))

    metal_results: dict = {}
    for S_kv in S_kvs:
        q = mx.array(rng.standard_normal((B, H, 1, D)).astype(np.float16))
        k_idx1 = mx.array(rng.integers(0, n_cents1, (B, H, S_kv, D), dtype=np.uint8))
        k_idx2 = mx.array(rng.integers(0, n_cents2, (B, H, S_kv, D), dtype=np.uint8))
        v_indices = mx.array(rng.integers(0, n_cents_v, (B, H, S_kv, n_sub_v), dtype=np.uint8))
        mx.eval(q, k_idx1, k_idx2, v_indices)

        ms = _bench_mlx(
            lambda _q=q, _k1=k_idx1, _k2=k_idx2, _c1=centroids1, _c2=centroids2, _vi=v_indices, _vc=v_codebook: (
                turboquant_fused_rvq_decode_attend(_q, _k1, _k2, _c1, _c2, _vi, _vc, b1, b2, bv)
            ),
            n_iter=n_iter,
        )
        metal_results[S_kv] = ms

    return S_kvs, metal_results, B, H, D


def plot_rvq_attend(S_kvs, metal_results, B, H, D, n_iter: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Fused RVQ Decode + Attend (Metal)  [B={B}, H={H}, D={D}]", fontsize=14, fontweight="bold"
    )

    ms_vals = [metal_results[S] for S in S_kvs]

    ax = axes[0]
    ax.plot(S_kvs, ms_vals, **STYLE["metal"])
    ax.set_xlabel("S_kv (KV sequence length)", fontsize=11)
    ax.set_ylabel("Latency (ms/iter)", fontsize=11)
    ax.set_title("Attend Latency vs S_kv", fontsize=12)
    ax.grid(True, alpha=0.3)
    for S, ms in zip(S_kvs, ms_vals):
        ax.annotate(
            f"{ms:.2f}", (S, ms), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center"
        )

    ax = axes[1]
    # ms per token
    ms_per_token = [ms / S for ms, S in zip(ms_vals, S_kvs)]
    ax.plot(S_kvs, ms_per_token, color=PALETTE[2], marker="D", linewidth=2, markersize=5)
    ax.set_xlabel("S_kv", fontsize=11)
    ax.set_ylabel("ms / KV token", fontsize=11)
    ax.set_title("Per-token Cost", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "fig6_rvq_attend_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Memory savings
# ---------------------------------------------------------------------------


def plot_memory_savings():
    S_kvs = [128, 256, 512, 1024, 2048, 4096, 8192]
    D = 128
    H = 8

    def fp16_bytes(S):
        return S * H * D * 2  # keys fp16

    def bits_quant_bytes(S, b):
        return S * H * D * b // 8  # b bits per element, byte-packed

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"KV Cache Memory Footprint  [H={H}, D={D}]", fontsize=14, fontweight="bold")

    ax = axes[0]
    fp16_mb = [fp16_bytes(S) / 1e6 for S in S_kvs]
    ax.plot(S_kvs, fp16_mb, label="fp16 (baseline)", color="gray", linewidth=2, linestyle="--")
    for i, b in enumerate([1, 2, 4]):
        mb = [bits_quant_bytes(S, b) / 1e6 for S in S_kvs]
        ax.plot(
            S_kvs,
            mb,
            label=f"{b}-bit (Metal kernel)",
            color=PALETTE[i],
            linewidth=2,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("Sequence length S_kv", fontsize=11)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title("Key Cache Memory", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, b in enumerate([1, 2, 4]):
        ratio = [fp16_bytes(S) / bits_quant_bytes(S, b) for S in S_kvs]
        ax.plot(
            S_kvs,
            ratio,
            label=f"{b}-bit compression",
            color=PALETTE[i],
            linewidth=2,
            marker="o",
            markersize=4,
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="no compression")
    ax.set_xlabel("Sequence length S_kv", fontsize=11)
    ax.set_ylabel("Compression ratio (fp16 / b-bit)", fontsize=11)
    ax.set_title("Compression Ratio", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 18)

    plt.tight_layout()
    path = OUT_DIR / "fig7_memory_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 8. Summary speedup bar chart
# ---------------------------------------------------------------------------


def plot_summary(all_speedups: dict):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "TurboQuant Metal Kernels — Summary Speedup vs NumPy/CPU", fontsize=14, fontweight="bold"
    )

    kernels = list(all_speedups.keys())
    speedups = [all_speedups[k] for k in kernels]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(kernels))]

    bars = ax.bar(
        range(len(kernels)), speedups, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1×)")
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels(kernels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs NumPy/CPU baseline", fontsize=11)
    ax.set_title("Peak Speedup (larger batch / sequence)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, s) in enumerate(zip(bars, speedups)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{s:.1f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    path = OUT_DIR / "fig8_summary_speedup.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Metal kernel benchmark")
    parser.add_argument("--n_iter", type=int, default=50, help="Timing iterations per measurement")
    args = parser.parse_args()
    n_iter = args.n_iter

    print(f"\n=== TurboQuant Metal Kernel Benchmark  (n_iter={n_iter}) ===\n")

    results_json: dict = {}
    all_speedups: dict = {}

    # ---- 1. Bit-pack ----
    print("1/6  Benchmarking bit_pack / bit_unpack ...")
    Ns, pack_r, unpack_r = bench_bit_pack(n_iter)
    plot_bit_pack(Ns, pack_r, unpack_r, n_iter)
    # peak speedup at largest N, b=4
    best_N = Ns[-1]
    su_pack = pack_r[4][best_N][1] / pack_r[4][best_N][0]
    all_speedups["bit_pack (b=4, N=65k)"] = su_pack
    results_json["bit_pack"] = {
        str(b): {str(N): {"metal_ms": pack_r[b][N][0], "numpy_ms": pack_r[b][N][1]} for N in Ns}
        for b in [1, 2, 4]
    }

    # ---- 2. Scalar quant ----
    print("2/6  Benchmarking scalar_quantize / dequantize ...")
    Ns_sq, sq_r, sdq_r = bench_scalar_quant(n_iter)
    plot_scalar_quant(Ns_sq, sq_r, sdq_r, n_iter)
    best_N_sq = Ns_sq[-1]
    su_sq = sq_r[4][best_N_sq][1] / sq_r[4][best_N_sq][0]
    all_speedups["scalar_quantize (b=4, N=256k)"] = su_sq
    results_json["scalar_quantize"] = {
        str(b): {str(N): {"metal_ms": sq_r[b][N][0], "numpy_ms": sq_r[b][N][1]} for N in Ns_sq}
        for b in [1, 2, 4]
    }

    # ---- 3. Hadamard quantize ----
    print("3/6  Benchmarking hadamard_quantize ...")
    Ds, d_r, Bs_hq, b_r, B_fix, D_fix = bench_hadamard_quantize(n_iter)
    plot_hadamard_quantize(Ds, d_r, Bs_hq, b_r, B_fix, D_fix, n_iter)
    # speedup vs sequential Hadamard + quantize
    best_D = Ds[-1]
    # Reference: measure MLX hadamard_transform + scalar_quantize separately
    rng2 = np.random.default_rng(99)
    try:
        cents = mx.array(_lloyd_max_centroids(4))
    except ImportError:
        cents = mx.linspace(-1.0, 1.0, 16).astype(mx.float32)
    diag = mx.array(rng2.choice([-1.0, 1.0], size=best_D).astype(np.float32))
    x_ref = mx.array(rng2.standard_normal((B_fix, best_D)).astype(np.float16))
    # Two-step reference: mx.hadamard_transform + scalar_quantize separately
    twostep_ms = _bench_mlx(
        lambda _x=x_ref, _d=diag, _c=cents: turboquant_scalar_quantize(
            mx.hadamard_transform(_x.astype(mx.float32) * _d),
            _c,
            4,
        ),
        n_iter=n_iter,
    )
    su_had = twostep_ms / d_r[best_D]
    all_speedups[f"hadamard_quant (D={best_D}, B={B_fix})"] = su_had
    results_json["hadamard_quantize"] = {
        "D_sweep": {str(D): d_r[D] for D in Ds},
        "B_sweep": {str(B): b_r[B] for B in Bs_hq},
        "twostep_ms_D1024": twostep_ms,
    }

    # ---- 4. QJL encode ----
    print("4/6  Benchmarking qjl_encode ...")
    Bs_qjl, metal_qjl, np_qjl, d_qjl, m_qjl = bench_qjl_encode(n_iter)
    plot_qjl_encode(Bs_qjl, metal_qjl, np_qjl, d_qjl, m_qjl, n_iter)
    best_B_qjl = Bs_qjl[-1]
    su_qjl = np_qjl[best_B_qjl] / metal_qjl[best_B_qjl]
    all_speedups[f"qjl_encode (B={best_B_qjl})"] = su_qjl
    results_json["qjl_encode"] = {
        str(B): {"metal_ms": metal_qjl[B], "numpy_ms": np_qjl[B]} for B in Bs_qjl
    }

    # ---- 5. QJL inner product ----
    print("5/6  Benchmarking qjl_inner_product ...")
    S_kvs_ip, metal_ip, H_ip, m_ip = bench_qjl_ip(n_iter)
    plot_qjl_ip(S_kvs_ip, metal_ip, H_ip, m_ip, n_iter)
    results_json["qjl_inner_product"] = {str(S): metal_ip[S] for S in S_kvs_ip}
    all_speedups["qjl_ip (S_kv=4096)"] = (
        metal_ip[S_kvs_ip[0]] / metal_ip[S_kvs_ip[-1]] * len(S_kvs_ip)
    )  # relative efficiency

    # ---- 6. RVQ attend ----
    print("6/6  Benchmarking fused_rvq_decode_attend ...")
    S_kvs_rv, metal_rv, B_rv, H_rv, D_rv = bench_rvq_attend(n_iter)
    plot_rvq_attend(S_kvs_rv, metal_rv, B_rv, H_rv, D_rv, n_iter)
    results_json["fused_rvq_attend"] = {str(S): metal_rv[S] for S in S_kvs_rv}
    all_speedups["rvq_attend (S_kv=2048)"] = (
        metal_rv[S_kvs_rv[0]] / metal_rv[S_kvs_rv[-1]] * len(S_kvs_rv)
    )

    # ---- 7. Memory ----
    plot_memory_savings()

    # ---- 8. Summary ----
    plot_summary(all_speedups)

    # Save JSON
    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # Print summary table
    print("\n=== Speedup summary ===")
    print(f"{'Kernel':<42}  {'Speedup':>10}")
    print("-" * 55)
    for k, v in all_speedups.items():
        print(f"  {k:<40}  {v:>8.2f}×")

    print(f"\nAll figures saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()

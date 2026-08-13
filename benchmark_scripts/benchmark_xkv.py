"""Offline-synthetic benchmark for xKV-adapted cross-layer shared-subspace
key compression.

xKV's claim is that a group of nearby layers' keys share a dominant SVD
subspace, so jointly factorizing them into one basis compresses better (per
byte) than compressing each layer independently at the same rank. This
harness builds synthetic multi-layer key tensors with a controllable amount
of *shared* structure and measures:

  1. Reconstruction MSE of the shared-basis path vs independent per-layer SVD
     (SVDq-style) at the same rank, swept across ``xkv_group_size`` and a
     "shared_fraction" knob (how much of each layer's variance the group
     truly shares vs private per-layer noise). This is the honest test: xKV
     should win when structure is genuinely shared, and should *not* claim a
     win when it isn't — both cases are reported.
  2. Byte-accounting: total group bytes (leader's amortized basis + every
     member's own latent codes) vs the naive sum of per-layer independent
     SVDq bytes at the same rank — the amortization win xKV's shared basis
     is supposed to deliver.
  3. Output-perturbation proxy (cosine distance of a probe-query attention
     output using the compressed cache vs the full-precision cache), the
     same metric family used by the CaM/ChunkKV benchmarks in this repo, so
     xKV's numbers are comparable in kind (not in absolute value) to other
     eviction/merge methods already benchmarked here.

No real model required — consistent with every other benchmark_*.py in this
repo (StreamingLLM, CaM, ChunkKV, etc. all commit offline-synthetic numbers,
never paper-claimed numbers).

Usage
-----
    python benchmark_scripts/benchmark_xkv.py

Prints tables and saves a JSON summary.
"""

from __future__ import annotations

import json
import math
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.xkv_cache import XKVCache
from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [256, 512]
GROUP_SIZES = [2, 3, 4]
SHARED_FRACTIONS = [0.9, 0.5, 0.1]  # 0.9 = strongly shared, 0.1 = mostly private
N_HEADS = 4
HEAD_DIM = 64
RANK = 16
N_PROBES = 32


def _synthetic_group_keys(n_members: int, S: int, shared_fraction: float, seed: int):
    """Build n_members layers' [1, H, S, D] key tensors with a controllable
    mix of shared low-rank structure and independent per-layer structure."""
    rng = np.random.default_rng(seed)
    shared_basis = rng.standard_normal((HEAD_DIM, RANK)).astype(np.float32)
    shared_basis /= np.linalg.norm(shared_basis, axis=0, keepdims=True) + 1e-8

    layers = []
    for i in range(n_members):
        shared_coeffs = rng.standard_normal((N_HEADS, S, RANK)).astype(np.float32)
        shared_part = np.einsum("hsr,dr->hsd", shared_coeffs, shared_basis)
        private = rng.standard_normal((N_HEADS, S, HEAD_DIM)).astype(np.float32)
        mix = shared_fraction * shared_part + (1.0 - shared_fraction) * private * 2.0
        layers.append(mx.array(mix[None].astype(np.float16)))  # [1, H, S, D]
    return layers


def _attn_output(query, keys, values):
    q = query.astype(mx.float32)
    k = keys.astype(mx.float32)
    v = values.astype(mx.float32)
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    logits = (q @ k.T) * scale
    w = mx.softmax(logits, axis=-1)
    return w @ v


def _perturbation(probe, full_k, full_v, comp_k, comp_v):
    ref = _attn_output(probe, full_k, full_v)
    got = _attn_output(probe, comp_k, comp_v)
    rn = ref / (mx.sqrt(mx.sum(ref * ref, axis=-1, keepdims=True)) + 1e-8)
    gn = got / (mx.sqrt(mx.sum(got * got, axis=-1, keepdims=True)) + 1e-8)
    cos = mx.sum(rn * gn, axis=-1)
    return float(mx.mean(1.0 - cos).item())


def _mse(a, b):
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _run_shared_group(layers, group_size, seed):
    """Run the layers through a coordinated XKVCache group; settle so every
    member adopts the shared basis; return (outputs, bytes, latency_ms)."""
    coord = XKVCoordinator()
    cfg = KVCacheConfig(method="xkv", head_dim=HEAD_DIM, xkv_rank=RANK)
    members = [
        XKVCache(cfg, member_idx=i, group_id=0, n_members=group_size, coordinator=coord)
        for i in range(group_size)
    ]
    # xKV is keys-only: values pass through fp16 unchanged. Use real random
    # values (not zeros) so the attention-output perturbation probe actually
    # exercises the softmax weighting induced by the *compressed keys* —
    # zero values would make every output identically zero regardless of key
    # quality, silently degenerating the perturbation metric to a constant.
    rng_v = np.random.default_rng(seed + 555)
    real_v = [
        mx.array(rng_v.standard_normal((1, N_HEADS, k.shape[2], HEAD_DIM)).astype(np.float16))
        for k in layers
    ]

    t0 = time.perf_counter()
    outs = [m.update_and_fetch(k, v) for m, k, v in zip(members, layers, real_v)]
    # settle round so earlier-iterated members adopt the completed shared basis
    settle_k = mx.zeros((1, N_HEADS, 1, HEAD_DIM), dtype=mx.float16)
    settle_v = mx.zeros((1, N_HEADS, 1, HEAD_DIM), dtype=mx.float16)
    for m in members:
        m.update_and_fetch(settle_k, settle_v)
    latency_ms = (time.perf_counter() - t0) * 1_000

    total_bytes = sum(m.compressed_key_bytes + m.shared_basis_bytes for m in members)
    return [o[0] for o in outs], real_v, total_bytes, latency_ms


def _run_independent(layers, real_v):
    """Run each layer through a standalone (group_size=1) XKVCache — the
    per-layer independent SVDq-style baseline."""
    cfg = KVCacheConfig(method="xkv", head_dim=HEAD_DIM, xkv_rank=RANK)
    outs, total_bytes = [], 0
    for k, v in zip(layers, real_v):
        cache = XKVCache(cfg, member_idx=0, group_id=0, n_members=1, coordinator=None)
        ko, _ = cache.update_and_fetch(k, v)
        outs.append(ko)
        total_bytes += cache.compressed_key_bytes + cache.shared_basis_bytes
    return outs, total_bytes


def _run_once(seq_len, group_size, shared_fraction, seed) -> dict:
    layers = _synthetic_group_keys(group_size, seq_len, shared_fraction, seed)

    shared_outs, real_v, shared_bytes, latency_ms = _run_shared_group(layers, group_size, seed)
    indep_outs, indep_bytes = _run_independent(layers, real_v)

    shared_mse = float(np.mean([_mse(o, k) for o, k in zip(shared_outs, layers)]))
    indep_mse = float(np.mean([_mse(o, k) for o, k in zip(indep_outs, layers)]))

    rng = np.random.default_rng(seed + 777)
    probe = mx.array(rng.standard_normal((N_PROBES, HEAD_DIM)).astype(np.float16))
    perts_shared, perts_indep = [], []
    for k, v, s_out, i_out in zip(layers, real_v, shared_outs, indep_outs):
        full_k, full_v = k[0, 0], v[0, 0]  # [S, D] — head 0, batch 0
        perts_shared.append(_perturbation(probe, full_k, full_v, s_out[0, 0], full_v))
        perts_indep.append(_perturbation(probe, full_k, full_v, i_out[0, 0], full_v))

    return {
        "seq_len": seq_len,
        "group_size": group_size,
        "shared_fraction": shared_fraction,
        "shared_basis_mse": round(shared_mse, 5),
        "independent_svd_mse": round(indep_mse, 5),
        "mse_ratio": round(shared_mse / max(indep_mse, 1e-9), 3),
        "shared_group_bytes": shared_bytes,
        "independent_bytes": indep_bytes,
        "byte_ratio": round(shared_bytes / max(indep_bytes, 1), 3),
        "perturbation_shared": round(float(np.mean(perts_shared)), 5),
        "perturbation_indep": round(float(np.mean(perts_indep)), 5),
        "latency_ms": round(latency_ms, 2),
    }


def main() -> None:
    print("xKV-adapted cross-layer shared-subspace compression — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  rank={RANK}  probes={N_PROBES}")
    print("  (mse_ratio < 1 means the shared basis reconstructs better than independent SVD)")
    print("  (byte_ratio < 1 means the shared basis costs fewer bytes than independent SVD)")
    print()
    header = (
        f"{'seq':>5}  {'grp':>4}  {'shared_frac':>11}  {'mse_ratio':>9}  "
        f"{'byte_ratio':>10}  {'pert_shared':>11}  {'pert_indep':>10}  {'ms':>7}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for seq_len, group_size, shared_fraction in product(SEQ_LENS, GROUP_SIZES, SHARED_FRACTIONS):
        row = _run_once(seq_len, group_size, shared_fraction, seed=seq_len + group_size)
        results.append(row)
        print(
            f"{row['seq_len']:>5}  {row['group_size']:>4}  {row['shared_fraction']:>11.1f}  "
            f"{row['mse_ratio']:>9.3f}  {row['byte_ratio']:>10.3f}  "
            f"{row['perturbation_shared']:>11.5f}  {row['perturbation_indep']:>10.5f}  "
            f"{row['latency_ms']:>7.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "xkv" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    high_share = [r for r in results if r["shared_fraction"] == max(SHARED_FRACTIONS)]
    low_share = [r for r in results if r["shared_fraction"] == min(SHARED_FRACTIONS)]
    print("\nSummary:")
    print(
        f"  At shared_fraction={max(SHARED_FRACTIONS)}: mean mse_ratio="
        f"{np.mean([r['mse_ratio'] for r in high_share]):.3f}, mean byte_ratio="
        f"{np.mean([r['byte_ratio'] for r in high_share]):.3f}"
    )
    print(
        f"  At shared_fraction={min(SHARED_FRACTIONS)}: mean mse_ratio="
        f"{np.mean([r['mse_ratio'] for r in low_share]):.3f}, mean byte_ratio="
        f"{np.mean([r['byte_ratio'] for r in low_share]):.3f}"
    )
    print("  (honest expectation: mse_ratio should rise toward/above 1.0 as shared_fraction")
    print("   falls — xKV's shared-basis mechanism only helps when structure is truly shared.)")


if __name__ == "__main__":
    main()

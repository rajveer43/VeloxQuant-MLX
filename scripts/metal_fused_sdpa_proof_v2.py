"""Phase 2 v2 — FlashAttention-style fused VecInfer SDPA Metal kernel.

The v1 naive kernel (scripts/metal_fused_sdpa_proof.py) is correct but
serial: 1 thread per output position can't saturate the GPU.  This v2
kernel parallelizes the S_kv reduction across a 32-wide SIMD group:

  Grid:        (B * H_q, S_q, 1)
  Threadgroup: (32, 1, 1)        — one SIMD group cooperates on one query

The 32 threads jointly:
  1. Fill a shared [n_sub, n_centroids] LUT in threadgroup memory
  2. For each tile of 32 K positions, each thread scores its own K
     via LUT lookups, simd_max+simd_sum for the running stats
  3. Each thread accumulates its share of the D-dim output stripe

We reuse the v1 reference for correctness validation.

Run from repo root:

    PYTHONPATH=. python scripts/metal_fused_sdpa_proof_v2.py
"""

from __future__ import annotations

import time
from typing import Tuple

import mlx.core as mx
import numpy as np

from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_queries,
    dequantize_vq,
    walsh_hadamard_matrix,
)
from veloxquant_mlx.metal import metal_available

# Import the v1 reference + fixture helpers
import sys

sys.path.insert(0, "scripts")
from metal_fused_sdpa_proof import (  # noqa: E402
    reference_sdpa,
    _make_test_inputs,
    _max_abs_diff,
)


# ===========================================================================
# Kernel v2 — FlashAttention-style tiled across SIMD group
# ===========================================================================
#
# Inputs (same as v1):
#   q          [B*H_q*S_q, D]               fp32 — already transformed
#   k_indices  [B*H_kv*S_kv, n_sub]         uint32
#   k_codebook [n_centroids, sub_dim]       fp32
#   v_indices  [B*H_kv*S_kv, n_sub_v]       uint32
#   v_codebook [n_centroids_v, sub_dim_v]   fp32
#   params     [10] uint32 packed shape
#   scale_arr  [1] fp32
#   slide_arr  [1] uint32
# Output:
#   out        [B*H_q*S_q, D]               fp32
#
# Thread-group cooperation rules:
#   - All 32 lanes hold the same (q_head_idx, q_pos) coordinates.
#   - The LUT lives in threadgroup memory so all lanes share one copy.
#   - The running output o[D] lives in threadgroup memory; lane k writes
#     dimensions D//32 apart starting at offset k.
#   - simd_max / simd_sum collapse 32 per-lane values to one in O(log N)
#     hardware ops — no manual reduction tree needed.
#
# n_centroids is baked in at compile time so the LUT array size is known.
_FUSED_SDPA_V2_SRC = r"""
    // Grid x = (B*H_q) * 32, so each threadgroup covers one query head;
    // inside that threadgroup the 32 lanes cooperate via SIMD reductions.
    uint q_head_idx = thread_position_in_grid.x / 32;   // 0 .. B*H_q
    uint q_pos      = thread_position_in_grid.y;        // 0 .. S_q
    uint lane       = thread_position_in_threadgroup.x; // 0 .. 31

    uint H_q       = params[0];
    uint H_kv      = params[1];
    uint S_q       = params[2];
    uint S_kv      = params[3];
    uint D         = params[4];
    uint n_sub     = params[5];
    uint sub_dim   = params[6];
    uint n_sub_v   = params[7];
    uint sub_dim_v = params[8];
    uint flags     = params[9];
    bool causal       = (flags & 1u) != 0u;
    bool use_window   = (flags & 2u) != 0u;
    uint window_width = slide_arr[0];
    float scale       = scale_arr[0];

    if (q_pos >= S_q) { return; }

    uint batch       = q_head_idx / H_q;
    uint h_q         = q_head_idx % H_q;
    uint h_kv        = (h_q * H_kv) / H_q;       // GQA integer div

    uint q_base     = q_head_idx * S_q * D + q_pos * D;
    uint k_base_b   = batch * H_kv * S_kv;       // per-batch base in K
    uint out_base   = q_head_idx * S_q * D + q_pos * D;

    constexpr uint kNCentroids = LUT_N_CENTROIDS;   // compile-time
    constexpr uint kMaxLut     = LUT_MAX_SIZE;      // n_sub * n_centroids
    constexpr uint kMaxD       = MAX_D;

    threadgroup float lut[kMaxLut];           // [n_sub, n_centroids]
    threadgroup float t_out[kMaxD];           // running weighted V accumulator
    threadgroup float tg_m_shared;            // running max, broadcast
    threadgroup float tg_d_shared;            // running denom, broadcast
    threadgroup float tg_factor;              // exp(m_old - m_new) per tile

    // -----------------------------------------------------------------
    // Phase 0: cooperatively fill LUT[sub, c]
    //   = q[sub*sub_dim:(sub+1)*sub_dim] @ k_codebook[c, :]
    // Lane k handles entries [k, k+32, k+64, ...] for coalesced access.
    // -----------------------------------------------------------------
    uint lut_total = n_sub * kNCentroids;
    for (uint idx = lane; idx < lut_total; idx += 32) {
        uint sub = idx / kNCentroids;
        uint c   = idx % kNCentroids;
        uint q_sub_off = q_base + sub * sub_dim;
        uint cb_off    = c * sub_dim;
        float dot = 0.0f;
        for (uint i = 0; i < sub_dim; ++i) {
            dot += q[q_sub_off + i] * k_codebook[cb_off + i];
        }
        lut[idx] = dot;
    }

    // -----------------------------------------------------------------
    // Phase 1: init running state
    // -----------------------------------------------------------------
    for (uint dim = lane; dim < D; dim += 32) {
        t_out[dim] = 0.0f;
    }
    if (lane == 0) {
        tg_m_shared = -INFINITY;
        tg_d_shared = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_abs = (S_kv - S_q) + q_pos;

    // -----------------------------------------------------------------
    // Phase 2: process S_kv in tiles of 32 (one tile per SIMD step)
    // -----------------------------------------------------------------
    for (uint tile_start = 0; tile_start < S_kv; tile_start += 32) {
        uint k_pos = tile_start + lane;

        // Per-lane score (or -INF if out of range or masked)
        float score = -INFINITY;
        bool valid = (k_pos < S_kv);
        if (valid && causal && k_pos > q_abs) valid = false;
        if (valid && use_window) {
            // Reference mask: k_pos < q_abs - window + 1, i.e. k_pos + window < q_abs + 1
            if (q_abs + 1u > window_width && k_pos + window_width < q_abs + 1u) {
                valid = false;
            }
        }
        if (valid) {
            uint k_row_idx = (k_base_b + h_kv * S_kv + k_pos) * n_sub;
            float s = 0.0f;
            for (uint sub = 0; sub < n_sub; ++sub) {
                uint c = k_indices[k_row_idx + sub];
                s += lut[sub * kNCentroids + c];
            }
            score = s * scale;
        }

        // Tile max across the 32 lanes
        float tile_max = simd_max(score);
        // If the whole tile is masked, skip it
        if (!isfinite(tile_max)) { continue; }

        // Merge tile_max into the running max; compute the rescale factor
        float m_old, m_new, factor;
        if (lane == 0) {
            m_old = tg_m_shared;
            m_new = max(m_old, tile_max);
            // factor for rescaling the running denom + output
            factor = isfinite(m_old) ? exp(m_old - m_new) : 0.0f;
            tg_m_shared = m_new;
            tg_factor   = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m_new  = tg_m_shared;
        factor = tg_factor;

        // Per-lane softmax weight (0 if masked since exp(-inf) = 0)
        float w = exp(score - m_new);
        if (!valid) w = 0.0f;

        // Reduce weights across the SIMD group → tile contribution to denom
        float tile_w_sum = simd_sum(w);

        // Update running denom (single thread)
        if (lane == 0) {
            tg_d_shared = tg_d_shared * factor + tile_w_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale prior output by `factor` (lanes split across D)
        for (uint dim = lane; dim < D; dim += 32) {
            t_out[dim] *= factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation — each output dim is computed by exactly one lane.
        // That lane walks all 32 tile-mates, shuffling in each mate's
        // weight + V index, computes the per-dim contribution.
        // This is conflict-free: every t_out[dim] is written by one lane.
        for (uint dim = lane; dim < D; dim += 32) {
            uint sub_v = dim / sub_dim_v;
            uint comp  = dim % sub_dim_v;
            float acc = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                uint k_l = tile_start + l;
                if (k_l >= S_kv) break;
                float w_l = simd_shuffle(w, l);
                if (w_l == 0.0f) continue;
                uint v_row_idx = (k_base_b + h_kv * S_kv + k_l) * n_sub_v;
                uint v_c = v_indices[v_row_idx + sub_v];
                acc += w_l * v_codebook[v_c * sub_dim_v + comp];
            }
            t_out[dim] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -----------------------------------------------------------------
    // Phase 3: normalize and write out
    // -----------------------------------------------------------------
    float inv_d = 1.0f / max(tg_d_shared, 1e-20f);
    for (uint dim = lane; dim < D; dim += 32) {
        out[out_base + dim] = t_out[dim] * inv_d;
    }
"""


_kernel_cache: dict = {}


def _get_kernel_v2(n_centroids: int, n_sub: int, D: int):
    """Compile a kernel specialized for (n_centroids, n_sub, D).

    All three are baked in as compile-time constants so the threadgroup
    memory arrays have known sizes.
    """
    if n_centroids > 256:
        raise ValueError(f"Fused SDPA v2: n_centroids <= 256, got {n_centroids}")
    if n_sub > 16:
        raise ValueError(f"Fused SDPA v2: n_sub <= 16, got {n_sub}")
    if D > 256:
        raise ValueError(f"Fused SDPA v2: D <= 256, got {D}")
    key = (n_centroids, n_sub, D)
    if key not in _kernel_cache:
        src = (
            _FUSED_SDPA_V2_SRC.replace("LUT_N_CENTROIDS", str(n_centroids))
            .replace("LUT_MAX_SIZE", str(n_sub * n_centroids))
            .replace("MAX_D", str(D))
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_fused_sdpa_v2_c{n_centroids}_s{n_sub}_d{D}",
            input_names=[
                "q",
                "k_indices",
                "k_codebook",
                "v_indices",
                "v_codebook",
                "params",
                "scale_arr",
                "slide_arr",
            ],
            output_names=["out"],
            source=src,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def metal_fused_sdpa_v2(
    q_tilde: mx.array,
    k_indices: mx.array,
    k_codebook: mx.array,
    v_indices: mx.array,
    v_codebook: mx.array,
    scale: float,
    causal: bool = True,
    sliding_window: int = 0,
) -> mx.array:
    B, H_q, S_q, D = q_tilde.shape
    _, H_kv, S_kv, n_sub = k_indices.shape
    n_centroids, sub_dim = k_codebook.shape
    _, _, _, n_sub_v = v_indices.shape
    _, sub_dim_v = v_codebook.shape

    if D != n_sub * sub_dim or D != n_sub_v * sub_dim_v:
        raise ValueError(f"D={D} mismatch with sub_dim layout")

    q_flat = q_tilde.reshape(B * H_q * S_q, D).astype(mx.float32)
    k_idx_flat = k_indices.reshape(B * H_kv * S_kv, n_sub).astype(mx.uint32)
    v_idx_flat = v_indices.reshape(B * H_kv * S_kv, n_sub_v).astype(mx.uint32)
    k_cb = k_codebook.astype(mx.float32)
    v_cb = v_codebook.astype(mx.float32)

    flags = 0
    if causal:
        flags |= 1
    if sliding_window and sliding_window > 0:
        flags |= 2

    params = mx.array(
        [H_q, H_kv, S_q, S_kv, D, n_sub, sub_dim, n_sub_v, sub_dim_v, flags],
        dtype=mx.uint32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    slide_arr = mx.array([int(sliding_window or 0)], dtype=mx.uint32)

    kernel = _get_kernel_v2(n_centroids, n_sub, D)

    outputs = kernel(
        inputs=[q_flat, k_idx_flat, k_cb, v_idx_flat, v_cb, params, scale_arr, slide_arr],
        output_shapes=[(B * H_q * S_q, D)],
        output_dtypes=[mx.float32],
        # Grid x: 32 lanes per query × B*H_q queries.  Each SIMD group
        # cooperates on one query position via threadgroup memory.
        grid=(B * H_q * 32, S_q, 1),
        threadgroup=(32, 1, 1),
    )
    return outputs[0].reshape(B, H_q, S_q, D)


# ===========================================================================
# Driver
# ===========================================================================
def main() -> int:
    if not metal_available():
        print("Metal unavailable — aborting.")
        return 1

    print(f"Device: {mx.default_device()}")

    # Realistic shape
    fixture = _make_test_inputs(
        B=1,
        H_q=32,
        H_kv=8,
        S_q=1,
        S_kv=2048,
        D=128,
        sub_dim=8,
        n_centroids=256,
    )

    print("\n=== Correctness (v2 vs reference) ===")
    for causal, sw in [(True, 0), (False, 0), (True, 128)]:
        q = fixture["q"]
        k_idx = fixture["k_indices"]
        v_idx = fixture["v_indices"]
        cb = fixture["codebook"]
        smooth = fixture["smooth"]
        H = fixture["H"]
        scale = fixture["scale"]

        out_ref = reference_sdpa(
            q=q,
            k_indices=k_idx,
            k_codebook=cb,
            smooth=smooth,
            H=H,
            v_indices=v_idx,
            v_codebook=cb,
            scale=scale,
            causal=causal,
            sliding_window=sw,
        )
        q_tilde = apply_dual_transform_queries(q.astype(mx.float32), smooth, H)
        out_v2 = metal_fused_sdpa_v2(
            q_tilde=q_tilde,
            k_indices=k_idx,
            k_codebook=cb,
            v_indices=v_idx,
            v_codebook=cb,
            scale=scale,
            causal=causal,
            sliding_window=sw,
        )
        mx.eval(out_ref, out_v2)
        d = _max_abs_diff(out_ref, out_v2)
        ok = d < 1e-2
        tag = "OK" if ok else "FAIL"
        print(f"  causal={causal!s:5s} sliding={sw:>4d} → max|diff|={d:.4e}  {tag}")

    # Benchmark sweep
    print("\n=== Throughput (median of 30, after 3 warmup) ===")
    print(f"  {'shape':<55s}  {'ref ms':>8s}  {'v2 ms':>8s}  {'speedup':>8s}")
    print(f"  {'-' * 55}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for B, H_q, H_kv, S_q, S_kv in [
        (1, 32, 8, 1, 512),
        (1, 32, 8, 1, 2048),
        (1, 32, 8, 1, 4096),
        (1, 32, 8, 1, 8192),
        (1, 32, 8, 1, 16384),
    ]:
        fx = _make_test_inputs(B, H_q, H_kv, S_q, S_kv, 128, 8, 256)
        q = fx["q"]
        k_idx = fx["k_indices"]
        v_idx = fx["v_indices"]
        cb = fx["codebook"]
        smooth = fx["smooth"]
        H = fx["H"]
        scale = fx["scale"]
        q_tilde = apply_dual_transform_queries(q.astype(mx.float32), smooth, H)

        # Warmup
        for _ in range(3):
            a = reference_sdpa(
                q=q,
                k_indices=k_idx,
                k_codebook=cb,
                smooth=smooth,
                H=H,
                v_indices=v_idx,
                v_codebook=cb,
                scale=scale,
                causal=True,
            )
            b = metal_fused_sdpa_v2(
                q_tilde=q_tilde,
                k_indices=k_idx,
                k_codebook=cb,
                v_indices=v_idx,
                v_codebook=cb,
                scale=scale,
                causal=True,
            )
            mx.eval(a, b)

        t_ref, t_v2 = [], []
        for _ in range(30):
            t0 = time.perf_counter()
            a = reference_sdpa(
                q=q,
                k_indices=k_idx,
                k_codebook=cb,
                smooth=smooth,
                H=H,
                v_indices=v_idx,
                v_codebook=cb,
                scale=scale,
                causal=True,
            )
            mx.eval(a)
            t_ref.append(time.perf_counter() - t0)
        for _ in range(30):
            t0 = time.perf_counter()
            b = metal_fused_sdpa_v2(
                q_tilde=q_tilde,
                k_indices=k_idx,
                k_codebook=cb,
                v_indices=v_idx,
                v_codebook=cb,
                scale=scale,
                causal=True,
            )
            mx.eval(b)
            t_v2.append(time.perf_counter() - t0)

        m_ref = float(np.median(t_ref)) * 1e3
        m_v2 = float(np.median(t_v2)) * 1e3
        sp = m_ref / m_v2 if m_v2 > 0 else float("inf")
        shape_str = f"B={B} H_q={H_q} H_kv={H_kv} S_q={S_q} S_kv={S_kv}"
        print(f"  {shape_str:<55s}  {m_ref:>7.2f}  {m_v2:>7.2f}  {sp:>7.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

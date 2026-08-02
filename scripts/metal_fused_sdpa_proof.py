"""Phase 2 proof: fused VecInfer SDPA Metal kernel correctness + benchmark.

This script is the acceptance gate for Phase 2.  Before any library
integration, the fused Metal kernel must produce attention output that
matches the pure-MLX reference (dequant → standard SDPA) to within fp16
tolerance, on a realistic shape.

Run from repo root:

    PYTHONPATH=. python scripts/metal_fused_sdpa_proof.py

Reference path (what we are replacing):
    1. Dequantize key indices via existing pure-MLX path.
    2. Apply the inverse smooth + Hadamard transform → fp16 K_hat.
    3. Same for values.
    4. Compute softmax(q @ K_hat.T * scale) @ V_hat in pure MLX.

Fused path (the new kernel):
    1. Transform queries: q_tilde = (q * smooth) @ H   (cheap, fp32).
    2. Precompute per-(B, H_q, S_q) LUT [n_sub, n_centroids] in the
       kernel: LUT[sub, c] = q_tilde[sub*sd:(sub+1)*sd] @ k_codebook[c, :].
    3. Online softmax over S_kv keys: score = sum_sub LUT[sub, k_idx[sub]].
    4. Accumulate weighted value via V codebook indices.
    5. Output [B, H_q, S_q, D] fp16.

Hard requirement: max-abs-diff < 1e-2 vs reference.  Test fails → kernel
is wrong; do not proceed to Step 2 integration.
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


# ===========================================================================
# Metal kernel — inline until proof passes; extracted to fused_sdpa.py later.
# ===========================================================================
#
# Thread layout: grid = (B * H_q, S_q, 1).  One thread per output position.
#
# Inputs (MLX names match input_names list):
#   q          [B*H_q*S_q, D]               fp32   (already smooth+Hadamard transformed)
#   k_indices  [B*H_kv*S_kv, n_sub]         uint32
#   k_codebook [n_centroids, sub_dim]       fp32
#   v_indices  [B*H_kv*S_kv, n_sub_v]       uint32
#   v_codebook [n_centroids_v, sub_dim_v]   fp32
#   params     [10]                         uint32  -- packed shape/mode params:
#       [0] H_q          query head count (per batch)
#       [1] H_kv         kv head count    (per batch)
#       [2] S_q          query sequence length (per batch)
#       [3] S_kv         kv sequence length    (per batch)
#       [4] D            head dim
#       [5] n_sub        D / sub_dim
#       [6] sub_dim      key sub-vector dim
#       [7] n_sub_v      D / sub_dim_v
#       [8] sub_dim_v    value sub-vector dim
#       [9] flags        bit 0 = causal, bit 1 = sliding window enabled
#   scale_arr  [1]                           fp32
#   slide_arr  [1]                           uint32 -- sliding window width
#
# Output:
#   out        [B*H_q*S_q, D]                fp32
#
# (Why a flat output and a packed params buffer? mx.fast.metal_kernel doesn't
# let us pass python scalars or 5-axis tensors directly; flattening keeps
# every binding ≤ 2D and 1D scalars get bundled into a small buffer.)
#
_FUSED_SDPA_SRC = r"""
    uint q_head_idx = thread_position_in_grid.x;   // 0 .. B*H_q
    uint q_pos      = thread_position_in_grid.y;   // 0 .. S_q

    // Unpack params
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
    uint h_kv        = (h_q * H_kv) / H_q;        // GQA broadcast: integer div

    // Base offsets into the flat tensors
    uint q_base   = q_head_idx * S_q * D + q_pos * D;
    uint k_base_batch = batch * H_kv * S_kv;         // per-batch base in K
    uint out_base = q_head_idx * S_q * D + q_pos * D;

    // -----------------------------------------------------------------
    // Step A: precompute LUT[sub, c] = q[sub*sd:(sub+1)*sd] @ k_codebook[c, :]
    // -----------------------------------------------------------------
    // LUT lives in registers.  Max budget: 16 sub × 256 centroids = 4096 floats.
    // The wrapper enforces n_sub <= 16 and n_centroids <= 256.
    float lut[16 * 256];

    uint n_centroids = LUT_N_CENTROIDS;   // template constant (set per-compile)

    for (uint sub = 0; sub < n_sub; ++sub) {
        uint q_sub_off = q_base + sub * sub_dim;
        for (uint c = 0; c < n_centroids; ++c) {
            float dot = 0.0f;
            uint cb_off = c * sub_dim;
            for (uint i = 0; i < sub_dim; ++i) {
                dot += q[q_sub_off + i] * k_codebook[cb_off + i];
            }
            lut[sub * n_centroids + c] = dot;
        }
    }

    // -----------------------------------------------------------------
    // Step B: online softmax over S_kv, accumulating weighted value sum.
    //   Running max m, running denom d, running output o
    //     score_k = sum_sub LUT[sub, k_idx[sub]]  * scale
    //     m'      = max(m, score_k)
    //     factor  = exp(m - m')
    //     d'      = d * factor + exp(score_k - m')
    //     o'_dim  = o_dim * factor + exp(score_k - m') * V_hat_k[dim]
    //   At the end: out[dim] = o[dim] / d
    // V_hat_k[dim] is computed from value indices + value codebook on the fly:
    //   V_hat_k[sub_v * sub_dim_v + i] = v_codebook[v_idx[sub_v], i]
    // -----------------------------------------------------------------
    float m = -INFINITY;
    float d = 0.0f;
    // o[D] in registers; D <= 256 in practice
    float o[256];
    for (uint i = 0; i < D; ++i) { o[i] = 0.0f; }

    uint q_abs_pos = q_pos;  // assumes S_q tokens align to the tail of S_kv

    for (uint k_pos = 0; k_pos < S_kv; ++k_pos) {
        // Causal / sliding window masks
        // Convention: prompt prefix + new tokens; the new q_pos correspond
        // to absolute positions [S_kv - S_q + q_pos] in the cached sequence.
        uint q_abs = (S_kv - S_q) + q_pos;
        if (causal && k_pos > q_abs) { continue; }
        if (use_window && (q_abs >= window_width) && (k_pos < q_abs - window_width + 1u)) {
            continue;
        }

        // Compute score via LUT lookup
        uint k_row_idx = (k_base_batch + h_kv * S_kv + k_pos) * n_sub;
        float score = 0.0f;
        for (uint sub = 0; sub < n_sub; ++sub) {
            uint c = k_indices[k_row_idx + sub];
            score += lut[sub * n_centroids + c];
        }
        score *= scale;

        // Online softmax update
        float m_new = max(m, score);
        float factor = exp(m - m_new);
        float w      = exp(score - m_new);
        d = d * factor + w;

        // Decode V_hat_k[dim] on the fly and accumulate
        uint v_row_idx = (k_base_batch + h_kv * S_kv + k_pos) * n_sub_v;
        for (uint sub_v = 0; sub_v < n_sub_v; ++sub_v) {
            uint v_c = v_indices[v_row_idx + sub_v];
            uint vcb_off = v_c * sub_dim_v;
            uint o_off = sub_v * sub_dim_v;
            for (uint i = 0; i < sub_dim_v; ++i) {
                o[o_off + i] = o[o_off + i] * factor + w * v_codebook[vcb_off + i];
            }
        }
        m = m_new;
    }

    float inv_d = 1.0f / max(d, 1e-20f);
    for (uint i = 0; i < D; ++i) {
        out[out_base + i] = o[i] * inv_d;
    }
"""


# ---------------------------------------------------------------------------
# Python wrapper around the kernel.  Handles reshaping and compile caching.
# ---------------------------------------------------------------------------
_kernel_cache: dict = {}


def _get_kernel(n_centroids: int):
    """Compile a kernel specialized for the given n_centroids.

    n_centroids is baked into the kernel as a template constant so that
    the per-thread LUT array can be a fixed-size stack buffer (Metal
    won't accept a variable-size array in registers).
    """
    if n_centroids > 256:
        raise ValueError(
            f"Fused SDPA kernel supports n_centroids <= 256, got {n_centroids}. "
            f"Use the pure-MLX path for larger codebooks."
        )
    if n_centroids not in _kernel_cache:
        src = _FUSED_SDPA_SRC.replace("LUT_N_CENTROIDS", str(n_centroids))
        _kernel_cache[n_centroids] = mx.fast.metal_kernel(
            name=f"vecinfer_fused_sdpa_c{n_centroids}",
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
    return _kernel_cache[n_centroids]


def metal_fused_sdpa(
    q_tilde: mx.array,  # [B, H_q, S_q, D] fp32 — already transformed
    k_indices: mx.array,  # [B, H_kv, S_kv, n_sub] int32 / uint32
    k_codebook: mx.array,  # [n_centroids, sub_dim] fp32
    v_indices: mx.array,  # [B, H_kv, S_kv, n_sub_v] int32 / uint32
    v_codebook: mx.array,  # [n_centroids_v, sub_dim_v] fp32
    scale: float,
    causal: bool = True,
    sliding_window: int = 0,
) -> mx.array:
    """Fused VecInfer attention from compressed indices — Metal kernel.

    Returns ``[B, H_q, S_q, D]`` fp32 attention output (caller can cast
    to fp16 if desired).
    """
    B, H_q, S_q, D = q_tilde.shape
    B2, H_kv, S_kv, n_sub = k_indices.shape
    n_centroids, sub_dim = k_codebook.shape
    _, H_kv2, S_kv2, n_sub_v = v_indices.shape
    n_centroids_v, sub_dim_v = v_codebook.shape
    assert B == B2, f"batch mismatch q={B} k={B2}"
    assert H_kv == H_kv2
    assert S_kv == S_kv2
    assert D == n_sub * sub_dim, f"D={D} != n_sub*sub_dim={n_sub * sub_dim}"
    assert D == n_sub_v * sub_dim_v, f"D={D} != n_sub_v*sub_dim_v={n_sub_v * sub_dim_v}"
    if n_sub > 16:
        raise ValueError(f"Fused SDPA: n_sub<=16 only, got {n_sub}")
    if n_centroids != n_centroids_v:
        raise ValueError(
            "Fused SDPA: key codebook and value codebook must have the same "
            f"n_centroids for now; got {n_centroids} vs {n_centroids_v}."
        )

    # Flatten to the layouts the kernel expects
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

    kernel = _get_kernel(n_centroids)

    outputs = kernel(
        inputs=[q_flat, k_idx_flat, k_cb, v_idx_flat, v_cb, params, scale_arr, slide_arr],
        output_shapes=[(B * H_q * S_q, D)],
        output_dtypes=[mx.float32],
        grid=(B * H_q, S_q, 1),
        threadgroup=(min(B * H_q, 32), min(S_q, 8), 1),
    )
    return outputs[0].reshape(B, H_q, S_q, D)


# ===========================================================================
# Pure-MLX reference: dequantize K and V, run standard SDPA
# ===========================================================================
def reference_sdpa(
    q: mx.array,  # [B, H_q, S_q, D] fp16   -- *untransformed* queries
    k_indices: mx.array,  # [B, H_kv, S_kv, n_sub]
    k_codebook: mx.array,  # [n_centroids, sub_dim]
    smooth: mx.array,  # [H_kv, D] or [D] fp32
    H: mx.array,  # [D, D] fp32
    v_indices: mx.array,  # [B, H_kv, S_kv, n_sub_v]
    v_codebook: mx.array,  # [n_centroids_v, sub_dim_v]
    scale: float,
    causal: bool = True,
    sliding_window: int = 0,
) -> mx.array:
    """Reference path: rebuild K_hat / V_hat in fp32, run pure-MLX SDPA.

    This matches what ``VecInferKVCache.update_and_fetch + mlx_lm SDPA``
    currently does, except we keep everything in fp32 for a fair
    correctness comparison (the fused kernel also accumulates in fp32).
    """
    B, H_q, S_q, D = q.shape
    _, H_kv, S_kv, n_sub = k_indices.shape
    _, _, _, n_sub_v = v_indices.shape

    # Dequantize keys back into transformed space, then invert
    k_hat_tilde = dequantize_vq(k_indices, k_codebook)  # [B, H_kv, S_kv, D]
    # invert: K_hat = (K_tilde_hat @ H.T) * lambda
    k_hat = k_hat_tilde.astype(mx.float32) @ H.T.astype(mx.float32)
    if smooth.ndim == 2 and k_hat.shape[-3] == smooth.shape[0]:
        sm_b = smooth[:, None, :].astype(mx.float32)
    elif smooth.ndim == 2:
        sm_b = mx.mean(smooth, axis=0).astype(mx.float32)
    else:
        sm_b = smooth.astype(mx.float32)
    k_hat = k_hat * sm_b

    # Values: simple dequant, no transform
    v_hat = dequantize_vq(v_indices, v_codebook).astype(mx.float32)

    # GQA broadcast — repeat KV heads to match query heads
    rep = H_q // H_kv
    if rep > 1:
        k_hat = mx.repeat(k_hat, repeats=rep, axis=1)
        v_hat = mx.repeat(v_hat, repeats=rep, axis=1)

    q32 = q.astype(mx.float32)
    # scores: [B, H_q, S_q, S_kv]
    scores = (q32 @ mx.swapaxes(k_hat, -2, -1)) * scale

    # Masks
    if causal or sliding_window:
        # absolute positions of queries within S_kv
        q_pos = mx.arange(S_q) + (S_kv - S_q)
        k_pos = mx.arange(S_kv)
        if causal:
            causal_mask = q_pos[:, None] < k_pos[None, :]
            scores = mx.where(causal_mask, mx.array(-1e9, dtype=mx.float32), scores)
        if sliding_window and sliding_window > 0:
            window_mask = k_pos[None, :] < (q_pos[:, None] - sliding_window + 1)
            scores = mx.where(window_mask, mx.array(-1e9, dtype=mx.float32), scores)

    weights = mx.softmax(scores, axis=-1)
    out = weights @ v_hat
    return out


# ===========================================================================
# Test harness
# ===========================================================================
def _make_test_inputs(
    B: int,
    H_q: int,
    H_kv: int,
    S_q: int,
    S_kv: int,
    D: int,
    sub_dim: int,
    n_centroids: int,
    seed: int = 42,
) -> dict:
    """Generate a consistent test fixture: random queries, calibrated codebook,
    and indices computed by quantizing random key vectors against the codebook.
    """
    rng = np.random.default_rng(seed)
    n_sub = D // sub_dim
    # Codebook: random unit-ish normal vectors
    cb = rng.standard_normal((n_centroids, sub_dim)).astype(np.float32) * 0.5
    # Generate random "true" key vectors then encode against the codebook
    # so the indices are consistent with the codebook (not just random ints).
    raw_keys = rng.standard_normal((B, H_kv, S_kv, D)).astype(np.float32) * 0.5
    # Per-sub-vector nearest-centroid (numpy, simple) — only run once at setup.
    raw_sub = raw_keys.reshape(B, H_kv, S_kv, n_sub, sub_dim)
    diff = raw_sub[..., None, :] - cb[None, None, None, None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    k_idx_np = np.argmin(d2, axis=-1).astype(np.uint32)  # [B, H_kv, S_kv, n_sub]

    # Same for V
    raw_vals = rng.standard_normal((B, H_kv, S_kv, D)).astype(np.float32) * 0.5
    raw_v_sub = raw_vals.reshape(B, H_kv, S_kv, n_sub, sub_dim)
    diff_v = raw_v_sub[..., None, :] - cb[None, None, None, None, :, :]
    d2_v = np.sum(diff_v * diff_v, axis=-1)
    v_idx_np = np.argmin(d2_v, axis=-1).astype(np.uint32)

    # Queries (fp16): mlx_lm passes fp16 queries
    q_np = rng.standard_normal((B, H_q, S_q, D)).astype(np.float32) * 0.3
    # Smooth factors: per-(H_kv, D) — what calibrate_smooth_factors returns
    smooth_np = (np.abs(raw_keys).max(axis=(0, 2)) + 1e-4) ** 0.5  # [H_kv, D]

    return {
        "q": mx.array(q_np).astype(mx.float16),
        "k_indices": mx.array(k_idx_np),
        "v_indices": mx.array(v_idx_np),
        "codebook": mx.array(cb),
        "smooth": mx.array(smooth_np.astype(np.float32)),
        "H": walsh_hadamard_matrix(D, dtype=mx.float32),
        "scale": 1.0 / float(D) ** 0.5,
        "shape": dict(
            B=B,
            H_q=H_q,
            H_kv=H_kv,
            S_q=S_q,
            S_kv=S_kv,
            D=D,
            sub_dim=sub_dim,
            n_sub=n_sub,
            n_centroids=n_centroids,
        ),
    }


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def correctness(fixture: dict, *, causal: bool, sliding_window: int = 0) -> Tuple[bool, float]:
    q = fixture["q"]
    k_idx = fixture["k_indices"]
    v_idx = fixture["v_indices"]
    cb = fixture["codebook"]
    smooth = fixture["smooth"]
    H = fixture["H"]
    scale = fixture["scale"]

    # Reference: full SDPA on dequantized K_hat / V_hat
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
        sliding_window=sliding_window,
    )

    # Fused: queries must be transformed first (smooth then Hadamard)
    q_tilde = apply_dual_transform_queries(q.astype(mx.float32), smooth, H)
    out_metal = metal_fused_sdpa(
        q_tilde=q_tilde,
        k_indices=k_idx,
        k_codebook=cb,
        v_indices=v_idx,
        v_codebook=cb,
        scale=scale,
        causal=causal,
        sliding_window=sliding_window,
    )
    mx.eval(out_ref, out_metal)
    diff = _max_abs_diff(out_ref, out_metal)
    ok = diff < 1e-2
    return ok, diff


def benchmark(fixture: dict, iters: int = 30, warmup: int = 3) -> dict:
    q = fixture["q"]
    k_idx = fixture["k_indices"]
    v_idx = fixture["v_indices"]
    cb = fixture["codebook"]
    smooth = fixture["smooth"]
    H = fixture["H"]
    scale = fixture["scale"]
    q_tilde = apply_dual_transform_queries(q.astype(mx.float32), smooth, H)

    # Warmup both paths
    for _ in range(warmup):
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
        b = metal_fused_sdpa(
            q_tilde=q_tilde,
            k_indices=k_idx,
            k_codebook=cb,
            v_indices=v_idx,
            v_codebook=cb,
            scale=scale,
            causal=True,
        )
        mx.eval(a, b)

    times_ref, times_met = [], []
    for _ in range(iters):
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
        times_ref.append(time.perf_counter() - t0)

    for _ in range(iters):
        t0 = time.perf_counter()
        b = metal_fused_sdpa(
            q_tilde=q_tilde,
            k_indices=k_idx,
            k_codebook=cb,
            v_indices=v_idx,
            v_codebook=cb,
            scale=scale,
            causal=True,
        )
        mx.eval(b)
        times_met.append(time.perf_counter() - t0)

    med_ref = float(np.median(times_ref)) * 1e3
    med_met = float(np.median(times_met)) * 1e3
    return {
        "pure_ms": med_ref,
        "metal_ms": med_met,
        "speedup": med_ref / med_met if med_met > 0 else float("inf"),
    }


def main() -> int:
    if not metal_available():
        print("Metal unavailable — aborting.")
        return 1

    print(f"Device: {mx.default_device()}")

    # The realistic shape from the prompt
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

    print("\n=== Correctness ===")
    ok_c, diff_c = correctness(fixture, causal=True)
    print(f"  causal=True, sliding=0  → max|diff|={diff_c:.4e}  {'OK' if ok_c else 'FAIL'}")
    ok_n, diff_n = correctness(fixture, causal=False)
    print(f"  causal=False             → max|diff|={diff_n:.4e}  {'OK' if ok_n else 'FAIL'}")
    ok_w, diff_w = correctness(fixture, causal=True, sliding_window=128)
    print(f"  causal=True, sliding=128 → max|diff|={diff_w:.4e}  {'OK' if ok_w else 'FAIL'}")

    all_ok = ok_c and ok_n and ok_w
    if not all_ok:
        print("\nCORRECTNESS FAILED — kernel is wrong.  Do not integrate.")
        return 2

    print("\n=== Benchmark (median of 30 iters) ===")
    perf = benchmark(fixture)
    print(f"  pure-MLX reference: {perf['pure_ms']:.2f} ms")
    print(f"  fused Metal kernel: {perf['metal_ms']:.2f} ms")
    print(f"  speedup:            {perf['speedup']:.2f}x")

    print("\nAll correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

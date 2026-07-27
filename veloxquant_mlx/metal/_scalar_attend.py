"""Universal fused group-affine decode + attention Metal kernel.

Single-dispatch scaled-dot-product attention directly over an
asymmetric group-min/max ("affine") quantized KV cache — the KIVI /
SKVQ / Kitty / group-quant family. Keys and values are stored as
uint8 codes plus per-group ``(scale, zero)``; the kernel reconstructs
each element in-register as ``x_hat = code * scale + zero`` and runs a
FlashAttention-style online softmax. No dequantized ``K_hat`` / ``V_hat``
tensor is ever materialized in DRAM.

This is the scalar/group-quant analogue of the codebook fused attends
(:mod:`_rvq_attend`, :mod:`fused_sdpa`, :mod:`_rabitq_attend`). It kills
the ``dequantize -> DRAM -> SDPA`` round-trip that the pure-MLX path
(reconstruct ``code*scale+zero`` into a full fp16 tensor, then call
``scaled_dot_product_attention``) pays every decode step. At long
context that round-trip dominates: the fp16 K_hat grows linearly with
S_kv while the packed codes are ``16/b`` times smaller.

Quantization layout (matches KIVI's ``_quant_dequant_along``)
------------------------------------------------------------
Keys — per-CHANNEL groups (group along the token axis):
  * ``k_codes [B, H, S_kv, D]``  uint8 codes in ``[0, 2^b - 1]``
  * ``k_scale [B, H, GK,   D]``  fp32, GK = ceil(S_kv / g)
  * ``k_zero  [B, H, GK,   D]``  fp32
  reconstruction: ``k_hat[.., sk, d] = k_codes[..,sk,d] * k_scale[..,sk/g,d] + k_zero[..,sk/g,d]``

Values — per-TOKEN groups (group along the channel axis):
  * ``v_codes [B, H, S_kv, D]``  uint8
  * ``v_scale [B, H, S_kv, GV]`` fp32, GV = ceil(D / g)
  * ``v_zero  [B, H, S_kv, GV]`` fp32
  reconstruction: ``v_hat[.., sk, d] = v_codes[..,sk,d] * v_scale[..,sk,d/g] + v_zero[..,sk,d/g]``

Score model (per kv slot sk, query (b, h, sq)):

    score_sk = (sum_d q[..,sq,d] * k_hat[..,sk,d]) * scale

``scale`` (typically ``1/sqrt(D)``) is passed in and applied by the
kernel; no other scaling is folded in.

Public API:
  - :func:`scalar_fused_decode_attend`
"""
from __future__ import annotations

import mlx.core as mx

_cache: dict = {}


# ===========================================================================
# Metal source — fused affine decode + flash-decoding attend
# ===========================================================================
# Grid:        (B * H * S_q * 32, NSG_C, 1) — MLX grid = total threads.
# Threadgroup: (32, NSG_C, 1)               — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup handles one query position (b, h, sq). A single
# 32-lane pass over S_kv under-fills the GPU for decode shapes
# (B*H*S_q small), so the kv axis is split flash-decoding style:
# SIMD-group sg processes slots sk = sg, sg + NSG_C, ... with its own
# online softmax (running_m, running_d, my_out[]). The NSG_C partial
# results are merged through threadgroup memory at the end (identical
# merge to _rabitq_attend).
#
# Within one SIMD-group the 32 lanes stripe the D-dim vectors in steps
# of 32; simd_sum reduces the partial dot product, so the per-slot loop
# needs no barriers. Only the final cross-SIMD-group merge barriers.
#
# GK (key groups along tokens) and GV (value groups along channels) and
# the group size G are read from the passed shapes / a param, so one
# compiled kernel serves any (S_kv, D, g).

_SCALAR_AFFINE_ATTEND_SRC = r"""
    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;

    uint B    = uint(q_shape[0]);
    uint H    = uint(q_shape[1]);
    uint S_q  = uint(q_shape[2]);
    uint D    = uint(q_shape[3]);
    uint S_kv = uint(k_codes_shape[2]);
    uint GK   = uint(k_scale_shape[2]);   // key groups along tokens
    uint GV   = uint(v_scale_shape[3]);   // value groups along channels
    (void)B;

    uint G        = uint(gsize[0]);       // group size
    float scale   = scale_arr[0];
    uint  NSG     = uint(NSG_C);

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H;
    uint b_idx  = tg / (S_q * H);

    uint q_base   = ((b_idx * H + h_idx) * S_q + sq_idx) * D;
    uint bh       = b_idx * H + h_idx;

    // ----- per-lane online-softmax state for this SIMD-group -----
    float running_m = -INFINITY;
    float running_d = 0.0f;
    float my_out[8];                       // D/32 <= 256/32 = 8
    for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;
    uint n_owned = (D + 31u) / 32u;

    // ----- main loop: this SIMD-group strides kv by NSG -----
    for (uint sk = sg; sk < S_kv; sk += NSG) {
        // key group index along tokens for this slot
        uint kg = sk / G;

        // partial dot product over lane-owned dims
        float partial_dot = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            uint  code_off = (bh * S_kv + sk) * D + d;
            uint  ks_off   = (bh * GK + kg) * D + d;
            float k_hat = float(k_codes[code_off]) * k_scale[ks_off]
                        + k_zero[ks_off];
            partial_dot += float(q[q_base + d]) * k_hat;
        }
        float score = simd_sum(partial_dot) * scale;

        // online softmax update (every lane holds identical score)
        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score      - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;
        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        // value decode + weighted accumulate (per-token groups over channels)
        for (uint d = lane; d < D; d += 32u) {
            uint  vg      = d / G;
            uint  code_off = (bh * S_kv + sk) * D + d;
            uint  vs_off   = (bh * S_kv + sk) * GV + vg;
            float v_hat = float(v_codes[code_off]) * v_scale[vs_off]
                        + v_zero[vs_off];
            uint  out_i = (d - lane) / 32u;
            my_out[out_i] += w * v_hat;
        }
    }

    // ----- merge the NSG partial softmaxes through threadgroup memory -----
    // sh_o layout: [NSG_C][8][32]  (SIMD-group, owned-slot, lane).
    threadgroup float sh_m[NSG_C];         // one slot per SIMD-group
    threadgroup float sh_d[NSG_C];
    threadgroup float sh_o[NSG_C * 8 * 32];

    // stash this SIMD-group's per-lane partials
    for (uint i = 0; i < n_owned; ++i) {
        sh_o[(sg * 8u + i) * 32u + lane] = my_out[i];
    }
    if (lane == 0) { sh_m[sg] = running_m; sh_d[sg] = running_d; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD-group 0 reduces across all groups
    if (sg == 0) {
        float gm = -INFINITY;
        for (uint s = 0; s < NSG; ++s) gm = metal::max(gm, sh_m[s]);
        float gd = 0.0f;
        for (uint s = 0; s < NSG; ++s) gd += sh_d[s] * metal::exp(sh_m[s] - gm);

        for (uint i = 0; i < n_owned; ++i) {
            float acc = 0.0f;
            for (uint s = 0; s < NSG; ++s) {
                acc += sh_o[(s * 8 + i) * 32 + lane]
                     * metal::exp(sh_m[s] - gm);
            }
            uint d = lane + i * 32u;
            if (d < D) {
                uint out_off = ((b_idx * H + h_idx) * S_q + sq_idx) * D + d;
                out[out_off] = half(acc / gd);
            }
        }
    }
"""


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def _scalar_affine_attend_kernel(D: int, nsg: int):
    key = ("scalar_affine_attend", D, nsg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"scalar_affine_attend_d{D}_nsg{nsg}",
            input_names=[
                "q",
                "k_codes", "k_scale", "k_zero",
                "v_codes", "v_scale", "v_zero",
                "gsize", "scale_arr",
            ],
            output_names=["out"],
            header=f"#define NSG_C {nsg}\n",
            source=_SCALAR_AFFINE_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scalar_fused_decode_attend(
    q: mx.array,
    k_codes: mx.array,
    k_scale: mx.array,
    k_zero: mx.array,
    v_codes: mx.array,
    v_scale: mx.array,
    v_zero: mx.array,
    group_size: int,
    scale: float,
    nsg: int = 4,
) -> mx.array:
    """Fused group-affine (KIVI-style) key/value decode + SDP attention.

    Reconstructs ``k_hat = k_codes*k_scale + k_zero`` (per-channel groups)
    and ``v_hat = v_codes*v_scale + v_zero`` (per-token groups) on-the-fly
    inside an online-softmax loop — no intermediate fp16 K_hat/V_hat.

    Args:
        q:        ``[B, H, S_q, D]`` fp16/fp32 queries (pre-rotated).
        k_codes:  ``[B, H, S_kv, D]`` uint8 key codes.
        k_scale:  ``[B, H, GK, D]``   fp32, GK = ceil(S_kv/group_size).
        k_zero:   ``[B, H, GK, D]``   fp32.
        v_codes:  ``[B, H, S_kv, D]`` uint8 value codes.
        v_scale:  ``[B, H, S_kv, GV]`` fp32, GV = ceil(D/group_size).
        v_zero:   ``[B, H, S_kv, GV]`` fp32.
        group_size: quantization group size g.
        scale:    attention scale applied to the raw dot (e.g. 1/sqrt(D)).
        nsg:      SIMD-groups per threadgroup splitting the kv axis.

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"scalar_fused_decode_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D > 256:
        raise ValueError(f"scalar_fused_decode_attend: D={D} must be <= 256")
    if not (1 <= nsg <= 32):
        raise ValueError(f"scalar_fused_decode_attend: nsg={nsg} must be in 1..32")

    n_tg = B * H * S_q
    gsize = mx.array([group_size], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    outputs = _scalar_affine_attend_kernel(D, nsg)(
        inputs=[
            q.astype(mx.float16),
            k_codes.astype(mx.uint8),
            k_scale.astype(mx.float32),
            k_zero.astype(mx.float32),
            v_codes.astype(mx.uint8),
            v_scale.astype(mx.float32),
            v_zero.astype(mx.float32),
            gsize,
            scale_arr,
        ],
        # MLX grid = total threads; NSG_C SIMD-groups of 32 lanes each
        grid=(n_tg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["scalar_fused_decode_attend"]

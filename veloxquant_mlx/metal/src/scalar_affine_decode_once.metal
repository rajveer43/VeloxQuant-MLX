// scalar_affine_decode_once.metal
// Extracted from veloxquant_mlx/metal/_scalar_attend.py (_SCALAR_AFFINE_DECODE_K_SRC /
// _SCALAR_AFFINE_DECODE_V_SRC) — see that file's docstring for the algorithm
// and issue #308's spike rationale. This file is read as plain text at
// import time via _read_kernel_source() and JIT-compiled by
// mx.fast.metal_kernel at call time; it is NOT separately compiled
// (see issue #64).
//
// Flat elementwise decode of one (batch, kv_head, s, d) code into fp16,
// writing a full [B, H_kv, S, D] device buffer once. Dispatched at
// B*H_kv*S*D threadgroups' worth of threads -- large and occupancy-friendly
// by construction, unlike the tiny B*H*S_q attend dispatch this feeds.
//
// Two group layouts share this same flat-index shape (K: groups along
// tokens; V: groups along channels) so this file is compiled twice with a
// DECODE_MODE_K / DECODE_MODE_V header switch selecting the group-index
// arithmetic; the element loop itself is identical.

    uint elem = thread_position_in_grid.x;
    uint N    = uint(codes_shape[0]) * uint(codes_shape[1]) * uint(codes_shape[2]) * uint(codes_shape[3]);
    if (elem >= N) return;

    uint D    = uint(codes_shape[3]);
    uint S    = uint(codes_shape[2]);
    uint G    = uint(gsize[0]);

    uint d  = elem % D;
    uint s  = (elem / D) % S;
    uint bh = elem / (D * S);   // flattened (b * H_kv + h_kv)

#if DECODE_MODE_K
    // K: per-CHANNEL groups (group along the token axis) -- scale/zero
    // shaped [B, H_kv, GK, D], GK = ceil(S/G).
    uint gk       = s / G;
    uint sz_off   = (bh * uint(scale_shape[2]) + gk) * D + d;
#else
    // V: per-TOKEN groups (group along the channel axis) -- scale/zero
    // shaped [B, H_kv, S, GV], GV = ceil(D/G).
    uint GV       = uint(scale_shape[3]);
    uint gv       = d / G;
    uint sz_off   = (bh * S + s) * GV + gv;
#endif

    float hat = float(codes[elem]) * scale[sz_off] + zero[sz_off];
    out[elem] = half(hat);

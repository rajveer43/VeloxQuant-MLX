"""Fused RaBitQ asymmetric decode + attention Metal kernel.

Single-dispatch attention over an asymmetric-precision KV cache:
1-bit packed key sign bits (RaBitQ style) scored via XOR + popcount,
fused with an online softmax and a 4-bit scalar-codebook value
gather-accumulate. No dequantized K matrix is ever materialized.
The kv axis is split across SIMD-groups flash-decoding style so
decode-shaped dispatches (B*H*S_q small) still fill the GPU.

Score model (per kv slot i, query position (b, h, sq)):

    score_i = (D - 2 * ham_i) * q_scale[b,h,sq] * k_mag[b,h,i] + k_const[b,h,i]

where ham_i = popcount(XOR(sign_bits(q), k_bits[i])). Under the RaBitQ
sign estimate q_hat ~ s_q * m_q, k_hat ~ s_k * m_k with s in {+-1}^D,
the inner product is <q_hat, k_hat> ~ (D - 2*ham) * m_q * m_k, so a
centroid-free cache passes q_scale = m_q, k_mag = m_k, k_const = 0.
Any attention scaling (1/sqrt(D)) must be folded into q_scale and
k_const by the caller — the kernel applies none of its own.

Bit conventions match the RaBitQ quantizer (_pack_signs): element
8*b + t maps to byte b, bit t (np.packbits bitorder='little'), and
q >= 0 encodes as bit 1.

Public API:
  - :func:`rabitq_fused_attend`
"""

from __future__ import annotations

import mlx.core as mx

_cache: dict = {}


# ===========================================================================
# Metal source — fused RaBitQ score + flash-decoding attend
# ===========================================================================
# Grid:        (B * H * S_q * 32, NSG_C, 1) — MLX grid = total threads.
# Threadgroup: (32, NSG_C, 1)               — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup handles one query position (b, h, sq). A single
# 32-thread pass over S_kv leaves the GPU nearly idle for decode-style
# shapes (B*H*S_q small), so the kv axis is split flash-decoding style:
# SIMD-group sg processes slots sk = sg, sg + NSG_C, ... with its own
# online softmax (running_m, running_d, my_out), and the NSG_C partial
# results are merged through threadgroup memory at the end.
#
# N_BYTES = D/8 <= 32 (enforced by the wrapper via D <= 256), so lane j
# owns packed byte j of the binarized query; lanes j >= N_BYTES
# contribute 0 to the Hamming sum. simd_sum spans one SIMD-group, so
# the per-slot loop needs no barriers — only the final merge does.
#
# Per kv-slot (within a SIMD-group):
#   1. Each owning lane XORs its query byte against the slot's packed
#      key byte and popcounts (portable bit-trick, as in _rabitq.py);
#      simd_sum yields the full Hamming distance.
#   2. Affine score from ham + per-query/per-key scalars (see module doc).
#   3. Online softmax update (running_m, running_d, factor).
#   4. Value gather v_cents[v_idx] striped across lanes, accumulate w * v.
#
# Merge: m* = max_sg(m_sg); each partial is rescaled by exp(m_sg - m*)
# and summed; empty SIMD-groups (S_kv < NSG_C) carry m_sg = -INF and
# d_sg = 0, so their rescale factor is exp(-INF) = 0 and they drop out.

_RABITQ_ATTEND_SRC = r"""
    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;

    uint B    = uint(q_shape[0]);
    uint H    = uint(q_shape[1]);
    uint S_q  = uint(q_shape[2]);
    uint D    = uint(q_shape[3]);
    uint S_kv = uint(k_bits_shape[2]);
    (void)B;

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H;
    uint b_idx  = tg / (S_q * H);

    uint q_base  = ((b_idx * H + h_idx) * S_q + sq_idx) * D;
    uint kv_base = (b_idx * H + h_idx) * S_kv;

    // Binarize the query once: lane j packs byte j (little-endian bit order,
    // q >= 0 -> bit 1, matching _pack_signs in the RaBitQ quantizer).
    uint my_qbyte = 0u;
    if (lane < uint(N_BYTES)) {
        for (uint t = 0; t < 8u; ++t) {
            float qv = float(q[q_base + lane * 8u + t]);
            my_qbyte |= (qv >= 0.0f ? 1u : 0u) << t;
        }
    }

    float qs = q_scale[(b_idx * H + h_idx) * S_q + sq_idx];

    float running_m = -INFINITY;
    float running_d = 0.0f;

    // Per-lane output accumulator; max 8 slots (D=256, 32 lanes)
    float my_out[8];
    for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;
    uint n_owned = (D + 31u) / 32u;

    for (uint sk = sg; sk < S_kv; sk += uint(NSG_C)) {
        // 1. Hamming distance via XOR + popcount — keys stay packed.
        uint partial_ham = 0u;
        if (lane < uint(N_BYTES)) {
            uint xr = my_qbyte ^ uint(k_bits[(kv_base + sk) * uint(N_BYTES) + lane]);
            // popcount via bit manipulation (portable across Metal versions)
            uint v = xr;
            v = v - ((v >> 1u) & 0x55u);
            v = (v & 0x33u) + ((v >> 2u) & 0x33u);
            v = (v + (v >> 4u)) & 0x0Fu;
            partial_ham = v;
        }
        uint ham = simd_sum(partial_ham);

        // 2. Affine score: <q, k> estimate = (D - 2*ham) * m_q * m_k + bias.
        float score = (float(D) - 2.0f * float(ham)) * qs * k_mag[kv_base + sk]
                    + k_const[kv_base + sk];

        // 3. Online softmax update
        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score     - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;

        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        // 4. 4-bit value codebook gather + weighted accumulate.
        // V_PACKED is a compile-time constant, so the branch folds away:
        // packed stores two 4-bit indices per byte (lo nibble = even dim).
        for (uint i = lane; i < D; i += 32u) {
            uint idx;
            if (V_PACKED != 0) {
                uint byte = uint(v_idx[(kv_base + sk) * (D >> 1u) + (i >> 1u)]);
                idx = (i & 1u) ? (byte >> 4u) : (byte & 0xFu);
            } else {
                idx = uint(v_idx[(kv_base + sk) * D + i]);
            }
            float vi = v_cents[idx];
            my_out[(i - lane) / 32u] += w * vi;
        }
    }

    // Publish per-SIMD-group partials, then merge across SIMD-groups.
    threadgroup float tg_m[NSG_C];
    threadgroup float tg_d[NSG_C];
    threadgroup float tg_out[NSG_C * MAX_D];

    if (lane == 0u) {
        tg_m[sg] = running_m;
        tg_d[sg] = running_d;
    }
    for (uint i = lane; i < D; i += 32u) {
        tg_out[sg * uint(MAX_D) + i] = my_out[(i - lane) / 32u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_star = -INFINITY;
    for (uint s = 0; s < uint(NSG_C); ++s) {
        m_star = metal::max(m_star, tg_m[s]);
    }
    float d_star = 0.0f;
    for (uint s = 0; s < uint(NSG_C); ++s) {
        d_star += metal::exp(tg_m[s] - m_star) * tg_d[s];
    }

    uint flat = sg * 32u + lane;
    for (uint i = flat; i < D; i += 32u * uint(NSG_C)) {
        float acc = 0.0f;
        for (uint s = 0; s < uint(NSG_C); ++s) {
            acc += metal::exp(tg_m[s] - m_star) * tg_out[s * uint(MAX_D) + i];
        }
        out[q_base + i] = half(acc / d_star);
    }
"""


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

# SIMD-groups per threadgroup for the flash-decoding kv split.
_N_SIMDGROUPS = 8


def _rabitq_attend_kernel(n_bytes: int, d: int, v_packed: bool):
    key = ("rabitq_fused_attend", n_bytes, d, v_packed)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rabitq_fused_attend_nb{n_bytes}_d{d}_vp{int(v_packed)}",
            input_names=[
                "q",
                "q_scale",
                "k_bits",
                "k_mag",
                "k_const",
                "v_idx",
                "v_cents",
            ],
            output_names=["out"],
            source=_RABITQ_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_fused_attend(
    q: mx.array,  # [B, H, S_q, D]    fp16 — queries (pre-rotated)
    q_scale: mx.array,  # [B, H, S_q]       fp32 — per-query score scale
    k_bits: mx.array,  # [B, H, S_kv, D/8] uint8 — packed 1-bit key signs
    k_mag: mx.array,  # [B, H, S_kv]      fp32 — per-key magnitude
    k_const: mx.array,  # [B, H, S_kv]      fp32 — per-key additive score bias
    v_idx: mx.array,  # [B, H, S_kv, D]   uint8 — value codebook indices
    v_cents: mx.array,  # [n_cents]         fp32 — scalar value codebook
) -> mx.array:
    """Fused asymmetric RaBitQ attention: 1-bit keys, codebook values.

    Scores every cached slot directly from packed sign bits
    (XOR + popcount), runs an online softmax, and accumulates values
    gathered from a scalar codebook — one dispatch, no dequantized K.

    Score per slot: ``(D - 2*ham) * q_scale * k_mag + k_const``. Callers
    must fold any attention scaling (``1/sqrt(D)``) into ``q_scale`` and
    ``k_const``; pass ``k_const = 0`` for the centroid-free scheme.

    Args:
        q:       ``[B, H, S_q, D]`` fp16 queries, pre-rotated into the
                 same space as the packed key bits. D divisible by 8,
                 D <= 256.
        q_scale: ``[B, H, S_q]`` fp32 per-query scale (e.g. ``L1(q)/D``).
        k_bits:  ``[B, H, S_kv, D//8]`` uint8 packed key sign bits
                 (little-endian bit order, ``>= 0`` -> 1).
        k_mag:   ``[B, H, S_kv]`` fp32 per-key magnitude (e.g. ``L1(k)/D``).
        k_const: ``[B, H, S_kv]`` fp32 additive score bias per key.
        v_idx:   ``[B, H, S_kv, D]`` uint8 per-element value indices, or
                 ``[B, H, S_kv, D//2]`` uint8 nibble-packed (two 4-bit
                 indices per byte, low nibble = even dim — see
                 :func:`rabitq_pack_values`). Format is detected from
                 the shape; packed halves value-cache memory and
                 bandwidth.
        v_cents: ``[n_cents]`` fp32 scalar value codebook (16 entries
                 for the 4-bit scheme; packed format requires <= 16).

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"rabitq_fused_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D % 8 != 0:
        raise ValueError(f"rabitq_fused_attend: D={D} must be divisible by 8")
    if D > 256:
        raise ValueError(f"rabitq_fused_attend: D={D} exceeds the 256 limit")
    n_bytes = D // 8

    if k_bits.ndim != 4 or k_bits.shape[:2] != (B, H) or k_bits.shape[3] != n_bytes:
        raise ValueError(
            f"rabitq_fused_attend: k_bits must be [B, H, S_kv, {n_bytes}], got {k_bits.shape}"
        )
    S_kv = k_bits.shape[2]
    if k_mag.shape != (B, H, S_kv):
        raise ValueError(f"rabitq_fused_attend: k_mag must be {(B, H, S_kv)}, got {k_mag.shape}")
    if k_const.shape != (B, H, S_kv):
        raise ValueError(
            f"rabitq_fused_attend: k_const must be {(B, H, S_kv)}, got {k_const.shape}"
        )
    if q_scale.shape != (B, H, S_q):
        raise ValueError(f"rabitq_fused_attend: q_scale must be {(B, H, S_q)}, got {q_scale.shape}")
    if v_idx.shape == (B, H, S_kv, D):
        v_packed = False
    elif v_idx.shape == (B, H, S_kv, D // 2):
        v_packed = True
    else:
        raise ValueError(
            f"rabitq_fused_attend: v_idx must be {(B, H, S_kv, D)} (one index "
            f"per element) or {(B, H, S_kv, D // 2)} (nibble-packed), got {v_idx.shape}"
        )
    if v_cents.ndim != 1:
        raise ValueError(f"rabitq_fused_attend: v_cents must be 1D, got {v_cents.shape}")
    if v_packed and v_cents.shape[0] > 16:
        raise ValueError(
            f"rabitq_fused_attend: nibble-packed v_idx can only address 16 "
            f"centroids, got v_cents with {v_cents.shape[0]}"
        )

    n_tg = B * H * S_q

    outputs = _rabitq_attend_kernel(n_bytes, D, v_packed)(
        inputs=[
            q.astype(mx.float16),
            q_scale.astype(mx.float32),
            k_bits.astype(mx.uint8),
            k_mag.astype(mx.float32),
            k_const.astype(mx.float32),
            v_idx.astype(mx.uint8),
            v_cents.astype(mx.float32),
        ],
        template=[
            ("N_BYTES", n_bytes),
            ("MAX_D", D),
            ("NSG_C", _N_SIMDGROUPS),
            ("V_PACKED", int(v_packed)),
        ],
        # MLX grid = total threads; one threadgroup of 32 x NSG threads
        # per query position (see kernel comment for the kv split).
        grid=(n_tg * 32, _N_SIMDGROUPS, 1),
        threadgroup=(32, _N_SIMDGROUPS, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["rabitq_fused_attend"]

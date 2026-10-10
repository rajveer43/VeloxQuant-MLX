// rvq_unpack_decode.metal
// Extracted from veloxquant_mlx/metal/_rvq_unpack_decode.py (_RVQ_UNPACK_DECODE_SRC) — see that
// file's docstring for the algorithm, memory-layout, and threadgroup strategy
// explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Fused TurboQuantRVQ packed-key decode: unpack both uint32 index streams,
// gather + sum the two codebooks, inverse randomized Hadamard, per-vector norm
// rescale with fp16 saturation. Replaces the ~10 MLX dispatches (and their
// full-size intermediates) in TurboQuantRVQKVCache._dequantize_range with one.
//
// Grid:        (N * D, 1, 1) — one thread per (vector, coordinate).
// Threadgroup: (D, 1, 1)     — one threadgroup per cached key vector.
//
// Every rounding step of the MLX reference is mirrored so the only source of
// difference is fp32 summation order inside the butterfly:
//   1. y        = half(c1[idx1] + c2[idx2])         (codebooks are fp16, the sum is fp16)
//   2. h        = fp32 WHT(y) / sqrt(D)             (mx.hadamard_transform is normalized)
//   3. x_unit   = half(h * diag[i])                 (apply_inverse casts back to fp16)
//   4. k_hat    = half(clamp(x_unit * norm, +-65504)) (_rescale_fp16: fp32 multiply, saturate)

    // Ping-pong buffers: one barrier per cross-SIMD-group stage instead of two.
    threadgroup float buf0[MAX_D];
    threadgroup float buf1[MAX_D];

    uint n    = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint D    = uint(MAX_D);

    constexpr uint ELEMS_PER_WORD = 32u / BITS;
    constexpr uint MASK           = (1u << BITS) - 1u;
    uint n_words = (D + ELEMS_PER_WORD - 1u) / ELEMS_PER_WORD;

    // Output row n = (bh, s) with s < S live tokens per (batch*head); the packed
    // streams and norms are read in place from a cache buffer with `cap` slots
    // per (batch*head), so no strided-slice copy is needed before the dispatch.
    uint S   = dims[0];
    uint cap = dims[1];
    uint bh  = n / S;
    uint row = bh * cap + (n - bh * S);

    // 1. Unpack: element `lane` lives in word lane/EPW at bit offset (lane%EPW)*BITS
    //    (LSB-first, matching _pack_indices / mx.quantize).
    uint w     = lane / ELEMS_PER_WORD;
    uint shift = (lane % ELEMS_PER_WORD) * BITS;
    uint idx1  = (packed1[row * n_words + w] >> shift) & MASK;
    uint idx2  = (packed2[row * n_words + w] >> shift) & MASK;

    // 2. Codebook gather; the reference sums two fp16 values in fp16.
    float v = float(half(centroids1[idx1] + centroids2[idx2]));

    // 3. WHT butterfly. Stages with stride < 32 stay inside one SIMD-group, so
    //    they exchange through simd_shuffle_xor with no barrier. Stage math is
    //    identical to hadamard_quantize.metal: upper lane = partner - self,
    //    lower lane = self + partner.
    constexpr uint SIMD_W = 32u;
    uint in_simd = (D < SIMD_W) ? D : SIMD_W;
    for (uint stride = 1; stride < in_simd; stride <<= 1) {
        float b = simd_shuffle_xor(v, ushort(stride));
        v = ((lane & stride) != 0u) ? (b - v) : (v + b);
    }
    // Remaining stages cross SIMD-groups: exchange through threadgroup memory.
    uint stage = 0u;
    for (uint stride = SIMD_W; stride < D; stride <<= 1) {
        threadgroup float* buf = (stage & 1u) ? buf1 : buf0;
        buf[lane] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float b = buf[lane ^ stride];
        v = ((lane & stride) != 0u) ? (b - v) : (v + b);
        ++stage;
    }

    // 4. Normalize, undo the sign diagonal, round to fp16 like apply_inverse.
    float h      = v * metal::rsqrt(float(D));
    float x_unit = float(half(h * diag[lane]));

    // 5. Norm rescale in fp32, saturate to fp16's range (see _rescale_fp16).
    float k_hat = x_unit * float(norms[row]);
    k_hat = metal::clamp(k_hat, -65504.0f, 65504.0f);
    out[n * D + lane] = half(k_hat);

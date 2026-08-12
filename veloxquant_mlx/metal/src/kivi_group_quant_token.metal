// kivi_group_quant_token.metal
// Extracted from veloxquant_mlx/metal/_kivi_quant.py (_KIVI_QUANT_TOKEN_SRC) — see that
// file's docstring for the algorithm, memory-layout, and threadgroup strategy
// explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// KIVI **per-token** group quantize->dequantize (values): the quantization
// group runs along the CHANNEL axis, one (min, max) pair per (token, channel
// group).
//
// Input is [BH, S, D] row-contiguous, so here the quantization axis (D) is the
// CONTIGUOUS one — the mirror image of the per-channel kernel, and it wants the
// opposite thread mapping.
//
// Grid:        (n_groups * 32, 1, 1); n_groups = BH * S * ceil(D/GROUP_SIZE).
// Threadgroup: (32, 1, 1) — exactly one SIMD group per quantization group.
//
// Lanes split the group's elements, so a warp's loads are 32 adjacent addresses
// (fully coalesced), then a `simd_shuffle_xor` butterfly reduces the per-lane
// (min, max) to a group-wide pair. The butterfly needs **no threadgroup memory
// and no barriers** — lanes in a SIMD group advance in lockstep — which is why
// this is preferred over a threadgroup tree reduction.
//
// At KIVI's default GROUP_SIZE=32 this is an exact fit: one lane per element,
// one butterfly, done. Larger groups grid-stride first; smaller groups leave
// the surplus lanes at the identity (+/-INFINITY), which reduces correctly but
// wastes lanes — acceptable, as sub-warp group sizes are not KIVI's regime.

    uint lane = thread_position_in_threadgroup.x;
    uint gid  = threadgroup_position_in_grid.x;

    const uint S   = x_shape[1];
    const uint NGD = (DHEAD + GROUP_SIZE - 1u) / GROUP_SIZE;  // channel groups

    // Whole threadgroups exit together, so every lane still reaches the
    // butterfly below — a divergent return would deadlock the shuffle.
    if (gid >= x_shape[0] * S * NGD) { return; }

    const uint gd       = gid % NGD;
    const uint row      = gid / NGD;        // flat (bh, s)
    const uint row_base = row * DHEAD;
    const uint d0       = gd * GROUP_SIZE;
    const uint d1       = min(d0 + GROUP_SIZE, DHEAD);

    // Both passes read global memory rather than carrying the group in
    // registers. Caching was measured at exactly 1.00x-1.02x here: the second
    // read hits cache, so it buys nothing and only adds code. (In the channel
    // kernel the same idea is actively harmful — 0.76x — because there a whole
    // group is GROUP_SIZE floats per thread.)
    float gmin =  INFINITY;
    float gmax = -INFINITY;
    for (uint i = lane; i < GROUP_SIZE; i += 32u) {
        uint d = d0 + i;
        if (d < d1) {
            float v = float(x[row_base + d]);
            gmin = min(gmin, v);
            gmax = max(gmax, v);
        }
    }
    // Ragged final group: the reference pads by replicating this row's LAST
    // channel. Folding it in on every lane is idempotent for min/max, so it
    // needs no lane-0 special case.
    if (d1 < d0 + GROUP_SIZE) {
        float pad_val = float(x[row_base + DHEAD - 1u]);
        gmin = min(gmin, pad_val);
        gmax = max(gmax, pad_val);
    }

    // Butterfly: after 5 XOR shuffles every lane holds the group-wide min/max.
    for (uint off = 16u; off > 0u; off >>= 1u) {
        gmin = min(gmin, simd_shuffle_xor(gmin, off));
        gmax = max(gmax, simd_shuffle_xor(gmax, off));
    }

    const float scale = max((gmax - gmin) / float(LEVELS), float(EPS));

    for (uint i = lane; i < GROUP_SIZE; i += 32u) {
        uint d = d0 + i;
        if (d < d1) {
            float v = float(x[row_base + d]);
            // rint == round-half-to-even, matching mx.round. metal::round would
            // be round-half-away-from-zero and would disagree on exact .5 codes.
            float q = clamp(rint((v - gmin) / scale), 0.0f, float(LEVELS));
            // Two separately-rounded ops, NOT an fma — contraction shifts
            // ~0.02% of elements by 1 ULP away from the MLX reference.
            float prod = q * scale;
            out[row_base + d] = T(prod + gmin);
        }
    }

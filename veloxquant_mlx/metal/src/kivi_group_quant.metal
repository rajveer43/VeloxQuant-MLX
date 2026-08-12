// kivi_group_quant.metal
// Extracted from veloxquant_mlx/metal/_kivi_quant.py (_KIVI_GROUP_QUANT_SRC) — see that
// file's docstring for the algorithm, memory-layout, and threadgroup strategy
// explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Fused KIVI asymmetric group quantize->dequantize round-trip.
//
// Input is viewed as [R, L] where L is the quantization axis: the caller
// moves the quant axis last (axis=-2 per-channel keys, axis=-1 per-token
// values) so this kernel is axis-agnostic. Each group of GROUP_SIZE
// consecutive elements along L gets one (min, max) pair.
//
// Grid:        (n_row_groups * TG, 1, 1) — one threadgroup per (row, group).
// Threadgroup: (TG, 1, 1).
//
// Each threadgroup owns exactly one group and does, in three phases:
//   1. Strided load of its <= GROUP_SIZE live elements, tracking a per-thread
//      running (min, max). Elements past L (the pad region) replicate the
//      row's LAST live element, matching the MLX path's edge-replication pad.
//   2. Threadgroup tree reduction to a single (gmin, gmax).
//   3. Re-read, quantize with scale = max((gmax-gmin)/levels, eps), round,
//      clip to [0, levels], reconstruct q*scale+gmin, and write back only the
//      live (non-pad) elements.
//
// Phase 1 and 3 both read the input rather than staging the group in
// threadgroup memory: GROUP_SIZE is a runtime #define and staging would cap
// the group size at whatever fits in threadgroup memory. The re-read hits the
// same cache lines that phase 1 just touched.
//
// Rounding note: MLX's mx.round is round-half-to-even; Metal's rint() under
// the default rounding mode is likewise round-half-to-even. metal::round()
// would be round-half-away-from-zero and would NOT match — this is the one
// place where the obvious function is the wrong one.

    uint tid   = thread_position_in_threadgroup.x;
    uint gid   = threadgroup_position_in_grid.x;   // flat (row, group) index

    // R and L arrive as runtime buffers, NOT #defines. Baking them into the
    // header would compile a fresh shader for every distinct sequence length
    // — i.e. one per decode step — which turns the dispatch cache into an
    // unbounded shader-compilation leak. Only GROUP_SIZE/LEVELS/EPS/T (a tiny
    // fixed set) are specialized.
    const uint R = shape[0];
    const uint L = shape[1];

    const uint n_groups = (L + GROUP_SIZE - 1u) / GROUP_SIZE;
    const uint row      = gid / n_groups;
    const uint grp      = gid % n_groups;

    if (row >= R) { return; }

    const uint base     = row * L;              // start of this row in x
    const uint g_start  = grp * GROUP_SIZE;     // first (virtual) index in group
    const uint g_end    = min(g_start + GROUP_SIZE, L);  // one past last LIVE index
    // Number of virtual (post-pad) slots this group covers. Only the final
    // group of the final row can be partially padded.
    const uint g_slots  = min(g_start + GROUP_SIZE, n_groups * GROUP_SIZE) - g_start;

    // The pad value: the MLX path appends copies of x[..., -1:] to fill the
    // last group, so every padded slot holds the row's last live element.
    const float pad_val = float(x[base + L - 1u]);

    // ---- phase 1: per-thread min/max over a strided slice of the group ----
    float t_min =  INFINITY;
    float t_max = -INFINITY;
    for (uint i = tid; i < g_slots; i += THREADS) {
        uint idx = g_start + i;
        float v  = (idx < g_end) ? float(x[base + idx]) : pad_val;
        t_min = min(t_min, v);
        t_max = max(t_max, v);
    }

    // ---- phase 2: threadgroup tree reduction ----
    threadgroup float s_min[THREADS];
    threadgroup float s_max[THREADS];
    s_min[tid] = t_min;
    s_max[tid] = t_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_min[tid] = min(s_min[tid], s_min[tid + stride]);
            s_max[tid] = max(s_max[tid], s_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float gmin  = s_min[0];
    const float gmax  = s_max[0];
    const float scale = max((gmax - gmin) / float(LEVELS), float(EPS));

    // ---- phase 3: quantize + dequantize, write live elements only ----
    for (uint i = tid; i < g_slots; i += THREADS) {
        uint idx = g_start + i;
        if (idx >= g_end) { continue; }         // padded slot: no output
        float v = float(x[base + idx]);
        // rint == round-half-to-even, matching mx.round (see header note).
        float q = rint((v - gmin) / scale);
        q = clamp(q, 0.0f, float(LEVELS));
        // Two separately-rounded ops, NOT an fma. The compiler will happily
        // contract `q * scale + gmin` into a single fused multiply-add with
        // one rounding step; MLX's elementwise path rounds the multiply and
        // the add independently, so contraction makes ~0.02% of elements
        // differ by 1 ULP. `fma` is the more accurate answer, but bit-exact
        // parity with the MLX reference is the contract here (issue #164), so
        // force the un-fused form.
        float prod = q * scale;
        out[base + idx] = T(prod + gmin);
    }

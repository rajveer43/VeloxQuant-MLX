// kivi_group_quant_channel.metal
// Extracted from veloxquant_mlx/metal/_kivi_quant.py (_KIVI_QUANT_CHANNEL_SRC) — see that
// file's docstring for the algorithm, memory-layout, and threadgroup strategy
// explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// KIVI **per-channel** group quantize->dequantize (keys): the quantization
// group runs along the TOKEN axis, one (min, max) pair per (channel, token
// group).
//
// Input is [BH, S, D] row-contiguous; element (bh, s, d) lives at
// bh*S*D + s*D + d. The quantization axis (S) is therefore the STRIDED one.
//
// Grid:        (BH * n_token_groups * D, 1, 1) — one thread per
//              (bh, token-group, channel).
// Threadgroup: (256, 1, 1) — threads never cooperate, so this is just an
//              occupancy knob.
//
// **One thread owns one whole group, and that is the entire trick.** Consecutive
// threads take consecutive channels `d`, so a warp's 32 simultaneous loads hit 32
// adjacent addresses — fully coalesced — even though each individual thread then
// walks its group with stride D. Because a thread owns its group outright, there
// is no cross-thread reduction: no threadgroup memory, no barriers, no butterfly.
//
// The alternative (transpose so the token axis becomes contiguous, then reduce
// across a warp) is what this kernel replaces. That transpose is a full-size
// materializing copy of the tensor, which costs more memory traffic than the
// arithmetic being fused — it made the kernel a net loss on the key path.
//
// D is a compile-time constant so `tid % D` / `tid / D` become shift/mask
// instead of integer division. That is safe to specialize on: D is the model's
// head_dim, fixed for the lifetime of a cache. S is NOT specialized — it grows
// every decode step, and baking it in would compile a fresh shader per token.

    uint tid = thread_position_in_grid.x;

    const uint BH = x_shape[0];
    const uint S  = x_shape[1];
    const uint NG = (S + GROUP_SIZE - 1u) / GROUP_SIZE;   // token groups

    if (tid >= BH * NG * DHEAD) { return; }

    const uint d   = tid % DHEAD;
    const uint r   = tid / DHEAD;
    const uint grp = r % NG;
    const uint bh  = r / NG;

    // Element (bh, s, d) == x[base + s * DHEAD]: walking this thread's group
    // means striding by one full row.
    const uint base = bh * S * DHEAD + d;
    const uint s0   = grp * GROUP_SIZE;
    const uint s1   = min(s0 + GROUP_SIZE, S);

    // Deliberately re-reads in the quantize pass below rather than caching the
    // group in registers. Caching was measured *slower* here (0.76x at S=2048):
    // a whole group is GROUP_SIZE floats per thread, and the occupancy that
    // costs outweighs the saved loads, which hit cache anyway. Tried and
    // rejected in the token kernel too, where it measured exactly neutral.
    float gmin =  INFINITY;
    float gmax = -INFINITY;
    for (uint s = s0; s < s1; ++s) {
        float v = float(x[base + s * DHEAD]);
        gmin = min(gmin, v);
        gmax = max(gmax, v);
    }
    // Ragged final group: the reference pads by replicating the LAST live
    // token's value for this channel. Every pad slot holds that same value, so
    // folding it in once is equivalent for min/max — no need to loop the pad.
    if (s1 < s0 + GROUP_SIZE) {
        float pad_val = float(x[base + (S - 1u) * DHEAD]);
        gmin = min(gmin, pad_val);
        gmax = max(gmax, pad_val);
    }

    const float scale = max((gmax - gmin) / float(LEVELS), float(EPS));

    for (uint s = s0; s < s1; ++s) {
        float v = float(x[base + s * DHEAD]);
        // rint == round-half-to-even, matching mx.round. metal::round would be
        // round-half-away-from-zero and would disagree on exact .5 codes.
        float q = clamp(rint((v - gmin) / scale), 0.0f, float(LEVELS));
        // Two separately-rounded ops, NOT an fma: the compiler would happily
        // contract this into a single fused multiply-add with one rounding,
        // which shifts ~0.02% of elements by 1 ULP away from the MLX reference.
        float prod = q * scale;
        out[base + s * DHEAD] = T(prod + gmin);
    }

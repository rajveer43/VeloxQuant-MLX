// crosskv_rope_recode.metal
// Extracted from veloxquant_mlx/metal/_crosskv_rope.py (_CROSSKV_ROPE_RECODE_SRC) —
// see that file's docstring for the algorithm and dispatch strategy. This file
// is read as plain text at import time via _read_kernel_source() and JIT-compiled
// by mx.fast.metal_kernel at call time; it is NOT separately compiled (see #64).
//
// Cross-model KV transfer, RoPE re-encode (paper §3.3). Converts keys rotated
// under the SOURCE model's rope_theta into keys rotated under the TARGET's,
// in a single pass.
//
// The Python path (veloxquant_mlx/transfer/rope.py) evaluates this as
// strip_rope then apply_rope: two full [N, D] rotations, each materializing
// cos/sin tables and an intermediate array. Because both rotations act on the
// same (d, d + D/2) pair with the same position p, they compose into ONE
// rotation by the per-dimension angle difference:
//
//     angle = p * (theta_t^(-d/half) - theta_s^(-d/half))
//
// so this kernel does one sincos and one 2x2 rotation per element pair, with
// no intermediate array. Note this is NOT the single-base delta rotation used
// by h2o_evict_apply.metal: there the base is shared and the angles collapse
// to (new_pos - old_pos) * inv_freq; here the BASES differ, so the difference
// must be taken per dimension, after the inverse-frequency exponentiation.
//
// Grid:        (BH * N * (D/2), 1, 1) — one thread per rotated element pair.
// Threadgroup: (256, 1, 1).
//
// Layout: keys are [BH, N, D] row-contiguous; thread g maps to
//   pair index  d  = g % half
//   token       n  = (g / half) % N
//   group       bh = g / (half * N)
// Each thread reads keys[bh,n,d] and keys[bh,n,d+half] and writes both back
// rotated. Threads never touch each other's elements, so no barriers are
// needed and the two dispatch phases of the eviction kernels are unnecessary.
//
// NeoX-style (first-half/second-half) pairing, matching MLX's default
// traditional=false convention and transfer/rope.py's _rotate_by.

    uint gid = thread_position_in_grid.x;

    uint N    = uint(keys_shape[1]);
    uint D    = uint(keys_shape[2]);
    uint half_d = D / 2u;
    uint BH   = uint(keys_shape[0]);

    uint total = BH * N * half_d;
    if (gid >= total) return;

    uint d  = gid % half_d;
    uint n  = (gid / half_d) % N;
    uint bh = gid / (half_d * N);

    // Positions are per-token and shared across bh groups: [N].
    float p = float(positions[n]);

    float src_base = rope_bases[0];
    float tgt_base = rope_bases[1];

    // exponent is identical for both bases; compute once.
    float exponent = float(d) / float(half_d);
    float inv_freq_s = 1.0f / metal::pow(src_base, exponent);
    float inv_freq_t = 1.0f / metal::pow(tgt_base, exponent);

    // Single fused angle: subtract BEFORE multiplying by p, so the two large
    // similar angles cancel in float before the trig call rather than after —
    // at p in the thousands, p*inv_freq_s and p*inv_freq_t are both large and
    // nearly equal for low d, and differencing them post-multiply loses
    // significant bits.
    float angle = p * (inv_freq_t - inv_freq_s);

    float c = metal::cos(angle);
    float s = metal::sin(angle);

    uint row = (bh * N + n) * D;
    float x1 = float(keys[row + d]);
    float x2 = float(keys[row + d + half_d]);

    out[row + d]          = static_cast<T>(x1 * c - x2 * s);
    out[row + d + half_d] = static_cast<T>(x1 * s + x2 * c);

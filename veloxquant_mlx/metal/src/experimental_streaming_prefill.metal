// experimental_streaming_prefill.metal
// Row-owned streaming causal prefill attention — see
// experimental_streaming_prefill_ARCHITECTURE.md in this directory for the
// full design rationale. Read as plain text at import time via
// _read_kernel_source() in _experimental_streaming_prefill.py and
// JIT-compiled by mx.fast.metal_kernel at call time (same pattern as
// flash_prefill.metal — see that file's header comment).
//
// This is a genuinely different computational decomposition from
// flash_prefill.metal: ownership is by QUERY ROW, not by matrix tile.
// One SIMD-group (32 lanes) owns exactly one query row for the whole
// kernel invocation; each lane owns a fixed, disjoint stride-32 slab of
// head-dims (D/32 dims/lane). There is no simdgroup_matrix usage, no
// threadgroup array staging K/V/scores/weights/output, and no
// threadgroup_barrier / simdgroup_barrier anywhere in this file.
//
// Template parameters:
//   MAX_D       — head dimension: 32, 64, or 128 (must be a multiple of 32).
//   KV_BLOCK    — KV tokens processed per loop iteration before one
//                 online-softmax update: 1, 2, 4, or 8.
//   ROWS_PER_TG — SIMD-groups (independent query rows) per threadgroup:
//                 1 for the single-row kernels, 4 for the "multirow"
//                 kernel. Purely a dispatch-granularity knob — no data is
//                 shared between SIMD-groups in the same threadgroup, so
//                 no barrier is needed even when ROWS_PER_TG > 1.
//
// Grid:        (B * H * n_qblk * 32, ROWS_PER_TG, 1) — MLX grid = threads.
// Threadgroup: (32, ROWS_PER_TG, 1).
// n_qblk = ceil(S_q / ROWS_PER_TG); threadgroup tgx owns ROWS_PER_TG
// consecutive query rows (one per SIMD-group) within one (b, h).
//
// Causal alignment: q_abs = (S_kv - S_q) + q_index; KV slot j is visible
// iff j <= q_abs. The KV loop bound is `visible = min(S_kv, q_abs + 1)`
// itself — future KV tokens are never loaded or computed, no masking.
//
// Precision: Q/K/V loaded fp16, widened to float for all arithmetic.
// Output accumulator is float, cast to half only at the final write.
// scale is pre-folded with log2(e) so the softmax inner loop uses
// fast::exp2 instead of exp (matches flash_prefill.metal / MLX's steel
// attention kernel).
//
// Note: the dims-per-lane loop index is named `dk` (not `k`) throughout —
// `k` is the MLX-generated device-buffer pointer name for the Keys input
// (from input_names=["q","k","v","scale"]), so reusing `k` as a loop
// variable would shadow it and break every `k[...]` load.

constexpr uint D             = uint(MAX_D);
constexpr uint KB            = uint(KV_BLOCK);
constexpr uint NSG           = uint(ROWS_PER_TG);
constexpr uint DIMS_PER_LANE = D / 32u;
constexpr float kLog2E       = 1.4426950408889634f;

uint tgx  = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg   = thread_position_in_threadgroup.y;

uint H     = uint(q_shape[1]);
uint S_q   = uint(q_shape[2]);
uint S_kv  = uint(k_shape[2]);

uint n_qblk = (S_q + NSG - 1u) / NSG;

// Threadgroup tgx owns one (b, h) pair and one block of NSG consecutive
// query rows: qblk = tgx % n_qblk, h = (tgx / n_qblk) % H, b = tgx / (n_qblk*H).
uint qblk  = tgx % n_qblk;
uint h_idx = (tgx / n_qblk) % H;
uint b_idx = tgx / (n_qblk * H);

uint g_row = qblk * NSG + sg;  // this SIMD-group's global query row index

uint q_base  = ((b_idx * H + h_idx) * S_q) * D;
uint kv_base = ((b_idx * H + h_idx) * S_kv) * D;

// scale pre-folded with log2(e): softmax uses exp2 instead of exp.
float sc = scale[0] * kLog2E;

// Queries align to the tail of the KV cache (fused_sdpa.metal convention,
// same as flash_prefill.metal).
int q_align = int(S_kv) - int(S_q);

// Per-lane owned output-dim accumulators (registers only — no
// threadgroup memory). o[dk] holds owned dim `lane + dk*32`.
float o[DIMS_PER_LANE];
for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) o[dk] = 0.0f;

float m = -INFINITY;
float l = 0.0f;

if (g_row < S_q) {
    int q_abs = q_align + int(g_row);
    // Pre-load this lane's owned Q dims (widened to float) once — reused
    // for every KV token's dot product.
    float q_local[DIMS_PER_LANE];
    for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
        q_local[dk] = float(q[q_base + g_row * D + (lane + dk * 32u)]);
    }

    if (q_abs >= 0) {
        uint visible = uint(metal::min(int(S_kv) - 1, q_abs)) + 1u;

        uint j0 = 0u;
        // ---- Blocked main loop: KB independent KV tokens per iteration ----
        for (; j0 + KB <= visible; j0 += KB) {
            float scores[KB];
            for (uint b = 0; b < KB; ++b) {
                uint slot = j0 + b;
                float local_dot = 0.0f;
                for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
                    float kv = float(k[kv_base + slot * D + (lane + dk * 32u)]);
                    local_dot += q_local[dk] * kv;
                }
                scores[b] = metal::simd_sum(local_dot) * sc;
            }

            // One online-softmax update over this KB-wide mini-block.
            float block_max = scores[0];
            for (uint b = 1; b < KB; ++b) block_max = metal::max(block_max, scores[b]);
            float m_new = metal::max(m, block_max);
            float alpha = metal::fast::exp2(m - m_new);

            float p[KB];
            float p_sum = 0.0f;
            for (uint b = 0; b < KB; ++b) {
                p[b] = metal::fast::exp2(scores[b] - m_new);
                p_sum += p[b];
            }

            for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
                float acc = o[dk] * alpha;
                for (uint b = 0; b < KB; ++b) {
                    uint slot = j0 + b;
                    float vv = float(v[kv_base + slot * D + (lane + dk * 32u)]);
                    acc += p[b] * vv;
                }
                o[dk] = acc;
            }

            l = l * alpha + p_sum;
            m = m_new;
        }

        // ---- Tail: remaining KV tokens (visible % KB != 0), one at a time ----
        for (uint slot = j0; slot < visible; ++slot) {
            float local_dot = 0.0f;
            for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
                float kv = float(k[kv_base + slot * D + (lane + dk * 32u)]);
                local_dot += q_local[dk] * kv;
            }
            float score = metal::simd_sum(local_dot) * sc;

            float m_new = metal::max(m, score);
            float alpha = metal::fast::exp2(m - m_new);
            float p = metal::fast::exp2(score - m_new);

            for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
                float vv = float(v[kv_base + slot * D + (lane + dk * 32u)]);
                o[dk] = o[dk] * alpha + p * vv;
            }

            l = l * alpha + p;
            m = m_new;
        }
    }

    // ---- Write out (l == 0 only for the causally-unreachable case,
    // guarded for safety/symmetry with flash_prefill.metal). ----
    for (uint dk = 0; dk < DIMS_PER_LANE; ++dk) {
        out[q_base + g_row * D + (lane + dk * 32u)] =
            half(l > 0.0f ? o[dk] / l : 0.0f);
    }
}

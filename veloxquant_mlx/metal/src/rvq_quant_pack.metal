// rvq_quant_pack.metal
// Extracted from veloxquant_mlx/metal/_rvq_quant_pack.py (_RVQ_QUANT_PACK_SRC) — see that
// file's docstring for the algorithm, memory-layout, and threadgroup strategy
// explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Fused TurboQuantRVQ stage-1 + stage-2 quantize, packed directly into uint32
// words — replaces {codebook1.quantize, codebook1.dequantize, codebook2.quantize,
// _pack_indices, _pack_indices} (five MLX dispatches, two full-size uint8 index
// intermediates) with one dispatch and zero intermediate index buffers (#251).
//
// Grid:        (N * D, 1, 1) — one thread per (vector, coordinate).
// Threadgroup: (D, 1, 1)     — one threadgroup per rotated vector `y`.
//
// Per thread:
//   1. idx1 = count(y > boundaries1[k])   for k in [0, K1-1)   -- stage-1 code
//      (boundary-count quantize: identical to ScalarCodebook.quantize's
//      broadcast-compare-and-sum, so output matches the MLX path bit-for-bit
//      on ties as well, since both reduce the same `>` comparisons in the
//      same left-to-right order over sorted boundaries.)
//   2. y_hat1 = centroids1[idx1]
//   3. r1 = y - y_hat1
//   4. idx2 = count(r1 > boundaries2[k])  for k in [0, K2-1)   -- stage-2 code
//   5. Each lane's (idx1, idx2) is a BITS-bit code. Lanes cooperatively pack
//      ELEMS_PER_WORD = 32 / BITS consecutive lanes' codes into one uint32,
//      LSB-first — the same layout as _pack_indices/mx.quantize. Packing
//      reads threadgroup memory (each lane's own idx1/idx2, staged there),
//      so no cross-lane shuffle is needed: lane `w * ELEMS_PER_WORD` writes
//      word `w` by looping its ELEMS_PER_WORD group.
//
// D is not required to be a multiple of ELEMS_PER_WORD; out-of-range lanes
// within the final word contribute zero bits (matching _pack_indices's
// zero-padding of the trailing partial word).

    threadgroup uint8_t idx1_buf[MAX_D];
    threadgroup uint8_t idx2_buf[MAX_D];

    uint n    = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint D    = uint(MAX_D);

    constexpr uint K1 = (1u << BITS) - 1u;  // stage-1 boundary count
    constexpr uint K2 = (1u << BITS) - 1u;  // stage-2 boundary count

    float y = float(rotated[n * D + lane]);

    uint idx1 = 0u;
    for (uint k = 0; k < K1; ++k) {
        idx1 += uint(y > float(boundaries1[k]));
    }
    float y_hat1 = float(centroids1[idx1]);
    float r1 = y - y_hat1;

    uint idx2 = 0u;
    for (uint k = 0; k < K2; ++k) {
        idx2 += uint(r1 > float(boundaries2[k]));
    }

    idx1_buf[lane] = uint8_t(idx1);
    idx2_buf[lane] = uint8_t(idx2);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint ELEMS_PER_WORD = 32u / BITS;
    constexpr uint MASK           = (1u << BITS) - 1u;

    // Each word's first lane packs its ELEMS_PER_WORD-wide group. Surplus
    // lanes (D not a multiple of ELEMS_PER_WORD, or lane belongs to a
    // partial trailing group) are simply idle here.
    if (lane % ELEMS_PER_WORD == 0u) {
        uint word_idx = lane / ELEMS_PER_WORD;
        uint w1 = 0u;
        uint w2 = 0u;
        for (uint j = 0; j < ELEMS_PER_WORD; ++j) {
            uint d = lane + j;
            if (d < D) {
                w1 |= (uint(idx1_buf[d]) & MASK) << (j * BITS);
                w2 |= (uint(idx2_buf[d]) & MASK) << (j * BITS);
            }
        }
        uint n_words = (D + ELEMS_PER_WORD - 1u) / ELEMS_PER_WORD;
        packed1[n * n_words + word_idx] = w1;
        packed2[n * n_words + word_idx] = w2;
    }

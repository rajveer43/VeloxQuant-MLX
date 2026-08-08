// rabitq_hamming_score.metal
// Extracted from veloxquant_mlx/metal/_rabitq.py (_HAMMING_SCORE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint i = thread_position_in_grid.x;
    if (i >= uint(N)) return;

    // XOR + popcount over N_BYTES packed bytes
    uint ham = 0u;
    uint base = i * uint(N_BYTES);
    for (uint b = 0; b < uint(N_BYTES); b++) {
        uint8_t xr = qbits[b] ^ bits[base + b];
        // popcount via bit manipulation (portable across Metal versions)
        uint v = uint(xr);
        v = v - ((v >> 1u) & 0x55u);
        v = (v & 0x33u) + ((v >> 2u) & 0x33u);
        v = (v + (v >> 4u)) & 0x0Fu;
        ham += v;
    }

    scores[i] = float(ham) * scale[0] + Cx[i];

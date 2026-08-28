"""Parity + benchmark tests for the fused RVQ quantize+pack kernel (#251).

``TurboQuantRVQKVCache.update_and_fetch`` previously ran stage-1 quantize,
dequantize, residual, stage-2 quantize as MLX ops, then bit-packed each
stage's uint8 index array via a *separate* ``_pack_indices`` MLX dispatch --
five MLX kernel launches and two full-size ``(N, D)`` uint8 intermediates
per flush. ``rvq_quant_pack`` fuses all of that (post-rotation) into one
Metal dispatch with no intermediate index buffers.

Must reproduce ``_pack_indices(ScalarCodebook.quantize(...), bits)``
**bit-for-bit**, not merely within tolerance -- see
:meth:`TurboQuantRVQ.encode_pack`'s docstring for why (same bit-exactness
bar as the KIVI fused kernel, #164).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.turboquant_rvq_cache import _pack_indices
from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import rvq_quant_pack
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


def _reference_pack(quantizer: TurboQuantRVQ, x: mx.array, bits: int):
    ev = quantizer.encode(x)
    p1 = _pack_indices(ev.indices, bits)
    p2 = _pack_indices(ev.signs.astype(mx.uint8), bits)
    return p1, p2


# ---------------------------------------------------------------------------
# Parity vs the MLX reference path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [1, 5, 33])
def test_rvq_quant_pack_bit_exact(D, bits, N):
    q = TurboQuantRVQ(d=D, b=bits, seed=D + bits + N, use_hadamard=True)
    rng = np.random.default_rng(D * 100 + bits * 10 + N)
    x = mx.array(rng.standard_normal((N, D)).astype(np.float16))

    p1_ref, p2_ref = _reference_pack(q, x, bits)
    p1_got, p2_got = q.encode_pack(x)
    mx.eval(p1_ref, p2_ref, p1_got, p2_got)

    assert p1_got.shape == p1_ref.shape
    assert p2_got.shape == p2_ref.shape
    assert p1_got.dtype == mx.uint32
    np.testing.assert_array_equal(np.array(p1_got), np.array(p1_ref))
    np.testing.assert_array_equal(np.array(p2_got), np.array(p2_ref))


def test_rvq_quant_pack_direct_kernel_entry_point():
    """Exercise rvq_quant_pack() directly (not via TurboQuantRVQ.encode_pack)."""
    D, bits, N = 128, 2, 17
    q = TurboQuantRVQ(d=D, b=bits, seed=0, use_hadamard=True)
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((N, D)).astype(np.float16))

    p1_ref, p2_ref = _reference_pack(q, x, bits)

    y = q._rotation.apply(x)
    p1_got, p2_got = rvq_quant_pack(
        y,
        q._codebook1.centroids_mx(),
        q._codebook1.boundaries_mx(),
        q._codebook2.boundaries_mx(),
        bits,
    )
    mx.eval(p1_ref, p2_ref, p1_got, p2_got)
    np.testing.assert_array_equal(np.array(p1_got), np.array(p1_ref))
    np.testing.assert_array_equal(np.array(p2_got), np.array(p2_ref))


# ---------------------------------------------------------------------------
# End-to-end: cache update_and_fetch takes the fused path and dequantizes
# correctly (round-trip through the real EncodedVector-based decode).
# ---------------------------------------------------------------------------


def test_rvq_cache_fused_path_matches_mlx_path():
    from veloxquant_mlx import KVCacheConfig, KVCacheFactory

    cfg = KVCacheConfig(method="turboquant_rvq", head_dim=128, bit_width_inlier=2, seed=0)
    c_fused = KVCacheFactory.create(cfg)
    c_mlx = KVCacheFactory.create(cfg)
    c_mlx._use_metal_pack = False
    assert c_fused._use_metal_pack is True

    rng = np.random.default_rng(5)
    keys = mx.array(rng.standard_normal((1, 4, 40, 128)).astype(np.float16))
    vals = mx.array(rng.standard_normal((1, 4, 40, 128)).astype(np.float16))

    k_fused, v_fused = c_fused.update_and_fetch(keys, vals)
    k_mlx, v_mlx = c_mlx.update_and_fetch(keys, vals)
    mx.eval(k_fused, v_fused, k_mlx, v_mlx)

    np.testing.assert_array_equal(np.array(k_fused), np.array(k_mlx))
    np.testing.assert_array_equal(np.array(v_fused), np.array(v_mlx))
    assert c_fused.compressed_key_bytes == c_mlx.compressed_key_bytes


def test_rvq_quant_pack_rejects_bad_shapes():
    D, bits = 128, 2
    q = TurboQuantRVQ(d=D, b=bits, seed=0, use_hadamard=True)
    y = mx.zeros((4, D), dtype=mx.float16)

    with pytest.raises(ValueError, match="must be 2D"):
        rvq_quant_pack(
            y[None],
            q._codebook1.centroids_mx(),
            q._codebook1.boundaries_mx(),
            q._codebook2.boundaries_mx(),
            bits,
        )
    with pytest.raises(ValueError, match="power of two"):
        bad_y = mx.zeros((4, 24), dtype=mx.float16)
        rvq_quant_pack(
            bad_y,
            mx.zeros(4, dtype=mx.float32),
            mx.zeros(3, dtype=mx.float32),
            mx.zeros(3, dtype=mx.float32),
            bits,
        )
    with pytest.raises(ValueError, match="bits must be"):
        rvq_quant_pack(
            y,
            q._codebook1.centroids_mx(),
            q._codebook1.boundaries_mx(),
            q._codebook2.boundaries_mx(),
            5,
        )
    with pytest.raises(ValueError, match="centroids"):
        rvq_quant_pack(
            y,
            mx.zeros(3, dtype=mx.float32),
            q._codebook1.boundaries_mx(),
            q._codebook2.boundaries_mx(),
            bits,
        )


# ---------------------------------------------------------------------------
# Benchmark: fused kernel latency vs the multi-stage MLX pipeline.
# ---------------------------------------------------------------------------


def test_rvq_quant_pack_benchmark(capsys):
    """Latency, effective bandwidth, and speedup vs the multi-stage MLX path.

    Follows the interleaved-sampling methodology of
    ``test_kivi_quant.py::test_kivi_quant_benchmark``: a rotating pool of
    distinct inputs (so MLX cannot reuse work across calls) and interleaved
    A/B sampling (so thermal drift lands evenly on both paths), reporting
    medians.
    """
    D, bits = 128, 2
    q = TurboQuantRVQ(d=D, b=bits, seed=0, use_hadamard=True)
    rng = np.random.default_rng(0)
    rows = []

    for N in (64, 1024, 8192):
        pool = [mx.array(rng.standard_normal((N, D)).astype(np.float16)) for _ in range(10)]
        mx.eval(*pool)

        def _mlx_path(i, _pool=pool):
            x = _pool[i % len(_pool)]
            return _reference_pack(q, x, bits)

        def _fused(i, _pool=pool):
            return q.encode_pack(_pool[i % len(_pool)])

        for i in range(20):  # warm + compile, outside all timing
            mx.eval(*_mlx_path(i))
            mx.eval(*_fused(i))
        mx.synchronize()

        def _sample(fn, off, iters=30):
            mx.synchronize()
            t0 = time.perf_counter()
            for i in range(iters):
                mx.eval(*fn(i + off))
            mx.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

        s_mlx, s_fused = [], []
        for rep in range(7):
            s_mlx.append(_sample(_mlx_path, rep * 5))
            s_fused.append(_sample(_fused, rep * 5))

        ms_mlx = float(np.median(s_mlx))
        ms_fused = float(np.median(s_fused))
        # Effective bandwidth: fp32 rotated input read once, two uint32
        # packed streams written; the MLX path additionally round-trips two
        # full uint8 (N, D) index buffers through global memory.
        n_words = -(-D // (32 // bits))
        fused_bytes = N * D * 4 + 2 * N * n_words * 4
        mlx_bytes = fused_bytes + 2 * N * D  # + two uint8 index intermediates
        gbps_fused = fused_bytes / (ms_fused * 1e-3) / 1e9
        gbps_mlx = mlx_bytes / (ms_mlx * 1e-3) / 1e9
        rows.append((N, ms_mlx, ms_fused, gbps_mlx, gbps_fused))

    with capsys.disabled():
        print(f"\n# RVQ quantize+pack  |  D={D} bits={bits}  |  MLX {mx.__version__}")
        print("| N | MLX ms | Fused ms | speedup | MLX GB/s | Fused GB/s |")
        print("|---|--------|----------|---------|----------|------------|")
        for N, ms_mlx, ms_fused, gbps_mlx, gbps_fused in rows:
            print(
                f"| {N} | {ms_mlx:.4f} | {ms_fused:.4f} | {ms_mlx / ms_fused:.2f}x "
                f"| {gbps_mlx:.2f} | {gbps_fused:.2f} |"
            )
        print(
            "\nMedians of interleaved samples over a rotating input pool (avoids\n"
            "flattering either path with repeated-input reuse). GB/s is nominal\n"
            "traffic (input read once + packed streams written), not a hardware\n"
            "counter; the MLX column additionally charges the two full-size\n"
            "uint8 index intermediates the fused kernel never materializes."
        )

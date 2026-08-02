"""Correctness + benchmark tests for TurboQuant Metal kernels.

Each test:
  1. Builds a reference result in pure MLX / numpy.
  2. Runs the Metal kernel.
  3. Asserts shape, dtype, and numerical parity.
  4. Benchmarks over 100 iterations and prints throughput.

Tests are skipped automatically on builds where Metal is unavailable.
"""

from __future__ import annotations

import math
import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import (
    turboquant_bit_pack,
    turboquant_bit_unpack,
    turboquant_scalar_quantize,
    turboquant_scalar_dequantize,
    turboquant_hadamard_quantize,
    qjl_encode,
    qjl_inner_product,
    turboquant_fused_rvq_decode_attend,
)

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bench(fn, n_warmup: int = 5, n_iter: int = 100) -> float:
    """Return mean wall-clock time per iteration in milliseconds."""
    for _ in range(n_warmup):
        result = fn()
        mx.eval(result)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        result = fn()
        mx.eval(result)
    return (time.perf_counter() - t0) / n_iter * 1_000


def _ref_scalar_quantize(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Reference numpy nearest-centroid quantization."""
    diffs = x[..., None] - centroids[None, :]  # [..., n_cents]
    return np.argmin(diffs**2, axis=-1).astype(np.uint8)


def _ref_hadamard(x: np.ndarray, diag: np.ndarray) -> np.ndarray:
    """Reference Walsh-Hadamard transform (Cooley-Tukey, O(d log d))."""
    y = x * diag
    d = y.shape[-1]
    stride = 1
    while stride < d:
        for i in range(0, d, stride * 2):
            for j in range(stride):
                a = y[..., i + j].copy()
                b = y[..., i + j + stride].copy()
                y[..., i + j] = a + b
                y[..., i + j + stride] = a - b
        stride <<= 1
    return y / math.sqrt(d)


# ===========================================================================
# KERNEL C — bit pack / unpack
# ===========================================================================


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_bit_pack_roundtrip(b: int, N: int):
    """pack → unpack must recover the original indices exactly."""
    rng = np.random.default_rng(42)
    n_cents = 1 << b
    indices_np = rng.integers(0, n_cents, size=N, dtype=np.uint8)
    indices_mx = mx.array(indices_np)

    packed = turboquant_bit_pack(indices_mx, b)
    mx.eval(packed)

    assert packed.shape == (N * b // 8,), f"packed shape wrong: {packed.shape}"
    assert packed.dtype == mx.uint8

    unpacked = turboquant_bit_unpack(packed, N, b)
    mx.eval(unpacked)

    assert unpacked.shape == (N,)
    assert unpacked.dtype == mx.uint8
    np.testing.assert_array_equal(
        np.array(unpacked).astype(np.uint8),
        indices_np,
        err_msg=f"bit_pack roundtrip failed for b={b}, N={N}",
    )


@pytest.mark.parametrize("b", [1, 2, 4])
def test_bit_pack_memory_ratio(b: int):
    """Packed size must be exactly N*b/8 bytes."""
    N = 1024
    indices_mx = mx.zeros((N,), dtype=mx.uint8)
    packed = turboquant_bit_pack(indices_mx, b)
    mx.eval(packed)
    assert packed.size == N * b // 8


@pytest.mark.parametrize("b", [1, 2, 4])
def test_bit_pack_bench(b: int, capsys):
    N = 4096
    indices_mx = mx.zeros((N,), dtype=mx.uint8)
    ms = _bench(lambda: turboquant_bit_pack(indices_mx, b))
    with capsys.disabled():
        print(f"\n[bench] bit_pack  b={b} N={N}: {ms:.3f} ms/iter")


# ===========================================================================
# KERNEL A — scalar quantize
# ===========================================================================


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_scalar_quantize_correctness(b: int):
    """Metal kernel must match numpy argmin nearest-centroid."""
    rng = np.random.default_rng(7)
    B, d = 8, 128
    n_cents = 1 << b
    # Centroids sorted so Lloyd-Max boundaries are midpoints
    cents_np = np.linspace(-2.0, 2.0, n_cents, dtype=np.float32)
    x_np = rng.standard_normal((B, d)).astype(np.float32)

    ref = _ref_scalar_quantize(x_np, cents_np)  # [B, d] uint8
    x_mx = mx.array(x_np.astype(np.float16))
    c_mx = mx.array(cents_np)

    out = turboquant_scalar_quantize(x_mx, c_mx, b)
    mx.eval(out)

    assert out.shape == (B, d)
    assert out.dtype == mx.uint8
    np.testing.assert_array_equal(
        np.array(out),
        ref,
        err_msg=f"scalar_quantize mismatch at b={b}",
    )


def test_scalar_quantize_bench(capsys):
    B, d, b = 64, 128, 4
    cents = mx.linspace(-2.0, 2.0, 1 << b)
    x = mx.random.normal((B, d)).astype(mx.float16)
    ms = _bench(lambda: turboquant_scalar_quantize(x, cents, b))
    with capsys.disabled():
        print(f"\n[bench] scalar_quantize  B={B} d={d} b={b}: {ms:.3f} ms/iter")


# ===========================================================================
# KERNEL B — scalar dequantize
# ===========================================================================


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_scalar_dequantize_correctness(b: int):
    """Dequantize must reproduce centroids[indices] in fp16."""
    rng = np.random.default_rng(13)
    B, d = 8, 128
    n_cents = 1 << b
    cents_np = np.linspace(-2.0, 2.0, n_cents, dtype=np.float32)
    idx_np = rng.integers(0, n_cents, size=(B, d), dtype=np.uint8)

    ref = cents_np[idx_np].astype(np.float16)  # [B, d] fp16

    idx_mx = mx.array(idx_np)
    c_mx = mx.array(cents_np)
    out = turboquant_scalar_dequantize(idx_mx, c_mx)
    mx.eval(out)

    assert out.shape == (B, d)
    assert out.dtype == mx.float16
    np.testing.assert_allclose(
        np.array(out, dtype=np.float32),
        ref.astype(np.float32),
        atol=1e-3,
        err_msg=f"scalar_dequantize mismatch at b={b}",
    )


def test_scalar_dequantize_bench(capsys):
    B, d, b = 64, 128, 4
    n_cents = 1 << b
    idx = mx.array(np.random.randint(0, n_cents, (B, d), dtype=np.uint8))
    cents = mx.linspace(-2.0, 2.0, n_cents)
    ms = _bench(lambda: turboquant_scalar_dequantize(idx, cents))
    with capsys.disabled():
        print(f"\n[bench] scalar_dequantize  B={B} d={d} b={b}: {ms:.3f} ms/iter")


def test_quantize_dequantize_roundtrip():
    """quantize → dequantize → SNR must be plausible for 4-bit Gaussian."""
    rng = np.random.default_rng(99)
    B, d, b = 32, 256, 4
    n_cents = 1 << b
    cents_np = np.linspace(-3.0, 3.0, n_cents, dtype=np.float32)
    x_np = rng.standard_normal((B, d)).astype(np.float32)

    x_mx = mx.array(x_np.astype(np.float16))
    c_mx = mx.array(cents_np)

    idx = turboquant_scalar_quantize(x_mx, c_mx, b)
    x_hat = turboquant_scalar_dequantize(idx, c_mx)
    mx.eval(x_hat)

    x_hat_np = np.array(x_hat, dtype=np.float32)
    mse = np.mean((x_np - x_hat_np) ** 2)
    signal = np.mean(x_np**2)
    snr_db = 10 * math.log10(signal / mse)
    # 4-bit quantization with uniform centroids should give > 15 dB SNR
    # (Lloyd-Max centroids give ~24 dB; linspace is a lower bar but still validates correctness)
    assert snr_db > 15.0, f"Quantize-dequantize SNR too low: {snr_db:.1f} dB"


# ===========================================================================
# KERNEL G — fused Hadamard + quantize
# ===========================================================================


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("b", [2, 4])
def test_hadamard_quantize_matches_twostep(D: int, b: int):
    """Fused kernel must match WHT-then-quantize in two separate steps."""
    rng = np.random.default_rng(21)
    B = 16
    n_cents = 1 << b
    cents_np = np.linspace(-3.0, 3.0, n_cents, dtype=np.float32)

    x_np = rng.standard_normal((B, D)).astype(np.float32)
    diag_np = rng.choice([-1, 1], size=D).astype(np.float32)

    # Reference: apply WHT then quantize
    y_np = _ref_hadamard(x_np.copy(), diag_np)
    ref_idx = _ref_scalar_quantize(y_np, cents_np)

    x_mx = mx.array(x_np.astype(np.float16))
    diag_mx = mx.array(diag_np.astype(np.float32))  # will be cast to float32 in kernel
    c_mx = mx.array(cents_np)

    out = turboquant_hadamard_quantize(x_mx, diag_mx, c_mx, b)
    mx.eval(out)

    assert out.shape == (B, D)
    assert out.dtype == mx.uint8
    # Expect bit-exact match; fp16 rounding may occasionally differ by 1 index
    # on exact boundary — allow at most 1% mismatch.
    mismatch_rate = np.mean(np.array(out) != ref_idx)
    assert mismatch_rate < 0.01, (
        f"hadamard_quantize mismatch rate {mismatch_rate:.3%} exceeds 1% (D={D}, b={b})"
    )


def test_hadamard_quantize_bench(capsys):
    B, D, b = 64, 128, 4
    x = mx.random.normal((B, D)).astype(mx.float16)
    diag = mx.array(np.random.choice([-1.0, 1.0], size=D).astype(np.float32))
    cents = mx.linspace(-3.0, 3.0, 1 << b)
    ms = _bench(lambda: turboquant_hadamard_quantize(x, diag, cents, b))
    with capsys.disabled():
        print(f"\n[bench] hadamard_quantize  B={B} D={D} b={b}: {ms:.3f} ms/iter")


# ===========================================================================
# KERNEL D — qjl_encode
# ===========================================================================


def test_qjl_encode_sign_correctness():
    """Metal sign bits must match numpy sign(S @ x)."""
    rng = np.random.default_rng(55)
    B, d, m = 8, 64, 64
    x_np = rng.standard_normal((B, d)).astype(np.float16)
    S_np = rng.standard_normal((m, d)).astype(np.float16) / math.sqrt(d)

    proj = x_np.astype(np.float32) @ S_np.T.astype(np.float32)  # [B, m]
    ref_signs = (proj >= 0).astype(np.uint8)  # [B, m] {0,1}

    x_mx = mx.array(x_np)
    S_mx = mx.array(S_np)

    packed_signs, norms = qjl_encode(x_mx, S_mx)
    mx.eval(packed_signs, norms)

    assert packed_signs.shape == (B, m // 8)
    assert packed_signs.dtype == mx.uint8
    assert norms.shape == (B,)
    assert norms.dtype == mx.float16

    # Unpack bits and compare against reference
    ps_np = np.array(packed_signs)  # [B, m//8]
    recovered = np.unpackbits(ps_np, axis=-1, bitorder="little")[:, :m]  # [B, m]

    mismatch = np.mean(recovered != ref_signs)
    assert mismatch < 0.02, f"qjl_encode sign mismatch: {mismatch:.3%}"


def test_qjl_encode_norm_correctness():
    """Stored norms must match ‖x‖₂ within fp16 tolerance."""
    rng = np.random.default_rng(66)
    B, d, m = 16, 128, 64
    x_np = rng.standard_normal((B, d)).astype(np.float16)
    S_np = rng.standard_normal((m, d)).astype(np.float16) / math.sqrt(d)

    ref_norms = np.linalg.norm(x_np.astype(np.float32), axis=-1).astype(np.float16)

    x_mx = mx.array(x_np)
    S_mx = mx.array(S_np)
    _, norms = qjl_encode(x_mx, S_mx)
    mx.eval(norms)

    np.testing.assert_allclose(
        np.array(norms, dtype=np.float32),
        ref_norms.astype(np.float32),
        rtol=1e-2,
        err_msg="qjl_encode norm mismatch",
    )


def test_qjl_encode_bench(capsys):
    B, d, m = 32, 128, 128
    x = mx.random.normal((B, d)).astype(mx.float16)
    S = (mx.random.normal((m, d)) / math.sqrt(d)).astype(mx.float16)
    ms = _bench(lambda: qjl_encode(x, S))
    with capsys.disabled():
        print(f"\n[bench] qjl_encode  B={B} d={d} m={m}: {ms:.3f} ms/iter")


# ===========================================================================
# KERNEL E — qjl_inner_product
# ===========================================================================


def test_qjl_inner_product_correctness():
    """Metal QJL IP scores must correlate with true inner products (unbiased estimator).

    Uses qjl_encode to build the packed representation so the bit-packing
    convention is guaranteed to match qjl_inner_product.
    """
    rng = np.random.default_rng(77)
    d, m = 64, 64
    S_kv, H = 64, 4

    # Build S (shared across encode + project)
    S_np = rng.standard_normal((m, d)).astype(np.float16) / math.sqrt(d)
    S_mx = mx.array(S_np)

    # Queries: one per head  [H, d]
    q_np = rng.standard_normal((H, d)).astype(np.float16)

    # Keys: one per (kv-slot, head)  [S_kv * H, d]
    k_np = rng.standard_normal((S_kv * H, d)).astype(np.float16)

    # Reference true inner products: [H, S_kv]
    q_f = q_np.astype(np.float32)  # [H, d]
    k_f = k_np.astype(np.float32)  # [S_kv*H, d]
    true_ip = np.zeros((H, S_kv), dtype=np.float32)
    for h in range(H):
        for s in range(S_kv):
            true_ip[h, s] = np.dot(q_f[h], k_f[s * H + h])

    # Encode keys: qjl_encode produces packed_signs [S_kv*H, m//8] and norms [S_kv*H]
    k_mx = mx.array(k_np)
    packed_flat, norms_flat = qjl_encode(k_mx, S_mx)  # [S_kv*H, m//8], [S_kv*H]
    mx.eval(packed_flat, norms_flat)

    # Reshape to [S_kv, H, m//8] and [S_kv, H] for the IP kernel
    packed_mx = packed_flat.reshape(S_kv, H, m // 8)
    norms_mx = norms_flat.reshape(S_kv, H)

    # Project queries: q_proj [H, m]
    q_proj_np = (q_f @ S_np.T.astype(np.float32)).astype(np.float16)
    q_proj_mx = mx.array(q_proj_np)

    scores = qjl_inner_product(q_proj_mx, packed_mx, norms_mx)
    mx.eval(scores)

    assert scores.shape == (H, S_kv)
    assert scores.dtype == mx.float16

    # QJL is an unbiased estimator. With m=d=64, theoretical correlation ≈ 0.67.
    # The test validates the kernel is numerically correct (i.e. matches the estimator),
    # not that the estimator is tight — use a threshold at the theoretical floor.
    scores_np = np.array(scores, dtype=np.float32)
    corr = np.corrcoef(true_ip.ravel(), scores_np.ravel())[0, 1]
    assert corr > 0.60, f"qjl_inner_product correlation with truth too low: {corr:.3f}"


def test_qjl_inner_product_bench(capsys):
    H, m, S_kv = 8, 128, 512
    q_proj = mx.random.normal((H, m)).astype(mx.float16)
    packed_signs = mx.zeros((S_kv, H, m // 8), dtype=mx.uint8)
    norms = mx.ones((S_kv, H), dtype=mx.float16)
    ms = _bench(lambda: qjl_inner_product(q_proj, packed_signs, norms))
    with capsys.disabled():
        print(f"\n[bench] qjl_inner_product  H={H} m={m} S_kv={S_kv}: {ms:.3f} ms/iter")


# ===========================================================================
# KERNEL F — fused RVQ decode + attend
# ===========================================================================


def _make_rvq_cache(B, H, S_kv, D, b1, b2, bv, sub_dim_v, rng):
    """Return (q, ki1, ki2, c1, c2, vi, vcb) for fused attend test."""
    n_c1 = 1 << b1
    n_c2 = 1 << b2
    n_cv = 1 << bv
    n_sub_v = D // sub_dim_v

    q = rng.standard_normal((B, H, 1, D)).astype(np.float16)
    ki1 = rng.integers(0, n_c1, (B, H, S_kv, D), dtype=np.uint8)
    ki2 = rng.integers(0, n_c2, (B, H, S_kv, D), dtype=np.uint8)
    c1 = np.linspace(-2.0, 2.0, n_c1, dtype=np.float32)
    c2 = np.linspace(-1.0, 1.0, n_c2, dtype=np.float32)
    vi = rng.integers(0, n_cv, (B, H, S_kv, n_sub_v), dtype=np.uint8)
    vcb = rng.standard_normal((n_cv, sub_dim_v)).astype(np.float16)
    return q, ki1, ki2, c1, c2, vi, vcb


def _ref_rvq_attend(q_np, ki1_np, ki2_np, c1_np, c2_np, vi_np, vcb_np):
    """Reference: decode keys, decode values, compute softmax attention."""
    B, H, S_q, D = q_np.shape
    S_kv = ki1_np.shape[2]
    sub_dim_v = vcb_np.shape[1]
    inv_sqrt = 1.0 / math.sqrt(D)
    out = np.zeros((B, H, S_q, D), dtype=np.float32)

    for b in range(B):
        for h in range(H):
            for sq in range(S_q):
                q_vec = q_np[b, h, sq, :].astype(np.float32)
                scores = np.zeros(S_kv, dtype=np.float32)
                for sk in range(S_kv):
                    k_vec = (c1_np[ki1_np[b, h, sk, :]] + c2_np[ki2_np[b, h, sk, :]]).astype(
                        np.float32
                    )
                    scores[sk] = np.dot(q_vec, k_vec) * inv_sqrt
                scores -= scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                for sk in range(S_kv):
                    n_sub_v = vi_np.shape[3]
                    v_vec = np.zeros(D, dtype=np.float32)
                    for s in range(n_sub_v):
                        cb_row = vcb_np[vi_np[b, h, sk, s]].astype(np.float32)
                        v_vec[s * sub_dim_v : (s + 1) * sub_dim_v] = cb_row
                    out[b, h, sq, :] += weights[sk] * v_vec
    return out.astype(np.float16)


def test_fused_rvq_attend_correctness():
    """Fused kernel output must match two-step reference within fp16 tolerance."""
    rng = np.random.default_rng(42)
    B, H, S_kv, D = 1, 2, 16, 64
    b1, b2, bv, sub_dim_v = 2, 2, 2, 8

    q_np, ki1_np, ki2_np, c1_np, c2_np, vi_np, vcb_np = _make_rvq_cache(
        B, H, S_kv, D, b1, b2, bv, sub_dim_v, rng
    )
    ref = _ref_rvq_attend(q_np, ki1_np, ki2_np, c1_np, c2_np, vi_np, vcb_np)

    out = turboquant_fused_rvq_decode_attend(
        mx.array(q_np),
        mx.array(ki1_np),
        mx.array(ki2_np),
        mx.array(c1_np),
        mx.array(c2_np),
        mx.array(vi_np),
        mx.array(vcb_np),
        b1=b1,
        b2=b2,
        bv=bv,
    )
    mx.eval(out)

    assert out.shape == (B, H, 1, D)
    assert out.dtype == mx.float16

    np.testing.assert_allclose(
        np.array(out, dtype=np.float32),
        ref.astype(np.float32),
        atol=1e-2,
        rtol=1e-2,
        err_msg="fused_rvq_decode_attend output mismatch",
    )


def test_fused_rvq_attend_bench(capsys):
    rng = np.random.default_rng(0)
    B, H, S_kv, D = 1, 8, 512, 128
    b1, b2, bv, sub_dim_v = 2, 2, 2, 8

    q_np, ki1_np, ki2_np, c1_np, c2_np, vi_np, vcb_np = _make_rvq_cache(
        B, H, S_kv, D, b1, b2, bv, sub_dim_v, rng
    )

    q_mx = mx.array(q_np)
    ki1_mx = mx.array(ki1_np)
    ki2_mx = mx.array(ki2_np)
    c1_mx = mx.array(c1_np)
    c2_mx = mx.array(c2_np)
    vi_mx = mx.array(vi_np)
    vcb_mx = mx.array(vcb_np)

    ms = _bench(
        lambda: turboquant_fused_rvq_decode_attend(
            q_mx,
            ki1_mx,
            ki2_mx,
            c1_mx,
            c2_mx,
            vi_mx,
            vcb_mx,
            b1=b1,
            b2=b2,
            bv=bv,
        )
    )
    with capsys.disabled():
        print(f"\n[bench] fused_rvq_decode_attend  B={B} H={H} S_kv={S_kv} D={D}: {ms:.3f} ms/iter")

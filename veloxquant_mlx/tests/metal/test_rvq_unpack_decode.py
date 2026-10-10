"""Parity + integration tests for the fused packed-RVQ key decode kernel.

``TurboQuantRVQKVCache._dequantize_range`` re-decodes the whole packed key
history on every ``update_and_fetch`` with ~10 MLX dispatches (unpack x2,
codebook gather x2, add, inverse Hadamard, fp32 norm rescale + saturate).
``rvq_unpack_decode`` fuses all of it into one dispatch.

Exactness policy: unpack + codebook sum is bit-exact (``D == 1`` test below
isolates it, since the Hadamard of a single element is the identity). The
inverse Hadamard butterfly sums in a different fp32 order than
``mx.hadamard_transform``, so the rotated output is compared with a tolerance
of 2 fp16 ulps of ``max|reference|`` (``2**-9`` relative). That is two orders
of magnitude below the quantization error itself (>= 5% of ``max|x|`` at the
bit-widths tested; see ``test_error_far_below_quantization_error``).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import veloxquant_mlx.metal.kernels as kernels_mod
from veloxquant_mlx.cache.turboquant_rvq_cache import (
    TurboQuantRVQKVCache,
    _rescale_fp16,
    _unpack_indices,
)
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.metal import _rvq_unpack_decode, metal_available
from veloxquant_mlx.metal.kernels import rvq_unpack_decode
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        not metal_available(),
        reason="Metal compute kernels not available on this build of mlx.",
    ),
]

# 2 fp16 ulps of the largest reference magnitude (fp16 has 10 mantissa bits).
REL_TOL = 2.0**-9


def _reference_decode(q: TurboQuantRVQ, p1, p2, norms, bits: int, d: int):
    """The exact MLX path from TurboQuantRVQKVCache._dequantize_range."""
    idx1 = _unpack_indices(p1, bits, d)
    idx2 = _unpack_indices(p2, bits, d)
    ev = EncodedVector(
        quantizer_type="turboquant_rvq",
        batch_size=idx1.shape[0],
        dim=d,
        indices=idx1,
        signs=idx2.astype(mx.int8),
    )
    return _rescale_fp16(q.decode(ev), norms.reshape(-1, 1))


def _unit_rows(n: int, d: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return mx.array(x.astype(np.float16))


def _norms(n: int, seed: int = 1, hi: float = 50.0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.uniform(0.1, hi, (n, 1)).astype(np.float32)).astype(mx.bfloat16)


def _assert_close(got: mx.array, ref: mx.array) -> None:
    mx.eval(got, ref)
    g = np.array(got).astype(np.float32)
    r = np.array(ref).astype(np.float32)
    assert np.isfinite(g).all()
    scale = max(float(np.abs(r).max()), 1e-6)
    assert float(np.abs(g - r).max()) <= REL_TOL * scale


# ---------------------------------------------------------------------------
# Parity vs the MLX reference path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64, 128, 256])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [1, 5, 33, 1000])
def test_rvq_unpack_decode_matches_mlx_path(D, bits, N):
    # bits=3 -> 10 codes per word, so D=32/64/128/256 all end in a padded word.
    q = TurboQuantRVQ(d=D, b=bits, seed=D + bits + N, use_hadamard=True)
    x = _unit_rows(N, D, seed=D * 100 + bits * 10 + N)
    norms = _norms(N)
    p1, p2 = q.encode_pack(x)

    ref = _reference_decode(q, p1, p2, norms, bits, D)
    got = q.decode_packed(p1, p2, norms)

    assert got.shape == (N, D)
    assert got.dtype == mx.float16
    _assert_close(got, ref)


@pytest.mark.parametrize("diag_sign", [1.0, -1.0])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_unpack_and_codebook_sum_are_bit_exact(bits, diag_sign):
    """D == 1: the Hadamard is the identity, so only unpack + gather + sum + rescale run.

    The quantizer can't be built at d=1, so borrow a d=64 quantizer's codebooks
    and drive the kernel directly with a one-element diagonal.
    """
    q = TurboQuantRVQ(d=64, b=bits, seed=0, use_hadamard=True)
    diag1 = mx.array([diag_sign], dtype=mx.float32)
    n_levels = 1 << bits
    rng = np.random.default_rng(bits)
    N = 512
    i1 = rng.integers(0, n_levels, N).astype(np.uint32)
    i2 = rng.integers(0, n_levels, N).astype(np.uint32)
    p1 = mx.array(i1.reshape(N, 1))  # D == 1 -> one code in the low bits of one word
    p2 = mx.array(i2.reshape(N, 1))
    norms = _norms(N, seed=7)

    got = rvq_unpack_decode(
        p1,
        p2,
        norms,
        q._codebook1.centroids_mx(),
        q._codebook2.centroids_mx(),
        diag1,
        bits,
        1,
    )

    c1 = np.array(q._codebook1.centroids_mx().astype(mx.float32)).astype(np.float16)
    c2 = np.array(q._codebook2.centroids_mx().astype(mx.float32)).astype(np.float16)
    diag = np.array(diag1)
    y = (c1[i1] + c2[i2]).astype(np.float16)  # fp16 sum, as in the reference
    x_unit = (y.astype(np.float32) * diag[0]).astype(np.float16)
    nrm = np.array(norms.astype(mx.float32)).reshape(-1)
    expect = np.clip(x_unit.astype(np.float32) * nrm, -65504.0, 65504.0).astype(np.float16)

    mx.eval(got)
    np.testing.assert_array_equal(np.array(got).reshape(-1), expect)


def test_round_trip_matches_encode_decode():
    """encode_pack -> decode_packed equals the unpacked encode -> decode path."""
    D, bits, N = 128, 2, 64
    q = TurboQuantRVQ(d=D, b=bits, seed=11, use_hadamard=True)
    x = _unit_rows(N, D, seed=3)
    ones = mx.ones((N, 1), dtype=mx.bfloat16)

    p1, p2 = q.encode_pack(x)
    got = q.decode_packed(p1, p2, ones)
    ref = q.decode(q.encode(x))
    _assert_close(got, ref)


def test_error_far_below_quantization_error():
    """The kernel-vs-MLX gap must be tiny next to the codec's own error."""
    D, bits, N = 128, 2, 256
    q = TurboQuantRVQ(d=D, b=bits, seed=5, use_hadamard=True)
    x = _unit_rows(N, D, seed=9)
    norms = _norms(N)
    p1, p2 = q.encode_pack(x)

    got = np.array(q.decode_packed(p1, p2, norms)).astype(np.float32)
    ref = np.array(_reference_decode(q, p1, p2, norms, bits, D)).astype(np.float32)
    truth = np.array(x).astype(np.float32) * np.array(norms.astype(mx.float32))

    kernel_gap = np.abs(got - ref).max()
    quant_err = np.abs(got - truth).max()
    assert kernel_gap <= quant_err / 20.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_saturates_like_rescale_fp16():
    D, bits, N = 64, 2, 8
    q = TurboQuantRVQ(d=D, b=bits, seed=2, use_hadamard=True)
    x = _unit_rows(N, D, seed=4)
    p1, p2 = q.encode_pack(x)
    # bf16 has fp32's range, so 2e5 is a legal stored norm and must clip to fp16 max.
    norms = mx.full((N, 1), 2.0e5, dtype=mx.bfloat16)

    got = q.decode_packed(p1, p2, norms)
    ref = _reference_decode(q, p1, p2, norms, bits, D)
    mx.eval(got, ref)
    g = np.array(got).astype(np.float32)
    assert np.isfinite(g).all()
    assert float(np.abs(g).max()) == 65504.0
    _assert_close(got, ref)


def test_near_limit_norm_stays_finite():
    D, bits, N = 128, 1, 16
    q = TurboQuantRVQ(d=D, b=bits, seed=6, use_hadamard=True)
    x = _unit_rows(N, D, seed=8)
    p1, p2 = q.encode_pack(x)
    norms = mx.full((N, 1), 6.0e4, dtype=mx.bfloat16)

    got = q.decode_packed(p1, p2, norms)
    ref = _reference_decode(q, p1, p2, norms, bits, D)
    _assert_close(got, ref)


def test_zero_norm_gives_zero_vector():
    D, bits, N = 32, 2, 4
    q = TurboQuantRVQ(d=D, b=bits, seed=1, use_hadamard=True)
    p1, p2 = q.encode_pack(_unit_rows(N, D, seed=1))
    got = q.decode_packed(p1, p2, mx.zeros((N, 1), dtype=mx.bfloat16))
    mx.eval(got)
    assert not np.array(got).any()


def test_non_contiguous_slice_input():
    """The cache hands the kernel a [..., :offset, :] view of a larger buffer."""
    D, bits, B, H, cap, n = 128, 2, 1, 4, 96, 40
    q = TurboQuantRVQ(d=D, b=bits, seed=3, use_hadamard=True)
    el = 32 // bits
    n_words = -(-D // el)
    full = B * H * cap
    p1, p2 = q.encode_pack(_unit_rows(full, D, seed=5))
    p1 = p1.reshape(B, H, cap, n_words)
    p2 = p2.reshape(B, H, cap, n_words)
    norms = _norms(full).reshape(B, H, cap, 1)

    s1 = p1[..., :n, :].reshape(-1, n_words)
    s2 = p2[..., :n, :].reshape(-1, n_words)
    sn = norms[..., :n, :].reshape(-1, 1)
    got = q.decode_packed(s1, s2, sn)
    ref = _reference_decode(q, s1, s2, sn, bits, D)
    _assert_close(got, ref)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_reads_over_allocated_cache_buffer_in_place(bits):
    """3D [BH, cap, n_words] + seq_len == decoding the sliced live region."""
    D, BH, cap, n = 128, 6, 70, 23
    q = TurboQuantRVQ(d=D, b=bits, seed=bits, use_hadamard=True)
    n_words = -(-D // (32 // bits))
    p1, p2 = q.encode_pack(_unit_rows(BH * cap, D, seed=bits))
    p1, p2 = p1.reshape(BH, cap, n_words), p2.reshape(BH, cap, n_words)
    norms = _norms(BH * cap, seed=bits).reshape(BH, cap, 1)

    got = q.decode_packed(p1, p2, norms, seq_len=n)

    s1 = p1[:, :n, :].reshape(-1, n_words)
    s2 = p2[:, :n, :].reshape(-1, n_words)
    sn = norms[:, :n, :].reshape(-1, 1)
    ref = _reference_decode(q, s1, s2, sn, bits, D)
    assert got.shape == (BH * n, D)
    _assert_close(got, ref)


def test_seq_len_validation():
    a = _valid_args(D=64, bits=2, N=4)
    with pytest.raises(ValueError, match="seq_len requires 3D"):
        rvq_unpack_decode(*a, seq_len=2)
    p1, p2 = a[0].reshape(1, 4, -1), a[1].reshape(1, 4, -1)
    with pytest.raises(ValueError, match="seq_len=9"):
        rvq_unpack_decode(p1, p2, a[2], *a[3:], seq_len=9)
    out = rvq_unpack_decode(p1, p2, a[2], *a[3:], seq_len=0)
    assert out.shape == (0, 64)


def test_zero_rows_returns_empty():
    q = TurboQuantRVQ(d=64, b=2, seed=0, use_hadamard=True)
    out = rvq_unpack_decode(
        mx.zeros((0, 4), dtype=mx.uint32),
        mx.zeros((0, 4), dtype=mx.uint32),
        mx.zeros((0, 1), dtype=mx.bfloat16),
        q._codebook1.centroids_mx(),
        q._codebook2.centroids_mx(),
        q._rotation._D,
        2,
        64,
    )
    assert out.shape == (0, 64)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _valid_args(D=64, bits=2, N=2):
    q = TurboQuantRVQ(d=D, b=bits, seed=0, use_hadamard=True)
    p1, p2 = q.encode_pack(_unit_rows(N, D, seed=0))
    return [
        p1,
        p2,
        _norms(N),
        q._codebook1.centroids_mx(),
        q._codebook2.centroids_mx(),
        q._rotation._D,
        bits,
        D,
    ]


def test_rejects_bad_arguments():
    a = _valid_args()
    with pytest.raises(ValueError, match="power of two"):
        rvq_unpack_decode(*a[:7], 96)
    with pytest.raises(ValueError, match="1024"):
        rvq_unpack_decode(*a[:7], 2048)
    with pytest.raises(ValueError, match="bits must be 1-4"):
        rvq_unpack_decode(*a[:6], 5, 64)
    with pytest.raises(ValueError, match="packed shape"):
        rvq_unpack_decode(a[0][:, :-1], a[1][:, :-1], *a[2:])
    with pytest.raises(ValueError, match="uint32"):
        rvq_unpack_decode(a[0].astype(mx.int32), *a[1:])
    with pytest.raises(ValueError, match="norms"):
        rvq_unpack_decode(a[0], a[1], _norms(5), *a[3:])
    with pytest.raises(ValueError, match="centroids"):
        rvq_unpack_decode(a[0], a[1], a[2], a[3][:-1], *a[4:])
    with pytest.raises(ValueError, match="diag"):
        rvq_unpack_decode(*a[:5], a[5][:-1], a[6], a[7])


def test_decode_packed_requires_hadamard():
    qr = TurboQuantRVQ(d=64, b=2, seed=0, use_hadamard=False)
    assert not qr.supports_fused_decode
    with pytest.raises(ValueError, match="use_hadamard"):
        qr.decode_packed(
            mx.zeros((1, 4), dtype=mx.uint32), mx.zeros((1, 4), dtype=mx.uint32), mx.ones((1, 1))
        )
    assert TurboQuantRVQ(d=64, b=2, seed=0, use_hadamard=True).supports_fused_decode
    # Hadamard-compatible but not a power of two: no fused path.
    assert not TurboQuantRVQ(d=96, b=2, seed=0, use_hadamard=True).supports_fused_decode


# ---------------------------------------------------------------------------
# Cache level
# ---------------------------------------------------------------------------


def _cfg(use_metal, head_dim=128, bits=2):
    return SimpleNamespace(
        head_dim=head_dim, bit_width_inlier=bits, seed=0, use_metal_kernels=use_metal
    )


def _kv(rng, n, h=4, d=128):
    k = mx.array(rng.standard_normal((1, h, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, h, n, d)).astype(np.float16))
    return k, v


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_cache_kernel_on_off_agree_across_decode_trim_grow(bits):
    on = TurboQuantRVQKVCache(_cfg(True, bits=bits))
    off = TurboQuantRVQKVCache(_cfg(False, bits=bits))
    assert on._use_metal_decode and not off._use_metal_decode
    rng = np.random.default_rng(bits)

    def step(n):
        k, v = _kv(rng, n)
        ko, vo = on.update_and_fetch(k, v)
        kf, vf = off.update_and_fetch(k, v)
        assert ko.shape == kf.shape
        _assert_close(ko, kf)
        np.testing.assert_array_equal(np.array(vo), np.array(vf))

    step(300)  # prefill chunk, crosses the 256-token growth step
    for _ in range(40):  # single-token decode
        step(1)
    on.trim(10)
    off.trim(10)
    for _ in range(230):  # decode past the next growth boundary (offset 330 -> 560)
        step(1)
    assert on.offset == off.offset == 560
    assert on._use_metal_decode, "kernel must stay enabled on the happy path"


def test_cache_zero_key_vectors_match():
    on = TurboQuantRVQKVCache(_cfg(True))
    off = TurboQuantRVQKVCache(_cfg(False))
    k = mx.zeros((1, 2, 8, 128), dtype=mx.float16)
    v = mx.ones((1, 2, 8, 128), dtype=mx.float16)
    ko, _ = on.update_and_fetch(k, v)
    kf, _ = off.update_and_fetch(k, v)
    _assert_close(ko, kf)


def test_cache_gate_conditions():
    assert TurboQuantRVQKVCache(_cfg(None))._use_metal_decode  # auto
    assert not TurboQuantRVQKVCache(_cfg(False))._use_metal_decode  # forced off
    assert not TurboQuantRVQKVCache(_cfg(None, head_dim=96))._use_metal_decode  # not pow2


def test_cache_non_fp16_keys_use_mlx_path():
    cache = TurboQuantRVQKVCache(_cfg(True))
    rng = np.random.default_rng(0)
    k, v = _kv(rng, 16)
    k = k.astype(mx.bfloat16)
    ko, _ = cache.update_and_fetch(k, v.astype(mx.bfloat16))
    assert ko.dtype == mx.bfloat16
    assert cache._use_metal_decode  # not latched off, just not used for bf16


def test_cache_latches_off_when_kernel_raises(monkeypatch):
    cache = TurboQuantRVQKVCache(_cfg(True))
    ref = TurboQuantRVQKVCache(_cfg(False))

    def boom(*args, **kwargs):
        raise RuntimeError("simulated kernel failure")

    monkeypatch.setattr(kernels_mod, "rvq_unpack_decode", boom)
    rng = np.random.default_rng(0)
    k, v = _kv(rng, 32)
    ko, _ = cache.update_and_fetch(k, v)
    kf, _ = ref.update_and_fetch(k, v)

    assert cache._use_metal_decode is False
    np.testing.assert_array_equal(np.array(ko), np.array(kf))  # MLX path, byte-identical


def test_cache_requires_metal_when_forced_on(monkeypatch):
    import veloxquant_mlx.cache.turboquant_rvq_cache as cache_mod

    monkeypatch.setattr(cache_mod, "metal_available", lambda: False)
    with pytest.raises(RuntimeError, match="use_metal_kernels=True"):
        TurboQuantRVQKVCache(_cfg(True))
    # None / False degrade silently.
    assert not TurboQuantRVQKVCache(_cfg(None))._use_metal_decode
    assert not TurboQuantRVQKVCache(_cfg(False))._use_metal_decode


def test_cache_state_roundtrip_rebuilds_gate():
    """from_state() bypasses __init__; the gate must still be rebuilt."""
    src = TurboQuantRVQKVCache(_cfg(True))
    rng = np.random.default_rng(1)
    k, v = _kv(rng, 20)
    expect, _ = src.update_and_fetch(k, v)

    clone = TurboQuantRVQKVCache.from_state(src.state, src.meta_state)
    assert clone._use_metal_decode
    got, _ = clone.update_and_fetch(*_kv(rng, 1))
    _assert_close(got[..., :20, :], expect)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_compiles_decode_kernel_with_same_key_as_real_call():
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.metal._warmup import warmup_for_config

    _rvq_unpack_decode._cache.clear()
    cfg = KVCacheConfig(method="turboquant_rvq", head_dim=128, bit_width_inlier=2, seed=42)
    warmup_for_config(cfg)
    warm_keys = set(_rvq_unpack_decode._cache.keys())
    assert ("rvq_unpack_decode", 128, 2) in warm_keys

    cache = TurboQuantRVQKVCache(cfg)
    k = mx.random.normal((1, 1, 1, 128)).astype(mx.float16)
    cache.update_and_fetch(k, k)
    assert set(_rvq_unpack_decode._cache.keys()) == warm_keys

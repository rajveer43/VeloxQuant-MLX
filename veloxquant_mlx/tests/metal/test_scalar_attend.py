"""Parity + benchmark tests for the fused group-affine decode + attend kernel.

The reference recomputes the exact KIVI reconstruction math and attention:
per-channel group-affine dequant of keys (``code*scale+zero``, groups along
the token axis), per-token group-affine dequant of values (groups along the
channel axis), scaled dot product, softmax, and the value matmul. The
kernel's fp16 output must match the float32 reference within tolerance.

The same codes / scales / zeros feed both the reference and the kernel, so
the only permitted divergence is the kernel's fp32 online-softmax
accumulation vs. the reference's direct softmax — which is *more* accurate,
not less.
"""

from __future__ import annotations

import math
import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal._scalar_attend import (
    scalar_decode_once,
    scalar_predecoded_attend,
)
from veloxquant_mlx.metal.kernels import scalar_fused_decode_attend

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Quantization helpers (KIVI-exact group-affine min/max)
# ---------------------------------------------------------------------------


def _quant_keys(k: np.ndarray, g: int, levels: int, eps=1e-8):
    """Per-CHANNEL group quant (group along tokens). k: [B,H,S,D]."""
    B, H, S, D = k.shape
    GK = (S + g - 1) // g
    pad = GK * g - S
    x = k.astype(np.float32)
    if pad:
        x = np.concatenate([x, np.broadcast_to(x[:, :, -1:, :], (B, H, pad, D))], axis=2)
    xg = x.reshape(B, H, GK, g, D)
    gmin = xg.min(axis=3, keepdims=True)
    gmax = xg.max(axis=3, keepdims=True)
    scale = np.maximum((gmax - gmin) / levels, eps)
    codes = np.clip(np.round((xg - gmin) / scale), 0, levels)
    codes = codes.reshape(B, H, GK * g, D)[:, :, :S, :].astype(np.uint8)
    return codes, scale.reshape(B, H, GK, D), gmin.reshape(B, H, GK, D)


def _quant_values(v: np.ndarray, g: int, levels: int, eps=1e-8):
    """Per-TOKEN group quant (group along channels). v: [B,H,S,D]."""
    B, H, S, D = v.shape
    GV = (D + g - 1) // g
    pad = GV * g - D
    x = v.astype(np.float32)
    if pad:
        x = np.concatenate([x, np.broadcast_to(x[:, :, :, -1:], (B, H, S, pad))], axis=3)
    xg = x.reshape(B, H, S, GV, g)
    gmin = xg.min(axis=4, keepdims=True)
    gmax = xg.max(axis=4, keepdims=True)
    scale = np.maximum((gmax - gmin) / levels, eps)
    codes = np.clip(np.round((xg - gmin) / scale), 0, levels)
    codes = codes.reshape(B, H, S, GV * g)[:, :, :, :D].astype(np.uint8)
    return codes, scale.reshape(B, H, S, GV), gmin.reshape(B, H, S, GV)


def _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale):
    """Reconstruct fp32 K_hat/V_hat then dense softmax attention (numpy).

    Supports GQA: if k/v carry fewer heads than q (kc.shape[1] < q.shape[1]),
    they are repeat-interleaved up to q's head count first — contiguous
    blocks of ``heads_per_kv`` query heads share one kv head, matching the
    kernel's own hkv_idx*heads_per_kv + hp mapping.
    """
    B, H, S_q, D = q.shape
    H_kv = kc.shape[1]
    if H_kv != H:
        assert H % H_kv == 0, f"H={H} not a multiple of H_kv={H_kv}"
        heads_per_kv = H // H_kv
        kc = np.repeat(kc, heads_per_kv, axis=1)
        ks = np.repeat(ks, heads_per_kv, axis=1)
        kz = np.repeat(kz, heads_per_kv, axis=1)
        vc = np.repeat(vc, heads_per_kv, axis=1)
        vs = np.repeat(vs, heads_per_kv, axis=1)
        vz = np.repeat(vz, heads_per_kv, axis=1)

    S_kv = kc.shape[2]
    kg = np.arange(S_kv) // g
    k_hat = kc.astype(np.float32) * ks[:, :, kg, :] + kz[:, :, kg, :]  # [B,H,Skv,D]
    vgi = np.arange(D) // g
    v_hat = vc.astype(np.float32) * vs[:, :, :, vgi] + vz[:, :, :, vgi]  # [B,H,Skv,D]

    scores = np.einsum("bhqd,bhsd->bhqs", q.astype(np.float32), k_hat) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("bhqs,bhsd->bhqd", w, v_hat).astype(np.float32)


def _make_inputs(B, H, S_kv, D, b, g, seed=0, H_kv=None):
    """Build q at H query heads and k/v at H_kv heads (defaults to H, i.e. MHA)."""
    if H_kv is None:
        H_kv = H
    rng = np.random.default_rng(seed)
    levels = (1 << b) - 1
    q = rng.standard_normal((B, H, 1, D)).astype(np.float16)
    kf = rng.standard_normal((B, H_kv, S_kv, D)).astype(np.float32)
    vf = rng.standard_normal((B, H_kv, S_kv, D)).astype(np.float32)
    kc, ks, kz = _quant_keys(kf, g, levels)
    vc, vs, vz = _quant_values(vf, g, levels)
    return q, kc, ks, kz, vc, vs, vz


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S_kv", [64, 512, 2048])
@pytest.mark.parametrize("nsg", [1, 2, 4, 8])
def test_scalar_attend_parity(S_kv, nsg):
    B, H, D, b, g = 1, 4, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H, S_kv, D, b, g)

    ref = _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale)
    got = scalar_fused_decode_attend(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=nsg,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    assert got_np.shape == ref.shape
    max_abs = np.abs(got_np - ref).max()
    assert max_abs < 2e-3, f"S_kv={S_kv} nsg={nsg}: max|abs|={max_abs:.3e}"


@pytest.mark.parametrize("b", [2, 3, 4])
def test_scalar_attend_bitwidths(b):
    B, H, D, g, S_kv = 1, 4, 128, 32, 1024
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H, S_kv, D, b, g, seed=b)
    ref = _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale)
    got = scalar_fused_decode_attend(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=4,
    )
    mx.eval(got)
    max_abs = np.abs(np.array(got).astype(np.float32) - ref).max()
    assert max_abs < 2e-3, f"b={b}: max|abs|={max_abs:.3e}"


# ---------------------------------------------------------------------------
# GQA head-packing parity
# ---------------------------------------------------------------------------


def _tg_mem_bytes(nsg, heads_per_kv):
    # Mirrors _scalar_attend.py's own budget check: sh_o + sh_m + sh_d, all fp32.
    n_slots = nsg * heads_per_kv
    return (n_slots * 8 * 32 + n_slots + n_slots) * 4


_GQA_CASES = [
    (H_q, H_kv, S_kv, nsg)
    for H_q, H_kv in [(8, 2), (32, 4), (32, 8)]
    for S_kv in [64, 512, 2048]
    for nsg in [1, 2, 4]
    if _tg_mem_bytes(nsg, H_q // H_kv) <= 32768
]


@pytest.mark.parametrize("H_q,H_kv,S_kv,nsg", _GQA_CASES)
def test_scalar_attend_gqa_parity(H_q, H_kv, S_kv, nsg):
    D, b, g = 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(1, H_q, S_kv, D, b, g, H_kv=H_kv)

    ref = _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale)
    got = scalar_fused_decode_attend(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=nsg,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    assert got_np.shape == ref.shape == (1, H_q, 1, D)
    max_abs = np.abs(got_np - ref).max()
    assert max_abs < 2e-3, f"H_q={H_q} H_kv={H_kv} S_kv={S_kv} nsg={nsg}: max|abs|={max_abs:.3e}"


def test_scalar_attend_gqa_parity_batched():
    # B=2 exercises the b_idx*H_kv term in bh_kv, which B=1 alone can't catch.
    B, H_q, H_kv, S_kv, D, b, g = 2, 8, 2, 512, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H_q, S_kv, D, b, g, H_kv=H_kv)

    ref = _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale)
    got = scalar_fused_decode_attend(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=4,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    assert got_np.shape == ref.shape == (B, H_q, 1, D)
    max_abs = np.abs(got_np - ref).max()
    assert max_abs < 2e-3, f"B={B}: max|abs|={max_abs:.3e}"


def test_scalar_attend_validation():
    q = mx.zeros((1, 4, 1, 128), dtype=mx.float16)
    z = mx.zeros((1, 4, 8, 128), dtype=mx.uint8)
    s = mx.zeros((1, 4, 1, 4), dtype=mx.float32)
    # D > 256 rejected
    with pytest.raises(ValueError):
        scalar_fused_decode_attend(
            mx.zeros((1, 4, 1, 512), dtype=mx.float16),
            z,
            z.astype(mx.float32),
            z.astype(mx.float32),
            z,
            s,
            s,
            32,
            0.1,
        )
    # bad nsg rejected
    with pytest.raises(ValueError):
        scalar_fused_decode_attend(
            q, z, z.astype(mx.float32), z.astype(mx.float32), z, s, s, 32, 0.1, nsg=0
        )
    # H_q not a multiple of H_kv rejected
    with pytest.raises(ValueError):
        scalar_fused_decode_attend(
            mx.zeros((1, 5, 1, 128), dtype=mx.float16),  # H_q=5
            mx.zeros((1, 2, 8, 128), dtype=mx.uint8),  # H_kv=2, 5 % 2 != 0
            mx.zeros((1, 2, 1, 128), dtype=mx.float32),
            mx.zeros((1, 2, 1, 128), dtype=mx.float32),
            mx.zeros((1, 2, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 2, 8, 4), dtype=mx.float32),
            mx.zeros((1, 2, 8, 4), dtype=mx.float32),
            32,
            0.1,
        )
    # heads_per_kv exceeding _MAX_HEADS_PER_KV rejected
    with pytest.raises(ValueError):
        scalar_fused_decode_attend(
            mx.zeros((1, 32, 1, 128), dtype=mx.float16),  # H_q=32
            mx.zeros((1, 1, 8, 128), dtype=mx.uint8),  # H_kv=1 -> heads_per_kv=32
            mx.zeros((1, 1, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
            mx.zeros((1, 1, 8, 4), dtype=mx.float32),
            32,
            0.1,
            nsg=1,
        )
    # threadgroup-memory budget overflow (nsg * heads_per_kv within the
    # heads_per_kv cap but still too large for the softmax merge buffers)
    with pytest.raises(ValueError):
        scalar_fused_decode_attend(
            mx.zeros((1, 32, 1, 128), dtype=mx.float16),  # H_q=32
            mx.zeros((1, 4, 8, 128), dtype=mx.uint8),  # H_kv=4 -> heads_per_kv=8
            mx.zeros((1, 4, 1, 128), dtype=mx.float32),
            mx.zeros((1, 4, 1, 128), dtype=mx.float32),
            mx.zeros((1, 4, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 4, 8, 4), dtype=mx.float32),
            mx.zeros((1, 4, 8, 4), dtype=mx.float32),
            32,
            0.1,
            nsg=4,  # nsg=4 * heads_per_kv=8 -> 33024B > 32768B budget
        )


# ---------------------------------------------------------------------------
# Benchmark (printed, not asserted) — before/after vs. dequant->MLX SDPA
# ---------------------------------------------------------------------------


def test_scalar_attend_benchmark(capsys):
    B, H, D, b, g = 1, 32, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    nsg = 8

    def _timeit(fn, iters=30, warmup=10):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    def _baseline(q, kc, ks, kz, vc, vs, vz):
        # reconstruct fp16 K_hat/V_hat then MLX SDPA — the round-trip we kill
        S_kv = kc.shape[2]
        kg = mx.arange(S_kv) // g
        k_hat = (kc.astype(mx.float32) * mx.take(ks, kg, axis=2) + mx.take(kz, kg, axis=2)).astype(
            mx.float16
        )
        vgi = mx.arange(D) // g
        v_hat = (
            vc.astype(mx.float32) * mx.take(vs, vgi, axis=3) + mx.take(vz, vgi, axis=3)
        ).astype(mx.float16)
        return mx.fast.scaled_dot_product_attention(q, k_hat, v_hat, scale=scale)

    with capsys.disabled():
        print(
            f"\n# fused group-affine decode-attend  |  B={B} H={H} D={D} b={b} "
            f"g={g} nsg={nsg}  |  MLX {mx.__version__}"
        )
        print("| S_kv | before (MLX) ms | after (fused) ms | speedup |")
        print("|------|-----------------|------------------|---------|")
        for S_kv in [512, 2048, 8192, 16384]:
            q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H, S_kv, D, b, g)
            aq = mx.array(q)
            akc = mx.array(kc)
            aks = mx.array(ks)
            akz = mx.array(kz)
            avc = mx.array(vc)
            avs = mx.array(vs)
            avz = mx.array(vz)
            mx.eval(aq, akc, aks, akz, avc, avs, avz)
            tb = _timeit(lambda: _baseline(aq, akc, aks, akz, avc, avs, avz))
            ta = _timeit(
                lambda: scalar_fused_decode_attend(
                    aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=nsg
                )
            )
            print(f"| {S_kv:5d} | {tb:15.3f} | {ta:16.3f} | {tb / ta:6.2f}x |")


def test_scalar_attend_gqa_packing_benchmark(capsys):
    """Packed single dispatch vs. heads_per_kv sequential unpacked dispatches.

    Output parity alone can't prove K/V decode is actually being shared
    (a buggy kernel that redundantly redecodes per head could still produce
    correct output). This times the packed kernel against heads_per_kv
    separate calls of the same (heads_per_kv=1) kernel, each against one
    query head sliced against the shared kv head's codes — i.e. simulating
    the redundant-redecode baseline with the same underlying kernel, so a
    real speedup at large S_kv is direct evidence sharing is happening.
    """
    B, H_q, H_kv, D, b, g = 1, 32, 4, 128, 2, 32
    heads_per_kv = H_q // H_kv
    scale = 1.0 / math.sqrt(D)
    nsg = 2  # nsg=4 would overflow the 32KB threadgroup-memory budget at heads_per_kv=8

    def _timeit(fn, iters=20, warmup=10):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    with capsys.disabled():
        print(
            f"\n# GQA head-packing  |  B={B} H_q={H_q} H_kv={H_kv} "
            f"heads_per_kv={heads_per_kv} D={D} b={b} g={g} nsg={nsg}  |  MLX {mx.__version__}"
        )
        print("| S_kv | packed ms | unpacked ms | speedup |")
        print("|------|-----------|-------------|---------|")
        for S_kv in [512, 2048, 8192, 16384]:
            q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H_q, S_kv, D, b, g, H_kv=H_kv)
            aq = mx.array(q)
            akc = mx.array(kc)
            aks = mx.array(ks)
            akz = mx.array(kz)
            avc = mx.array(vc)
            avs = mx.array(vs)
            avz = mx.array(vz)
            mx.eval(aq, akc, aks, akz, avc, avs, avz)

            def _packed():
                return scalar_fused_decode_attend(
                    aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=nsg
                )

            def _unpacked():
                outs = []
                for hkv in range(H_kv):
                    for hp in range(heads_per_kv):
                        hq = hkv * heads_per_kv + hp
                        outs.append(
                            scalar_fused_decode_attend(
                                aq[:, hq : hq + 1],
                                akc[:, hkv : hkv + 1],
                                aks[:, hkv : hkv + 1],
                                akz[:, hkv : hkv + 1],
                                avc[:, hkv : hkv + 1],
                                avs[:, hkv : hkv + 1],
                                avz[:, hkv : hkv + 1],
                                g,
                                scale,
                                nsg=nsg,
                            )
                        )
                return mx.concatenate(outs, axis=1)

            tp = _timeit(_packed)
            tu = _timeit(_unpacked)
            print(f"| {S_kv:5d} | {tp:9.3f} | {tu:11.3f} | {tu / tp:6.2f}x |")


# ---------------------------------------------------------------------------
# Issue #308 spike: two-pass decode-once + predecoded-attend
# ---------------------------------------------------------------------------


def _two_pass_attend(q, kc, ks, kz, vc, vs, vz, g, scale, nsg=4):
    """scalar_decode_once (K, V) followed by scalar_predecoded_attend."""
    k_hat = scalar_decode_once(kc, ks, kz, g, mode="K")
    v_hat = scalar_decode_once(vc, vs, vz, g, mode="V")
    return scalar_predecoded_attend(q, k_hat, v_hat, scale, nsg=nsg)


@pytest.mark.parametrize("S_kv", [64, 512, 2048])
@pytest.mark.parametrize("H_q,H_kv", [(4, 4), (8, 2), (32, 4)])
def test_scalar_decode_once_predecoded_attend_parity(H_q, H_kv, S_kv):
    """Two-pass output must match the same numpy reference as the fused kernel."""
    B, D, b, g = 1, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H_q, S_kv, D, b, g, H_kv=H_kv)

    aq = mx.array(q)
    akc, aks, akz = mx.array(kc), mx.array(ks), mx.array(kz)
    avc, avs, avz = mx.array(vc), mx.array(vs), mx.array(vz)

    out = _two_pass_attend(aq, akc, aks, akz, avc, avs, avz, g, scale)
    ref = _reference_attend(q, kc, ks, kz, vc, vs, vz, g, scale)

    err = np.abs(np.array(out.astype(mx.float32)) - ref)
    assert err.max() < 2e-3, f"max abs error {err.max():.6f} at H_q={H_q},H_kv={H_kv},S_kv={S_kv}"


def test_scalar_decode_once_validation():
    with pytest.raises(ValueError, match="mode must be"):
        scalar_decode_once(
            mx.zeros((1, 2, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 2, 1, 128), dtype=mx.float32),
            mx.zeros((1, 2, 1, 128), dtype=mx.float32),
            group_size=32,
            mode="bogus",
        )
    with pytest.raises(ValueError, match="H_q=.*must be a multiple"):
        scalar_predecoded_attend(
            mx.zeros((1, 5, 1, 128), dtype=mx.float16),  # H_q=5
            mx.zeros((1, 2, 8, 128), dtype=mx.float16),  # H_kv=2 -- 5 % 2 != 0
            mx.zeros((1, 2, 8, 128), dtype=mx.float16),
            scale=1.0,
        )


def test_scalar_attend_two_pass_benchmark(capsys):
    """Three-way timing: packed (307 pt.2) vs. unpacked-redundant vs. two-pass (308).

    Issue #308's spike question: does paying one DRAM round-trip to decode
    K/V exactly once, then running the existing full-occupancy per-head
    attend dispatch against the decoded buffer, beat redundantly
    re-decoding K/V on-the-fly per query head (today's fastest baseline,
    per #307's addendum)? Printed only -- the go/no-go call is made from
    reading these numbers, not asserted here, matching
    test_scalar_attend_gqa_packing_benchmark's style.
    """
    B, H_q, H_kv, D, b, g = 1, 32, 4, 128, 2, 32
    heads_per_kv = H_q // H_kv
    scale = 1.0 / math.sqrt(D)
    nsg = 2  # matches the nsg used in the #307 packed-vs-unpacked benchmark

    def _timeit(fn, iters=20, warmup=10):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    with capsys.disabled():
        print(
            f"\n# Two-pass decode-once + predecoded-attend (issue #308)  |  "
            f"B={B} H_q={H_q} H_kv={H_kv} heads_per_kv={heads_per_kv} "
            f"D={D} b={b} g={g} nsg={nsg}  |  MLX {mx.__version__}"
        )
        print("| S_kv | packed ms | unpacked ms | two-pass ms | two-pass vs unpacked |")
        print("|------|-----------|-------------|-------------|-----------------------|")
        for S_kv in [512, 2048, 8192, 16384]:
            q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H_q, S_kv, D, b, g, H_kv=H_kv)
            aq = mx.array(q)
            akc, aks, akz = mx.array(kc), mx.array(ks), mx.array(kz)
            avc, avs, avz = mx.array(vc), mx.array(vs), mx.array(vz)
            mx.eval(aq, akc, aks, akz, avc, avs, avz)

            def _packed():
                return scalar_fused_decode_attend(
                    aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=nsg
                )

            def _unpacked():
                outs = []
                for hkv in range(H_kv):
                    for hp in range(heads_per_kv):
                        hq = hkv * heads_per_kv + hp
                        outs.append(
                            scalar_fused_decode_attend(
                                aq[:, hq : hq + 1],
                                akc[:, hkv : hkv + 1],
                                aks[:, hkv : hkv + 1],
                                akz[:, hkv : hkv + 1],
                                avc[:, hkv : hkv + 1],
                                avs[:, hkv : hkv + 1],
                                avz[:, hkv : hkv + 1],
                                g,
                                scale,
                                nsg=nsg,
                            )
                        )
                return mx.concatenate(outs, axis=1)

            def _two_pass():
                return _two_pass_attend(aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=nsg)

            tp = _timeit(_packed)
            tu = _timeit(_unpacked)
            tt = _timeit(_two_pass)
            print(f"| {S_kv:5d} | {tp:9.3f} | {tu:11.3f} | {tt:11.3f} | {tu / tt:21.2f}x |")

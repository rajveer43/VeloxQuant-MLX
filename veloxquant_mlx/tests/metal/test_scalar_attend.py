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
    _TG_MEM_BUDGET_BYTES,
    _auto_nsg,
    _d_slots,
    scalar_decode_once,
    scalar_predecoded_attend,
)
from veloxquant_mlx.metal.kernels import (
    scalar_fused_decode_attend,
    scalar_fused_decode_attend_batched,
)

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


def _tg_mem_bytes(nsg, heads_per_kv, D=128):
    # Mirrors _scalar_attend.py's own budget check: sh_o (half) + sh_m, sh_d
    # (both fp32). sh_o is sized by DSLOTS_C = ceil(D/32), not a fixed 8 — so
    # the admissible nsg depends on the head dim.
    n_slots = nsg * heads_per_kv
    d_slots = (D + 31) // 32
    return n_slots * d_slots * 32 * 2 + n_slots * 4 * 2


_GQA_CASES = [
    (H_q, H_kv, S_kv, nsg)
    for H_q, H_kv in [(8, 2), (32, 4), (32, 8)]
    for S_kv in [64, 512, 2048]
    for nsg in [1, 2, 4, 8]
    if _tg_mem_bytes(nsg, H_q // H_kv, 128) <= 32768
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
    # threadgroup-memory budget overflow. The budget scales with
    # DSLOTS_C = ceil(D/32) and sh_o is half, so overflow now requires a
    # genuinely large combination: D=256 (8 slots) x heads_per_kv=8 x nsg=8
    # -> 33280B. At D=128 these same nsg/heads_per_kv legitimately fit.
    with pytest.raises(ValueError, match="threadgroup-memory budget"):
        scalar_fused_decode_attend(
            mx.zeros((1, 32, 1, 256), dtype=mx.float16),  # H_q=32, D=256
            mx.zeros((1, 4, 8, 256), dtype=mx.uint8),  # H_kv=4 -> heads_per_kv=8
            mx.zeros((1, 4, 1, 256), dtype=mx.float32),
            mx.zeros((1, 4, 1, 256), dtype=mx.float32),
            mx.zeros((1, 4, 8, 256), dtype=mx.uint8),
            mx.zeros((1, 4, 8, 8), dtype=mx.float32),
            mx.zeros((1, 4, 8, 8), dtype=mx.float32),
            32,
            0.1,
            nsg=8,  # nsg=8 * hpk=8 * 8 slots -> 33280B > 32768B budget
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


# ---------------------------------------------------------------------------
# Cross-layer batched dispatch (issue #307 part 1)
# ---------------------------------------------------------------------------


def _make_layer_stack(NL, B, H, S_kv, D, b, g, seed=0, H_kv=None):
    """Build NL independently-random layers' worth of _make_inputs, pre-stacked
    along a new leading axis — the precondition scalar_fused_decode_attend_batched
    documents (caller stacks; kernel does not gather)."""
    qs, kcs, kss, kzs, vcs, vss, vzs = [], [], [], [], [], [], []
    for layer_idx in range(NL):
        q, kc, ks, kz, vc, vs, vz = _make_inputs(
            B, H, S_kv, D, b, g, seed=seed + layer_idx, H_kv=H_kv
        )
        qs.append(q)
        kcs.append(kc)
        kss.append(ks)
        kzs.append(kz)
        vcs.append(vc)
        vss.append(vs)
        vzs.append(vz)
    return (
        np.stack(qs, axis=0),
        np.stack(kcs, axis=0),
        np.stack(kss, axis=0),
        np.stack(kzs, axis=0),
        np.stack(vcs, axis=0),
        np.stack(vss, axis=0),
        np.stack(vzs, axis=0),
    )


@pytest.mark.parametrize("NL", [1, 4, 32])
def test_scalar_attend_batched_parity_vs_single_layer_loop(NL):
    """Load-bearing correctness test: batched dispatch vs. calling the
    single-layer kernel NL times in a loop and stacking the outputs must be
    bit-identical — the new NL grid axis is pure batching, not a semantic
    change to the math."""
    B, H, S_kv, D, b, g = 1, 8, 512, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_layer_stack(NL, B, H, S_kv, D, b, g, seed=100)

    loop_outs = []
    for layer_idx in range(NL):
        out = scalar_fused_decode_attend(
            mx.array(q[layer_idx]),
            mx.array(kc[layer_idx]),
            mx.array(ks[layer_idx]),
            mx.array(kz[layer_idx]),
            mx.array(vc[layer_idx]),
            mx.array(vs[layer_idx]),
            mx.array(vz[layer_idx]),
            g,
            scale,
            nsg=2,
        )
        mx.eval(out)
        loop_outs.append(np.array(out))
    ref = np.stack(loop_outs, axis=0)

    got = scalar_fused_decode_attend_batched(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=2,
    )
    mx.eval(got)
    got_np = np.array(got)

    assert got_np.shape == ref.shape == (NL, B, H, 1, D)
    max_abs = np.abs(got_np.astype(np.float32) - ref.astype(np.float32)).max()
    assert max_abs == 0.0, (
        f"NL={NL}: batched must be bit-identical to the single-layer loop, got {max_abs:.3e}"
    )


@pytest.mark.parametrize("NL", [1, 8])
@pytest.mark.parametrize("H_q,H_kv", [(4, 4), (8, 2), (32, 4), (32, 8)])
def test_scalar_attend_batched_parity_vs_numpy_reference(NL, H_q, H_kv):
    """Batched output must match the same numpy reference used for the
    single-layer kernel, applied independently per layer."""
    B, S_kv, D, b, g = 1, 256, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_layer_stack(NL, B, H_q, S_kv, D, b, g, seed=200, H_kv=H_kv)

    got = scalar_fused_decode_attend_batched(
        mx.array(q),
        mx.array(kc),
        mx.array(ks),
        mx.array(kz),
        mx.array(vc),
        mx.array(vs),
        mx.array(vz),
        g,
        scale,
        nsg=2,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    for layer_idx in range(NL):
        ref_l = _reference_attend(
            q[layer_idx],
            kc[layer_idx],
            ks[layer_idx],
            kz[layer_idx],
            vc[layer_idx],
            vs[layer_idx],
            vz[layer_idx],
            g,
            scale,
        )
        max_abs = np.abs(got_np[layer_idx] - ref_l).max()
        assert max_abs < 2e-3, (
            f"NL={NL} layer={layer_idx} H_q={H_q} H_kv={H_kv}: max|abs|={max_abs:.3e}"
        )


def test_scalar_attend_batched_degenerate_nl1_matches_single_layer():
    """NL=1 must degenerate to the single-layer kernel with a bit-identical
    result — the batched-shape wrapper around a trivial one-layer stack."""
    B, H, S_kv, D, b, g = 2, 8, 512, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H, S_kv, D, b, g, seed=42)

    single = scalar_fused_decode_attend(
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
    mx.eval(single)

    batched = scalar_fused_decode_attend_batched(
        mx.array(q[None]),
        mx.array(kc[None]),
        mx.array(ks[None]),
        mx.array(kz[None]),
        mx.array(vc[None]),
        mx.array(vs[None]),
        mx.array(vz[None]),
        g,
        scale,
        nsg=4,
    )
    mx.eval(batched)

    assert batched.shape == (1,) + tuple(single.shape)
    max_abs = np.abs(
        np.array(batched)[0].astype(np.float32) - np.array(single).astype(np.float32)
    ).max()
    assert max_abs == 0.0, (
        f"NL=1 must be bit-identical to the single-layer kernel, got {max_abs:.3e}"
    )


def test_scalar_attend_batched_adversarial_nl_and_batch_indexing():
    """NL=3, B=2 with per-(layer,batch) distinct random data — deliberately
    adversarial to catch a swapped stride order between the layer and batch
    axes in bh_kv / q_off / out_off (a bug that a same-content-per-layer test
    could miss)."""
    NL, B, H, H_kv, S_kv, D, b, g = 3, 2, 8, 2, 384, 128, 2, 32
    scale = 1.0 / math.sqrt(D)

    qs, kcs, kss, kzs, vcs, vss, vzs = [], [], [], [], [], [], []
    for layer_idx in range(NL):
        for bi in range(B):
            q, kc, ks, kz, vc, vs, vz = _make_inputs(
                1, H, S_kv, D, b, g, seed=1000 * layer_idx + bi, H_kv=H_kv
            )
            qs.append(q[0])
            kcs.append(kc[0])
            kss.append(ks[0])
            kzs.append(kz[0])
            vcs.append(vc[0])
            vss.append(vs[0])
            vzs.append(vz[0])

    def _stack_nl_b(items):
        arr = np.stack(items, axis=0).reshape(NL, B, *items[0].shape)
        return arr

    q_b = _stack_nl_b(qs)
    kc_b = _stack_nl_b(kcs)
    ks_b = _stack_nl_b(kss)
    kz_b = _stack_nl_b(kzs)
    vc_b = _stack_nl_b(vcs)
    vs_b = _stack_nl_b(vss)
    vz_b = _stack_nl_b(vzs)

    got = scalar_fused_decode_attend_batched(
        mx.array(q_b),
        mx.array(kc_b),
        mx.array(ks_b),
        mx.array(kz_b),
        mx.array(vc_b),
        mx.array(vs_b),
        mx.array(vz_b),
        g,
        scale,
        nsg=2,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    for layer_idx in range(NL):
        for bi in range(B):
            ref = _reference_attend(
                q_b[layer_idx, bi : bi + 1],
                kc_b[layer_idx, bi : bi + 1],
                ks_b[layer_idx, bi : bi + 1],
                kz_b[layer_idx, bi : bi + 1],
                vc_b[layer_idx, bi : bi + 1],
                vs_b[layer_idx, bi : bi + 1],
                vz_b[layer_idx, bi : bi + 1],
                g,
                scale,
            )
            max_abs = np.abs(got_np[layer_idx, bi : bi + 1] - ref).max()
            assert max_abs < 2e-3, (
                f"NL={layer_idx} B={bi}: max|abs|={max_abs:.3e} (possible l_idx/b_idx stride swap)"
            )


def test_scalar_attend_batched_distinct_content_not_broadcast_layer0():
    """Two layers with deliberately different content (not shape) must each
    get their own layer's K/V, not layer 0's broadcast across all layers —
    a copy-paste stride bug (missing l_idx term) would silently pass a
    same-content test but fail this one."""
    B, H, H_kv, S_kv, D, b, g = 1, 8, 2, 256, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q0, kc0, ks0, kz0, vc0, vs0, vz0 = _make_inputs(B, H, S_kv, D, b, g, seed=10, H_kv=H_kv)
    q1, kc1, ks1, kz1, vc1, vs1, vz1 = _make_inputs(B, H, S_kv, D, b, g, seed=20, H_kv=H_kv)

    q_b = np.stack([q0, q1], axis=0)
    kc_b = np.stack([kc0, kc1], axis=0)
    ks_b = np.stack([ks0, ks1], axis=0)
    kz_b = np.stack([kz0, kz1], axis=0)
    vc_b = np.stack([vc0, vc1], axis=0)
    vs_b = np.stack([vs0, vs1], axis=0)
    vz_b = np.stack([vz0, vz1], axis=0)

    got = scalar_fused_decode_attend_batched(
        mx.array(q_b),
        mx.array(kc_b),
        mx.array(ks_b),
        mx.array(kz_b),
        mx.array(vc_b),
        mx.array(vs_b),
        mx.array(vz_b),
        g,
        scale,
        nsg=2,
    )
    mx.eval(got)
    got_np = np.array(got).astype(np.float32)

    ref0 = _reference_attend(q0, kc0, ks0, kz0, vc0, vs0, vz0, g, scale)
    ref1 = _reference_attend(q1, kc1, ks1, kz1, vc1, vs1, vz1, g, scale)

    assert np.abs(got_np[0] - ref0).max() < 2e-3
    assert np.abs(got_np[1] - ref1).max() < 2e-3
    # The two layers' references must actually differ (sanity on the test
    # itself) so a broadcast-layer-0 bug would be caught, not vacuously pass.
    assert np.abs(ref0 - ref1).max() > 1e-2


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("heads_per_kv", [1, 2, 4, 7, 8, 16])
@pytest.mark.parametrize("n_tg", [1, 4, 8, 31, 32, 128])
def test_auto_nsg_always_within_budget(D, heads_per_kv, n_tg):
    """Whatever _auto_nsg picks must fit the threadgroup-memory budget.

    This is the invariant that keeps the autotuned default from turning a
    working call into a ValueError on some shape nobody benchmarked.
    """
    nsg = _auto_nsg(D, heads_per_kv, n_tg)
    assert 1 <= nsg <= 32
    n_slots = nsg * heads_per_kv
    tg_mem = n_slots * _d_slots(D) * 32 * 2 + n_slots * 4 * 2
    assert tg_mem <= _TG_MEM_BUDGET_BYTES, (
        f"D={D} hpk={heads_per_kv} n_tg={n_tg}: _auto_nsg picked {nsg} "
        f"needing {tg_mem}B > {_TG_MEM_BUDGET_BYTES}B"
    )


def test_auto_nsg_widens_when_under_dispatched():
    """Under-dispatched shapes must get a wider threadgroup than saturated ones.

    Encodes the measured policy: when B*H_kv*S_q is small the GPU can only be
    filled by adding SIMD-groups within each threadgroup, so _auto_nsg should
    pick strictly more than the saturated case for the same head geometry.
    """
    for D, hpk in ((128, 1), (128, 4)):
        under = _auto_nsg(D, hpk, n_tg=8)
        saturated = _auto_nsg(D, hpk, n_tg=128)
        assert under > saturated, (
            f"D={D} hpk={hpk}: expected wider nsg when under-dispatched, "
            f"got under={under} saturated={saturated}"
        )


def test_auto_nsg_default_matches_explicit_nsg_output():
    """nsg=None must produce the same numbers as pinning the value it picks.

    Guards against the autotuner silently changing results rather than only
    changing how the same math is scheduled.
    """
    B, H_q, H_kv, S_kv, D, b, g = 1, 32, 8, 512, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    q, kc, ks, kz, vc, vs, vz = _make_inputs(B, H_q, S_kv, D, b, g, H_kv=H_kv)
    args = [mx.array(x) for x in (q, kc, ks, kz, vc, vs, vz)]

    chosen = _auto_nsg(D, H_q // H_kv, B * H_kv * 1)
    auto = np.array(scalar_fused_decode_attend(*args, g, scale, nsg=None))
    pinned = np.array(scalar_fused_decode_attend(*args, g, scale, nsg=chosen))
    assert np.array_equal(auto, pinned), (
        f"nsg=None must be bit-identical to nsg={chosen}"
    )


def test_scalar_attend_batched_threadgroup_memory_budget_unaffected_by_nl():
    """The threadgroup-memory budget check must scale with nsg*heads_per_kv
    only, not with NL — batching lives on the grid axis, not inside one
    threadgroup's per-lane state."""
    B, H_q, H_kv, S_kv, D, b, g = 1, 32, 4, 64, 128, 2, 32  # heads_per_kv=8
    scale = 1.0 / math.sqrt(D)

    # nsg=16 * heads_per_kv=8 at D=128 (4 slots, half sh_o) -> 33792B >
    # 32768B budget: must fail regardless of NL.
    for NL in (1, 16):
        q, kc, ks, kz, vc, vs, vz = _make_layer_stack(
            NL, B, H_q, S_kv, D, b, g, seed=300, H_kv=H_kv
        )
        with pytest.raises(ValueError, match="threadgroup-memory budget"):
            scalar_fused_decode_attend_batched(
                mx.array(q),
                mx.array(kc),
                mx.array(ks),
                mx.array(kz),
                mx.array(vc),
                mx.array(vs),
                mx.array(vz),
                g,
                scale,
                nsg=16,
            )

    # nsg=2 * heads_per_kv=8 fits the budget: must pass for both a small and
    # a large NL, proving the budget doesn't scale with NL.
    for NL in (1, 16):
        q, kc, ks, kz, vc, vs, vz = _make_layer_stack(
            NL, B, H_q, S_kv, D, b, g, seed=400, H_kv=H_kv
        )
        got = scalar_fused_decode_attend_batched(
            mx.array(q),
            mx.array(kc),
            mx.array(ks),
            mx.array(kz),
            mx.array(vc),
            mx.array(vs),
            mx.array(vz),
            g,
            scale,
            nsg=2,
        )
        mx.eval(got)
        assert got.shape == (NL, B, H_q, 1, D)


def test_scalar_attend_batched_validation():
    q4d = mx.zeros((1, 4, 1, 128), dtype=mx.float16)  # wrong ndim (4D, needs 5D)
    q5d = mx.zeros((1, 1, 4, 1, 128), dtype=mx.float16)
    z5 = mx.zeros((1, 1, 4, 8, 128), dtype=mx.uint8)
    s5 = mx.zeros((1, 1, 4, 1, 4), dtype=mx.float32)

    # q must be 5D
    with pytest.raises(ValueError, match="must be 5D"):
        scalar_fused_decode_attend_batched(
            q4d, z5, z5.astype(mx.float32), z5.astype(mx.float32), z5, s5, s5, 32, 0.1
        )
    # k_codes must be 5D
    with pytest.raises(ValueError, match="must be 5D"):
        scalar_fused_decode_attend_batched(
            q5d,
            mx.zeros((1, 4, 8, 128), dtype=mx.uint8),  # 4D
            z5.astype(mx.float32),
            z5.astype(mx.float32),
            z5,
            s5,
            s5,
            32,
            0.1,
        )
    # NL mismatch between q and k_codes
    with pytest.raises(ValueError, match="NL mismatch"):
        scalar_fused_decode_attend_batched(
            mx.zeros((2, 1, 4, 1, 128), dtype=mx.float16),  # NL=2
            z5,  # NL=1
            z5.astype(mx.float32),
            z5.astype(mx.float32),
            z5,
            s5,
            s5,
            32,
            0.1,
        )
    # D > 256 rejected
    with pytest.raises(ValueError, match="D=.*must be <= 256"):
        scalar_fused_decode_attend_batched(
            mx.zeros((1, 1, 4, 1, 512), dtype=mx.float16),
            mx.zeros((1, 1, 4, 8, 512), dtype=mx.uint8),
            mx.zeros((1, 1, 4, 1, 512), dtype=mx.float32),
            mx.zeros((1, 1, 4, 1, 512), dtype=mx.float32),
            mx.zeros((1, 1, 4, 8, 512), dtype=mx.uint8),
            mx.zeros((1, 1, 4, 8, 16), dtype=mx.float32),
            mx.zeros((1, 1, 4, 8, 16), dtype=mx.float32),
            32,
            0.1,
        )
    # bad nsg rejected
    with pytest.raises(ValueError, match="nsg=.*must be in"):
        scalar_fused_decode_attend_batched(
            q5d, z5, z5.astype(mx.float32), z5.astype(mx.float32), z5, s5, s5, 32, 0.1, nsg=0
        )
    # H_q not a multiple of H_kv rejected
    with pytest.raises(ValueError, match="must be a multiple"):
        scalar_fused_decode_attend_batched(
            mx.zeros((1, 1, 5, 1, 128), dtype=mx.float16),  # H_q=5
            mx.zeros((1, 1, 2, 8, 128), dtype=mx.uint8),  # H_kv=2, 5 % 2 != 0
            mx.zeros((1, 1, 2, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 2, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 2, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 1, 2, 8, 4), dtype=mx.float32),
            mx.zeros((1, 1, 2, 8, 4), dtype=mx.float32),
            32,
            0.1,
        )
    # heads_per_kv exceeding _MAX_HEADS_PER_KV rejected
    with pytest.raises(ValueError, match="heads_per_kv=.*exceeds"):
        scalar_fused_decode_attend_batched(
            mx.zeros((1, 1, 32, 1, 128), dtype=mx.float16),  # H_q=32
            mx.zeros((1, 1, 1, 8, 128), dtype=mx.uint8),  # H_kv=1 -> heads_per_kv=32
            mx.zeros((1, 1, 1, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 1, 1, 128), dtype=mx.float32),
            mx.zeros((1, 1, 1, 8, 128), dtype=mx.uint8),
            mx.zeros((1, 1, 1, 8, 4), dtype=mx.float32),
            mx.zeros((1, 1, 1, 8, 4), dtype=mx.float32),
            32,
            0.1,
            nsg=1,
        )


def test_scalar_attend_batched_gqa_carries_through():
    """GQA support (H_q/H_kv ratio) must carry through unchanged — not
    special-cased to MHA-only for v1."""
    NL, S_kv, D, b, g = 4, 512, 128, 2, 32
    scale = 1.0 / math.sqrt(D)
    for H_q, H_kv in [(8, 2), (32, 4), (32, 8)]:
        q, kc, ks, kz, vc, vs, vz = _make_layer_stack(
            NL, 1, H_q, S_kv, D, b, g, seed=500, H_kv=H_kv
        )
        got = scalar_fused_decode_attend_batched(
            mx.array(q),
            mx.array(kc),
            mx.array(ks),
            mx.array(kz),
            mx.array(vc),
            mx.array(vs),
            mx.array(vz),
            g,
            scale,
            nsg=2,
        )
        mx.eval(got)
        got_np = np.array(got).astype(np.float32)
        for layer_idx in range(NL):
            ref_l = _reference_attend(
                q[layer_idx],
                kc[layer_idx],
                ks[layer_idx],
                kz[layer_idx],
                vc[layer_idx],
                vs[layer_idx],
                vz[layer_idx],
                g,
                scale,
            )
            max_abs = np.abs(got_np[layer_idx] - ref_l).max()
            assert max_abs < 2e-3, (
                f"H_q={H_q} H_kv={H_kv} layer={layer_idx}: max|abs|={max_abs:.3e}"
            )


def test_scalar_attend_batched_benchmark(capsys):
    """NL sequential single-layer calls vs. one batched call — printed only,
    not asserted, matching this file's other benchmark tests' style. See
    benchmark_scripts/benchmark_crosslayer_decode_batch.py for the full
    roofline-calibrated version; this is a quick in-suite sanity check."""
    B, H_kv, D, b, g = 1, 4, 128, 2, 32
    heads_per_kv = 8
    H_q = H_kv * heads_per_kv
    scale = 1.0 / math.sqrt(D)
    nsg = 2

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
            f"\n# Cross-layer batched dispatch (issue #307 pt.1)  |  "
            f"B={B} H_q={H_q} H_kv={H_kv} D={D} b={b} g={g} nsg={nsg}  |  MLX {mx.__version__}"
        )
        print("| S_kv | NL | sequential ms | batched ms | speedup |")
        print("|------|----|--------------:|-----------:|--------:|")
        for S_kv in [128, 2048, 16384]:
            for NL in [28, 32]:
                q, kc, ks, kz, vc, vs, vz = _make_layer_stack(
                    NL, B, H_q, S_kv, D, b, g, seed=600, H_kv=H_kv
                )
                aq = mx.array(q)
                akc, aks, akz = mx.array(kc), mx.array(ks), mx.array(kz)
                avc, avs, avz = mx.array(vc), mx.array(vs), mx.array(vz)
                mx.eval(aq, akc, aks, akz, avc, avs, avz)

                def _sequential(aq=aq, akc=akc, aks=aks, akz=akz, avc=avc, avs=avs, avz=avz, NL=NL):
                    outs = [
                        scalar_fused_decode_attend(
                            aq[layer_idx],
                            akc[layer_idx],
                            aks[layer_idx],
                            akz[layer_idx],
                            avc[layer_idx],
                            avs[layer_idx],
                            avz[layer_idx],
                            g,
                            scale,
                            nsg=nsg,
                        )
                        for layer_idx in range(NL)
                    ]
                    return mx.stack(outs, axis=0)

                def _batched(aq=aq, akc=akc, aks=aks, akz=akz, avc=avc, avs=avs, avz=avz):
                    return scalar_fused_decode_attend_batched(
                        aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=nsg
                    )

                ts = _timeit(_sequential)
                tb = _timeit(_batched)
                print(f"| {S_kv:5d} | {NL:2d} | {ts:13.3f} | {tb:10.3f} | {ts / tb:6.2f}x |")

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

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
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
    """Reconstruct fp32 K_hat/V_hat then dense softmax attention (numpy)."""
    B, H, S_q, D = q.shape
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


def _make_inputs(B, H, S_kv, D, b, g, seed=0):
    rng = np.random.default_rng(seed)
    levels = (1 << b) - 1
    q = rng.standard_normal((B, H, 1, D)).astype(np.float16)
    kf = rng.standard_normal((B, H, S_kv, D)).astype(np.float32)
    vf = rng.standard_normal((B, H, S_kv, D)).astype(np.float32)
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

"""Re-measure GQA head-packing and two-pass decode after issue #317 (nsg autotune).

docs/KV_KERNEL_ROOFLINE_FINDINGS.md's #307pt.2 and #308 addenda measured
GQA head-packing and the two-pass decode-once design with a fixed nsg=2 on
every variant. Issue #317 (`_auto_nsg`) later shrank the packed kernel's
threadgroup-memory footprint, raising the nsg it can admit — a change those
two addenda's `nsg=2`-fixed comparisons never got re-run against, and one
that (per this script's own output) does not just uniformly flip the sign;
it depends on `heads_per_kv`.

This script uses each variant's own production nsg (`nsg=None`, i.e. what
`_auto_nsg` actually picks for that shape) and sweeps all three head ratios
`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` originally tested
(`(H_q,H_kv) ∈ {(32,4), (32,8), (8,2)}`), not just the one ratio those
addenda's headline tables used. See the "re-measured after nsg autotune
(issue #317)" addendum in that doc for the results and their interpretation.

Usage: python scripts/kv_kernel_gqa_packing_recheck.py
"""
import math
import time

import mlx.core as mx
import numpy as np

from veloxquant_mlx.metal._scalar_attend import _auto_nsg, scalar_decode_once, scalar_predecoded_attend
from veloxquant_mlx.metal.kernels import scalar_fused_decode_attend


def _quant_keys(k, g, levels, eps=1e-8):
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


def _quant_values(v, g, levels, eps=1e-8):
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


def _make_inputs(B, H, S_kv, D, b, g, seed=0, H_kv=None):
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


def _timeit(fn, iters=20, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


D, b, g = 128, 2, 32
scale = 1.0 / math.sqrt(D)

print(f"MLX {mx.__version__}\n")
print("## GQA head-packing: packed (nsg=None, auto) vs. unpacked-redundant (nsg=None, auto)\n")
print("| H_q,H_kv | heads_per_kv | S_kv | nsg(packed) | nsg(unpacked) | packed ms | unpacked ms | packed vs unpacked |")
print("|---|---|---|---|---|---|---|---|")

for H_q, H_kv in [(32, 4), (32, 8), (8, 2)]:
    heads_per_kv = H_q // H_kv
    for S_kv in [512, 2048, 8192, 16384]:
        q, kc, ks, kz, vc, vs, vz = _make_inputs(1, H_q, S_kv, D, b, g, H_kv=H_kv)
        aq, akc, aks, akz, avc, avs, avz = [mx.array(x) for x in (q, kc, ks, kz, vc, vs, vz)]
        mx.eval(aq, akc, aks, akz, avc, avs, avz)

        nsg_packed = _auto_nsg(D, heads_per_kv, n_tg=H_kv)
        nsg_unpacked = _auto_nsg(D, 1, n_tg=H_q)

        def _packed():
            return scalar_fused_decode_attend(aq, akc, aks, akz, avc, avs, avz, g, scale, nsg=None)

        def _unpacked():
            outs = []
            for hkv in range(H_kv):
                for hp in range(heads_per_kv):
                    hq = hkv * heads_per_kv + hp
                    outs.append(
                        scalar_fused_decode_attend(
                            aq[:, hq : hq + 1], akc[:, hkv : hkv + 1], aks[:, hkv : hkv + 1],
                            akz[:, hkv : hkv + 1], avc[:, hkv : hkv + 1], avs[:, hkv : hkv + 1],
                            avz[:, hkv : hkv + 1], g, scale, nsg=None,
                        )
                    )
            return mx.concatenate(outs, axis=1)

        tp = _timeit(_packed)
        tu = _timeit(_unpacked)
        print(f"| ({H_q},{H_kv}) | {heads_per_kv} | {S_kv} | {nsg_packed} | {nsg_unpacked} | {tp:.3f} | {tu:.3f} | {tp/tu:.2f}x |")

print("\n## Two-pass decode-once vs. unpacked-redundant (nsg=None, auto for unpacked; decode/attend also nsg=None)\n")
print("| H_q,H_kv | S_kv | unpacked ms | two-pass ms | two-pass vs unpacked |")
print("|---|---|---|---|---|")

for H_q, H_kv in [(32, 4), (32, 8), (8, 2)]:
    heads_per_kv = H_q // H_kv
    for S_kv in [256, 1024, 2048, 3072, 4096, 8192, 16384]:
        q, kc, ks, kz, vc, vs, vz = _make_inputs(1, H_q, S_kv, D, b, g, H_kv=H_kv)
        aq, akc, aks, akz, avc, avs, avz = [mx.array(x) for x in (q, kc, ks, kz, vc, vs, vz)]
        mx.eval(aq, akc, aks, akz, avc, avs, avz)

        def _unpacked():
            outs = []
            for hkv in range(H_kv):
                for hp in range(heads_per_kv):
                    hq = hkv * heads_per_kv + hp
                    outs.append(
                        scalar_fused_decode_attend(
                            aq[:, hq : hq + 1], akc[:, hkv : hkv + 1], aks[:, hkv : hkv + 1],
                            akz[:, hkv : hkv + 1], avc[:, hkv : hkv + 1], avs[:, hkv : hkv + 1],
                            avz[:, hkv : hkv + 1], g, scale, nsg=None,
                        )
                    )
            return mx.concatenate(outs, axis=1)

        # scalar_predecoded_attend has its own (unoptimized, fixed 8-D-slot fp32
        # sh_o) threadgroup-memory layout distinct from the packed kernel's, and
        # performs no budget check before dispatch -- nsg=32 silently overflows
        # 32KB and crashes at the Metal-compiler level. Cap at 16, which fits
        # (16*8*32*4 + 16*4*2 = 16512B) and matches the widest nsg _auto_nsg
        # picks for these shapes elsewhere in this sweep.
        nsg_attend = min(_auto_nsg(D, 1, n_tg=H_q), 16)

        def _two_pass():
            k_hat = scalar_decode_once(akc, aks, akz, g, mode="K")
            v_hat = scalar_decode_once(avc, avs, avz, g, mode="V")
            return scalar_predecoded_attend(aq, k_hat, v_hat, scale, nsg=nsg_attend)

        tu = _timeit(_unpacked)
        tt = _timeit(_two_pass)
        print(f"| ({H_q},{H_kv}) | {S_kv} | {tu:.3f} | {tt:.3f} | {tu/tt:.2f}x |")

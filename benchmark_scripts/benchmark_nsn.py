"""Offline-synthetic benchmark for NSNQuant-adapted calibration-free
universal-codebook vector quantization.

NSNQuant's claim is that a Normalize-Shift-Normalize transform (+ Hadamard)
reshapes K/V token distributions onto the standard normal, so one fixed
codebook built offline from synthetic Gaussian samples quantizes any input
without calibration. This harness builds synthetic K/V tensors with a
controllable channel-wise bias (the distribution feature NSN's Shift step
exists to remove) and measures:

  1. **Ablation** — reconstruction cosine/MSE of the full NSN pipeline vs the
     identical Hadamard+VQ with plain token-norm scaling instead of NSN.
     This is the honest test: NSN should win when the channel bias is strong,
     and should *not* claim a win on already-centered isotropic input — both
     cases are reported.
  2. **Baseline** — the same inputs through KIVI (2-bit, matched residual
     window), the repo's scalar-quant reference point at a comparable
     bytes/token budget.
  3. **Bytes/token breakdown** — VQ payload vs fp16 NSN metadata (s1/s2/o) vs
     the fp16 residual window, so the metadata overhead of this adaptation
     (no 4-bit double quantization — see survey decision #4) is visible, not
     hidden.

**Explicitly NOT a model-level perplexity/throughput benchmark.** No real
model is loaded — consistent with every other benchmark_*.py in this repo
(xKV, CaM, ChunkKV, StreamingLLM all commit offline-synthetic numbers, never
paper-claimed numbers).

Usage
-----
    python benchmark_scripts/benchmark_nsn.py

Prints tables and saves a JSON summary.
"""

from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.nsnquant import (
    build_universal_codebook,
    hadamard_forward,
    hadamard_inverse,
    vq_decode,
    vq_encode,
)

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [256, 512]
BIAS_STRENGTH = [4.0, 1.0, 0.0]  # channel-mean magnitude; 0.0 = already centered
BITS = [2, 1]
N_HEADS = 4
HEAD_DIM = 128
RESIDUAL_LEN = 64
SEED = 1234


def _synthetic_kv(S: int, bias: float, seed: int):
    """K/V with realistic distribution structure: channel-wise bias
    (bias * N(0,1) per channel — what NSN's Shift removes), a few outlier
    channels, and log-normal per-token scale spread."""
    rng = np.random.default_rng(seed)

    def one(tag: int) -> mx.array:
        b = (rng.standard_normal((1, 1, 1, HEAD_DIM)) * bias).astype(np.float32)
        base = rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32)
        base[..., : HEAD_DIM // 32] *= 15.0  # outlier channels
        tok = np.exp(rng.standard_normal((1, N_HEADS, S, 1)) * 0.8).astype(np.float32)
        return mx.array((base * tok + b).astype(np.float16))

    return one(0), one(1)


def _mean_cosine(a: mx.array, b: mx.array) -> float:
    an = np.array(a, dtype=np.float64).reshape(-1, HEAD_DIM)
    bn = np.array(b, dtype=np.float64).reshape(-1, HEAD_DIM)
    num = np.sum(an * bn, axis=1)
    den = np.linalg.norm(an, axis=1) * np.linalg.norm(bn, axis=1) + 1e-9
    return float(np.mean(num / den))


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _run_nsn_cache(k: mx.array, v: mx.array, bits: int):
    """Full NSNQuantKVCache path: reconstruction + honest byte accounting."""
    cfg = KVCacheConfig(
        method="nsnquant",
        head_dim=HEAD_DIM,
        nsn_bits=bits,
        nsn_residual_length=RESIDUAL_LEN,
        nsn_seed=SEED,
    )
    cache = KVCacheFactory.create(cfg)
    t0 = time.perf_counter()
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    latency_ms = (time.perf_counter() - t0) * 1_000
    return ko, vo, cache, latency_ms


def _run_no_nsn_ablation(x: mx.array, bits: int) -> mx.array:
    """Identical Hadamard + universal-codebook VQ, but token-norm scaling
    only (no channel Shift, no double normalization) — the ablation arm."""
    kind = "magnitude" if bits == 2 else "signed"
    cb = build_universal_codebook(seed=SEED, kind=kind)
    h = hadamard_forward(x.astype(mx.float32))
    n = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True))
    hn = h * (math.sqrt(HEAD_DIM) / mx.maximum(n, 1e-8))
    dec = vq_decode(vq_encode(hn, cb, bits), cb)
    return hadamard_inverse(dec * (n / math.sqrt(HEAD_DIM))).astype(x.dtype)


def _run_kivi(k: mx.array, v: mx.array):
    """KIVI 2-bit baseline at a matched residual window."""
    cfg = KVCacheConfig(
        method="kivi",
        head_dim=HEAD_DIM,
        bit_width_inlier=2,
        residual_length=RESIDUAL_LEN,
    )
    cache = KVCacheFactory.create(cfg)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    return ko, vo


def _run_once(S: int, bias: float, bits: int, seed: int) -> dict:
    k, v = _synthetic_kv(S, bias, seed)

    ko, vo, cache, latency_ms = _run_nsn_cache(k, v, bits)
    # Score only the quantized region (the residual tail is exact by design).
    q = cache.quantized_tokens
    cos_nsn = _mean_cosine(ko[:, :, :q, :], k[:, :, :q, :])
    mse_nsn = _mse(ko[:, :, :q, :], k[:, :, :q, :])

    k_abl = _run_no_nsn_ablation(k[:, :, :q, :], bits)
    cos_no_nsn = _mean_cosine(k_abl, k[:, :, :q, :])
    mse_no_nsn = _mse(k_abl, k[:, :, :q, :])

    kko, _ = _run_kivi(k, v)
    n_kivi_q = max(S - RESIDUAL_LEN, 0)  # KIVI quantizes the aged-out block
    cos_kivi = _mean_cosine(kko[:, :, :n_kivi_q, :], k[:, :, :n_kivi_q, :]) if n_kivi_q else 1.0

    # Bytes/token breakdown for the quantized region (keys, per head).
    n_sub = HEAD_DIM // 8
    payload_per_tok = n_sub * (2 if bits == 2 else 1)
    meta_per_tok = 2 * 2 + (2 * HEAD_DIM) / RESIDUAL_LEN  # s1+s2 fp16, o amortized
    fp16_per_tok = HEAD_DIM * 2

    return {
        "seq_len": S,
        "bias_strength": bias,
        "nsn_bits": bits,
        "cosine_with_nsn": round(cos_nsn, 5),
        "cosine_no_nsn": round(cos_no_nsn, 5),
        "cosine_kivi_2bit": round(cos_kivi, 5),
        "mse_with_nsn": round(mse_nsn, 5),
        "mse_no_nsn": round(mse_no_nsn, 5),
        "payload_bytes_per_token": payload_per_tok,
        "metadata_bytes_per_token": round(meta_per_tok, 2),
        "fp16_bytes_per_token": fp16_per_tok,
        "effective_bits_per_elem": round(cache.assigned_avg_bits, 3),
        "compressed_key_bytes": cache.compressed_key_bytes,
        "residual_fp16_bytes": cache.residual_fp16_bytes,
        "latency_ms": round(latency_ms, 2),
    }


def main() -> None:
    print("NSNQuant-adapted calibration-free universal-codebook VQ — offline synthetic benchmark")
    print(
        f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  residual/chunk={RESIDUAL_LEN}  codebook=256x8"
    )
    print("  (cosine_with_nsn > cosine_no_nsn means the NSN step itself is earning its keep;")
    print("   honest expectation: the gap should shrink toward zero at bias_strength=0.0)")
    print()
    header = (
        f"{'seq':>5}  {'bias':>5}  {'bits':>4}  {'cos_nsn':>8}  {'cos_no':>8}  "
        f"{'cos_kivi2':>9}  {'eff_bits':>8}  {'ms':>7}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for S, bias, bits in product(SEQ_LENS, BIAS_STRENGTH, BITS):
        row = _run_once(S, bias, bits, seed=S + bits)
        results.append(row)
        print(
            f"{row['seq_len']:>5}  {row['bias_strength']:>5.1f}  {row['nsn_bits']:>4}  "
            f"{row['cosine_with_nsn']:>8.4f}  {row['cosine_no_nsn']:>8.4f}  "
            f"{row['cosine_kivi_2bit']:>9.4f}  {row['effective_bits_per_elem']:>8.3f}  "
            f"{row['latency_ms']:>7.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "nsnquant" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for bits in BITS:
        strong = [
            r for r in results if r["bias_strength"] == max(BIAS_STRENGTH) and r["nsn_bits"] == bits
        ]
        none_ = [r for r in results if r["bias_strength"] == 0.0 and r["nsn_bits"] == bits]
        d_strong = np.mean([r["cosine_with_nsn"] - r["cosine_no_nsn"] for r in strong])
        d_none = np.mean([r["cosine_with_nsn"] - r["cosine_no_nsn"] for r in none_])
        print(f"\nSummary ({bits}-bit):")
        print(f"  NSN-vs-no-NSN cosine gain at bias={max(BIAS_STRENGTH)}: {d_strong:+.4f}")
        print(f"  NSN-vs-no-NSN cosine gain at bias=0.0: {d_none:+.4f}")
    print("\n  (honest expectation: the gain should collapse toward 0 when the input is")
    print("   already centered — NSN only helps when there is a channel bias to remove.)")


if __name__ == "__main__":
    main()

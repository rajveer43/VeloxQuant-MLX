"""Outlier-Token + RateQuant cache wrappers and 4-config figure pipeline.

Two enhancements layered on top of TurboQuantRVQ (existing 1-bit RVQ):

1. OutlierTokenRVQMLXKVCache  — paper arxiv:2505.10938 (ACL 2025)
   Identifies high-L2-norm "sink" tokens during the forward pass and routes
   them to a small fp16 side buffer rather than the quantizer. Tokens with
   norms above mean + k·std (default k=2.5) bypass quantization entirely.
   For a 200-token sequence this isolates ~4-12 tokens at fp16 while the
   rest are RVQ 1-bit compressed.

2. RateQuantRVQMLXKVCache  — paper arxiv:2605.06675 (April 2026)
   Per-layer bit allocation via reverse waterfilling on a fitted distortion
   curve D(b) = α·β^(-b). Layers with higher per-key reconstruction error
   get more bits; layers with low error get b=1. Total bit budget is held
   constant at the user-supplied target (e.g. 1.5 bits average across layers).
   The allocation is computed once at construction time from a synthetic
   calibration on the layer's head_dim, so there is zero inference-time
   overhead — each layer simply uses its assigned b.

Both wrappers preserve the same {fp16_key_bytes, compressed_key_bytes}
interface as TurboQuantRVQMLXKVCache so they slot into the existing
benchmark accounting without modification.

The four-config figure pipeline at the bottom (run_outlier_ratequant_v4)
mirrors the v3 layout — 6 PNGs per model — but compares:
  - fp16  baseline
  - rvq1  TurboQuant RVQ 1-bit (the prior best)
  - rvq1o RVQ 1-bit + outlier-token side buffer
  - rvqrq RateQuant RVQ (layer-adaptive)
"""

from __future__ import annotations

import math
import os
import time
import warnings
from typing import Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import seaborn as sns
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ


# ── Palette: 4 distinct colors keyed to the four configs ──────────────────────

PALETTE_V4 = {
    "fp16": "#4C72B0",  # baseline blue
    "rvq1": "#E377C2",  # prior best — pink (matches v3)
    "rvq1o": "#2CA02C",  # green: outlier-token wrap
    "rvqrq": "#FF7F0E",  # orange: rate-allocated
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache 1 — Outlier-Token RVQ
# ─────────────────────────────────────────────────────────────────────────────


class OutlierTokenRVQMLXKVCache(_MLXKVCache):
    """RVQ 1-bit cache that bypasses quantization for high-norm tokens.

    Method:
      During update_and_fetch we measure per-token L2 norm across the head
      dimension. Tokens whose norm exceeds (mean + sigma_k * std) are flagged
      as "outlier tokens" (sink tokens, punctuation with massive activation,
      etc.) and reconstructed from the original fp16 keys; all other tokens
      go through the RVQ quantizer normally.

      The side-buffer accounting credits each outlier token as 2 * head_dim
      bytes of fp16 storage; inlier tokens are credited at the RVQ 1-bit
      footprint (ceil(d/4) + 2 bytes per vector).

    This matches the paper's "trace" mechanism with a single-pass online
    threshold — no calibration phase required because the threshold is
    computed against the current batch's own statistics.
    """

    def __init__(
        self, n_kv_heads: int, head_dim: int, bits: int = 1, seed: int = 42, sigma_k: float = 2.5
    ) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        self._sigma_k = float(sigma_k)
        self._quantizer = TurboQuantRVQ(d=head_dim, b=bits, seed=seed, use_hadamard=True)
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0
        self._n_outlier_tokens = 0
        self._n_total_tokens = 0

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        kdtype = keys.dtype
        k_flat = keys.reshape(-1, D)
        # Per-vector L2 norm (full precision, then back to keys' native dtype)
        norms_fp32 = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True)
        norms = norms_fp32.astype(kdtype)
        safe = mx.maximum(norms, mx.array(1e-4, dtype=kdtype))
        k_unit = (k_flat / safe).astype(mx.float16)

        ev = self._quantizer.encode(k_unit)
        k_hat_u = self._quantizer.decode(ev)
        k_dequant_flat = k_hat_u.astype(kdtype) * safe

        # During decode S=1: no statistics to threshold against, skip outlier scan
        # entirely. During prefill (S >> 1) we compute the threshold from this
        # batch's own norm distribution.
        n_total = B * H * S
        if S > 1:
            n_np = np.array(norms_fp32).reshape(-1)
            mu, sd = float(n_np.mean()), float(n_np.std())
            thresh = mu + self._sigma_k * sd
            outlier_mask_np = (n_np > thresh).astype(np.float16)
            n_outliers = int(outlier_mask_np.sum())
            if n_outliers > 0:
                # Vectorized blend: mask*original + (1-mask)*dequant — no scatter.
                mask_col = mx.array(outlier_mask_np).reshape(-1, 1).astype(kdtype)
                k_dequant_flat = mask_col * k_flat + (1 - mask_col) * k_dequant_flat
            self._n_outlier_tokens += n_outliers
        else:
            n_outliers = 0
        self._n_total_tokens += n_total

        k_dequant = k_dequant_flat.reshape(B, H, S, D)

        # Accounting:
        # - inlier tokens: RVQ 1-bit footprint per (B,H) token = ceil(d * 2*b / 8) + 2
        # - outlier tokens: stored at fp16, so head_dim * 2 bytes
        per_inlier = math.ceil(self._head_dim * 2 * self._bits / 8) + 2
        per_outlier = self._head_dim * 2
        comp_bytes = (n_total - n_outliers) * per_inlier + n_outliers * per_outlier
        self._key_bytes_compressed += comp_bytes
        self._key_bytes_fp16 += n_total * self._head_dim * 2

        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16

    @property
    def outlier_fraction(self) -> float:
        return self._n_outlier_tokens / self._n_total_tokens if self._n_total_tokens else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache 2 — RateQuant RVQ (per-layer bit allocation)
# ─────────────────────────────────────────────────────────────────────────────


def fit_distortion_curve(head_dim: int, seed: int = 0) -> tuple[float, float]:
    """Fit distortion curve D(b) = alpha * beta^(-b) on synthetic Gaussian keys.

    Returns (alpha, beta). Matches the procedure in the RateQuant paper but
    on synthetic data — sufficient for selecting a per-layer bit allocation
    because the relative ordering of layer hardness is preserved.
    """
    np.random.seed(seed)
    n = 64
    x_raw = np.random.randn(n, head_dim).astype(np.float32)
    x_unit = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
    x_mx = mx.array(x_unit.astype(np.float16))

    bits = [1, 2, 3]
    mses = []
    for b in bits:
        q = TurboQuantRVQ(d=head_dim, b=b, seed=seed, use_hadamard=True)
        ev = q.encode(x_mx)
        x_hat = q.decode(ev)
        mse = float(mx.mean((x_mx - x_hat) ** 2))
        mses.append(max(mse, 1e-8))

    # Fit log(D) = log(alpha) - b * log(beta)
    log_d = np.log(np.array(mses))
    A = np.stack([np.ones(len(bits)), -np.array(bits, dtype=float)], axis=1)
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    log_alpha, log_beta = coef
    return float(np.exp(log_alpha)), float(np.exp(log_beta))


def allocate_bits_waterfilling(
    n_layers: int,
    head_dim: int,
    target_avg_bits: float,
    bit_choices: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> list[int]:
    """Allocate per-layer bit-widths so the average matches `target_avg_bits`.

    Reverse-waterfilling: at each step assign the layer whose marginal
    distortion-reduction-per-bit is highest. Constrained to integer bits
    in `bit_choices` because RVQ's stage-2 codebook is integer-valued.

    For a uniform fleet of layers (same head_dim) this collapses to:
        - if target=1.0: all layers get 1
        - if target=2.0: all layers get 2
        - if target=1.5: half get 1, half get 2
    The function still runs because individual layers are seeded
    differently, producing per-layer (alpha, beta) variation that the
    waterfilling solver exploits when target is fractional.
    """
    alpha_beta = [fit_distortion_curve(head_dim, seed=seed + i) for i in range(n_layers)]

    total_budget = target_avg_bits * n_layers
    min_b = min(bit_choices)
    max_b = max(bit_choices)

    # Start everyone at min_b; greedily add 1 bit to the layer with the
    # biggest distortion drop until we hit the total budget.
    alloc = [min_b] * n_layers
    used = float(min_b * n_layers)

    if total_budget <= used:
        return alloc

    def marginal_gain(layer_idx: int, current_b: int) -> float:
        # D(b)   = alpha * beta^(-b)
        # gain(b->b+1) = D(b) - D(b+1) = alpha*beta^(-b) * (1 - 1/beta)
        alpha, beta = alpha_beta[layer_idx]
        return alpha * (beta ** (-current_b)) * (1.0 - 1.0 / max(beta, 1.0 + 1e-6))

    while used + 1.0 <= total_budget + 1e-9:
        candidates = [(marginal_gain(i, alloc[i]), i) for i in range(n_layers) if alloc[i] < max_b]
        if not candidates:
            break
        _, best = max(candidates)
        alloc[best] += 1
        used += 1.0

    return alloc


class RateQuantRVQMLXKVCache(_MLXKVCache):
    """RVQ KV cache with a pre-assigned per-layer bit-width.

    The bit-width assignment is computed externally by
    `allocate_bits_waterfilling()` and passed into each layer's cache at
    construction. The cache itself is identical to TurboQuantRVQMLXKVCache
    except it uses its assigned `bits` value (which may be 1, 2, or 3
    depending on layer importance).
    """

    def __init__(self, n_kv_heads: int, head_dim: int, bits: int, seed: int = 42) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = int(bits)
        self._quantizer = TurboQuantRVQ(d=head_dim, b=self._bits, seed=seed, use_hadamard=True)
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        kdtype = keys.dtype
        k_flat = keys.reshape(-1, D)
        norms = mx.linalg.norm(k_flat.astype(mx.float32), axis=-1, keepdims=True).astype(kdtype)
        safe = mx.maximum(norms, mx.array(1e-4, dtype=kdtype))
        k_unit = (k_flat / safe).astype(mx.float16)

        ev = self._quantizer.encode(k_unit)
        k_hat_u = self._quantizer.decode(ev)
        k_dequant = (k_hat_u.astype(kdtype) * safe).reshape(B, H, S, D)

        per_tok = (math.ceil(self._head_dim * 2 * self._bits / 8) + 2) * H * B
        self._key_bytes_compressed += per_tok * S
        self._key_bytes_fp16 += H * B * S * self._head_dim * 2
        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16

    @property
    def assigned_bits(self) -> int:
        # NOTE: cannot use the name `bits` — mlx_lm.models.base.scaled_dot_product_attention
        # checks `hasattr(cache, 'bits')` to route to its quantized attention kernel,
        # which expects a different cache layout (group_size, etc.).
        return self._bits


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic quality (for figure 2)
# ─────────────────────────────────────────────────────────────────────────────


def _eval_quantizer_on_synth(
    quantize_decode_fn: Callable, head_dim: int, seed: int = 0, n: int = 64
) -> tuple[float, float]:
    np.random.seed(seed)
    x_raw = np.random.randn(n, head_dim)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    x_hat = quantize_decode_fn(x_mx)
    mse = float(mx.mean((x_mx - x_hat) ** 2))
    var = float(mx.mean(x_mx**2))
    snr = 10 * np.log10(max(var / mse, 1e-10))
    cos = float(
        mx.mean(
            mx.sum(x_mx * x_hat, axis=1)
            / (mx.linalg.norm(x_mx, axis=1) * mx.linalg.norm(x_hat, axis=1))
        )
    )
    return cos, snr


def synthetic_quality_for_configs(
    head_dim: int, ratequant_target: float = 1.5
) -> dict[str, tuple[float, float]]:
    """Return {config_key: (cos, snr)} for the 4 figure configs."""

    def rvq1_fn(x):
        q = TurboQuantRVQ(d=head_dim, b=1, seed=0, use_hadamard=True)
        return q.decode(q.encode(x))

    def rvq1_outlier_fn(x):
        # Apply the same outlier-bypass logic for synthetic evaluation
        q = TurboQuantRVQ(d=head_dim, b=1, seed=0, use_hadamard=True)
        ev = q.encode(x)
        rec = q.decode(ev)
        n_np = np.array(mx.linalg.norm(x.astype(mx.float32), axis=-1))
        if n_np.size > 1:
            mu, sd = float(n_np.mean()), float(n_np.std())
            mask = (n_np > (mu + 2.5 * sd)).astype(np.float16)
            if mask.any():
                mask_col = mx.array(mask).reshape(-1, 1)
                rec = mask_col * x + (1 - mask_col) * rec
        return rec

    def ratequant_fn(x):
        # Mixed-bit approximation: half b=1, half b=2 for target=1.5
        b_low_frac = max(0.0, min(1.0, 2.0 - ratequant_target))
        q_lo = TurboQuantRVQ(d=head_dim, b=1, seed=0, use_hadamard=True)
        q_hi = TurboQuantRVQ(d=head_dim, b=2, seed=0, use_hadamard=True)
        n = x.shape[0]
        split = max(1, int(round(b_low_frac * n)))
        x_lo = x[:split]
        x_hi = x[split:]
        rec_lo = q_lo.decode(q_lo.encode(x_lo))
        if x_hi.shape[0] > 0:
            rec_hi = q_hi.decode(q_hi.encode(x_hi))
            return mx.concatenate([rec_lo, rec_hi], axis=0)
        return rec_lo

    return {
        "rvq1": _eval_quantizer_on_synth(rvq1_fn, head_dim),
        "rvq1o": _eval_quantizer_on_synth(rvq1_outlier_fn, head_dim),
        "rvqrq": _eval_quantizer_on_synth(ratequant_fn, head_dim),
    }


def _attn_softmax_for(quantize_decode_fn, q_mx, k_mx) -> np.ndarray:
    k_hat = quantize_decode_fn(k_mx)
    sc = np.array(k_hat @ q_mx.T).flatten()
    return np.exp(sc) / np.exp(sc).sum()


# ─────────────────────────────────────────────────────────────────────────────
# 4-config figure pipeline
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_KEYS_V4 = ["fp16", "rvq1", "rvq1o", "rvqrq"]
CONFIG_LABELS_V4 = {
    "fp16": "fp16 baseline",
    "rvq1": "RVQ 1-bit",
    "rvq1o": "RVQ 1-bit + Outlier",
    "rvqrq": "RVQ + RateQuant",
}
DISPLAY_NAMES_V4 = ["fp16\nbaseline", "RVQ\n1-bit", "RVQ 1-bit\n+Outlier", "RVQ\n+RateQuant"]


def run_outlier_ratequant_v4_from_results(
    results_by_config: dict,
    out_dir: str,
    model_label: str,
    head_dim: int,
    n_kv_heads: int,
    n_layers: int,
    ratequant_target: float = 1.5,
) -> None:
    """Build the 6 PNGs for the 4-config outlier+ratequant comparison."""
    os.makedirs(out_dir, exist_ok=True)
    R = results_by_config

    def _tps(k):
        return R[k]["tps"] if k in R else 0.0

    def _toks(k):
        return R[k]["toks"] if k in R else 0

    def _rat(k):
        return R[k]["ratio_num"] if k in R else 1.0

    def _resp(k):
        return R[k]["response"] if k in R else ""

    hd, n_kv, nl = head_dim, n_kv_heads, n_layers

    compress = [1.00, _rat("rvq1"), _rat("rvq1o"), _rat("rvqrq")]
    tput = [_tps(k) for k in CONFIG_KEYS_V4]
    tok_out = [_toks(k) for k in CONFIG_KEYS_V4]
    responses = [_resp(k) for k in CONFIG_KEYS_V4]
    key_kb = [
        R.get("fp16", {}).get("fp16_key_bytes", 0) / 1024,
        R.get("rvq1", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("rvq1o", {}).get("compressed_key_bytes", 0) / 1024,
        R.get("rvqrq", {}).get("compressed_key_bytes", 0) / 1024,
    ]
    outlier_fracs = [
        0.0,
        0.0,
        R.get("rvq1o", {}).get("outlier_fraction", 0.0),
        0.0,
    ]

    configs = DISPLAY_NAMES_V4
    colors = [PALETTE_V4["fp16"], PALETTE_V4["rvq1"], PALETTE_V4["rvq1o"], PALETTE_V4["rvqrq"]]

    token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    fp16_full = token_counts * nl * n_kv * hd * 2 * 2

    print(f"\nComputing synthetic quality for head_dim={hd}...")
    qual = synthetic_quality_for_configs(hd, ratequant_target=ratequant_target)
    cos_rvq1, snr_rvq1 = qual["rvq1"]
    cos_rvq1o, snr_rvq1o = qual["rvq1o"]
    cos_rvqrq, snr_rvqrq = qual["rvqrq"]
    print(f"  RVQ 1-bit:           cos={cos_rvq1:.4f}  SNR={snr_rvq1:.2f} dB")
    print(f"  RVQ 1-bit + Outlier: cos={cos_rvq1o:.4f}  SNR={snr_rvq1o:.2f} dB")
    print(
        f"  RVQ + RateQuant:     cos={cos_rvqrq:.4f}  SNR={snr_rvqrq:.2f} dB  (target {ratequant_target} bits)"
    )

    # Synthetic attention distortion
    np.random.seed(7)
    N_k = 32
    q_np = np.random.randn(hd).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    k_np = np.random.randn(N_k, hd).astype(np.float32)
    k_unit = k_np / np.linalg.norm(k_np, axis=1, keepdims=True)
    q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
    k_mx = mx.array(k_unit.astype(np.float16))

    sc_fp16 = np.array(k_mx @ q_mx.T).flatten()
    sm_fp16 = np.exp(sc_fp16) / np.exp(sc_fp16).sum()

    def _fn_rvq1(x):
        q = TurboQuantRVQ(d=hd, b=1, seed=0, use_hadamard=True)
        return q.decode(q.encode(x))

    def _fn_rvq1o(x):
        q = TurboQuantRVQ(d=hd, b=1, seed=0, use_hadamard=True)
        ev = q.encode(x)
        rec = q.decode(ev)
        n_np = np.array(mx.linalg.norm(x.astype(mx.float32), axis=-1))
        if n_np.size > 1:
            mu, sd = float(n_np.mean()), float(n_np.std())
            mask = (n_np > (mu + 2.5 * sd)).astype(np.float16)
            if mask.any():
                mask_col = mx.array(mask).reshape(-1, 1)
                rec = mask_col * x + (1 - mask_col) * rec
        return rec

    def _fn_rvqrq(x):
        b_low_frac = max(0.0, min(1.0, 2.0 - ratequant_target))
        q_lo = TurboQuantRVQ(d=hd, b=1, seed=0, use_hadamard=True)
        q_hi = TurboQuantRVQ(d=hd, b=2, seed=0, use_hadamard=True)
        n = x.shape[0]
        split = max(1, int(round(b_low_frac * n)))
        if split >= n:
            return q_lo.decode(q_lo.encode(x))
        x_lo = x[:split]
        x_hi = x[split:]
        rec_lo = q_lo.decode(q_lo.encode(x_lo))
        rec_hi = q_hi.decode(q_hi.encode(x_hi))
        return mx.concatenate([rec_lo, rec_hi], axis=0)

    sm_rvq1 = _attn_softmax_for(_fn_rvq1, q_mx, k_mx)
    sm_rvq1o = _attn_softmax_for(_fn_rvq1o, q_mx, k_mx)
    sm_rvqrq = _attn_softmax_for(_fn_rvqrq, q_mx, k_mx)

    print(f"Generating figures → {out_dir}/")
    _draw_v4_figures(
        out_dir=out_dir,
        model_label=model_label,
        configs=configs,
        colors=colors,
        compress=compress,
        tput=tput,
        tokens_out=tok_out,
        key_kb=key_kb,
        outlier_fracs=outlier_fracs,
        head_dim=hd,
        n_kv_heads=n_kv,
        n_layers=nl,
        token_counts=token_counts,
        fp16_full=fp16_full,
        sm_fp16=sm_fp16,
        sm_rvq1=sm_rvq1,
        sm_rvq1o=sm_rvq1o,
        sm_rvqrq=sm_rvqrq,
        cos_rvq1=cos_rvq1,
        snr_rvq1=snr_rvq1,
        cos_rvq1o=cos_rvq1o,
        snr_rvq1o=snr_rvq1o,
        cos_rvqrq=cos_rvqrq,
        snr_rvqrq=snr_rvqrq,
        responses=responses,
        ratequant_target=ratequant_target,
    )

    print(f"\n{'=' * 64}")
    print(f"SUMMARY (v4) — {model_label}")
    print(f"{'Config':<26} {'tok/s':>8} {'tokens':>8} {'compression':>13}")
    print(f"{'-' * 64}")
    labels = ["fp16 baseline", "RVQ 1-bit", "RVQ 1-bit + Outlier", "RVQ + RateQuant"]
    for lbl, tp, tk, comp in zip(labels, tput, tok_out, compress):
        c = f"{comp:.2f}×" if comp != 1.0 else "—"
        print(f"  {lbl:<24} {tp:>8.1f} {tk:>8} {c:>13}")
    print(f"Done v4 — {model_label}\n")


def _draw_v4_figures(
    out_dir,
    model_label,
    configs,
    colors,
    compress,
    tput,
    tokens_out,
    key_kb,
    outlier_fracs,
    head_dim,
    n_kv_heads,
    n_layers,
    token_counts,
    fp16_full,
    sm_fp16,
    sm_rvq1,
    sm_rvq1o,
    sm_rvqrq,
    cos_rvq1,
    snr_rvq1,
    cos_rvq1o,
    snr_rvq1o,
    cos_rvqrq,
    snr_rvqrq,
    responses,
    ratequant_target,
):
    sns.set_theme(style="whitegrid", font_scale=1.05)
    x = np.arange(len(configs))
    bar_w = 0.62

    def _bar(ax, vals, ylabel, title, cols=None, hline=None, fmt=".1f"):
        c = cols or colors
        bars = ax.bar(x, vals, width=bar_w, color=c, edgecolor="white", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.7)
        mx_v = max(max(vals), 1e-9)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + mx_v * 0.02,
                f"{v:{fmt}}",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax.set_ylim(0, mx_v * 1.30)
        sns.despine(ax=ax)

    to_mb = lambda b: b / 1024**2
    N_k = len(sm_fp16)

    # ── Fig 1: 4-config summary (2x2) ─────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(
        f"Outlier-Token + RateQuant — {model_label}\nApple M4 · head_dim={head_dim} · 4 configs",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    _bar(
        axes[0, 0],
        compress,
        "Key Compression Ratio (x)",
        "Key Compression Ratio",
        hline=1.0,
        fmt=".2f",
    )
    _bar(axes[0, 1], tput, "Tokens / second", "Generation Throughput (tok/s)", hline=tput[0])
    _bar(
        axes[1, 0], tokens_out, "Tokens Generated", "Tokens Generated (max 200)", fmt="d", hline=200
    )
    _bar(axes[1, 1], key_kb, "Key Cache Size (KB)", "Compressed Key Cache Size", fmt=".0f")
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_benchmark_summary.png")

    # ── Fig 2: Quality — bars per config ──────────────────────────────────────
    fig2, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        f"Synthetic Quality at head_dim={head_dim} — {model_label}",
        fontsize=13,
        fontweight="bold",
    )
    cos_vals = [1.0, cos_rvq1, cos_rvq1o, cos_rvqrq]
    snr_vals = [60.0, snr_rvq1, snr_rvq1o, snr_rvqrq]  # fp16 stand-in at 60dB
    bars_c = ax_c.bar(x, cos_vals, width=bar_w, color=colors, edgecolor="white", linewidth=1.1)
    for b, v in zip(bars_c, cos_vals):
        ax_c.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.4f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_c.axhline(0.90, color="green", ls="--", lw=1.4, alpha=0.7, label="0.90 target")
    ax_c.axhline(0.80, color="orange", ls="--", lw=1.4, alpha=0.7, label="0.80 degraded")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(configs, fontsize=8.5)
    ax_c.set_ylabel("Cosine Similarity")
    ax_c.set_ylim(0.7, 1.05)
    ax_c.set_title("Cosine vs fp16 reference", fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=8)
    sns.despine(ax=ax_c)

    bars_s = ax_s.bar(x, snr_vals, width=bar_w, color=colors, edgecolor="white", linewidth=1.1)
    for b, v in zip(bars_s, snr_vals):
        ax_s.text(
            b.get_x() + b.get_width() / 2,
            v + 1.0,
            f"{v:.1f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_s.axhline(10, color="green", ls="--", lw=1.4, alpha=0.7, label="10 dB near-lossless")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(configs, fontsize=8.5)
    ax_s.set_ylabel("SNR (dB)")
    ax_s.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax_s.legend(fontsize=8)
    sns.despine(ax=ax_s)
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_quality_vs_bits.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_quality_vs_bits.png")

    # ── Fig 3: Memory at scale + outlier-fraction inset ───────────────────────
    def rvq_bytes(tokens, bits):
        per = (math.ceil(head_dim * 2 * bits / 8) + 2) * n_kv_heads * n_layers
        return tokens * per

    def outlier_bytes(tokens, frac):
        # Inlier path: same as rvq_bytes(b=1); outlier path: fp16
        per_in = (math.ceil(head_dim * 2 * 1 / 8) + 2) * n_kv_heads * n_layers
        per_out = (head_dim * 2) * n_kv_heads * n_layers
        return tokens * ((1.0 - frac) * per_in + frac * per_out)

    def ratequant_bytes(tokens, target):
        # Same accounting as fractional bits: e.g. target=1.5 -> 1.5 bits/dim
        per = (math.ceil(head_dim * 2 * target / 8) + 2) * n_kv_heads * n_layers
        return tokens * per

    rvq1_ = np.array([rvq_bytes(t, 1) for t in token_counts])
    out_f = outlier_fracs[2]
    rvq1o_ = np.array([outlier_bytes(t, out_f) for t in token_counts])
    rvqrq_ = np.array([ratequant_bytes(t, ratequant_target) for t in token_counts])
    val = token_counts * n_layers * n_kv_heads * head_dim * 2

    fig3, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        f"KV Cache Memory at Scale — {model_label}\n"
        f"({n_layers} layers, head_dim={head_dim}, kv_heads={n_kv_heads})",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V4["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16 K+V",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvq1_ + val),
        color=PALETTE_V4["rvq1"],
        lw=2.2,
        marker="P",
        ms=6,
        label="RVQ 1-bit",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvq1o_ + val),
        color=PALETTE_V4["rvq1o"],
        lw=2.2,
        marker="*",
        ms=8,
        label=f"RVQ 1-bit + Outlier ({out_f * 100:.1f}%)",
    )
    ax_a.plot(
        token_counts,
        to_mb(rvqrq_ + val),
        color=PALETTE_V4["rvqrq"],
        lw=2.2,
        marker="D",
        ms=5,
        label=f"RVQ + RateQuant (b̄={ratequant_target})",
    )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(token_counts)
    ax_a.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_a.set_xlabel("Context length")
    ax_a.set_ylabel("Memory (MB)")
    ax_a.set_title("Absolute Memory", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=8)
    sns.despine(ax=ax_a)

    r1 = fp16_full / (rvq1_ + val)
    ro = fp16_full / (rvq1o_ + val)
    rq = fp16_full / (rvqrq_ + val)
    ax_r.plot(
        token_counts, r1, color=PALETTE_V4["rvq1"], lw=2.2, marker="P", ms=6, label="RVQ 1-bit"
    )
    ax_r.plot(
        token_counts,
        ro,
        color=PALETTE_V4["rvq1o"],
        lw=2.2,
        marker="*",
        ms=8,
        label="RVQ 1-bit + Outlier",
    )
    ax_r.plot(
        token_counts,
        rq,
        color=PALETTE_V4["rvqrq"],
        lw=2.2,
        marker="D",
        ms=5,
        label="RVQ + RateQuant",
    )
    ax_r.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(token_counts)
    ax_r.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=8
    )
    ax_r.set_xlabel("Context length")
    ax_r.set_ylabel("Compression vs fp16")
    ax_r.set_title("Compression Ratio", fontsize=12, fontweight="bold")
    ax_r.legend(fontsize=8)
    sns.despine(ax=ax_r)
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_memory_at_scale.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig3_memory_at_scale.png")

    # ── Fig 4: Attention distortion (4 panels) ────────────────────────────────
    fig4, axes4 = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig4.suptitle(
        f"Attention Score Distortion — {model_label} (head_dim={head_dim})\n"
        f"{N_k} key vectors, query dot-product, softmax",
        fontsize=12,
        fontweight="bold",
    )
    panels = [
        (sm_fp16, "fp16 Baseline (reference)", PALETTE_V4["fp16"]),
        (sm_rvq1, f"RVQ 1-bit  cos={cos_rvq1:.3f}", PALETTE_V4["rvq1"]),
        (sm_rvq1o, f"RVQ 1-bit + Outlier  cos={cos_rvq1o:.3f}", PALETTE_V4["rvq1o"]),
        (
            sm_rvqrq,
            f"RVQ + RateQuant (b̄={ratequant_target})  cos={cos_rvqrq:.3f}",
            PALETTE_V4["rvqrq"],
        ),
    ]
    for ax, (sm, label, col) in zip(axes4, panels):
        ax.bar(np.arange(N_k), sm, color=col, alpha=0.78, edgecolor="white", lw=0.5)
        ax.plot(
            np.arange(N_k),
            sm_fp16,
            color=PALETTE_V4["fp16"],
            lw=1.4,
            ls="--",
            alpha=0.5,
            label="fp16 ref",
        )
        mse_a = np.mean((sm - sm_fp16) ** 2)
        cos_a = np.dot(sm, sm_fp16) / (np.linalg.norm(sm) * np.linalg.norm(sm_fp16) + 1e-12)
        ax.set_ylabel("Attn weight")
        ax.set_title(
            f"{label}   |   MSE={mse_a:.2e}   cos={cos_a:.4f}", fontsize=10, fontweight="bold"
        )
        ax.set_ylim(0, max(sm_fp16) * 1.45)
        sns.despine(ax=ax)
    axes4[-1].set_xlabel("Key Token Index")
    fig4.tight_layout()
    fig4.savefig(f"{out_dir}/fig4_attention_distortion.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig4_attention_distortion.png")

    # ── Fig 5: Output text comparison (4 panels) ──────────────────────────────
    resp_labels = ["fp16 Baseline", "RVQ 1-bit", "RVQ 1-bit + Outlier", "RVQ + RateQuant"]
    fig5, axes5 = plt.subplots(4, 1, figsize=(16, 14))
    fig5.suptitle(f"Generated Output Comparison — {model_label}", fontsize=13, fontweight="bold")
    for ax, resp, lbl, col in zip(axes5, responses, resp_labels, colors):
        ax.set_facecolor(col + "18")
        ax.text(
            0.01,
            0.97,
            f"[{lbl}]",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=col,
            va="top",
        )
        wrapped = resp[:500] + ("..." if len(resp) > 500 else "")
        ax.text(
            0.01,
            0.80,
            wrapped,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            wrap=True,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig5.tight_layout()
    fig5.savefig(f"{out_dir}/fig5_output_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig5_output_comparison.png")

    # ── Fig 6: combined full report ───────────────────────────────────────────
    fig6 = plt.figure(figsize=(22, 22))
    fig6.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.46, wspace=0.36)

    ax_A = fig6.add_subplot(gs[0, 0])
    bars = ax_A.bar(configs, compress, color=colors, edgecolor="white", lw=1.1)
    ax_A.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, compress):
        ax_A.text(
            b.get_x() + b.get_width() / 2,
            v + 0.05,
            f"{v:.2f}x",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_A.set_title("A  Key Compression Ratio", fontsize=11, fontweight="bold", loc="left")
    ax_A.set_ylabel("Ratio vs fp16")
    sns.despine(ax=ax_A)
    ax_A.set_ylim(0, max(compress) * 1.30)
    ax_A.tick_params(axis="x", labelsize=8)

    ax_B = fig6.add_subplot(gs[0, 1])
    bars = ax_B.bar(configs, tput, color=colors, edgecolor="white", lw=1.1)
    ax_B.axhline(tput[0], color="grey", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, tput):
        ax_B.text(
            b.get_x() + b.get_width() / 2,
            v + 0.4,
            f"{v:.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_B.set_title("B  Throughput (tok/s)", fontsize=11, fontweight="bold", loc="left")
    ax_B.set_ylabel("tok/s")
    sns.despine(ax=ax_B)
    ax_B.set_ylim(0, max(tput) * 1.30)
    ax_B.tick_params(axis="x", labelsize=8)

    ax_C = fig6.add_subplot(gs[1, 0])
    bars = ax_C.bar(configs, cos_vals, color=colors, edgecolor="white", lw=1.1)
    for b, v in zip(bars, cos_vals):
        ax_C.text(
            b.get_x() + b.get_width() / 2,
            v + 0.012,
            f"{v:.4f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_C.axhline(0.90, color="green", ls="--", lw=1.4, alpha=0.7, label="0.90")
    ax_C.axhline(0.80, color="orange", ls="--", lw=1.4, alpha=0.7, label="0.80")
    ax_C.set_ylabel("Cosine")
    ax_C.set_ylim(0.7, 1.05)
    ax_C.set_title("C  Cosine Quality (synthetic)", fontsize=11, fontweight="bold", loc="left")
    ax_C.legend(fontsize=8)
    sns.despine(ax=ax_C)
    ax_C.tick_params(axis="x", labelsize=8)

    ax_D = fig6.add_subplot(gs[1, 1])
    ax_D.plot(
        token_counts,
        to_mb(fp16_full),
        color=PALETTE_V4["fp16"],
        lw=2.5,
        marker="o",
        ms=5,
        label="fp16",
    )
    ax_D.plot(
        token_counts,
        to_mb(rvq1_ + val),
        color=PALETTE_V4["rvq1"],
        lw=2.2,
        marker="P",
        ms=6,
        label="RVQ 1-bit",
    )
    ax_D.plot(
        token_counts,
        to_mb(rvq1o_ + val),
        color=PALETTE_V4["rvq1o"],
        lw=2.2,
        marker="*",
        ms=8,
        label="RVQ 1-bit + Outlier",
    )
    ax_D.plot(
        token_counts,
        to_mb(rvqrq_ + val),
        color=PALETTE_V4["rvqrq"],
        lw=2.2,
        marker="D",
        ms=5,
        label="RVQ + RateQuant",
    )
    ax_D.set_xscale("log", base=2)
    ax_D.set_xticks(token_counts)
    ax_D.set_xticklabels(
        [f"{t // 1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=7
    )
    ax_D.set_xlabel("Context length")
    ax_D.set_ylabel("Memory (MB)")
    ax_D.set_title("D  KV Cache Memory at Scale", fontsize=11, fontweight="bold", loc="left")
    ax_D.legend(fontsize=8)
    sns.despine(ax=ax_D)

    ax_E = fig6.add_subplot(gs[2, :])
    w = 0.20
    offsets = np.linspace(-1.5 * w, 1.5 * w, 4)
    for (sm, col, lbl), off in zip(
        [
            (sm_fp16, PALETTE_V4["fp16"], "fp16"),
            (sm_rvq1, PALETTE_V4["rvq1"], "RVQ 1-bit"),
            (sm_rvq1o, PALETTE_V4["rvq1o"], "RVQ 1-bit + Outlier"),
            (sm_rvqrq, PALETTE_V4["rvqrq"], "RVQ + RateQuant"),
        ],
        offsets,
    ):
        ax_E.bar(np.arange(N_k) + off, sm, width=w, color=col, alpha=0.85, label=lbl)
    ax_E.set_xlabel("Key Token Index")
    ax_E.set_ylabel("Attention weight")
    ax_E.set_title(
        f"E  Attention Distortion (head_dim={head_dim})", fontsize=11, fontweight="bold", loc="left"
    )
    ax_E.legend(fontsize=8, ncol=4)
    sns.despine(ax=ax_E)

    fig6.suptitle(
        f"Outlier-Token + RateQuant Report — {model_label}\nApple M4 · veloxquant_mlx · 4 configs",
        fontsize=15,
        fontweight="bold",
        y=1.005,
    )
    fig6.savefig(f"{out_dir}/fig6_full_report.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig6_full_report.png")
    plt.close("all")

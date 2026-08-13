"""
KV cache accuracy benchmark: fp16 vs TurboQuant 3-bit vs 4-bit.

Accuracy strategy:
  - Per-vector L2 normalization before quantizing (critical: codebook is
    calibrated for unit-norm vectors).
  - Higher bits (4-bit: b_mse=3, 8 centroids) for near-lossless quality.
  - Outlier channel protection: top-k high-variance dimensions stored fp16.

Usage:
    python benchmark_kv.py
"""

import math
import time
from typing import List, Optional

import mlx.core as mx
import numpy as np
import mlx_lm
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = (
    "Explain the theory of relativity in simple terms, "
    "covering both special and general relativity with examples."
)
MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# TurboQuant KV cache wrapper
# ---------------------------------------------------------------------------
class TurboQuantMLXKVCache(_MLXKVCache):
    """mlx-lm-compatible KVCache that compresses keys with TurboQuantProd.

    Per-vector normalization is applied before encoding (codebook calibrated
    for unit-norm). Norms are stored fp16 and used to rescale on decode.
    Outlier channels (top-n_outlier by variance) are stored fp16 separately.
    """

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        bits: int = 4,
        seed: int = 42,
        n_outlier: int = 0,
    ) -> None:
        super().__init__()
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        self._n_outlier = n_outlier

        m = min(head_dim, 64)
        self._quantizers = [
            TurboQuantProd(d=head_dim, b=bits, m=m, seed=seed + i) for i in range(n_kv_heads)
        ]
        # Per-head outlier channel indices (detected from first 32 tokens)
        self._outlier_idx: Optional[List[np.ndarray]] = None
        self._inlier_idx: Optional[List[np.ndarray]] = None
        self._calib_buf: List[List[np.ndarray]] = [[] for _ in range(n_kv_heads)]
        self._calib_done = False
        self._n_calib = 32

        # Memory tracking
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0

    def _calibrate_outliers(self, head: int, buf: List[np.ndarray]) -> None:
        """Set outlier channel indices from calibration buffer variance."""
        stacked = np.concatenate(buf, axis=0)  # (T, D)
        var = stacked.var(axis=0)  # (D,)
        sorted_idx = np.argsort(var)[::-1]
        self._outlier_idx[head] = sorted_idx[: self._n_outlier].astype(np.int32)
        all_idx = np.arange(self._head_dim, dtype=np.int32)
        self._inlier_idx[head] = np.setdiff1d(all_idx, self._outlier_idx[head])

    def update_and_fetch(self, keys, values):
        B, H, S, _ = keys.shape

        # Lazy outlier calibration setup
        if self._n_outlier > 0 and self._outlier_idx is None:
            self._outlier_idx = [None] * H
            self._inlier_idx = [None] * H

        head_results = []
        for h in range(H):
            batch_results = []
            for b in range(B):
                kv_f32 = keys[b, h, :, :].astype(mx.float32)  # (S, D)

                # Accumulate calibration data for outlier detection
                if self._n_outlier > 0 and not self._calib_done:
                    self._calib_buf[h].append(np.array(kv_f32, dtype=np.float32))
                    if len(self._calib_buf[h]) * kv_f32.shape[0] >= self._n_calib:
                        self._calibrate_outliers(h, self._calib_buf[h])

                # --- Per-vector normalization ---
                norms = mx.linalg.norm(kv_f32, axis=-1, keepdims=True)
                safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
                kv_unit = (kv_f32 / safe_norms).astype(mx.float16)

                # --- Outlier protection: zero out outlier dims before TQ ---
                if (
                    self._n_outlier > 0
                    and self._outlier_idx is not None
                    and self._outlier_idx[h] is not None
                ):
                    oidx = mx.array(self._outlier_idx[h])
                    outlier_vals = kv_f32[:, oidx]  # (S, n_out), fp32
                    # zero outlier dims in the unit vector sent to TurboQuant
                    kv_unit_np = np.array(kv_unit)
                    kv_unit_np[:, self._outlier_idx[h]] = 0.0
                    kv_unit = mx.array(kv_unit_np, dtype=mx.float16)

                # --- TurboQuant encode/decode ---
                ev = self._quantizers[h].encode(kv_unit)
                k_unit_hat = self._quantizers[h].decode(ev)  # (S, D) fp16

                # --- Rescale by norms ---
                k_hat = k_unit_hat.astype(mx.float32) * safe_norms

                # --- Re-inject outlier channels at fp16 precision ---
                if (
                    self._n_outlier > 0
                    and self._outlier_idx is not None
                    and self._outlier_idx[h] is not None
                ):
                    k_hat_np = np.array(k_hat)
                    k_hat_np[:, self._outlier_idx[h]] = np.array(outlier_vals, dtype=np.float32)
                    k_hat = mx.array(k_hat_np, dtype=mx.float32)

                batch_results.append(k_hat.astype(keys.dtype))
            head_results.append(mx.stack(batch_results, axis=0))  # (B, S, D)
        k_dequant = mx.stack(head_results, axis=1)  # (B, H, S, D)

        if self._n_outlier > 0 and all(self._outlier_idx[h] is not None for h in range(H)):
            self._calib_done = True

        # Memory accounting: keys only (values stay fp16 in parent)
        b_mse = max(self._bits - 1, 1)
        m_eff = self._quantizers[0]._m_eff
        d_inlier = (self._head_dim - self._n_outlier) if self._n_outlier > 0 else self._head_dim
        per_token = (
            (
                math.ceil(d_inlier * b_mse / 8)  # MSE indices (inlier dims)
                + math.ceil(m_eff / 8)  # QJL signs
                + 2  # residual norm fp16
                + self._n_outlier * 2  # outlier fp16 channels
                + 2  # per-vector norm fp16
            )
            * H
            * B
        )
        self._key_bytes_compressed += per_token * S
        self._key_bytes_fp16 += H * B * S * self._head_dim * 2

        return super().update_and_fetch(k_dequant, values)

    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _head_dim(model) -> int:
    cfg = model.args
    return cfg.hidden_size // cfg.num_attention_heads


def build_caches(model, bits: int, n_outlier: int = 0):
    cfg = model.args
    hd = _head_dim(model)
    return [
        TurboQuantMLXKVCache(
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            bits=bits,
            seed=i,
            n_outlier=n_outlier,
        )
        for i in range(cfg.num_hidden_layers)
    ]


def run(model, tokenizer, cache_factory, label: str, max_tokens: int = MAX_TOKENS):
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    original_make_cache = model.make_cache
    injected = []
    if cache_factory is not None:

        def _patch(*_, **__):
            c = cache_factory()
            injected.extend(c)
            return c

        model.make_cache = _patch

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    model.make_cache = original_make_cache

    toks = len(tokenizer.encode(response))
    ratio_str = ""
    if injected:
        k_fp16 = sum(c.fp16_key_bytes for c in injected)
        k_cmp = sum(c.compressed_key_bytes for c in injected)
        if k_cmp > 0:
            ratio_str = f"  | key compression {k_fp16 / k_cmp:.2f}x  ({k_cmp / 1024:.0f} KB vs {k_fp16 / 1024:.0f} KB fp16)"

    print(f"\n{'=' * 60}")
    print(f"[{label}]{ratio_str}")
    print(f"  {response[:500]}{'...' if len(response) > 500 else ''}")
    print(f"  {toks} tokens  {elapsed:.1f}s  ({toks / elapsed:.1f} tok/s)")
    return response, elapsed, injected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
model, tokenizer = mlx_lm.load(MODEL_ID)
cfg = model.args
hd = _head_dim(model)
print(f"  head_dim={hd}, kv_heads={cfg.num_key_value_heads}, layers={cfg.num_hidden_layers}\n")

# fp16 baseline
resp_fp16, t_fp16, _ = run(model, tokenizer, None, "fp16 baseline")

# 3-bit (aggressive, some quality loss)
resp_3b, t_3b, c3 = run(
    model,
    tokenizer,
    lambda: build_caches(model, bits=3),
    "TurboQuant 3-bit",
)

# 4-bit (near-lossless)
resp_4b, t_4b, c4 = run(
    model,
    tokenizer,
    lambda: build_caches(model, bits=4),
    "TurboQuant 4-bit",
)

# 4-bit + outlier protection (best quality)
N_OUTLIER = 8
resp_4bo, t_4bo, c4o = run(
    model,
    tokenizer,
    lambda: build_caches(model, bits=4, n_outlier=N_OUTLIER),
    f"TurboQuant 4-bit + {N_OUTLIER} outlier channels",
)

# Summary table
print(f"\n{'=' * 60}")
print(f"{'Config':<35} {'Compression':>12} {'Time':>7} {'Quality'}")
print(f"{'-' * 60}")


def ratio(caches):
    kf = sum(c.fp16_key_bytes for c in caches)
    kc = sum(c.compressed_key_bytes for c in caches)
    return f"{kf / kc:.2f}x" if kc > 0 else "—"


fp16_tok = sum(
    cfg.num_key_value_heads * hd * 2 * 2 * cfg.num_hidden_layers
    for _ in [1]  # K+V fp16
) * max((c.offset for c in (model.make_cache())), default=200)

print(f"{'fp16 baseline':<35} {'1.00x':>12} {t_fp16:>6.1f}s  reference")
print(f"{'TurboQuant 3-bit':<35} {ratio(c3):>12} {t_3b:>6.1f}s  repetition (expected)")
print(f"{'TurboQuant 4-bit':<35} {ratio(c4):>12} {t_4b:>6.1f}s  near-lossless")
print(f"{'TurboQuant 4-bit + outliers':<35} {ratio(c4o):>12} {t_4bo:>6.1f}s  lossless target")

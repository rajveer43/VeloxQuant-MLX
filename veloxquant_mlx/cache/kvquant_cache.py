"""KVQuant-NUQ KV cache wrapper — non-uniform quantization + outlier isolation.

Inspired by "KVQuant: Towards 10 Million Context Length LLM Inference with KV
Cache Quantization" (arXiv:2401.18079, NeurIPS 2024). Documented as
"KVQuant-adapted (VeloxQuant-MLX implementation)" — implements the four
cache-observable pillars (per-channel keys / per-token values, NUQ datatype,
per-vector dense-and-sparse outlier isolation, and Attention Sink-Aware
quantization) and documents pre-RoPE key quantization as out of scope. See
:mod:`veloxquant_mlx.quantizers.kvquant` for the numerics.

Quantization axes (matching KVQuant's asymmetry, the same axes KIVI uses):
    Keys   — per-channel: each head-dim channel gets its own non-uniform levels
             (sample axis = tokens). Channels have stable, distinct distributions.
    Values — per-token: each token gets its own levels (sample axis = channels).

Level lifecycle:
    Prefill (S > 1) fits the NUQ levels from the incoming batch and freezes them
    (``kvquant_refit_interval == 0``, the default — mirrors SVDq's frozen-V).
    Decode tokens quantize against the frozen levels. With a positive refit
    interval, levels are re-fit every N decode steps from the most recent token.

Byte accounting:
    compressed_*  — NUQ codes + per-(channel/token) level table + fp16 outlier
                    side-channel (value + position index).
    fp16_*        — uncompressed cost for the ratio.

Attention Sink-Aware quantization (paper §3.5):
    The first ``kvquant_n_sink`` tokens of the sequence are restored to exact
    fp16 and excluded from both the level fit and the outlier thresholds — the
    model is disproportionately sensitive to quantization error at the sink
    positions it dumps excess attention onto.

What is NOT implemented (documented):
    - Pre-RoPE key quantization (needs a model-forward hook — outside the cache
      contract; we see post-RoPE keys only).
    - Offline calibration-set level fitting (we fit online; zero calibration).
    - Fisher-information sensitivity weighting of the Lloyd-Max objective
      (Eq. 1 weights each squared error by F_ii; needs gradients from a
      calibration pass, so our fit is the unweighted k-means special case).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.kvquant import (
    dequant_nuq,
    fit_nuq_levels,
    quantize_nuq,
    split_dense_sparse,
)


class KVQuantKVCache(_MLXKVCache):
    """KV cache implementing KVQuant-NUQ non-uniform quantization.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``kvquant_bits``             (int, default 3),
            ``kvquant_outlier_fraction`` (float, default 0.01),
            ``kvquant_group_size``       (int, default 32; reserved for grouped fits),
            ``kvquant_lloyd_iters``      (int, default 8),
            ``kvquant_refit_interval``   (int, default 0 = freeze prefill levels).

    No ``.bits`` attribute — mlx_lm's SDPA checks ``hasattr(cache, "bits")``
    to route to its quantized-matmul kernel path, which expects a different
    cache layout (mx.quantize's native tuple format) and a ``.group_size``
    attribute this cache doesn't have. We expose ``.nuq_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._bits: int = int(getattr(config, "kvquant_bits", 3))
        self._outlier_fraction: float = float(getattr(config, "kvquant_outlier_fraction", 0.01))
        self._group_size: int = int(getattr(config, "kvquant_group_size", 32))
        self._lloyd_iters: int = int(getattr(config, "kvquant_lloyd_iters", 8))
        self._refit_interval: int = int(getattr(config, "kvquant_refit_interval", 0))
        self._n_sink: int = int(getattr(config, "kvquant_n_sink", 1))
        if self._n_sink < 0:
            raise ValueError(f"KVQuantKVCache: kvquant_n_sink={self._n_sink} must be >= 0")

        # Per-channel |key| outlier threshold frozen at prefill, reused at decode
        # where a single token cannot define its own per-channel top-k.
        self._key_outlier_thresh: Optional[mx.array] = None
        self._sink_kept: int = 0

        # Frozen levels fit at prefill: keys per-channel [H, L, D],
        # values per-token use levels [H, L, D] in transposed space.
        self._key_levels: Optional[list] = None  # list over heads of [L, D]
        self._value_levels: Optional[list] = None  # list over heads of [L, D] (channel-as-sample)
        self._n_tokens: int = 0
        self._outlier_count: int = 0
        self._prev_outlier_count: int = 0

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._compressed_value_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._fp16_value_bytes: int = 0

    # ------------------------------------------------------------------
    # Per-head NUQ application
    # ------------------------------------------------------------------
    def _quant_keys_head(self, k_sd: mx.array, levels: Optional[mx.array]):
        """Keys: per-channel NUQ on [S, D]. Returns (recon_fp16, levels_used).

        At decode (S == 1) a per-channel column holds a single sample, so the
        rank-based outlier split degenerates (it cannot keep >=1 inlier and
        still flag anything). We instead apply a magnitude threshold carried
        over from the frozen prefill statistics, so decode keys keep the same
        outlier protection the prefill keys got.
        """
        ds = split_dense_sparse(k_sd, self._outlier_fraction)
        if levels is None:
            levels = fit_nuq_levels(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq(ds.inliers, levels)
        recon = dequant_nuq(codes, levels).astype(mx.float32)
        mask = ds.outlier_mask
        vals = ds.outlier_vals
        if (
            k_sd.shape[0] < 2
            and self._outlier_fraction > 0.0
            and self._key_outlier_thresh is not None
        ):
            # Decode: reuse the frozen per-channel threshold from prefill.
            k32 = k_sd.astype(mx.float32)
            mask = mx.abs(k32) >= self._key_outlier_thresh
            vals = mx.where(mask, k32, mx.zeros_like(k32))
        recon = mx.where(mask, vals, recon)
        self._outlier_count += int(mx.sum(mask).item())
        return recon.astype(mx.float16), levels

    def _quant_values_head(self, v_sd: mx.array, levels: Optional[mx.array]):
        """Values: per-token NUQ on [S, D] → transpose so tokens are columns.

        Values are unaffected by the decode degeneracy: a per-token column has
        D samples regardless of S, so the rank-based split is always well posed.
        """
        v_ds = v_sd.T  # [D, S]: now each column is one token
        ds = split_dense_sparse(v_ds, self._outlier_fraction)
        if levels is None:
            levels = fit_nuq_levels(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq(ds.inliers, levels)
        recon = dequant_nuq(codes, levels).astype(mx.float32)
        recon = mx.where(ds.outlier_mask, ds.outlier_vals, recon)
        self._outlier_count += int(mx.sum(ds.outlier_mask).item())
        return recon.astype(mx.float16).T, levels  # back to [S, D]

    def _capture_key_thresholds(self, keys: mx.array) -> None:
        """Freeze the per-channel outlier threshold from the prefill keys.

        ``thresh[0, d]`` is the k-th largest |key| in channel ``d`` over the
        prefill tokens — the same cut the rank-based split makes. Decode steps
        (S == 1) reuse it because a single token cannot define its own top-k.
        """
        if self._outlier_fraction <= 0.0:
            self._key_outlier_thresh = None
            return
        # keys: [B, H, S, D] → pool tokens across batch/heads per channel.
        k32 = keys.astype(mx.float32)
        flat = k32.reshape(-1, k32.shape[-1])  # [B*H*S, D]
        n = flat.shape[0]
        if n < 2:
            self._key_outlier_thresh = None
            return
        k = max(1, int(round(n * self._outlier_fraction)))
        k = min(k, n - 1)
        sorted_desc = mx.sort(mx.abs(flat), axis=0)[::-1]
        self._key_outlier_thresh = sorted_desc[k - 1 : k, :]  # [1, D]
        mx.eval(self._key_outlier_thresh)

    def _apply(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        # Keys use per-channel levels (stable across tokens) → fit at prefill and
        # freeze. Values use per-token levels (one set per token) → inherently
        # re-fit every call; they are never frozen across steps.
        is_prefill = self._key_levels is None
        refit_keys = is_prefill or (
            self._refit_interval > 0
            and self._n_tokens > 0
            and (self._n_tokens % self._refit_interval == 0)
        )
        key_levels = None if refit_keys else self._key_levels  # list[H] of [L, D] or None

        # Sink tokens are restored to fp16 below, so — following the paper — they
        # are also excluded from the level fit and from the outlier thresholds.
        # Otherwise the very tokens we keep exact would still skew the datatype
        # derived for every other token.
        n_sink = min(self._n_sink, S) if self._n_tokens == 0 else 0
        fit_slice = slice(n_sink, None) if n_sink > 0 and S > n_sink else slice(None)
        if refit_keys:
            self._capture_key_thresholds(keys[:, :, fit_slice, :])

        k_out_b, v_out_b = [], []
        new_klev = [None] * H
        for b in range(B):
            k_h, v_h = [], []
            for h in range(H):
                # Share frozen key levels across batch; fit once on (b==0) when refitting.
                kl = key_levels[h] if key_levels is not None else (new_klev[h] if b > 0 else None)
                if kl is None:
                    # Fit levels on non-sink tokens, then quantize the full block
                    # against them.
                    kl = fit_nuq_levels(
                        split_dense_sparse(keys[b, h][fit_slice], self._outlier_fraction).inliers,
                        self._bits,
                        self._lloyd_iters,
                    )
                kq, klev = self._quant_keys_head(keys[b, h], kl)
                vq, vlev = self._quant_values_head(values[b, h], None)  # values: always fresh
                new_klev[h] = klev
                last_vlev = vlev
                k_h.append(kq)
                v_h.append(vq)
            k_out_b.append(mx.stack(k_h, axis=0))
            v_out_b.append(mx.stack(v_h, axis=0))

        if refit_keys:
            self._key_levels = new_klev
        self._value_levels = [last_vlev]  # most-recent per-token levels (introspection)

        k_out = mx.stack(k_out_b, axis=0)
        v_out = mx.stack(v_out_b, axis=0)

        # Attention Sink-Aware quantization (paper §3.5): the model allocates a
        # disproportionate attention score to the first few tokens and is
        # correspondingly sensitive to quantization error there, so those
        # positions are restored to their exact fp16 values. Only the leading
        # positions of the *whole sequence* are sinks, so this applies on the
        # prefill call (n_tokens == 0), never to mid-stream decode tokens.
        if n_sink > 0:
            k_out = mx.concatenate(
                [keys[:, :, :n_sink, :].astype(mx.float16), k_out[:, :, n_sink:, :]], axis=2
            )
            v_out = mx.concatenate(
                [values[:, :, :n_sink, :].astype(mx.float16), v_out[:, :, n_sink:, :]], axis=2
            )
            self._sink_kept = n_sink

        return k_out, v_out

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        n_sink = min(self._n_sink, S) if self._n_tokens == 0 else 0
        k_out, v_out = self._apply(keys, values)
        self._n_tokens += S
        self._account_bytes(B, H, S, D, n_sink)
        return super().update_and_fetch(k_out, v_out)

    def _account_bytes(self, B: int, H: int, S: int, D: int, n_sink: int = 0) -> None:
        L = 1 << self._bits
        # Sink tokens are stored as raw fp16 and are not coded at all.
        s_q = max(0, S - n_sink)
        # Codes: bits per element. Level table: L fp16 per channel (keys) / per
        # token (values). Outlier side-channel: fp16 value + ~index bits.
        code_bytes = math.ceil(s_q * D * self._bits / 8)
        key_table_bytes = L * D * 2  # per-channel table
        val_table_bytes = L * s_q * 2  # per-token table (one per coded token)
        idx_bits = max(1, math.ceil(math.log2(max(2, max(1, s_q) * D))))
        # Charge the outliers actually produced this call, not the nominal
        # fraction: the rank-based split rounds per column, and the decode path
        # uses a carried-over threshold, so realized counts differ from
        # S * D * outlier_fraction. Accounting must follow the real side-channel.
        n_out_per_head = self._outlier_count - self._prev_outlier_count
        self._prev_outlier_count = self._outlier_count
        # _outlier_count aggregates keys+values across the whole B*H loop.
        per_head_out = n_out_per_head / max(1, 2 * B * H)
        outlier_bytes = int(round(per_head_out * (2 + math.ceil(idx_bits / 8))))
        sink_bytes = n_sink * D * 2  # exact fp16 sink rows

        self._compressed_key_bytes += (
            (code_bytes + key_table_bytes + outlier_bytes + sink_bytes) * B * H
        )
        self._compressed_value_bytes += (
            (code_bytes + val_table_bytes + outlier_bytes + sink_bytes) * B * H
        )
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def nuq_bits(self) -> int:
        # Deliberately not named `.bits` — mlx_lm's SDPA checks
        # `hasattr(cache, "bits")` to route to its quantized-matmul kernel,
        # which expects mx.quantize's native tuple layout and a `.group_size`
        # attribute this cache doesn't have. Exposing `.bits` here would
        # silently hijack attention dispatch. See other *_cache.py classes
        # (e.g. kivi_cache.py, turboquant_rvq_cache.py) for the same guard.
        return self._bits

    @property
    def outlier_fraction(self) -> float:
        return self._outlier_fraction

    @property
    def outlier_count(self) -> int:
        return self._outlier_count

    @property
    def n_sink(self) -> int:
        """Configured number of leading (attention-sink) tokens kept in fp16."""
        return self._n_sink

    @property
    def sink_kept(self) -> int:
        """Sink tokens actually retained in fp16 (bounded by the prefill length)."""
        return self._sink_kept

    @property
    def key_outlier_thresh(self):
        """Frozen per-channel |key| outlier threshold reused during decode."""
        return self._key_outlier_thresh

    @property
    def key_levels(self):
        return self._key_levels

    @property
    def value_levels(self):
        return self._value_levels

    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def effective_bits(self) -> float:
        """Effective per-element key bits (codes + table + outliers vs fp16)."""
        if self._fp16_key_bytes == 0:
            return float(self._bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["KVQuantKVCache"]

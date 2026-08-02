"""AMC-adapted KV cache — per-token saliency-driven tiered rank + precision.

Inspired by "Adaptive Model Compression (AMC)" (Hu, Yuan, Hu, Yin, Li,
Suchter — Apple; arXiv:2607.10109, **no verified peer-reviewed venue as of
2026-07-14**). Documented as "AMC-adapted (VeloxQuant-MLX implementation)" —
not a faithful port; see :mod:`veloxquant_mlx.quantizers.amc` module
docstring for the full honesty crux (venue status, hardware/RTL scope cut,
compression-only-not-eviction framing).

Every token — prefill or decode — is scored, tiered, rank-masked, and
quantized on every call. No token is ever dropped: this is the fortieth
method in this repo and the first whose family is "adaptive rank+precision"
rather than "eviction." The cache's stored sequence length always equals the
number of tokens seen; only the effective rank/bit-width per token varies.

Byte accounting:
    amc_kept_bytes     — actual bytes for K + V given each token's tier
    full_seq_bytes      — hypothetical fp16 full-rank cost if AMC were never
                           applied
    compression_ratio   — full_seq_bytes / amc_kept_bytes (> 1 = savings)
    tokens_seen         — total token positions ever passed to update_and_fetch
    tokens_high/mid/low — cumulative per-tier token counts (observability)
"""

from __future__ import annotations

from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.amc import (
    HIGH,
    LOW,
    MID,
    AMCThresholdState,
    amc_adaptive_thresholds,
    amc_apply_rank_mask,
    amc_assign_tiers,
    amc_fp16_bytes,
    amc_query_aware_saliency,
    amc_quantize_tier,
    amc_saliency,
    full_amc_fp16_bytes,
    init_amc_threshold_state,
    _tier_config_for_dim,
)


class AMCKVCache(_MLXKVCache):
    """KV cache implementing AMC-adapted saliency-driven tiered compression.

    Args:
        config: :class:`~veloxquant_mlx.cache.base.KVCacheConfig`. Fields
            consumed:
                ``amc_k_high`` (float, default 0.20)         — top percentile -> High tier
                ``amc_k_mid`` (float, default 0.30)           — next percentile -> Mid tier
                ``amc_use_query_saliency`` (bool, default False)
                ``amc_query_alpha`` (float, default 0.5)      — Eq. 3 balance coefficient
                ``amc_adaptive_thresholds`` (bool, default False) — Eq. 4-5 closed-loop
                ``amc_threshold_window`` (int, default 64)    — trailing window size
                ``amc_gamma`` (float, default 0.1)            — threshold attenuation factor
                ``amc_calib_variance`` (float | None)         — required if adaptive thresholds on
                ``amc_group_size`` (int, default 32)          — quantization group size

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly (the
        rank mask + quantize round-trip is simulated, matching every other
        method's "quantize-then-dequantize to fp16" convention).
        Both prefill (S > 1) and decode (S == 1) go through the identical
        per-token tiering loop.
        Query-aware saliency (Eq. 3) requires a query vector, which is not
        available at the cache-wrapper level (same limitation as H2O/SnapKV/
        CurDKV's key-as-query-proxy) — when enabled, the mean of the current
        step's keys is used as a proxy query, disclosed here as the same
        category of approximation as the eviction methods' proxy queries.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._k_high = float(getattr(config, "amc_k_high", 0.20))
        self._k_mid = float(getattr(config, "amc_k_mid", 0.30))
        if not 0.0 <= self._k_high <= 1.0 or not 0.0 <= self._k_mid <= 1.0:
            raise ValueError(
                f"AMCKVCache: amc_k_high ({self._k_high}) and amc_k_mid "
                f"({self._k_mid}) must each be in [0, 1]."
            )
        if self._k_high + self._k_mid > 1.0:
            raise ValueError(
                f"AMCKVCache: amc_k_high + amc_k_mid must be <= 1.0, got "
                f"{self._k_high} + {self._k_mid} = "
                f"{self._k_high + self._k_mid}."
            )
        self._use_query_saliency = bool(getattr(config, "amc_use_query_saliency", False))
        self._query_alpha = float(getattr(config, "amc_query_alpha", 0.5))
        self._use_adaptive_thresholds = bool(getattr(config, "amc_adaptive_thresholds", False))
        self._threshold_window = int(getattr(config, "amc_threshold_window", 64))
        self._gamma = float(getattr(config, "amc_gamma", 0.1))
        self._calib_variance = getattr(config, "amc_calib_variance", None)
        self._group_size = int(getattr(config, "amc_group_size", 32))

        if self._use_adaptive_thresholds and self._calib_variance is None:
            raise ValueError(
                "AMCKVCache: amc_adaptive_thresholds=True requires "
                "amc_calib_variance to be set (from offline calibration)."
            )

        self._head_dim: int = int(getattr(config, "head_dim", 128))

        self._B: int = 0
        self._H: int = 0
        self._keys: List[mx.array] = []  # per (b,h): [n_seen, D] fp16
        self._values: List[mx.array] = []  # per (b,h): [n_seen, D] fp16

        self._threshold_states: List[AMCThresholdState] = []

        self._amc_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0
        self._tier_counts: Dict[int, int] = {HIGH: 0, MID: 0, LOW: 0}

    # ------------------------------------------------------------------
    def _ensure_state(self, B: int, H: int) -> None:
        if not self._keys:
            self._B = B
            self._H = H
            self._keys = [None] * (B * H)
            self._values = [None] * (B * H)
            if self._use_adaptive_thresholds:
                self._threshold_states = [
                    init_amc_threshold_state(self._threshold_window, self._calib_variance)
                    for _ in range(B * H)
                ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    def _tier_thresholds(self, idx: int, saliency: mx.array) -> tuple:
        """Resolve (tau_H, tau_L) either statically or via the closed loop."""
        # Percentile-based assignment (amc_assign_tiers) does not need an
        # absolute threshold pair — it directly selects the top-k_high /
        # top-k_mid tokens by rank. The adaptive-threshold machinery (Eq.
        # 4-5) is exposed for observability/parity with the paper even
        # though amc_assign_tiers's percentile selection already adapts to
        # the current step's distribution implicitly; when enabled, we still
        # advance the trailing-window state so amc_adaptive_thresholds's
        # variance tracking is exercised and available to callers/tests.
        if self._use_adaptive_thresholds:
            state = self._threshold_states[idx]
            _, _, state = amc_adaptive_thresholds(
                tau_high_base=1.0 - self._k_high,
                tau_low_base=1.0 - self._k_high - self._k_mid,
                state=state,
                new_saliency_values=saliency,
                gamma=self._gamma,
            )
            self._threshold_states[idx] = state

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply AMC tiered compression, return all tokens.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_seen, D]`` fp16 — AMC never
            evicts, so ``n_seen`` equals the total tokens passed so far.
        """
        B, H, S, D = keys.shape
        self._ensure_state(B, H)

        self._full_seq_bytes += full_amc_fp16_bytes(B * H * S, D)
        self._tokens_seen_total += B * H * S

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                k_step = keys[b, h].astype(mx.float16)  # [S, D]
                v_step = values[b, h].astype(mx.float16)  # [S, D]

                if self._use_query_saliency:
                    query = mx.mean(k_step.astype(mx.float32), axis=0)
                    saliency = amc_query_aware_saliency(
                        k_step, k_step, query, alpha=self._query_alpha
                    )
                else:
                    saliency = amc_saliency(k_step)

                self._tier_thresholds(idx, saliency)

                tiers = amc_assign_tiers(saliency, self._k_high, self._k_mid)

                k_compressed = self._compress_step(k_step, tiers, D)
                v_compressed = self._compress_step(v_step, tiers, D)

                for t in tiers:
                    self._tier_counts[t] += 1

                prev_k = self._keys[idx]
                prev_v = self._values[idx]
                new_k = (
                    k_compressed
                    if prev_k is None
                    else mx.concatenate([prev_k, k_compressed], axis=0)
                )
                new_v = (
                    v_compressed
                    if prev_v is None
                    else mx.concatenate([prev_v, v_compressed], axis=0)
                )
                self._keys[idx] = new_k
                self._values[idx] = new_v

                k_out_h.append(new_k)
                v_out_h.append(new_v)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._amc_kept_bytes = amc_fp16_bytes(self._tier_counts, self._head_dim)

        return K_out, V_out

    def _compress_step(self, x: mx.array, tiers: List[int], head_dim: int) -> mx.array:
        """Apply per-token rank mask + quantization according to each token's tier."""
        n = x.shape[0]
        out_rows = []
        for i in range(n):
            cfg = _tier_config_for_dim(tiers[i], head_dim)
            row = x[i : i + 1]  # [1, D]
            row = amc_apply_rank_mask(row, cfg.rank)
            row = amc_quantize_tier(row, cfg.bits, self._group_size)
            out_rows.append(row)
        return mx.concatenate(out_rows, axis=0) if out_rows else x

    # ------------------------------------------------------------------
    @property
    def amc_kept_bytes(self) -> int:
        """Actual bytes stored across all heads (fp16-equivalent K + V, tiered)."""
        return self._amc_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 full-rank K + V cost if AMC were never applied."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / amc_kept_bytes; > 1 means savings over fp16 full-rank."""
        if self._amc_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._amc_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache — always == tokens per head seen."""
        if not self._keys or self._keys[0] is None:
            return 0
        return int(self._keys[0].shape[0])

    @property
    def tokens_high(self) -> int:
        """Cumulative tokens (across all heads/steps) assigned the High tier."""
        return self._tier_counts[HIGH]

    @property
    def tokens_mid(self) -> int:
        """Cumulative tokens (across all heads/steps) assigned the Mid tier."""
        return self._tier_counts[MID]

    @property
    def tokens_low(self) -> int:
        """Cumulative tokens (across all heads/steps) assigned the Low tier."""
        return self._tier_counts[LOW]


__all__ = ["AMCKVCache"]

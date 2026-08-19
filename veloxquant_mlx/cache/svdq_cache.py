"""SVDq KV cache wrapper — sub-2-bit key compression via offline SVD.

Inspired by "SVDq: Singular Value Decomposition-based KV Cache Quantization"
(arXiv:2502.15304, Feb 2025, unreviewed preprint).  Documented as
"SVDq-adapted (VeloxQuant-MLX implementation)" — not a faithful port. See
``quantizers/svdq.py`` module docstring for the full list of remaining
deviations (per-sequence SVD instead of offline calibration, invented rank
heuristic, no model-level accuracy evidence).

Design:

  Prefill (first call where S > 1):
    1. Compute truncated SVD of the incoming key batch K ∈ R^{S×D}.
    2. Store the right singular vectors V [D, r] and mean key K̄ [D] as layer
       state.  These are O(D²) and negligible for long sequences.
    3. Project keys into latent space L = (K - K̄) @ V → [S, r].
    4. Apply mixed-precision group quantization to L using the paper's 8-group
       fixed bit schedule (Eq. 6): the r latent channels are split into
       ``len(bit_schedule)`` equal-size contiguous groups (largest singular
       values first), each quantized at its own bit width — 0 bits truncates
       a group to exactly zero.
    5. Reconstruct fp16 keys for the downstream SDPA call.
    6. Accumulate latents in a growing list (quantized-then-dequantized, so
       the downstream cache sees fp16, consistent with all other wrappers).

  Decode (S == 1 or S < prefill threshold):
    1. Project new key into the already-stored V space.
    2. Quantize and reconstruct fp16.
    3. Pass through to the underlying mlx_lm KVCache.

  Values are left at fp16 throughout (the paper notes values have weak
  low-rank structure; compressing values is left to stacked wrappers).

Byte accounting:
  compressed_key_bytes  — latent storage at the schedule's per-group bit rate
  fp16_key_bytes        — what full fp16 would cost (for ratio computation)
  value_fp16_bytes      — values are always fp16 (reported separately)

  Effective key bit-width ≈ (r/D) * b̄, where b̄ is the schedule's mean
  bit-width (paper Eq. 6: b̄ = mean(bit_schedule)). For the default schedule
  (8,4,2,1,1,0,0,0), b̄ = 2. At r = 0.5D: effective ≈ 0.5 * 2 = 1.0 bit/element.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.svdq import (
    DEFAULT_BIT_SCHEDULE,
    equivalent_bit_width,
    latent_group_slices,
    min_safe_rank,
    quantize_latents_mixed,
    reconstruct_keys,
    svd_compress_keys,
)


class SVDqKVCache(_MLXKVCache):
    """KV cache implementing SVDq sub-2-bit key compression.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``head_dim`` (D),
            ``svdq_rank`` (int | None — explicit rank; None → energy threshold),
            ``svdq_energy_threshold`` (float, default 0.95),
            ``svdq_bit_schedule`` (Sequence[int], default (8,4,2,1,1,0,0,0) —
                paper Eq. 6's 8-group per-group bit widths, most-significant
                group first; a 0 truncates that group to exactly zero),
            ``svdq_group_size`` (int, default 32 — quantization group size
                along the token axis, independent of the latent-channel
                grouping above).

    Notes:
        Does not expose ``.bits`` — mlx_lm's SDPA checks ``hasattr(cache, "bits")``
        to route to a quantized kernel; we keep that path clean.
        Exposes ``.assigned_avg_bits`` with the effective key bit-width.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._rank: Optional[int] = getattr(config, "svdq_rank", None)
        self._energy_threshold: float = float(getattr(config, "svdq_energy_threshold", 0.95))
        self._bit_schedule: tuple[int, ...] = tuple(
            getattr(config, "svdq_bit_schedule", DEFAULT_BIT_SCHEDULE)
        )
        if not self._bit_schedule or any(b < 0 for b in self._bit_schedule):
            raise ValueError(
                f"svdq: svdq_bit_schedule entries must be >= 0, got {self._bit_schedule}"
            )
        self._group_size: int = int(getattr(config, "svdq_group_size", 32))

        # SVD state — set on first prefill call
        self._V: Optional[mx.array] = None  # [D, r] fp32
        self._K_mean: Optional[mx.array] = None  # [D] fp32
        self._singular_values: Optional[mx.array] = None  # [r] fp32
        self._r: int = 0  # actual rank used
        # Schedule actually used for quantization, resolved at prefill time —
        # may differ from self._bit_schedule if the small-rank guard degraded
        # it gracefully (automatic-rank case only; see _resolve_safe_schedule).
        self._effective_schedule: tuple[int, ...] = self._bit_schedule

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0
        self._tokens_seen: int = 0

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------
    def _run_prefill_svd(self, keys: mx.array) -> mx.array:
        """Compute SVD on keys [B, H, S, D], store projection, return reconstructed keys."""
        B, H, S, D = keys.shape
        # Process head 0 of batch 0 for the SVD; apply the same V to all heads.
        # Keys across heads share the same D-dimensional space.
        k0 = keys[0, 0].astype(mx.float32)  # [S, D]
        L, V, K_mean, s_vals = svd_compress_keys(
            k0, rank=self._rank, energy_threshold=self._energy_threshold
        )
        self._V = V  # [D, r]
        self._K_mean = K_mean  # [D]
        self._singular_values = s_vals  # [r]
        self._r = int(V.shape[1])
        mx.eval(self._V, self._K_mean, self._singular_values)

        self._effective_schedule = self._resolve_safe_schedule()

        # Project and quantize all heads
        return self._project_quantize_reconstruct(keys)

    def _resolve_safe_schedule(self) -> tuple[int, ...]:
        """Guard against the truncation failure mode where 0-bit groups in
        the schedule wipe out real signal because rank is too small for
        len(bit_schedule) groups to each cover a safe channel span.

        See quantizers/svdq.py's min_safe_rank / MIN_SAFE_CHANNELS_PER_GROUP
        and test_small_rank_near_group_count_is_rejected, which demonstrates
        SVDq losing to naive 2-bit quantization once groups shrink to size 1.
        Only relevant when the schedule actually truncates (has a 0-bit
        group) — a schedule with no zeros doesn't have this failure mode
        regardless of rank.

        Behavior differs by how rank was chosen:
          - Explicit ``svdq_rank``: the user made a specific, informed choice
            that turned out unsafe for this schedule — raise immediately so
            the misconfiguration is caught at the source rather than
            producing quietly-bad reconstructions.
          - Automatic (energy-threshold) rank: the rank depends on how many
            prefill tokens happen to arrive (e.g. a short sequence can only
            support a small rank), which is not something the caller
            explicitly chose. Raising here would make the *default* config
            crash on short sequences. Instead, fall back for this layer to a
            "no truncation" schedule (every group gets >= 1 bit,
            proportioned the same way as the configured schedule) so
            behavior degrades gracefully rather than failing outright.
        """
        n_groups = len(self._bit_schedule)
        if 0 not in self._bit_schedule or self._r >= min_safe_rank(n_groups):
            return self._bit_schedule

        if self._rank is not None:
            floor = min_safe_rank(n_groups)
            raise ValueError(
                f"svdq: explicit svdq_rank={self._rank} is too small for the "
                f"{n_groups}-group bit schedule {self._bit_schedule}, which "
                f"truncates trailing groups to 0 bits. With this rank, "
                f"groups would average {self._r / n_groups:.1f} channels "
                f"each, so a 0-bit group would wipe out individual channels "
                f"that may still carry real signal rather than a genuinely "
                f"negligible energy tail (the schedule assumes ~d/{n_groups} "
                f"channels per group, per the paper's own configs). Use "
                f"svdq_rank>={floor}, or pass a svdq_bit_schedule with no "
                f"0-bit groups."
            )

        # Automatic rank landed below the safe floor (e.g. a short prefill) —
        # degrade gracefully: replace 0-bit groups with 1-bit rather than
        # dropping real signal outright.
        return tuple(max(b, 1) for b in self._bit_schedule)

    def _project_quantize_reconstruct(self, keys: mx.array) -> mx.array:
        """Project keys → latent → quantize → reconstruct for all [B, H, S, D]."""
        B, H, S, D = keys.shape
        V = self._V
        K_mean = self._K_mean
        sv = self._singular_values

        out_heads = []
        for b in range(B):
            out_batch = []
            for h in range(H):
                k_bh = keys[b, h].astype(mx.float32)  # [S, D]
                k_centered = k_bh - K_mean[None, :]
                L = k_centered @ V  # [S, r]
                L_q = quantize_latents_mixed(
                    L,
                    sv,
                    bit_schedule=self._effective_schedule,
                    group_size=self._group_size,
                )
                k_hat = reconstruct_keys(L_q, V, K_mean)  # [S, D] fp16
                out_batch.append(k_hat)
            out_heads.append(mx.stack(out_batch, axis=0))  # [H, S, D]
        return mx.stack(out_heads, axis=0)  # [B, H, S, D]

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape

        if self._V is None:
            # First call — run SVD on the incoming batch (prefill)
            k_out = self._run_prefill_svd(keys)
        else:
            # Subsequent calls — project into existing V
            k_out = self._project_quantize_reconstruct(keys)

        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, values)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        r = self._r if self._r > 0 else D
        schedule = self._effective_schedule
        slices = latent_group_slices(r, n_groups=len(schedule))

        # Latent storage: each group's channels at its own bit width, plus
        # group-quant overhead (scale + zero per group, fp16). A 0-bit group
        # costs nothing beyond that overhead (paper Eq. 6 truncation).
        def _latent_bytes(n_tokens: int, n_ch: int, b: int) -> int:
            if n_ch <= 0:
                return 0
            code_bytes = math.ceil(n_tokens * n_ch * b / 8) if b > 0 else 0
            n_groups = math.ceil(n_tokens / self._group_size)
            param_bytes = n_groups * n_ch * 2 * 2 if b > 0 else 0  # scale + zero, fp16
            return (code_bytes + param_bytes) * H * B

        key_bytes = sum(
            _latent_bytes(S, end - start, schedule[i]) for i, (start, end) in enumerate(slices)
        )
        # V [D, r] + K_mean [D] stored once — amortized over tokens seen
        projection_bytes = (D * r + D) * 4 * H * B  # fp32

        self._compressed_key_bytes += key_bytes + projection_bytes
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2
        self._tokens_seen += S

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def value_fp16_bytes(self) -> int:
        return self._value_fp16_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Effective key bit-width: schedule's mean bit-width scaled by r/D."""
        schedule = self._effective_schedule
        if self._r == 0 or self._D == 0:
            return equivalent_bit_width(self._r, schedule)
        b_bar = equivalent_bit_width(self._r, schedule)
        # Scale by r/D — latent dim is smaller than original
        return b_bar * self._r / self._D

    @property
    def rank(self) -> int:
        """Actual SVD rank used after energy-threshold selection."""
        return self._r


__all__ = ["SVDqKVCache"]

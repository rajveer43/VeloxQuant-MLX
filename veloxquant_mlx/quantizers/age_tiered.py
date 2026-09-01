"""AgeTieredKV — position/age-driven three-tier KV precision (issue #256).

Motivating question (issue #256): does KV-cache precision need to be
uniform, or can token position/age be used to assign different bit-widths
without hurting quality more than a uniform budget-matched baseline?

Where this sits relative to existing methods in this repo:
  * :mod:`veloxquant_mlx.cache.kivi_cache` already treats recency specially
    — a fixed-length fp16 "residual window" of the newest tokens, with
    everything older quantized at one fixed bit-width. That is a two-level
    scheme (fp16 vs. quantized) gated purely by age.
  * :mod:`veloxquant_mlx.quantizers.amc` already implements a three-tier
    discrete precision ladder (High/Mid/Low), but tiers by per-token
    activation *saliency*, not position.

AgeTieredKV combines the two: three discrete quantization tiers, gated
purely by how long ago a token was written (``current_position -
token_position``), reusing this repo's existing group quantizer rather than
inventing a new numeric scheme:
  * Recent tokens (age < ``age_recent_boundary``)      -> high tier (default 8-bit)
  * Mid-age tokens (age < ``age_mid_boundary``)         -> mid tier  (default 4-bit)
  * Old tokens (age >= ``age_mid_boundary``)            -> low tier  (default 2-bit)

Every token is retained — like AMC, and unlike the eviction-family methods
(H2O, SnapKV, ...) — only its bit-width changes as it ages. Tokens are
re-quantized at coarser precision each time they cross a tier boundary,
mirroring KIVI's flush-on-boundary-crossing model rather than being
quantized once and left alone; see :class:`~veloxquant_mlx.cache.age_tiered_cache.AgeTieredKVCache`
for that boundary bookkeeping.

Unlike AMC, there is no rank masking here — the varying signal is purely
bit-width, so :func:`age_tier_quantize` is a thin wrapper around the shared
:func:`~veloxquant_mlx.quantizers._quant_utils._group_quant_dequant`
primitive (the same one KIVI and AMC's Low/Mid quantization both use), not
a reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

RECENT, MID, OLD = 0, 1, 2


@dataclass(frozen=True)
class AgeTierConfig:
    """One age tier's bit-width."""

    tier: int
    bits: int


def default_age_tiers(bits_recent: int, bits_mid: int, bits_old: int) -> Tuple[AgeTierConfig, ...]:
    """Build the three (tier, bits) pairs from the configured bit-widths."""
    return (
        AgeTierConfig(tier=RECENT, bits=bits_recent),
        AgeTierConfig(tier=MID, bits=bits_mid),
        AgeTierConfig(tier=OLD, bits=bits_old),
    )


def assign_age_tiers(
    ages: List[int],
    age_recent_boundary: int,
    age_mid_boundary: int,
) -> List[int]:
    """Map each token's age (in positions) to a tier id.

    Args:
        ages: Per-token age, ``current_position - token_position``. Age 0
            means "written this step."
        age_recent_boundary: Tokens with ``age < age_recent_boundary`` are
            RECENT. Must be > 0.
        age_mid_boundary: Tokens with ``age_recent_boundary <= age <
            age_mid_boundary`` are MID; ``age >= age_mid_boundary`` are OLD.
            Must be >= ``age_recent_boundary``.

    Returns:
        List of tier ids, same length as ``ages``.
    """
    tiers = []
    for age in ages:
        if age < age_recent_boundary:
            tiers.append(RECENT)
        elif age < age_mid_boundary:
            tiers.append(MID)
        else:
            tiers.append(OLD)
    return tiers


def age_tier_quantize(x: mx.array, bits: int, group_size: int = 32) -> mx.array:
    """Quantize-then-dequantize a contiguous run of tokens at ``bits`` precision.

    Args:
        x: ``[N, D]`` activations, a contiguous same-tier slice.
        bits: Target bit-width. ``bits >= 16`` is a no-op (cast to fp16 only).
        group_size: Token-axis group size for the shared min/max quantizer.

    Returns:
        ``[N, D]`` fp16 quantized-then-dequantized activations.
    """
    if x.shape[0] == 0:
        return x
    if bits >= 16:
        return x.astype(mx.float16)
    return _group_quant_dequant(x, bits, group_size)


def age_tiered_bytes(tier_counts: dict, tiers: Tuple[AgeTierConfig, ...], head_dim: int) -> int:
    """Actual stored bytes given per-tier token counts (K + V combined).

    Args:
        tier_counts: Mapping ``{RECENT: n, MID: n, OLD: n}``.
        tiers: The ``(tier, bits)`` config, e.g. from :func:`default_age_tiers`.
        head_dim: Channel dimension ``D``.

    Returns:
        Total bytes for K + V combined, across all tiers.
    """
    by_tier = {cfg.tier: cfg.bits for cfg in tiers}
    total = 0
    for tier_id, n in tier_counts.items():
        if n <= 0:
            continue
        bits = by_tier[tier_id]
        bytes_per_token = (head_dim * bits + 7) // 8
        total += n * bytes_per_token * 2  # K + V
    return total


def full_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V byte cost if every token stayed 16-bit."""
    return tokens_seen * head_dim * 2 * 2  # K + V, fp16 (2 bytes)


__all__ = [
    "RECENT",
    "MID",
    "OLD",
    "AgeTierConfig",
    "default_age_tiers",
    "assign_age_tiers",
    "age_tier_quantize",
    "age_tiered_bytes",
    "full_fp16_bytes",
]

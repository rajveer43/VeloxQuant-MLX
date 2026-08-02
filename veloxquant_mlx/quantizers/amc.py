"""AMC-adapted quantizer — saliency-driven per-token tiered rank + precision.

Inspired by "Adaptive Model Compression (AMC): Saliency-Driven Resource
Allocation for Ultra-Low-Power Transformer Inference" (Hu, Yuan, Hu, Yin, Li,
Suchter — Apple; arXiv:2607.10109v1 [cs.IR], submitted 2026-07-11). Documented
as "AMC-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

**No verified peer-reviewed venue as of 2026-07-14** — single-version
preprint, no Comments/journal-ref field. This is the **second of 40** methods
in this repo shipped without a verified venue (the first was NestedKV-adapted,
see :mod:`veloxquant_mlx.quantizers.nestedkv`), at the user's explicit
direction. This is a one-time exception, not a new standing precedent — the
next new method reverts to requiring a verified venue. State this plainly
everywhere AMC is documented.

**Scope cut (read before using this module):** AMC's source paper is a
hardware/software co-design — roughly half of it (Sections IV-V: 45nm CMOS
RTL, Verilog clock-gating, the Precision-Gated Systolic Array, the
Narrow-Width SRAM write-back buffer, all pJ/µJ energy figures, the EDAP/Pareto
silicon comparisons) targets physical silicon that has no analogue in a
software MLX library. **None of that is implemented here.** This module ports
only the portable software half: Section II-A's saliency engine (L1-norm
score, three-tier percentile partitioning, the query-aware semantic-saliency
blend, and the sequence-adaptive closed-loop threshold adjustment) and Section
III's adaptive resource scaling (Hadamard rank masking + linear
fixed-point-style quantization), plus the offline SVD/PCA channel-order
calibration from Algorithm 1 Phase I (see
:mod:`veloxquant_mlx.quantizers.amc_calibration`). The paper's headline
59.2%-energy / 2.24x-throughput / 3.6%-accuracy numbers are the paper's own
hardware-measured figures under its own synthetic 3-layer setup — not
reproduced by this software port. See ``benchmark_scripts/benchmark_amc.py``
for this repo's own (unrelated, software-only) synthetic benchmark.

**Compression-only, never eviction:** unlike every eviction-family method in
this repo (H2O, SnapKV, PyramidKV, CurDKV, NestedKV, ...), AMC never drops a
token. Every token is retained; only its rank (how many hidden-dim channels
survive) and bit-width (quantization precision) are reduced for low-saliency
tokens. This is a structurally different family: "adaptive rank+precision,"
not "eviction."

Mechanism (per token, every prefill *and* decode step — the paper's own
design applies tiering continuously, not once at prefill like NestedKV):
  1. Saliency score ``S_i`` — mean absolute activation magnitude (L1-norm,
     Eq. 1-2), optionally blended with query-aware cosine similarity (Eq. 3).
  2. Percentile-threshold tier assignment: top ``k_high`` → High, next
     ``k_mid`` → Mid, remainder → Low (Algorithm 1 Phase II).
  3. Rank masking (Eq. 6): zero out channels beyond the tier's rank, on
     channels already reordered by the offline calibration permutation so the
     surviving prefix is the highest-variance subspace.
  4. Linear quantization (Eq. 7) to the tier's bit-width.

Adaptation notes (stated plainly, mirroring every other "-adapted" method's
honest-deviation convention):
  - Query-aware saliency (Eq. 3) and sequence-adaptive closed-loop thresholds
    (Eq. 4-5) are opt-in and off by default — the default path is pure
    magnitude-only scoring (Eq. 1-2), matching the paper's primary reported
    configuration.
  - Requires the offline calibration permutation (see
    :mod:`veloxquant_mlx.quantizers.amc_calibration`) to be meaningful; using
    AMC without it truncates arbitrary, not lowest-variance, channels — the
    same category of footgun as Palu/SVDq/RaBitQ's calibration requirement.
  - `cs.IR` is an unusual arXiv category for what is fundamentally a hardware
    architecture paper — noted as a minor oddity, not a disqualifier.

Byte accounting:
    amc_kept_bytes     — sum of actual per-tier bit-width bytes for K + V
    full_seq_bytes      — hypothetical fp16 cost if every token were 16-bit,
                           full rank
    compression_ratio   — full_seq_bytes / amc_kept_bytes (> 1 = savings)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import mlx.core as mx

from veloxquant_mlx.dsa.bit_pack import BitPackBuffer
from veloxquant_mlx.dsa.heap import MaxHeap
from veloxquant_mlx.dsa.ring_buffer import RingBuffer
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

# ---------------------------------------------------------------------------
# Tier definitions — Algorithm 1 (paper), exact rank/bit values
# ---------------------------------------------------------------------------

HIGH, MID, LOW = 0, 1, 2


@dataclass(frozen=True)
class AMCTierConfig:
    """One tier's (rank, bit-width) pair — Algorithm 1's Golden Model values."""

    tier: int  # HIGH=0, MID=1, LOW=2
    rank: int  # channels retained
    bits: int  # quantization bit-width


AMC_TIERS: Tuple[AMCTierConfig, ...] = (
    AMCTierConfig(tier=HIGH, rank=128, bits=16),
    AMCTierConfig(tier=MID, rank=43, bits=8),
    AMCTierConfig(tier=LOW, rank=8, bits=4),
)


def _tier_config_for_dim(tier: int, head_dim: int) -> AMCTierConfig:
    """Scale the paper's D=128 tier ranks to an arbitrary head_dim.

    The paper's Algorithm 1 fixes ranks at (128, 43, 8) for D=128 — roughly
    (100%, 33.6%, 6.25%) of the full dimension. We preserve those fractions
    for other head_dim values (e.g. D=32/64 used in this repo's small-shape
    tests) rather than hard-coding 128/43/8, which would be nonsensical for
    D < 128.
    """
    base = AMC_TIERS[tier]
    if head_dim == 128:
        rank = base.rank
    else:
        frac = base.rank / 128.0
        rank = max(1, min(head_dim, round(head_dim * frac)))
    return AMCTierConfig(tier=tier, rank=rank, bits=base.bits)


# ---------------------------------------------------------------------------
# Saliency scoring — Eq. 1-2 (faithful port), Eq. 3 (opt-in query-aware)
# ---------------------------------------------------------------------------


def amc_saliency(x: mx.array) -> mx.array:
    """Magnitude-based saliency score (Eq. 1-2): mean(|x|) clamped to [0, 1].

    Args:
        x: ``[N, D]`` token activations.

    Returns:
        ``[N]`` saliency scores in ``[0, 1]``.
    """
    s = mx.mean(mx.abs(x.astype(mx.float32)), axis=-1)
    return mx.clip(s, 0.0, 1.0)


def amc_query_aware_saliency(
    x: mx.array,
    keys: mx.array,
    query: mx.array,
    alpha: float = 0.5,
) -> mx.array:
    """Query-aware semantic saliency (Eq. 3): blend of magnitude and cosine
    similarity to a query/prompt vector.

    ``S_i = alpha * mean(|x_i|) + (1 - alpha) * cosine_similarity(query, k_i)``

    Args:
        x: ``[N, D]`` token activations (for the magnitude term).
        keys: ``[N, D]`` key projection vectors (for the semantic term).
        query: ``[D]`` embedded query/prompt vector.
        alpha: Balance coefficient in ``[0, 1]``; 1.0 == pure magnitude
            (:func:`amc_saliency`), 0.0 == pure semantic relevance.

    Returns:
        ``[N]`` saliency scores in ``[0, 1]``.
    """
    mag = mx.mean(mx.abs(x.astype(mx.float32)), axis=-1)  # [N]

    k32 = keys.astype(mx.float32)
    q32 = query.astype(mx.float32)
    k_norm = mx.sqrt(mx.sum(k32 * k32, axis=-1))  # [N]
    q_norm = mx.sqrt(mx.sum(q32 * q32))  # scalar
    eps = 1e-8
    denom = mx.maximum(k_norm * q_norm, eps)
    cos_sim = (k32 @ q32) / denom  # [N]
    cos_sim = mx.clip((cos_sim + 1.0) * 0.5, 0.0, 1.0)  # map [-1,1] -> [0,1]

    s = alpha * mx.clip(mag, 0.0, 1.0) + (1.0 - alpha) * cos_sim
    return mx.clip(s, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tier assignment — Algorithm 1 Phase II, top-k selection via dsa.MaxHeap
# ---------------------------------------------------------------------------


def amc_assign_tiers(
    saliency: mx.array,
    k_high: float = 0.20,
    k_mid: float = 0.30,
) -> List[int]:
    """Assign each token a tier id via percentile-threshold partitioning.

    Uses :class:`veloxquant_mlx.dsa.heap.MaxHeap` to select the top
    ``ceil(k_high * N)`` and next ``ceil(k_mid * N)`` tokens by saliency in
    ``O(N log N)`` via heap pops — bounded selection rather than a full
    ``sorted()`` call, consistent with this repo's DSA-first convention
    (the same heap already backs :class:`~veloxquant_mlx.dsa.heap.SortedChannelIndex`
    elsewhere in the codebase).

    Args:
        saliency: ``[N]`` saliency scores.
        k_high: Fraction of tokens routed to the High tier.
        k_mid: Fraction of tokens routed to the Mid tier (after High).

    Returns:
        List of length ``N`` with tier ids (``HIGH``/``MID``/``LOW``).
    """
    n = int(saliency.shape[0])
    if n == 0:
        return []

    n_high = max(1, math.ceil(k_high * n)) if n > 0 else 0
    n_mid = max(1, math.ceil(k_mid * n)) if n > 1 else 0
    n_high = min(n_high, n)
    n_mid = min(n_mid, n - n_high)

    heap = MaxHeap()
    scores = saliency.tolist()
    for i, sc in enumerate(scores):
        heap.push(float(sc), i)

    tiers = [LOW] * n
    for _ in range(n_high):
        if len(heap) == 0:
            break
        _, idx = heap.pop()
        tiers[idx] = HIGH
    for _ in range(n_mid):
        if len(heap) == 0:
            break
        _, idx = heap.pop()
        tiers[idx] = MID
    # Remaining heap contents stay LOW (already the default fill).
    return tiers


# ---------------------------------------------------------------------------
# Sequence-adaptive closed-loop thresholds — Eq. 4-5, dsa.RingBuffer-backed
# ---------------------------------------------------------------------------


@dataclass
class AMCThresholdState:
    """Trailing-window saliency variance tracker for closed-loop thresholding."""

    window: RingBuffer = field(default_factory=lambda: RingBuffer(64))
    calib_variance: float = 1.0


def init_amc_threshold_state(window_size: int, calib_variance: float) -> AMCThresholdState:
    """Create a fresh threshold-adaptation state.

    Args:
        window_size: Trailing window length (RingBuffer capacity).
        calib_variance: Nominal activation variance from offline calibration
            (``sigma^2_calib`` in Eq. 4-5). Must be > 0.
    """
    return AMCThresholdState(
        window=RingBuffer(max(1, window_size)),
        calib_variance=max(calib_variance, 1e-8),
    )


def amc_adaptive_thresholds(
    tau_high_base: float,
    tau_low_base: float,
    state: AMCThresholdState,
    new_saliency_values: mx.array,
    gamma: float = 0.1,
) -> Tuple[float, float, AMCThresholdState]:
    """Sequence-adaptive closed-loop threshold adjustment (Eq. 4-5).

    Pushes ``new_saliency_values`` into the trailing window, computes the
    window's variance, and depresses/raises ``tau_H``/``tau_L`` by a
    log-ratio against the calibration-time variance.

    Args:
        tau_high_base: Baseline High-tier threshold (from offline calibration).
        tau_low_base: Baseline Mid-tier threshold.
        state: Current :class:`AMCThresholdState` (mutated in place — the
            returned state is the same object, appended to).
        new_saliency_values: ``[N]`` saliency scores from the current step,
            pushed into the trailing window before computing variance.
        gamma: Attenuation scaling factor.

    Returns:
        ``(tau_H, tau_L, state)`` — updated thresholds and the (mutated)
        state object.
    """
    for v in new_saliency_values.tolist():
        state.window.append(float(v))

    if len(state.window) < 2:
        return tau_high_base, tau_low_base, state

    vals = state.window.to_list()
    mean_v = sum(vals) / len(vals)
    seq_variance = sum((v - mean_v) ** 2 for v in vals) / len(vals)

    eps = 1e-8
    ratio = max(seq_variance, eps) / max(state.calib_variance, eps)
    adj = 1.0 - gamma * math.log(ratio)

    tau_h = tau_high_base * adj
    tau_l = tau_low_base * adj
    return tau_h, tau_l, state


# ---------------------------------------------------------------------------
# Rank masking — Eq. 6, Hadamard/index masking on calibration-ordered channels
# ---------------------------------------------------------------------------


def amc_apply_rank_mask(x: mx.array, rank: int) -> mx.array:
    """Zero channels ``[rank:D)`` of a (calibration-ordered) activation vector.

    Args:
        x: ``[N, D]`` activations, already permuted by
            :func:`veloxquant_mlx.quantizers.amc_calibration.amc_calibrate_channel_order`
            so the surviving prefix is the highest-variance subspace.
        rank: Number of leading channels to keep.

    Returns:
        ``[N, D]`` with columns ``rank:D`` zeroed.
    """
    n, d = x.shape
    rank = max(0, min(rank, d))
    if rank == d:
        return x
    mask = mx.concatenate([mx.ones((rank,), dtype=x.dtype), mx.zeros((d - rank,), dtype=x.dtype)])
    return x * mask[None, :]


# ---------------------------------------------------------------------------
# Precision scaling — Eq. 7, reuses the shared group quantizer
# ---------------------------------------------------------------------------


def amc_quantize_tier(x: mx.array, bits: int, group_size: int = 32) -> mx.array:
    """Quantize-then-dequantize a tier's activations to ``bits`` precision.

    Reuses the repo's shared asymmetric min/max group quantizer
    (:func:`veloxquant_mlx.quantizers._quant_utils._group_quant_dequant`)
    rather than hand-rolling Eq. 7's fixed-point rounding — same simulated
    quantize/dequantize round-trip, same convention as every other method
    here.

    Args:
        x: ``[N, D]`` activations (already rank-masked).
        bits: Target bit-width (16, 8, or 4 in AMC's tier scheme, but any
            value the shared group quantizer accepts works).
        group_size: Token-axis group size for min/max quantization.

    Returns:
        ``[N, D]`` fp16 quantized-then-dequantized activations.
    """
    if bits >= 16:
        return x.astype(mx.float16)
    return _group_quant_dequant(x, bits, group_size)


# ---------------------------------------------------------------------------
# 4-bit dense packing for the Low tier — dsa.BitPackBuffer
# ---------------------------------------------------------------------------


def amc_pack_low_tier(quantized_codes: "mx.array") -> Tuple[bytes, int]:
    """Densely pack Low-tier 4-bit integer codes via :class:`BitPackBuffer`.

    Args:
        quantized_codes: ``[N]`` uint8 codes in ``[0, 15]`` (already
            quantized to 4-bit integer indices, e.g. via a group-quant
            ``codes`` array before dequantization).

    Returns:
        ``(packed_bytes, n)`` where ``n`` is the original element count
        (needed to unpack).
    """
    import numpy as np

    codes_np = np.asarray(quantized_codes, dtype=np.uint8)
    packer = BitPackBuffer(4)
    packed = packer.pack(codes_np)
    return packed.tobytes(), len(codes_np)


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def amc_fp16_bytes(tier_counts: dict, head_dim: int) -> int:
    """Compute actual stored bytes given per-tier token counts (K + V).

    Args:
        tier_counts: Mapping ``{HIGH: n_high, MID: n_mid, LOW: n_low}``.
        head_dim: Full channel dimension ``D`` (used to scale tier ranks).

    Returns:
        Total bytes for K + V combined, across all tiers.
    """
    total = 0
    for tier_id, n in tier_counts.items():
        if n <= 0:
            continue
        cfg = _tier_config_for_dim(tier_id, head_dim)
        bytes_per_token = math.ceil(cfg.rank * cfg.bits / 8)
        total += n * bytes_per_token * 2  # K + V
    return total


def full_amc_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 full-rank K + V byte cost if AMC were never applied."""
    return tokens_seen * head_dim * 2 * 2  # K + V, fp16 (2 bytes), full rank


__all__ = [
    "HIGH",
    "MID",
    "LOW",
    "AMCTierConfig",
    "AMC_TIERS",
    "AMCThresholdState",
    "init_amc_threshold_state",
    "amc_saliency",
    "amc_query_aware_saliency",
    "amc_assign_tiers",
    "amc_adaptive_thresholds",
    "amc_apply_rank_mask",
    "amc_quantize_tier",
    "amc_pack_low_tier",
    "amc_fp16_bytes",
    "full_amc_fp16_bytes",
]

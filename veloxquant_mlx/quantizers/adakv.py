"""AdaKV-proxy quantizer — per-head adaptive bit allocation over KIVI.

Inspired by "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference" (arXiv:2407.11550, 2024). Documented
as "AdaKV-proxy (VeloxQuant-MLX implementation)" — a *proxy* adaptation, not a
faithful port: true Ada-KV adapts the per-head *eviction* budget using softmax
attention weights, which live outside the ``update_and_fetch`` contract. We
instead adapt the per-head *bit* budget using an attention-free proxy for head
importance.

Algorithm:
    For a key tensor K ∈ R^{B×H×S×D}:
      1. Per head h, compute the inter-token L2-norm variance of the keys:
             head_importance[h] = Var_t( ‖K[:, h, t, :]‖₂ )
         averaged over the batch. High variance ⇒ the head's key magnitudes
         spread widely across tokens — a proxy for high attention entropy and
         thus for a head that benefits from extra precision.
      2. Normalise importances to sum to 1, scale by ``n_heads × target_avg_bits``
         to obtain a real-valued per-head bit budget, then snap each head to the
         nearest value in the allowed set (e.g. {2, 3, 4}).
      3. A greedy round-trip correction nudges heads up/down — picking the head
         closest to a rounding boundary each time — until the total bit budget
         matches ``n_heads × target_avg_bits`` as closely as the allowed set
         permits.
      4. Each head's keys are quantized at its assigned bit-width with KIVI-style
         asymmetric min/max group quantization.

Effective bit-width:
    assigned_avg_bits = Σ_h head_bits[h] / H,  with  head_bits[h] ∈ allowed_bits
    and  Σ_h head_bits[h] ≈ H × target_avg_bits.

Adaptation notes:
    - True Ada-KV (head-adaptive eviction budget) needs softmax attention
      weights — outside the update_and_fetch contract. Documented as the
      theoretical basis only.
    - Cross-layer budget sharing (a low-importance layer donating budget to a
      high-importance layer) is out of scope — it would break the single-wrapper
      contract.
    - Values are left at fp16 — AdaKV-proxy is a key-only method.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


def compute_head_norm_variance(keys: mx.array) -> mx.array:
    """Inter-token key-norm variance per head.

    For each head h, computes the variance over the sequence (token) axis of
    the per-token key L2 norms ‖k_t‖₂, then averages that variance over the
    batch dimension.

    Args:
        keys: Shape [B, H, S, D] fp16 or fp32.

    Returns:
        [H] fp32 array of per-head norm variances. Returns zeros if S < 2.
    """
    B, H, S, D = keys.shape
    k32 = keys.astype(mx.float32)
    norms = mx.sqrt(mx.sum(k32 * k32, axis=-1))  # [B, H, S]
    if S < 2:
        return mx.zeros((H,), dtype=mx.float32)
    var = mx.var(norms, axis=-1)  # [B, H]
    return mx.mean(var, axis=0)  # [H]


def allocate_head_bits(
    head_importance: Sequence[float] | mx.array,
    target_avg_bits: float,
    allowed_bits: Sequence[int],
    n_heads: int,
) -> list[int]:
    """Allocate per-head bit-widths under a global average-bits budget.

    Steps:
        1. Normalise importances to sum to 1, scale by ``n_heads × target``
           to get a real-valued per-head budget.
        2. Snap each head to the nearest value in ``allowed_bits``.
        3. Greedy correction: while the integer total over/undershoots
           ``n_heads × target``, bump the head whose snap is closest to a
           rounding boundary in the corrective direction (without leaving the
           allowed set), until the total can no longer move closer to target.

    Args:
        head_importance: [H] importance scores (>= 0). All-equal (incl. all-zero)
            degrades to a uniform target allocation.
        target_avg_bits: Global average bits/element target.
        allowed_bits: Sorted-or-unsorted set of permissible bit-widths.
        n_heads: Number of heads H (must match len(head_importance)).

    Returns:
        [H] list of assigned integer bit-widths, each in ``allowed_bits``.
    """
    allowed = sorted(set(int(b) for b in allowed_bits))
    lo, hi = allowed[0], allowed[-1]

    imp = [
        float(x)
        for x in (
            head_importance.tolist() if isinstance(head_importance, mx.array) else head_importance
        )
    ]
    if len(imp) != n_heads:
        raise ValueError(
            f"allocate_head_bits: head_importance has {len(imp)} entries but n_heads={n_heads}."
        )

    # Single head: no allocation freedom — snap target to the allowed set.
    if n_heads == 1:
        return [min(allowed, key=lambda b: abs(b - target_avg_bits))]

    total_imp = sum(imp)
    if total_imp <= 0.0:
        # No signal: uniform target allocation.
        norm = [1.0 / n_heads] * n_heads
    else:
        norm = [x / total_imp for x in imp]

    budget_total = n_heads * float(target_avg_bits)
    # Real-valued per-head budget, then clamp to the allowed range.
    real = [min(max(w * budget_total, float(lo)), float(hi)) for w in norm]

    # Snap each head to the nearest allowed bit value.
    bits = [min(allowed, key=lambda b: abs(b - r)) for r in real]

    # --- Greedy round-trip correction toward the integer budget -------------
    target_total = budget_total

    def _neighbor(b: int, direction: int) -> int | None:
        """Next allowed bit above (direction=+1) or below (direction=-1) b."""
        i = allowed.index(b)
        j = i + direction
        if 0 <= j < len(allowed):
            return allowed[j]
        return None

    # Move heads one allowed-step at a time, each time choosing the head whose
    # real budget is closest to the boundary we are crossing (smallest added
    # snap error). Stop when no single step reduces |total - target|.
    max_iters = n_heads * len(allowed) * 4
    for _ in range(max_iters):
        cur_total = sum(bits)
        diff = target_total - cur_total
        if abs(diff) < 1e-9:
            break
        direction = 1 if diff > 0 else -1

        best_h = -1
        best_cost = None
        for h in range(n_heads):
            nb = _neighbor(bits[h], direction)
            if nb is None:
                continue
            # Cost = how far this head's real budget already is from the new
            # snap; prefer bumping heads whose real value most supports it.
            cost = abs(real[h] - nb)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_h = h

        if best_h < 0:
            break  # nothing left to move in this direction

        new_bits = list(bits)
        new_bits[best_h] = _neighbor(bits[best_h], direction)  # type: ignore[arg-type]
        # Only accept the step if it gets the total strictly closer to target.
        if abs(target_total - sum(new_bits)) < abs(diff):
            bits = new_bits
        else:
            break

    return bits


def quantize_head(keys_h: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Quantize one head's keys at ``b`` bits with KIVI-style group quant.

    Args:
        keys_h: [S, D] fp16 or fp32 — a single head's key tensor.
        b: Bit-width for this head.
        group_size: Group size along the token axis.

    Returns:
        Reconstructed keys [S, D] fp16.
    """
    return _group_quant_dequant(keys_h, b, group_size).astype(mx.float16)


__all__ = [
    "compute_head_norm_variance",
    "allocate_head_bits",
    "quantize_head",
]

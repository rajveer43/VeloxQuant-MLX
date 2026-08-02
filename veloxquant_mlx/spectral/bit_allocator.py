from __future__ import annotations

import numpy as np


def water_fill_bits(
    eigenvalues: np.ndarray,
    total_bit_budget: int,
    min_bits: int = 1,
    max_bits: int = 8,
) -> np.ndarray:
    """Allocate bits proportionally to signal strength (water-filling).

    Dimensions with higher eigenvalue (more signal) receive more bits.
    The allocation sums to total_bit_budget (subject to min/max caps).

    When min/max constraints make the exact budget unreachable, the result
    is the closest feasible allocation (capped at min or max per dim).

    Args:
        eigenvalues: Per-dimension variance/eigenvalue, shape (d,), non-negative.
        total_bit_budget: Total bits to distribute across all d dimensions.
        min_bits: Minimum bits per dimension.
        max_bits: Maximum bits per dimension.

    Returns:
        Integer bit allocation array of shape (d,).
    """
    d = len(eigenvalues)
    ev = np.clip(eigenvalues, 0, None).astype(np.float64)
    ev_sum = ev.sum()

    if ev_sum < 1e-12 or total_bit_budget <= 0:
        uniform = max(min_bits, min(max_bits, total_bit_budget // d))
        return np.full(d, uniform, dtype=np.int32)

    # Iterative water-filling: proportionally allocate, then redistribute
    # bits from capped dims to uncapped dims until convergence.
    bits = np.full(d, min_bits, dtype=np.int32)
    remaining_budget = total_bit_budget - d * min_bits
    active = np.ones(d, dtype=bool)  # dims that can still receive more bits

    for _ in range(d + 2):  # at most d iterations to converge
        if remaining_budget <= 0:
            break
        active_ev = ev.copy()
        active_ev[~active] = 0.0
        active_sum = active_ev.sum()
        if active_sum < 1e-12:
            # Distribute remaining bits uniformly across still-active dims
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            per_dim = remaining_budget // len(active_indices)
            for i in active_indices:
                add = min(per_dim, max_bits - bits[i])
                bits[i] += add
                remaining_budget -= add
            break

        proportions = active_ev / active_sum
        alloc = proportions * remaining_budget
        proposed = bits.copy()
        proposed[active] += np.round(alloc[active]).astype(np.int32)
        proposed = np.clip(proposed, min_bits, max_bits)

        # Find newly capped dims
        newly_capped = active & (proposed >= max_bits)
        bits[newly_capped] = max_bits
        remaining_budget -= int((bits * newly_capped).sum()) - int(
            (bits * ~active * newly_capped).sum()
        )

        # Recompute remaining budget from scratch
        bits = np.where(newly_capped, max_bits, bits)
        remaining_budget = total_bit_budget - int(bits.sum())
        active[newly_capped] = False

        if not newly_capped.any():
            # No new caps: do final allocation
            active_ev2 = ev.copy()
            active_ev2[~active] = 0.0
            s2 = active_ev2.sum()
            if s2 > 1e-12:
                raw = active_ev2 / s2 * remaining_budget
                for i in np.where(active)[0]:
                    add = int(round(raw[i]))
                    add = max(0, min(add, max_bits - bits[i]))
                    bits[i] += add
            break

    # Final fix: exact budget correction with greedy adjustment
    diff = total_bit_budget - int(bits.sum())
    if diff != 0:
        # Sort by how much room is available in the desired direction
        if diff > 0:
            # Want to add bits: prioritize high-eigenvalue, under-max dims
            order = np.argsort(-(ev * (bits < max_bits).astype(float)))
        else:
            # Want to remove bits: prioritize low-eigenvalue, above-min dims
            order = np.argsort(ev * (bits > min_bits).astype(float))
        for i in order:
            if diff == 0:
                break
            new_val = int(bits[i]) + (1 if diff > 0 else -1)
            if min_bits <= new_val <= max_bits:
                bits[i] = new_val
                diff += -1 if diff > 0 else 1

    return bits

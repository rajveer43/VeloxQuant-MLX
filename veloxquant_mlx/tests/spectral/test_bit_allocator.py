"""Tests for water_fill_bits (spectral/bit_allocator.py).

Covers the budget-exactness contract: the allocation sums to
total_bit_budget whenever that's feasible under min/max caps, for both
the main iterative (non-degenerate eigenvalues) path and the near-zero
eigenvalue / non-positive budget early-exit path.
"""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.spectral.bit_allocator import water_fill_bits


# ---------------------------------------------------------------------------
# Regression for #75: the near-zero-eigenvalue early-exit branch used floor
# division with no remainder distribution, silently dropping
# total_bit_budget % d bits whenever d didn't evenly divide the budget.
# ---------------------------------------------------------------------------
def test_zero_eigenvalues_exact_budget_issue_repro():
    """Issue #75's exact repro: sum must equal the requested budget, not
    floor(budget / d)."""
    bits = water_fill_bits(np.array([0.0, 0.0]), total_bit_budget=5, min_bits=1, max_bits=6)
    assert bits.sum() == 5
    assert list(bits) in ([3, 2], [2, 3])  # remainder goes to one dim, either is valid


def test_zero_eigenvalues_exact_budget_sweep():
    """Broader sweep over d, budget, and min/max caps -- all-zero eigenvalues
    must still hit the exact feasible budget."""
    for d in range(2, 12):
        for budget in range(0, d * 6 + 3):
            bits = water_fill_bits(np.zeros(d), total_bit_budget=budget, min_bits=1, max_bits=6)
            feasible = max(d * 1, min(d * 6, budget))
            assert bits.sum() == feasible, (d, budget, bits.tolist())
            assert np.all(bits >= 1) and np.all(bits <= 6)


def test_near_zero_eigenvalues_exact_budget():
    """Eigenvalues just under the 1e-12 near-zero threshold hit the same
    early-exit branch and must still be budget-exact."""
    ev = np.full(7, 1e-13)
    bits = water_fill_bits(ev, total_bit_budget=20, min_bits=1, max_bits=8)
    assert bits.sum() == 20


def test_non_positive_budget_still_respects_min_bits_floor():
    """total_bit_budget <= 0 also hits the early-exit branch: the achievable
    result is d * min_bits (can't go below the floor), not the (infeasible)
    requested budget -- this is correct, not a regression."""
    bits = water_fill_bits(np.zeros(4), total_bit_budget=0, min_bits=2, max_bits=8)
    assert bits.sum() == 4 * 2
    assert np.all(bits == 2)

    bits_neg = water_fill_bits(np.zeros(4), total_bit_budget=-10, min_bits=2, max_bits=8)
    assert bits_neg.sum() == 4 * 2


def test_zero_eigenvalues_randomized_budget_feasibility_sweep():
    """Randomized sweep mirroring the issue's 2000-trial check: across d,
    min_bits/max_bits, and budget, the near-zero-eigenvalue path must hit
    exactly the feasible target (budget clamped to [d*min_bits, d*max_bits])."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        d = int(rng.integers(2, 40))
        min_bits = int(rng.integers(0, 3))
        max_bits = int(rng.integers(min_bits + 1, min_bits + 10))
        budget = int(rng.integers(0, d * max_bits + 1))
        bits = water_fill_bits(np.zeros(d), budget, min_bits=min_bits, max_bits=max_bits)
        feasible = max(d * min_bits, min(d * max_bits, budget))
        assert bits.sum() == feasible, (d, min_bits, max_bits, budget, bits.tolist())
        assert np.all(bits >= min_bits) and np.all(bits <= max_bits)


# ---------------------------------------------------------------------------
# Non-degenerate path stays exact too (already correct pre-fix; guards
# against a regression from routing the zero-eigenvalue case through the
# shared "final fix" correction block).
# ---------------------------------------------------------------------------
def test_nondegenerate_eigenvalues_exact_budget():
    ev = np.array([10.0, 5.0, 1.0, 0.1])
    bits = water_fill_bits(ev, total_bit_budget=11, min_bits=1, max_bits=8)
    assert bits.sum() == 11

"""Tests for the KVTC DP-optimal per-component bit allocator (allocators/kvtc_dp.py).

Covers: init guards, the uniform-variance collapse (the pinned reduction —
analogue of SVDq's fixed-split baseline / MorphKV's window=1==TOVA / KVzip's
probe="latest"==TOVA collapses), budget respected exactly / local optimality
against a brute-force reference on small n, monotonicity (higher variance
never gets fewer bits), can-assign-exactly-0, and determinism (no RNG).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from veloxquant_mlx.allocators.kvtc_dp import (
    DEFAULT_BETA,
    DEFAULT_BIT_CHOICES,
    _distortion,
    dp_allocate_bits,
)

_CONTIGUOUS_CHOICES = tuple(range(0, 9))  # 0..8, no gaps


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_rejects_negative_budget():
    with pytest.raises(ValueError, match="total_bit_budget must be >= 0"):
        dp_allocate_bits(np.array([1.0, 2.0]), total_bit_budget=-1)


def test_rejects_negative_variance():
    with pytest.raises(ValueError, match="non-negative"):
        dp_allocate_bits(np.array([1.0, -0.5]), total_bit_budget=4)


def test_rejects_empty_variances():
    with pytest.raises(ValueError, match="non-empty"):
        dp_allocate_bits(np.array([]), total_bit_budget=4)


def test_rejects_empty_bit_choices():
    with pytest.raises(ValueError, match="bit_choices must be non-empty"):
        dp_allocate_bits(np.array([1.0, 2.0]), total_bit_budget=4, bit_choices=())


def test_rejects_negative_bit_choice():
    with pytest.raises(ValueError, match="non-negative"):
        dp_allocate_bits(np.array([1.0]), total_bit_budget=4, bit_choices=(-1, 0, 2))


def test_budget_zero_gives_all_zero_bits():
    bits = dp_allocate_bits(np.array([5.0, 1.0, 9.0]), total_bit_budget=0)
    assert np.array_equal(bits, np.zeros(3, dtype=bits.dtype))


# ---------------------------------------------------------------------------
# uniform-variance collapse — THE pinned reduction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,budget",
    [
        (5, 0),
        (5, 1),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 11),
        (5, 17),
        (7, 0),
        (7, 13),
        (7, 7 * 8),
        (1, 3),
        (1, 0),
    ],
)
def test_uniform_variance_collapses_to_floor_plus_remainder(n, budget):
    """With equal variances and a CONTIGUOUS bit_choices range, the DP-optimal
    allocation is exactly floor(budget/n) per component, with the remainder
    (budget % n) distributed one extra bit to the first components in index
    order — the same allocation a naive uniform splitter would produce.
    """
    v = np.full(n, 3.7)  # any positive constant
    bits = dp_allocate_bits(v, total_bit_budget=budget, bit_choices=_CONTIGUOUS_CHOICES)

    base = budget // n
    rem = budget % n
    expected = np.array([base + 1 if i < rem else base for i in range(n)])
    assert np.array_equal(bits, expected), f"got {bits}, expected {expected}"


def test_uniform_variance_collapse_matches_naive_uniform_splitter_helper():
    """Same claim, phrased as an explicit comparison against a hand-written
    'naive uniform splitter' rather than an inline expected array."""

    def naive_uniform_split(n: int, budget: int) -> np.ndarray:
        base, rem = divmod(budget, n)
        return np.array([base + 1 if i < rem else base for i in range(n)])

    for n, budget in [(4, 10), (6, 6), (10, 37)]:
        v = np.full(n, 1.0)
        bits = dp_allocate_bits(v, total_bit_budget=budget, bit_choices=_CONTIGUOUS_CHOICES)
        assert np.array_equal(bits, naive_uniform_split(n, budget))


# ---------------------------------------------------------------------------
# budget respected + local/global optimality (brute force on small n)
# ---------------------------------------------------------------------------
def test_budget_never_exceeded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(1, 12)
        v = rng.exponential(2.0, size=n)
        budget = int(rng.integers(0, 40))
        bits = dp_allocate_bits(v, total_bit_budget=budget)
        assert bits.sum() <= budget
        assert all(b in DEFAULT_BIT_CHOICES for b in bits.tolist())


def _brute_force_optimum(
    v: np.ndarray, budget: int, choices: tuple[int, ...], beta: float
) -> float:
    """Exhaustive search over all bit assignments for small n — returns the
    minimum achievable total distortion (not the argmin allocation, since
    ties make the allocation itself non-unique; the DP is checked against
    the achieved distortion VALUE, which is unique).
    """
    n = len(v)
    best = float("inf")
    for combo in itertools.product(choices, repeat=n):
        if sum(combo) > budget:
            continue
        total = sum(_distortion(v[i], combo[i], beta) for i in range(n))
        best = min(best, total)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_optimum_small_n(seed):
    rng = np.random.default_rng(seed)
    n = 4
    v = rng.exponential(2.0, size=n)
    choices = (0, 1, 2, 3, 4)
    budget = int(rng.integers(0, 4 * n))

    bits = dp_allocate_bits(v, total_bit_budget=budget, bit_choices=choices)
    achieved = sum(_distortion(v[i], int(bits[i]), DEFAULT_BETA) for i in range(n))
    optimum = _brute_force_optimum(v, budget, choices, DEFAULT_BETA)

    assert bits.sum() <= budget
    assert achieved == pytest.approx(optimum, rel=1e-9, abs=1e-12)


def test_local_optimality_no_single_reallocation_improves():
    """No feasible single-component bit-swap (one component +delta, another
    -delta, staying within bit_choices and the budget) can lower total
    distortion — a necessary condition for a global optimum, checked
    directly since the DP claims exact optimality.
    """
    rng = np.random.default_rng(1)
    v = rng.exponential(2.0, size=6)
    choices = DEFAULT_BIT_CHOICES
    budget = 20
    bits = dp_allocate_bits(v, total_bit_budget=budget, bit_choices=choices)
    base_distortion = sum(_distortion(v[i], int(bits[i]), DEFAULT_BETA) for i in range(6))

    for i, j in itertools.permutations(range(6), 2):
        for c_i in choices:
            for c_j in choices:
                trial = bits.copy()
                trial[i] = c_i
                trial[j] = c_j
                if trial.sum() > budget:
                    continue
                d = sum(_distortion(v[k], int(trial[k]), DEFAULT_BETA) for k in range(6))
                assert d >= base_distortion - 1e-9, (
                    f"reallocation {trial} beats DP allocation {bits}: {d} < {base_distortion}"
                )


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------
def test_monotonic_higher_variance_never_fewer_bits():
    rng = np.random.default_rng(2)
    for _ in range(10):
        n = 8
        v = np.sort(rng.exponential(2.0, size=n))[::-1]  # descending
        budget = int(rng.integers(n, 6 * n))
        bits = dp_allocate_bits(v, total_bit_budget=budget)
        # v is sorted descending, so bits should be non-increasing.
        assert all(bits[i] >= bits[i + 1] for i in range(n - 1)), (v, bits)


# ---------------------------------------------------------------------------
# can assign exactly 0 to a near-zero-variance component under a tight budget
# ---------------------------------------------------------------------------
def test_near_zero_variance_component_gets_zero_bits():
    v = np.array([10.0, 10.0, 1e-6])
    bits = dp_allocate_bits(v, total_bit_budget=4)
    assert bits[2] == 0
    assert bits[0] > 0 and bits[1] > 0


def test_zero_variance_component_gets_zero_bits_under_a_tight_budget():
    """When the budget is exactly enough to saturate the informative
    component, a zero-variance component (whose distortion is 0 regardless
    of bit-width — nothing left to encode) should not be given any of that
    scarce budget.
    """
    v = np.array([5.0, 0.0])
    bits = dp_allocate_bits(v, total_bit_budget=8, bit_choices=_CONTIGUOUS_CHOICES)
    assert bits[1] == 0
    assert bits[0] == 8  # all useful budget goes to the informative component


def test_zero_variance_component_never_worsens_informative_components_share():
    """With slack budget beyond what any component can usefully absorb
    (max bit_choice reached), a zero-variance component MAY receive the
    leftover bits (its distortion is 0 either way, so this is not a
    sub-optimal allocation) — but the informative component must still
    reach the maximum bit-width available, i.e. the zero-variance component
    never steals budget the informative one could still use.
    """
    v = np.array([5.0, 0.0])
    bits = dp_allocate_bits(v, total_bit_budget=16, bit_choices=_CONTIGUOUS_CHOICES)
    assert bits[0] == 8  # informative component saturated at max bit_choice


# ---------------------------------------------------------------------------
# determinism (no RNG in the DP itself)
# ---------------------------------------------------------------------------
def test_deterministic_repeated_calls():
    v = np.array([4.0, 1.0, 9.0, 0.2, 6.0])
    a = dp_allocate_bits(v, total_bit_budget=20)
    b = dp_allocate_bits(v, total_bit_budget=20)
    assert np.array_equal(a, b)


def test_deterministic_across_fresh_processes_equivalent_call():
    """Same inputs called independently (simulating a fresh call site) give
    bit-for-bit identical output — no hidden global state."""
    v1 = np.array([2.0, 2.0, 2.0])
    v2 = np.array([2.0, 2.0, 2.0])
    assert np.array_equal(
        dp_allocate_bits(v1, total_bit_budget=9),
        dp_allocate_bits(v2, total_bit_budget=9),
    )

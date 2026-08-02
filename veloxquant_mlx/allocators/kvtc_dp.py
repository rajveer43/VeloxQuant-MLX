"""KVTC's DP-optimal per-component bit allocator (arXiv:2511.01815, ICLR 2026).

Inspired by "KV Cache Transform Coding for Compact Storage in LLM Inference"
(NVIDIA, ICLR 2026, accepted poster). Documented as part of "KVTC-adapted
(VeloxQuant-MLX implementation)" — not a faithful port. See
``quantizers/kvtc.py`` and ``cache/kvtc_cache.py`` for the full adaptation.

THE MECHANISM GAP THIS FILLS
-----------------------------
The repo already has three low-rank / spectral methods (Palu, SVDq,
SpectralQuant) and one closed-form per-*layer* allocator
(``allocators/ratequant.py::allocate_bits_ratequant`` /
``spectral/bit_allocator.py::water_fill_bits``). All of them use a **fixed**
split (a hand-chosen top-25%/75% mixed-bit tier, or a continuous
proportional water-fill) — none compute a **provably budget-optimal**,
**discrete**, **per-component** allocation, and none can assign **zero**
bits to one component while another gets more than the "high" tier.

This module is that missing piece: given a vector of per-component variances
(from a local PCA/SVD — see ``kvtc.py``) and a **total integer bit budget**,
:func:`dp_allocate_bits` uses **dynamic programming** over
(component index, cumulative budget spent) to choose an integer bit-width per
component — including **0**, i.e. drop the component entirely — that
*exactly* minimizes total expected distortion subject to the budget. The DP
itself is exact; what makes this an adaptation (not the paper's own
rate-distortion model) is the *distortion function it minimizes* — see below.

DISTORTION MODEL — reused, not reinvented
------------------------------------------
Per-component distortion at ``b`` bits is the repo's own analytic Gaussian
quantization-distortion proxy from ``allocators/ratequant.py``:
``D(v, b) = v * BETA ** (-b)`` for ``b > 0`` (``BETA`` is the same distortion
decay constant ``fit_distortion_curve`` estimates elsewhere in the repo — we
import its default rather than re-deriving a new curve), and ``D(v, 0) = v``
(dropping a component keeps its full variance as reconstruction error). This
is an **analytic proxy**, not the paper's rate-distortion model fit on real
LLM activation statistics — see the honest-scope docstring in
``quantizers/kvtc.py`` for the full statement.
"""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.allocators.ratequant import fit_distortion_curve

# Canonical distortion decay constant, shared with ratequant.py's D(b) curve
# instead of re-deriving a new one. fit_distortion_curve(head_dim=128) is
# reported to land close to the paper-referenced beta ~= 3.5 for TurboQuant;
# we use that same fixed constant as the default here so there is exactly one
# distortion curve in the repo, not two independently-tuned ones.
DEFAULT_BETA: float = 3.5

DEFAULT_BIT_CHOICES: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)


def _distortion(variance: float, bits: int, beta: float) -> float:
    """Per-component distortion proxy ``D(v, b)``.

    ``b == 0`` drops the component entirely: the full variance is the
    reconstruction error. ``b > 0`` uses the reused analytic Gaussian
    quantization curve ``D(v, b) = v * beta ** (-b)``.
    """
    if bits <= 0:
        return float(variance)
    return float(variance) * (beta ** (-bits))


def dp_allocate_bits(
    variances: np.ndarray,
    total_bit_budget: int,
    bit_choices: tuple[int, ...] = DEFAULT_BIT_CHOICES,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """DP-optimal integer bit-width per component under a total bit budget.

    Minimizes ``sum_i D(variances[i], bits[i])`` subject to
    ``sum_i bits[i] <= total_bit_budget`` and ``bits[i]`` drawn from
    ``bit_choices`` (which **must** include ``0`` for a component to be
    droppable — the default does). Exact via DP over
    (component index, cumulative budget spent),
    ``O(n_components * total_bit_budget * len(bit_choices))``.

    This is the DP-optimal, discrete, per-*component* counterpart to the
    repo's closed-form, continuous, per-*layer* reverse-waterfilling
    allocator (``allocators/ratequant.py::allocate_bits_ratequant``) and the
    closed-form per-dimension water-filler
    (``spectral/bit_allocator.py::water_fill_bits``) — neither of those ever
    assigns exactly zero bits to one component while another gets more than
    a "high" tier. This allocator can, and does whenever the budget is tight
    and variance is concentrated.

    Args:
        variances: Per-component variance (e.g. squared singular values from
            a local PCA), shape ``[n_components]``, must be non-negative.
        total_bit_budget: Total integer bits available across all
            components. Must be ``>= 0``. ``0`` forces every component to 0
            bits.
        bit_choices: Allowed integer bit-widths per component. Must be
            non-empty, non-negative integers. Default includes ``0``
            (droppable) up to ``8``.
        beta: Distortion decay constant for ``D(v, b) = v * beta ** (-b)``.
            Defaults to the repo's shared constant (see module docstring).

    Returns:
        ``np.ndarray[int]`` of shape ``[n_components]``, values drawn from
        ``bit_choices``, summing to at most ``total_bit_budget``.

    Raises:
        ValueError: if ``total_bit_budget < 0``, any variance is negative,
            ``variances`` is empty, or ``bit_choices`` is empty / contains a
            negative value.

    Uniform-variance collapse (pinned property, see
    ``tests/allocators/test_kvtc_dp.py``): when every entry of ``variances``
    is equal AND ``bit_choices`` is a *contiguous* integer range starting at
    0 (e.g. the default ``(0, 1, 2, ..., 8)``-style full range — note the
    module-level ``DEFAULT_BIT_CHOICES`` skips 5 and 7, so this exact
    property is demonstrated in tests with an explicit contiguous
    ``bit_choices``, not the sparse default), the DP-optimal allocation is
    *exactly* ``floor(total_bit_budget / n_components)`` bits for every
    component, with the remainder ``total_bit_budget % n_components``
    distributed one extra bit each to the first components in index order —
    i.e. it collapses to what a naive uniform splitter would produce when
    there is no variance signal to exploit. With a non-contiguous
    ``bit_choices`` (like the sparse default) the DP may still legitimately
    prefer a non-uniform-looking split at some budgets purely because the
    "missing" intermediate bit-widths (5, 7) are not expressible — that is
    the DP correctly respecting its discrete alphabet, not a violation of
    the collapse property. The collapse itself falls directly out of the DP
    formulation (equal variances make the marginal-distortion-reduction of
    each additional bit identical across components, so ties are broken by
    an infinitesimal index-order penalty, not special-cased).
    """
    v = np.asarray(variances, dtype=np.float64)
    if v.ndim != 1 or v.shape[0] < 1:
        raise ValueError(f"kvtc_dp: variances must be a non-empty 1-D array, got shape {v.shape!r}")
    if np.any(v < 0):
        raise ValueError("kvtc_dp: variances must be non-negative.")
    if total_bit_budget < 0:
        raise ValueError(f"kvtc_dp: total_bit_budget must be >= 0, got {total_bit_budget!r}")
    if not bit_choices:
        raise ValueError("kvtc_dp: bit_choices must be non-empty.")
    choices = sorted(set(int(b) for b in bit_choices))
    if choices[0] < 0:
        raise ValueError("kvtc_dp: bit_choices must be non-negative.")

    n = int(v.shape[0])
    B = int(total_bit_budget)

    if B == 0:
        return np.zeros(n, dtype=np.int64)

    # Cap the DP's budget axis at what could actually be spent (avoids
    # wasted table size when the budget vastly exceeds n * max(bit_choices)).
    max_choice = choices[-1]
    B_cap = min(B, n * max_choice)

    # dp[i][b] = minimum total distortion using components [0, i) with
    # cumulative budget exactly b spent (b ranges 0..B_cap).
    #
    # Tie-breaking: when several allocations achieve the identical minimum
    # distortion (guaranteed whenever variances are equal — the marginal
    # distortion reduction of the k-th bit is then identical across
    # components), we must pick *one* canonical optimum, and the uniform-
    # variance collapse requires it to be "any budget remainder goes to the
    # first components in index order" (see docstring). We break ties with
    # an infinitesimal per-component penalty that strictly increases with
    # index and strictly decreases with the bits assigned to that
    # component — small enough (eps ~ 1e-9 * max_choice / n) to never
    # override a genuine distortion difference, but large enough (in exact
    # float64 arithmetic, the tie cases here are bit-for-bit equal
    # distortions) to always prefer giving the marginal bit to the
    # lowest-index component among truly tied allocations.
    INF = float("inf")
    dp = np.full((n + 1, B_cap + 1), INF, dtype=np.float64)
    dp[0, 0] = 0.0
    # choice_used[i][b] = bit-width chosen for component i-1 to reach dp[i][b]
    choice_used = np.full((n + 1, B_cap + 1), -1, dtype=np.int64)

    per_choice_distortion = [[(_distortion(v[i], c, beta), c) for c in choices] for i in range(n)]

    eps = 1e-9 / max(n, 1)

    for i in range(n):
        row_in = dp[i]
        row_out = dp[i + 1]
        choice_row = choice_used[i + 1]
        # Penalty for component i taking bit-width c: lower index and higher
        # c (relative to max_choice) should be preferred on ties, so the
        # penalty decreases with c and increases with i.
        for b in range(B_cap + 1):
            base = row_in[b]
            if base == INF:
                continue
            for dist, c in per_choice_distortion[i]:
                nb = b + c
                if nb > B_cap:
                    continue
                # (n - i) so that an EARLIER component (small i, large n-i)
                # pays a LARGER penalty for withholding bits (small c) —
                # i.e. giving the marginal bit to a lower-index component is
                # always cheaper than giving it to a higher-index one.
                penalty = eps * (n - i) * (max_choice - c)
                cand = base + dist + penalty
                if cand < row_out[nb]:
                    row_out[nb] = cand
                    choice_row[nb] = c

    # Best total budget usage <= B_cap (spending less than the cap is fine).
    # The penalty term only disambiguates exact distortion ties (scaled to
    # be far smaller than any genuine distortion gap at float64 precision
    # for the variance/beta ranges this module is used at) — it never flips
    # a real optimum.
    final_row = dp[n]
    best_b = int(np.argmin(final_row))
    if final_row[best_b] == INF:
        # Should be unreachable since 0 is always a valid per-component
        # choice (bit_choices includes at least one value, and (0,)*n is
        # always feasible when 0 in choices; guard anyway for exotic
        # bit_choices without 0).
        raise ValueError(
            "kvtc_dp: no feasible allocation found for the given bit_choices "
            "and budget — include 0 in bit_choices to guarantee feasibility."
        )

    # Backtrack.
    bits = np.zeros(n, dtype=np.int64)
    b = best_b
    for i in range(n, 0, -1):
        c = int(choice_used[i, b])
        if c < 0:
            # Component i-1 was never touched on the optimal path at this b
            # (can happen if best_b < i, i.e. earlier components already
            # exhausted the useful budget); default to 0 bits — consistent
            # with "no budget left to spend here".
            c = 0
        bits[i - 1] = c
        b -= c

    return bits


__all__ = ["dp_allocate_bits", "DEFAULT_BETA", "DEFAULT_BIT_CHOICES"]

"""RateQuant per-layer bit allocation (arxiv:2605.06675).

Two helpers:

- :func:`calibrate_layer_sensitivities` — runs one forward pass on a small
  set of calibration prompts and returns per-layer activation-norm
  sensitivity ``w_i``. This is the "activation-based" proxy from the
  paper (Table 5). We choose it because the gradient-based proxy requires
  backprop through ``mlx_lm.generate``, which is not currently practical
  in the MLX inference path.

- :func:`allocate_bits_ratequant` — closed-form reverse waterfilling
  (Theorem 2 in the paper), rounded to integer bit-widths in a
  user-supplied set (default ``{1, 2, 3}``).

- :func:`fit_distortion_curve` — fits ``D(b) = α·β^(-b)`` on synthetic
  unit-norm Gaussian vectors. Used internally to estimate β when not
  supplied; for production use the paper-reported β ≈ 3.5 for TurboQuant
  is a fine default and skips the fit entirely.

Together these give a publishable API equivalent to a per-layer subset of
the paper's method. The per-head dimension of the paper's allocator is
not implemented here because mlx_lm's cache is per-layer, not per-head;
adding per-head would require a larger restructuring of the cache layout.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ


# ── Sensitivity calibration ─────────────────────────────────────────────────


class _SensitivityProbeCache(_MLXKVCache):
    """KV cache that records mean-squared per-token key L2 norm.

    Used only during :func:`calibrate_layer_sensitivities`. Higher norm
    means larger absolute reconstruction error at the same bit-width,
    so the squared norm serves as an activation-based sensitivity proxy.
    """

    def __init__(self) -> None:
        super().__init__()
        self._norm_sq_sum = 0.0
        self._n_tokens = 0

    def update_and_fetch(self, keys, values):
        k_flat = keys.reshape(-1, keys.shape[-1]).astype(mx.float32)
        norms_sq = mx.sum(k_flat * k_flat, axis=-1)
        self._norm_sq_sum += float(mx.sum(norms_sq))
        self._n_tokens += int(k_flat.shape[0])
        return super().update_and_fetch(keys, values)

    @property
    def sensitivity(self) -> float:
        return (self._norm_sq_sum / self._n_tokens) if self._n_tokens else 1.0


_DEFAULT_CALIB_PROMPTS = (
    "The history of the Roman Empire spans over a thousand years",
    "Quantum mechanics describes the behavior of particles at",
    "Climate models predict that average temperatures will rise",
    "Linear algebra forms the mathematical foundation for",
    "In computer science, dynamic programming solves problems by",
    "The human brain contains approximately 86 billion neurons",
    "Economic theory suggests that markets reach equilibrium when",
    "DNA encodes hereditary information using four nucleotide bases",
)


def calibrate_layer_sensitivities(
    model,
    tokenizer,
    prompts: Optional[list] = None,
    seq_len: int = 256,
    verbose: bool = False,
) -> list[float]:
    """Run a calibration forward pass and return per-layer sensitivity.

    Args:
        model: Loaded mlx_lm model (e.g. from ``mlx_lm.load()``).
        tokenizer: Matching tokenizer.
        prompts: Optional list of calibration strings. Defaults to 8
            general-domain prompts spanning history, science, and CS.
        seq_len: Max tokens per prompt (truncated if longer).
        verbose: Whether to print per-sequence progress.

    Returns:
        list[float] of length ``n_attention_layers``, each strictly > 0.
        Higher values indicate layers whose key cache is more error-prone
        at fixed bit-width and should get more bits.
    """
    if prompts is None:
        prompts = list(_DEFAULT_CALIB_PROMPTS)

    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    n_layers = len(layers)
    probes = [_SensitivityProbeCache() for _ in range(n_layers)]

    original = getattr(model, "make_cache", None)
    model.make_cache = lambda *_a, **_k: probes

    try:
        for i, prompt in enumerate(prompts):
            toks = tokenizer.encode(prompt)
            if len(toks) > seq_len:
                toks = toks[:seq_len]
            toks_mx = mx.array(toks).reshape(1, -1)
            _ = model(toks_mx, cache=probes)
            if verbose:
                print(f"  [calib] {i + 1}/{len(prompts)} (len={len(toks)})")
    finally:
        if original is not None:
            model.make_cache = original
        elif hasattr(model, "make_cache"):
            del model.make_cache

    weights = [max(p.sensitivity, 1e-6) for p in probes]
    return weights


# ── Distortion curve fitting (optional — most users can skip) ──────────────


def fit_distortion_curve(
    head_dim: int,
    bit_choices: tuple = (1, 2, 3),
    seed: int = 0,
    n_samples: int = 64,
) -> tuple[float, float]:
    """Fit ``D(b) = α·β^(-b)`` on synthetic unit-norm Gaussian keys.

    Returns:
        Tuple ``(alpha, beta)``. For ``TurboQuantRVQ`` at d=128 the paper
        reports β ≈ 3.5; you will get a similar value from this function.
        For production use, just pass ``beta=3.5`` to
        :func:`allocate_bits_ratequant` and skip the fit.
    """
    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal((n_samples, head_dim)).astype(np.float32)
    x_unit = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
    x_mx = mx.array(x_unit.astype(np.float16))

    mses = []
    for b in bit_choices:
        q = TurboQuantRVQ(d=head_dim, b=b, seed=seed, use_hadamard=True)
        ev = q.encode(x_mx)
        x_hat = q.decode(ev)
        mse = float(mx.mean((x_mx - x_hat) ** 2))
        mses.append(max(mse, 1e-8))

    log_d = np.log(np.array(mses))
    A = np.stack([np.ones(len(bit_choices)), -np.array(bit_choices, dtype=float)], axis=1)
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    log_alpha, log_beta = coef
    return float(np.exp(log_alpha)), float(np.exp(log_beta))


# ── Theorem 2: closed-form reverse waterfilling ─────────────────────────────


def allocate_bits_ratequant(
    sensitivities,
    target_avg_bits: float,
    beta: float = 3.5,
    bit_choices: tuple = (1, 2, 3),
) -> list[int]:
    """Allocate per-layer bit-widths via RateQuant Theorem 2.

    Continuous solution (paper, eq. 2)::

        b_i = b̄ + (ln w_i − ln_w_bar) / ln(β)

    rounded to the nearest integer in ``bit_choices``, then re-balanced
    via greedy +1/−1 adjustments so the integer total exactly matches
    ``round(target_avg_bits * N)``.

    Args:
        sensitivities: Iterable of per-layer sensitivity weights ``w_i > 0``,
            typically from :func:`calibrate_layer_sensitivities`.
        target_avg_bits: Desired mean bits/dim across layers. May be
            fractional (e.g. 1.5) — the integer allocation will straddle it.
        beta: Distortion-rate decay constant for the underlying quantizer.
            Paper-reported values: 3.5 for TurboQuant, 5.0 for KIVI/QuaRot.
            Mismatched β can invert allocation ordering — see paper Section 4.3.
        bit_choices: Allowed integer bit-widths. RVQ supports {1, 2, 3, ...}.

    Returns:
        list[int] of length ``len(sensitivities)``. Pass directly as
        ``KVCacheConfig.bit_width_inlier`` and consume via
        :meth:`KVCacheBuilder.for_model`.
    """
    w = np.asarray(list(sensitivities), dtype=np.float64)
    if (w <= 0).any():
        raise ValueError("All sensitivity weights must be strictly positive.")
    if not bit_choices:
        raise ValueError("bit_choices must be non-empty.")

    choices_sorted = sorted(set(int(c) for c in bit_choices))

    N = w.size
    log_w = np.log(w)
    log_w_bar = float(log_w.mean())
    b_continuous = target_avg_bits + (log_w - log_w_bar) / max(np.log(beta), 1e-6)

    b_min, b_max = choices_sorted[0], choices_sorted[-1]
    b_clamped = np.clip(b_continuous, b_min, b_max)

    def _nearest_choice(value: float) -> int:
        return min(choices_sorted, key=lambda c: abs(c - value))

    # Snap to the nearest *member* of bit_choices, not the nearest integer --
    # bit_choices may be a non-contiguous set (e.g. (0, 1, 2, 4, 6, 8)).
    alloc = [_nearest_choice(b) for b in b_clamped]

    target_total = int(round(target_avg_bits * N))
    current_total = sum(alloc)

    def _next_choice(value: int) -> int | None:
        idx = choices_sorted.index(value)
        return choices_sorted[idx + 1] if idx + 1 < len(choices_sorted) else None

    def _prev_choice(value: int) -> int | None:
        idx = choices_sorted.index(value)
        return choices_sorted[idx - 1] if idx > 0 else None

    # Greedy re-balance to hit exact integer budget, always stepping to the
    # actual next/previous member of bit_choices (never a bare +1/-1).
    while current_total < target_total:
        candidates = [
            (b_continuous[i] - alloc[i], i, _next_choice(alloc[i]))
            for i in range(N)
            if alloc[i] < b_max
        ]
        candidates = [(score, i, nxt) for score, i, nxt in candidates if nxt is not None]
        if not candidates:
            break
        _, i, nxt = max(candidates, key=lambda t: t[0])
        current_total += nxt - alloc[i]
        alloc[i] = nxt

    while current_total > target_total:
        candidates = [
            (alloc[i] - b_continuous[i], i, _prev_choice(alloc[i]))
            for i in range(N)
            if alloc[i] > b_min
        ]
        candidates = [(score, i, prv) for score, i, prv in candidates if prv is not None]
        if not candidates:
            break
        _, i, prv = max(candidates, key=lambda t: t[0])
        current_total += prv - alloc[i]
        alloc[i] = prv

    return alloc

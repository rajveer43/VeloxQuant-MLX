"""PALU quantizer — low-rank K *and* V projection with group-head decomposition.

Inspired by "PALU: Compressing KV-Cache with Low-Rank Projection"
(arXiv:2407.21118, ICLR 2025, Chang et al.).  Documented as
"PALU-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

What makes this PALU and not SVDq (already in the repo):

  * Both **keys and values** are low-rank projected (SVDq is keys-only,
    values fp16).
  * The cache stores the latent codes ``[S, r]`` *directly* — it never holds
    full fp16 ``[S, D]`` keys/values for storage; the parent fp16 buffer is
    bypassed entirely (see :class:`~veloxquant_mlx.cache.palu_cache.PALUKVCache`).
    SVDq reconstructs full fp16 keys and hands them to the parent cache, so its
    win is bandwidth-accounting only.  PALU stores ``[S, r]`` and reconstructs
    to fp16 *only* at attend time, so it also wins peak storage.
  * **Group-head low-rank decomposition (G-LRD):** heads are partitioned into
    ``n_head_groups`` groups; all heads in a group share one projection matrix
    fit from their concatenated keys (resp. values).  This is PALU's middle
    ground between a single whole-matrix projection (max compression, worst
    fidelity) and per-head projections (best fidelity, projection overhead
    grows with head count).
  * The stored latents are further **mixed-bit quantized** (reusing the SVDq
    latent quantizer) so the compression is ``(D / r) * (16 / avg_bits)`` on
    both tensors.

Algorithm:

  Prefill (first call, S > 1):
    1. For keys and (optionally) values independently:
       a. Partition the H heads into ``n_head_groups`` contiguous groups.
       b. For each group g, stack the group's heads along the token axis and
          compute a truncated SVD → projection ``V_g [D, r]`` + mean ``mu_g [D]``.
       c. Project every head in the group into its group's latent space and
          mixed-bit quantize the resulting ``[S, r]`` latents.
    2. Store ``V_g``, ``mu_g``, singular values per group, and the quantized
       latents as cache state.

  Decode (S == 1):
    1. Project the new key/value into the already-stored group projections.
    2. Mixed-bit quantize and append to the latent buffers.

This module holds the pure projection/reconstruction maths.  The cache wrapper
(:class:`PALUKVCache`) owns the per-group state and the true-latent storage.

Adaptation notes:
  - Rank defaults to ``energy_threshold = 0.90`` (a touch more aggressive than
    SVDq's 0.95, since PALU pays the rank cost on both tensors).
  - Values use the same group-head SVD path; set ``quantize_values=False`` at
    the cache level for a low-rank-only (fp16 latent) value path.
  - PALU's fused low-rank-reconstruction attention CUDA kernel is *not* ported:
    we reconstruct fp16 then call MLX SDPA.  Documented as a known simplification.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from numbers import Integral, Real
from typing import Any

import mlx.core as mx
import numpy as np

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers._quant_utils import (
    _group_quant_dequant,
    _group_quant_dequant_batched,
)

_MAX_BITS = 8


def _as_int(value: object) -> int | None:
    """``value`` as an int if it is integral (``8`` or ``8.0``), else ``None``.

    Configs routinely carry whole numbers as floats (JSON/YAML), and the old
    ``int(...)`` coercion accepted them, so they must keep working. A fractional
    value (``2.5``) is rejected rather than truncated; bools, strings, NaN and
    Inf are not integers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return int(f)
    return None


def _validate_int(value: object, name: str, *, lo: int, hi: int | None = None) -> int:
    n = _as_int(value)
    if n is None or n < lo or (hi is not None and n > hi):
        bound = f"an integer >= {lo}" if hi is None else f"an integer in [{lo}, {hi}]"
        raise QuantizerConfigError(f"palu: {name}={value!r} must be {bound}.")
    return n


def _validate_bits(bits: object, name: str) -> int:
    return _validate_int(bits, name, lo=1, hi=_MAX_BITS)


def _validate_group_size(group_size: object, name: str = "group_size") -> int:
    return _validate_int(group_size, name, lo=1)


def _validate_rank(rank: object, name: str = "rank") -> int | None:
    """``None`` (energy-chosen) or an integer ``>= 1`` (a latent needs a channel)."""
    return None if rank is None else _validate_int(rank, name, lo=1)


def _validate_n_groups(n_groups: object, name: str = "n_head_groups") -> int:
    return _validate_int(n_groups, name, lo=1)


def _validate_energy_threshold(t: object, name: str = "energy_threshold") -> float:
    ok = (
        not isinstance(t, bool)
        and isinstance(t, Real)
        and math.isfinite(float(t))
        and 0.0 < float(t) <= 1.0
    )
    if not ok:
        raise QuantizerConfigError(f"palu: {name}={t!r} must be a number in (0, 1].")
    return float(t)


def _validate_hi_fraction(frac: object, name: str = "hi_fraction") -> float:
    ok = (
        not isinstance(frac, bool)
        and isinstance(frac, Real)
        and math.isfinite(float(frac))
        and 0.0 <= float(frac) <= 1.0
    )
    if not ok:
        raise QuantizerConfigError(f"palu: {name} must be in [0, 1], got {frac!r}")
    return float(frac)


def validate_palu_config(config: Any) -> dict[str, Any]:
    """Validate and coerce the ``palu_*`` fields of a :class:`KVCacheConfig`.

    Returns the coerced values keyed by field suffix (``rank``, ``energy``,
    ``n_groups``, ``hi_bit``, ``lo_bit``, ``hi_fraction``, ``group_size``,
    ``quantize_values``, ``min_fit_rows``). Whole-number floats (``8.0``) are
    accepted and coerced to int; everything else that is not valid raises
    :class:`~veloxquant_mlx.core.exceptions.QuantizerConfigError` (a ``ValueError``).
    """

    def get(name: str, default: Any) -> Any:
        return getattr(config, f"palu_{name}", default)

    quantize_values = get("quantize_values", True)
    if not isinstance(quantize_values, (bool, np.bool_)):
        raise QuantizerConfigError(
            f"palu: palu_quantize_values={quantize_values!r} must be a bool."
        )
    return {
        "rank": _validate_rank(get("rank", None), "palu_rank"),
        "energy": _validate_energy_threshold(
            get("energy_threshold", 0.90), "palu_energy_threshold"
        ),
        "n_groups": _validate_n_groups(get("n_head_groups", 4), "palu_n_head_groups"),
        "hi_bit": _validate_bits(get("hi_bit", 4), "palu_hi_bit"),
        "lo_bit": _validate_bits(get("lo_bit", 2), "palu_lo_bit"),
        "hi_fraction": _validate_hi_fraction(get("hi_fraction", 0.25), "palu_hi_fraction"),
        "group_size": _validate_group_size(get("group_size", 32), "palu_group_size"),
        "quantize_values": bool(quantize_values),
        "min_fit_rows": _validate_int(get("min_fit_rows", 0), "palu_min_fit_rows", lo=0),
    }


@lru_cache(maxsize=1024)
def n_hi_channels(r: int, hi_fraction: float) -> int:
    """Number of latent channels (of ``r``) that get ``hi_bit``: ``max(1, floor(r * f))``.

    Evaluated on the decimal value of ``hi_fraction`` rather than its binary
    float, so ``50 * 0.58`` is 29 and not ``int(28.999999999999996) == 28``.
    The result is the same wherever it is used (quantizer, byte accounting).
    """
    return max(1, int(Fraction(repr(float(hi_fraction))) * r))


def head_group_bounds(n_heads: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition ``n_heads`` into ``n_groups`` contiguous ``[lo, hi)`` ranges.

    Groups are as even as possible; the first ``n_heads % n_groups`` groups
    take one extra head.  ``n_groups`` is clamped to ``[1, n_heads]``.

    Returns:
        List of ``(lo, hi)`` half-open head-index ranges, length ``<= n_groups``.
    """
    n_groups = max(1, min(int(n_groups), int(n_heads)))
    base = n_heads // n_groups
    rem = n_heads % n_groups
    bounds: list[tuple[int, int]] = []
    lo = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        bounds.append((lo, lo + size))
        lo += size
    return bounds


def group_head_svd(
    x_group: mx.array,
    rank: int | None = None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fit a shared low-rank projection for a group of heads.

    The group's heads are stacked along the token axis so a single projection
    captures the group's dominant subspace (PALU's group-head decomposition).

    Args:
        x_group: Shape ``[G, S, D]`` (G = heads in this group), fp16 or fp32.
        rank: Explicit latent rank ``r`` (an integer ``>= 1``; clamped down to the
            available rank ``min(G*S, D)``).  If None, chosen by ``energy_threshold``.
        energy_threshold: Fraction of singular-value energy to retain when
            ``rank`` is None; a number in ``(0, 1]``.

    Returns:
        ``(V, mu, singular_values)`` where
          V               — right singular vectors ``[D, r]`` fp32
          mu              — mean vector ``[D]`` fp32 (shared across the group)
          singular_values — ``[r]`` fp32, descending (for mixed-bit ranking)
    """
    if x_group.ndim != 3 or 0 in x_group.shape:
        raise QuantizerConfigError(
            f"palu: x_group must be a non-empty [G, S, D] array, got shape {tuple(x_group.shape)}."
        )
    rank = _validate_rank(rank)
    energy_threshold = _validate_energy_threshold(energy_threshold)
    G, S, D = x_group.shape
    x = x_group.astype(mx.float32).reshape(G * S, D)  # stack heads along tokens
    mu = mx.mean(x, axis=0)  # [D]

    # Only V and the singular values are used, so decompose the D x D Gram
    # matrix instead of SVD-ing the tall [G*S, D] block (#846). It is formed
    # and diagonalised in float64: squaring the conditioning is harmless at
    # that precision, whereas in float32 it would blur the weaker retained
    # directions.
    x_np = np.asarray(x, dtype=np.float64)
    x_np -= x_np.mean(axis=0, keepdims=True)
    gram = x_np.T @ x_np
    # One NaN/Inf would make the shared basis NaN for *every* head in the group
    # (and so every token), with no error. Check the D x D Gram instead of the
    # whole [G*S, D] input: a finite fp16/bf16/fp32 value cannot overflow when
    # squared in float64, so a non-finite input always leaves a non-finite
    # diagonal entry. O(D^2) instead of an extra pass over every row.
    if not np.isfinite(gram).all():
        raise QuantizerConfigError(
            "palu: x_group must be finite (found NaN or Inf); the model produced NaN/Inf KV states."
        )
    evals, evecs = np.linalg.eigh(gram)
    k = min(G * S, D)  # an SVD of an [N, D] block has min(N, D) singular values
    order = np.argsort(evals)[::-1][:k]
    s_vals = mx.array(np.sqrt(np.clip(evals[order], 0.0, None)).astype(np.float32))
    Vt = mx.array(np.ascontiguousarray(evecs[:, order].T).astype(np.float32))
    mx.eval(s_vals, Vt)

    if rank is None:
        total = float(mx.sum(s_vals).item())
        if total < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = int(s_vals.shape[0])
            for i, sv in enumerate(s_vals.tolist()):
                cumsum += sv
                if cumsum / total >= energy_threshold:
                    rank = i + 1
                    break
    rank = max(1, min(rank, int(s_vals.shape[0]), D))

    V = Vt[:rank, :].T  # [D, r]
    s_r = s_vals[:rank]  # [r]
    return V, mu, s_r


def project_to_latent(x: mx.array, V: mx.array, mu: mx.array) -> mx.array:
    """Project ``[S, D]`` into the latent space ``[S, r]``: ``(x - mu) @ V``."""
    return (x.astype(mx.float32) - mu[None, :]) @ V


def reconstruct_from_latent(L: mx.array, V: mx.array, mu: mx.array) -> mx.array:
    """Reconstruct ``[S, D]`` fp16 from latents: ``L @ V.T + mu``."""
    out = L.astype(mx.float32) @ V.T + mu[None, :]
    return out.astype(mx.float16)


def _quant_args_uncached(
    hi_bit: object, lo_bit: object, hi_fraction: object, group_size: object
) -> tuple[int, int, float, int]:
    return (
        _validate_bits(hi_bit, "hi_bit"),
        _validate_bits(lo_bit, "lo_bit"),
        _validate_hi_fraction(hi_fraction),
        _validate_group_size(group_size),
    )


# These four are the same on every decode call (per layer, per group, per tensor),
# so validate each distinct combination once.
# typed=True: otherwise a cached valid ``1`` would also pass for ``True``.
_checked_quant_args = lru_cache(maxsize=256, typed=True)(_quant_args_uncached)


def _validate_quant_args(
    singular_values: mx.array,
    r: int,
    hi_bit: object,
    lo_bit: object,
    hi_fraction: object,
    group_size: object,
) -> tuple[int, int, float, int]:
    if singular_values.shape != (r,):
        raise QuantizerConfigError(
            f"palu: singular_values must have shape ({r},) to match the {r} latent channels, "
            f"got {tuple(singular_values.shape)}."
        )
    try:
        return _checked_quant_args(hi_bit, lo_bit, hi_fraction, group_size)
    except TypeError:  # unhashable argument: take the uncached path, which rejects it
        return _quant_args_uncached(hi_bit, lo_bit, hi_fraction, group_size)


def quantize_latent(
    L: mx.array,
    singular_values: mx.array,
    hi_bit: int,
    lo_bit: int,
    hi_fraction: float,
    group_size: int,
) -> mx.array:
    """Mixed-bit quantize latents ``[S, r]`` → reconstructed fp16 latents ``[S, r]``.

    Top-``hi_fraction`` channels by singular value magnitude get ``hi_bit``;
    the rest get ``lo_bit``. PALU's own 2-tier split — kept separate from
    SVDq's 8-group fixed schedule (``svdq.quantize_latents_mixed``), which
    follows a different paper (arXiv:2502.15304 Eq. 6) with a different
    grouping shape and 0-bit truncation semantics that don't apply here.
    """
    if L.ndim != 2:
        raise QuantizerConfigError(f"palu: L must be [S, r], got shape {tuple(L.shape)}.")
    S, r = L.shape
    hi_bit, lo_bit, hi_fraction, group_size = _validate_quant_args(
        singular_values, r, hi_bit, lo_bit, hi_fraction, group_size
    )
    n_hi = n_hi_channels(r, hi_fraction)
    sv_np = singular_values.tolist()
    sorted_idx = sorted(range(r), key=lambda i: -sv_np[i])
    hi_idx = sorted(sorted_idx[:n_hi])
    lo_idx = sorted(sorted_idx[n_hi:])

    parts = [None] * r
    if hi_idx:
        recon_hi = _group_quant_dequant(L[:, hi_idx], hi_bit, group_size)
        for new_col_idx, orig_col_idx in enumerate(hi_idx):
            parts[orig_col_idx] = recon_hi[:, new_col_idx]
    if lo_idx:
        recon_lo = _group_quant_dequant(L[:, lo_idx], lo_bit, group_size)
        for new_col_idx, orig_col_idx in enumerate(lo_idx):
            parts[orig_col_idx] = recon_lo[:, new_col_idx]
    return mx.stack(parts, axis=1).astype(mx.float16)


def project_to_latent_batched(x: mx.array, V: mx.array, mu: mx.array) -> mx.array:
    """Batched-leading-axis equivalent of :func:`project_to_latent`.

    Args:
        x:  ``[G, S, D]`` — one group's heads (contiguous head-group range),
            fp16 or fp32.
        V:  ``[D, r]`` — this group's shared projection basis.
        mu: ``[D]`` — this group's shared mean.

    Returns:
        ``[G, S, r]`` fp32 latents.
    """
    return (x.astype(mx.float32) - mu[None, None, :]) @ V[None]


def reconstruct_from_latent_batched(
    L: mx.array, V: mx.array, mu: mx.array, out_dtype: mx.Dtype = mx.float16
) -> mx.array:
    """Batched-leading-axis equivalent of :func:`reconstruct_from_latent`.

    Args:
        L:  ``[G, S, r]`` — one group's heads' latents.
        V:  ``[D, r]`` — this group's shared projection basis.
        mu: ``[D]`` — this group's shared mean.

    Returns:
        ``[G, S, D]``, cast to ``out_dtype`` (default fp16).
    """
    out = L.astype(mx.float32) @ mx.swapaxes(V, -1, -2)[None] + mu[None, None, :]
    return out.astype(out_dtype)


def quantize_latent_batched(
    L: mx.array,
    singular_values: mx.array,
    hi_bit: int,
    lo_bit: int,
    hi_fraction: float,
    group_size: int,
) -> mx.array:
    """Batched-leading-axis equivalent of :func:`quantize_latent`.

    All rows of the leading ``G`` axis share the same ``singular_values``
    (true for every real caller: heads in the same PALU head-group share one
    projection and therefore one channel importance ranking), so the
    hi/lo channel split is computed once and both tiers are quantized with
    one :func:`_group_quant_dequant_batched` call each instead of looping
    :func:`quantize_latent` per head.

    Args:
        L: ``[G, S, r]`` latents for every head in this group.
        singular_values: ``[r]``, shared across the group.
        hi_bit / lo_bit / hi_fraction / group_size: see :func:`quantize_latent`.

    Returns:
        ``[G, S, r]`` fp16 reconstructed (quantize-then-dequantize) latents.
    """
    if L.ndim != 3:
        raise QuantizerConfigError(f"palu: L must be [G, S, r], got shape {tuple(L.shape)}.")
    g, s, r = L.shape
    hi_bit, lo_bit, hi_fraction, group_size = _validate_quant_args(
        singular_values, r, hi_bit, lo_bit, hi_fraction, group_size
    )
    n_hi = n_hi_channels(r, hi_fraction)
    sv_np = singular_values.tolist()
    sorted_idx = sorted(range(r), key=lambda i: -sv_np[i])
    hi_idx = sorted(sorted_idx[:n_hi])
    lo_idx = sorted(sorted_idx[n_hi:])

    parts = [None] * r
    if hi_idx:
        recon_hi = _group_quant_dequant_batched(
            mx.take(L, mx.array(hi_idx), axis=2), hi_bit, group_size
        )
        for new_col_idx, orig_col_idx in enumerate(hi_idx):
            parts[orig_col_idx] = recon_hi[:, :, new_col_idx]
    if lo_idx:
        recon_lo = _group_quant_dequant_batched(
            mx.take(L, mx.array(lo_idx), axis=2), lo_bit, group_size
        )
        for new_col_idx, orig_col_idx in enumerate(lo_idx):
            parts[orig_col_idx] = recon_lo[:, :, new_col_idx]
    return mx.stack(parts, axis=2).astype(mx.float16)


__all__ = [
    "n_hi_channels",
    "validate_palu_config",
    "head_group_bounds",
    "group_head_svd",
    "project_to_latent",
    "reconstruct_from_latent",
    "quantize_latent",
    "project_to_latent_batched",
    "reconstruct_from_latent_batched",
    "quantize_latent_batched",
]

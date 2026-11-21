"""Argument validators shared by quantizers that reject bad configs loudly.

Policy (the same one GEAR and PALU apply): a whole number given as a float
(``8.0``, as JSON/YAML configs often carry) is accepted and coerced to ``int``;
fractions, ``bool``, strings, NaN and Inf are rejected rather than silently
truncated, clamped or coerced. Every failure is a
:class:`~veloxquant_mlx.core.exceptions.QuantizerConfigError` (a ``ValueError``)
that names the method and the argument.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from veloxquant_mlx.core.exceptions import QuantizerConfigError

#: Widest group-quantization code the shared kernels document (``Bit width (1-8)``).
MAX_BITS = 8


def as_int(value: object) -> int | None:
    """``value`` as an int if it is integral (``8`` or ``8.0``), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return int(f)
    return None


def validate_int(value: object, name: str, *, who: str, lo: int, hi: int | None = None) -> int:
    """An integral ``value`` in ``[lo, hi]`` (``hi=None``: no upper bound), as an ``int``."""
    n = as_int(value)
    if n is None or n < lo or (hi is not None and n > hi):
        bound = f"an integer >= {lo}" if hi is None else f"an integer in [{lo}, {hi}]"
        raise QuantizerConfigError(f"{who}: {name}={value!r} must be {bound}.")
    return n


def validate_rank(rank: object, name: str, *, who: str) -> int | None:
    """``None`` (chosen by energy) or an integer ``>= 1`` (a latent needs a channel)."""
    return None if rank is None else validate_int(rank, name, who=who, lo=1)


def validate_energy_threshold(t: object, name: str, *, who: str) -> float:
    """A finite fraction in ``(0, 1]``."""
    f = float(t) if isinstance(t, Real) and not isinstance(t, bool) else math.nan  # type: ignore[arg-type]
    if not (math.isfinite(f) and 0.0 < f <= 1.0):
        raise QuantizerConfigError(f"{who}: {name}={t!r} must be a number in (0, 1].")
    return f


def validate_fraction(value: object, name: str, *, who: str, allow_zero: bool = True) -> float:
    """A finite fraction in ``[0, 1]`` (``allow_zero=False``: in ``(0, 1]``), as a ``float``.

    ``bool`` and strings are rejected; ``float(True)`` is ``1.0`` and ``float("0.2")`` is
    ``0.2``, and neither is a number a config meant.
    """
    f = float(value) if isinstance(value, Real) and not isinstance(value, bool) else math.nan  # type: ignore[arg-type]
    if not (math.isfinite(f) and (f >= 0.0 if allow_zero else f > 0.0) and f <= 1.0):
        lo = "[0" if allow_zero else "(0"
        raise QuantizerConfigError(f"{who}: {name}={value!r} must be a number in {lo}, 1].")
    return f


def validate_flag(value: object, name: str, *, who: str) -> bool:
    """A real ``bool`` (or ``numpy.bool_``); ``bool("false")`` is ``True``, so strings are rejected."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise QuantizerConfigError(f"{who}: {name}={value!r} must be True or False.")


def validate_bit_schedule(
    schedule: object, name: str, *, who: str, allow_all_zero: bool = True
) -> tuple[int, ...]:
    """A non-empty sequence of per-group bit widths, each an integer in ``[0, MAX_BITS]``.

    ``0`` truncates a group to zero. ``allow_all_zero=False`` additionally rejects a
    schedule that truncates every group, which stores no information at all.
    """
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, (Sequence, np.ndarray)):
        raise QuantizerConfigError(
            f"{who}: {name}={schedule!r} must be a non-empty sequence of integers in "
            f"[0, {MAX_BITS}]."
        )
    items = list(schedule)
    if not items:
        raise QuantizerConfigError(
            f"{who}: {name}={schedule!r} must be a non-empty sequence of integers in "
            f"[0, {MAX_BITS}]."
        )
    out: list[int] = []
    for i, b in enumerate(items):
        n = as_int(b)
        if n is None or not 0 <= n <= MAX_BITS:
            raise QuantizerConfigError(
                f"{who}: {name}[{i}]={b!r} must be an integer in [0, {MAX_BITS}] "
                f"(entries must be >= 0; 0 truncates the group), got {tuple(items)!r}."
            )
        out.append(n)
    if not allow_all_zero and not any(out):
        raise QuantizerConfigError(
            f"{who}: {name}={tuple(out)!r} truncates every group to 0 bits, so nothing "
            "would be stored; give at least one group a positive bit width."
        )
    return tuple(out)


def check_finite_gram(gram: np.ndarray, what: str, *, who: str) -> None:
    """Raise if the data behind a ``[..., D, D]`` Gram matrix held NaN or Inf.

    A finite fp16/bf16/fp32 value cannot overflow when squared in float64, so any
    non-finite input leaves a non-finite entry on the Gram diagonal. Checking the
    ``O(D^2)`` diagonal avoids a second pass over all the rows.
    """
    if not np.isfinite(np.einsum("...ii->...i", gram)).all():
        raise QuantizerConfigError(
            f"{who}: {what} must be finite (found NaN or Inf); the model produced NaN/Inf "
            "KV states."
        )

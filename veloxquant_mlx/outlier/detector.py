"""Streaming detector for high-magnitude outlier channels in key vectors.

Provides :class:`OutlierDetector`, which observes key vectors one token at
a time during prefill and, after a calibration window, identifies the
channels with highest mean absolute magnitude for mixed-precision
handling (e.g. TurboQuant's outlier-channel path in
:mod:`veloxquant_mlx.cache.turboquant_cache`).

:meth:`observe` runs once per token — the hot path, over the whole prefill —
while :meth:`get_outlier_channels` is queried exactly once, right when
calibration completes (see the call site in
:mod:`veloxquant_mlx.cache.turboquant_cache`). So per-token work is kept to
the O(d) vectorized magnitude accumulation that's unavoidable (every channel
has to be looked at at least once), and the top-k selection itself —
:func:`numpy.argpartition`, O(d) — runs only at that single query, rather
than being incrementally maintained (e.g. via a heap) across every token for
a query that never repeats.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from veloxquant_mlx.core.constants import DEFAULT_N_CALIB_TOKENS, DEFAULT_N_OUTLIER_CHANNELS


class OutlierDetector:
    """Detects high-magnitude outlier channels from streaming key vectors.

    During the prefill phase, key vectors are observed one at a time.
    After n_calib tokens, the detector identifies the n_outliers channels
    with the highest mean absolute magnitude — these are the outlier channels.

    Args:
        n_outliers: Number of outlier channels to identify.
        n_calib: Minimum number of tokens before the detector is calibrated.
    """

    def __init__(
        self,
        n_outliers: int = DEFAULT_N_OUTLIER_CHANNELS,
        n_calib: int = DEFAULT_N_CALIB_TOKENS,
    ) -> None:
        self._n_outliers = n_outliers
        self._n_calib = n_calib
        self._sum_abs: np.ndarray | None = None
        self._count: int = 0

    def observe(self, k: Any) -> None:
        """Record a new key vector for calibration.

        Args:
            k: Key vector, shape (d,) or (1, d). Can be an MLX array or numpy array.
        """
        arr = np.array(k, dtype=np.float32).reshape(-1)
        if self._sum_abs is None:
            self._sum_abs = np.abs(arr)
        else:
            self._sum_abs += np.abs(arr)
        self._count += 1

    def get_outlier_channels(self) -> np.ndarray:
        """Return indices of the top-n_outliers high-magnitude channels.

        Returns:
            Sorted array of channel indices (ascending), shape
            (min(n_outliers, number of channels observed),).
        """
        if self._sum_abs is None or self._n_outliers <= 0:
            return np.array([], dtype=np.int32)

        mean_abs = self._sum_abs / self._count
        k = min(self._n_outliers, mean_abs.shape[0])
        # argpartition finds the top-k unordered in O(d); the caller only
        # needs the set of channel indices (sorted ascending for a stable,
        # readable order), not a full ranking by magnitude.
        top_k = np.argpartition(mean_abs, -k)[-k:]
        return np.sort(top_k).astype(np.int32)

    @property
    def is_calibrated(self) -> bool:
        """True once at least n_calib tokens have been observed."""
        return self._count >= self._n_calib

    @property
    def n_observed(self) -> int:
        """Number of key vectors observed so far."""
        return self._count

    def reset(self) -> None:
        """Reset the detector state for a new sequence."""
        self._sum_abs = None
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"OutlierDetector(n_outliers={self._n_outliers}, "
            f"n_calib={self._n_calib}, "
            f"observed={self._count}, "
            f"calibrated={self.is_calibrated})"
        )

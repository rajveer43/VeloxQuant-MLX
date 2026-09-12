from __future__ import annotations

import numpy as np

from veloxquant_mlx.core.constants import DEFAULT_N_CALIB_TOKENS, DEFAULT_N_OUTLIER_CHANNELS
from veloxquant_mlx.dsa.heap import SortedChannelIndex


class OutlierDetector:
    """Detects high-magnitude outlier channels from streaming key vectors.

    During the prefill phase, key vectors are observed one at a time.
    After n_calib tokens, the detector identifies the n_outliers channels
    with the highest mean absolute magnitude — these are the outlier channels.

    Internally uses SortedChannelIndex for efficient top-k tracking.

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
        self._index = SortedChannelIndex()
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

        # Update index with running means
        mean_abs = self._sum_abs / self._count
        for ch_idx, mag in enumerate(mean_abs):
            self._index.insert(ch_idx, float(mag))

    def get_outlier_channels(self) -> np.ndarray:
        """Return indices of the top-n_outliers high-magnitude channels.

        Returns:
            Sorted array of channel indices (ascending), shape (n_outliers,).
        """
        top_k = self._index.top_k(self._n_outliers)
        return np.array(sorted(top_k), dtype=np.int32)

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
        self._index = SortedChannelIndex()
        self._sum_abs = None
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"OutlierDetector(n_outliers={self._n_outliers}, "
            f"n_calib={self._n_calib}, "
            f"observed={self._count}, "
            f"calibrated={self.is_calibrated})"
        )


# Allow bare 'Any' usage without import in function signatures
from typing import Any  # noqa: E402

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook
from veloxquant_mlx.math.lloyd_max import lloyd_max


class AdaptiveScalarCodebook:
    """ScalarCodebook that refits its centroids from observed key vectors.

    Phase 1 (calibration, first n_calib batches passed through observe()):
        - Buffer raw post-rotation vectors.
        - quantize() / dequantize() proxy to a default N(0, 1/d) ScalarCodebook
          so encode/decode keep working before calibration completes.

    Phase 2 (after n_calib vectors observed):
        - Build an empirical histogram from the buffer.
        - Run Lloyd-Max with that empirical PDF to fit centroids.
        - Replace the proxy codebook with a ScalarCodebook of fitted centroids.

    The wrapper is API-compatible with ScalarCodebook (quantize/dequantize/
    centroids_mx/centroids_numpy/k/b), so existing quantizers can use it as
    a drop-in replacement.

    Args:
        b: Bit-width.
        d: Vector dimension (used for default codebook shape).
        n_calib: Number of vectors to buffer before fitting.
        default_codebook: Initial ScalarCodebook used during calibration.
        n_hist_bins: Histogram bin count for empirical PDF estimation.
    """

    def __init__(
        self,
        b: int,
        d: int,
        n_calib: int = 64,
        default_codebook: ScalarCodebook | None = None,
        n_hist_bins: int = 64,
        seed: int = 0,
    ) -> None:
        self._b = int(b)
        self._d = int(d)
        self._k = 2**self._b
        self._n_calib = int(n_calib)
        self._n_hist_bins = int(n_hist_bins)
        self._rng = np.random.default_rng(seed)

        if default_codebook is None:
            from veloxquant_mlx.codebooks.base import CodebookFactory

            distribution = "gaussian" if d >= 64 else "beta"
            default_codebook = CodebookFactory.create(distribution, b=b, d=d)
        self._codebook: ScalarCodebook = default_codebook  # type: ignore[assignment]

        self._buffer: list[np.ndarray] = []
        self._n_observed = 0
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def k(self) -> int:
        return self._k

    @property
    def b(self) -> int:
        return self._b

    def get_codebook(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (centroids, boundaries) of the current codebook.

        Boundaries are recomputed as midpoints between sorted centroids.
        """
        c = self._codebook.centroids_numpy()
        c_sorted = np.sort(c)
        boundaries = np.concatenate(
            [
                [-np.inf],
                (c_sorted[:-1] + c_sorted[1:]) / 2.0,
                [np.inf],
            ]
        )
        return c_sorted, boundaries

    def observe(self, y: Any) -> None:
        """Accumulate post-rotation vectors during calibration.

        The full batch is buffered (not truncated to the first `remaining`
        rows) so a single large call -- e.g. an entire prefill batch -- does
        not silently bias calibration toward whichever rows happened to be
        first. `_fit()` randomly subsamples the buffer down to `n_calib`.

        Args:
            y: Array of shape (batch, d), fp16 mx or numpy.
        """
        if self._is_calibrated:
            return
        y_np = np.array(y, dtype=np.float32).reshape(-1, self._d)
        self._buffer.append(y_np)
        self._n_observed += y_np.shape[0]
        if self._n_observed >= self._n_calib:
            self._fit()

    def _fit(self) -> None:
        if self._is_calibrated:
            return
        buffered = np.concatenate(self._buffer, axis=0)
        if buffered.shape[0] > self._n_calib:
            idx = self._rng.choice(buffered.shape[0], size=self._n_calib, replace=False)
            buffered = buffered[idx]
        flat = buffered.reshape(-1).astype(np.float64)
        # Empirical PDF via histogram
        lo = float(np.quantile(flat, 0.001))
        hi = float(np.quantile(flat, 0.999))
        if hi - lo < 1e-6:
            lo, hi = -1.0, 1.0
        bin_edges = np.linspace(lo, hi, self._n_hist_bins + 1)
        hist, _ = np.histogram(flat, bins=bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def pdf_fn(x: np.ndarray) -> np.ndarray:
            return np.interp(x, bin_centers, hist, left=0.0, right=0.0)

        centroids, _ = lloyd_max(pdf_fn, support=(lo, hi), n_levels=self._k)
        self._codebook = ScalarCodebook(centroids.astype(np.float32))
        self._is_calibrated = True
        self._buffer = []  # release memory

    def quantize(self, y: Any) -> Any:
        return self._codebook.quantize(y)

    def dequantize(self, idx: Any) -> Any:
        return self._codebook.dequantize(idx)

    def centroids_numpy(self) -> np.ndarray:
        return self._codebook.centroids_numpy()

    def centroids_mx(self) -> Any:
        return self._codebook.centroids_mx()

    def __repr__(self) -> str:
        return (
            f"AdaptiveScalarCodebook(b={self._b}, d={self._d}, "
            f"calibrated={self._is_calibrated}, observed={self._n_observed}/{self._n_calib})"
        )

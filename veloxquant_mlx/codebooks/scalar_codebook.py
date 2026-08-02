from __future__ import annotations

from typing import Any

import numpy as np

from veloxquant_mlx.core.abstractions import Codebook
from veloxquant_mlx.core.exceptions import CodebookDimensionMismatch
from veloxquant_mlx.dsa.avl_tree import VoronoiTree


class ScalarCodebook(Codebook):
    """MLX-based scalar codebook for nearest-centroid quantisation.

    Quantisation uses broadcast argmin over the codebook (O(d·k) per batch).
    Dequantisation is a simple gather (O(d) per batch).

    For large k (> VoronoiTree.LINEAR_THRESHOLD), a VoronoiTree provides
    O(log k) per-coordinate search in pure Python (used only when MLX
    is unavailable or for verification).

    Args:
        centroids: 1-D array of sorted centroid values, shape (k,).
    """

    def __init__(self, centroids: np.ndarray) -> None:
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 1:
            raise CodebookDimensionMismatch(
                f"ScalarCodebook expects 1-D centroids, got shape {centroids.shape}"
            )
        self._k = len(centroids)
        self._b = int(np.log2(self._k))
        if 2**self._b != self._k:
            raise CodebookDimensionMismatch(
                f"ScalarCodebook: number of centroids must be a power of 2, got {self._k}"
            )

        import mlx.core as mx

        # Centroids must be sorted for searchsorted-based quantize.
        sort_idx = np.argsort(centroids)
        centroids = centroids[sort_idx]
        self._centroids_mx: Any = mx.array(centroids.astype(np.float16))
        self._centroids_np: np.ndarray = centroids
        # Precompute Voronoi boundaries (midpoints between sorted centroids).
        # Used by quantize() via mx.searchsorted — single kernel, no (b,d,k) broadcast.
        boundaries_np = (centroids[:-1] + centroids[1:]) / 2.0
        self._boundaries_mx: Any = mx.array(boundaries_np.astype(np.float16))

        self._voronoi = VoronoiTree()
        self._voronoi.build(centroids)

    @property
    def k(self) -> int:
        """Number of centroids (2^b)."""
        return self._k

    @property
    def b(self) -> int:
        """Bit-width of this codebook."""
        return self._b

    def quantize(self, y: Any) -> Any:
        """Map coordinates to nearest-centroid indices using broadcast argmin.

        dist_{i,j,k} = |y_{i,j} - c_k| → argmin_k

        Args:
            y: Input array of shape (batch, d), fp16.

        Returns:
            Index array of shape (batch, d), dtype uint8.
        """
        import mlx.core as mx

        # Boundary-sum quantize: count how many boundaries y exceeds.
        # That count is exactly the centroid index. Drops the abs() and argmin()
        # kernels of the prior path; still uses (batch, d, k-1) broadcast but
        # over k-1 boundaries instead of k centroids, and with cheaper (>) op.
        # Output is identical to the broadcast argmin in exact arithmetic; fp16
        # tie-breaking on a boundary may flip to the other side.
        cmp = y[:, :, None] > self._boundaries_mx[None, None, :]
        return mx.sum(cmp.astype(mx.uint8), axis=-1).astype(mx.uint8)

    def dequantize(self, idx: Any) -> Any:
        """Retrieve centroid values via gather.

        Args:
            idx: Index array of shape (batch, d), dtype uint8.

        Returns:
            Centroid values of shape (batch, d), fp16.
        """
        import mlx.core as mx

        return self._centroids_mx[idx]

    def nearest_numpy(self, value: float) -> int:
        """O(log k) nearest-centroid lookup for scalar queries.

        Falls back to linear for k <= 16. Used for verification.

        Args:
            value: Scalar float query.

        Returns:
            Index of the nearest centroid.
        """
        return self._voronoi.nearest(value)

    def centroids_numpy(self) -> np.ndarray:
        """Return centroid array as float32 numpy array."""
        return self._centroids_np.copy()

    def centroids_mx(self) -> Any:
        """Return centroid array as MLX fp16 array."""
        return self._centroids_mx

    def __repr__(self) -> str:
        return f"ScalarCodebook(k={self._k}, b={self._b})"

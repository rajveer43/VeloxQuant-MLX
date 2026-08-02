from __future__ import annotations

from typing import Any, Optional

import numpy as np

from veloxquant_mlx.core.abstractions import Quantizer
from veloxquant_mlx.core.context import EncodedVector


class CompositeQuantizer(Quantizer):
    """Split-channel quantizer that routes outlier and inlier channels separately.

    Outlier channels are quantized with a high-bit-width quantizer.
    Inlier channels are quantized with a low-bit-width quantizer.

    The outlier_encoded and inlier_encoded fields of the returned EncodedVector
    nest the children's outputs.

    Args:
        outlier_quantizer: Quantizer for high-magnitude (outlier) channels.
        inlier_quantizer: Quantizer for low-magnitude (inlier) channels.
        outlier_idx: Integer indices of outlier channels in the original vector.
        total_dim: Total dimension d of the original vector.
    """

    def __init__(
        self,
        outlier_quantizer: Quantizer,
        inlier_quantizer: Quantizer,
        outlier_idx: np.ndarray,
        total_dim: int,
    ) -> None:
        self._outlier_q = outlier_quantizer
        self._inlier_q = inlier_quantizer
        self._outlier_idx = np.asarray(outlier_idx, dtype=np.int32)
        self._inlier_idx = np.setdiff1d(np.arange(total_dim), self._outlier_idx)
        self._d = total_dim

    def encode(self, x: Any) -> EncodedVector:
        """Encode by routing channels to respective child quantizers.

        Args:
            x: Array of shape (batch, d), fp16.

        Returns:
            CompositeEncodedVector with nested outlier and inlier encodings.
        """
        if x.ndim == 1:
            x = x[None]

        x_out = x[:, self._outlier_idx]
        x_in = x[:, self._inlier_idx]

        ev_out = self._outlier_q.encode(x_out)
        ev_in = self._inlier_q.encode(x_in)

        return EncodedVector(
            quantizer_type="composite",
            batch_size=x.shape[0],
            dim=self._d,
            outlier_idx=self._outlier_idx,
            outlier_encoded=ev_out,
            inlier_encoded=ev_in,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Decode and reassemble channels.

        Args:
            ev: CompositeEncodedVector.

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """
        import mlx.core as mx

        x_out = self._outlier_q.decode(ev.outlier_encoded)  # (batch, n_out)
        x_in = self._inlier_q.decode(ev.inlier_encoded)  # (batch, n_in)

        batch = x_out.shape[0]
        out_np = np.zeros((batch, self._d), dtype=np.float32)
        out_np[:, self._outlier_idx] = np.array(x_out, dtype=np.float32)
        out_np[:, self._inlier_idx] = np.array(x_in, dtype=np.float32)
        return mx.array(out_np).astype(x_out.dtype)

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ by summing contributions from both children.

        Args:
            q: Query vector, shape (d,) or (1, d).
            ev: CompositeEncodedVector.

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        q_flat = q.reshape(-1)
        q_out = q_flat[self._outlier_idx]
        q_in = q_flat[self._inlier_idx]

        ip_out = self._outlier_q.estimate_inner_product(q_out, ev.outlier_encoded)
        ip_in = self._inlier_q.estimate_inner_product(q_in, ev.inlier_encoded)
        return ip_out + ip_in

    def __repr__(self) -> str:
        return (
            f"CompositeQuantizer("
            f"n_outliers={len(self._outlier_idx)}, "
            f"n_inliers={len(self._inlier_idx)}, "
            f"d={self._d})"
        )

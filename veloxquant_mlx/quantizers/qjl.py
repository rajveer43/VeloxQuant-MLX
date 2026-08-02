from __future__ import annotations

from typing import Any, Optional

import numpy as np

from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.constants import SQRT_PI_OVER_2
from veloxquant_mlx.core.context import EncodedVector, QuantizationContext
from veloxquant_mlx.core.registry import QuantizerRegistry
from veloxquant_mlx.math.rotation import make_jl_matrix
from veloxquant_mlx.preconditioners.jl_sketch import QJLEncoder


@QuantizerRegistry.register("qjl")
class QJLQuantizer(Quantizer):
    """1-bit Quantized Johnson-Lindenstrauss key-vector quantizer.

    Pipeline: x → sign(S·x) + ‖x‖

    Encode stores:
        signs: (batch, m) int8 — sign(S·x)
        norm:  (batch,) fp16  — ‖x‖₂

    Inner product estimation (unbiased):
        ProdQJL(q, k) = √(π/2)/m · ‖k‖ · ⟨S·q, sign(S·k)⟩

    Args:
        d: Input dimension.
        m: JL projection dimension (default = d).
        seed: Random seed for JL matrix.
        b: Unused (QJL is always 1-bit for signs; stored for API consistency).
        store: Optional ArtifactStore.
    """

    def __init__(
        self,
        d: int,
        m: int = 128,
        seed: int = 42,
        b: int = 1,
        store: Optional[ArtifactStore] = None,
        **kwargs: Any,
    ) -> None:
        self._d = d
        self._m = m
        self._seed = seed

        import mlx.core as mx

        if store is not None and store.exists("jl", d=d, m=m, seed=seed):
            S = store.load_jl_matrix(d, m, seed)
        else:
            S_np = make_jl_matrix(d, m, seed=seed)
            S = mx.array(S_np.astype(np.float16))
            if store is not None:
                store.save_jl_matrix(S_np, d=d, m=m, seed=seed)

        self._encoder = QJLEncoder(S)

    def encode(self, x: Any) -> EncodedVector:
        """Encode a batch of key vectors via QJL.

        Args:
            x: Array of shape (batch, d), fp16.

        Returns:
            EncodedVector with signs and norm populated.
        """
        import mlx.core as mx

        if x.ndim == 1:
            x = x[None]
        signs, norm = self._encoder.encode_key(x)
        return EncodedVector(
            quantizer_type="qjl",
            batch_size=x.shape[0],
            dim=self._d,
            signs=signs,
            norm=norm,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Approximate reconstruction: x̃ = ‖k‖ · (√(π/2)/m) · S^T · sign(S·k).

        Note: this is biased (not a true inverse); used only for MSE testing.

        Args:
            ev: EncodedVector with signs and norm.

        Returns:
            Approximate array of shape (batch, d), fp16.
        """
        import mlx.core as mx

        scale = SQRT_PI_OVER_2 / self._m
        x_hat = ev.norm[:, None] * scale * (ev.signs.astype(mx.float16) @ self._encoder._S)
        return x_hat

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ for each encoded key.

        Args:
            q: Query vector, shape (d,) or (1, d).
            ev: Encoded keys (signs and norm).

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        return self._encoder.estimate_ip(q, ev.signs, ev.norm)

    def __repr__(self) -> str:
        return f"QJLQuantizer(d={self._d}, m={self._m}, seed={self._seed})"

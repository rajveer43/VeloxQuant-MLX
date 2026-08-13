from __future__ import annotations

import math
from typing import Any, Literal, Optional

from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.core.registry import QuantizerRegistry


class QuantizerFactory:
    """Factory for creating Quantizer instances via the QuantizerRegistry.

    All quantizer construction should go through this factory.

    Example::

        q = QuantizerFactory.create("qjl", d=128, m=128, seed=42)
        q = QuantizerFactory.create("turboquant_mse", d=128, b=2, seed=42)
        q = QuantizerFactory.create("turboquant_prod", d=128, b=3, m=128, seed=42)
        q = QuantizerFactory.create("polar", d=128, b=2, seed=42)
    """

    @staticmethod
    def create(
        method: Literal["qjl", "turboquant_mse", "turboquant_prod", "polar"],
        d: int,
        b: int = 2,
        m: Optional[int] = None,
        seed: int = 42,
        store: Optional[ArtifactStore] = None,
        **kwargs: Any,
    ) -> Quantizer:
        """Instantiate a Quantizer by method name.

        Args:
            method: Algorithm name registered in QuantizerRegistry.
            d: Vector dimension.
            b: Bit-width (default 2).
            m: JL projection dimension (default = d for QJL methods).
            seed: Random seed.
            store: ArtifactStore for loading precomputed artifacts. If None,
                artifacts are generated on-the-fly.
            **kwargs: Additional keyword arguments forwarded to the quantizer.

        Returns:
            Configured Quantizer instance.

        Raises:
            QuantizerConfigError: If parameters are invalid.
            KeyError: If method is not registered.
        """
        if not math.log2(d).is_integer():
            raise QuantizerConfigError(f"QuantizerFactory: d={d} must be a power of 2")
        if b < 1:
            raise QuantizerConfigError(f"QuantizerFactory: b={b} must be >= 1")

        if m is None:
            m = d

        cls = QuantizerRegistry.get(method)
        return cls(d=d, b=b, m=m, seed=seed, store=store, **kwargs)

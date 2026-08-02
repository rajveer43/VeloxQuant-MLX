from __future__ import annotations

from typing import Any, Literal

from veloxquant_mlx.core.abstractions import Preconditioner
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.core.registry import PreconditionerRegistry


class PreconditionerFactory:
    """Factory for creating Preconditioner instances.

    Example::

        pre = PreconditionerFactory.create("rotation", d=128, m=128, Pi=Pi_array)
    """

    @staticmethod
    def create(
        kind: Literal["rotation", "jl", "hadamard"],
        d: int,
        m: int,
        **kwargs: Any,
    ) -> Preconditioner:
        """Instantiate a Preconditioner.

        Args:
            kind: Type of preconditioner.
                - ``"rotation"``: Orthogonal rotation Π (requires ``Pi`` kwarg).
                - ``"jl"``: JL sketch S (requires ``S`` kwarg).
                - ``"hadamard"``: Structured Hadamard (not yet implemented).
            d: Input dimension.
            m: Output dimension (for JL) or same as d (for rotation).
            **kwargs: Extra constructor arguments (``Pi`` or ``S``).

        Returns:
            Configured Preconditioner instance.

        Raises:
            QuantizerConfigError: If kind is unknown or required kwargs are missing.
        """
        if kind == "rotation":
            if "Pi" not in kwargs:
                raise QuantizerConfigError("PreconditionerFactory: 'rotation' requires 'Pi' kwarg.")
            cls = PreconditionerRegistry.get("rotation")
            return cls(kwargs["Pi"])
        elif kind == "jl":
            if "S" not in kwargs:
                raise QuantizerConfigError("PreconditionerFactory: 'jl' requires 'S' kwarg.")
            cls = PreconditionerRegistry.get("jl")
            return cls(kwargs["S"])
        elif kind == "hadamard":
            if "D" not in kwargs:
                raise QuantizerConfigError(
                    "PreconditionerFactory: 'hadamard' requires 'D' kwarg (±1 diagonal vector)."
                )
            cls = PreconditionerRegistry.get("hadamard")
            return cls(kwargs["D"])
        else:
            raise QuantizerConfigError(
                f"PreconditionerFactory: unknown kind '{kind}'. Choices: rotation, jl, hadamard."
            )

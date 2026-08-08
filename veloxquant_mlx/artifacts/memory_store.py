from __future__ import annotations

from typing import Any

import numpy as np

from veloxquant_mlx.core.abstractions import ArtifactStore
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError


class InMemoryArtifactStore(ArtifactStore):
    """In-memory artifact store for testing — performs no disk I/O.

    All artifacts are stored in plain Python dicts keyed by descriptor tuples.
    Arrays are stored as float16 numpy arrays and wrapped in MLX on load.

    Args:
        None.
    """

    def __init__(self) -> None:
        self._rotations: dict[tuple, np.ndarray] = {}
        self._codebooks: dict[tuple, np.ndarray] = {}
        self._jls: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def load_rotation_matrix(self, d: int, seed: int) -> Any:
        key = (d, seed)
        if key not in self._rotations:
            raise ArtifactNotFoundError(
                f"InMemoryArtifactStore: rotation d={d} seed={seed} not found."
            )
        import mlx.core as mx

        return mx.array(self._rotations[key])

    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None:
        self._rotations[(d, seed)] = np.array(Pi, dtype=np.float16)

    # ------------------------------------------------------------------
    # Codebook
    # ------------------------------------------------------------------

    def load_codebook(self, distribution: str, b: int, d: int) -> Any:
        key = (distribution, b, d)
        if key not in self._codebooks:
            raise ArtifactNotFoundError(
                f"InMemoryArtifactStore: codebook dist={distribution} b={b} d={d} not found."
            )
        import mlx.core as mx

        return mx.array(self._codebooks[key])

    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None:
        self._codebooks[(distribution, b, d)] = np.array(cb, dtype=np.float16)

    # ------------------------------------------------------------------
    # JL matrix
    # ------------------------------------------------------------------

    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any:
        key = (d, m, seed)
        if key not in self._jls:
            raise ArtifactNotFoundError(
                f"InMemoryArtifactStore: JL d={d} m={m} seed={seed} not found."
            )
        import mlx.core as mx

        return mx.array(self._jls[key])

    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None:
        self._jls[(d, m, seed)] = np.array(S, dtype=np.float16)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, artifact_type: str, **kwargs: Any) -> bool:
        if artifact_type == "rotation":
            return (kwargs["d"], kwargs["seed"]) in self._rotations
        if artifact_type == "codebook":
            return (kwargs["distribution"], kwargs["b"], kwargs["d"]) in self._codebooks
        if artifact_type == "jl":
            return (kwargs["d"], kwargs["m"], kwargs["seed"]) in self._jls
        return False

    def __repr__(self) -> str:
        return (
            f"InMemoryArtifactStore("
            f"rotations={len(self._rotations)}, "
            f"codebooks={len(self._codebooks)}, "
            f"jls={len(self._jls)})"
        )

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from veloxquant_mlx.core.abstractions import ArtifactStore
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` to ``path`` via a temp file + atomic rename.

    Prevents concurrent readers/writers targeting the same path (e.g. two
    workers lazily constructing the same quantizer config) from observing a
    partially-written ``.npy`` file: ``np.save`` writes directly to the
    destination and is not atomic, but ``os.replace`` is atomic on POSIX and
    Windows. The temp name is PID- and object-id-qualified so concurrent
    writers never collide with each other's temp files either.
    """
    tmp_path = path.with_name(f".{path.name}.tmp{os.getpid()}-{id(arr)}.npy")
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


class NpyArtifactStore(ArtifactStore):
    """Artifact store that reads and writes ``.npy`` files from a local directory.

    File naming conventions:
        rotation_d{d}_seed{seed}.npy
        codebook_{distribution}_b{b}_d{d}.npy
        jl_d{d}_m{m}_seed{seed}.npy

    Args:
        root_dir: Path to the directory where artifacts are stored.
            Created automatically on first save if absent.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def _rotation_path(self, d: int, seed: int) -> Path:
        return self._root / f"rotation_d{d}_seed{seed}.npy"

    def load_rotation_matrix(self, d: int, seed: int) -> Any:
        path = self._rotation_path(d, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Rotation matrix not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d}` first."
            )
        import mlx.core as mx

        return mx.array(np.load(path).astype(np.float16))

    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None:
        path = self._rotation_path(d, seed)
        arr = np.array(Pi, dtype=np.float16)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # Codebook
    # ------------------------------------------------------------------

    def _codebook_path(self, distribution: str, b: int, d: int) -> Path:
        return self._root / f"codebook_{distribution}_b{b}_d{d}.npy"

    def load_codebook(self, distribution: str, b: int, d: int) -> Any:
        path = self._codebook_path(distribution, b, d)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Codebook not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d} --bits {b}` first."
            )
        import mlx.core as mx

        return mx.array(np.load(path).astype(np.float16))

    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None:
        path = self._codebook_path(distribution, b, d)
        arr = np.array(cb, dtype=np.float16)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # JL matrix
    # ------------------------------------------------------------------

    def _jl_path(self, d: int, m: int, seed: int) -> Path:
        return self._root / f"jl_d{d}_m{m}_seed{seed}.npy"

    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any:
        path = self._jl_path(d, m, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"JL matrix not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d} --jl_dim {m}` first."
            )
        import mlx.core as mx

        return mx.array(np.load(path).astype(np.float16))

    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None:
        path = self._jl_path(d, m, seed)
        arr = np.array(S, dtype=np.float16)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, artifact_type: str, **kwargs: Any) -> bool:
        if artifact_type == "rotation":
            return self._rotation_path(kwargs["d"], kwargs["seed"]).exists()
        if artifact_type == "codebook":
            return self._codebook_path(kwargs["distribution"], kwargs["b"], kwargs["d"]).exists()
        if artifact_type == "jl":
            return self._jl_path(kwargs["d"], kwargs["m"], kwargs["seed"]).exists()
        return False

    def __repr__(self) -> str:
        return f"NpyArtifactStore(root={self._root!r})"

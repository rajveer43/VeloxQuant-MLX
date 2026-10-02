"""Filesystem-backed :class:`ArtifactStore` that persists artifacts as ``.npy`` files.

The production ``ArtifactStore`` implementation: rotation matrices,
codebooks, and JL sketch matrices are precomputed once (via ``python -m
veloxquant_mlx precompute``) and cached under a root directory using a fixed
naming scheme (``rotation_d{d}_seed{seed}.npy``, etc.), so subsequent cache
construction can load them instead of recomputing. Writes go through
:func:`_atomic_save` (temp file + atomic rename) so concurrent
readers/writers targeting the same artifact never observe a partial file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from veloxquant_mlx.core.abstractions import ArtifactStore
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` to ``path`` via a temp file + atomic rename.

    Prevents concurrent readers/writers targeting the same path (e.g. two
    workers lazily constructing the same quantizer config) from observing a
    partially-written ``.npy`` file: ``np.save`` writes directly to the
    destination and is not atomic, but ``Path.replace`` is atomic on POSIX
    and Windows. The temp file comes from ``tempfile.mkstemp`` in the same
    directory, so its name is unique across processes and threads (a
    per-process counter is not: every process starts at 1, #634).
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        tmp_path.replace(path)
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
        """Load the rotation matrix at ``rotation_d{d}_seed{seed}.npy``, raising if absent."""
        path = self._rotation_path(d, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Rotation matrix not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d}` first."
            )
        arr = np.load(path)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        return mx.array(arr)

    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None:
        """Atomically write ``Pi`` (cast to fp16) to ``rotation_d{d}_seed{seed}.npy``, silently overwriting any prior file."""
        path = self._rotation_path(d, seed)
        arr = np.array(Pi, dtype=np.float16)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # Codebook
    # ------------------------------------------------------------------

    def _codebook_path(self, distribution: str, b: int, d: int) -> Path:
        return self._root / f"codebook_{distribution}_b{b}_d{d}.npy"

    def load_codebook(self, distribution: str, b: int, d: int) -> Any:
        """Load the codebook at ``codebook_{distribution}_b{b}_d{d}.npy``, raising if absent."""
        path = self._codebook_path(distribution, b, d)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Codebook not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d} --bits {b}` first."
            )
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return mx.array(arr)

    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None:
        """Atomically write ``cb`` (as fp32) to ``codebook_{distribution}_b{b}_d{d}.npy``, silently overwriting any prior file.

        Kept in fp32: fp16 rounding of tightly spaced centroids changes the
        nearest-centroid choice, so a warm store would encode differently
        from the cold run that computed the codebook (#659).
        """
        path = self._codebook_path(distribution, b, d)
        arr = np.array(cb, dtype=np.float32)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # JL matrix
    # ------------------------------------------------------------------

    def _jl_path(self, d: int, m: int, seed: int) -> Path:
        return self._root / f"jl_d{d}_m{m}_seed{seed}.npy"

    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any:
        """Load the JL matrix at ``jl_d{d}_m{m}_seed{seed}.npy``, raising if absent."""
        path = self._jl_path(d, m, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"JL matrix not found at {path}. "
                f"Run `python -m veloxquant_mlx precompute --head_dim {d} --jl_dim {m}` first."
            )
        arr = np.load(path)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        return mx.array(arr)

    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None:
        """Atomically write ``S`` (cast to fp16) to ``jl_d{d}_m{m}_seed{seed}.npy``, silently overwriting any prior file."""
        path = self._jl_path(d, m, seed)
        arr = np.array(S, dtype=np.float16)
        _atomic_save(path, arr)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, artifact_type: str, **kwargs: Any) -> bool:
        """Check whether a rotation/codebook/JL artifact file is already on disk.

        ``artifact_type`` must be one of ``"rotation"``, ``"codebook"``, or
        ``"jl"``; ``kwargs`` must supply that type's identifying parameters
        (e.g. ``d=``/``seed=`` for a rotation). Returns ``False`` for an
        unrecognized ``artifact_type`` rather than raising.
        """
        if artifact_type == "rotation":
            return self._rotation_path(kwargs["d"], kwargs["seed"]).exists()
        if artifact_type == "codebook":
            return self._codebook_path(kwargs["distribution"], kwargs["b"], kwargs["d"]).exists()
        if artifact_type == "jl":
            return self._jl_path(kwargs["d"], kwargs["m"], kwargs["seed"]).exists()
        return False

    def __repr__(self) -> str:
        return f"NpyArtifactStore(root={self._root!r})"

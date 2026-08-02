"""CLI-runnable precomputation script for codebooks, rotation matrices, and JL matrices.

Usage::

    python -m veloxquant_mlx.codebooks.precompute \\
        --head_dim 128 \\
        --bits 1 2 3 4 \\
        --jl_dim 128 \\
        --seed 42 \\
        --output_dir ./artifacts/
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore
from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.math.rotation import make_jl_matrix, make_rotation_matrix


def precompute(
    head_dim: int,
    bits: list[int],
    jl_dim: int,
    seed: int,
    output_dir: str,
    distributions: list[str] | None = None,
) -> None:
    """Precompute and save rotation matrix, JL matrix, and codebooks.

    Args:
        head_dim: Attention head dimension (d).
        bits: List of bit-widths to precompute codebooks for.
        jl_dim: JL projection dimension (m).
        seed: Random seed.
        output_dir: Directory for .npy output files.
        distributions: List of distributions to compute. Defaults to
            ["gaussian", "beta"].
    """
    if distributions is None:
        distributions = ["gaussian", "beta"]

    store = NpyArtifactStore(output_dir)
    d = head_dim

    # Rotation matrix
    print(f"[precompute] Rotation matrix d={d} seed={seed} ...", end=" ", flush=True)
    Pi = make_rotation_matrix(d, seed=seed)
    store.save_rotation_matrix(Pi, d=d, seed=seed)
    print("done")

    # JL matrix
    m = min(jl_dim, d)
    print(f"[precompute] JL matrix d={d} m={m} seed={seed} ...", end=" ", flush=True)
    S = make_jl_matrix(d, m=m, seed=seed)
    store.save_jl_matrix(S, d=d, m=m, seed=seed)
    print("done")

    # Codebooks
    for b in bits:
        for dist in distributions:
            print(
                f"[precompute] Codebook dist={dist} b={b} d={d} ...",
                end=" ",
                flush=True,
            )
            try:
                cb = CodebookFactory.create(dist, b=b, d=d)
                centroids = cb.centroids_numpy()  # type: ignore[attr-defined]
                store.save_codebook(centroids, distribution=dist, b=b, d=d)
                print("done")
            except Exception as exc:
                print(f"FAILED: {exc}")

    # Polar angle codebooks per level
    import math

    n_levels = 4
    for level in range(1, n_levels + 1):
        for b in bits:
            print(
                f"[precompute] Polar codebook level={level} b={b} d={d} ...",
                end=" ",
                flush=True,
            )
            try:
                cb = CodebookFactory.create("polar_level", b=b, d=d, polar_level=level)
                centroids = cb.centroids_numpy()  # type: ignore[attr-defined]
                store.save_codebook(centroids, distribution=f"polar_level{level}", b=b, d=d)
                print("done")
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"[precompute] All artifacts saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute codebooks, rotation matrices, and JL matrices."
    )
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--jl_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./artifacts/")
    args = parser.parse_args()

    precompute(
        head_dim=args.head_dim,
        bits=args.bits,
        jl_dim=args.jl_dim,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

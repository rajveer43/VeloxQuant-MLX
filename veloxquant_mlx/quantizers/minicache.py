"""MiniCache quantizer — cross-layer KV merge via SLERP + token retention.

Inspired by "MiniCache: KV Cache Compression in Depth Dimension for Large
Language Models" (Liu et al., **NeurIPS 2024**, arXiv:2405.14366). Documented as
"MiniCache-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

How it differs from XQuant (also cross-layer, already in the repo):
  * XQuant **reuses the quantized codes** of an anchor layer; each reuse layer
    still has its own (param-corrected) reconstruction.
  * MiniCache **merges the tensors themselves**: a pair of adjacent layers store
    *one* shared KV obtained by spherical-linear-interpolating their direction
    while keeping each token's per-layer magnitude. The merge is the storage
    win — two layers cost roughly one.

Core mechanism (per token vector, per head):
  1. Decompose each layer's KV vector into magnitude (L2 norm) and direction
     (unit vector). MiniCache's observation: adjacent middle-to-deep layers have
     nearly identical *directions*.
  2. **SLERP** the two directions into one shared unit vector; store it once.
  3. Keep each layer's **scalar magnitude** (cheap — one float per token per
     layer) so reconstruction is ``magnitude_layer * shared_direction``.
  4. **Token retention:** token pairs whose directions are too dissimilar
     (cosine below a threshold) are *not* merged — both layers' full vectors are
     retained. This caps the worst-case error MiniCache's ablation shows matters.

This module holds the pure numerics. The cross-layer coordination (which layer
pairs with which, who writes the shared direction) is handled by
:class:`~veloxquant_mlx.cache.minicache_coordinator.MiniCacheCoordinator`.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


def pair_layers_depth(
    n_layers: int, start_frac: float = 0.5, group_size: int = 2
) -> list[tuple[str, int]]:
    """Assign merge roles, only pairing layers in the middle-to-deep range.

    MiniCache merges adjacent layers in the *middle-to-deep* portion; early
    layers (where adjacent-layer similarity is low) are left unmerged.

    Args:
        n_layers: Number of attention-bearing layers.
        start_frac: Fraction of depth below which layers are never merged
            (kept as standalone "anchor" full-precision-direction layers).
        group_size: Layers per merge group (2 = pairs, MiniCache's default).

    Returns:
        Length-``n_layers`` list of ``(role, group_id)`` where role is
        ``"primary"`` (writes the shared direction) or ``"merge"`` (reads it).
        Layers below the start depth are all ``"primary"`` with a unique group.
    """
    if group_size < 2:
        raise ValueError(f"pair_layers_depth: group_size must be >= 2, got {group_size}.")
    start = int(n_layers * start_frac)
    roles: list[tuple[str, int]] = []
    gid = 0
    i = 0
    while i < n_layers:
        if i < start:
            roles.append(("primary", gid))
            gid += 1
            i += 1
        else:
            # form a merge group of up to group_size layers
            roles.append(("primary", gid))
            i += 1
            for _ in range(group_size - 1):
                if i < n_layers:
                    roles.append(("merge", gid))
                    i += 1
            gid += 1
    return roles


def to_mag_dir(x: mx.array) -> tuple[mx.array, mx.array]:
    """Split [..., D] into (magnitude [..., 1], unit direction [..., D])."""
    x32 = x.astype(mx.float32)
    mag = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
    direction = x32 / mx.maximum(mag, 1e-8)
    return mag, direction


def slerp(d0: mx.array, d1: mx.array, t: float = 0.5) -> mx.array:
    """Spherical linear interpolation between two unit-direction tensors.

    Args:
        d0, d1: [..., D] unit vectors (last axis is the vector dim).
        t: interpolation factor in [0, 1] (0.5 = midpoint).

    Returns:
        [..., D] interpolated unit vector. Falls back to normalized lerp when
        the two directions are nearly collinear (sin(omega) ~ 0).
    """
    dot = mx.sum(d0 * d1, axis=-1, keepdims=True)
    dot = mx.clip(dot, -1.0, 1.0)
    omega = mx.arccos(dot)  # [..., 1]
    sin_omega = mx.sin(omega)
    near = sin_omega < 1e-6
    # SLERP coefficients
    a = mx.sin((1.0 - t) * omega) / mx.maximum(sin_omega, 1e-8)
    b = mx.sin(t * omega) / mx.maximum(sin_omega, 1e-8)
    out = a * d0 + b * d1
    # collinear fallback → normalized linear interpolation
    lerp = (1.0 - t) * d0 + t * d1
    out = mx.where(near, lerp, out)
    norm = mx.sqrt(mx.sum(out * out, axis=-1, keepdims=True))
    return out / mx.maximum(norm, 1e-8)


class MergeResult(NamedTuple):
    """Output of merging two layers' KV for one head.

    Attributes:
        shared_dir: [S, D] shared unit direction (stored once for the pair).
        mag_primary: [S, 1] primary layer's per-token magnitude.
        mag_merge:   [S, 1] merge layer's per-token magnitude.
        retained:    [S] bool — True where the pair was NOT merged (kept full).
        full_primary / full_merge: [S, D] full vectors for retained tokens
            (only meaningful where ``retained`` is True; elsewhere ignored).
    """

    shared_dir: mx.array
    mag_primary: mx.array
    mag_merge: mx.array
    retained: mx.array
    full_primary: mx.array
    full_merge: mx.array


def merge_pair(
    x_primary: mx.array,
    x_merge: mx.array,
    retention_threshold: float = 0.9,
    t: float = 0.5,
) -> MergeResult:
    """Merge two layers' [S, D] KV (one head) into a shared direction + magnitudes.

    Args:
        x_primary: [S, D] primary layer's keys or values.
        x_merge:   [S, D] merge layer's keys or values (same head).
        retention_threshold: cosine below which a token pair is NOT merged.
        t: SLERP factor.

    Returns:
        MergeResult.
    """
    mag_p, dir_p = to_mag_dir(x_primary)
    mag_m, dir_m = to_mag_dir(x_merge)
    cos = mx.sum(dir_p * dir_m, axis=-1, keepdims=True)  # [S, 1]
    retained = (cos < retention_threshold).reshape(-1)  # [S]
    shared = slerp(dir_p, dir_m, t=t)  # [S, D]
    return MergeResult(
        shared_dir=shared,
        mag_primary=mag_p,
        mag_merge=mag_m,
        retained=retained,
        full_primary=x_primary.astype(mx.float16),
        full_merge=x_merge.astype(mx.float16),
    )


def reconstruct_layer(res: MergeResult, which: str) -> mx.array:
    """Reconstruct one layer's [S, D] fp16 KV from a MergeResult.

    Merged tokens: ``magnitude * shared_direction``. Retained tokens: the full
    stored vector for that layer.

    Args:
        res: a MergeResult.
        which: ``"primary"`` or ``"merge"``.

    Returns:
        [S, D] fp16 reconstruction.
    """
    if which == "primary":
        mag, full = res.mag_primary, res.full_primary
    elif which == "merge":
        mag, full = res.mag_merge, res.full_merge
    else:
        raise ValueError(f"reconstruct_layer: which must be primary|merge, got {which!r}")
    merged = (mag * res.shared_dir).astype(mx.float16)  # [S, D]
    mask = res.retained.reshape(-1, 1)  # [S, 1]
    return mx.where(mask, full.astype(mx.float16), merged)


def merge_similarity(a: mx.array, b: mx.array) -> dict:
    """Diagnostic: mean direction cosine + magnitude ratio between two tensors."""
    _, da = to_mag_dir(a)
    _, db = to_mag_dir(b)
    cos = mx.sum(da * db, axis=-1)
    ma = mx.sqrt(mx.sum(a.astype(mx.float32) ** 2, axis=-1))
    mb = mx.sqrt(mx.sum(b.astype(mx.float32) ** 2, axis=-1))
    return {
        "dir_cosine": float(mx.mean(cos).item()),
        "mag_ratio": float(mx.mean(ma / mx.maximum(mb, 1e-8)).item()),
    }


__all__ = [
    "pair_layers_depth",
    "to_mag_dir",
    "slerp",
    "MergeResult",
    "merge_pair",
    "reconstruct_layer",
    "merge_similarity",
]

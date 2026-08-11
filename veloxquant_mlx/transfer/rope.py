"""Content-space RoPE handling for cross-model KV transfer (paper §3.3).

The KV cache stores keys *after* RoPE has been applied, so a stored key
confounds two things: the token's semantic content and its absolute position.
Fitting a cross-model map on rotated keys ties the fitted weights to the
position distribution seen during calibration (1024 tokens in the paper). The
paper's fix is to map in *content space*: un-rotate the source's keys, fit and
apply the linear map there, then re-rotate with the *target's* RoPE.

    K̂_t = (K_s · R_Θs⁻¹(t) · W_K + b_K) · R_Θt(t)

Because ``R_Θ`` is orthogonal, the inversion is exact and costs no accuracy.
Values carry no positional encoding and are never touched by this module.

Why this is not :func:`~veloxquant_mlx.quantizers.a2ats_rope.rope_remap_positions`:
that function composes a de-rotation and a re-rotation *at the same RoPE base*,
so the two collapse into one rotation by ``new_pos - old_pos``. Here the source
and target are different models and generally have **different ``rope_theta``**
(e.g. Qwen3 0.6B and 1.7B need not agree). With ``θ_s ≠ θ_t`` the per-dimension
frequencies differ, the angles do not share a common factor, and the rotations
do **not** collapse — the composition must be evaluated as two genuine
rotations, or equivalently one rotation by the per-dimension angle difference
``p·(1/θ_t^(d/half) − 1/θ_s^(d/half))``. :func:`recode_rope` does the latter,
which is both cheaper and numerically better than rotating twice.

Convention: NeoX-style (``traditional=False``) first-half/second-half pairing,
matching MLX's default and every ``mlx_lm`` Llama/Mistral/Qwen attention module
this repo targets. ``traditional=True`` models are not handled.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["rope_angles", "strip_rope", "apply_rope", "recode_rope"]


def rope_angles(positions: mx.array, head_dim: int, base: float) -> mx.array:
    """Per-position, per-dimension-pair RoPE angles.

    Args:
        positions: ``[N]`` absolute positions.
        head_dim: Even head dimension ``D``.
        base: RoPE frequency base (``rope_theta``).

    Returns:
        ``[N, D // 2]`` float32 angles ``position * base^(-d/half)``.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))
    return positions.astype(mx.float32)[:, None] * inv_freq[None, :]


def _rotate_by(x: mx.array, angles: mx.array) -> mx.array:
    """Rotate ``[..., N, D]`` by ``[N, D//2]`` angles, NeoX-style.

    Computed in float32 and cast back to ``x``'s dtype at the end: the
    calibration path accumulates over ~128K tokens per head, and doing the
    trigonometry in fp16 loses enough precision to show up in the ridge fit.
    """
    out_dtype = x.dtype
    half = x.shape[-1] // 2
    cos, sin = mx.cos(angles), mx.sin(angles)
    x1 = x[..., :half].astype(mx.float32)
    x2 = x[..., half:].astype(mx.float32)
    rotated = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.astype(out_dtype)


def strip_rope(keys: mx.array, positions: mx.array, base: float) -> mx.array:
    """Remove RoPE from already-rotated keys, yielding content-space keys.

    The inverse rotation is the forward rotation at negated angles, exact
    because ``R_Θ`` is orthogonal.

    Args:
        keys: ``[..., N, D]`` keys as stored in the cache (post-RoPE).
        positions: ``[N]`` absolute positions the keys were rotated at.
        base: The source model's ``rope_theta``.

    Returns:
        ``[..., N, D]`` position-free content-space keys.
    """
    if keys.shape[-2] == 0:
        return keys
    return _rotate_by(keys, -rope_angles(positions, keys.shape[-1], base))


def apply_rope(keys: mx.array, positions: mx.array, base: float) -> mx.array:
    """Apply RoPE to content-space keys at the given positions.

    Args:
        keys: ``[..., N, D]`` content-space (un-rotated) keys.
        positions: ``[N]`` absolute positions to rotate to.
        base: The target model's ``rope_theta``.

    Returns:
        ``[..., N, D]`` rotated keys, ready to serve as a target KV cache.
    """
    if keys.shape[-2] == 0:
        return keys
    return _rotate_by(keys, rope_angles(positions, keys.shape[-1], base))


def recode_rope(
    keys: mx.array,
    positions: mx.array,
    source_base: float,
    target_base: float,
    use_metal: bool = True,
) -> mx.array:
    """Re-encode keys from the source model's RoPE to the target's, in one pass.

    Equivalent to ``apply_rope(strip_rope(k, p, θ_s), p, θ_t)`` but evaluated as
    a single rotation by the per-dimension angle *difference*, so the
    trigonometry runs once instead of twice. Used when a mapper is identity in
    content space (or for testing the RoPE path in isolation) — the fitted
    path in :mod:`veloxquant_mlx.transfer.apply` cannot use this shortcut,
    because the linear map has to be applied *between* the two rotations.

    On a 3-D ``[BH, N, D]`` input with Metal available this dispatches to
    :func:`~veloxquant_mlx.metal._crosskv_rope.crosskv_rope_recode`, which
    fuses the whole thing into one kernel (~13× faster at 8K context). Any
    other rank, or no GPU, falls back to the MLX path below; both compute the
    same angles, so results agree to floating-point rounding.

    Args:
        keys: ``[..., N, D]`` keys rotated under ``source_base``.
        positions: ``[N]`` absolute positions.
        source_base: Source model's ``rope_theta``.
        target_base: Target model's ``rope_theta``.
        use_metal: Set False to force the pure-MLX path (used by the kernel's
            own equivalence tests).

    Returns:
        ``[..., N, D]`` keys rotated as though generated by the target model.
    """
    if keys.shape[-2] == 0:
        return keys
    if use_metal and keys.ndim == 3:
        from veloxquant_mlx.metal._crosskv_rope import crosskv_rope_recode, is_available

        if is_available():
            return crosskv_rope_recode(keys, positions, source_base, target_base)
    d = keys.shape[-1]
    delta = rope_angles(positions, d, target_base) - rope_angles(positions, d, source_base)
    return _rotate_by(keys, delta)

"""A2ATS-adapted windowed RoPE — distance-gated exact/approximate rotation.

Inspired by "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu,
Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings,
aclanthology.org/2025.findings-acl.644). Documented as "A2ATS-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

This module ports the paper's RoPE-decoupling idea: tokens within a trailing
window of the current decode position get *exact* RoPE (cheap — the window is
small and re-derived every step anyway); tokens outside the window are left
**unrotated** on the key side (Eq. 12, ``k̃_i = k_i``), with the constant
``R_b`` rotation representing the "far" distance class applied to the *query*
instead (Eq. 11, ``u_ij = q_i R_b k_j^T``). This avoids a distinct exact
rotation per absolute position for the bulk of a long sequence, and — the
actual point — leaves far keys position-free so a single shared codebook can
quantize them (§3.1, Observation 2). This is a different axis from CommVQ-adapted
(:mod:`veloxquant_mlx.quantizers.comm_vq`), which instead constrains the VQ
*codebook* to a RoPE-commuting subspace and applies exact RoPE uniformly to
every position after decode. A2ATS-adapted changes *when* exact-vs-approximate
RoPE is paid for, gated by token distance from the query; CommVQ-adapted
changes what the codebook can represent. The two are independent axes and, in
principle, composable — not attempted here.

Adaptation notes (stated plainly):
  - The windowed approximation trades accuracy for avoiding per-position exact
    rotation on distant tokens. It is not free: every far token's true
    relative position ``i - j`` is replaced by the single constant ``b``, so
    far-token attention scores carry approximation error by construction.
    State this, don't imply strictly-better.
  - No CUDA kernel fusion reproduced — same MLX/Metal disclaimer as every
    other VQ-family method in this repo (VecInfer-adapted, CommVQ-adapted):
    the benefit on Apple Silicon is memory footprint, not throughput.

Public API:
  a2ats_apply_exact_rope      — standard RoPE at each token's own position
  a2ats_apply_windowed_rope   — key side: exact within window, unrotated outside
  a2ats_apply_far_query_rope  — query side: constant R_b for the far bucket
  rope_remap_positions        — de-rotate + re-rotate already-RoPE'd vectors
                                to a new position (used by H2O-adapted, see
                                cache/h2o_cache.py, to fix up positions after
                                eviction removes rows from the middle of a
                                stored sequence — not A2ATS-specific, kept
                                here because it shares this module's RoPE
                                cos/sin/rotate primitives)
"""

from __future__ import annotations

import mlx.core as mx


def _rope_cos_sin(positions: mx.array, head_dim: int, base: float) -> tuple:
    """Per-position RoPE cos/sin tables.

    Args:
        positions: ``[N]`` int/float positions (need not be contiguous or
            sorted — each token supplies its own absolute position).
        head_dim: Even hidden dimension ``D``.
        base: RoPE frequency base.

    Returns:
        ``(cos, sin)`` each ``[N, D // 2]`` float16.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))  # [half]
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]  # [N, half]
    return mx.cos(angles).astype(mx.float16), mx.sin(angles).astype(mx.float16)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply the standard RoPE rotation given precomputed per-token cos/sin."""
    half = x.shape[-1] // 2
    x1, x2 = x[:, :half], x[:, half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def a2ats_apply_exact_rope(
    x: mx.array,
    positions: mx.array,
    base: float = 10000.0,
) -> mx.array:
    """Apply exact RoPE to each token at its own absolute position.

    Args:
        x: ``[N, D]`` fp16/fp32 token vectors (dequantized, pre-RoPE).
        positions: ``[N]`` int/float absolute positions, one per token.
        base: RoPE frequency base.

    Returns:
        ``[N, D]`` fp16 rotated vectors.
    """
    if x.shape[0] == 0:
        return x.astype(mx.float16)
    cos, sin = _rope_cos_sin(positions, x.shape[-1], base)
    return _rotate(x.astype(mx.float16), cos, sin)


def rope_remap_positions(
    x: mx.array,
    old_positions: mx.array,
    new_positions: mx.array,
    base: float = 10000.0,
) -> mx.array:
    """De-rotate already-RoPE'd vectors from ``old_positions`` and re-rotate
    them at ``new_positions`` — used to fix up stored keys after an eviction
    cache (H2O-adapted, and structurally any other row-dropping eviction
    cache) removes rows and leaves the survivors' *storage index* out of sync
    with the absolute position baked into their rotation.

    RoPE rotates each ``(x1, x2)`` pair by angle ``position * inv_freq``; the
    inverse rotation is the same formula at ``-position``. Composing "rotate
    by ``-old_position``" then "rotate by ``new_position``" is equivalent to a
    single rotation by ``new_position - old_position``, computed directly here
    (one rotation, not two) for numerical accuracy.

    Args:
        x: ``[N, D]`` fp16/fp32 vectors, already RoPE'd at ``old_positions``.
        old_positions: ``[N]`` absolute positions each row was originally
            rotated at (must be the *exact* positions used at rotation time).
        new_positions: ``[N]`` absolute positions to re-rotate each row to
            (typically contiguous indices reflecting a post-eviction gap-free
            layout).
        base: RoPE frequency base — must match the base the model's attention
            module actually used, or the de-rotation will not cancel out.

    Returns:
        ``[N, D]`` fp16 vectors, equivalent to having been generated fresh and
        rotated directly at ``new_positions``.

    Caveat: assumes MLX's default ``traditional=False`` (NeoX-style first/second-
    half split) RoPE convention — the same one :func:`a2ats_apply_exact_rope`
    uses and the one ``mlx_lm``'s Llama/Mistral/Qwen attention modules default
    to. Models configured with ``traditional=True`` (adjacent-pair
    interleaving) are not handled by this function.
    """
    if x.shape[0] == 0:
        return x.astype(mx.float16)
    delta = new_positions.astype(mx.float32) - old_positions.astype(mx.float32)
    cos, sin = _rope_cos_sin(delta, x.shape[-1], base)
    return _rotate(x.astype(mx.float16), cos, sin)


def a2ats_apply_windowed_rope(
    x: mx.array,
    positions: mx.array,
    query_position: int,
    window: int = 128,
    base: float = 10000.0,
) -> mx.array:
    """Windowed RoPE on the **key** side (paper Eq. 11-12).

    Tokens with ``query_position - position < window`` ("near") get exact
    RoPE at their own position. Tokens outside the window ("far") are
    returned **unrotated** — that is the paper's Eq. (12), ``k̃_i = k_i``:

        Since post-PE key ``k̃_i`` states are identical to their pre-PE
        counterparts ``k_i``, WRoPE decouples the positional dependency
        from post-PE representations.

    That decoupling is the whole point of WRoPE — it is what makes a single
    shared codebook viable across inputs (paper §3.1, Observation 2). The
    relative-position information for far tokens is *not* dropped: it is
    carried on the **query** side by the constant rotation ``R_b``, applied
    via :func:`a2ats_apply_far_query_rope`, so that the far-token attention
    score is ``u_ij = q_i R_b k_j^T`` exactly as Eq. (11) specifies.

    .. note::
       Before the fix for :issue:`29` this function rotated far keys by a
       fixed ``R_window``. That is a different operation from Eq. (11)
       (``q_i (k_j R_w)^T = q_i R_w^T k_j^T``, rotation on the wrong operand
       and in the wrong direction), it injected reconstruction error the
       paper never incurs, and it left far keys in a rotated frame —
       precisely the situation §3.1 identifies as the problem WRoPE exists
       to solve.

    Boundary behavior (both intentional, not error cases):
      - ``window <= 0``: every token is treated as "far" — every key is
        returned unrotated (pure pre-RoPE frame).
      - ``window`` at or beyond the trailing distance of every token in
        ``positions``: every token is "near" — always exact, equivalent to
        :func:`a2ats_apply_exact_rope` (matches CommVQ-adapted's uniform
        treatment).

    Args:
        x: ``[N, D]`` fp16/fp32 token vectors (dequantized, pre-RoPE).
        positions: ``[N]`` int/float absolute positions, one per token.
        query_position: Absolute position of the current decode step.
        window: Trailing distance (in positions) treated as "near"/exact.
        base: RoPE frequency base.

    Returns:
        ``[N, D]`` fp16 vectors: exact-rotated for near tokens, unrotated
        (pre-RoPE) for far tokens.
    """
    if x.shape[0] == 0:
        return x.astype(mx.float16)

    x16 = x.astype(mx.float16)
    distance = mx.array(query_position, dtype=mx.float32) - positions.astype(mx.float32)
    near_mask = distance < float(window)  # [N] bool; window<=0 -> all False

    exact = a2ats_apply_exact_rope(x16, positions, base=base)

    # Eq. (12): far keys stay in their pre-RoPE frame. The constant R_b that
    # encodes "far" relative position lives on the query side instead — see
    # a2ats_apply_far_query_rope.
    return mx.where(near_mask[:, None], exact, x16)


def a2ats_apply_far_query_rope(
    q: mx.array,
    b: int = 2048,
    base: float = 10000.0,
) -> mx.array:
    """Apply the paper's constant far-token rotation ``R_b`` to the query.

    This is the query-side half of WRoPE. Paper Eq. (11) scores a far token
    as ``u_ij = q_i R_b k_j^T``; with far keys left unrotated by
    :func:`a2ats_apply_windowed_rope` (Eq. 12), the ``R_b`` factor must be
    applied here for far-token attention scores to be correct.

    ``b`` is a hyperparameter **independent of the window size** ``w``. The
    paper's §5.1 Implementation Details sets ``w = 64`` and ``b = 2048`` —
    they differ by a factor of 32, so they are plainly not one value. (Before
    the fix for :issue:`29` this repo derived the far offset from ``window``,
    hardcoding ``b == w`` and making the paper's own configuration
    inexpressible.)

    Args:
        q: ``[D]`` or ``[N, D]`` query vector(s), pre-RoPE.
        b: Constant relative position representing the "far" distance class.
        base: RoPE frequency base.

    Returns:
        Same shape as ``q``, fp16, rotated by the constant ``R_b``.
    """
    squeeze = q.ndim == 1
    q2 = q[None, :] if squeeze else q
    if q2.shape[0] == 0:
        return q.astype(mx.float16)

    offset = mx.array([float(b)], dtype=mx.float32)
    cos, sin = _rope_cos_sin(offset, q2.shape[-1], base)  # [1, half]
    out = _rotate(q2.astype(mx.float16), cos, sin)  # broadcasts over N
    return out[0] if squeeze else out


__all__ = [
    "a2ats_apply_exact_rope",
    "a2ats_apply_windowed_rope",
    "a2ats_apply_far_query_rope",
    "rope_remap_positions",
]

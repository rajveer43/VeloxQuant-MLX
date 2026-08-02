"""A2ATS-adapted windowed RoPE — distance-gated exact/approximate rotation.

Inspired by "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu,
Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings,
aclanthology.org/2025.findings-acl.644). Documented as "A2ATS-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

This module ports the paper's RoPE-decoupling idea: tokens within a trailing
window of the current decode position get *exact* RoPE (cheap — the window is
small and re-derived every step anyway); tokens outside the window get a
single *fixed-offset approximate* rotation representing the "far" distance
class, avoiding a distinct exact rotation per absolute position for the bulk
of a long sequence. This is a different axis from CommVQ-adapted
(:mod:`veloxquant_mlx.quantizers.comm_vq`), which instead constrains the VQ
*codebook* to a RoPE-commuting subspace and applies exact RoPE uniformly to
every position after decode. A2ATS-adapted changes *when* exact-vs-approximate
RoPE is paid for, gated by token distance from the query; CommVQ-adapted
changes what the codebook can represent. The two are independent axes and, in
principle, composable — not attempted here.

Adaptation notes (stated plainly):
  - The windowed approximation trades accuracy for avoiding per-position exact
    rotation on distant tokens. It is not free: distant-token reconstruction
    has nonzero RoPE error by construction. State this, don't imply strictly-
    better.
  - No CUDA kernel fusion reproduced — same MLX/Metal disclaimer as every
    other VQ-family method in this repo (VecInfer-adapted, CommVQ-adapted):
    the benefit on Apple Silicon is memory footprint, not throughput.

Public API:
  a2ats_apply_exact_rope     — standard RoPE at each token's own position
  a2ats_apply_windowed_rope  — distance-gated exact/approximate RoPE
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


def a2ats_apply_windowed_rope(
    x: mx.array,
    positions: mx.array,
    query_position: int,
    window: int = 128,
    base: float = 10000.0,
) -> mx.array:
    """Distance-gated RoPE: exact within the trailing window, fixed-offset
    approximate outside it.

    Tokens with ``query_position - position < window`` get exact RoPE at
    their own position. Tokens outside the window all receive a single
    shared approximate rotation, computed once at the representative
    "far" offset ``window`` (rather than each token's true, more distant,
    offset) — the paper's approximation for the retrieval-irrelevant bulk
    of a long sequence.

    Boundary behavior (both intentional, not error cases):
      - ``window <= 0``: every token is treated as "far" — always
        approximate.
      - ``window`` at or beyond the trailing distance of every token in
        ``positions``: every token is treated as "near" — always exact,
        equivalent to :func:`a2ats_apply_exact_rope` (matches CommVQ-adapted's
        uniform treatment).

    Args:
        x: ``[N, D]`` fp16/fp32 token vectors (dequantized, pre-RoPE).
        positions: ``[N]`` int/float absolute positions, one per token.
        query_position: Absolute position of the current decode step.
        window: Trailing distance (in positions) treated as "near"/exact.
        base: RoPE frequency base.

    Returns:
        ``[N, D]`` fp16 rotated vectors.
    """
    if x.shape[0] == 0:
        return x.astype(mx.float16)

    x16 = x.astype(mx.float16)
    distance = mx.array(query_position, dtype=mx.float32) - positions.astype(mx.float32)
    near_mask = distance < float(window)  # [N] bool; window<=0 -> all False

    exact = a2ats_apply_exact_rope(x16, positions, base=base)

    # Single representative "far" offset: the trailing edge of the window
    # (or 0 if window<=0, degenerating to position-0 i.e. unrotated-frame
    # reconstruction — the paper's coarsest approximation bucket).
    far_offset = mx.array([max(float(window), 0.0)], dtype=mx.float32)
    far_cos, far_sin = _rope_cos_sin(far_offset, x.shape[-1], base)  # [1, half]
    approx = _rotate(x16, far_cos, far_sin)  # broadcasts [1, half] against [N, half]

    return mx.where(near_mask[:, None], exact, approx)


__all__ = [
    "a2ats_apply_exact_rope",
    "a2ats_apply_windowed_rope",
]

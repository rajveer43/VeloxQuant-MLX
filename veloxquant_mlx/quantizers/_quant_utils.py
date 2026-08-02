"""Shared low-level quantization helpers used by multiple quantizers."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _group_quant_dequant(x: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Asymmetric min/max group quantization along axis 0. Returns fp16.

    Groups are formed along axis 0 (token/sequence axis). Each group is
    independently scaled and zero-pointed using the group's min/max values.

    Args:
        x: Input array [N, D] fp16 or fp32.
        b: Bit width (1–8).
        group_size: Number of rows per quantization group.

    Returns:
        Quantized-then-dequantized array [N, D] fp16.
    """
    n, d = x.shape
    gs = group_size
    n_groups = (n + gs - 1) // gs
    pad = n_groups * gs - n
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    xg = x32.reshape(n_groups, gs, d)
    gmin = mx.min(xg, axis=1, keepdims=True)
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << b) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    recon = codes * scale + gmin
    return recon.reshape(n_groups * gs, d)[:n].astype(mx.float16)


def _truncated_svd(
    x: mx.array,
    rank: Optional[int] = None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array, mx.array, mx.array]:
    """Truncated SVD of a centered-or-raw matrix ``[N, D]`` → ``(U_r, s_r, Vt_r)``.

    Computes the economy SVD on the CPU stream (MLX ``linalg.svd`` is CPU-only)
    and truncates to ``rank`` singular components. When ``rank`` is None the rank
    is chosen as the smallest prefix whose cumulative singular-value energy
    reaches ``energy_threshold``.

    This is the shared truncation logic used by SVDq (signal SVD), PALU
    (group-head signal SVD), and GEAR (quantization-residual SVD). The caller is
    responsible for any mean-centering it needs before calling.

    Args:
        x: Input matrix ``[N, D]`` fp16 or fp32.
        rank: Explicit rank ``r``. If None, chosen by ``energy_threshold``.
        energy_threshold: Fraction of singular-value energy to retain when
            ``rank`` is None.

    Returns:
        ``(U_r, s_r, Vt_r)`` where ``U_r`` is ``[N, r]``, ``s_r`` is ``[r]``
        (descending), and ``Vt_r`` is ``[r, D]`` — all fp32.
    """
    x32 = x.astype(mx.float32)
    n, d = x32.shape
    U, s_vals, Vt = mx.linalg.svd(x32, stream=mx.cpu)
    mx.eval(U, s_vals, Vt)

    if rank is None:
        total = float(mx.sum(s_vals).item())
        if total < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = int(s_vals.shape[0])
            for i, sv in enumerate(s_vals.tolist()):
                cumsum += sv
                if cumsum / total >= energy_threshold:
                    rank = i + 1
                    break
    rank = max(1, min(int(rank), int(s_vals.shape[0]), d))

    return U[:, :rank], s_vals[:rank], Vt[:rank, :]


__all__ = ["_group_quant_dequant", "_truncated_svd"]

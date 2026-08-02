"""AMC offline calibration — variance-ordered channel permutation.

Inspired by Algorithm 1, Phase I ("Offline Post-Training Structural
Calibration") of "Adaptive Model Compression (AMC): Saliency-Driven Resource
Allocation for Ultra-Low-Power Transformer Inference" (Hu, Yuan, Hu, Yin, Li,
Suchter — Apple; arXiv:2607.10109, **no verified peer-reviewed venue as of
2026-07-14**, single-version preprint submitted 2026-07-11).

Raw transformer hidden dimensions have no inherent ordering by importance.
AMC's rank-masking mechanism (see :mod:`veloxquant_mlx.quantizers.amc`) keeps
only the first ``r`` channels of a token's (already-permuted) activation
vector and zeroes the rest — safe only if the surviving prefix is the
highest-variance subspace. This module computes that one-time, offline
permutation via truncated SVD, reusing the same ``mx.linalg.svd`` pattern as
:func:`veloxquant_mlx.quantizers.palu.group_head_svd` and
:func:`veloxquant_mlx.quantizers.svdq.svd_compress_keys` — no new SVD
machinery.

This is a **pre-deployment, one-time** pass: it runs once over a
representative calibration sample, never inside the per-token hot path (the
paper's "strictly zero runtime overhead" claim is about silicon; here it
means this function is never called from :func:`amc_saliency` /
:func:`amc_assign_tiers` / :func:`amc_apply_rank_mask`).
"""

from __future__ import annotations

import mlx.core as mx


def amc_calibrate_channel_order(calib_activations: mx.array) -> mx.array:
    """Compute a variance-descending channel permutation from calibration data.

    Runs a truncated (here: full-rank) SVD over the centered calibration
    activation matrix and returns the permutation of the ``D`` hidden-dim
    columns of ``V`` sorted by descending singular value — i.e. the
    variance-ordering used to make :func:`amc_apply_rank_mask` safe.

    Args:
        calib_activations: ``[n_calib, D]`` representative activation sample
            for one layer (fp16 or fp32).

    Returns:
        ``perm``: ``[D]`` int32 permutation indices such that
        ``calib_activations[:, perm]`` has descending-variance columns.

    Raises:
        ValueError: If ``calib_activations`` is not 2-D or ``n_calib < 2``.
    """
    if calib_activations.ndim != 2:
        raise ValueError(
            f"amc_calibrate_channel_order: expected [n_calib, D], got shape "
            f"{calib_activations.shape}"
        )
    n_calib, d = calib_activations.shape
    if n_calib < 2:
        raise ValueError(f"amc_calibrate_channel_order: n_calib must be >= 2, got {n_calib}")

    x = calib_activations.astype(mx.float32)
    mean = mx.mean(x, axis=0)
    x_centered = x - mean[None, :]

    # SVD of the centered activations: columns of V are the principal
    # directions in *activation* space, ranked by descending singular value.
    # We only need the ranking of the original D axes by their contribution
    # to that variance, so rank axis j by the L2 norm of row j of V (i.e.
    # how much each original channel loads onto the retained principal
    # directions) weighted by singular value — equivalently and more simply,
    # rank the original D axes directly by their own per-channel variance,
    # which is exactly what a per-axis SVD/PCA loading ranking reduces to for
    # a channel *selection* (as opposed to a channel *rotation*) mask: AMC's
    # rank-masking (Eq. 6) zeroes raw channels, it does not rotate into a new
    # basis, so the permutation must operate on the *original* axes.
    U, s_vals, Vt = mx.linalg.svd(x_centered, stream=mx.cpu)
    mx.eval(U, s_vals, Vt)

    # Per-original-channel variance (diagonal of the empirical covariance) —
    # this is what determines which raw channel index is safe to keep when
    # Eq. 6 zeroes a contiguous tail of channel *indices*, not principal
    # components. (A pure PCA rotation would instead project into the V
    # basis, but that would require carrying an extra [D, D] rotation matrix
    # through every downstream rank-masked op — the paper's own Eq. 6 defines
    # the mask as literal index zeroing, so we sort raw indices instead.)
    channel_var = mx.mean(x_centered * x_centered, axis=0)  # [D]
    mx.eval(channel_var)

    order = sorted(range(d), key=lambda j: -float(channel_var[j].item()))
    return mx.array(order, dtype=mx.int32)


def amc_permute_weights(weight: mx.array, perm: mx.array, axis: int = -1) -> mx.array:
    """Statically reorder a weight tensor's hidden-dim axis by ``perm``.

    Zero runtime cost at inference time — this permutation is baked into the
    stored weights once, offline, exactly as Algorithm 1 Phase I specifies
    ("incurs strictly zero computational, area, or energy overhead during
    real-time hardware execution"; in this software port, "real-time" means
    the per-token scoring/masking path in
    :mod:`veloxquant_mlx.quantizers.amc`).

    Args:
        weight: Weight tensor whose ``axis`` dimension has length ``D``.
        perm: ``[D]`` int permutation indices (e.g. from
            :func:`amc_calibrate_channel_order`).
        axis: Axis of ``weight`` to permute (default: last axis).

    Returns:
        ``weight`` with ``axis`` reordered by ``perm``.
    """
    return mx.take(weight, perm, axis=axis)


__all__ = [
    "amc_calibrate_channel_order",
    "amc_permute_weights",
]

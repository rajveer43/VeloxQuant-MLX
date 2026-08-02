"""NSNQuant quantizer — calibration-free universal-codebook vector quantization.

Inspired by "NSNQuant: A Double Normalization Approach for Calibration-Free
Low-Bit Vector Quantization of KV Cache" (Donghyun Son, Euntae Choi, Sungjoo
Yoo — NeurIPS 2025; arXiv:2505.18231, OpenReview id boNYskaXnO). Documented as
"NSNQuant-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

How it differs from the repo's other VQ methods: every existing VQ method
either fits its codebook to the data at hand (RVQ's per-sequence k-means,
CommVQ) or uses a data-independent *geometric* code (RaBitQ sign codes,
VecInfer binary, PolarQuant polar grids, QJL sketches). NSNQuant inverts the
relationship — it **reshapes the data to match a fixed code**: a
Normalize-Shift-Normalize (NSN) transform plus a Hadamard rotation maps K/V
token vectors onto (approximately) the standard normal distribution, so one
codebook built *offline from synthetic Gaussian samples* — never from model
activations — quantizes any model, layer, or dataset. Calibration-free by
construction.

Core mechanism (per chunk of tokens):
  1. **Normalize** (token-wise): scale each token to norm sqrt(d); keep the
     scale ``s1``.
  2. **Shift** (channel-wise): subtract the chunk's own channel mean ``o``
     (online statistics — never global, never calibrated).
  3. **Normalize** again (token-wise): rescale to norm sqrt(d); keep ``s2``.
     (The second normalization slightly perturbs the zero mean; the paper
     shows the deviation is negligible.)
  4. **Hadamard transform**: rotate so channels decorrelate and the empirical
     distribution closely matches an isotropic standard normal.
  5. **Vector quantization**: 8-dim subvectors matched by cosine against a
     universal codebook. 2-bit: uint8 sign mask + uint8 index into a
     positive-orthant "magnitude" codebook (2 bits/element). 1-bit: uint8
     index into a "signed" codebook only (1 bit/element).
  6. **Restoration**: ``x_hat = s1 * (s2 * x_nsn + o)`` after inverse
     Hadamard.

What we do NOT implement (see ``paper/NEW_METHOD_SURVEY_V11.md`` for the full
rationale):
  * Pre-RoPE key handling + the paper's RoPE-aware attention kernel — our
    cache wrappers receive **post-RoPE** keys from ``update_and_fetch``, so
    NSN + Hadamard run post-RoPE on keys. This is the central simplification
    of the adaptation.
  * Value-side Hadamard fused into the projection layers (model surgery) —
    we apply the Hadamard explicitly to cached values instead.
  * Gradient fine-tuning of the codebook — deterministic seeded spherical
    k-means on synthetic standard-normal samples only.
  * 4-bit double quantization of the NSN metadata — ``s1``/``s2``/``o`` are
    stored fp16 and counted honestly in the byte accounting.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np

from veloxquant_mlx.math.rotation import is_hadamard_compatible

_EPS = 1e-8

# Module-level codebook cache: built once per process per parameter set.
# Deterministic (seeded numpy) — never persisted to disk, never committed.
_CODEBOOK_CACHE: dict[tuple[int, int, int, str], np.ndarray] = {}


# ----------------------------------------------------------------------
# NSN transform
# ----------------------------------------------------------------------


def nsn_transform(x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Apply the Normalize-Shift-Normalize transform to a chunk of tokens.

    Args:
        x: ``(..., T, d)`` array — a self-contained chunk. The channel mean
           ``o`` is computed from *this chunk only* (online statistics).

    Returns:
        ``(x_nsn, s1, o, s2)`` where ``x_nsn`` is fp32 ``(..., T, d)`` with
        per-token norm sqrt(d), ``s1``/``s2`` are fp16 ``(..., T, 1)``
        per-token scales, and ``o`` is the fp16 ``(..., 1, d)`` per-chunk
        channel mean.
    """
    d = x.shape[-1]
    sqrt_d = math.sqrt(d)
    x32 = x.astype(mx.float32)

    n1 = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
    s1 = mx.maximum(n1 / sqrt_d, _EPS)
    x1 = x32 / s1

    o = mx.mean(x1, axis=-2, keepdims=True)
    x2 = x1 - o

    n2 = mx.sqrt(mx.sum(x2 * x2, axis=-1, keepdims=True))
    s2 = mx.maximum(n2 / sqrt_d, _EPS)
    x_nsn = x2 / s2

    return (
        x_nsn,
        s1.astype(mx.float16),
        o.astype(mx.float16),
        s2.astype(mx.float16),
    )


def nsn_inverse(x_nsn: mx.array, s1: mx.array, o: mx.array, s2: mx.array) -> mx.array:
    """Restore a chunk from its NSN form: ``x_hat = s1 * (s2 * x_nsn + o)``.

    Exact (to fp16 metadata precision) when ``x_nsn`` is unquantized.
    """
    return s1.astype(mx.float32) * (
        s2.astype(mx.float32) * x_nsn.astype(mx.float32) + o.astype(mx.float32)
    )


# ----------------------------------------------------------------------
# Universal codebook (offline, synthetic Gaussian — calibration-free)
# ----------------------------------------------------------------------


def build_universal_codebook(
    codebook_size: int = 256,
    subvector_dim: int = 8,
    seed: int = 1234,
    n_samples: int = 262_144,
    iters: int = 25,
    kind: str = "signed",
) -> np.ndarray:
    """Build the model-independent codebook via seeded spherical k-means.

    Samples are synthetic standard-normal vectors (``np.random.default_rng``)
    — **never model activations**, which is what keeps NSNQuant
    calibration-free. Deterministic: identical arguments yield bitwise
    identical output. The paper additionally fine-tunes its codebook with
    gradient descent on a cosine objective; we skip that step (adaptation
    decision — expect a slightly worse codebook than the paper's).

    Args:
        codebook_size: Number of centroids (256 -> uint8 indices).
        subvector_dim: Dimension of each subvector (8 in the paper).
        seed: RNG seed.
        n_samples: Synthetic training samples.
        iters: Spherical k-means iterations.
        kind: ``"signed"`` (1-bit: k-means on raw samples) or ``"magnitude"``
            (2-bit: k-means on ``|samples|`` — the codebook lives in the
            positive orthant and a per-subvector sign mask restores
            orientation).

    Returns:
        Float32 array of shape ``(codebook_size, subvector_dim)`` with
        unit-norm rows.
    """
    if kind not in ("signed", "magnitude"):
        raise ValueError(f"build_universal_codebook: unknown kind {kind!r}")
    key = (codebook_size, subvector_dim, seed, kind)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_samples, subvector_dim)).astype(np.float32)
    if kind == "magnitude":
        samples = np.abs(samples)
    norms = np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), _EPS)
    samples = samples / norms

    # Deterministic init: first draw of centroid indices.
    init_idx = rng.choice(n_samples, size=codebook_size, replace=False)
    centroids = samples[init_idx].copy()

    chunk = 65_536  # bound the (chunk, codebook_size) similarity matrix
    for _ in range(iters):
        assign = np.empty(n_samples, dtype=np.int64)
        for start in range(0, n_samples, chunk):
            block = samples[start : start + chunk]
            assign[start : start + block.shape[0]] = np.argmax(block @ centroids.T, axis=1)
        # Mean of members, then re-project to the unit sphere (spherical
        # k-means update). Empty clusters are re-seeded deterministically.
        for c in range(codebook_size):
            members = samples[assign == c]
            if members.shape[0] == 0:
                centroids[c] = samples[int(rng.integers(n_samples))]
            else:
                centroids[c] = members.mean(axis=0)
        cn = np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), _EPS)
        centroids = centroids / cn

    centroids = centroids.astype(np.float32)
    _CODEBOOK_CACHE[key] = centroids
    return centroids


# ----------------------------------------------------------------------
# Subvector VQ encode / decode
# ----------------------------------------------------------------------


def _check_subvector_dim(d: int, subvector_dim: int) -> None:
    if d % subvector_dim != 0:
        raise ValueError(
            f"nsnquant: last dim {d} must be divisible by subvector_dim {subvector_dim}"
        )


def vq_encode(
    x_nsn: mx.array,
    codebook: np.ndarray,
    bits: int,
) -> dict:
    """Encode NSN-space tokens against the universal codebook.

    Args:
        x_nsn: ``(..., T, d)`` fp32 array (NSN + Hadamard domain).
        codebook: ``(K, sub_d)`` unit-norm float32 codebook —
            ``kind="magnitude"`` for ``bits=2``, ``kind="signed"`` for
            ``bits=1`` (see :func:`build_universal_codebook`).
        bits: 2 (sign mask + index) or 1 (index only).

    Returns:
        Dict with ``idx`` (uint8 ``(..., T, d/sub_d)``), ``signs`` (uint8
        bitmask, 2-bit only), and the shape metadata needed by
        :func:`vq_decode`.
    """
    if bits not in (1, 2):
        raise ValueError(f"nsnquant: bits must be 1 or 2, got {bits}")
    d = x_nsn.shape[-1]
    sub_d = int(codebook.shape[1])
    _check_subvector_dim(d, sub_d)

    cb = mx.array(codebook)  # (K, sub_d)
    sub = x_nsn.reshape(x_nsn.shape[:-1] + (d // sub_d, sub_d))

    if bits == 2:
        # Positive-orthant match: cosine argmax against |subvector| (argmax of
        # the dot with unit centroids is scale-invariant, so no normalization
        # of the query is needed).
        mags = mx.abs(sub)
        idx = mx.argmax(mags @ cb.T, axis=-1).astype(mx.uint8)
        pow2 = mx.array((1 << np.arange(sub_d, dtype=np.uint32)).astype(np.uint32))
        signs = mx.sum((sub >= 0).astype(mx.uint32) * pow2, axis=-1).astype(
            mx.uint8 if sub_d <= 8 else mx.uint32
        )
        return {"idx": idx, "signs": signs, "d": d, "sub_d": sub_d, "bits": 2}

    idx = mx.argmax(sub @ cb.T, axis=-1).astype(mx.uint8)
    return {"idx": idx, "signs": None, "d": d, "sub_d": sub_d, "bits": 1}


def vq_decode(encoded: dict, codebook: np.ndarray) -> mx.array:
    """Decode :func:`vq_encode` output back to NSN-space tokens.

    Codebook entries are unit-subvector approximations, so after lookup each
    token is renormalized to norm sqrt(d) — the scale-adjustment analog that
    keeps the stored ``s2`` valid on restoration.
    """
    idx = encoded["idx"]
    d, sub_d, bits = encoded["d"], encoded["sub_d"], encoded["bits"]
    cb = mx.array(codebook)

    sub_hat = cb[idx.astype(mx.int32)]  # (..., T, d/sub_d, sub_d)
    if bits == 2:
        pow2 = mx.array((1 << np.arange(sub_d, dtype=np.uint32)).astype(np.uint32))
        sign_bits = (encoded["signs"].astype(mx.uint32)[..., None] // pow2) % 2
        sub_hat = sub_hat * (2.0 * sign_bits.astype(mx.float32) - 1.0)

    x_hat = sub_hat.reshape(sub_hat.shape[:-2] + (d,))
    n = mx.sqrt(mx.sum(x_hat * x_hat, axis=-1, keepdims=True))
    return x_hat * (math.sqrt(d) / mx.maximum(n, _EPS))


# ----------------------------------------------------------------------
# Hadamard wrappers
# ----------------------------------------------------------------------


def hadamard_forward(x: mx.array) -> mx.array:
    """Plain (sign-free) normalized Hadamard rotation on the last axis.

    ``mx.hadamard_transform`` is normalized (self-inverse) and
    norm-preserving, which is what lets the Hadamard run *after* NSN without
    disturbing the ``s1``/``o``/``s2`` semantics: token norms are unchanged,
    so restoration in the rotated domain stays exact. Equivalent to
    ``HadamardPreconditioner`` with an all-ones diagonal — NSNQuant uses the
    paper's practical plain-Hadamard choice, not the randomized-sign variant.
    """
    d = x.shape[-1]
    if not is_hadamard_compatible(d):
        raise ValueError(
            f"nsnquant: head_dim {d} unsupported by mx.hadamard_transform "
            f"(needs d = m * 2^k, m in {{1, 12, 20, 28}})"
        )
    dtype = x.dtype
    return mx.hadamard_transform(x.astype(mx.float32)).astype(dtype)


def hadamard_inverse(y: mx.array) -> mx.array:
    """Inverse of :func:`hadamard_forward` (the transform is self-inverse)."""
    return hadamard_forward(y)


__all__ = [
    "nsn_transform",
    "nsn_inverse",
    "build_universal_codebook",
    "vq_encode",
    "vq_decode",
    "hadamard_forward",
    "hadamard_inverse",
]

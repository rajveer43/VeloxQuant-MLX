"""Unit tests for NSNQuant calibration-free universal-codebook VQ primitives.

Covers:
  - nsn_transform / nsn_inverse: exact round-trip, post-NSN statistics
  - hadamard_forward / hadamard_inverse: self-inverse on head_dim 64 and 128
  - build_universal_codebook: determinism, shape/norm, magnitude orthant
  - vq_encode / vq_decode: 2-bit and 1-bit round-trip cosine floors
  - full pipeline: NSN materially beats the same VQ without NSN on
    channel-biased input (the mechanism-validation test)
  - error paths (d % sub_d != 0) and odd shapes (T=1)
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.nsnquant import (
    build_universal_codebook,
    hadamard_forward,
    hadamard_inverse,
    nsn_inverse,
    nsn_transform,
    vq_decode,
    vq_encode,
)

# Small-but-adequate codebooks for test speed; the module-level cache makes
# repeated calls free within the test session.
_CB_KW = dict(n_samples=131_072, iters=15)


def _cb(kind: str) -> np.ndarray:
    return build_universal_codebook(kind=kind, **_CB_KW)


def _pipeline(x: mx.array, cb: np.ndarray, bits: int) -> mx.array:
    x_nsn, s1, o, s2 = nsn_transform(x)
    enc = vq_encode(hadamard_forward(x_nsn), cb, bits)
    dec = vq_decode(enc, cb)
    return nsn_inverse(hadamard_inverse(dec), s1, o, s2)


def _mean_cosine(a: mx.array, b: mx.array) -> float:
    an = np.array(a, dtype=np.float64).reshape(-1, a.shape[-1])
    bn = np.array(b, dtype=np.float64).reshape(-1, b.shape[-1])
    num = np.sum(an * bn, axis=1)
    den = np.linalg.norm(an, axis=1) * np.linalg.norm(bn, axis=1) + 1e-9
    return float(np.mean(num / den))


# ------------------------------------------------------------------
# NSN transform
# ------------------------------------------------------------------


def test_nsn_roundtrip_exact_without_vq() -> None:
    mx.random.seed(0)
    x = mx.random.normal((2, 4, 37, 128)) * 3.0 + 0.5
    x_nsn, s1, o, s2 = nsn_transform(x)
    x_hat = nsn_inverse(x_nsn, s1, o, s2)
    rel = float(mx.max(mx.abs(x_hat - x.astype(mx.float32))) / mx.max(mx.abs(x)))
    # Metadata is fp16, so the round-trip is exact to fp16 precision only.
    assert rel < 5e-3


def test_nsn_output_statistics() -> None:
    """Post-NSN tokens have norm sqrt(d) and ~zero channel mean.

    The second Normalize slightly perturbs the zero mean produced by the
    Shift; the paper notes the deviation is negligible, hence the loose
    channel-mean tolerance here.
    """
    mx.random.seed(1)
    d = 128
    x = mx.random.normal((1, 2, 256, d)) * 5.0 + 2.0
    x_nsn, _, _, _ = nsn_transform(x)
    norms = mx.sqrt(mx.sum(x_nsn * x_nsn, axis=-1))
    assert float(mx.max(mx.abs(norms - math.sqrt(d)))) < 1e-2
    chan_mean = mx.mean(x_nsn, axis=-2)
    assert float(mx.max(mx.abs(chan_mean))) < 0.15


def test_nsn_metadata_dtypes_and_shapes() -> None:
    x = mx.random.normal((1, 2, 16, 64))
    x_nsn, s1, o, s2 = nsn_transform(x)
    assert s1.dtype == mx.float16 and s2.dtype == mx.float16
    assert o.dtype == mx.float16
    assert s1.shape == (1, 2, 16, 1) and s2.shape == (1, 2, 16, 1)
    assert o.shape == (1, 2, 1, 64)
    assert x_nsn.shape == x.shape


# ------------------------------------------------------------------
# Hadamard wrappers
# ------------------------------------------------------------------


@pytest.mark.parametrize("d", [64, 128])
def test_hadamard_roundtrip(d: int) -> None:
    mx.random.seed(2)
    x = mx.random.normal((2, 3, 8, d))
    y = hadamard_forward(x)
    z = hadamard_inverse(y)
    assert float(mx.max(mx.abs(z - x))) < 1e-4
    # Norm-preserving — the property that lets Hadamard follow NSN.
    assert abs(float(mx.linalg.norm(x)) - float(mx.linalg.norm(y))) < 1e-2


def test_hadamard_rejects_incompatible_dim() -> None:
    with pytest.raises(ValueError, match="hadamard"):
        hadamard_forward(mx.random.normal((1, 2, 4, 9)))


# ------------------------------------------------------------------
# Universal codebook
# ------------------------------------------------------------------


def test_codebook_deterministic() -> None:
    a = build_universal_codebook(seed=7, n_samples=32_768, iters=5)
    build_universal_codebook.__globals__["_CODEBOOK_CACHE"].clear()
    b = build_universal_codebook(seed=7, n_samples=32_768, iters=5)
    c = build_universal_codebook(seed=8, n_samples=32_768, iters=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_codebook_shape_and_norms() -> None:
    cb_s = _cb("signed")
    cb_m = _cb("magnitude")
    assert cb_s.shape == (256, 8) and cb_m.shape == (256, 8)
    assert np.allclose(np.linalg.norm(cb_s, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(cb_m, axis=1), 1.0, atol=1e-5)
    # Magnitude codebook lives in the positive orthant (signs stored apart).
    assert (cb_m >= 0.0).all()


def test_codebook_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        build_universal_codebook(kind="bogus")


# ------------------------------------------------------------------
# VQ round-trips (floors calibrated empirically once, then pinned)
# ------------------------------------------------------------------


def test_vq_2bit_gaussian_cosine_floor() -> None:
    mx.random.seed(3)
    x = mx.random.normal((1, 2, 256, 128))
    assert _mean_cosine(_pipeline(x, _cb("magnitude"), 2), x) > 0.92


def test_vq_1bit_gaussian_cosine_floor() -> None:
    mx.random.seed(4)
    x = mx.random.normal((1, 2, 256, 128))
    assert _mean_cosine(_pipeline(x, _cb("signed"), 1), x) > 0.78


def test_vq_2bit_beats_1bit() -> None:
    mx.random.seed(5)
    x = mx.random.normal((1, 2, 256, 128))
    c2 = _mean_cosine(_pipeline(x, _cb("magnitude"), 2), x)
    c1 = _mean_cosine(_pipeline(x, _cb("signed"), 1), x)
    assert c2 > c1


def test_nsn_beats_no_nsn_on_channel_biased_input() -> None:
    """Mechanism validation: on input with a strong channel-wise bias (the
    distribution shape NSN's Shift step exists for), the full NSN pipeline
    must materially beat the identical VQ with token-norm scaling only."""
    rng = np.random.default_rng(0)
    d = 128
    bias = (rng.standard_normal((1, 1, 1, d)) * 4.0).astype(np.float32)
    base = rng.standard_normal((1, 2, 256, d)).astype(np.float32)
    base[..., :4] *= 15.0  # outlier channels
    tok_scale = np.exp(rng.standard_normal((1, 2, 256, 1)) * 0.8).astype(np.float32)
    x = mx.array(base * tok_scale + bias)

    cb = _cb("magnitude")
    with_nsn = _mean_cosine(_pipeline(x, cb, 2), x)

    # Ablation: same Hadamard + VQ, but token-norm scaling instead of NSN.
    h = hadamard_forward(x.astype(mx.float32))
    n = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True))
    hn = h * (math.sqrt(d) / mx.maximum(n, 1e-8))
    dec = vq_decode(vq_encode(hn, cb, 2), cb)
    no_nsn = _mean_cosine(hadamard_inverse(dec * (n / math.sqrt(d))), x)

    assert with_nsn > no_nsn + 0.02


def test_vq_rejects_indivisible_dim() -> None:
    cb = _cb("signed")
    with pytest.raises(ValueError, match="divisible"):
        vq_encode(mx.random.normal((1, 2, 4, 12)), cb, 1)


def test_vq_rejects_bad_bits() -> None:
    cb = _cb("signed")
    with pytest.raises(ValueError, match="bits"):
        vq_encode(mx.random.normal((1, 2, 4, 64)), cb, 3)


def test_full_pipeline_single_token() -> None:
    """T=1 (single decode token) works through the whole pipeline."""
    mx.random.seed(6)
    x = mx.random.normal((1, 2, 1, 128))
    out = _pipeline(x, _cb("magnitude"), 2)
    assert out.shape == x.shape
    # A single token is fully described by (s1, o, s2) + its unit direction,
    # so reconstruction is near-exact regardless of the codebook.
    assert _mean_cosine(out, x) > 0.999

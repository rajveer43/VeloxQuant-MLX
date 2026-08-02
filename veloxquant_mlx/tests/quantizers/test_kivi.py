"""Tests for KIVIQuantizer — asymmetric group quantization (arXiv:2402.02750).

KIVI is deterministic (min/max group quant, no codebook training, no RNG),
so every reconstruction-quality assertion below is exact run-to-run.  All
synthetic data is seeded for reproducibility regardless.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.core.registry import QuantizerRegistry
from veloxquant_mlx.quantizers.kivi import KIVIQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Registry / construction
# ---------------------------------------------------------------------------


def test_registered() -> None:
    assert QuantizerRegistry.is_registered("kivi")
    assert QuantizerRegistry.get("kivi") is KIVIQuantizer


def test_rejects_bad_bits() -> None:
    with pytest.raises(QuantizerConfigError):
        KIVIQuantizer(d=128, b=0)
    with pytest.raises(QuantizerConfigError):
        KIVIQuantizer(d=128, b=9)


def test_rejects_bad_axis() -> None:
    with pytest.raises(QuantizerConfigError):
        KIVIQuantizer(d=128, b=2, axis="diagonal")


# ---------------------------------------------------------------------------
# Shape / dtype preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["channel", "token"])
def test_encode_decode_shape_dtype(axis: str) -> None:
    X = mx.array(_unit_gaussian(200, 128))
    q = KIVIQuantizer(d=128, b=4, group_size=32, axis=axis)
    ev = q.encode(X)
    assert ev.quantizer_type == "kivi"
    assert ev.batch_size == 200 and ev.dim == 128
    assert ev.indices.dtype == mx.uint8
    out = q.decode(ev)
    assert out.shape == (200, 128)
    assert out.dtype == mx.float16


def test_encode_wrong_dim_raises() -> None:
    q = KIVIQuantizer(d=128, b=2)
    with pytest.raises(QuantizerConfigError):
        q.encode(mx.array(_unit_gaussian(10, 64)))


# ---------------------------------------------------------------------------
# Reconstruction quality (deterministic; tolerances justified inline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "b, min_cos",
    # KIVI per-channel group quant on unit-norm Gaussian keys.  These
    # thresholds are conservative lower bounds measured on this synthetic
    # distribution; higher b ⇒ higher fidelity.  b=2 is genuinely lossy
    # (this is why KIVI keeps an fp16 residual window in the cache).
    [(2, 0.90), (3, 0.96), (4, 0.99), (8, 0.999)],
)
def test_channel_reconstruction_cosine(b: int, min_cos: float) -> None:
    X = _unit_gaussian(512, 128, seed=b)
    q = KIVIQuantizer(d=128, b=b, group_size=32, axis="channel")
    Xhat = np.array(q.decode(q.encode(mx.array(X)))).astype(np.float32)
    cos = _cosine(X, Xhat)
    assert cos >= min_cos, f"b={b}: cosine {cos:.4f} < {min_cos}"


def test_monotone_quality_in_bits() -> None:
    """More bits must not reduce reconstruction quality."""
    X = _unit_gaussian(512, 128, seed=7)
    prev = -1.0
    for b in (2, 3, 4, 6, 8):
        q = KIVIQuantizer(d=128, b=b, group_size=32, axis="channel")
        Xhat = np.array(q.decode(q.encode(mx.array(X)))).astype(np.float32)
        cos = _cosine(X, Xhat)
        assert cos >= prev - 1e-4, f"quality dropped at b={b}: {cos} < {prev}"
        prev = cos


def test_high_bit_near_lossless() -> None:
    """At b=8 the max group-quant error is bounded by range/255 per group;
    on unit-norm data the reconstruction MSE must be tiny."""
    X = _unit_gaussian(256, 128, seed=1)
    q = KIVIQuantizer(d=128, b=8, group_size=32, axis="channel")
    Xhat = np.array(q.decode(q.encode(mx.array(X)))).astype(np.float32)
    mse = float(np.mean((X - Xhat) ** 2))
    assert mse < 1e-4, f"b=8 MSE {mse:.2e} too high"


def test_determinism() -> None:
    """Same input ⇒ bit-identical codes on repeat (no RNG anywhere)."""
    X = mx.array(_unit_gaussian(128, 128, seed=3))
    q = KIVIQuantizer(d=128, b=3, group_size=32, axis="channel")
    a = np.array(q.encode(X).indices)
    b = np.array(q.encode(X).indices)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Asymmetric scheme: per-token (value) path
# ---------------------------------------------------------------------------


def test_token_axis_roundtrip() -> None:
    X = _unit_gaussian(300, 128, seed=5)
    q = KIVIQuantizer(d=128, b=4, group_size=32, axis="token")
    Xhat = np.array(q.decode(q.encode(mx.array(X)))).astype(np.float32)
    assert _cosine(X, Xhat) >= 0.99


# ---------------------------------------------------------------------------
# Ragged final group (n not divisible by group_size)
# ---------------------------------------------------------------------------


def test_ragged_group_handled() -> None:
    # 100 tokens, group_size 32 → groups of 32,32,32,4 (padded internally)
    X = _unit_gaussian(100, 64, seed=9)
    q = KIVIQuantizer(d=64, b=4, group_size=32, axis="channel")
    out = q.decode(q.encode(mx.array(X)))
    assert out.shape == (100, 64)
    assert _cosine(X, np.array(out).astype(np.float32)) >= 0.99


# ---------------------------------------------------------------------------
# Inner-product estimate
# ---------------------------------------------------------------------------


def test_inner_product_tracks_exact() -> None:
    X = _unit_gaussian(256, 128, seed=2)
    q = KIVIQuantizer(d=128, b=8, group_size=32, axis="channel")
    ev = q.encode(mx.array(X))
    rng = np.random.default_rng(11)
    qv = rng.standard_normal(128).astype(np.float32)
    est = np.array(q.estimate_inner_product(mx.array(qv), ev)).astype(np.float32)
    exact = X @ qv
    # b=8 ⇒ estimate should track exact dot products closely.
    corr = float(np.corrcoef(est, exact)[0, 1])
    assert corr >= 0.999, f"IP correlation {corr:.5f} too low"

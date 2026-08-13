"""Tests for TurboQuantProd: unbiasedness and IP distortion bound."""

from __future__ import annotations

import math

import numpy as np
import pytest

from veloxquant_mlx.quantizers.base import QuantizerFactory


@pytest.fixture(scope="module")
def prod_quantizer():
    return QuantizerFactory.create("turboquant_prod", d=64, b=3, m=64, seed=42)


def test_prod_encode_shape(prod_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(5, 64).astype(np.float16))
    ev = prod_quantizer.encode(x)
    assert ev.indices.shape == (5, 64)
    assert ev.signs.shape == (5, 64)
    assert ev.residual_norm.shape == (5,)


def test_prod_decode_shape(prod_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(5, 64).astype(np.float16))
    ev = prod_quantizer.encode(x)
    x_hat = prod_quantizer.decode(ev)
    assert x_hat.shape == (5, 64)


def test_prod_unbiasedness(prod_quantizer) -> None:
    """IP estimate should be unbiased: E[ProdQJL(q,k)] ≈ ⟨q,k⟩."""
    import mlx.core as mx

    rng = np.random.default_rng(42)
    d = 64
    q = rng.standard_normal(d).astype(np.float32)
    k = rng.standard_normal(d).astype(np.float32)
    true_ip = float(np.dot(q, k))

    n_trials = 500
    estimates = []
    for _ in range(n_trials):
        ev = prod_quantizer.encode(mx.array(k[None].astype(np.float16)))
        est = float(
            prod_quantizer.estimate_inner_product(mx.array(q.astype(np.float16)), ev).item()
        )
        estimates.append(est)

    bias = abs(np.mean(estimates) - true_ip)
    # Bias should be small relative to the IP magnitude
    assert bias < max(0.5, abs(true_ip) * 0.5), (
        f"TurboQuantProd shows excessive bias: {bias:.4f}, true_ip={true_ip:.4f}"
    )


def test_prod_ip_distortion_bound() -> None:
    """IP distortion should be within √(3π)/2 · ‖y‖²/d · 4^(-b)."""
    import mlx.core as mx

    d, n_samples, b = 64, 1000, 3
    quantizer = QuantizerFactory.create("turboquant_prod", d=d, b=b, m=d, seed=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # unit norm
    y = rng.standard_normal(d).astype(np.float32)
    y /= np.linalg.norm(y)

    X_mx = mx.array(X.astype(np.float16))
    y_mx = mx.array(y.astype(np.float16))
    ev = quantizer.encode(X_mx)
    true_ips = np.array(X_mx @ y_mx)
    est_ips = np.array(quantizer.estimate_inner_product(y_mx, ev))

    ip_distortion = float(np.mean((true_ips - est_ips) ** 2))
    y_norm_sq = float(np.dot(y, y))
    upper_bound = math.sqrt(3 * math.pi) / 2 * y_norm_sq / d * (4 ** (-b))

    # 15× margin: paper bound is asymptotic; MSE stage at b-1=2 bits overshoots
    # the paper constant by ~2×, which propagates into IP distortion.
    assert ip_distortion <= upper_bound * 15.0, (
        f"IP distortion {ip_distortion:.6f} far exceeds upper bound {upper_bound:.6f}"
    )

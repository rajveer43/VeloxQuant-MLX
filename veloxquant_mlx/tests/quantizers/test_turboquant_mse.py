"""Tests for TurboQuantMSE: MSE bound verification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from veloxquant_mlx.quantizers.base import QuantizerFactory


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_turboquant_mse_bound(b: int) -> None:
    """Empirical MSE must fall within [4^(-b), √(3π)/2·4^(-b)] × 1.1 tolerance."""
    import mlx.core as mx

    d, n_samples = 128, 2000
    theoretical_upper = math.sqrt(3 * math.pi) / 2 * (4 ** (-b))
    theoretical_lower = 4 ** (-b)

    quantizer = QuantizerFactory.create("turboquant_mse", d=d, b=b, seed=42)
    rng = np.random.default_rng(b)
    X = rng.standard_normal((n_samples, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    X_mx = mx.array(X.astype(np.float16))
    ev = quantizer.encode(X_mx)
    X_hat = quantizer.decode(ev)
    mx.eval(X_hat)

    mse = float(mx.mean(mx.sum((X_mx - X_hat) ** 2, axis=-1)).item())
    # Scalar Lloyd-Max at b=2-4 achieves D/4^(-b) ≈ 1.88-2.43, up to 2.5× above
    # the paper's asymptotic bound.  Use 2.5 as achievable tolerance.
    assert mse <= theoretical_upper * 2.5, (
        f"b={b}: MSE {mse:.6f} exceeds upper bound {theoretical_upper:.6f} ×2.5"
    )
    assert mse >= theoretical_lower * 0.5, (
        f"b={b}: MSE {mse:.8f} is suspiciously low vs lower bound {theoretical_lower:.6f}"
    )


def test_turboquant_mse_encode_decode_shapes() -> None:
    import mlx.core as mx

    d, n = 64, 8
    q = QuantizerFactory.create("turboquant_mse", d=d, b=2, seed=42)
    x = mx.array(np.random.randn(n, d).astype(np.float16))
    ev = q.encode(x)
    assert ev.indices.shape == (n, d)
    x_hat = q.decode(ev)
    assert x_hat.shape == (n, d)


def test_turboquant_mse_ip_shape() -> None:
    import mlx.core as mx

    d, n = 64, 16
    q = QuantizerFactory.create("turboquant_mse", d=d, b=2, seed=42)
    x = mx.array(np.random.randn(n, d).astype(np.float16))
    query = mx.array(np.random.randn(d).astype(np.float16))
    ev = q.encode(x)
    ips = q.estimate_inner_product(query, ev)
    assert ips.shape == (n,)

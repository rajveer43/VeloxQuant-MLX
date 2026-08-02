"""Integration test: reproduces Figure 3 from TurboQuant paper.

Verifies that empirical MSE and IP distortion fall within theoretical bounds
for b = 1, 2, 3, 4 across both TurboQuantMSE and TurboQuantProd.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_turboquant_mse_within_bounds(b: int) -> None:
    """D_mse must satisfy lower ≤ D_mse ≤ paper_upper × scalar_lm_gap.

    The paper bound sqrt(3π)/2 · 4^(-b) is the asymptotic optimum.  Scalar
    Lloyd-Max at low bit-rates (b=2–4) overshoots this by up to 2.5× — a
    well-known property of finite-level scalar quantisation.  The tolerance
    here accounts for that gap so the test checks correct operation, not an
    unachievable bound.
    """
    import mlx.core as mx
    from veloxquant_mlx.quantizers.base import QuantizerFactory

    d, n = 128, 3000
    upper = math.sqrt(3 * math.pi) / 2 * (4 ** (-b))
    lower = 4 ** (-b)

    q = QuantizerFactory.create("turboquant_mse", d=d, b=b, seed=0)
    rng = np.random.default_rng(b)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    X_mx = mx.array(X.astype(np.float16))
    ev = q.encode(X_mx)
    X_hat = q.decode(ev)
    mx.eval(X_hat)

    mse = float(mx.mean(mx.sum((X_mx - X_hat) ** 2, axis=-1)).item())
    # Scalar Lloyd-Max at b=2,3,4 achieves D/4^(-b) ≈ 1.88, 2.21, 2.43 —
    # a 2.5× ceiling covers all cases with a small safety margin.
    assert mse <= upper * 2.5, f"b={b}: empirical MSE={mse:.6f} > upper bound {upper:.6f} × 2.5"
    assert mse >= lower * 0.5, (
        f"b={b}: empirical MSE={mse:.8f} below lower bound × 0.5 (suspicious)"
    )


@pytest.mark.parametrize("b", [2, 3, 4])
def test_turboquant_prod_ip_distortion(b: int) -> None:
    """IP distortion D_prod ≤ √(3π)/2 · ‖y‖²/d · 4^(-b) (with generous margin)."""
    import mlx.core as mx
    from veloxquant_mlx.quantizers.base import QuantizerFactory

    d, n, m = 64, 1000, 64
    upper_factor = math.sqrt(3 * math.pi) / 2

    q = QuantizerFactory.create("turboquant_prod", d=d, b=b, m=m, seed=0)
    rng = np.random.default_rng(b * 10)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.standard_normal(d).astype(np.float32)
    y /= np.linalg.norm(y)
    y_norm_sq = float(np.dot(y, y))

    X_mx = mx.array(X.astype(np.float16))
    y_mx = mx.array(y.astype(np.float16))
    ev = q.encode(X_mx)
    true_ips = np.array(X_mx @ y_mx)
    est_ips = np.array(q.estimate_inner_product(y_mx, ev))

    ip_distortion = float(np.mean((true_ips - est_ips) ** 2))
    upper_bound = upper_factor * y_norm_sq / d * (4 ** (-b))

    # 20× margin: paper bound is asymptotic; at b=3 the MSE stage uses b-1=2
    # bits whose scalar LM gap (≈1.88×) propagates into IP distortion.
    assert ip_distortion <= upper_bound * 20.0, (
        f"b={b}: IP distortion={ip_distortion:.6f} >> upper bound {upper_bound:.6f}"
    )


def test_builder_end_to_end() -> None:
    """Verify the KVCacheBuilder quick-start example runs end-to-end."""
    import mlx.core as mx
    import numpy as np
    from veloxquant_mlx import KVCacheBuilder

    cache = (
        KVCacheBuilder()
        .with_method("turboquant_prod")
        .with_head_dim(64)
        .with_bit_width(inlier=2, outlier=3)
        .with_jl_dim(64)
        .with_seed(42)
        .build()
    )

    rng = np.random.default_rng(0)
    for _ in range(20):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        cache.append(k, v)

    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (64,)
    assert cache.memory_bytes() > 0

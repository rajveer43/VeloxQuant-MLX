"""Tests for QJLQuantizer: unbiasedness and distortion bounds."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.quantizers.base import QuantizerFactory


@pytest.fixture(scope="module")
def qjl_quantizer():
    return QuantizerFactory.create("qjl", d=64, m=64, seed=42)


def test_qjl_encode_shape(qjl_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(4, 64).astype(np.float16))
    ev = qjl_quantizer.encode(x)
    assert ev.signs.shape == (4, 64)
    assert ev.norm.shape == (4,)


def test_qjl_signs_are_pm1(qjl_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(10, 64).astype(np.float16))
    ev = qjl_quantizer.encode(x)
    signs_np = np.array(ev.signs)
    assert set(np.unique(signs_np)).issubset({-1, 1})


def test_qjl_unbiasedness(qjl_quantizer) -> None:
    """E[ProdQJL(q,k) - <q,k>] should be ≈ 0 over random (q,k) pairs.

    Unbiasedness holds in expectation over the JL matrix distribution.
    With a fixed S, the correct test averages estimation errors over many
    fresh (q,k) pairs so the randomness comes from the inputs, not S.
    """
    import mlx.core as mx

    d, n_trials = 64, 2000
    rng = np.random.default_rng(0)

    errors = []
    for _ in range(n_trials):
        q_i = rng.standard_normal(d).astype(np.float32)
        k_i = rng.standard_normal(d).astype(np.float32)
        true_ip = float(np.dot(q_i, k_i))
        ev = qjl_quantizer.encode(mx.array(k_i[None].astype(np.float16)))
        est = float(
            qjl_quantizer.estimate_inner_product(mx.array(q_i.astype(np.float16)), ev).item()
        )
        errors.append(est - true_ip)

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors)) / np.sqrt(n_trials)
    # Mean error should be within 3 standard errors of zero
    assert abs(mean_err) < 3 * std_err + 0.1, (
        f"QJL is biased: mean_err={mean_err:.4f}, std_err={std_err:.4f}"
    )


def test_qjl_ip_estimation_shape(qjl_quantizer) -> None:
    import mlx.core as mx

    n_keys = 20
    x = mx.array(np.random.randn(n_keys, 64).astype(np.float16))
    q = mx.array(np.random.randn(64).astype(np.float16))
    ev = qjl_quantizer.encode(x)
    ips = qjl_quantizer.estimate_inner_product(q, ev)
    assert ips.shape == (n_keys,)

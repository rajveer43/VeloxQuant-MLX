"""Tests for PolarQuantizer: reconstruction error and angle distribution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from veloxquant_mlx.quantizers.base import QuantizerFactory


@pytest.fixture(scope="module")
def polar_quantizer():
    return QuantizerFactory.create("polar", d=64, b=2, seed=42)


def test_polar_encode_shape(polar_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(4, 64).astype(np.float16))
    ev = polar_quantizer.encode(x)
    assert ev.angles is not None
    assert len(ev.angles) == 4  # n_levels
    assert ev.final_radius is not None


def test_polar_decode_shape(polar_quantizer) -> None:
    import mlx.core as mx

    x = mx.array(np.random.randn(4, 64).astype(np.float16))
    ev = polar_quantizer.encode(x)
    x_hat = polar_quantizer.decode(ev)
    assert x_hat.shape == (4, 64)


def test_polar_reconstruction_error() -> None:
    """Reconstruction error should be below a reasonable threshold at b=3."""
    import mlx.core as mx

    d, n_samples, b = 64, 500, 3
    quantizer = QuantizerFactory.create("polar", d=d, b=b, seed=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    X_mx = mx.array(X.astype(np.float16))
    ev = quantizer.encode(X_mx)
    X_hat = quantizer.decode(ev)
    mx.eval(X_hat)

    mse = float(mx.mean(mx.sum((X_mx - X_hat) ** 2, axis=-1)).item())
    # PolarQuant at b=3 should be reasonably accurate. With the level-1
    # angle range fixed (#69: forward() now folds atan2's (-pi, pi] output
    # into the [0, 2*pi) the codebook assumes), empirical MSE is ~0.055;
    # allow headroom to 0.2. Before the fix this was ~1.28 -- if this
    # threshold ever needs loosening back toward that range, the level-1
    # angle convention is broken again.
    assert mse < 0.2, f"PolarQuant b=3 MSE={mse:.4f} is too large"


def test_polar_ip_estimation(polar_quantizer) -> None:
    """IP estimation should be closer to truth than random."""
    import mlx.core as mx

    d = 64
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, d)).astype(np.float32)
    q = rng.standard_normal(d).astype(np.float32)

    X_mx = mx.array(X.astype(np.float16))
    q_mx = mx.array(q.astype(np.float16))
    ev = polar_quantizer.encode(X_mx)
    true_ips = np.array(X_mx @ q_mx)
    est_ips = np.array(polar_quantizer.estimate_inner_product(q_mx, ev))

    corr = float(np.corrcoef(true_ips, est_ips)[0, 1])
    # With the level-1 angle range fixed (#69), correlation at b=2 is ~0.94.
    # Before the fix this was only ~0.3-0.4 (level-1 angles quantized against
    # a codebook covering the wrong range).
    assert corr > 0.8, f"PolarQuant IP correlation {corr:.3f} is too low"


class TestLevel1AngleRange:
    """Regression for #69: level-1 angles must land in [0, 2*pi), matching
    the range polar_angle_pdf/PolarAngleSamplingStrategy assume when
    building the level-1 codebook -- atan2's native range is (-pi, pi]."""

    def test_level1_angles_are_never_negative(self) -> None:
        import mlx.core as mx

        from veloxquant_mlx.transforms.polar import RecursivePolarTransform

        d = 32
        rng = np.random.default_rng(2)
        x = mx.array(rng.standard_normal((300, d)).astype(np.float32).astype(np.float16))

        transform = RecursivePolarTransform(n_levels=2)
        result = transform.forward(x)
        level1_angles = np.array(result.angles[0])

        assert (level1_angles >= 0.0).all(), (
            f"level-1 angles must be in [0, 2*pi), found negative values: min={level1_angles.min()}"
        )
        # fp16 storage of values near true 2*pi can round to/past fp16's
        # nearest representable 2*pi (~6.28125) -- allow a small tolerance
        # rather than asserting a mathematically exact open interval.
        assert (level1_angles < 2 * math.pi + 1e-2).all(), level1_angles.max()

    def test_level1_angles_actually_span_full_circle(self) -> None:
        """A zero-mean rotated input should produce angles spread across
        the whole [0, 2*pi) range, not clustered in (0, pi] -- otherwise
        the fold could be silently mapping negative angles to a narrow band
        instead of the correct symmetric range."""
        import mlx.core as mx

        from veloxquant_mlx.transforms.polar import RecursivePolarTransform

        d = 32
        rng = np.random.default_rng(3)
        x = mx.array(rng.standard_normal((2000, d)).astype(np.float32).astype(np.float16))

        transform = RecursivePolarTransform(n_levels=1)
        result = transform.forward(x)
        level1_angles = np.array(result.angles[0]).reshape(-1)

        # Roughly uniform on [0, 2*pi): a healthy fraction should exceed pi.
        frac_above_pi = float((level1_angles > math.pi).mean())
        assert 0.3 < frac_above_pi < 0.7, (
            f"expected ~half of level-1 angles above pi, got {frac_above_pi:.2f} "
            f"-- level-1 angles may not be using the full [0, 2*pi) range"
        )

    def test_level1_codebook_range_matches_transform_output_range(self) -> None:
        """The level-1 codebook's centroid range and the actual angles
        produced by forward() must live in the same convention."""
        import mlx.core as mx

        from veloxquant_mlx.codebooks.base import CodebookFactory
        from veloxquant_mlx.transforms.polar import RecursivePolarTransform

        d = 32
        rng = np.random.default_rng(4)
        x = mx.array(rng.standard_normal((500, d)).astype(np.float32).astype(np.float16))

        transform = RecursivePolarTransform(n_levels=1)
        result = transform.forward(x)
        level1_angles = np.array(result.angles[0]).reshape(-1)

        cb = CodebookFactory.create("polar_level", b=4, d=d, polar_level=1)
        centroids = np.asarray(cb.centroids_numpy())

        assert centroids.min() >= 0.0
        assert centroids.max() < 2 * math.pi

        idx = cb.quantize(result.angles[0])
        dequant = np.array(cb.dequantize(idx)).reshape(-1)
        mean_angle_err = np.mean(np.abs(level1_angles - dequant))
        # At b=4 (16 centroids) over a 2*pi range, a well-matched codebook
        # should give error well under 2*pi/16 ~= 0.39. Before the fix this
        # was ~0.9 (angles quantized against a mismatched-range codebook).
        assert mean_angle_err < 0.3, f"mean level-1 angle quant error {mean_angle_err:.4f}"

    def test_reconstruction_error_decreases_with_bit_width(self) -> None:
        """Error should shrink monotonically as bit-width grows -- the kind
        of quantitative check that would have caught #69 immediately."""
        import mlx.core as mx

        d = 32
        rng = np.random.default_rng(5)
        x = mx.array(rng.standard_normal((300, d)).astype(np.float32).astype(np.float16))

        errors = []
        for b in (2, 4, 6, 8):
            q = QuantizerFactory.create("polar", d=d, b=b, n_levels=2, seed=0)
            ev = q.encode(x)
            x_hat = q.decode(ev)
            err = float(mx.mean(mx.abs(x.astype(mx.float32) - x_hat.astype(mx.float32))))
            errors.append(err)

        assert errors == sorted(errors, reverse=True), f"errors not decreasing: {errors}"
        # At b=4, error should be small relative to signal magnitude (~0.78
        # for unit-Gaussian coordinates), not comparable to it.
        assert errors[1] < 0.2, f"b=4 mean abs error {errors[1]:.4f} too large"

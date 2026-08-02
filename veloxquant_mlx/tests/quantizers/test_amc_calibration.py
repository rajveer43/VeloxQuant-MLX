"""Tests for AMC's offline calibration — variance-ordered channel permutation.

amc_calibrate_channel_order (Algorithm 1 Phase I) must rank the original
D hidden-dim channel indices by descending empirical variance so that AMC's
rank-masking (Eq. 6, veloxquant_mlx.quantizers.amc.amc_apply_rank_mask) is
safe to apply to raw index prefixes. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.amc_calibration import (
    amc_calibrate_channel_order,
    amc_permute_weights,
)


def _calib_matrix(n: int, d: int, variances: list, seed: int = 0) -> mx.array:
    """Build an [n, d] matrix where column j has approximately `variances[j]`."""
    rng = np.random.default_rng(seed)
    cols = [rng.standard_normal(n) * np.sqrt(v) for v in variances]
    return mx.array(np.stack(cols, axis=1).astype(np.float32))


# ---------------------------------------------------------------------------
# Core ordering behaviour
# ---------------------------------------------------------------------------


def test_orders_channels_by_descending_variance() -> None:
    # Column 2 has the highest variance, then 0, then 3, then 1 (lowest).
    variances = [4.0, 0.01, 25.0, 1.0]
    x = _calib_matrix(500, 4, variances, seed=1)
    perm = amc_calibrate_channel_order(x)
    order = perm.tolist()
    assert order[0] == 2  # highest variance first
    assert order[-1] == 1  # lowest variance last


def test_permutation_is_a_valid_permutation() -> None:
    x = _calib_matrix(200, 8, [float(i + 1) for i in range(8)], seed=2)
    perm = amc_calibrate_channel_order(x)
    assert sorted(perm.tolist()) == list(range(8))


def test_permuted_columns_have_descending_variance() -> None:
    variances = [0.1, 9.0, 3.0, 0.5, 16.0]
    x = _calib_matrix(500, 5, variances, seed=3)
    perm = amc_calibrate_channel_order(x)
    x_np = np.array(x)
    permuted = x_np[:, perm.tolist()]
    col_vars = permuted.var(axis=0)
    # Non-increasing (allow small numerical slack between adjacent columns)
    for i in range(len(col_vars) - 1):
        assert col_vars[i] >= col_vars[i + 1] - 1e-3


# ---------------------------------------------------------------------------
# Weight permutation
# ---------------------------------------------------------------------------


def test_permute_weights_last_axis() -> None:
    w = mx.array(np.arange(12).reshape(3, 4).astype(np.float32))
    perm = mx.array([3, 1, 0, 2], dtype=mx.int32)
    out = amc_permute_weights(w, perm, axis=-1)
    expected = np.array([[3, 1, 0, 2], [7, 5, 4, 6], [11, 9, 8, 10]], dtype=np.float32)
    assert np.allclose(np.array(out), expected)


def test_permute_weights_roundtrip_identity() -> None:
    w = mx.array(np.random.default_rng(4).standard_normal((5, 6)).astype(np.float32))
    perm = mx.array(list(range(6)), dtype=mx.int32)
    out = amc_permute_weights(w, perm, axis=-1)
    assert np.allclose(np.array(out), np.array(w))


# ---------------------------------------------------------------------------
# Edge cases / guards
# ---------------------------------------------------------------------------


def test_rejects_non_2d_input() -> None:
    x = mx.array(np.random.default_rng(5).standard_normal((10,)).astype(np.float32))
    with pytest.raises(ValueError):
        amc_calibrate_channel_order(x)


def test_rejects_too_few_calib_rows() -> None:
    x = mx.array(np.random.default_rng(6).standard_normal((1, 4)).astype(np.float32))
    with pytest.raises(ValueError):
        amc_calibrate_channel_order(x)


def test_deterministic() -> None:
    x = _calib_matrix(300, 6, [1.0, 5.0, 2.0, 8.0, 0.5, 3.0], seed=7)
    p1 = amc_calibrate_channel_order(x)
    p2 = amc_calibrate_channel_order(x)
    assert p1.tolist() == p2.tolist()


def test_all_equal_variance_no_crash() -> None:
    """Degenerate case: all channels have identical variance — must not crash."""
    rng = np.random.default_rng(8)
    x = mx.array(rng.standard_normal((100, 4)).astype(np.float32))
    perm = amc_calibrate_channel_order(x)
    assert sorted(perm.tolist()) == list(range(4))

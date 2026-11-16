"""#970: unusable smooth factors must be rejected at construction.

The Metal key kernel skips the smooth divide for ``s <= 1e-8`` but always
multiplies by ``s`` on the way out, so such a channel came back as exactly 0
(and the pure-MLX path gave inf/NaN). The cache now validates once in __init__.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.allocators.vecinfer import calibrate_smooth_factors
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
from veloxquant_mlx.core.exceptions import QuantizerConfigError

D = 16


def _build(smooth):
    cfg = SimpleNamespace(
        head_dim=D,
        key_sub_dim=4,
        value_sub_dim=4,
        key_codebook_bits=5,
        value_codebook_bits=5,
        smooth_factors=smooth,
        use_metal_kernels=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VecInferKVCache(cfg)


def _with(value, shape=(D,)):
    arr = np.ones(shape, dtype=np.float32)
    arr.reshape(-1)[3] = value
    return mx.array(arr)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1e-9, 1e-8, math.nan, math.inf, -math.inf])
def test_rejects_unusable_factor_1d(bad):
    with pytest.raises(QuantizerConfigError, match="smooth_factors"):
        _build(_with(bad))


def test_rejects_unusable_factor_2d_and_reports_count():
    arr = np.ones((4, D), dtype=np.float32)
    arr[1, 2] = 0.0
    arr[3, 5] = -1.0
    with pytest.raises(QuantizerConfigError, match="2 of 64"):
        _build(mx.array(arr))


def test_rejects_numpy_input():
    arr = np.ones(D, dtype=np.float32)
    arr[0] = 0.0
    with pytest.raises(QuantizerConfigError):
        _build(arr)


def test_rejects_float16_smooth():
    arr = mx.ones((D,), dtype=mx.float16)
    arr[3] = 0.0
    with pytest.raises(QuantizerConfigError):
        _build(arr)


@pytest.mark.parametrize("ok", [0.01, 1e-4, 1.0, 50.0])
def test_accepts_positive_factors(ok):
    c = _build(_with(ok))
    assert c._smooth is not None


def test_none_smooth_is_still_allowed():
    assert _build(None)._smooth is None


def test_calibrated_factors_are_accepted():
    mx.random.seed(0)
    keys = mx.random.normal((4, 64, D))
    # a dead channel exercises calibrate's floor
    keys = mx.concatenate([keys[..., :3], mx.zeros((4, 64, 1)), keys[..., 4:]], axis=-1)
    c = _build(calibrate_smooth_factors(keys))
    assert c._smooth is not None

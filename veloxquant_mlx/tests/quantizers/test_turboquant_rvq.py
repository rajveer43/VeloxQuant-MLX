"""Tests for TurboQuantRVQ's codebook memoization (VeloxQuant-MLX#508).

Stage-1's codebook already went through CodebookFactory (now memoized).
Stage-2's Laplacian-fit codebook was previously rebuilt inline on every
TurboQuantRVQ() construction -- the dominant cost of per-layer/per-request
quantizer construction, since it's a pure function of (b, residual_scale)
with no seed dependency, unlike the per-layer Hadamard rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ, _stage2_codebook


@pytest.fixture(autouse=True)
def _clear_stage2_cache():
    _stage2_codebook.cache_clear()
    yield
    _stage2_codebook.cache_clear()


def test_stage2_codebook_shared_across_layers_with_different_seeds():
    """Layers only differ by seed (config.seed + i); stage-2 must be shared."""
    q1 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    q2 = TurboQuantRVQ(d=128, b=2, seed=43, use_hadamard=True)
    assert q1._codebook2 is q2._codebook2


def test_stage2_codebook_differs_by_bits():
    q1 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    q2 = TurboQuantRVQ(d=128, b=3, seed=42, use_hadamard=True)
    assert q1._codebook2 is not q2._codebook2


def test_stage2_codebook_differs_by_dim():
    q1 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    q2 = TurboQuantRVQ(d=64, b=2, seed=42, use_hadamard=True)
    assert q1._codebook2 is not q2._codebook2


def test_rotation_still_differs_per_layer_despite_shared_codebooks():
    """Memoizing codebooks must not accidentally share the seed-dependent rotation."""
    import mlx.core as mx

    q1 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    q2 = TurboQuantRVQ(d=128, b=2, seed=43, use_hadamard=True)
    assert q1._codebook2 is q2._codebook2
    assert q1._rotation is not q2._rotation

    x = mx.random.normal((4, 128)).astype(mx.float16)
    y1 = np.array(q1._rotation.apply(x))
    y2 = np.array(q2._rotation.apply(x))
    assert not np.array_equal(y1, y2)


def test_cached_stage2_matches_uncached_centroids():
    q_cached = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    _stage2_codebook.cache_clear()
    q_fresh = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True)
    np.testing.assert_array_equal(
        q_cached._codebook2.centroids_numpy(), q_fresh._codebook2.centroids_numpy()
    )


def test_explicit_residual_scale_bypasses_default_derivation_but_still_caches():
    q1 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True, residual_scale=0.05)
    q2 = TurboQuantRVQ(d=128, b=2, seed=99, use_hadamard=True, residual_scale=0.05)
    assert q1._codebook2 is q2._codebook2

    q3 = TurboQuantRVQ(d=128, b=2, seed=42, use_hadamard=True, residual_scale=0.1)
    assert q1._codebook2 is not q3._codebook2

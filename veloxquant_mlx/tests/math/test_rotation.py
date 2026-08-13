"""Tests for is_hadamard_compatible, including the bare-m regression for #67."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.math.rotation import is_hadamard_compatible


class TestBareMValuesAreRejected:
    """Regression for #67: d exactly equal to a non-trivial m in
    {12, 20, 28} (i.e. k=0 doublings) crashes mx.hadamard_transform during
    Metal shader compilation and must be reported as incompatible."""

    @pytest.mark.parametrize("d", [12, 20, 28])
    def test_bare_m_is_not_compatible(self, d: int) -> None:
        assert not is_hadamard_compatible(d)

    @pytest.mark.parametrize("d", [12, 20, 28])
    def test_bare_m_actually_crashes_mlx(self, d: int) -> None:
        """Confirms the premise: mx.hadamard_transform really does fail for
        these inputs on the installed MLX version, so the gate must reject
        them. If this test starts failing (transform succeeds), the MLX
        version's kernel constraints have changed and the gate can be
        relaxed accordingly."""
        import mlx.core as mx

        x = mx.array(np.random.randn(d).astype(np.float32))
        with pytest.raises(RuntimeError):
            y = mx.hadamard_transform(x)
            mx.eval(y)


class TestDoubledMValuesAreAccepted:
    """d = m * 2^k for k >= 1 must remain compatible -- these work fine."""

    @pytest.mark.parametrize("d", [24, 48, 96, 40, 80, 56, 112])
    def test_doubled_m_is_compatible(self, d: int) -> None:
        assert is_hadamard_compatible(d)

    @pytest.mark.parametrize("d", [24, 40, 56])
    def test_doubled_m_actually_works_in_mlx(self, d: int) -> None:
        import mlx.core as mx

        x = mx.array(np.random.randn(d).astype(np.float32))
        y = mx.hadamard_transform(x)
        mx.eval(y)


class TestPowersOfTwoAreAccepted:
    @pytest.mark.parametrize("d", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_power_of_two_is_compatible(self, d: int) -> None:
        assert is_hadamard_compatible(d)

    @pytest.mark.parametrize("d", [1, 2, 64, 128])
    def test_power_of_two_actually_works_in_mlx(self, d: int) -> None:
        import mlx.core as mx

        x = mx.array(np.random.randn(d).astype(np.float32))
        y = mx.hadamard_transform(x)
        mx.eval(y)


class TestIncompatibleValues:
    @pytest.mark.parametrize("d", [0, -1, -12, 3, 5, 9])
    def test_incompatible_or_invalid_d_returns_false(self, d: int) -> None:
        assert not is_hadamard_compatible(d)

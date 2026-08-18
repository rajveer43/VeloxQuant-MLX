"""Tests for is_hadamard_compatible, including the bare-m regression for #67."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from veloxquant_mlx.math.rotation import is_hadamard_compatible


@functools.lru_cache(maxsize=1)
def _mlx_supports_bare_m() -> bool:
    """Does the installed MLX handle d == m for non-trivial m (no doublings)?

    Probed rather than version-compared: the fix landed in mlx 0.32.1, but the
    behaviour is what these tests care about, and probing also covers backports
    and any future regression. d=12 stands in for the whole {12, 20, 28} set --
    they are one kernel path and move together.
    """
    import mlx.core as mx

    try:
        y = mx.hadamard_transform(mx.array(np.zeros(12, dtype=np.float32)))
        mx.eval(y)
    except RuntimeError:
        return False
    return True


class TestBareMValuesAreRejected:
    """Regression for #67: d exactly equal to a non-trivial m in
    {12, 20, 28} (i.e. k=0 doublings) crashes mx.hadamard_transform during
    Metal shader compilation and must be reported as incompatible."""

    @pytest.mark.parametrize("d", [12, 20, 28])
    def test_bare_m_is_not_compatible(self, d: int) -> None:
        assert not is_hadamard_compatible(d)

    @pytest.mark.parametrize("d", [12, 20, 28])
    def test_bare_m_behaviour_matches_installed_mlx(self, d: int) -> None:
        """Pins what the installed MLX actually does with bare m, in both
        directions.

        Through MLX 0.32.0 these inputs crash mx.hadamard_transform during
        Metal shader compilation (#67), which is why the gate rejects them.
        MLX 0.32.1 fixed the kernel and they now return a correct orthonormal
        Hadamard. The project still supports mlx>=0.18, so the gate stays
        conservative and rejects them on every version -- a user on 0.31 must
        not be handed a d that crashes their MLX. This test therefore asserts
        whichever behaviour the installed version has, so it keeps working as
        a regression guard instead of encoding one version's bug as a
        permanent fact.

        If the floor is ever raised to >=0.32.1, is_hadamard_compatible can be
        relaxed to accept bare m and this test can drop the older branch.
        """
        import mlx.core as mx

        x = mx.array(np.random.randn(d).astype(np.float32))
        if _mlx_supports_bare_m():
            y = mx.hadamard_transform(x)
            mx.eval(y)
            assert np.isfinite(np.array(y)).all()
        else:
            with pytest.raises(RuntimeError):
                y = mx.hadamard_transform(x)
                mx.eval(y)

    @pytest.mark.parametrize("d", [12, 20, 28])
    def test_bare_m_transform_is_orthonormal_when_supported(self, d: int) -> None:
        """Where MLX does support bare m, what it returns is a real Hadamard.

        Guards the claim behind the note above: relaxing the gate on a newer
        MLX would be safe on correctness grounds. Applying the transform to
        the identity recovers the matrix itself; it must be orthonormal with
        entries +-1/sqrt(d). Note it is NOT symmetric for d=20 and d=28 (the
        Paley construction), so H(H(x)) == x does not hold there and is not a
        valid check.
        """
        import mlx.core as mx

        if not _mlx_supports_bare_m():
            pytest.skip(f"installed MLX {mx.__version__} does not support bare m={d}")

        matrix = np.array(mx.hadamard_transform(mx.array(np.eye(d, dtype=np.float32))))
        gram = matrix @ matrix.T
        np.testing.assert_allclose(gram, np.eye(d), atol=1e-5)
        np.testing.assert_allclose(np.abs(matrix) * np.sqrt(d), np.ones((d, d)), atol=1e-4)


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

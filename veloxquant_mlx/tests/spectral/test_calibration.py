from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_U(d: int = 64, seed: int = 0) -> np.ndarray:
    """Random orthogonal (d,d) matrix — columns are eigenvectors."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
    return U


def _make_rotation_entry(d: int = 64, seed: int = 0) -> tuple:
    key_U = _make_U(d, seed)
    val_U = _make_U(d, seed + 1)
    key_ev = np.ones(d, dtype=np.float64)
    val_ev = np.ones(d, dtype=np.float64)
    key_ds = 4
    val_ds = 50
    return (key_U, val_U, key_ev, val_ev, key_ds, val_ds)


def test_save_and_load_rotations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VELOXQUANT_CACHE_DIR", str(tmp_path))
    import importlib
    import veloxquant_mlx.spectral.calibrate as calib_mod

    importlib.reload(calib_mod)

    d = 64
    rotations = {
        0: _make_rotation_entry(d, seed=0),
        1: _make_rotation_entry(d, seed=2),
    }
    calib_mod.save_rotations("test_model", rotations)
    loaded = calib_mod.load_cached_rotations("test_model")

    assert loaded is not None
    assert set(loaded.keys()) == {0, 1}
    for layer_idx in (0, 1):
        orig = rotations[layer_idx]
        got = loaded[layer_idx]
        np.testing.assert_allclose(orig[0], got[0], atol=1e-5)  # key_U
        np.testing.assert_allclose(orig[1], got[1], atol=1e-5)  # val_U
        assert got[4] == orig[4]  # key_ds
        assert got[5] == orig[5]  # val_ds


def test_load_returns_none_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VELOXQUANT_CACHE_DIR", str(tmp_path))
    import importlib
    import veloxquant_mlx.spectral.calibrate as calib_mod

    importlib.reload(calib_mod)

    result = calib_mod.load_cached_rotations("nonexistent_model")
    assert result is None


def test_rotation_matrix_is_orthonormal():
    U = _make_U(64)
    # U^T U ≈ I (columns orthonormal)
    product = U.T @ U
    np.testing.assert_allclose(product, np.eye(64), atol=1e-5)


def test_model_name_with_slash_is_safe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VELOXQUANT_CACHE_DIR", str(tmp_path))
    import importlib
    import veloxquant_mlx.spectral.calibrate as calib_mod

    importlib.reload(calib_mod)

    d = 32
    rotations = {0: _make_rotation_entry(d, seed=0)}
    calib_mod.save_rotations("Qwen/Qwen2.5-0.5B", rotations)
    loaded = calib_mod.load_cached_rotations("Qwen/Qwen2.5-0.5B")
    assert loaded is not None
    assert 0 in loaded


def test_calibrate_from_vectors_produces_orthonormal_U():
    """calibrate_from_vectors() should return orthonormal U matrices."""
    from veloxquant_mlx.spectral.calibrate import calibrate_from_vectors

    d = 128
    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(rng.standard_normal((d, 4)).astype(np.float32))
    keys = rng.standard_normal((256, 4)).astype(np.float32) @ basis.T
    vals = rng.standard_normal((256, d)).astype(np.float32)

    rotations = calibrate_from_vectors({0: keys}, {0: vals}, model_name="test_ortho")
    key_U, val_U, key_ev, val_ev, key_ds, val_ds = rotations[0]

    # U^T U ≈ I
    np.testing.assert_allclose(key_U.T @ key_U, np.eye(d), atol=1e-4)
    np.testing.assert_allclose(val_U.T @ val_U, np.eye(d), atol=1e-4)

    # Eigenvalues should be non-negative and descending
    assert np.all(key_ev >= -1e-8)
    assert np.all(np.diff(key_ev) <= 1e-6), "key eigenvalues should be non-increasing"

    # d_s should be a reasonable small number for low-rank key data
    assert 1 <= key_ds <= d


def test_calibrate_from_vectors_detects_low_d_eff():
    """Calibrating on rank-4 data should detect d_s ≈ 4."""
    from veloxquant_mlx.spectral.calibrate import calibrate_from_vectors

    d = 128
    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(rng.standard_normal((d, 4)).astype(np.float32))
    keys = rng.standard_normal((512, 4)).astype(np.float32) @ basis.T
    keys += rng.standard_normal((512, d)).astype(np.float32) * 0.01  # small noise

    rotations = calibrate_from_vectors({0: keys}, {0: keys}, model_name="test_deff")
    _, _, key_ev, _, key_ds, _ = rotations[0]

    # For nearly rank-4 data, d_s should be in range [1, 15]
    assert 1 <= key_ds <= 15, f"Expected d_s ≈ 4 for rank-4 data, got {key_ds}"

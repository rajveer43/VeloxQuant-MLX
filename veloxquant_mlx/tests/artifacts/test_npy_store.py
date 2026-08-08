"""Tests for NpyArtifactStore, including the atomic-write regression for #71."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore, _atomic_save
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError


class TestRoundtrip:
    def test_rotation_matrix_roundtrip(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        Pi = np.eye(4, dtype=np.float32)
        store.save_rotation_matrix(Pi, d=4, seed=0)
        loaded = np.asarray(store.load_rotation_matrix(d=4, seed=0))
        assert np.allclose(loaded, Pi.astype(np.float16))

    def test_codebook_roundtrip(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        cb = np.arange(16, dtype=np.float32).reshape(4, 4)
        store.save_codebook(cb, distribution="gaussian", b=2, d=4)
        loaded = np.asarray(store.load_codebook(distribution="gaussian", b=2, d=4))
        assert np.allclose(loaded, cb.astype(np.float16))

    def test_jl_matrix_roundtrip(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        S = np.ones((4, 2), dtype=np.float32)
        store.save_jl_matrix(S, d=4, m=2, seed=0)
        loaded = np.asarray(store.load_jl_matrix(d=4, m=2, seed=0))
        assert np.allclose(loaded, S.astype(np.float16))

    def test_exists_false_before_save_true_after(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        assert not store.exists("rotation", d=8, seed=0)
        store.save_rotation_matrix(np.eye(8), d=8, seed=0)
        assert store.exists("rotation", d=8, seed=0)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.load_rotation_matrix(d=4, seed=0)


class TestAtomicSave:
    """Regression for #71: saves must not leave partial files visible or
    stray temp files behind, even under concurrent access."""

    def test_no_leftover_tmp_files_after_save(self, tmp_path: Path) -> None:
        store = NpyArtifactStore(tmp_path)
        store.save_rotation_matrix(np.eye(4), d=4, seed=0)
        names = os.listdir(tmp_path)
        assert names == ["rotation_d4_seed0.npy"]

    def test_final_file_has_no_partial_write_window(self, tmp_path: Path, monkeypatch) -> None:
        """Simulate a slow write: a concurrent reader polling for the final
        path must see either "not there yet" or a fully-formed, loadable
        file -- never a truncated/corrupt one."""
        path = tmp_path / "target.npy"
        arr = np.arange(10_000, dtype=np.float64).reshape(100, 100)

        real_save = np.save
        release_writer = threading.Event()

        def slow_save(file, data, *a, **k):
            real_save(file, data, *a, **k)
            # Hold the temp file in place before the atomic rename happens,
            # widening the window a racing reader would need to exploit.
            release_writer.wait(timeout=2)

        monkeypatch.setattr(np, "save", slow_save)

        errors: list[Exception] = []

        def writer():
            try:
                _atomic_save(path, arr)
            except Exception as e:  # pragma: no cover - diagnostic only
                errors.append(e)

        def reader():
            deadline = time.time() + 2
            while time.time() < deadline:
                if path.exists():
                    # Any file observable at `path` must be a complete,
                    # loadable, correct array -- since os.replace only
                    # exposes it once the write is fully done.
                    loaded = np.load(path)
                    assert loaded.shape == arr.shape
                    assert np.array_equal(loaded, arr)
                    return
                time.sleep(0.001)

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        t_writer.start()
        t_reader.start()
        time.sleep(0.05)
        release_writer.set()
        t_writer.join(timeout=3)
        t_reader.join(timeout=3)

        assert not errors
        assert np.array_equal(np.load(path), arr)
        assert os.listdir(tmp_path) == ["target.npy"]

    def test_concurrent_saves_to_same_path_leave_one_valid_complete_file(
        self, tmp_path: Path
    ) -> None:
        """Two 'workers' racing to save the same artifact path must never
        leave a corrupt/partial file -- the loser's write is simply
        overwritten atomically, not interleaved."""
        path = tmp_path / "shared.npy"
        arr_a = np.full((50, 50), 1.0, dtype=np.float64)
        arr_b = np.full((50, 50), 2.0, dtype=np.float64)

        errors: list[Exception] = []

        def save(arr):
            try:
                for _ in range(20):
                    _atomic_save(path, arr)
            except Exception as e:  # pragma: no cover - diagnostic only
                errors.append(e)

        threads = [
            threading.Thread(target=save, args=(arr_a,)),
            threading.Thread(target=save, args=(arr_b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        # File must always be fully one array or the other, never a mix.
        final = np.load(path)
        assert np.array_equal(final, arr_a) or np.array_equal(final, arr_b)
        assert [n for n in os.listdir(tmp_path) if n != "shared.npy"] == []

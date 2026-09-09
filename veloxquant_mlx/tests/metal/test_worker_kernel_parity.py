from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal GPU is required", allow_module_level=True)

from veloxquant_mlx.cli.worker import _bit_pack_file, _rope_recode_file
from veloxquant_mlx.transfer.rope import recode_rope


def test_worker_bit_pack_file_matches_exact_reference(tmp_path):
    source = tmp_path / "input.npy"
    target = tmp_path / "output.npy"
    values = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
    np.save(source, values, allow_pickle=False)

    result = _bit_pack_file({"inputPath": str(source), "outputPath": str(target), "bits": 2})

    assert result["backend"] == "metal"
    np.testing.assert_array_equal(np.load(target, allow_pickle=False), np.array([228, 228], dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_worker_rope_file_matches_reference(tmp_path, dtype):
    source = tmp_path / "keys.npy"
    positions_path = tmp_path / "positions.npy"
    target = tmp_path / "output.npy"
    keys = np.arange(2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8).astype(dtype)
    positions = np.arange(4, dtype=np.int32)
    np.save(source, keys, allow_pickle=False)
    np.save(positions_path, positions, allow_pickle=False)

    result = _rope_recode_file({
        "inputPath": str(source), "positionsPath": str(positions_path),
        "outputPath": str(target), "sourceBase": 10000.0, "targetBase": 100000.0,
    })

    actual = mx.array(np.load(target, allow_pickle=False))
    expected = recode_rope(mx.array(keys), mx.array(positions), 10000.0, 100000.0, use_metal=False)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3)
    assert result["shape"] == [2, 4, 8]

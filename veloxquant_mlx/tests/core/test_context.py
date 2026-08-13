"""Tests for EncodedVector.memory_bytes(), including the outlier_idx
undercounting regression for #65."""

from __future__ import annotations

import numpy as np

from veloxquant_mlx.core.context import EncodedVector


class TestMemoryBytesOutlierIdx:
    """Regression for #65: outlier_idx (populated by CompositeQuantizer)
    was silently excluded from memory_bytes()'s total."""

    def test_outlier_idx_contributes_to_total(self) -> None:
        outlier_idx = np.array([1, 5, 9, 20], dtype=np.int32)
        ev_with = EncodedVector(
            quantizer_type="composite",
            batch_size=1,
            dim=64,
            indices=np.zeros((1, 64), dtype=np.uint8),
            outlier_idx=outlier_idx,
        )
        ev_without = EncodedVector(
            quantizer_type="composite",
            batch_size=1,
            dim=64,
            indices=np.zeros((1, 64), dtype=np.uint8),
            outlier_idx=None,
        )
        assert ev_with.memory_bytes() - ev_without.memory_bytes() == outlier_idx.nbytes

    def test_exact_byte_count_matches_int32_itemsize(self) -> None:
        outlier_idx = np.arange(10, dtype=np.int32)
        ev = EncodedVector(
            quantizer_type="composite", batch_size=1, dim=64, outlier_idx=outlier_idx
        )
        assert ev.memory_bytes() == 10 * 4

    def test_shape_matching_composite_encode_output_counts_outlier_idx(self) -> None:
        """An EncodedVector shaped like CompositeQuantizer.encode()'s actual
        output (outlier_idx + nested outlier/inlier_encoded, per
        composite.py:58-65) must include the outlier_idx bytes."""
        outlier_idx = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int32)
        outlier_encoded = EncodedVector(
            quantizer_type="polar",
            batch_size=4,
            dim=8,
            angles=None,
            final_radius=np.zeros(4, dtype=np.float16),
        )
        inlier_encoded = EncodedVector(
            quantizer_type="polar",
            batch_size=4,
            dim=8,
            angles=None,
            final_radius=np.zeros(4, dtype=np.float16),
        )
        ev = EncodedVector(
            quantizer_type="composite",
            batch_size=4,
            dim=16,
            outlier_idx=outlier_idx,
            outlier_encoded=outlier_encoded,
            inlier_encoded=inlier_encoded,
        )

        without_outlier = ev.memory_bytes() - outlier_idx.nbytes
        ev.outlier_idx = None
        assert ev.memory_bytes() == without_outlier


class TestMemoryBytesFieldCombinations:
    def test_all_none_fields_is_zero(self) -> None:
        ev = EncodedVector(quantizer_type="empty", batch_size=1, dim=4)
        assert ev.memory_bytes() == 0

    def test_angles_list_contributes_per_level(self) -> None:
        import mlx.core as mx

        angles = [mx.zeros((2, 4), dtype=mx.float16), mx.zeros((2, 2), dtype=mx.float16)]
        ev = EncodedVector(quantizer_type="polar", batch_size=2, dim=8, angles=angles)
        assert ev.memory_bytes() == (2 * 4 + 2 * 2) * 2

    def test_nested_outlier_and_inlier_encoded_are_summed_recursively(self) -> None:
        inner_outlier = EncodedVector(
            quantizer_type="polar",
            batch_size=1,
            dim=3,
            indices=np.zeros((1, 3), dtype=np.uint8),
        )
        inner_inlier = EncodedVector(
            quantizer_type="polar",
            batch_size=1,
            dim=13,
            indices=np.zeros((1, 13), dtype=np.uint8),
        )
        outer = EncodedVector(
            quantizer_type="composite",
            batch_size=1,
            dim=16,
            outlier_idx=np.array([0, 5, 9], dtype=np.int32),
            outlier_encoded=inner_outlier,
            inlier_encoded=inner_inlier,
        )
        expected = (
            inner_outlier.memory_bytes() + inner_inlier.memory_bytes() + outer.outlier_idx.nbytes
        )
        assert outer.memory_bytes() == expected

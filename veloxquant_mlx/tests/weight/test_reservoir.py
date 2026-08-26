"""Tests for weight/reservoir.py: mmap-backed serialization of already
quantize_model()-processed weights (see docs/WEIGHT_RESERVOIR_IDEATION.md)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from veloxquant_mlx.weight.model_quantizer import quantize_model
from veloxquant_mlx.weight.quantized_linear import QuantizedLinear
from veloxquant_mlx.weight.reservoir import graft_reservoir, load_reservoir, save_reservoir


class _ToyModel(nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 96, bias=bias)
        self.fc2 = nn.Linear(96, 32, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.fc1(x))


class _ToyModelWithQRFallback(nn.Module):
    """in_features=50 is not Hadamard-compatible (see
    veloxquant_mlx.math.rotation.is_hadamard_compatible), so quantize_model()
    falls back to the QR-derived RotationPreconditioner for fc1. Exercises
    the persist_rotation=True/False split in reservoir.py, which only
    applies to this fallback path -- Hadamard layers are unaffected."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(50, 32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc1(x)


class TestSaveLoadRoundTrip:
    """Reservoir load must be bit-exact against the freshly quantized model:
    identical indices, norms, bias, and forward-pass output."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_is_bit_exact(self, tmp_path, bits: int) -> None:
        mx.random.seed(0)
        model = quantize_model(_ToyModel(), bits=bits, use_hadamard=True)

        x = mx.random.normal((2, 64))
        y_before = model(x)
        mx.eval(y_before)

        path = tmp_path / "toy.vqrs"
        save_reservoir(model, path)

        model2 = quantize_model(_ToyModel(), bits=bits, use_hadamard=True)
        model2 = graft_reservoir(model2, path)
        y_after = model2(x)
        mx.eval(y_after)

        assert float(mx.abs(y_before - y_after).max()) == 0.0

        for name, layer in model.named_modules():
            if not isinstance(layer, QuantizedLinear):
                continue
            other = dict(model2.named_modules())[name]
            assert mx.array_equal(layer._w_indices, other._w_indices)
            assert mx.array_equal(layer._w_norms, other._w_norms)

    def test_bias_round_trips_exactly(self, tmp_path) -> None:
        mx.random.seed(1)
        model = quantize_model(_ToyModel(bias=True), bits=4)
        path = tmp_path / "bias.vqrs"
        save_reservoir(model, path)

        layers = load_reservoir(path)
        assert mx.array_equal(layers["fc1"]._bias, model.fc1._bias)
        assert layers["fc2"]._bias is None

    def test_load_reservoir_skips_requantization(self, tmp_path) -> None:
        """load_reservoir must not recompute indices via argmin -- it should
        assign the persisted indices directly. A cheap proxy: a layer whose
        weights would quantize differently under a different seed still
        loads the *original* seed's indices, not a re-derived assignment."""
        mx.random.seed(2)
        model = quantize_model(_ToyModel(), bits=4, seed=7)
        path = tmp_path / "seeded.vqrs"
        save_reservoir(model, path)

        layers = load_reservoir(path)
        assert layers["fc1"]._seed == model.fc1._seed
        assert mx.array_equal(layers["fc1"]._w_indices, model.fc1._w_indices)


class TestQRFallbackRotationPersistence:
    """persist_rotation controls whether QR-fallback layers store their full
    d x d rotation matrix (fast load, larger file) or re-derive it from the
    seed on load (slow load via np.linalg.qr, small file). Both must be
    bit-exact -- see Finding 4 in docs/WEIGHT_RESERVOIR_IDEATION.md."""

    @pytest.mark.parametrize("persist_rotation", [False, True])
    def test_qr_fallback_round_trip_is_bit_exact(self, tmp_path, persist_rotation: bool) -> None:
        mx.random.seed(5)
        model = quantize_model(_ToyModelWithQRFallback(), bits=4, use_hadamard=True)

        from veloxquant_mlx.preconditioners.rotation import RotationPreconditioner

        assert isinstance(model.fc1._preconditioner, RotationPreconditioner), (
            "test setup assumption failed: in_features=50 should trigger QR fallback"
        )

        x = mx.random.normal((2, 50))
        y_before = model(x)
        mx.eval(y_before)

        path = tmp_path / "qr.vqrs"
        save_reservoir(model, path, persist_rotation=persist_rotation)

        layers = load_reservoir(path)
        raw_skeleton = _ToyModelWithQRFallback()
        grafted = graft_reservoir(raw_skeleton, path)
        y_after = grafted(x)
        mx.eval(y_after)

        assert float(mx.abs(y_before - y_after).max()) == 0.0
        assert mx.array_equal(layers["fc1"]._preconditioner._Pi, model.fc1._preconditioner._Pi)

    def test_persist_rotation_false_keeps_file_small(self, tmp_path) -> None:
        mx.random.seed(6)
        model = quantize_model(_ToyModelWithQRFallback(), bits=4, use_hadamard=True)

        small_path = tmp_path / "small.vqrs"
        large_path = tmp_path / "large.vqrs"
        save_reservoir(model, small_path, persist_rotation=False)
        save_reservoir(model, large_path, persist_rotation=True)

        # d=50 QR matrix is 50*50*4 = 10000 bytes (one page); persisting it
        # must make the file at least one page larger.
        assert large_path.stat().st_size > small_path.stat().st_size


class TestGraftOntoRawSkeleton:
    """graft_reservoir must work on a raw (never-quantized) nn.Linear
    skeleton -- it must not require quantize_model() to have already run,
    since paying a full quantize pass just to graft over it would defeat
    the point (see scripts/bench_weight_reservoir.py)."""

    def test_graft_onto_unquantized_model(self, tmp_path) -> None:
        mx.random.seed(4)
        source = quantize_model(_ToyModel(), bits=4)
        x = mx.random.normal((2, 64))
        y_before = source(x)
        mx.eval(y_before)

        path = tmp_path / "raw_graft.vqrs"
        save_reservoir(source, path)

        raw_skeleton = _ToyModel()  # never passed through quantize_model()
        grafted = graft_reservoir(raw_skeleton, path)
        y_after = grafted(x)
        mx.eval(y_after)

        assert float(mx.abs(y_before - y_after).max()) == 0.0
        assert isinstance(grafted.fc1, QuantizedLinear)


class TestFileFormat:
    def test_rejects_bad_magic(self, tmp_path) -> None:
        path = tmp_path / "bad.vqrs"
        path.write_bytes(b"NOPE" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a VeloxQuant reservoir"):
            load_reservoir(path)

    def test_layer_names_preserved(self, tmp_path) -> None:
        mx.random.seed(3)
        model = quantize_model(_ToyModel(), bits=4)
        path = tmp_path / "names.vqrs"
        save_reservoir(model, path)

        layers = load_reservoir(path)
        assert set(layers.keys()) == {"fc1", "fc2"}

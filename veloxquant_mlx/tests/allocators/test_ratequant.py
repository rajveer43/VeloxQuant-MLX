"""Tests for the RateQuant per-layer bit allocator."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx import (
    KVCacheBuilder,
    KVCacheConfig,
    KVCacheFactory,
    QuantizerConfigError,
    allocate_bits_ratequant,
    calibrate_layer_sensitivities,
    fit_distortion_curve,
)


class TestAllocateBitsRateQuant:
    def test_uniform_weights_collapse_to_target(self) -> None:
        """Equal sensitivities should give every layer the same bits (= target)."""
        alloc = allocate_bits_ratequant([1.0] * 32, target_avg_bits=2.0)
        assert alloc == [2] * 32
        assert sum(alloc) == 64

    def test_non_uniform_weights_prefer_high_sensitivity(self) -> None:
        """Higher-sensitivity layers get more bits."""
        # Layer 0 is 10x more sensitive than layer 1; others equal
        w = [10.0, 1.0] + [3.0] * 30
        alloc = allocate_bits_ratequant(w, target_avg_bits=1.5, beta=3.5, bit_choices=(1, 2, 3))
        assert alloc[0] >= alloc[1]
        assert alloc[0] in (2, 3)
        assert alloc[1] in (1, 2)

    def test_target_is_exactly_hit(self) -> None:
        """Integer total must equal round(target * N) after re-balance."""
        rng = np.random.default_rng(0)
        w = rng.lognormal(0, 1.0, 28).tolist()
        for target in (1.0, 1.5, 1.7, 2.0, 2.5):
            alloc = allocate_bits_ratequant(
                w, target_avg_bits=target, beta=3.5, bit_choices=(1, 2, 3)
            )
            assert sum(alloc) == round(target * len(w)), (
                f"target={target} not hit: got {sum(alloc)}/{len(w)}"
            )

    def test_bit_choices_are_respected(self) -> None:
        """No allocated bit-width may fall outside the supplied set."""
        rng = np.random.default_rng(0)
        w = rng.lognormal(0, 1.5, 16).tolist()
        alloc = allocate_bits_ratequant(w, target_avg_bits=2.0, bit_choices=(2, 4))
        assert all(b in (2, 4) for b in alloc)

    def test_non_contiguous_bit_choices_membership(self) -> None:
        """Regression for #72: every allocated value must be an actual member
        of a non-contiguous bit_choices set, not merely within [min, max]."""
        sensitivities = [1.0, 5.0, 0.2, 3.0, 0.5]
        choices = (0, 1, 2, 4, 6, 8)
        alloc = allocate_bits_ratequant(
            sensitivities, target_avg_bits=2.6, bit_choices=choices
        )
        assert all(a in choices for a in alloc), alloc

    def test_non_contiguous_bit_choices_membership_randomized(self) -> None:
        """Broader sweep over random sparse bit_choices sets and targets."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(2, 12))
            w = rng.lognormal(0, 1.0, n).tolist()
            target = float(rng.uniform(0.5, 7.0))
            pool = rng.choice(9, size=int(rng.integers(2, 6)), replace=False)
            choices = tuple(sorted(set(int(c) for c in pool)))
            alloc = allocate_bits_ratequant(w, target_avg_bits=target, bit_choices=choices)
            assert all(a in choices for a in alloc), (choices, alloc)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            allocate_bits_ratequant([1.0, -0.5, 2.0], target_avg_bits=1.5)

    def test_empty_bit_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="bit_choices"):
            allocate_bits_ratequant([1.0, 2.0], target_avg_bits=1.5, bit_choices=())


class TestFitDistortionCurve:
    def test_returns_positive_alpha_beta(self) -> None:
        alpha, beta = fit_distortion_curve(head_dim=128)
        assert alpha > 0
        assert beta > 1.0

    def test_rvq_beta_near_paper_value(self) -> None:
        """Paper reports β ≈ 3.5 for TurboQuant on head_dim=128."""
        _, beta = fit_distortion_curve(head_dim=128)
        assert 2.0 < beta < 6.0, f"beta={beta} far from paper-reported ~3.5"


class TestKVCacheConfigListBitWidth:
    def test_config_accepts_list_bit_width(self) -> None:
        cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=[1, 2, 1, 3])
        assert cfg.bit_width_inlier == [1, 2, 1, 3]

    def test_factory_rejects_list(self) -> None:
        """Direct factory takes single int; list flows through for_model."""
        cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=[1, 2])
        with pytest.raises(QuantizerConfigError, match="single int"):
            KVCacheFactory.create(cfg)

    def test_builder_rejects_invalid_list(self) -> None:
        with pytest.raises(QuantizerConfigError):
            KVCacheBuilder().with_method("turboquant_rvq").with_head_dim(128).with_bit_width(
                [1, "oops", 2]
            ).build()

    def test_builder_accepts_int(self) -> None:
        c = (
            KVCacheBuilder()
            .with_method("turboquant_rvq")
            .with_head_dim(128)
            .with_bit_width(1)
            .with_seed(0)
            .build()
        )
        assert c.assigned_bits == 1


class TestForModelPerLayer:
    """`KVCacheBuilder.for_model()` consumes per-layer bit-width lists."""

    def _make_mock_model(self, n_layers: int):
        class _MockAttn:
            head_dim = 128

        class _MockLayer:
            def __init__(self) -> None:
                self.self_attn = _MockAttn()

        class _MockArgs:
            hidden_size = 1024
            num_attention_heads = 8

        class _MockModel:
            args = _MockArgs()
            layers = [_MockLayer() for _ in range(n_layers)]

        return _MockModel()

    def test_uniform_int_propagates(self) -> None:
        model = self._make_mock_model(4)
        cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=0)
        caches = KVCacheBuilder.for_model(model, cfg)
        assert len(caches) == 4
        assert all(c.assigned_bits == 2 for c in caches)

    def test_per_layer_list_propagates(self) -> None:
        model = self._make_mock_model(4)
        cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=[3, 1, 2, 1], seed=0)
        caches = KVCacheBuilder.for_model(model, cfg)
        assert [c.assigned_bits for c in caches] == [3, 1, 2, 1]

    def test_wrong_length_list_raises(self) -> None:
        model = self._make_mock_model(4)
        cfg = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=[1, 1, 1], seed=0)
        with pytest.raises(QuantizerConfigError, match="length"):
            KVCacheBuilder.for_model(model, cfg)


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [ord(c) % 100 for c in prompt][:16]


class TestCalibrateLayerSensitivitiesRestoresMakeCache:
    """Regression for #73: a raising forward pass must not permanently leave
    model.make_cache pointed at the calibration lambda."""

    class _RaisingModel:
        def __init__(self) -> None:
            self.layers = [1, 2, 3]
            self.make_cache = "original_make_cache"

        def __call__(self, *_a, **_k):
            raise RuntimeError("simulated forward-pass failure")

    class _RaisingModelNoOriginal:
        """A model that never had its own make_cache attribute."""

        def __init__(self) -> None:
            self.layers = [1, 2]

        def __call__(self, *_a, **_k):
            raise RuntimeError("simulated forward-pass failure")

    def test_restores_original_make_cache_after_forward_pass_raises(self) -> None:
        model = self._RaisingModel()
        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            calibrate_layer_sensitivities(model, _FakeTokenizer(), prompts=["hello world"])
        assert model.make_cache == "original_make_cache"

    def test_deletes_patched_make_cache_after_raise_when_no_original(self) -> None:
        model = self._RaisingModelNoOriginal()
        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            calibrate_layer_sensitivities(model, _FakeTokenizer(), prompts=["hello world"])
        assert not hasattr(model, "make_cache")

    def test_deletes_patched_make_cache_after_success_when_no_original(self) -> None:
        """Even on the success path, a model with no original make_cache must
        not be left with the calibration lambda attached."""

        class _SucceedingModel:
            def __init__(self) -> None:
                self.layers = [1, 2]

            def __call__(self, *_a, **_k):
                return None

        model = _SucceedingModel()
        calibrate_layer_sensitivities(model, _FakeTokenizer(), prompts=["hi"])
        assert not hasattr(model, "make_cache")

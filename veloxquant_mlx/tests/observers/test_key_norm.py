"""Tests for KeyNormObserver."""

from __future__ import annotations

from veloxquant_mlx import KeyNormObserver
from veloxquant_mlx.observers import QuantizationEvent


def _event(norm_sq) -> QuantizationEvent:
    return QuantizationEvent(stage="test", input_shape=(1,), metadata={"key_l2_norm_sq": norm_sq})


def test_empty_report() -> None:
    obs = KeyNormObserver()
    r = obs.report()
    assert r.n_tokens == 0


def test_accepts_scalar_norm() -> None:
    obs = KeyNormObserver()
    obs.on_event(_event(4.0))
    obs.on_event(_event(9.0))
    r = obs.report()
    assert r.n_tokens == 2
    assert r.mean_norm_sq == 6.5
    assert r.min_norm_sq == 4.0
    assert r.max_norm_sq == 9.0


def test_accepts_iterable_norm() -> None:
    obs = KeyNormObserver()
    obs.on_event(_event([1.0, 4.0, 9.0, 16.0]))
    r = obs.report()
    assert r.n_tokens == 4
    assert r.mean_norm_sq == 7.5


def test_heterogeneity_ratio() -> None:
    obs = KeyNormObserver()
    obs.on_event(_event([1.0, 100.0]))
    r = obs.report()
    assert r.heterogeneity_ratio == 100.0


def test_ignores_events_without_metadata() -> None:
    obs = KeyNormObserver()
    obs.on_event(QuantizationEvent(stage="other", input_shape=(1,)))
    assert obs.report().n_tokens == 0


def test_reset_clears_state() -> None:
    obs = KeyNormObserver()
    obs.on_event(_event(5.0))
    obs.reset()
    assert obs.report().n_tokens == 0

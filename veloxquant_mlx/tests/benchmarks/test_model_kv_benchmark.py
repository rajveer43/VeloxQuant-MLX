"""Regression tests for the perplexity helper in model_kv_benchmark (#660)."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

# matplotlib is the optional [plots] extra; the release gate installs only [test].
pytest.importorskip("matplotlib")

from veloxquant_mlx.benchmarks import model_kv_benchmark as mkb  # noqa: E402

_VOCAB = 11


class _FakeTokenizer:
    def encode(self, text):
        return [ord(c) % _VOCAB for c in text]


class _FakeModel:
    """Logits depend on the cache contents, so a model that ignores the
    cache and one that uses it give different perplexity."""

    def __init__(self):
        self.make_cache_calls = 0
        self.chunk_lengths = []

    def make_cache(self):
        self.make_cache_calls += 1
        return [{"seen": 0}]

    def __call__(self, ids, cache=None):
        self.chunk_lengths.append(ids.shape[1])
        offset = cache[0]["seen"] if cache is not None else 0
        if cache is not None:
            cache[0]["seen"] += ids.shape[1]
        pos = mx.arange(ids.shape[1])[None, :, None] + offset
        classes = mx.arange(_VOCAB)[None, None, :]
        return mx.cos(pos * 0.7 + classes * 0.3).astype(mx.float32)


@pytest.mark.parametrize("fn", ["compute_perplexity", "compute_perplexity_stable"])
def test_perplexity_runs_through_the_models_cache(fn):
    model = _FakeModel()
    ppl = getattr(mkb, fn)(model, _FakeTokenizer(), "x" * 3 + "abcdefghij" * 20, max_tokens=150)
    assert model.make_cache_calls == 1
    assert math.isfinite(ppl)


def test_perplexity_prefill_is_chunked_and_covers_every_position():
    model = _FakeModel()
    ids = mx.zeros((1, 150), dtype=mx.int32)
    logits = mkb._teacher_forced_logits(model, ids, chunk_size=64)
    assert model.chunk_lengths == [64, 64, 22]
    assert logits.shape == (1, 150, _VOCAB)

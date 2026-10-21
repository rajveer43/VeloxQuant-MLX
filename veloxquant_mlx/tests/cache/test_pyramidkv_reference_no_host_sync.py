"""pyramidkv's default (reference) eviction must not sync to the host (#846).

The reference path called ``int(mx.argmin(...).item())`` once per evicted
token, which made a 2,048-token prefill ~14x slower than the mlx backend.
Timing is too noisy for CI, so this pins the cause: eviction must not build a
Python index list (which needed the host-side argmin), and the result must
still match the batched mlx backend.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers import pyramidkv
from veloxquant_mlx.quantizers.pyramidkv import init_pyramid_state, pyramid_update

D, S, BUDGET, SINK = 16, 80, 24, 4


def _run(backend):
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    state = init_pyramid_state(SINK, BUDGET, D)
    return pyramid_update(state, k, v, backend=backend)


def test_reference_eviction_gathers_on_device(monkeypatch):
    """The host-sync path built a Python index list (``range``) per eviction."""
    real_range, calls = range, []

    def spy(*args):
        calls.append(args)
        return real_range(*args)

    monkeypatch.setattr(pyramidkv, "range", spy, raising=False)
    state = _run("reference")
    assert state.keys.shape[0] == BUDGET
    assert len(calls) == 1, f"per-eviction Python index loops: {len(calls) - 1}"


def test_reference_matches_mlx_backend():
    ref, fast = _run("reference"), _run("mlx")
    np.testing.assert_array_equal(np.array(ref.keys), np.array(fast.keys))
    np.testing.assert_array_equal(np.array(ref.scores), np.array(fast.scores))

"""chunkkv prefill must not run eviction machinery on steps that cannot evict (#846).

Chunk eviction drops ``chunk_size`` rows at once, so with the default size of 8
it fires about once per 8 tokens. The batched path used to build the chunk
one-hot, argsort and gather four buffers on every step anyway. Timing is too
noisy for CI, so this pins the work count, and checks the state still matches
the unbatched per-head reference.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers import chunkkv
from veloxquant_mlx.quantizers.chunkkv import chunkkv_update_batched

BH, D, S, BUDGET, CHUNK, SINK = 3, 16, 400, 64, 8, 4


def _run():
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((BH, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((BH, S, D)).astype(np.float16))
    pos = mx.broadcast_to(mx.arange(S, dtype=mx.int32)[None], (BH, S))
    return chunkkv_update_batched(
        None, None, None, None, None, k, v, pos, SINK, BUDGET, CHUNK, "attn_mass"
    )


def test_eviction_only_runs_on_steps_that_can_evict(monkeypatch):
    real, calls = chunkkv._evict_lowest_chunk_batched, []

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(chunkkv, "_evict_lowest_chunk_batched", spy)
    keys, _, _, _, lengths = _run()
    mx.eval(keys, lengths)
    # Without the skip this is one call per token (S).
    assert len(calls) < S // 3, f"{len(calls)} eviction passes for {S} tokens"
    assert int(mx.max(lengths).item()) <= BUDGET


def test_state_matches_per_head_reference():
    keys, values, _, positions, lengths = _run()
    rng = np.random.default_rng(0)
    k = mx.array(rng.standard_normal((BH, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((BH, S, D)).astype(np.float16))
    for row in range(BH):
        state = chunkkv.init_chunkkv_state(SINK, BUDGET, D, CHUNK)
        state = chunkkv.chunkkv_update(state, k[row], v[row])
        n = int(lengths[row].item())
        assert n == state.keys.shape[0]
        np.testing.assert_array_equal(np.array(keys[row, :n]), np.array(state.keys))
        np.testing.assert_array_equal(np.array(values[row, :n]), np.array(state.values))

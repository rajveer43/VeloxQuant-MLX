"""Parity checks for ChunkKV's batched per-head eviction dispatch (#564).

``ChunkKVCache.update_and_fetch`` looped ``for b in range(B): for h in
range(H):``, dispatching each row to ``chunkkv_update`` (leader) or
``chunkkv_apply_reuse_indices`` (follower) one at a time — every call, on
both prefill and decode. The outer dispatch is independent per row (each
head owns its own state and the loop only routes data); the per-token
chunk-eviction decision *inside* those functions is a genuine recurrence
(chunk boundaries and scores accumulate token-by-token) and stays untouched.

Unlike every other eviction method fixed in this series (H2O, CaM, Squeeze,
KVZip, Keyformer, MorphKV -- always exactly one row evicted per step, so
state stays perfectly rectangular), a ChunkKV eviction removes a whole
*chunk* (``chunk_size`` rows, or fewer for the ragged tail chunk) -- or zero,
for a row not yet over budget -- so different ``(b,h)`` rows can genuinely
hold different valid lengths within one multi-token absorption call. The
batched primitives therefore carry state as a padded ``[BH, n_max, D]`` /
``[BH, n_max]`` buffer plus an explicit ``[BH]`` ``lengths`` array (the
standard batched-ragged-sequence pattern) instead of H2O-style always-
rectangular ``[BH, n, D]`` state, only re-aligning to a common rectangular
shape at the very end of each call via ``chunkkv_trim_batched`` -- exactly
mirroring what the per-head loop + the existing scalar ``chunkkv_trim_to``
already did every call, just now also batched.

Three batched primitives, verified independently and end-to-end:
  - ``chunkkv_update_batched``: the leader path (own scoring/eviction).
  - ``chunkkv_apply_reuse_indices_batched``: the follower path (externally
    supplied per-head kept-index lists from the coordinator).
  - ``chunkkv_trim_batched``: the batched cross-head min-length alignment.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
from veloxquant_mlx.cache.chunkkv_coordinator import ChunkKVIndexReuseCoordinator
from veloxquant_mlx.quantizers.chunkkv import (
    chunkkv_apply_reuse_indices,
    chunkkv_apply_reuse_indices_batched,
    chunkkv_get_kv,
    chunkkv_trim_batched,
    chunkkv_trim_to,
    chunkkv_update,
    chunkkv_update_batched,
    init_chunkkv_state,
)


def _ref_loop_with_positions(
    bh, s, d, n_sink, budget, chunk_size, score_mode, keys_seq, values_seq
):
    states = [
        init_chunkkv_state(n_sink, budget, d, chunk_size=chunk_size, score_mode=score_mode)
        for _ in range(bh)
    ]
    bh_positions = [None] * bh
    for t in range(s):
        new_positions = mx.arange(t, t + 1, dtype=mx.int32)
        for i in range(bh):
            states[i], kept = chunkkv_update(
                states[i],
                keys_seq[i, t : t + 1],
                values_seq[i, t : t + 1],
                record_kept_positions=True,
            )
            p_i = new_positions
            positions = (
                p_i if bh_positions[i] is None else mx.concatenate([bh_positions[i], p_i], axis=0)
            )
            bh_positions[i] = positions[kept[0]]
    ks, vs = [], []
    for i in range(bh):
        k, v = chunkkv_get_kv(states[i])
        ks.append(k)
        vs.append(v)
    return ks, vs, bh_positions


def _run_leader_case(desc, bh, s, d, n_sink, budget, chunk_size, score_mode, data_seed):
    rng = np.random.default_rng(data_seed)
    keys_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    values_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)

    ref_ks, ref_vs, ref_pos = _ref_loop_with_positions(
        bh, s, d, n_sink, budget, chunk_size, score_mode, keys_seq, values_seq
    )

    keys = values = scores = positions = lengths = None
    new_positions = mx.broadcast_to(mx.arange(s, dtype=mx.int32)[None, :], (bh, s))
    keys, values, scores, positions, lengths = chunkkv_update_batched(
        keys,
        values,
        scores,
        positions,
        lengths,
        keys_seq,
        values_seq,
        new_positions,
        n_sink,
        budget,
        chunk_size,
        score_mode,
    )
    mx.eval(keys, values, positions, lengths)

    for i in range(bh):
        n_i = int(lengths[i].item())
        assert n_i == ref_ks[i].shape[0], f"{desc}: row {i} length mismatch"
        assert mx.array_equal(ref_ks[i], keys[i, :n_i]).item(), f"{desc}: row {i} key mismatch"
        assert mx.array_equal(ref_vs[i], values[i, :n_i]).item(), f"{desc}: row {i} value mismatch"
        assert mx.array_equal(ref_pos[i], positions[i, :n_i]).item(), (
            f"{desc}: row {i} position mismatch"
        )


@pytest.mark.parametrize(
    "desc,bh,s,d,n_sink,budget,chunk_size,score_mode,data_seed",
    [
        ("attn_mass basic", 5, 40, 8, 2, 10, 3, "attn_mass", 1),
        ("key_norm basic", 5, 40, 8, 2, 10, 3, "key_norm", 1),
        ("chunk_size=1 == H2O", 4, 35, 6, 1, 8, 1, "attn_mass", 2),
        ("large chunk, large budget", 3, 60, 8, 3, 20, 8, "attn_mass", 3),
        ("no sinks", 3, 30, 4, 0, 9, 3, "attn_mass", 4),
        ("no sinks key_norm", 3, 30, 4, 0, 9, 3, "key_norm", 4),
        ("ragged tail chunk", 6, 50, 8, 2, 13, 4, "attn_mass", 5),
        ("single head", 1, 25, 8, 1, 7, 2, "attn_mass", 6),
        ("decode-shaped many S=1 steps", 8, 20, 16, 2, 10, 3, "attn_mass", 7),
        ("chunk_size close to budget", 4, 45, 8, 2, 15, 13, "attn_mass", 8),
    ],
)
def test_update_batched_matches_looped(
    desc, bh, s, d, n_sink, budget, chunk_size, score_mode, data_seed
):
    _run_leader_case(desc, bh, s, d, n_sink, budget, chunk_size, score_mode, data_seed)


def _follower_ref(bh, kept_positions_per_row, keys_seq, values_seq, n_sink, budget, chunk_size):
    states = [
        init_chunkkv_state(
            n_sink, budget, keys_seq.shape[-1], chunk_size=chunk_size, score_mode="attn_mass"
        )
        for _ in range(bh)
    ]
    for i in range(bh):
        states[i] = chunkkv_apply_reuse_indices(
            states[i], keys_seq[i], values_seq[i], kept_positions_per_row[i]
        )
    ks, vs = [], []
    for i in range(bh):
        k, v = chunkkv_get_kv(states[i])
        ks.append(k)
        vs.append(v)
    return ks, vs


@pytest.mark.parametrize(
    "desc,bh,s,d,n_sink,budget,chunk_size,data_seed",
    [
        ("follower basic", 5, 40, 8, 2, 10, 3, 11),
        ("follower ragged tail", 6, 50, 8, 2, 13, 4, 12),
        ("follower decode-shaped", 8, 20, 16, 2, 10, 3, 13),
    ],
)
def test_apply_reuse_indices_batched_matches_looped(
    desc, bh, s, d, n_sink, budget, chunk_size, data_seed
):
    rng = np.random.default_rng(data_seed)
    leader_keys = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    leader_values = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    kept_positions_per_row = []
    for i in range(bh):
        st = init_chunkkv_state(n_sink, budget, d, chunk_size=chunk_size, score_mode="attn_mass")
        st, kept = chunkkv_update(st, leader_keys[i], leader_values[i], record_kept_positions=True)
        kept_positions_per_row.append(kept)

    rng2 = np.random.default_rng(data_seed + 1000)
    fol_keys = mx.array(rng2.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    fol_values = mx.array(rng2.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)

    ref_ks, ref_vs = _follower_ref(
        bh, kept_positions_per_row, fol_keys, fol_values, n_sink, budget, chunk_size
    )

    keys = values = positions = lengths = None
    new_positions = mx.zeros((bh, s), dtype=mx.int32)
    keys, values, positions, lengths = chunkkv_apply_reuse_indices_batched(
        keys,
        values,
        positions,
        lengths,
        fol_keys,
        fol_values,
        new_positions,
        kept_positions_per_row,
    )
    mx.eval(keys, values, lengths)

    for i in range(bh):
        n_i = int(lengths[i].item())
        assert n_i == ref_ks[i].shape[0], f"{desc}: row {i} length mismatch"
        assert mx.array_equal(ref_ks[i], keys[i, :n_i]).item(), f"{desc}: row {i} key mismatch"
        assert mx.array_equal(ref_vs[i], values[i, :n_i]).item(), f"{desc}: row {i} value mismatch"


def test_trim_batched_matches_looped():
    bh, d, n_sink, budget, chunk_size, s = 5, 8, 2, 10, 3, 40
    rng = np.random.default_rng(21)
    keys_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    values_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)

    ref_states = [
        init_chunkkv_state(n_sink, budget, d, chunk_size=chunk_size, score_mode="attn_mass")
        for _ in range(bh)
    ]
    for i in range(bh):
        ref_states[i] = chunkkv_update(ref_states[i], keys_seq[i], values_seq[i])
    min_kept = min(chunkkv_get_kv(st)[0].shape[0] for st in ref_states)
    ref_states = [chunkkv_trim_to(st, min_kept) for st in ref_states]
    ref_ks = [chunkkv_get_kv(st)[0] for st in ref_states]
    ref_vs = [chunkkv_get_kv(st)[1] for st in ref_states]

    keys = values = scores = positions = lengths = None
    new_positions = mx.broadcast_to(mx.arange(s, dtype=mx.int32)[None, :], (bh, s))
    keys, values, scores, positions, lengths = chunkkv_update_batched(
        keys,
        values,
        scores,
        positions,
        lengths,
        keys_seq,
        values_seq,
        new_positions,
        n_sink,
        budget,
        chunk_size,
        "attn_mass",
    )
    min_kept_b = int(mx.min(lengths).item())
    keys, values, scores, positions, lengths = chunkkv_trim_batched(
        keys, values, scores, positions, lengths, n_sink, min_kept_b
    )
    mx.eval(keys, values, lengths)

    assert min_kept_b == min_kept
    for i in range(bh):
        n_i = int(lengths[i].item())
        assert n_i == ref_ks[i].shape[0]
        assert mx.array_equal(ref_ks[i], keys[i, :n_i]).item()
        assert mx.array_equal(ref_vs[i], values[i, :n_i]).item()


# ---------------------------------------------------------------------------
# Cache-level: real ChunkKVCache, B>1/H>1, multi-call (including a growing
# buffer across varied call sizes -- the scenario that surfaced a real
# buffer-undersizing bug in chunkkv_apply_reuse_indices_batched during
# development, now covered here).
# ---------------------------------------------------------------------------


def _kv(B, H, S, D, seed):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**cfg):
    base = {"method": "chunkkv", "head_dim": 8}
    base.update(cfg)
    return ChunkKVCache(KVCacheConfig(**base))


@pytest.mark.parametrize(
    "desc,B,H,D,n_sink,budget,chunk_size,score_mode,call_sizes,seed",
    [
        ("B=2,H=3 varied calls attn_mass", 2, 3, 8, 2, 10, 3, "attn_mass", [20, 1, 1, 1, 30, 1], 1),
        ("B=1,H=4 key_norm", 1, 4, 8, 1, 8, 2, "key_norm", [15, 1, 1, 40, 1], 2),
        ("B=3,H=2 chunk_size=1 == H2O", 3, 2, 6, 1, 6, 1, "attn_mass", [10, 1, 1, 1, 1], 3),
        ("B=2,H=2 many decode steps", 2, 2, 8, 2, 9, 3, "attn_mass", [1] * 15, 4),
    ],
)
def test_real_cache_matches_reference_loop(
    desc, B, H, D, n_sink, budget, chunk_size, score_mode, call_sizes, seed
):
    rng = np.random.default_rng(seed)
    ref_states = [
        init_chunkkv_state(n_sink, budget, D, chunk_size=chunk_size, score_mode=score_mode)
        for _ in range(B * H)
    ]
    cache = _make(
        head_dim=D,
        chunkkv_budget=budget,
        chunkkv_n_sink=n_sink,
        chunkkv_chunk_size=chunk_size,
        chunkkv_score=score_mode,
    )
    for s in call_sizes:
        k = mx.array(rng.standard_normal((B, H, s, D)).astype(np.float32)).astype(mx.float16)
        v = mx.array(rng.standard_normal((B, H, s, D)).astype(np.float32)).astype(mx.float16)
        cache.update_and_fetch(k, v)
        for b in range(B):
            for h in range(H):
                idx = b * H + h
                ref_states[idx] = chunkkv_update(ref_states[idx], k[b, h], v[b, h])
        min_kept_ref = min(chunkkv_get_kv(st)[0].shape[0] for st in ref_states)
        ref_states = [chunkkv_trim_to(st, min_kept_ref) for st in ref_states]

    min_kept = int(cache._bh_lengths[0].item())
    for b in range(B):
        for h in range(H):
            idx = b * H + h
            ref_k, ref_v = chunkkv_get_kv(ref_states[idx])
            got_k = cache._bh_keys[idx, :min_kept]
            got_v = cache._bh_values[idx, :min_kept]
            assert mx.array_equal(ref_k, got_k).item(), f"{desc}: key mismatch (b={b},h={h})"
            assert mx.array_equal(ref_v, got_v).item(), f"{desc}: value mismatch (b={b},h={h})"


@pytest.mark.parametrize(
    "desc,B,H,D,n_sink,budget,chunk_size,call_sizes,seed",
    [
        ("leader/follower B=2,H=3 varied calls", 2, 3, 8, 2, 10, 3, [20, 1, 1, 30, 1], 11),
        (
            "leader/follower B=1,H=2 large prefill then decode",
            1,
            2,
            16,
            2,
            12,
            4,
            [50, 1, 1, 1, 1, 1],
            30,
        ),
    ],
)
def test_leader_follower_multi_batch_multi_call(
    desc, B, H, D, n_sink, budget, chunk_size, call_sizes, seed
):
    """A follower's output must stay bit-identical to its leader's across
    B>1/H>1 and varying call sizes -- this includes a growing-buffer
    scenario (large prefill then several small decode steps, or vice versa)
    that surfaced a real buffer-undersizing bug in
    chunkkv_apply_reuse_indices_batched (its capacity was only ever sized
    once, on the first call, against that call's own kept-index widths --
    fixed to re-validate/grow on every call)."""
    coord = ChunkKVIndexReuseCoordinator(n_layers=2, reuse_layers=2)
    cfg = KVCacheConfig(
        method="chunkkv",
        head_dim=D,
        chunkkv_budget=budget,
        chunkkv_n_sink=n_sink,
        chunkkv_chunk_size=chunk_size,
    )
    leader = ChunkKVCache(cfg, layer_id=0, coordinator=coord)
    follower = ChunkKVCache(cfg, layer_id=1, coordinator=coord)

    rng = np.random.default_rng(seed)
    for s in call_sizes:
        k = mx.array(rng.standard_normal((B, H, s, D)).astype(np.float32)).astype(mx.float16)
        v = mx.array(rng.standard_normal((B, H, s, D)).astype(np.float32)).astype(mx.float16)
        Kl, Vl = leader.update_and_fetch(k, v)
        Kf, Vf = follower.update_and_fetch(k, v)
        assert Kl.shape == Kf.shape, f"{desc}: shape mismatch at call size {s}"
        assert bool(mx.all(Kl == Kf).item()), f"{desc}: K mismatch at call size {s}"
        assert bool(mx.all(Vl == Vf).item()), f"{desc}: V mismatch at call size {s}"

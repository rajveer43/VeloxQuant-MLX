"""Equivalence tests: batched [B*H, ...] AnchorKV primitives vs. a per-head
reference loop calling the unbatched functions.

select_anchors_batched/assign_and_project_batched/key_value_utility_batched
replace AnchorKVKVCache._process_prefill's Python loop over (b, h) with one
batched call per primitive across all B*H heads. Anchor SELECTION differs
per row (each row scores/samples off its own data), but the batching is
still bit-for-bit exact: select_anchors_batched passes the same `seed` to
every row's RNG draw (matching the old per-head loop, which always called
select_anchors with the same self._seed for every head), so nothing about
*which* positions get chosen changes -- only the dispatch does.

allocate_residual_budget is deliberately NOT batched (kept as one call per
row, exactly as before) per the filing issue: at this call site it only
ever sees one row's utilities, so batching that call site would silently
start using its cross-head pooling capability -- a distinct design decision
out of scope here.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.anchorkv_cache import AnchorKVKVCache
from veloxquant_mlx.quantizers.anchorkv import (
    assign_and_project,
    assign_and_project_batched,
    key_value_utility,
    key_value_utility_batched,
    select_anchors,
    select_anchors_batched,
)


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "desc,bh,S,k,window,rho,seed",
    [
        ("single_row", 1, 10, 4, 2, 0.7, 0),
        ("multi_row", 4, 12, 5, 2, 0.7, 1),
        ("no_scoring_all_uniform", 4, 12, 5, 0, 0.0, 2),
        ("no_uniform_all_scored", 4, 12, 5, 2, 1.0, 3),
        ("window_covers_all", 3, 4, 4, 4, 0.5, 4),
        ("k_below_window", 3, 10, 2, 4, 0.5, 5),
        ("single_token", 3, 1, 4, 4, 0.5, 6),
    ],
)
def test_select_anchors_batched_matches_reference(desc, bh, S, k, window, rho, seed):
    keys = _rand((bh, S, 8), seed)
    anchors_b = select_anchors_batched(keys, k=k, window=window, rho=rho, seed=42)
    for row in range(bh):
        ref = select_anchors(keys[row], k=k, window=window, rho=rho, seed=42)
        assert mx.array_equal(anchors_b[row], ref), f"{desc}: row {row} diverged"


@pytest.mark.parametrize(
    "desc,bh,S,D,seed",
    [
        ("single_row", 1, 10, 8, 0),
        ("multi_row", 4, 12, 8, 1),
        ("odd_head_dim", 3, 10, 7, 2),
    ],
)
def test_assign_and_project_batched_matches_reference(desc, bh, S, D, seed):
    x = _rand((bh, S, D), seed)
    anchors = select_anchors_batched(x, k=5, window=2, rho=0.7, seed=42)
    res_b = assign_and_project_batched(x, anchors)
    for row in range(bh):
        ref = assign_and_project(x[row], anchors[row])
        assert mx.array_equal(res_b.assign_idx[row], ref.assign_idx), f"{desc}: row {row}"
        assert mx.array_equal(res_b.gamma[row], ref.gamma), f"{desc}: row {row}"
        assert mx.array_equal(res_b.residual[row], ref.residual), f"{desc}: row {row}"


@pytest.mark.parametrize(
    "desc,bh,S,D,m,seed",
    [
        ("single_row", 1, 10, 8, 4, 0),
        ("multi_row", 4, 12, 8, 4, 1),
    ],
)
def test_key_value_utility_batched_matches_reference(desc, bh, S, D, m, seed):
    keys = _rand((bh, S, D), seed)
    values = _rand((bh, S, D), seed + 1)
    kres = _rand((bh, S, D), seed + 2)
    vres = _rand((bh, S, D), seed + 3)
    proxy = keys[:, -m:]

    uk_b, uv_b = key_value_utility_batched(proxy, keys, values, kres, vres)
    for row in range(bh):
        uk_ref, uv_ref = key_value_utility(proxy[row], keys[row], values[row], kres[row], vres[row])
        assert mx.array_equal(uk_b[row], uk_ref), f"{desc}: row {row} u_key"
        assert mx.array_equal(uv_b[row], uv_ref), f"{desc}: row {row} u_value"


# ---------------------------------------------------------------------------
# Real AnchorKVKVCache end-to-end vs. a manual per-head reference using the
# original unbatched primitives directly (not a re-import of master).
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {
        "method": "anchorkv",
        "head_dim": 16,
        "anchorkv_theta": 0.3,
        "anchorkv_window": 4,
        "anchorkv_rho": 0.7,
        "anchorkv_anchor_frac": 0.2,
        "anchorkv_seed": 7,
    }
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize("B,H,seed", [(1, 1, 0), (2, 1, 1), (1, 4, 2), (2, 3, 3)])
def test_real_cache_prefill_matches_reference_loop(B, H, seed):
    from veloxquant_mlx.quantizers.anchorkv import (
        ResidualCodec,
        allocate_residual_budget,
        anchorkv_budget_slots,
    )

    D = 16
    S = 24
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(head_dim=D)
    k_out, v_out = cache.update_and_fetch(k, v)

    # Reference: the pre-batching per-head algorithm, called directly.
    theta, window, rho, anchor_frac, res_bits, seed_cfg = 0.3, 4, 0.7, 0.2, 2, 7
    codec = ResidualCodec(head_dim=D, seed=seed_cfg, bits=res_bits)
    k_budget = max(1, int(round(S * anchor_frac)))

    k_out_b, v_out_b = [], []
    for b in range(B):
        k_out_h, v_out_h = [], []
        for h in range(H):
            keys_h, values_h = k[b, h], v[b, h]
            anchors = select_anchors(
                keys_h.astype(mx.float32), k=k_budget, window=window, rho=rho, seed=seed_cfg
            )
            n_anchor = int(anchors.shape[0])
            key_assign = assign_and_project(keys_h, anchors)
            value_assign = assign_and_project(values_h, anchors)
            m = min(window, S)
            proxy_q = keys_h.astype(mx.float32)[-m:]
            u_key, u_value = key_value_utility(
                proxy_q,
                keys_h.astype(mx.float32),
                values_h.astype(mx.float32),
                key_assign.residual,
                value_assign.residual,
            )
            anchor_set = {int(a) for a in anchors.tolist()}
            non_anchor_mask = mx.array([i not in anchor_set for i in range(S)])
            neg_inf = mx.where(non_anchor_mask, mx.zeros((S,)), mx.full((S,), -1e30))
            u_key = u_key + neg_inf
            u_value = u_value + neg_inf

            n_slots = anchorkv_budget_slots(
                seq_len=S,
                head_dim=D,
                n_anchor=n_anchor,
                theta=theta,
                residual_codec_bytes=codec.bytes_per_residual,
            )
            n_key_slots = n_slots // 2
            n_value_slots = n_slots - n_key_slots
            key_mask = allocate_residual_budget([u_key], n_key_slots)[0]
            value_mask = allocate_residual_budget([u_value], n_value_slots)[0]

            def recon(x, assign, mask):
                chosen_anchor = x.astype(mx.float32)[assign.anchor_positions][assign.assign_idx]
                x_tilde = assign.gamma[:, None] * chosen_anchor
                codes, scale = codec.encode(assign.residual)
                decoded = codec.decode(codes, scale)
                term = mx.where(mask[:, None], decoded, mx.zeros_like(decoded))
                return (x_tilde + term).astype(mx.float16)

            k_out_h.append(recon(keys_h, key_assign, key_mask))
            v_out_h.append(recon(values_h, value_assign, value_mask))
        k_out_b.append(mx.stack(k_out_h, axis=0))
        v_out_b.append(mx.stack(v_out_h, axis=0))

    ref_k = mx.stack(k_out_b, axis=0)
    ref_v = mx.stack(v_out_b, axis=0)

    assert mx.array_equal(k_out, ref_k)
    assert mx.array_equal(v_out, ref_v)

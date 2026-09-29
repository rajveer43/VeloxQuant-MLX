"""Parity checks for PALU's batched group-head projection/reconstruction (#561).

``PALUKVCache``'s ``_TensorLowRank.append``/``.reconstruct`` used to loop
``for h in range(H):``, projecting/quantizing/reconstructing each head one
at a time through its assigned group's frozen basis — called up to 4x per
token (K-append, K-reconstruct, V-append, V-reconstruct). Heads within a
head-group share the same basis (``head_group_bounds`` returns contiguous
``[lo, hi)`` ranges, and ``group_head_svd`` fits one basis per group), so
the fix batches each group's heads into one matmul/quantize call instead of
one call per head — the same "group heads by shared parameter, batch within
group" recipe as AdaKV's ``quantize_heads_batched``.

``project_to_latent_batched`` / ``reconstruct_from_latent_batched`` /
``quantize_latent_batched`` are the batched primitives; this file checks
each against its per-head counterpart, then checks the cache end-to-end.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.palu import (
    project_to_latent,
    project_to_latent_batched,
    quantize_latent,
    quantize_latent_batched,
    reconstruct_from_latent,
    reconstruct_from_latent_batched,
)


def _make(**cfg):
    base = {"method": "palu", "head_dim": 16}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Primitive parity: per-head looped vs. group-batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g,s,d,r", [(3, 20, 16, 6), (1, 5, 8, 3), (5, 12, 32, 10)])
def test_project_to_latent_batched_matches_looped(g: int, s: int, d: int, r: int) -> None:
    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((g, s, d)).astype(np.float32))
    V = mx.array(rng.standard_normal((d, r)).astype(np.float32))
    mu = mx.array(rng.standard_normal((d,)).astype(np.float32))
    ref = mx.stack([project_to_latent(x[i], V, mu) for i in range(g)])
    batched = project_to_latent_batched(x, V, mu)
    mx.eval(ref, batched)
    assert mx.allclose(ref, batched, atol=1e-5).item()


@pytest.mark.parametrize("g,s,d,r", [(3, 20, 16, 6), (1, 5, 8, 3), (5, 12, 32, 10)])
def test_reconstruct_from_latent_batched_matches_looped(g: int, s: int, d: int, r: int) -> None:
    rng = np.random.default_rng(7)
    latents = mx.array(rng.standard_normal((g, s, r)).astype(np.float32))
    V = mx.array(rng.standard_normal((d, r)).astype(np.float32))
    mu = mx.array(rng.standard_normal((d,)).astype(np.float32))
    ref = mx.stack([reconstruct_from_latent(latents[i], V, mu) for i in range(g)])
    batched = reconstruct_from_latent_batched(latents, V, mu)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


@pytest.mark.parametrize(
    "g,s,r,hi_bit,lo_bit,hi_frac,gs",
    [
        (3, 40, 8, 4, 2, 0.25, 8),
        (1, 10, 4, 8, 4, 0.5, 4),
        (5, 33, 12, 4, 2, 0.3, 16),  # non-divisible S -> padded last group
    ],
)
def test_quantize_latent_batched_matches_looped(g, s, r, hi_bit, lo_bit, hi_frac, gs) -> None:
    rng = np.random.default_rng(7)
    latents = mx.array(rng.standard_normal((g, s, r)).astype(np.float32))
    sv = mx.array(np.abs(rng.standard_normal((r,))).astype(np.float32))
    ref = mx.stack([quantize_latent(latents[i], sv, hi_bit, lo_bit, hi_frac, gs) for i in range(g)])
    batched = quantize_latent_batched(latents, sv, hi_bit, lo_bit, hi_frac, gs)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


# ---------------------------------------------------------------------------
# 2. Cache-level: reconstructed output matches a per-head reference loop
# ---------------------------------------------------------------------------


def _encode_step_ref(lr, x_step: mx.array) -> list[mx.array]:
    """Per-head reference for one _TensorLowRank.append() call: [H, S, D] -> per-head [S, r]."""
    H = x_step.shape[0]
    out = []
    for h in range(H):
        g = lr._head_group[h]
        L = project_to_latent(x_step[h].astype(mx.float32), lr._V[g], lr._mu[g])
        if lr.quantize:
            L = quantize_latent(L, lr._sv[g], lr.hi_bit, lr.lo_bit, lr.hi_fraction, lr.group_size)
        out.append(L.astype(mx.float16))
    return out


def _reconstruct_ref(lr, latents_per_head: list[mx.array]) -> mx.array:
    heads = []
    for h, L in enumerate(latents_per_head):
        g = lr._head_group[h]
        heads.append(reconstruct_from_latent(L, lr._V[g], lr._mu[g]))
    return mx.stack(heads, axis=0)[None]


def test_cache_reconstruction_matches_reference_loop() -> None:
    """Feed a prefill + several decode steps through PALUKVCache (batched
    append/reconstruct) and, after every step, confirm the reconstructed
    K/V match a per-head reference loop fed the SAME per-step chunks (so
    quantization group boundaries line up exactly as they did for the old
    per-head-loop code, which was also called once per update_and_fetch)."""
    H, D = 6, 16
    n_prefill, n_decode = 24, 5
    cfg = {"palu_n_head_groups": 3, "palu_rank": 6, "head_dim": D}

    rng = np.random.default_rng(21)
    all_k = rng.standard_normal((n_prefill + n_decode, H, D)).astype(np.float16)
    all_v = rng.standard_normal((n_prefill + n_decode, H, D)).astype(np.float16)

    cache = _make(**cfg)
    k_prefill = mx.array(all_k[:n_prefill]).transpose(1, 0, 2)[None]  # [1, H, n_prefill, D]
    v_prefill = mx.array(all_v[:n_prefill]).transpose(1, 0, 2)[None]
    k_out, v_out = cache.update_and_fetch(k_prefill, v_prefill)

    ref_k_latents = _encode_step_ref(cache._keys_lr, mx.array(all_k[:n_prefill]).transpose(1, 0, 2))
    ref_v_latents = _encode_step_ref(cache._vals_lr, mx.array(all_v[:n_prefill]).transpose(1, 0, 2))
    ref_k = _reconstruct_ref(cache._keys_lr, ref_k_latents)
    ref_v = _reconstruct_ref(cache._vals_lr, ref_v_latents)
    mx.eval(k_out, v_out, ref_k, ref_v)
    assert mx.array_equal(k_out, ref_k).item()
    assert mx.array_equal(v_out, ref_v).item()

    for t in range(n_prefill, n_prefill + n_decode):
        k_step = mx.array(all_k[t])[None, :, None, :]  # [1, H, 1, D]
        v_step = mx.array(all_v[t])[None, :, None, :]
        k_out, v_out = cache.update_and_fetch(k_step, v_step)

        new_k = _encode_step_ref(cache._keys_lr, mx.array(all_k[t])[:, None, :])
        new_v = _encode_step_ref(cache._vals_lr, mx.array(all_v[t])[:, None, :])
        ref_k_latents = [
            mx.concatenate([prev, new], axis=0)
            for prev, new in zip(ref_k_latents, new_k, strict=True)
        ]
        ref_v_latents = [
            mx.concatenate([prev, new], axis=0)
            for prev, new in zip(ref_v_latents, new_v, strict=True)
        ]
        ref_k = _reconstruct_ref(cache._keys_lr, ref_k_latents)
        ref_v = _reconstruct_ref(cache._vals_lr, ref_v_latents)
        mx.eval(k_out, v_out, ref_k, ref_v)
        assert mx.array_equal(k_out, ref_k).item(), f"key mismatch at decode step {t}"
        assert mx.array_equal(v_out, ref_v).item(), f"value mismatch at decode step {t}"

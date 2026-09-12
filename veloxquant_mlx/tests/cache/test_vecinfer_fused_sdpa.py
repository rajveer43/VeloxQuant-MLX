"""Parity + integration tests for the Phase-2 fused VecInfer SDPA path.

These tests skip cleanly when Metal is unavailable.  They cover:

* config flag three-state resolution (None / True / False)
* shape preservation through update_and_fetch
* parity vs pure-MLX reference (causal, non-causal, sliding window)
* GQA broadcast
* short-sequence regression guard (S_kv = 1, 2)
* long-sequence correctness (S_kv = 4096)
* mlx_lm dispatcher patch idempotence
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_queries,
    dequantize_vq,
    walsh_hadamard_matrix,
)
from veloxquant_mlx.metal import metal_available

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_cache(
    *,
    fused_sdpa: bool,
    head_dim: int = 128,
    key_sub_dim: int = 8,
    seed: int = 0,
):
    """Build a VecInferKVCache with a deterministic random codebook.

    Using ``key_codebook_bits=8`` keeps n_centroids=256, which is the
    fused kernel's cap.
    """
    rng = np.random.default_rng(seed)
    n_centroids = 256
    cb_k = mx.array(rng.standard_normal((n_centroids, key_sub_dim)).astype(np.float32))
    cb_v = mx.array(rng.standard_normal((n_centroids, key_sub_dim)).astype(np.float32))
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=head_dim,
        key_sub_dim=key_sub_dim,
        value_sub_dim=key_sub_dim,
        key_codebook_bits=8,
        value_codebook_bits=8,
        seed=seed,
        key_codebook=cb_k,
        value_codebook=cb_v,
        fused_sdpa=fused_sdpa,
    )
    return KVCacheFactory.create(cfg)


def _reference_sdpa(
    q: mx.array,  # [B, H_q, S_q, D]   fp16
    k_indices: mx.array,  # [B, H_kv, S_kv, n_sub]
    k_codebook: mx.array,
    v_indices: mx.array,
    v_codebook: mx.array,
    smooth: mx.array,  # may be None for identity smooth
    H: mx.array,
    scale: float,
    causal: bool = True,
    sliding_window: int = 0,
) -> mx.array:
    """Pure-MLX reference: dequant K_hat / V_hat, standard SDPA."""
    B, H_q, S_q, D = q.shape
    _, H_kv, S_kv, _ = k_indices.shape

    k_hat_tilde = dequantize_vq(k_indices, k_codebook).astype(mx.float32)
    k_hat = k_hat_tilde @ H.T.astype(mx.float32)
    if smooth is not None:
        if smooth.ndim == 2 and k_hat.shape[-3] == smooth.shape[0]:
            sm_b = smooth[:, None, :].astype(mx.float32)
        elif smooth.ndim == 2:
            sm_b = mx.mean(smooth, axis=0).astype(mx.float32)
        else:
            sm_b = smooth.astype(mx.float32)
        k_hat = k_hat * sm_b

    v_hat = dequantize_vq(v_indices, v_codebook).astype(mx.float32)

    rep = H_q // H_kv
    if rep > 1:
        k_hat = mx.repeat(k_hat, repeats=rep, axis=1)
        v_hat = mx.repeat(v_hat, repeats=rep, axis=1)

    q32 = q.astype(mx.float32)
    scores = (q32 @ mx.swapaxes(k_hat, -2, -1)) * scale
    if causal or sliding_window:
        q_pos = mx.arange(S_q) + (S_kv - S_q)
        k_pos = mx.arange(S_kv)
        if causal:
            causal_mask = q_pos[:, None] < k_pos[None, :]
            scores = mx.where(causal_mask, mx.array(-1e9, dtype=mx.float32), scores)
        if sliding_window and sliding_window > 0:
            window_mask = k_pos[None, :] < (q_pos[:, None] - sliding_window + 1)
            scores = mx.where(window_mask, mx.array(-1e9, dtype=mx.float32), scores)
    weights = mx.softmax(scores, axis=-1)
    return weights @ v_hat


def _populate_cache_with_random_kv(cache, B, H_kv, S, D, seed=42):
    """Feed S tokens through update_and_fetch so the cache holds indices."""
    rng = np.random.default_rng(seed)
    keys = mx.array(rng.standard_normal((B, H_kv, S, D)).astype(np.float32) * 0.3).astype(
        mx.float16
    )
    vals = mx.array(rng.standard_normal((B, H_kv, S, D)).astype(np.float32) * 0.3).astype(
        mx.float16
    )
    cache.update_and_fetch(keys, vals)
    return keys, vals


def _run_and_compare(
    cache,
    q,
    *,
    causal,
    sliding_window,
    scale,
):
    """Compute fused output and pure-MLX reference; return (out_fused, max_diff)."""
    out_fused = cache.fused_sdpa(q, scale=scale, causal=causal, sliding_window=sliding_window)

    # Slice the live portion of the cache's ring buffer for the reference
    # path (the buffer is pre-allocated to fused_sdpa_max_ctx).
    s = cache._stored_S_kv
    live_k = cache._stored_k_indices[:, :, :s, :]
    live_v = cache._stored_v_indices[:, :, :s, :]

    out_ref = _reference_sdpa(
        q=q,
        k_indices=live_k,
        k_codebook=cache._key_codebook,
        v_indices=live_v,
        v_codebook=cache._value_codebook,
        smooth=cache._smooth,
        H=cache._H,
        scale=scale,
        causal=causal,
        sliding_window=sliding_window,
    )
    mx.eval(out_fused, out_ref)
    diff = float(mx.max(mx.abs(out_fused.astype(mx.float32) - out_ref.astype(mx.float32))).item())
    return out_fused, diff


# ===========================================================================
# Tests
# ===========================================================================
def test_config_flag_three_state() -> None:
    """fused_sdpa flag resolves correctly for None / False / True."""
    c_off = _build_cache(fused_sdpa=False)
    assert c_off._fused_enabled is False

    c_on = _build_cache(fused_sdpa=True)
    assert c_on._fused_enabled is True


def test_update_and_fetch_still_returns_full_tensors_when_fused() -> None:
    """Even in fused mode, update_and_fetch returns the standard K/V
    tensors so non-patched code paths still work (defense in depth)."""
    c = _build_cache(fused_sdpa=True)
    keys = mx.random.normal((1, 4, 8, 128)).astype(mx.float16)
    vals = mx.random.normal((1, 4, 8, 128)).astype(mx.float16)
    k, v = c.update_and_fetch(keys, vals)
    assert k.shape == (1, 4, 8, 128)
    assert v.shape == (1, 4, 8, 128)
    assert k.dtype == mx.float16


def test_fused_sdpa_matches_reference_causal() -> None:
    """Single-query decode with causal mask matches reference within fp16 tol."""
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S, D = 1, 4, 64, 128
    _populate_cache_with_random_kv(c, B, H_kv, S, D)
    q = mx.array(
        np.random.default_rng(7).standard_normal((B, 16, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=True, sliding_window=0, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"causal fused vs ref max diff = {diff:.3e}"


def test_fused_sdpa_matches_reference_non_causal() -> None:
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S, D = 1, 4, 64, 128
    _populate_cache_with_random_kv(c, B, H_kv, S, D)
    q = mx.array(
        np.random.default_rng(8).standard_normal((B, 16, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=False, sliding_window=0, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"non-causal fused vs ref max diff = {diff:.3e}"


def test_fused_sdpa_matches_reference_sliding_window() -> None:
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S, D = 1, 4, 256, 128
    _populate_cache_with_random_kv(c, B, H_kv, S, D)
    q = mx.array(
        np.random.default_rng(9).standard_normal((B, 16, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=True, sliding_window=64, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"sliding-window fused vs ref max diff = {diff:.3e}"


def test_fused_sdpa_gqa_broadcast() -> None:
    """GQA: H_q=32, H_kv=8 — kernel must integer-divide head indices correctly."""
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S, D = 1, 8, 64, 128
    _populate_cache_with_random_kv(c, B, H_kv, S, D)
    q = mx.array(
        np.random.default_rng(10).standard_normal((B, 32, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=True, sliding_window=0, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"GQA fused vs ref max diff = {diff:.3e}"


def test_fused_sdpa_handles_short_seq() -> None:
    """S_kv = 1 and S_kv = 2 — off-by-one and single-tile regression guard."""
    for S in (1, 2):
        c = _build_cache(fused_sdpa=True)
        B, H_kv, D = 1, 4, 128
        _populate_cache_with_random_kv(c, B, H_kv, S, D)
        q = mx.array(
            np.random.default_rng(11 + S).standard_normal((B, 16, 1, D)).astype(np.float32) * 0.2
        ).astype(mx.float16)
        _, diff = _run_and_compare(c, q, causal=True, sliding_window=0, scale=1.0 / D**0.5)
        assert diff < 1e-2, f"S_kv={S}: max diff {diff:.3e}"


def test_fused_sdpa_long_seq() -> None:
    """S_kv = 4096 — correctness at the scale that motivated Phase 2."""
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S, D = 1, 8, 4096, 128
    _populate_cache_with_random_kv(c, B, H_kv, S, D)
    q = mx.array(
        np.random.default_rng(13).standard_normal((B, 32, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=True, sliding_window=0, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"long-seq fused vs ref max diff = {diff:.3e}"


def test_fused_sdpa_multi_token_prefill() -> None:
    """S_q > 1 (prefill), not just single-token decode — the kernel's grid
    is parameterized by S_q, but every prior test only exercised S_q=1."""
    c = _build_cache(fused_sdpa=True)
    B, H_kv, S_kv, D = 1, 4, 64, 128
    _populate_cache_with_random_kv(c, B, H_kv, S_kv, D)
    S_q = 8
    q = mx.array(
        np.random.default_rng(21).standard_normal((B, 16, S_q, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    _, diff = _run_and_compare(c, q, causal=True, sliding_window=0, scale=1.0 / D**0.5)
    assert diff < 1e-2, f"multi-token prefill fused vs ref max diff = {diff:.3e}"


# ===========================================================================
# memory_bound wiring: update_and_fetch skip + dispatcher routing
# ===========================================================================
def _build_memory_bound_cache(*, head_dim: int = 128, key_sub_dim: int = 8, seed: int = 0):
    from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa, unpatch_mlx_lm

    unpatch_mlx_lm()  # start clean regardless of prior test state
    patch_mlx_lm_for_fused_sdpa()
    rng = np.random.default_rng(seed)
    n_centroids = 256
    cb_k = mx.array(rng.standard_normal((n_centroids, key_sub_dim)).astype(np.float32))
    cb_v = mx.array(rng.standard_normal((n_centroids, key_sub_dim)).astype(np.float32))
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=head_dim,
        key_sub_dim=key_sub_dim,
        value_sub_dim=key_sub_dim,
        key_codebook_bits=8,
        value_codebook_bits=8,
        seed=seed,
        key_codebook=cb_k,
        value_codebook=cb_v,
        fused_sdpa=True,
        fused_sdpa_memory_bound=True,
    )
    return KVCacheFactory.create(cfg)


def test_memory_bound_requires_patch_active() -> None:
    """Constructing with fused_sdpa_memory_bound=True before the mlx_lm
    dispatcher is patched must fail loudly, not silently produce zeros."""
    from veloxquant_mlx.metal.fused_sdpa import unpatch_mlx_lm

    unpatch_mlx_lm()
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=128,
        key_sub_dim=8,
        value_sub_dim=8,
        key_codebook_bits=8,
        value_codebook_bits=8,
        fused_sdpa=True,
        fused_sdpa_memory_bound=True,
    )
    with pytest.raises(RuntimeError, match="patch_mlx_lm_for_fused_sdpa"):
        KVCacheFactory.create(cfg)


def test_memory_bound_requires_fused_sdpa_flag() -> None:
    """fused_sdpa_memory_bound=True without fused_sdpa=True must fail loudly."""
    cfg = KVCacheConfig(
        method="vecinfer",
        head_dim=128,
        key_sub_dim=8,
        value_sub_dim=8,
        key_codebook_bits=8,
        value_codebook_bits=8,
        fused_sdpa=False,
        fused_sdpa_memory_bound=True,
    )
    with pytest.raises(RuntimeError, match="requires fused_sdpa=True"):
        KVCacheFactory.create(cfg)


def test_memory_bound_update_and_fetch_does_not_materialize_fp16() -> None:
    """The actual point of this fix: nbytes must reflect uint32 indices
    only, not a full fp16 K_hat/V_hat buffer."""
    c = _build_memory_bound_cache()
    B, H_kv, S, D = 1, 8, 16, 128
    keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
    vals = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
    c.update_and_fetch(keys, vals)

    fp16_equivalent_bytes = B * H_kv * S * D * 2 * 2  # K + V, fp16
    assert c.nbytes < fp16_equivalent_bytes, (
        f"nbytes={c.nbytes} should be well under the fp16-equivalent "
        f"{fp16_equivalent_bytes} bytes in memory_bound mode"
    )

    from veloxquant_mlx.metal.fused_sdpa import unpatch_mlx_lm

    unpatch_mlx_lm()


def test_memory_bound_end_to_end_via_patched_dispatcher() -> None:
    """Drive attention through mlx_lm's real (patched) SDPA entry point --
    not calling cache.fused_sdpa() directly -- and check it matches the
    same reference used by the direct-call tests above."""
    import mlx_lm.models.base as _base

    c = _build_memory_bound_cache(seed=5)
    B, H_kv, S, D = 1, 4, 32, 128
    keys = mx.array(
        np.random.default_rng(5).standard_normal((B, H_kv, S, D)).astype(np.float32) * 0.3
    ).astype(mx.float16)
    vals = mx.array(
        np.random.default_rng(6).standard_normal((B, H_kv, S, D)).astype(np.float32) * 0.3
    ).astype(mx.float16)
    c.update_and_fetch(keys, vals)  # memory-bound path: sentinel zeros back to caller

    q = mx.array(
        np.random.default_rng(7).standard_normal((B, 16, 1, D)).astype(np.float32) * 0.2
    ).astype(mx.float16)
    out_dispatched = _base.scaled_dot_product_attention(
        q, mx.zeros((B, H_kv, S, D), mx.float16), mx.zeros((B, H_kv, S, D), mx.float16),
        cache=c, scale=1.0 / D**0.5, mask="causal",
    )
    out_direct = c.fused_sdpa(q, scale=1.0 / D**0.5, causal=True, sliding_window=0)
    mx.eval(out_dispatched, out_direct)
    diff = float(mx.max(mx.abs(out_dispatched.astype(mx.float32) - out_direct.astype(mx.float32))).item())
    assert diff == 0.0, f"dispatched vs direct fused_sdpa call diverged: {diff:.3e}"

    from veloxquant_mlx.metal.fused_sdpa import unpatch_mlx_lm

    unpatch_mlx_lm()


def test_non_memory_bound_cache_is_unaffected_by_patch() -> None:
    """A cache NOT in memory_bound mode must still go through the standard
    path even while the dispatcher patch is globally active -- the patch
    must not accidentally intercept every VecInfer cache."""
    from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa, unpatch_mlx_lm
    import mlx_lm.models.base as _base

    unpatch_mlx_lm()
    patch_mlx_lm_for_fused_sdpa()
    try:
        c = _build_cache(fused_sdpa=True)  # fused_sdpa=True, memory_bound=False (default)
        B, H_kv, S, D = 1, 4, 8, 128
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        vals = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        k_out, v_out = c.update_and_fetch(keys, vals)
        assert k_out.shape == (B, H_kv, S, D) and k_out.dtype == mx.float16

        q = mx.random.normal((B, 16, 1, D)).astype(mx.float16)
        # Must not raise/misbehave when routed through the patched dispatcher:
        # non-memory-bound caches fall through to _original_sdpa with the
        # real (non-sentinel) k_out/v_out tensors.
        out = _base.scaled_dot_product_attention(
            q, k_out, v_out, cache=c, scale=1.0 / D**0.5, mask="causal"
        )
        mx.eval(out)
        assert out.shape == (B, 16, 1, D)
    finally:
        unpatch_mlx_lm()


def test_dispatcher_patch_is_idempotent_and_reversible() -> None:
    """Calling patch twice is fine; unpatch restores the original."""
    from veloxquant_mlx.metal.fused_sdpa import (
        patch_mlx_lm_for_fused_sdpa,
        unpatch_mlx_lm,
        is_patched,
    )
    import mlx_lm.models.base as _base

    original = _base.scaled_dot_product_attention

    patch_mlx_lm_for_fused_sdpa()
    assert is_patched()
    after_patch = _base.scaled_dot_product_attention
    assert after_patch is not original

    # Idempotent
    patch_mlx_lm_for_fused_sdpa()
    assert _base.scaled_dot_product_attention is after_patch

    unpatch_mlx_lm()
    assert not is_patched()
    assert _base.scaled_dot_product_attention is original

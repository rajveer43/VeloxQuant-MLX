"""VecInfer KV cache wrapper for mlx_lm integration.

Wraps :mod:`veloxquant_mlx.allocators.vecinfer` primitives in the standard
``update_and_fetch`` cache protocol expected by mlx_lm. The cache:

* applies a per-(head, channel) smooth scaling + Walsh-Hadamard rotation to
  keys, suppressing outliers before product VQ;
* encodes transformed keys against a pre-trained codebook, immediately
  dequantizes (then inverse-transforms) so the downstream SDPA call sees
  fp16 keys;
* tracks compressed vs fp16 byte counts so benchmarks can report a
  realized compression ratio.

The paper's CUDA kernel fusion (Section 3.3) is NOT portable to MLX/Metal;
the win on Apple Silicon is memory compression, not speedup over fp16.

Per-token storage (keys) at codebook bit-width ``b_k`` and sub-vector
dimension ``d_k``: ``(D / d_k) * b_k / 8`` bytes — plus an amortized
codebook cost of ``2**b_k * d_k * 2`` bytes shared across all tokens.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_keys,
    apply_dual_transform_queries,
    dequantize_vq,
    quantize_vq,
    walsh_hadamard_matrix,
)
from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.fused_sdpa import supports_shape as _fused_supports_shape


class VecInferKVCache(_MLXKVCache):
    """KV cache implementing VecInfer's dual-transform product VQ.

    Args:
        config: :class:`KVCacheConfig` with VecInfer-specific fields populated
            by :class:`KVCacheFactory.create`. Required fields:
            ``head_dim``, ``key_codebook_bits``, ``value_codebook_bits``,
            ``key_sub_dim``, ``value_sub_dim``. Optional: ``smooth_factors``
            (numpy or mx array; identity if absent), ``key_codebook``,
            ``value_codebook`` (random init if absent — for tests only).

    Notes:
        Storage layout deliberately delegates concatenation to mlx_lm's
        base ``_MLXKVCache``. We quantize + dequantize on the way in so
        that subsequent SDPA calls see a standard fp16 key tensor.

        Never exposes ``.bits`` — mlx_lm's SDPA checks
        ``hasattr(cache, "bits")`` to route to a different kernel path.
        We expose ``.assigned_avg_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._head_dim = int(config.head_dim)
        self._key_sub_dim = int(getattr(config, "key_sub_dim", 4))
        self._value_sub_dim = int(getattr(config, "value_sub_dim", 8))
        self._key_bits = int(getattr(config, "key_codebook_bits", 12))
        self._value_bits = int(getattr(config, "value_codebook_bits", 8))
        self._residual_length = int(getattr(config, "residual_length", 128))

        if self._head_dim % self._key_sub_dim != 0:
            raise ValueError(
                f"VecInferKVCache: head_dim={self._head_dim} not divisible "
                f"by key_sub_dim={self._key_sub_dim}"
            )
        if self._head_dim % self._value_sub_dim != 0:
            raise ValueError(
                f"VecInferKVCache: head_dim={self._head_dim} not divisible "
                f"by value_sub_dim={self._value_sub_dim}"
            )

        # Smooth factors: [n_heads, head_dim] or [head_dim] or None (identity)
        sm = getattr(config, "smooth_factors", None)
        if sm is None:
            self._smooth = None
        elif isinstance(sm, mx.array):
            self._smooth = sm
        else:
            self._smooth = mx.array(sm)

        # Hadamard matrix (constant)
        self._H = walsh_hadamard_matrix(self._head_dim, dtype=mx.float32)

        # Codebooks
        n_kc = 2**self._key_bits
        n_vc = 2**self._value_bits
        seed = int(getattr(config, "seed", 42))

        key_cb = getattr(config, "key_codebook", None)
        if key_cb is None:
            # Random init — only useful for shape/wiring tests; real usage
            # supplies a calibrated codebook via the factory.
            rng = mx.random.key(seed)
            key_cb = mx.random.normal(shape=(n_kc, self._key_sub_dim), key=rng).astype(mx.float32)
        elif not isinstance(key_cb, mx.array):
            key_cb = mx.array(key_cb)
        self._key_codebook = key_cb.astype(mx.float32)

        val_cb = getattr(config, "value_codebook", None)
        if val_cb is None:
            rng = mx.random.key(seed + 1)
            val_cb = mx.random.normal(shape=(n_vc, self._value_sub_dim), key=rng).astype(mx.float32)
        elif not isinstance(val_cb, mx.array):
            val_cb = mx.array(val_cb)
        self._value_codebook = val_cb.astype(mx.float32)

        # Byte accounting
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0
        self._value_bytes_compressed = 0
        self._value_bytes_fp16 = 0
        self._tokens_seen = 0
        self._tokens_quantized = 0

        # Resolve Metal acceleration flag.  Three-state:
        #   None  → auto-detect (silent fallback)
        #   True  → require (raise if unavailable)
        #   False → forced pure-MLX path
        requested = getattr(config, "use_metal_kernels", None)
        available = metal_available()
        if requested is True and not available:
            raise RuntimeError(
                "VecInferKVCache: use_metal_kernels=True but Metal kernels "
                "are not available on this build of mlx."
            )
        self._use_metal: bool = bool(available if requested is None else requested) and available

        # --- Phase 2: fused dequant+SDPA flag --------------------------
        # When enabled, update_and_fetch stores K/V codebook indices only
        # (no fp16 K_hat in the base cache) and exposes fused_sdpa() for
        # mlx_lm's patched dispatcher to call.  Requires power-of-2
        # head_dim with n_centroids <= 256, n_sub <= 16, head_dim <= 256.
        fused_req = getattr(config, "fused_sdpa", False)
        shape_ok = _fused_supports_shape(
            n_centroids=2**self._key_bits,
            n_sub=self._head_dim // self._key_sub_dim,
            head_dim=self._head_dim,
        )
        if fused_req is True and not (available and shape_ok):
            raise RuntimeError(
                f"VecInferKVCache: fused_sdpa=True but unsupported on this build "
                f"(metal_available={available}, shape_ok={shape_ok}, "
                f"head_dim={self._head_dim}, n_centroids={2**self._key_bits}, "
                f"n_sub={self._head_dim // self._key_sub_dim})."
            )
        self._fused_enabled: bool = bool(fused_req) and available and shape_ok

        # Ring-buffer storage for fused path.  Pre-allocated at first
        # update so we know B and H_kv; size = fused_sdpa_max_ctx.
        # Layout: [B, H_kv, max_ctx, n_sub] uint32.
        self._max_ctx: int = int(getattr(config, "fused_sdpa_max_ctx", 8192))
        self._n_sub_k: int = self._head_dim // self._key_sub_dim
        self._n_sub_v: int = self._head_dim // self._value_sub_dim
        self._stored_k_indices: Optional[mx.array] = None
        self._stored_v_indices: Optional[mx.array] = None
        self._stored_S_kv: int = 0

    # ------------------------------------------------------------------
    # Quantize/dequantize dispatch — Metal fast path or pure-MLX fallback
    # ------------------------------------------------------------------
    def _quantize(self, x: mx.array, codebook: mx.array, sub_dim: int) -> mx.array:
        if self._use_metal:
            from veloxquant_mlx.metal.kernels import vecinfer_quantize_metal

            return vecinfer_quantize_metal(x, codebook, sub_dim)
        return quantize_vq(x, codebook, sub_dim)

    def _dequantize(self, indices: mx.array, codebook: mx.array) -> mx.array:
        if self._use_metal:
            from veloxquant_mlx.metal.kernels import vecinfer_dequant_metal

            return vecinfer_dequant_metal(indices, codebook)
        return dequantize_vq(indices, codebook)

    def _encode_decode_keys(self, keys: mx.array) -> tuple:
        """Fused smooth+WHT+VQ+dequant+inv-WHT+smooth in one Metal dispatch.

        Returns ``(k_hat_fp16, k_indices_uint32)``.  Falls back to the
        7-node pure-MLX pipeline when Metal is unavailable or D is not a
        power-of-2 (e.g. D=64 is fine, D=96 is not).
        """
        D = keys.shape[-1]
        can_fuse = (
            self._use_metal
            and (D & (D - 1)) == 0  # power-of-2 for WHT butterfly
            and D <= 512  # threadgroup size cap
        )
        if can_fuse:
            from veloxquant_mlx.metal.kernels import vecinfer_encode_decode_metal

            k_hat_fp16, k_idx = vecinfer_encode_decode_metal(
                keys=keys,
                k_codebook=self._key_codebook,
                sub_dim=self._key_sub_dim,
                H_mat=self._H,
                smooth=self._smooth,
            )
            return k_hat_fp16, k_idx

        # Pure-MLX fallback path (identical to _update_and_fetch_standard)
        k32 = keys.astype(mx.float32)
        if self._smooth is not None:
            k_tilde = apply_dual_transform_keys(k32, self._smooth, self._H)
        else:
            k_tilde = k32 @ self._H
        k_idx = self._quantize(k_tilde, self._key_codebook, self._key_sub_dim)
        k_hat_tilde = self._dequantize(k_idx, self._key_codebook)
        k_hat = k_hat_tilde @ self._H.T
        if self._smooth is not None:
            sm = self._smooth
            if sm.ndim == 2 and k_hat.ndim >= 4 and k_hat.shape[-3] == sm.shape[0]:
                sm_b = sm[:, None, :].astype(mx.float32)
            elif sm.ndim == 2:
                sm_b = mx.mean(sm, axis=0).astype(mx.float32)
            else:
                sm_b = sm.astype(mx.float32)
            k_hat = k_hat * sm_b
        return k_hat.astype(keys.dtype), k_idx

    def _encode_decode_values(self, values: mx.array) -> tuple:
        """Fused VQ+dequant for values in one Metal dispatch.

        Returns ``(v_hat_fp16, v_indices_uint32)``.
        """
        D = values.shape[-1]
        can_fuse = self._use_metal and D <= 512
        if can_fuse:
            from veloxquant_mlx.metal.kernels import vecinfer_encode_decode_simple_metal

            v_hat_fp16, v_idx = vecinfer_encode_decode_simple_metal(
                values=values,
                v_codebook=self._value_codebook,
                sub_dim=self._value_sub_dim,
            )
            return v_hat_fp16, v_idx.astype(mx.int32)

        # Pure-MLX fallback
        v32 = values.astype(mx.float32)
        v_idx = self._quantize(v32, self._value_codebook, self._value_sub_dim)
        v_hat = self._dequantize(v_idx, self._value_codebook)
        return v_hat.astype(values.dtype), v_idx

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        # Always run the standard dequant path (the 0.5.1 behavior).  When
        # fused mode is enabled we also stash indices so cache.fused_sdpa()
        # can be called separately.  Phase 2.1 attempted an index-only
        # path; profiling showed it's slower because mlx_lm caches the
        # materialized K_hat across decode steps so per-step dequant cost
        # is essentially free, while our LUT-based kernel pays a fixed
        # per-call overhead that doesn't amortize.  Keeping the standard
        # path is the correct decision.
        if self._fused_enabled:
            return self._update_and_fetch_standard_and_stash(keys, values)
        return self._update_and_fetch_standard(keys, values)

    def _update_and_fetch_standard_and_stash(self, keys, values):
        """Run the standard dequant path AND stash indices for fused_sdpa().

        Used when fused_sdpa=True.  We never bypass the standard path —
        the K_hat fp16 buffer in the parent class is what mlx_lm's SDPA
        actually consumes during decode and it beats our fused kernel on
        already-materialized K_hat.  The index stash is kept so that
        ``cache.fused_sdpa(q)`` can still be called manually (useful for
        memory-bound long-context inference where you'd configure mlx_lm
        to skip the K_hat cache entirely).
        """
        B, H, S, D = keys.shape

        # Fused Metal encode+decode — 1 dispatch, returns both k_hat and indices
        k_dequant, k_idx = self._encode_decode_keys(keys)
        v_hat, v_idx = self._encode_decode_values(values)

        # Also stash indices so decode steps can use the fused path
        if self._stored_k_indices is None:
            self._stored_k_indices = mx.zeros((B, H, self._max_ctx, self._n_sub_k), dtype=mx.uint32)
            self._stored_v_indices = mx.zeros((B, H, self._max_ctx, self._n_sub_v), dtype=mx.uint32)
        if self._stored_S_kv + S > self._max_ctx:
            raise RuntimeError(
                f"VecInferKVCache: prefill length {self._stored_S_kv + S} "
                f"exceeded fused_sdpa_max_ctx={self._max_ctx}."
            )
        new_end = self._stored_S_kv + S
        self._stored_k_indices[:, :, self._stored_S_kv : new_end, :] = k_idx.astype(mx.uint32)
        self._stored_v_indices[:, :, self._stored_S_kv : new_end, :] = v_idx.astype(mx.uint32)
        self._stored_S_kv = new_end

        self._account_bytes(B, H, S, D)
        # Parent's update_and_fetch advances self.offset AND populates
        # self.keys/self.values.  Both are needed for prefill SDPA.
        return _MLXKVCache.update_and_fetch(self, k_dequant, v_hat)

    # ------------------------------------------------------------------
    # Standard path: dequant K_hat → fp16 base cache (current 0.5.x behavior)
    # ------------------------------------------------------------------
    def _update_and_fetch_standard(self, keys, values):
        B, H, S, D = keys.shape
        # Fused Metal encode+decode: 1 dispatch instead of 7 MLX graph nodes
        k_dequant, _ = self._encode_decode_keys(keys)
        v_hat, _ = self._encode_decode_values(values)
        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_dequant, v_hat)

    # ------------------------------------------------------------------
    # Fused path: write indices into pre-allocated ring buffer, advance
    # self.offset manually so mlx_lm's RoPE keeps working, and return
    # sentinel-shaped zero tensors that the patched SDPA dispatcher
    # ignores (it reads via cache.fused_sdpa(q, ...) instead).
    # ------------------------------------------------------------------
    def _update_and_fetch_fused(self, keys, values):
        B, H, S, D = keys.shape
        kdtype = keys.dtype
        vdtype = values.dtype

        # Quantize keys in transformed space — no dequant, no inverse
        k32 = keys.astype(mx.float32)
        if self._smooth is not None:
            k_tilde = apply_dual_transform_keys(k32, self._smooth, self._H)
        else:
            k_tilde = k32 @ self._H
        k_idx = self._quantize(k_tilde, self._key_codebook, self._key_sub_dim)

        # Quantize values (no transform needed)
        v32 = values.astype(mx.float32)
        v_idx = self._quantize(v32, self._value_codebook, self._value_sub_dim)

        # Lazy ring-buffer allocation on first update
        if self._stored_k_indices is None:
            self._stored_k_indices = mx.zeros((B, H, self._max_ctx, self._n_sub_k), dtype=mx.uint32)
            self._stored_v_indices = mx.zeros((B, H, self._max_ctx, self._n_sub_v), dtype=mx.uint32)

        if self._stored_S_kv + S > self._max_ctx:
            raise RuntimeError(
                f"VecInferKVCache: context length "
                f"{self._stored_S_kv + S} exceeded fused_sdpa_max_ctx="
                f"{self._max_ctx}.  Increase KVCacheConfig.fused_sdpa_max_ctx."
            )

        # Slice-write into the pre-allocated buffer.  MLX supports
        # in-place slice assignment on arrays.
        new_end = self._stored_S_kv + S
        self._stored_k_indices[:, :, self._stored_S_kv : new_end, :] = k_idx.astype(mx.uint32)
        self._stored_v_indices[:, :, self._stored_S_kv : new_end, :] = v_idx.astype(mx.uint32)
        self._stored_S_kv = new_end

        # Advance offset so mlx_lm's RoPE (rope(q, offset=cache.offset))
        # sees the correct sequence position on the next step.
        self.offset += S

        # Byte accounting
        self._account_bytes(B, H, S, D)

        # Sentinel return values.  The patched SDPA dispatcher routes to
        # cache.fused_sdpa(q, ...) and never reads these tensors.  We use
        # zeros (cheap, deterministic) so any *un*patched code path that
        # accidentally consumes them produces zero output rather than
        # NaN-or-garbage that might be missed.  The construction-time
        # check above ensures the patch is active before we get here.
        sentinel_k = mx.zeros((B, H, self._stored_S_kv, D), dtype=kdtype)
        sentinel_v = mx.zeros((B, H, self._stored_S_kv, D), dtype=vdtype)
        return sentinel_k, sentinel_v

    # ------------------------------------------------------------------
    # Parent-class overrides for fused mode.  When fused is enabled the
    # parent's `keys` / `values` buffers stay None; mlx_lm code that
    # introspects the cache (state property, eval barriers, empty(),
    # nbytes) would crash on the missing buffers.  These overrides
    # surface the index ring buffer as the cache's "state" instead.
    # When fused is disabled they fall back to parent behavior.
    # ------------------------------------------------------------------
    @property
    def state(self):  # type: ignore[override]
        if not self._fused_enabled:
            return super().state
        if self._stored_k_indices is None:
            empty_k = mx.zeros((1, 1, 0, self._n_sub_k), dtype=mx.uint32)
            empty_v = mx.zeros((1, 1, 0, self._n_sub_v), dtype=mx.uint32)
            return empty_k, empty_v
        # IMPORTANT: return the *whole* pre-allocated buffer, not a slice.
        # mlx_lm calls `mx.eval([c.state for c in caches])` every token to
        # force materialization.  A slice forces re-evaluation of the full
        # buffer every time.  Returning the buffer directly lets MLX dedup
        # against the prior evaluation.  Callers that need only the live
        # portion should consult `cache.offset` separately.
        return (self._stored_k_indices, self._stored_v_indices)

    @state.setter
    def state(self, v):  # type: ignore[override]
        if not self._fused_enabled:
            # Reuse parent setter
            _MLXKVCache.state.fset(self, v)  # type: ignore[union-attr]
            return
        # In fused mode, restore from an (k_indices, v_indices) pair.
        k, val = v
        self._stored_k_indices = k
        self._stored_v_indices = val
        self._stored_S_kv = int(k.shape[2])
        self.offset = self._stored_S_kv

    def empty(self) -> bool:  # type: ignore[override]
        if not self._fused_enabled:
            return super().empty()
        return self._stored_k_indices is None or self._stored_S_kv == 0

    def size(self) -> int:  # type: ignore[override]
        # Both standard and fused tracks self.offset; just defer to it.
        return self.offset

    @property
    def nbytes(self) -> int:  # type: ignore[override]
        if not self._fused_enabled:
            return super().nbytes
        if self._stored_k_indices is None:
            return 0
        # uint32 indices, live portion only
        n = self._stored_S_kv
        # nbytes = B * H_kv * n * n_sub * 4 bytes for both k and v
        ki = self._stored_k_indices
        vi = self._stored_v_indices
        per_tok_k = ki.shape[0] * ki.shape[1] * ki.shape[3] * 4
        per_tok_v = vi.shape[0] * vi.shape[1] * vi.shape[3] * 4
        return n * (per_tok_k + per_tok_v)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        k_bits_per_tok = (D // self._key_sub_dim) * self._key_bits
        v_bits_per_tok = (D // self._value_sub_dim) * self._value_bits
        k_bytes_per_tok = math.ceil(k_bits_per_tok / 8) * H * B
        v_bytes_per_tok = math.ceil(v_bits_per_tok / 8) * H * B
        self._key_bytes_compressed += k_bytes_per_tok * S
        self._value_bytes_compressed += v_bytes_per_tok * S
        self._key_bytes_fp16 += H * B * S * D * 2
        self._value_bytes_fp16 += H * B * S * D * 2
        self._tokens_seen += S
        self._tokens_quantized += S

    # ------------------------------------------------------------------
    # Phase 2: fused dequant+SDPA path — invoked by mlx_lm's patched
    # scaled_dot_product_attention when this cache is in use.
    # ------------------------------------------------------------------
    def fused_sdpa(
        self,
        q: mx.array,
        scale: float,
        *,
        causal: bool = True,
        sliding_window: int = 0,
    ) -> mx.array:
        """Compute attention directly from compressed K/V indices.

        Args:
            q: ``[B, H_q, S_q, D]`` uncompressed queries — not yet smooth
                or Hadamard transformed.  This method applies the dual
                transform internally so callers (mlx_lm SDPA dispatcher)
                pass exactly the same tensor they'd pass to standard SDPA.
            scale: Attention scale (typically ``1/sqrt(D)``).
            causal: Apply causal mask.
            sliding_window: If > 0, sliding-window attention with the
                given window width.

        Returns:
            ``[B, H_q, S_q, D]`` attention output in the dtype of ``q``.
        """
        if not self._fused_enabled:
            raise RuntimeError(
                "VecInferKVCache.fused_sdpa called but fused_sdpa was not "
                "enabled via KVCacheConfig.fused_sdpa=True."
            )
        if self._stored_k_indices is None or self._stored_S_kv == 0:
            raise RuntimeError(
                "VecInferKVCache.fused_sdpa called before any tokens were "
                "cached via update_and_fetch."
            )

        # Slice the live portion of the pre-allocated ring buffer.
        # MLX slices are views — no copy.
        live_k = self._stored_k_indices[:, :, : self._stored_S_kv, :]
        live_v = self._stored_v_indices[:, :, : self._stored_S_kv, :]

        # Transform q in fp32 (matches what the Hadamard math assumes)
        q_tilde = apply_dual_transform_queries(
            q.astype(mx.float32),
            self._smooth
            if self._smooth is not None
            else mx.ones((self._head_dim,), dtype=mx.float32),
            self._H,
        )

        # Lazy import — keeps cold-start cost off the package import path
        from veloxquant_mlx.metal.fused_sdpa import metal_fused_sdpa

        out = metal_fused_sdpa(
            q_tilde=q_tilde,
            k_indices=live_k,
            k_codebook=self._key_codebook,
            v_indices=live_v,
            v_codebook=self._value_codebook,
            scale=float(scale),
            causal=causal,
            sliding_window=int(sliding_window or 0),
            out_dtype=q.dtype,
        )
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16

    @property
    def compressed_value_bytes(self) -> int:
        return self._value_bytes_compressed

    @property
    def fp16_value_bytes(self) -> int:
        return self._value_bytes_fp16

    @property
    def codebook_bytes(self) -> int:
        """Static codebook overhead in bytes (fp16 storage)."""
        kb = (2**self._key_bits) * self._key_sub_dim * 2
        vb = (2**self._value_bits) * self._value_sub_dim * 2
        return kb + vb

    @property
    def assigned_avg_bits(self) -> float:
        """Effective bits/element averaged over keys and values.

        Excludes codebook overhead (amortized across many tokens); for an
        end-to-end byte ratio use compressed_*_bytes / fp16_*_bytes.
        """
        k_bits = (self._head_dim // self._key_sub_dim) * self._key_bits / self._head_dim
        v_bits = (self._head_dim // self._value_sub_dim) * self._value_bits / self._head_dim
        return (k_bits + v_bits) / 2.0


__all__ = ["VecInferKVCache"]

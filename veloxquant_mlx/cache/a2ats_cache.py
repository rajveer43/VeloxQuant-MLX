"""A2ATS-adapted KV cache — windowed RoPE + query-aware retrieval VQ.

Inspired by "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu,
Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings,
aclanthology.org/2025.findings-acl.644). Documented as "A2ATS-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

**No query visible at cache level.** Like every other query-aware method in
this repo (AMC-adapted's ``amc_use_query_saliency``, H2O-adapted's key-as-query
proxy, SnapKV-adapted's prefill window), a cache wrapper's ``update_and_fetch``
only ever receives keys and values — the true decode-time query vector is not
part of the mlx_lm cache protocol. This port substitutes the incoming key
vector itself as a proxy query for both the retrieval-set selection
(:func:`~veloxquant_mlx.quantizers.a2ats.a2ats_select_retrieval_set`) and the
query-aware codebook assignment
(:func:`~veloxquant_mlx.quantizers.a2ats.a2ats_query_aware_assignment`) — the
same category of approximation, not a new one.

**Retrieval set gets preferential codebook assignment, not eviction.** Every
token is quantized and retained; the retrieval-fraction split only changes
*which* centroid a token is matched against (query-aware vs. plain
nearest-centroid for the bulk). No token is ever dropped — a
compression-only method, not an eviction method (same family framing as
AMC-adapted).

**Windowed RoPE is applied at fetch time, every step.** The parent buffer
holds the *pre-RoPE* reconstruction; ``update_and_fetch`` re-applies windowed
RoPE to the whole accumulated cache against the current decode position on
each call, so a token's near/far class tracks the advancing query as Eq. (11)
requires. Baking the rotation in at write time would freeze that class at the
moment the token was written (:issue:`29`, finding 3).

**Far keys are returned unrotated** (Eq. 12, ``k̃_i = k_i``) — that is what
makes a shared codebook viable across inputs (§3.1). The constant ``R_b``
encoding "far" relative position belongs on the *query* side (Eq. 11); since
this cache never sees the query, it is exposed via :meth:`A2ATSKVCache.far_query_rope`
for callers that do.

**Offline codebook calibration required** — same footgun class as
VecInfer-adapted/CommVQ-adapted/Palu-adapted/SVDq-adapted/AMC-adapted: using
``a2ats`` with the default random-init codebook (present only so
wiring/shape tests don't require a calibration pass) degrades to
near-random quantization. Supply a calibrated ``a2ats_codebook`` via config
for real use.

**No CUDA kernel fusion reproduced** — same MLX/Metal disclaimer as every
other VQ-family method in this repo: the benefit on Apple Silicon is memory
footprint, not throughput.

Byte accounting:
    compressed_key_bytes / compressed_value_bytes — actual stored bytes
    fp16_key_bytes / fp16_value_bytes             — hypothetical full-rank cost
    compression_ratio                              — fp16 / compressed (K + V)
"""
from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.allocators.vecinfer import dequantize_vq
from veloxquant_mlx.quantizers.a2ats import (
    a2ats_cholesky_factor,
    a2ats_h_weighted_assignment,
    a2ats_query_aware_assignment,
    a2ats_select_retrieval_set,
)
from veloxquant_mlx.quantizers.a2ats_rope import (
    a2ats_apply_far_query_rope,
    a2ats_apply_windowed_rope,
)


class A2ATSKVCache(_MLXKVCache):
    """KV cache implementing A2ATS-adapted windowed RoPE + query-aware VQ.

    Args:
        config: :class:`~veloxquant_mlx.cache.base.KVCacheConfig`. Fields
            consumed:
                ``head_dim`` (int, required) — must be even (RoPE requirement)
                    and divisible by ``a2ats_sub_dim``.
                ``a2ats_codebook_bits`` (int, default 8) — codebook size 2^bits.
                ``a2ats_sub_dim`` (int, default 8) — VQ sub-vector width.
                ``a2ats_window`` (int, default 128) — trailing exact-RoPE
                    window ``w``.
                ``a2ats_b`` (int, default 2048) — constant far-token relative
                    position ``b`` (Eq. 11), independent of ``w``. The paper's
                    §5.1 uses ``w=64``, ``b=2048``. Consumed by
                    :meth:`far_query_rope`, which callers apply to the query.
                ``a2ats_use_query_aware`` (bool, default True) — paper's primary
                    reported path; off degrades to plain nearest-centroid VQ.
                ``a2ats_beta`` (float, default 0.5) — query/reconstruction blend,
                    must be in [0, 1].
                ``a2ats_retrieval_fraction`` (float, default 0.20) — fraction of
                    tokens routed to the query-aware assignment path, must be
                    in [0, 1].
                ``a2ats_rope_base`` (float, default 10000.0).
                ``a2ats_codebook`` (mx.array | np.ndarray | None) — pre-trained
                    codebook; random-init fallback if absent (wiring/tests only,
                    see module docstring's calibration warning).

    Notes:
        Never exposes ``.bits`` — keeps mlx_lm SDPA on the clean fp16 path,
        same convention as every other VQ-family cache in this repo.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._head_dim = int(config.head_dim)
        self._sub_dim = int(getattr(config, "a2ats_sub_dim", 8))
        self._bits = int(getattr(config, "a2ats_codebook_bits", 8))
        self._window = int(getattr(config, "a2ats_window", 128))
        self._b = int(getattr(config, "a2ats_b", 2048))
        self._use_query_aware = bool(getattr(config, "a2ats_use_query_aware", True))
        self._beta = float(getattr(config, "a2ats_beta", 0.5))
        self._retrieval_fraction = float(getattr(config, "a2ats_retrieval_fraction", 0.20))
        self._rope_base = float(getattr(config, "a2ats_rope_base", 10000.0))

        if self._head_dim % 2 != 0:
            raise ValueError(
                f"A2ATSKVCache: head_dim={self._head_dim} must be even "
                "(required by RoPE)."
            )
        if self._head_dim % self._sub_dim != 0:
            raise ValueError(
                f"A2ATSKVCache: head_dim={self._head_dim} not divisible by "
                f"a2ats_sub_dim={self._sub_dim}."
            )
        if not 0.0 <= self._beta <= 1.0:
            raise ValueError(
                f"A2ATSKVCache: a2ats_beta must be in [0, 1], got {self._beta}"
            )
        if not 0.0 <= self._retrieval_fraction <= 1.0:
            raise ValueError(
                "A2ATSKVCache: a2ats_retrieval_fraction must be in [0, 1], "
                f"got {self._retrieval_fraction}"
            )

        n_cb = 2 ** self._bits
        seed = int(getattr(config, "seed", 42))
        codebook = getattr(config, "a2ats_codebook", None)
        if codebook is None:
            # Random init — wiring/shape tests only; real usage supplies a
            # calibrated codebook (see module docstring).
            rng = mx.random.key(seed)
            codebook = mx.random.normal(shape=(n_cb, self._sub_dim), key=rng)
        elif not isinstance(codebook, mx.array):
            codebook = mx.array(codebook)
        self._codebook = codebook.astype(mx.float32)

        self._n_sub = self._head_dim // self._sub_dim

        # Paper Eq. (13)/(14): if a calibrated query second-moment matrix H is
        # supplied, retrieval-set tokens are assigned by the paper's actual
        # H-weighted objective (via the Eq. 15-18 Cholesky identity). Without
        # it, fall back to the cosine blend — a substitute, not Eq. (14); see
        # a2ats_query_aware_assignment's docstring (:issue:`29`, finding 4).
        query_h = getattr(config, "a2ats_query_h", None)
        if query_h is None:
            self._cholesky_factor = None
        else:
            if not isinstance(query_h, mx.array):
                query_h = mx.array(query_h)
            if query_h.shape != (self._sub_dim, self._sub_dim):
                raise ValueError(
                    "A2ATSKVCache: a2ats_query_h must have shape "
                    f"({self._sub_dim}, {self._sub_dim}) to match "
                    f"a2ats_sub_dim={self._sub_dim}, got {tuple(query_h.shape)}."
                )
            self._cholesky_factor = a2ats_cholesky_factor(query_h.astype(mx.float32))

        # Byte accounting
        self._key_bytes_compressed = 0
        self._value_bytes_compressed = 0
        self._key_bytes_fp16 = 0
        self._value_bytes_fp16 = 0
        self._tokens_seen = 0
        self._tokens_retrieved = 0

        # Absolute position tracking for windowed RoPE (prefill + decode).
        self._next_position = 0

    # ------------------------------------------------------------------
    # Core per-(batch, head) compression step
    # ------------------------------------------------------------------
    def _quantize_head(self, k_bh: mx.array, positions: mx.array) -> mx.array:
        """Compress + reconstruct one head's keys ``[S, D]``, **pre-RoPE**.

        ``positions`` are this head's absolute token positions (pre-RoPE
        keys are assumed — the cache never sees RoPE applied upstream in
        this repo's convention, matching CommVQ-adapted).

        Windowed RoPE is deliberately *not* applied here. It is applied in
        :meth:`update_and_fetch` to the whole accumulated cache against the
        current decode position, so that a token's near/far class tracks the
        advancing query as Eq. (11) requires — see that method's notes.
        """
        S, D = k_bh.shape

        if self._use_query_aware and S > 0:
            proxy_query = k_bh[-1]   # incoming key as query proxy (see module docstring)
            retrieval_idx, bulk_idx = a2ats_select_retrieval_set(
                k_bh, proxy_query, retrieval_fraction=self._retrieval_fraction
            )
            self._tokens_retrieved += int(retrieval_idx.shape[0])

            idx_parts = []
            for sub_i in range(self._n_sub):
                start = sub_i * self._sub_dim
                end = start + self._sub_dim
                sub = k_bh[:, start:end]

                sub_idx = mx.zeros((S,), dtype=mx.int32)
                if retrieval_idx.shape[0] > 0:
                    ret_sub = mx.take(sub, retrieval_idx, axis=0)
                    if self._cholesky_factor is not None:
                        # Paper Eq. (14), computed exactly via Eq. (17).
                        ret_assign = a2ats_h_weighted_assignment(
                            ret_sub, self._codebook, self._cholesky_factor
                        )
                    else:
                        # Cosine-blend substitute — no calibrated H available.
                        ret_assign = a2ats_query_aware_assignment(
                            ret_sub, self._codebook, proxy_query[start:end],
                            beta=self._beta,
                        )
                    sub_idx = _scatter_1d(sub_idx, retrieval_idx, ret_assign)
                if bulk_idx.shape[0] > 0:
                    bulk_assign = _nearest_centroid(
                        mx.take(sub, bulk_idx, axis=0), self._codebook
                    )
                    sub_idx = _scatter_1d(sub_idx, bulk_idx, bulk_assign)
                idx_parts.append(sub_idx)
            indices = mx.stack(idx_parts, axis=1)   # [S, n_sub]
        else:
            k_reshaped = k_bh.reshape(S, self._n_sub, self._sub_dim) if S > 0 else k_bh
            idx_parts = []
            for sub_i in range(self._n_sub):
                sub = k_bh[:, sub_i * self._sub_dim:(sub_i + 1) * self._sub_dim]
                idx_parts.append(_nearest_centroid(sub, self._codebook))
            indices = mx.stack(idx_parts, axis=1) if S > 0 else mx.zeros((0, self._n_sub), dtype=mx.int32)

        return dequantize_vq(indices, self._codebook).astype(mx.float16)  # [S, D], pre-RoPE

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize, store pre-RoPE, and return keys rotated against the
        *current* decode position.

        The parent :class:`mlx_lm.models.cache.KVCache` concatenates whatever
        it is handed and never revisits it. Writing rotated keys into it
        therefore freezes each token's near/far class at the moment it was
        written — a token classified "near" during prefill keeps its exact
        rotation forever, even thousands of steps later, which silently
        defeats the distance gating the method is named for (:issue:`29`,
        finding 3).

        So the parent buffer holds the **pre-RoPE** reconstruction, and
        windowed RoPE is re-applied to the full accumulated slice on every
        call, against ``query_position = self._next_position - 1``. Tokens
        cross from "near" to "far" as the decode position advances, matching
        Eq. (11), where ``i`` is the position of the decoding query.

        Cost: the rotation is ``O(total_tokens)`` per step rather than
        ``O(S)``. That is inherent to honest distance gating under this
        protocol — the near/far split is a function of the current query, so
        it cannot be precomputed. Far tokens are returned unrotated (Eq. 12),
        so only the ``window``-sized near slice does nontrivial work.
        """
        B, H, S, D = keys.shape
        positions = mx.arange(self._next_position, self._next_position + S)

        out_heads_k = []
        for b in range(B):
            per_head = []
            for h in range(H):
                per_head.append(self._quantize_head(keys[b, h], positions))
            out_heads_k.append(mx.stack(per_head, axis=0))
        k_out = mx.stack(out_heads_k, axis=0)   # [B, H, S, D]

        # Values: plain nearest-centroid VQ, no RoPE (values are never
        # position-rotated), no retrieval-set preferential assignment —
        # the paper's query-aware mechanism targets keys (retrieval
        # relevance is a key/query alignment concept); values follow the
        # simpler uniform path, same choice ZipCache-adapted/Palu-adapted
        # make for their "values follow the safer default" fields.
        v32 = values.astype(mx.float32)
        v_flat_shape = v32.shape
        v_reshaped = v32.reshape(-1, D)
        idx_parts = []
        for sub_i in range(self._n_sub):
            sub = v_reshaped[:, sub_i * self._sub_dim:(sub_i + 1) * self._sub_dim]
            idx_parts.append(_nearest_centroid(sub, self._codebook))
        v_indices = mx.stack(idx_parts, axis=1) if v_reshaped.shape[0] > 0 else mx.zeros((0, self._n_sub), dtype=mx.int32)
        v_hat = dequantize_vq(v_indices, self._codebook).astype(mx.float16)
        v_out = v_hat.reshape(v_flat_shape)

        self._account_bytes(B, H, S, D)
        self._next_position += S

        # Store the PRE-RoPE reconstruction in the parent buffer, then rotate
        # the whole accumulated slice below. See this method's docstring for
        # why the rotation cannot be baked in at write time.
        k_all, v_all = super().update_and_fetch(k_out, v_out)

        # Eq. (11)'s near/far split is relative to the *current* decoding
        # query i, which advances every step — so it must be recomputed here,
        # against the full cache, on every call.
        query_position = self._next_position - 1
        total = k_all.shape[2]
        all_positions = mx.arange(total)
        k_rot = a2ats_apply_windowed_rope(
            k_all.reshape(-1, D),
            mx.broadcast_to(all_positions[None, None, :], (B, H, total)).reshape(-1),
            query_position=query_position,
            window=self._window,
            base=self._rope_base,
        ).reshape(B, H, total, D)

        return k_rot, v_all

    def far_query_rope(self, query: mx.array) -> mx.array:
        """Apply the paper's constant far-token rotation ``R_b`` to a query.

        WRoPE splits into two halves: far **keys** stay unrotated (Eq. 12,
        handled in :meth:`update_and_fetch`), and the constant ``R_b`` that
        encodes "far" relative position rides on the **query** instead
        (Eq. 11, ``u_ij = q_i R_b k_j^T``). This method is the query-side
        half, using this cache's configured ``a2ats_b``.

        It is exposed rather than applied internally because
        ``update_and_fetch`` never receives the query — the mlx_lm cache
        protocol passes only keys and values (see the module docstring).
        A caller with real query access applies this before scoring far
        tokens; the proxy-query paths inside this cache cannot.
        """
        return a2ats_apply_far_query_rope(query, b=self._b, base=self._rope_base)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        bits_per_tok = self._n_sub * self._bits
        bytes_per_tok = math.ceil(bits_per_tok / 8) * H * B
        self._key_bytes_compressed += bytes_per_tok * S
        self._value_bytes_compressed += bytes_per_tok * S
        self._key_bytes_fp16 += H * B * S * D * 2
        self._value_bytes_fp16 += H * B * S * D * 2
        self._tokens_seen += S

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
        return (2 ** self._bits) * self._sub_dim * 2   # fp16 storage

    @property
    def compression_ratio(self) -> float:
        total_compressed = self._key_bytes_compressed + self._value_bytes_compressed
        total_fp16 = self._key_bytes_fp16 + self._value_bytes_fp16
        if total_compressed == 0:
            return 1.0
        return total_fp16 / total_compressed

    @property
    def assigned_avg_bits(self) -> float:
        return (self._n_sub * self._bits) / self._head_dim

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def tokens_retrieved(self) -> int:
        return self._tokens_retrieved


def _nearest_centroid(x: mx.array, codebook: mx.array) -> mx.array:
    """Plain nearest-centroid assignment (no query awareness). ``[N] int32``."""
    if x.shape[0] == 0:
        return mx.zeros((0,), dtype=mx.int32)
    diff = x.astype(mx.float32)[:, None, :] - codebook.astype(mx.float32)[None, :, :]
    d2 = mx.sum(diff * diff, axis=-1)
    return mx.argmin(d2, axis=-1).astype(mx.int32)


def _scatter_1d(base: mx.array, idx: mx.array, values: mx.array) -> mx.array:
    """Functional scatter: ``base`` with ``base[idx[i]] = values[i]``."""
    if idx.shape[0] == 0:
        return base
    return base.at[idx].add(values - mx.take(base, idx, axis=0))


__all__ = ["A2ATSKVCache"]

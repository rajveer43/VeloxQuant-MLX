"""xKV KV cache wrapper — cross-layer *shared-subspace* key compression.

Inspired by "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector
Extraction" (arXiv:2503.18893, preprint). Documented as "xKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port: fixed contiguous
grouping instead of CKA-validated grouping, no "Selective Reconstruction"
decode-time optimization, keys only (values fp16 throughout).

Per-layer roles (assigned at build time by ``pair_layers_grouped``):
    Every layer in a group publishes its own raw prefill keys to a shared
    ``XKVCoordinator``. Once every member of the group has published for the
    same token range, the coordinator computes one joint SVD over the
    *stacked* keys and returns a shared basis (V_g, K_mean_g). Every member —
    including the one whose publish triggered the computation — fetches and
    locally caches that identical basis, then projects its own keys into it
    for the rest of the generation. No further coordinator calls are made
    after the first successful fetch (decode-time projection is purely local).

Byte accounting:
    ``compressed_key_bytes``  — this layer's own latent codes only.
    ``shared_basis_bytes``    — the ``V_g``/``K_mean_g`` storage cost, reported
        as nonzero *only* by the group's leader (``member_idx == 0``) to avoid
        double-counting when a benchmark naively sums per-layer bytes across
        a model's layers. Followers report 0 here.
    ``fp16_key_bytes`` / ``value_fp16_bytes`` — always the uncompressed cost.

Degenerate case: with ``coordinator=None`` (or a group of size 1) the cache
behaves as standalone per-layer SVD compression — no basis sharing, useful for
unit-testing the projection/reconstruction path in isolation and for the
group-of-1 equivalence check against SVDq's mechanism.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator
from veloxquant_mlx.quantizers.xkv import (
    joint_svd_compress,
    project_into_shared_basis,
    quantize_latents_uniform,
    reconstruct_from_shared_basis,
)


class XKVCache(_MLXKVCache):
    """KV cache implementing xKV cross-layer shared-subspace key compression
    for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``head_dim``, ``xkv_rank``, ``xkv_energy_threshold``,
            ``xkv_latent_bits``, ``xkv_group_quant_size``.
        member_idx: This layer's 0-indexed position within its group
            (0 = leader by convention — the only layer that reports nonzero
            ``shared_basis_bytes``).
        group_id: Cross-layer group this layer belongs to.
        n_members: Number of layers in this group (the coordinator waits for
            all of them before computing the joint SVD).
        coordinator: Shared :class:`XKVCoordinator` (``None`` -> degenerate
            standalone per-layer SVD compression, equivalent to
            ``xkv_group_size=1``).
    """

    def __init__(
        self,
        config: Any,
        member_idx: int = 0,
        group_id: int = 0,
        n_members: int = 1,
        coordinator: Optional[XKVCoordinator] = None,
    ) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._member_idx = int(member_idx)
        self._group_id = int(group_id)
        self._n_members = max(1, int(n_members))
        self._coord: Optional[XKVCoordinator] = coordinator if self._n_members > 1 else None

        self._rank: Optional[int] = getattr(config, "xkv_rank", None)
        self._energy_threshold: float = float(getattr(config, "xkv_energy_threshold", 0.95))
        self._latent_bits: int = int(getattr(config, "xkv_latent_bits", 4))
        self._group_quant_size: int = int(getattr(config, "xkv_group_quant_size", 32))

        # Shared-basis state — set once, on the first successful fetch/compute.
        self._V_g: Optional[mx.array] = None  # [D, r] fp32
        self._K_mean_g: Optional[mx.array] = None  # [D] fp32
        self._singular_values: Optional[mx.array] = None  # [r] fp32
        self._r: int = 0
        self._token_offset: int = 0
        self._basis_token_start: Optional[int] = None

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._shared_basis_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0
        self._tokens_seen: int = 0

    # ------------------------------------------------------------------
    # Basis acquisition
    # ------------------------------------------------------------------
    def _standalone_basis(self, k0: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        V_g, K_mean_g, s_g = joint_svd_compress(
            [k0], rank=self._rank, energy_threshold=self._energy_threshold
        )
        mx.eval(V_g, K_mean_g, s_g)
        return V_g, K_mean_g, s_g

    def _try_acquire_shared_basis(self) -> bool:
        """Poll the coordinator for this group's shared basis (already
        published by a prior call). Returns True and caches it locally if
        available. Never publishes — publishing only happens in
        _acquire_basis on this layer's own prefill call."""
        basis = self._coord.get_shared_basis(
            self._group_id,
            self._basis_token_start,
            self._n_members,
            rank=self._rank,
            energy_threshold=self._energy_threshold,
        )
        if basis is None:
            return False
        V_g, K_mean_g, s_g = basis
        self._V_g, self._K_mean_g, self._singular_values = V_g, K_mean_g, s_g
        self._r = int(V_g.shape[1])
        self._account_basis_bytes(int(V_g.shape[0]), is_leader=(self._member_idx == 0))
        return True

    def _acquire_basis(self, keys: mx.array) -> None:
        """Obtain the shared (or standalone, if degenerate) basis for this
        layer. Sets self._V_g / self._K_mean_g / self._singular_values.

        The basis is anchored to this layer's *first* call's token range
        (``self._basis_token_start``) — every group member must publish for
        that same range before the joint SVD can run, so a member that
        hasn't yet adopted the shared basis keeps re-publishing its first
        keys at that fixed offset (not a moving one) on every subsequent
        call, until it observes the completed basis. Once adopted,
        ``self._V_g`` is set and this method is never called again.
        """
        B, H, S, D = keys.shape

        if self._coord is None:
            k0 = keys[0, 0].astype(mx.float32)
            V_g, K_mean_g, s_g = self._standalone_basis(k0)
            self._V_g, self._K_mean_g, self._singular_values = V_g, K_mean_g, s_g
            self._r = int(V_g.shape[1])
            self._account_basis_bytes(D, is_leader=True)
            return

        if self._basis_token_start is None:
            self._basis_token_start = self._token_offset
            k0 = keys[0, 0].astype(mx.float32)
            self._coord.publish_member_keys(
                self._group_id, self._member_idx, self._basis_token_start, S, k0
            )
        self._try_acquire_shared_basis()

    def _project_quantize_reconstruct(self, keys: mx.array) -> mx.array:
        B, H, S, D = keys.shape
        V_g, K_mean_g = self._V_g, self._K_mean_g

        out_heads = []
        for b in range(B):
            out_batch = []
            for h in range(H):
                k_bh = keys[b, h].astype(mx.float32)
                L = project_into_shared_basis(k_bh, V_g, K_mean_g)
                L_q = quantize_latents_uniform(
                    L, bits=self._latent_bits, group_size=self._group_quant_size
                )
                k_hat = reconstruct_from_shared_basis(L_q, V_g, K_mean_g)
                out_batch.append(k_hat)
            out_heads.append(mx.stack(out_batch, axis=0))
        return mx.stack(out_heads, axis=0)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape

        if self._V_g is None:
            # Publish this call's own keys (needed both so a still-forming
            # group's max_ctx guard sees every step, and so the joint SVD has
            # this member's latest data) and poll for the shared basis. Once
            # every member of the group has published for the *same*
            # token_start, the basis is computed and adopted by whichever
            # member observes completion first; every other member adopts it
            # on its own next call.
            self._acquire_basis(keys)

        if self._V_g is not None:
            k_out = self._project_quantize_reconstruct(keys)
        else:
            # Still no shared basis available — produce output from a private
            # one-shot basis for this call only (not stored, not accounted
            # against shared_basis_bytes since it is never adopted).
            k0 = keys[0, 0].astype(mx.float32)
            V_g, K_mean_g, s_g = self._standalone_basis(k0)
            k_out = self._project_with(keys, V_g, K_mean_g)
            r = int(V_g.shape[1])
            code_bytes = math.ceil(S * r * self._latent_bits / 8)
            n_groups = math.ceil(S / self._group_quant_size)
            param_bytes = n_groups * r * 2 * 2
            self._compressed_key_bytes += (code_bytes + param_bytes) * H * B
            self._fp16_key_bytes += B * H * S * D * 2
            self._value_fp16_bytes += B * H * S * D * 2
            self._tokens_seen += S
            self._token_offset += S
            return super().update_and_fetch(k_out, values)

        self._account_key_bytes(B, H, S, D)
        self._token_offset += S
        return super().update_and_fetch(k_out, values)

    def _project_with(self, keys: mx.array, V_g: mx.array, K_mean_g: mx.array) -> mx.array:
        B, H, S, D = keys.shape
        out_heads = []
        for b in range(B):
            out_batch = []
            for h in range(H):
                k_bh = keys[b, h].astype(mx.float32)
                L = project_into_shared_basis(k_bh, V_g, K_mean_g)
                L_q = quantize_latents_uniform(
                    L, bits=self._latent_bits, group_size=self._group_quant_size
                )
                k_hat = reconstruct_from_shared_basis(L_q, V_g, K_mean_g)
                out_batch.append(k_hat)
            out_heads.append(mx.stack(out_batch, axis=0))
        return mx.stack(out_heads, axis=0)

    # ------------------------------------------------------------------
    # Byte accounting
    # ------------------------------------------------------------------
    def _account_basis_bytes(self, D: int, is_leader: bool) -> None:
        if not is_leader or self._r == 0:
            return
        # V_g [D, r] + K_mean_g [D], fp32, charged once (amortized across the
        # group by convention: only the leader reports it).
        self._shared_basis_bytes += (D * self._r + D) * 4

    def _account_key_bytes(self, B: int, H: int, S: int, D: int) -> None:
        r = self._r if self._r > 0 else D
        code_bytes = math.ceil(S * r * self._latent_bits / 8)
        n_groups = math.ceil(S / self._group_quant_size)
        param_bytes = n_groups * r * 2 * 2  # scale + zero, fp16
        self._compressed_key_bytes += (code_bytes + param_bytes) * H * B
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2
        self._tokens_seen += S

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def member_idx(self) -> int:
        return self._member_idx

    @property
    def group_id(self) -> int:
        return self._group_id

    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def shared_basis_bytes(self) -> int:
        return self._shared_basis_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def value_fp16_bytes(self) -> int:
        return self._value_fp16_bytes

    @property
    def rank(self) -> int:
        return self._r

    @property
    def assigned_avg_bits(self) -> float:
        """Effective per-element key bit-width, including this layer's share
        of the amortized basis cost (leader only; followers report just their
        own latent-coding cost, matching the ``shared_basis_bytes`` convention)."""
        if self._fp16_key_bytes == 0:
            return float(self._latent_bits)
        total = self._compressed_key_bytes + self._shared_basis_bytes
        return 16.0 * total / self._fp16_key_bytes


__all__ = ["XKVCache"]

"""xKV KV cache wrapper — cross-layer *shared-subspace* key compression.

Inspired by "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector
Extraction" (arXiv:2503.18893, preprint). Documented as "xKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port: fixed contiguous
grouping instead of CKA-validated grouping, no "Selective Reconstruction"
decode-time optimization, keys only (values fp16 throughout).

Per-layer roles (assigned at build time by ``pair_layers_grouped``):
    Every layer in a group adds its own raw prefill keys to a shared
    ``XKVCoordinator``, which pools them (as running Gram statistics) and
    returns the joint basis (V_g, K_mean_g) fit on everything pooled so far.
    Each multi-row call (prefill chunk) refits it, so the basis is not frozen
    from a short first chunk (#892). A layer iterated before its peers has not
    yet seen their keys for the current chunk, so its basis lags theirs by one
    chunk; on single-token decode calls it adopts the latest pooled basis if
    the group has moved on, and otherwise keeps projecting through the one it
    holds.

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
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator
from veloxquant_mlx.quantizers.xkv import (
    JointGramStats,
    project_quantize_reconstruct_batched,
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
        coordinator: XKVCoordinator | None = None,
    ) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._member_idx = int(member_idx)
        self._group_id = int(group_id)
        self._n_members = max(1, int(n_members))
        self._coord: XKVCoordinator | None = coordinator if self._n_members > 1 else None

        self._rank: int | None = getattr(config, "xkv_rank", None)
        self._energy_threshold: float = float(getattr(config, "xkv_energy_threshold", 0.95))
        self._latent_bits: int = int(getattr(config, "xkv_latent_bits", 4))
        self._group_quant_size: int = int(getattr(config, "xkv_group_quant_size", 32))

        # Shared-basis state — set once, on the first successful fetch/compute.
        self._V_g: mx.array | None = None  # [D, r] fp32
        self._K_mean_g: mx.array | None = None  # [D] fp32
        self._singular_values: mx.array | None = None  # [r] fp32
        self._r: int = 0
        self._token_offset: int = 0
        # True until the first call carrying more than one row (#840).
        self._single_token_only: bool = True
        # Version of the pooled statistics the held basis was fit on; a later
        # coordinator version means a peer has added keys since (#892).
        self._basis_version: int = 0
        # Standalone (no coordinator) layers pool only their own keys.
        self._local_stats: JointGramStats | None = JointGramStats() if self._coord is None else None

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._shared_basis_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0
        self._tokens_seen: int = 0

    # ------------------------------------------------------------------
    # Basis acquisition
    # ------------------------------------------------------------------
    def _set_basis(self, basis: tuple[mx.array, mx.array, mx.array], D: int) -> None:
        self._V_g, self._K_mean_g, self._singular_values = basis
        self._r = int(self._V_g.shape[1])
        self._account_basis_bytes(D, is_leader=(self._member_idx == 0))

    def _refit_basis(self, keys: mx.array) -> None:
        """Add this call's keys to the pooled statistics and adopt the basis fit on everything pooled so far."""
        D = keys.shape[-1]
        k0 = keys[0, 0].astype(mx.float32)
        if self._coord is None:
            self._local_stats.add(k0)
            basis = self._local_stats.basis(
                rank=self._rank, energy_threshold=self._energy_threshold
            )
            mx.eval(*basis)
            self._set_basis(basis, D)
            return
        self._coord.add_member_keys(self._group_id, self._member_idx, k0)
        self._adopt_latest_basis(D)

    def _adopt_latest_basis(self, D: int) -> None:
        basis, version = self._coord.shared_basis(
            self._group_id, rank=self._rank, energy_threshold=self._energy_threshold
        )
        self._set_basis(basis, D)
        self._basis_version = version

    def _project_quantize_reconstruct(
        self, keys: mx.array, V_g: mx.array | None = None, K_mean_g: mx.array | None = None
    ) -> mx.array:
        """Project + quantize + reconstruct this layer's own keys against a
        shared (or ad-hoc, if explicitly passed) basis, batched across B*H in
        three vectorized ops instead of a Python loop per (b, h) — see
        :func:`project_quantize_reconstruct_batched`."""
        if V_g is None:
            V_g, K_mean_g = self._V_g, self._K_mean_g
        return project_quantize_reconstruct_batched(
            keys, V_g, K_mean_g, bits=self._latent_bits, group_size=self._group_quant_size
        )

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Refit the group's shared SVD basis on each multi-row call (prefill chunk), then project + quantize keys into it; values pass through fp16 unchanged."""
        B, H, S, D = keys.shape

        if self._V_g is None and S == 1 and self._single_token_only:
            # No call so far has carried more than one row, so a basis (the
            # shared one or the private one-shot below) would be fit on a
            # single row: rank-1, and frozen for the group (#840). Keep the row
            # lossless and charge it at fp16; the basis forms on the first
            # call with S > 1.
            raw = B * H * S * D * 2
            self._compressed_key_bytes += raw
            self._fp16_key_bytes += raw
            self._value_fp16_bytes += raw
            self._tokens_seen += S
            self._token_offset += S
            return super().update_and_fetch(keys, values)
        self._single_token_only = False

        if S > 1 or self._V_g is None:
            self._refit_basis(keys)
        elif self._coord is not None and self._coord.version(self._group_id) != self._basis_version:
            # Single-token decode: a peer has added keys since this layer's
            # basis was fit, so take the group's current basis once.
            self._adopt_latest_basis(D)

        k_out = self._project_quantize_reconstruct(keys)
        self._account_key_bytes(B, H, S, D)
        self._token_offset += S
        return super().update_and_fetch(k_out.astype(keys.dtype), values)

    def trim(self, n: int) -> int:
        """Drop the last ``n`` tokens and keep this cache's own state in step.

        The inherited ``trim`` only moves ``offset``.  Left alone, a cache
        trimmed to empty would project the next sequence through the previous
        sequence's shared SVD basis (#897).

        Trimming to empty resets the basis, the token counters, the byte
        accounting and this group's coordinator state (a new sequence starts
        from scratch).  A partial trim keeps the basis, which was fit on a
        surviving prefix, and scales the byte accounting with the token count
        (an estimate: per-group quantization parameters are not tracked per
        token).
        """
        old_tokens = self._token_offset
        trimmed = super().trim(n)
        if not trimmed:
            return trimmed
        if self.offset <= 0:
            self._V_g = None
            self._K_mean_g = None
            self._singular_values = None
            self._r = 0
            self._token_offset = 0
            self._single_token_only = True
            self._basis_version = 0
            if self._local_stats is not None:
                self._local_stats = JointGramStats()
            self._compressed_key_bytes = 0
            self._shared_basis_bytes = 0
            self._fp16_key_bytes = 0
            self._value_fp16_bytes = 0
            self._tokens_seen = 0
            if self._coord is not None:
                self._coord.reset()
            return trimmed
        self._token_offset = self.offset
        if old_tokens > 0:
            keep = self._token_offset / old_tokens
            self._compressed_key_bytes = int(self._compressed_key_bytes * keep)
            self._fp16_key_bytes = int(self._fp16_key_bytes * keep)
            self._value_fp16_bytes = int(self._value_fp16_bytes * keep)
            self._tokens_seen = int(self._tokens_seen * keep)
        return trimmed

    # ------------------------------------------------------------------
    # Byte accounting
    # ------------------------------------------------------------------
    def _account_basis_bytes(self, D: int, is_leader: bool) -> None:
        if not is_leader or self._r == 0:
            return
        # V_g [D, r] + K_mean_g [D], fp32, for the basis currently held
        # (amortized across the group by convention: only the leader reports
        # it). Refit per chunk, so this tracks the current rank rather than
        # accumulating.
        self._shared_basis_bytes = (D * self._r + D) * 4

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
        """This layer's 0-indexed position within its cross-layer group (0 = leader)."""
        return self._member_idx

    @property
    def group_id(self) -> int:
        """Cross-layer group this layer belongs to."""
        return self._group_id

    @property
    def compressed_key_bytes(self) -> int:
        """Realized stored bytes for this layer's own latent key codes (excludes the shared basis unless this is the group leader)."""
        return self._compressed_key_bytes

    @property
    def shared_basis_bytes(self) -> int:
        """Storage cost of the group's shared SVD basis; nonzero only for the group leader (``member_idx == 0``), to avoid double-counting across layers."""
        return self._shared_basis_bytes

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._fp16_key_bytes

    @property
    def value_fp16_bytes(self) -> int:
        """Actual value cost — values are stored fp16 throughout (keys-only compression)."""
        return self._value_fp16_bytes

    @property
    def rank(self) -> int:
        """Latent rank of the shared (or standalone) SVD basis."""
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

    # Without this, XKVCache inherits the base mlx_lm KVCache.merge()
    # classmethod unchanged, so hasattr(cache, "merge") is True and mlx_lm
    # treats this cache as batchable. Unlike some other #358 occurrences,
    # update_and_fetch here *does* populate the base class's self.keys via
    # super().update_and_fetch, so the inherited merge() does not crash on a
    # missing-keys AttributeError -- it succeeds silently, substituting a
    # plain BatchKVCache built from the reconstructed fp16 keys. That loses
    # not just this layer's own SVD-projection/quantization state but the
    # entire cross-layer XKVCoordinator relationship (member_idx, group_id,
    # the shared V_g/K_mean_g basis) -- every layer in the group is affected,
    # not just the one that got merged, since the coordinator's basis-sharing
    # protocol assumes all group members stay XKVCache instances for the
    # life of the request. See VeloxQuant-MLX#358; found verifying
    # VeloxQuant-Studio issue #34 (18th occurrence).
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError("XKVCache does not support merge() — see VeloxQuant-MLX#358")
        )
    )


__all__ = ["XKVCache"]

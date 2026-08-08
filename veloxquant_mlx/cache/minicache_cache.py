"""MiniCache KV cache wrapper — cross-layer depth-dimension merging.

Inspired by "MiniCache: KV Cache Compression in Depth Dimension for Large
Language Models" (Liu et al., NeurIPS 2024, arXiv:2405.14366). Documented as
"MiniCache-adapted (VeloxQuant-MLX implementation)" — faithful to the
SLERP-merge + token-retention core, adapted at the integration boundary via a
shared :class:`MiniCacheCoordinator` rather than a modified attention forward.

Per-layer roles (assigned at build time by ``pair_layers_depth``):
    Primary layer — stores its true KV to the coordinator and reconstructs
        itself losslessly (it is seen before its merge partner).
    Merge layer — fetches the paired primary's KV, performs the SLERP merge
        (shared direction + per-layer magnitudes, with a cosine-threshold
        retention set), and reconstructs *itself* from the merge. The pair now
        costs one direction tensor + two magnitude scalars per token.

Early layers (below ``minicache_start_frac`` of depth) are never merged — they
get a standalone primary role and behave as plain fp16 passthrough, matching
MiniCache's finding that adjacent-layer similarity only holds middle-to-deep.

Byte accounting (charged to the *merge* layer, which is where the win lands):
    Merged token: shared_dir is shared with the primary (counted once, on the
    primary side as it owns the segment) + this layer's per-token magnitude
    (one fp16 scalar). Retained token: a full fp16 vector.
    Primary layer: full fp16 (it is the reference) — but its direction is the
    one shared with the merge layer, so the *pair's* effective cost is ~1 layer.

Degenerate case: no coordinator (isolated layer) → behaves as a lossless fp16
passthrough primary, useful for unit-testing.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.minicache_coordinator import MiniCacheCoordinator
from veloxquant_mlx.quantizers.minicache import merge_pair, reconstruct_layer


class MiniCacheKVCache(_MLXKVCache):
    """KV cache implementing MiniCache cross-layer merging for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``minicache_retention_threshold`` (float, default 0.9),
            ``minicache_slerp_t``             (float, default 0.5).
        role: ``"primary"`` or ``"merge"`` (default ``"primary"``).
        group_id: Cross-layer merge group this layer belongs to.
        coordinator: Shared :class:`MiniCacheCoordinator` (None → degenerate).
        n_readers: Number of merge layers in this primary's group (group_size - 1).
            Only meaningful for ``role="primary"``; tells the coordinator how many
            fetches to expect before a published entry can be reclaimed.
    """

    def __init__(
        self,
        config: Any,
        role: str = "primary",
        group_id: int = 0,
        coordinator: Optional[MiniCacheCoordinator] = None,
        n_readers: int = 1,
    ) -> None:
        super().__init__()
        self._role = role if coordinator is not None else "primary"
        self._group_id = int(group_id)
        self._coord = coordinator
        self._n_readers = int(n_readers)
        self._ret = float(getattr(config, "minicache_retention_threshold", 0.9))
        self._t = float(getattr(config, "minicache_slerp_t", 0.5))

        self._token_offset = 0

        # Byte accounting
        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0
        self._n_retained = 0
        self._n_merged = 0
        self._n_retained_this_call = 0

    # ------------------------------------------------------------------
    def _merge_reconstruct(self, t_self: mx.array, t_primary: mx.array, is_key: bool) -> mx.array:
        """Merge this (merge-role) layer with the primary's tensor, per head.

        Reconstructs *this* layer from the shared-direction merge. Accumulates
        the merged/retained token accounting (key side only, to avoid double).
        """
        B, H, S, D = t_self.shape
        out_b = []
        n_ret = 0
        for b in range(B):
            out_h = []
            for h in range(H):
                res = merge_pair(
                    t_primary[b, h],
                    t_self[b, h],
                    retention_threshold=self._ret,
                    t=self._t,
                )
                out_h.append(reconstruct_layer(res, "merge"))
                if is_key:
                    n_ret += int(mx.sum(res.retained).item())
            out_b.append(mx.stack(out_h, axis=0))
        out = mx.stack(out_b, axis=0)
        if is_key:
            n_total = B * H * S
            self._n_retained += n_ret
            self._n_merged += n_total - n_ret
        return out

    def _account_merge(self, B: int, H: int, S: int, D: int) -> None:
        """Merge layer storage: per-token magnitude (fp16) + retained full vectors."""
        n_total = B * H * S
        # Magnitudes for all tokens (1 fp16 each) + full fp16 vectors for retained.
        # Use the running merged/retained split accumulated this step.
        # Approximate per-call share from the global ratio when needed.
        mag_bytes = n_total * 2
        retained_bytes = self._n_retained_this_call * D * 2
        self._compressed_key_bytes += mag_bytes + retained_bytes
        self._compressed_value_bytes += mag_bytes + retained_bytes
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    def _account_primary(self, B: int, H: int, S: int, D: int) -> None:
        """Primary layer stores the shared direction + its own magnitude.

        Direction [S, D] fp16 (shared with the merge partner — counted once here)
        + per-token magnitude. When there is no merge partner (early/degenerate
        layer) it is a plain fp16 reference.
        """
        if self._role == "primary" and self._coord is not None:
            dir_bytes = B * H * S * D * 2  # shared direction (counted once, here)
            mag_bytes = B * H * S * 2
            self._compressed_key_bytes += dir_bytes + mag_bytes
            self._compressed_value_bytes += dir_bytes + mag_bytes
        else:
            self._compressed_key_bytes += B * H * S * D * 2
            self._compressed_value_bytes += B * H * S * D * 2
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        tok_start = self._token_offset
        self._n_retained_this_call = 0

        if self._role == "merge" and self._coord is not None:
            seg = self._coord.fetch_primary(self._group_id, tok_start)
            if seg is None:
                # Primary not seen yet (mis-ordered) — fall back to lossless self.
                k_out, v_out = keys, values
                self._account_primary(B, H, S, D)
            else:
                ret_before = self._n_retained
                k_out = self._merge_reconstruct(keys, seg.keys, is_key=True)
                v_out = self._merge_reconstruct(values, seg.values, is_key=False)
                self._n_retained_this_call = self._n_retained - ret_before
                self._account_merge(B, H, S, D)
        else:
            # primary: store true KV for the merge partner, reconstruct losslessly
            if self._coord is not None:
                self._coord.publish_primary(
                    self._group_id, tok_start, S, keys, values, n_readers=self._n_readers
                )
            k_out, v_out = keys, values
            self._account_primary(B, H, S, D)

        self._token_offset += S
        return super().update_and_fetch(k_out, v_out)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def role(self) -> str:
        return self._role

    @property
    def group_id(self) -> int:
        return self._group_id

    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def n_retained(self) -> int:
        return self._n_retained

    @property
    def n_merged(self) -> int:
        return self._n_merged

    @property
    def retention_rate(self) -> float:
        total = self._n_retained + self._n_merged
        return self._n_retained / total if total else 0.0


__all__ = ["MiniCacheKVCache"]

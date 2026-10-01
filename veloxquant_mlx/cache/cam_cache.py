"""CaM-adapted KV cache — Cache Merging (merge evicted tokens instead of dropping).

Inspired by "CaM: Cache Merging for Memory-efficient LLMs Inference" (Zhang, Du,
Luo, Zhong, Zhang, Liu & Ji, ICML 2024, PMLR 235:58840-58850). Documented as
"CaM-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Every other eviction cache in the repo (SnapKV, StreamingLLM, H2O, TOVA,
PyramidKV, SqueezeAttention, ChunkKV) permanently **drops** the tokens it evicts.
CaM instead **merges** each evicted token into the surviving token it most
resembles (a cosine-similarity-weighted blend of the value rows, and optionally
the keys), then removes only the now-redundant slot — so the information is
folded into a neighbour rather than discarded. This is the first method on the
**merge-vs-drop** axis; the eviction *choice* is H2O's, only the disposition
differs.

Like H2O/TOVA/ChunkKV and unlike XQuant/MiniCache/SqueezeAttention, CaM needs
**no runtime coordinator** — every layer/head merges independently — so the
default ``KVCacheBuilder.for_model`` path (one ``CaMKVCache`` per layer) is all it
needs. With ``cam_merge="drop"`` the blend weight is zero and CaM reduces
**bit-for-bit** to H2O-adapted.

This is the eighth distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted     : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted        : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted       : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted  : H2O scoring with a fixed per-layer pyramid budget.
  - SqueezeAttention-adapted : H2O scoring with a data-driven per-layer budget.
  - ChunkKV-adapted    : H2O/key-norm scoring, evicted at CHUNK granularity.
  - CaM-adapted        : H2O scoring + eviction, but the loser is MERGED into a
    survivor (cosine-weighted) rather than dropped.

Merge gate (Eq. 14): the paper does not merge every over-budget loser
unconditionally — it first samples a Bernoulli mask whose probability scales
with the loser's accumulated attention mass relative to its merge target.
This is on by default (``cam_merge_gate=True``) and is load-bearing per the
paper's own ablation (Table 2: unconditional merging underperforms plain
eviction). Set ``cam_merge_gate=False`` to reproduce unconditional merging.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted) for both the importance score and
    the merge-target similarity.
  - Cosine-similarity merge *weight* rather than the paper's attention-
    prominence weight (which is ~0 for a just-appended token that overflows
    before it accumulates mass — the common case at the streaming eviction
    boundary). The merge *gate* still uses accumulated attention mass, per
    the paper.
  - Single nearest-survivor merge target (cosine nearest-neighbour), not the
    paper's local window of ``m`` contiguous tokens.
  - No RoPE position-ID remapping after merge.
  - Uniform budget across heads within a layer.

Byte accounting:
    cam_kept_bytes    — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / cam_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
    merge_mode        — this cache's merge disposition (diagnostic)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.quantizers.cam import cam_update_batched


class CaMKVCache(_MLXKVCache):
    """KV cache implementing CaM-adapted cache-merging eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``cam_budget`` (int, default 512)   — maximum tokens retained.
            ``cam_n_sink`` (int, default 4)     — leading positions never evicted.
            ``cam_merge`` (str, default "sim_weighted") — "sim_weighted" |
                "mean" | "drop"; "drop" reduces bit-for-bit to H2O-adapted.
            ``cam_merge_keys`` (bool, default False) — merge keys too (values are
                always merged).
            ``cam_merge_gate`` (bool, default True) — apply the paper's Eq. 14
                Bernoulli merge gate (whether to merge at all, not just how
                strongly). False unconditionally merges every over-budget loser
                (the paper's ablated "w.o. Merge Mask" configuration).
            ``seed`` (int) — base seed for the gate's deterministic draws.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        merge loop. Per-head state is lazily initialised on the first call.
        Because CaM merges (not drops) it always trims to exactly ``budget`` — the
        output is rectangular ``[B, H, budget, D]`` once past budget, so no
        cross-head alignment is needed.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token merge state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "cam_budget", 512))
        self._n_sink = int(getattr(config, "cam_n_sink", 4))
        self._merge_mode = str(getattr(config, "cam_merge", "sim_weighted"))
        self._merge_keys = bool(getattr(config, "cam_merge_keys", False))
        self._merge_gate = bool(getattr(config, "cam_merge_gate", True))
        self._seed = int(getattr(config, "seed", 0))

        self._head_dim: int = 0
        self._B: int = 0
        self._H: int = 0
        self._initialised: bool = False
        self._next_pos: int = 0
        self._draw_count: int = 0
        self._seeds: list[int] = []

        # Flat [BH, n, D] / [BH, n] state — replaces the old per-(b,h)
        # CaMState list. Batching every head into one call (instead of a
        # Python loop calling cam_update once per (b,h) pair) removes the
        # O(B*H) Python-dispatch bottleneck (issue #563), following the same
        # template as H2OKVCache/#504.
        self._bh_keys: mx.array | None = None
        self._bh_values: mx.array | None = None
        self._bh_scores: mx.array | None = None
        self._bh_positions: mx.array | None = None

        self._cam_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # True absolute token position, independent of how many rows survive
        # eviction/merge. Reported as ``self.offset`` so mlx_lm's RoPE stays
        # correct after tokens are dropped/merged (see #171-style handling
        # elsewhere).
        self._true_offset: int = 0

        # [B, n_kept] int32 true absolute position of each currently-stored
        # (head 0) row — see make_mask() and update_and_fetch()'s #370
        # deferred-eviction docstrings. None before the first update.
        self._kept_positions: mx.array | None = None

        # (K_out, V_out) actually returned by the last call — since #370's
        # deferred eviction, generally NOT the same as this call's full
        # (capped) retained state. A following S==0 no-op call must return
        # this unchanged.
        self._last_returned: tuple[mx.array, mx.array] | None = None

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily record shape and per-row gate seeds on first call."""
        if not self._initialised:
            self._B = B
            self._H = H
            self._head_dim = D
            # Distinct draw stream per (b,h) row, same derivation as the
            # per-head loop's ``seed=self._seed + head_idx``.
            self._seeds = [self._seed + head_idx for head_idx in range(B * H)]
            self._initialised = True

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply CaM merge-eviction, return retained window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` for THIS call's own attention — the full,
            un-evicted concatenation of whatever was stored before this call
            plus the ``S`` new tokens (see #370 below), NOT capped at
            ``cam_budget``. What gets *stored* afterward (visible to the
            next call) is capped as before.

        mlx_lm builds the attention mask for this call from hidden states —
        before q/k/v projections exist, let alone this cache's own
        ``update_and_fetch`` — so it is fixed (as either the "causal" string
        or an explicit array from ``make_mask``, called with only this
        call's query count ``N``) before eviction can possibly run. If this
        method shrank what it returns to fewer than ``N`` keys via merge-
        eviction, that already-fixed mask would silently desync from the
        shape it was built for (VeloxQuant-MLX#370). So eviction is
        deferred: this call returns the full pre-eviction concatenation
        (matching the mask ``make_mask`` already built from the previous
        call's true kept positions — see that method), and only the stored
        per-head states shrink, for the *next* call's ``make_mask`` to
        reflect correctly.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        if S == 0:
            if self._last_returned is not None:
                return self._last_returned
            return keys.astype(keys.dtype), values.astype(values.dtype)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16-equivalent accounting
        self._tokens_seen_total += B * H * S

        new_keys_flat = keys.reshape(B * H, S, D)
        new_values_flat = values.reshape(B * H, S, D)

        # This call's own (deferred) attention return gets the full
        # pre-eviction concatenation — captured before cam_update_batched
        # (below) evicts/merges anything. See #370.
        prev_keys_flat = self._bh_keys
        prev_values_flat = self._bh_values

        (
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_positions,
            self._next_pos,
            self._draw_count,
        ) = cam_update_batched(
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_positions,
            new_keys_flat,
            new_values_flat,
            self._n_sink,
            self._budget,
            self._merge_mode,
            self._merge_keys,
            self._merge_gate,
            self._seeds,
            self._next_pos,
            self._draw_count,
        )

        n_kept = self._bh_keys.shape[1]
        K_out = self._bh_keys.reshape(B, H, n_kept, D)  # [B, H, n_kept, D] — for STORAGE
        V_out = self._bh_values.reshape(B, H, n_kept, D)

        if prev_keys_flat is None:
            K_full = new_keys_flat.reshape(B, H, S, D)
            V_full = new_values_flat.reshape(B, H, S, D)
        else:
            n_full = prev_keys_flat.shape[1] + S
            K_full = mx.concatenate([prev_keys_flat, new_keys_flat], axis=1).reshape(
                B, H, n_full, D
            )
            V_full = mx.concatenate([prev_values_flat, new_values_flat], axis=1).reshape(
                B, H, n_full, D
            )

        # Byte accounting: bytes currently retained across all (b,h) rows.
        self._cam_kept_bytes = B * H * n_kept * D * 2 * 2

        # head-0 true kept positions per batch element, for the NEXT call's
        # make_mask (see that method) — not this call's own mask, already
        # fixed by the time we get here.
        self._kept_positions = self._bh_positions.reshape(B, H, n_kept)[:, 0, :]

        # K_out/V_out is the full retained state every call, not a delta —
        # reset so the base class's append-only buffer starts fresh instead
        # of stacking on top of the previous call's rows. Without this,
        # self.keys/self.values/self.offset stay at __init__ defaults
        # forever, and mlx_lm's generate() crashes on `cache.state` during
        # chunked prefill (see #83).
        self.keys = None
        self.values = None
        self.offset = 0
        super().update_and_fetch(K_out, V_out)
        self._true_offset += S
        self.offset = self._true_offset
        out = (K_full, V_full)
        self._last_returned = out
        return out

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Explicit position-based causal mask — see VeloxQuant-MLX#370.

        Called BEFORE this step's own ``update_and_fetch`` (and thus before
        this step's own eviction, which ``update_and_fetch`` defers past
        this step's return anyway — see its docstring). ``self._kept_positions``
        holds the true positions of whatever every head's state already
        stores from the *previous* call, which is exactly what
        ``update_and_fetch`` will concatenate its ``N`` new tokens onto —
        so a mask sized ``[B, 1, N, len(kept) + N]`` covers this call's
        actual returned key count precisely.
        """
        if self._kept_positions is None:
            return super().make_mask(N, return_array=return_array, window_size=window_size)
        B = self._kept_positions.shape[0]
        prev_positions = self._kept_positions
        new_positions = mx.arange(self.offset, self.offset + N, dtype=mx.int32)
        new_positions = mx.broadcast_to(new_positions[None, :], (B, N))
        key_positions = mx.concatenate([prev_positions, new_positions], axis=1)
        query_positions = new_positions
        return eviction_make_mask(
            query_positions, key_positions, N, return_array=return_array, window_size=window_size
        )

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def layer_budget(self) -> int:
        """This cache's per-layer token budget."""
        return self._budget

    @property
    def merge_mode(self) -> str:
        """This cache's merge disposition (diagnostic)."""
        return self._merge_mode

    @property
    def merge_gate(self) -> bool:
        """Whether the Eq. 14 Bernoulli merge gate is active (diagnostic)."""
        return self._merge_gate

    @property
    def cam_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._cam_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / cam_kept_bytes; > 1 means memory savings over fp16."""
        if self._cam_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._cam_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if self._bh_keys is None:
            return 0
        return int(self._bh_keys.shape[1])


__all__ = ["CaMKVCache"]

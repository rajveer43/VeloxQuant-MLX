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
from veloxquant_mlx.quantizers.cam import (
    CaMState,
    cam_fp16_bytes,
    cam_get_kv,
    cam_update,
    init_cam_state,
)


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
        self._states: list[CaMState] = []
        self._B: int = 0
        self._H: int = 0

        self._cam_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # True absolute token position, independent of how many rows survive
        # eviction/merge. Reported as ``self.offset`` so mlx_lm's RoPE stays
        # correct after tokens are dropped/merged (see #171-style handling
        # elsewhere).
        self._true_offset: int = 0

        # Per-(b,h) list of [n] int32 true absolute positions, parallel to
        # each head's CaMState.keys/values. None entries mean "nothing
        # stored yet for this head" (mirrors CaMState.keys is None).
        self._bh_positions: list[mx.array | None] = []

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
        """Lazily initialise per-head CaMState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_cam_state(
                    self._n_sink,
                    self._budget,
                    D,
                    merge_mode=self._merge_mode,
                    merge_keys=self._merge_keys,
                    merge_gate=self._merge_gate,
                    seed=self._seed + head_idx,  # distinct draw stream per head
                )
                for head_idx in range(B * H)
            ]
            self._bh_positions = [None for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

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
            return keys.astype(mx.float16), values.astype(mx.float16)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        new_positions = mx.arange(self._true_offset, self._true_offset + S, dtype=mx.int32)

        # Capture each head's pre-update stored K/V (for THIS call's own
        # deferred return) BEFORE running cam_update — see #370.
        previous_k = [cam_get_kv(st)[0] if st.keys is not None else None for st in self._states]
        previous_v = [cam_get_kv(st)[1] if st.keys is not None else None for st in self._states]

        k_out_b, v_out_b = [], []
        k_full_b, v_full_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            k_full_h, v_full_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st, pos = cam_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                    self._bh_positions[idx],
                    new_positions,
                )
                self._states[idx] = st
                self._bh_positions[idx] = pos
                k_h, v_h = cam_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
                new_k_bh = keys[b, h].astype(mx.float16)
                new_v_bh = values[b, h].astype(mx.float16)
                if previous_k[idx] is None:
                    k_full_h.append(new_k_bh)
                    v_full_h.append(new_v_bh)
                else:
                    k_full_h.append(mx.concatenate([previous_k[idx], new_k_bh], axis=0))
                    v_full_h.append(mx.concatenate([previous_v[idx], new_v_bh], axis=0))
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))
            k_full_b.append(mx.stack(k_full_h, axis=0))  # [H, n_full, D]
            v_full_b.append(mx.stack(v_full_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D] — for STORAGE
        V_out = mx.stack(v_out_b, axis=0)
        K_full = mx.stack(k_full_b, axis=0)  # [B, H, n_full, D] — this call's RETURN
        V_full = mx.stack(v_full_b, axis=0)

        # Byte accounting: sum across all head states.
        self._cam_kept_bytes = sum(cam_fp16_bytes(st) for st in self._states)

        # head-0 true kept positions per batch element, for the NEXT call's
        # make_mask (see that method) — not this call's own mask, already
        # fixed by the time we get here.
        head0_positions = [self._bh_positions[self._head_idx(b, 0)] for b in range(B)]
        self._kept_positions = mx.stack(head0_positions, axis=0)

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
        if not self._states or self._states[0].keys is None:
            return 0
        return int(self._states[0].keys.shape[0])


__all__ = ["CaMKVCache"]

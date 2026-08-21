"""RocketKV-adapted KV cache — two-stage compression (SnapKV eviction + HSA).

Inspired by "RocketKV: Accelerating Long-Context LLM Inference via Two-Stage
KV Cache Compression" (Behnam, Fu, Zhao, Tsai, Yu, Tumanov; NVIDIA/Georgia
Tech; ICML 2025, arXiv:2502.14051). Documented as "RocketKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port. See
``veloxquant_mlx/quantizers/rocketkv.py`` for the full list of adaptation
decisions and ``paper/research/surveys/NEW_METHOD_SURVEY_V23.md`` for the
write-up.

Design (paper §3.2-3.6):
    Prefill (first call, S > 1):
        1. Derive the stage-1/stage-2 split from ``rocketkv_compression_ratio``
           via :func:`~veloxquant_mlx.quantizers.rocketkv.split_compression_ratio`.
        2. Run stage-1 coarse-grain PERMANENT eviction — directly reuses
           :func:`~veloxquant_mlx.quantizers.snapkv.snapkv_compress` (the
           paper adopts SnapKV verbatim for this stage) — down to a
           stage-1 token budget derived from that split.
        3. Build a paged max/min key summary over the surviving subset
           (stage 2's index).

    Decode (subsequent calls, S == 1 per step):
        1. Append the new key's contribution to the paged summary
           incrementally (:func:`~veloxquant_mlx.quantizers.rocketkv.append_paged_summary`).
        2. Using the incoming key as a proxy query (same convention as
           SnapKV-adapted / A2ATS-adapted / every other query-aware method in
           this repo — the cache wrapper never sees the true query), run HSA:
           top-k1 head-dim channels, per-page approximate scores, top-k2 page
           selection.
        3. Gather the exact K/V rows for the selected pages *plus* the new
           decode token itself (always kept), and expose that subset as the
           cache's return value for this step — the eviction axis (stage 1)
           determines what is ever STORED; the selection axis (stage 2)
           determines what is FETCHED per step. Because mlx_lm's cache
           protocol has no notion of "fetch a subset, store everything," this
           wrapper stores only the stage-1 survivors (bounded, honest
           storage) and, at decode, narrows the *returned* K/V to the HSA
           selection — the base class's buffer keeps growing with new decode
           tokens (never evicted, matching RocketKV's paper: stage 1 only
           touches the *input* prompt).

Byte accounting:
    stage1_bytes / stage2_aux_bytes — storage after eviction (fp16 kept
        tokens) and HSA's paged max/min auxiliary summary, respectively
    full_fp16_bytes                 — hypothetical cost without any compression
    compression_ratio                — full_fp16_bytes / (stage1_bytes + stage2_aux_bytes)
    tokens_kept / tokens_total      — diagnostic token counters (stage 1 only —
        stage 2 never drops a stored token, only narrows what a given decode
        step attends to)

Limitations (stated plainly):
  - Key-as-query proxy at both stage-1 eviction and stage-2 HSA selection —
    inherited from SnapKV-adapted, not a new approximation.
  - No fused kernel: HSA gather/attend happens in eager MLX ops each step.
  - No RocketKV-MT (multi-turn) variant — see issue #239.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.rocketkv import (
    PagedKeySummary,
    append_paged_summary,
    build_paged_summary,
    hsa_approx_scores,
    hsa_fp32_bytes,
    select_topk_pages,
    split_compression_ratio,
    split_hsa_dims,
)
from veloxquant_mlx.quantizers.snapkv import full_fp16_bytes, snapkv_compress


class RocketKVKVCache(_MLXKVCache):
    """KV cache implementing RocketKV-adapted two-stage compression for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``rocketkv_compression_ratio`` (float, default 8.0) — overall
                target compression ratio ``c``; drives the adaptive stage-1/
                stage-2 split (paper §3.6). Must be > 1.
            ``rocketkv_page_size``   (Optional[int], default None) — HSA page
                size; ``None`` derives it from the adaptive split instead of
                a fixed value.
            ``rocketkv_head_topk1``  (Optional[int], default None) — head-dim
                channels kept per HSA step; ``None`` derives it from the
                adaptive split (``head_dim / head_dim_ratio``).
            ``rocketkv_obs_window``  (int, default 32) — stage-1 SnapKV
                observation window.
            ``rocketkv_n_sink``      (int, default 4) — stage-1 SnapKV sink
                tokens always kept.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Single-layer (no coordinator); ``for_model`` propagates all
        ``rocketkv_*`` fields automatically via ``dataclasses.replace``.
        Stage 1 (eviction) runs once at prefill, exactly like
        ``SnapKVKVCache``. Stage 2 (HSA) runs every decode step and narrows
        the *returned* K/V subset without shrinking the stored buffer —
        decode tokens accumulate in storage (paper's stage 1 only compresses
        the input prompt), but each attention call only touches the HSA-
        selected pages plus the running decode tail.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._compression_ratio = float(getattr(config, "rocketkv_compression_ratio", 8.0))
        self._page_size_cfg = getattr(config, "rocketkv_page_size", None)
        self._head_topk1_cfg = getattr(config, "rocketkv_head_topk1", None)
        self._obs_window = int(getattr(config, "rocketkv_obs_window", 32))
        self._n_sink = int(getattr(config, "rocketkv_n_sink", 4))

        stage1_ratio, stage2_ratio = split_compression_ratio(self._compression_ratio)
        self._stage1_ratio = stage1_ratio
        derived_page_size, self._head_dim_ratio = split_hsa_dims(stage2_ratio)
        self._page_size = (
            int(self._page_size_cfg) if self._page_size_cfg is not None else derived_page_size
        )

        self._head_dim: int = 0
        self._head_topk1: int = 0

        self._prefill_done = False
        self._summaries: list[list[PagedKeySummary]] = []  # [B][H]

        self._B = 0
        self._H = 0

        self._stage1_bytes = 0
        self._stage2_aux_bytes = 0
        self._full_fp16_bytes = 0
        self._tokens_kept = 0
        self._tokens_total = 0

        # True absolute position, mirroring SnapKVKVCache's offset split.
        self._in_base: bool = False
        self._row_offset: int = 0
        self._true_offset: int = 0

    @property
    def offset(self) -> int:
        """True absolute token position (NOT the retained row count).

        See ``SnapKVKVCache.offset`` for why the two must not be conflated —
        stage 1 here drops tokens exactly the same way.
        """
        return self._row_offset if self._in_base else self._true_offset

    @offset.setter
    def offset(self, value: int) -> None:
        self._row_offset = value

    # ------------------------------------------------------------------
    def _resolve_head_topk1(self, head_dim: int) -> int:
        if self._head_topk1_cfg is not None:
            return int(self._head_topk1_cfg)
        return max(1, int(round(head_dim / self._head_dim_ratio)))

    def _stage1_head(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array, int]:
        """Stage-1 eviction for one head's ``[S, D]`` K/V via SnapKV reuse."""
        S = int(keys.shape[0])
        budget = max(1, int(round(S / self._stage1_ratio)))
        state = snapkv_compress(
            keys, values, budget=budget, obs_window=self._obs_window, n_sink=self._n_sink
        )
        return state.kept_keys, state.kept_values, state.n_kept

    def _process_prefill(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        self._B, self._H, self._head_dim = B, H, D
        self._head_topk1 = self._resolve_head_topk1(D)

        k_out_b, v_out_b = [], []
        self._summaries = [[None] * H for _ in range(B)]  # type: ignore[list-item]
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                k_h, v_h, n_kept = self._stage1_head(keys[b, h], values[b, h])
                k_out_h.append(k_h)
                v_out_h.append(v_h)
                self._summaries[b][h] = build_paged_summary(k_h, self._page_size)

                self._stage1_bytes += n_kept * D * 2 * 2  # K + V, fp16
                self._stage2_aux_bytes += hsa_fp32_bytes(
                    self._summaries[b][h].page_max, self._summaries[b][h].page_min
                )
                self._full_fp16_bytes += full_fp16_bytes(S, D)
                self._tokens_kept += n_kept
                self._tokens_total += S
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))
        self._prefill_done = True
        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

    def _process_decode(self, keys: mx.array, values: mx.array):
        """Append decode tokens exactly; update HSA paged summaries incrementally.

        Decode tokens are always kept (stage 1 only compresses the input
        prompt); this method's job is bookkeeping so future HSA selection
        (via :meth:`select_indices`) has an up-to-date paged summary, not to
        narrow what gets stored.
        """
        B, H, S, D = keys.shape
        for b in range(B):
            for h in range(H):
                prior_n_pages = int(self._summaries[b][h].page_max.shape[0])
                self._summaries[b][h] = append_paged_summary(self._summaries[b][h], keys[b, h])
                new_n_pages = int(self._summaries[b][h].page_max.shape[0])
                # Only the newly created pages (if any) add auxiliary storage —
                # folding new tokens into an existing partial page is free.
                if new_n_pages > prior_n_pages:
                    self._stage2_aux_bytes += (new_n_pages - prior_n_pages) * D * 2 * 2

        fp16_cost = B * H * S * D * 2
        self._stage1_bytes += fp16_cost
        self._full_fp16_bytes += B * H * S * D * 2 * 2
        self._tokens_kept += B * H * S
        self._tokens_total += B * H * S
        return keys.astype(mx.float16), values.astype(mx.float16)

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        is_prefill = keys.shape[2] > 1
        if is_prefill:
            if not self._prefill_done:
                k_out, v_out = self._process_prefill(keys, values)
            else:
                # Chunked prefill of the same prompt: fold this chunk in as
                # if it were the tail of the original prefill, re-running
                # stage 1 over the accumulated kept set (SnapKVKVCache
                # convention) — RocketKV inherits this because stage 1 IS
                # SnapKV.
                B, H, S, D = keys.shape
                prev_kept = self._row_offset
                k_out_b, v_out_b = [], []
                for b in range(B):
                    k_out_h, v_out_h = [], []
                    for h in range(H):
                        prior_k = self.keys[b, h, :prev_kept, :]
                        prior_v = self.values[b, h, :prev_kept, :]
                        cat_k = mx.concatenate([prior_k, keys[b, h]], axis=0)
                        cat_v = mx.concatenate([prior_v, values[b, h]], axis=0)
                        k_h, v_h, n_kept = self._stage1_head(cat_k, cat_v)
                        k_out_h.append(k_h)
                        v_out_h.append(v_h)
                        self._summaries[b][h] = build_paged_summary(k_h, self._page_size)
                    k_out_b.append(mx.stack(k_out_h, axis=0))
                    v_out_b.append(mx.stack(v_out_h, axis=0))
                k_out = mx.stack(k_out_b, axis=0)
                v_out = mx.stack(v_out_b, axis=0)
                self._full_fp16_bytes += full_fp16_bytes(S, D) * B * H
                self._tokens_total += B * H * S
                self.offset = 0
                self.keys = None
                self.values = None
        else:
            k_out, v_out = self._process_decode(keys, values)

        self._true_offset += keys.shape[2]
        self._in_base = True
        try:
            return super().update_and_fetch(k_out, v_out)
        finally:
            self._in_base = False

    # ------------------------------------------------------------------
    def select_indices(self, query: mx.array, b: int, h: int, keep_recent: int = 0) -> mx.array:
        """HSA stage-2: approximate top-k page selection for one (b, h) query.

        Exposed for callers that want to actually exercise sparse attention
        with this cache (the base ``update_and_fetch`` return value stays
        dense — see the module docstring's storage-vs-fetch distinction).

        Args:
            query: ``[D]`` proxy query vector (key-as-query convention).
            b, h: Batch and head index into the stored paged summaries.
            keep_recent: Trailing token count always included regardless of
                HSA score (e.g. the newest decode tokens).

        Returns:
            ``[n_selected]`` int32 token indices, ascending, deduplicated,
            covering the HSA-selected pages and the trailing ``keep_recent``
            window.
        """
        from veloxquant_mlx.quantizers.rocketkv import gather_page_tokens

        summary = self._summaries[b][h]
        n_pages = int(summary.page_max.shape[0])
        k2 = max(1, int(round(n_pages / self._head_dim_ratio)))
        scores = hsa_approx_scores(query, summary, self._head_topk1)
        pages = select_topk_pages(scores, k2)
        selected = set(gather_page_tokens(pages, self._page_size, summary.n_tokens).tolist())
        if keep_recent > 0:
            selected.update(range(max(0, summary.n_tokens - keep_recent), summary.n_tokens))
        return mx.array(sorted(selected), dtype=mx.int32)

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal stage-1 kept-token accounting or stage-2 paged
        max/min summaries, silently desynchronizing both from the trimmed
        buffer on the next call.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def stage1_bytes(self) -> int:
        """Bytes stored for kept fp16 K + V rows after stage-1 eviction."""
        return self._stage1_bytes

    @property
    def stage2_aux_bytes(self) -> int:
        """Auxiliary bytes for HSA's paged max/min summaries (fp16-packed)."""
        return self._stage2_aux_bytes

    @property
    def full_fp16_bytes_total(self) -> int:
        """Hypothetical fp16 K + V cost without any compression."""
        return self._full_fp16_bytes

    @property
    def compression_ratio(self) -> float:
        """full_fp16_bytes / (stage1_bytes + stage2_aux_bytes); > 1 means savings."""
        total = self._stage1_bytes + self._stage2_aux_bytes
        if total == 0:
            return 1.0
        return self._full_fp16_bytes / total

    @property
    def tokens_kept(self) -> int:
        return self._tokens_kept

    @property
    def tokens_total(self) -> int:
        return self._tokens_total

    @property
    def keep_rate(self) -> float:
        if self._tokens_total == 0:
            return 1.0
        return self._tokens_kept / self._tokens_total

    @property
    def stage1_ratio(self) -> float:
        return self._stage1_ratio

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def head_topk1(self) -> int:
        return self._head_topk1


__all__ = ["RocketKVKVCache"]

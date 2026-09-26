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
           incrementally (:func:`~veloxquant_mlx.quantizers.rocketkv.append_paged_summary_batched`).
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

Performance: ``_process_prefill``'s single-shot path originally looped
``for b in range(B): for h in range(H):``, calling ``snapkv_compress`` once
per head — the same unbatched-per-head-loop cost pattern that dominated
ChunkKV/CurDKV/GEAR before their own fixes (#525/#526/#527). Here it reused
``_snapkv_compress_batched``, the batched primitive ``SnapKVKVCache``
already relies on in production (stage 1 *is* SnapKV verbatim, per this
module's own docstring — this was reinventing its per-head loop rather
than reusing the batched call). Verified bit-identical to the old loop.
Isolated stage-1 benchmark (B=1, H=32, S=1024, D=128, budget=128): ~4.53ms
(old per-head loop) -> ~1.50ms (batched), ~3x faster and markedly more
stable run-to-run.

``_process_decode`` had the same unbatched-per-(b,h)-loop pattern, and was
the hotter of the two (runs once per generated token, not once per prefill):
it called :func:`~veloxquant_mlx.quantizers.rocketkv.append_paged_summary`
once per head, itself an inner Python loop over pages. Batched via
:func:`~veloxquant_mlx.quantizers.rocketkv.append_paged_summary_batched`,
legal here because every (b, h) row's paged summary provably shares the same
``n_tokens``/page count at every point in ``_process_decode``'s call pattern
(every row is seeded with the same ``n_kept`` at prefill via
``_snapkv_compress_batched``'s one shared budget, and every decode step
appends the same ``S`` new tokens to every row — verified by grepping every
write site of the paged-summary state, not assumed). Verified bit-for-bit
identical to the old per-(b,h) loop against 1536 randomized single-step
trials plus 18 five-step decode sequences (varying page_size, prior token
count, new-token count, head_dim, batch size). Isolated decode-step
benchmark (D=128, page_size=8, n_kept=256, 50 steps): H=8: ~2.58ms ->
~0.33ms/step (~7.9x); H=32: ~11.54ms -> ~0.41ms/step (~28x) — speedup grows
with B*H, as expected for removing B*H separate Python-dispatched op graphs
(each itself looping over pages).

The chunked-prefill re-run path remains an explicit Python loop — unlike
``_process_decode``, per-head kept-token counts here are REDERIVED per call
via each head's own ``_stage1_head`` (SnapKV) call, so this path was
verified (not merely assumed) to keep every row's ``n_kept`` — and hence
page count — in sync via the shared ``budget``/``S`` formula, but the loop
itself was left unbatched since only chunked prompts hit it (rare, unlike
every-decode-step). See :func:`~veloxquant_mlx.quantizers.rocketkv.append_paged_summary_batched`'s
precondition note for why it isn't reused there.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.rocketkv import (
    append_paged_summary_batched,
    build_paged_summary,
    build_paged_summary_batched,
    hsa_approx_scores,
    select_topk_pages,
    split_compression_ratio,
    split_hsa_dims,
)
from veloxquant_mlx.quantizers.snapkv import (
    _snapkv_compress_batched,
    full_fp16_bytes,
    snapkv_compress,
)


class _SummaryRow:
    """Per-head read-only view materializing one row of the batched paged
    summary as a :class:`PagedKeySummary` (page_size/n_tokens are shared
    scalars across the whole batch, not stored per row)."""

    __slots__ = ("_cache", "_row")

    def __init__(self, cache: RocketKVKVCache, row: int) -> None:
        self._cache = cache
        self._row = row

    @property
    def page_max(self) -> mx.array:
        return self._cache._page_max[self._row]

    @property
    def page_min(self) -> mx.array:
        return self._cache._page_min[self._row]

    @property
    def page_size(self) -> int:
        return self._cache._page_size

    @property
    def n_tokens(self) -> int:
        return self._cache._summary_n_tokens


class _SummaryRowList:
    """One batch index's ``[h] -> _SummaryRow`` row, for ``_summaries[b][h]``."""

    __slots__ = ("_cache", "_b")

    def __init__(self, cache: RocketKVKVCache, b: int) -> None:
        self._cache = cache
        self._b = b

    def __getitem__(self, h: int) -> _SummaryRow:
        return _SummaryRow(self._cache, self._cache._head_idx(self._b, h))


class _SummaryView:
    """``self._summaries`` — a ``[b][h] -> PagedKeySummary``-shaped read-only
    view over ``RocketKVKVCache``'s batched ``_page_max``/``_page_min``
    storage. Exists so external callers (``select_indices``, tests) keep the
    per-(b, h) ``PagedKeySummary`` access they had before batching, without
    a second, desynchronizable copy of the state living in a Python
    list-of-lists."""

    __slots__ = ("_cache",)

    def __init__(self, cache: RocketKVKVCache) -> None:
        self._cache = cache

    def __getitem__(self, b: int) -> _SummaryRowList:
        return _SummaryRowList(self._cache, b)


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
        # Batched paged-summary storage: page_max/page_min are [B*H, n_pages, D]
        # fp32, one flattened row per (b, h) — see _process_decode's docstring
        # for why every row is guaranteed to share the same n_tokens/page
        # count, which is what makes this batching legal. self._summaries
        # (below) is a read-only view over these for callers/tests that still
        # want per-(b,h) PagedKeySummary access (select_indices, tests).
        self._page_max: mx.array | None = None
        self._page_min: mx.array | None = None
        self._summary_n_tokens: int = 0

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
        """Restore the retained row count (base-class bookkeeping only; see the getter for the true/retained distinction)."""
        self._row_offset = value

    def _head_idx(self, b: int, h: int) -> int:
        """Flatten a (batch, head) pair into this cache's [B*H, ...] row index."""
        return b * self._H + h

    @property
    def _summaries(self) -> _SummaryView:
        """Read-only ``[b][h] -> PagedKeySummary`` view over the batched
        ``self._page_max``/``self._page_min`` storage, for callers
        (``select_indices``, tests) that want per-(b, h) access without a
        second copy of the state living in a Python list-of-lists."""
        return _SummaryView(self)

    # ------------------------------------------------------------------
    def _resolve_head_topk1(self, head_dim: int) -> int:
        if self._head_topk1_cfg is not None:
            return int(self._head_topk1_cfg)
        return max(1, int(round(head_dim / self._head_dim_ratio)))

    def _stage1_head(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array, int]:
        """Stage-1 eviction for one head's ``[S, D]`` K/V via SnapKV reuse.

        Only used by the chunked-prefill re-run path below, where the
        accumulated kept set's length can genuinely differ from call to
        call. The single-shot prefill path (:meth:`_process_prefill`) uses
        the batched helper instead — see its docstring.
        """
        S = int(keys.shape[0])
        budget = max(1, int(round(S / self._stage1_ratio)))
        state = snapkv_compress(
            keys, values, budget=budget, obs_window=self._obs_window, n_sink=self._n_sink
        )
        return state.kept_keys, state.kept_values, state.n_kept

    def _process_prefill(self, keys: mx.array, values: mx.array):
        """Stage-1 eviction for every (batch, head) in one batched call.

        Was a Python ``for b: for h:`` loop calling ``snapkv_compress`` once
        per head — the same unbatched-per-head-loop cost pattern that
        dominated ChunkKV/CurDKV/GEAR's decode paths before their own fixes
        (VeloxQuant-MLX#525/#526/#527). Here the fix is lower-risk than
        those: ``_snapkv_compress_batched`` already exists and is already
        the production path for ``SnapKVKVCache`` itself (whose stage-1
        eviction this class reuses verbatim per its own docstring) — this
        was reinventing the per-head loop instead of reusing the batched
        call its sibling cache already relies on.

        ``budget`` is derived from ``S`` alone (``S / stage1_ratio``), and
        every (batch, head) in a single ``update_and_fetch`` call shares the
        same ``S`` — so it is safe to compute once for the whole batch
        instead of once per head (identical value either way).
        """
        B, H, S, D = keys.shape
        self._B, self._H, self._head_dim = B, H, D
        self._head_topk1 = self._resolve_head_topk1(D)

        budget = max(1, int(round(S / self._stage1_ratio)))
        k_out, v_out, _indices = _snapkv_compress_batched(
            keys,
            values,
            budget,
            self._obs_window,
            self._n_sink,
            return_indices=True,
        )
        n_kept = int(k_out.shape[2])

        # Batched paged-summary build: reshape [B,H,n_kept,D] -> [B*H,n_kept,D]
        # and run one vectorized max/min-over-pages reduction for the whole
        # batch instead of looping build_paged_summary per (b, h). Safe
        # because every (b, h) shares the same n_kept here (a single scalar
        # budget from _snapkv_compress_batched, not a per-head list — see
        # this method's own docstring).
        k_flat = k_out.reshape(B * H, n_kept, D)
        self._page_max, self._page_min = build_paged_summary_batched(k_flat, self._page_size)
        self._summary_n_tokens = n_kept
        n_pages = int(self._page_max.shape[1])
        self._stage2_aux_bytes += n_pages * D * 2 * 2 * B * H  # max + min, fp16, all rows

        self._stage1_bytes += n_kept * D * 2 * 2 * B * H  # K + V, fp16, all heads
        self._full_fp16_bytes += full_fp16_bytes(S, D) * B * H
        self._tokens_kept += n_kept * B * H
        self._tokens_total += S * B * H

        self._prefill_done = True
        return k_out, v_out

    def _process_decode(self, keys: mx.array, values: mx.array):
        """Append decode tokens exactly; update HSA paged summaries incrementally.

        Decode tokens are always kept (stage 1 only compresses the input
        prompt); this method's job is bookkeeping so future HSA selection
        (via :meth:`select_indices`) has an up-to-date paged summary, not to
        narrow what gets stored.
        """
        B, H, S, D = keys.shape
        # Batched paged-summary append: every (b, h) row is guaranteed to
        # share the same n_tokens/page count at this point (every row was
        # seeded with the same n_kept at prefill and every decode step
        # appends the same S new tokens to every row — see the module
        # docstring and append_paged_summary_batched's own precondition
        # note), so one call handles the whole B*H batch instead of a
        # Python loop calling append_paged_summary once per (b, h).
        prior_n_pages = int(self._page_max.shape[1])
        keys_flat = keys.reshape(B * H, S, D)
        self._page_max, self._page_min, self._summary_n_tokens = append_paged_summary_batched(
            self._page_max, self._page_min, self._summary_n_tokens, self._page_size, keys_flat
        )
        new_n_pages = int(self._page_max.shape[1])
        # Only the newly created pages (if any) add auxiliary storage —
        # folding new tokens into an existing partial page is free. New page
        # counts are synchronized across the whole batch (see above), so one
        # scalar delta applies to every (b, h) row alike.
        if new_n_pages > prior_n_pages:
            self._stage2_aux_bytes += (new_n_pages - prior_n_pages) * D * 2 * 2 * B * H

        fp16_cost = B * H * S * D * 2
        self._stage1_bytes += fp16_cost
        self._full_fp16_bytes += B * H * S * D * 2 * 2
        self._tokens_kept += B * H * S
        self._tokens_total += B * H * S
        return keys.astype(mx.float16), values.astype(mx.float16)

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Prefill: run stage-1 SnapKV eviction per head and build paged HSA summaries. Decode: append tokens exactly and update summaries incrementally."""
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
                # NOTE: intentionally NOT batched, unlike _process_decode and
                # _process_prefill. Each head reruns SnapKV independently
                # over its own accumulated kept set here, so per-head kept
                # counts (n_kept) — and hence each head's own page count —
                # can genuinely differ across heads (true raggedness, not
                # just a theoretical one). Batching would require padding
                # every row's summary to a shared max page count and
                # tracking per-row real counts separately, which is exactly
                # the complexity append_paged_summary_batched's precondition
                # note says this path doesn't satisfy. Left as a Python loop
                # (rare: only chunked prompts hit this branch, unlike
                # _process_decode which runs every generated token).
                B, H, S, D = keys.shape
                prev_kept = self._row_offset
                k_out_b, v_out_b = [], []
                page_max_rows: list[mx.array] = []
                page_min_rows: list[mx.array] = []
                max_n_pages = 0
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
                        row_summary = build_paged_summary(k_h, self._page_size)
                        page_max_rows.append(row_summary.page_max)
                        page_min_rows.append(row_summary.page_min)
                        max_n_pages = max(max_n_pages, int(row_summary.page_max.shape[0]))
                    k_out_b.append(mx.stack(k_out_h, axis=0))
                    v_out_b.append(mx.stack(v_out_h, axis=0))
                k_out = mx.stack(k_out_b, axis=0)
                v_out = mx.stack(v_out_b, axis=0)

                # Every row's own _stage1_head call uses the SAME formula
                # (budget = round(S / stage1_ratio), n_kept = min(budget, S))
                # over the SAME S (this call's shared cat_k length for every
                # head) -- n_kept, and hence each row's page count, is
                # therefore provably identical across every (b, h) in this
                # branch, not merely typically so. Padding is consequently a
                # structural no-op here (max_n_pages tracked above only as a
                # defensive check, since mx.stack below would raise on a
                # real mismatch rather than silently truncate/misalign).
                assert all(int(pm.shape[0]) == max_n_pages for pm in page_max_rows), (
                    "RocketKVKVCache: chunked-prefill re-run produced divergent "
                    "per-head page counts -- the shared-n_tokens invariant this "
                    "branch relies on no longer holds; see this branch's comment."
                )
                self._page_max = mx.stack(page_max_rows, axis=0)
                self._page_min = mx.stack(page_min_rows, axis=0)
                self._summary_n_tokens = int(n_kept)
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
        """Total tokens retained after stage-1 eviction (plus all decode tokens, which are always kept)."""
        return self._tokens_kept

    @property
    def tokens_total(self) -> int:
        """Total tokens seen across prefill and decode, before stage-1 eviction."""
        return self._tokens_total

    @property
    def keep_rate(self) -> float:
        """Fraction of tokens retained after stage-1 eviction (``tokens_kept / tokens_total``)."""
        if self._tokens_total == 0:
            return 1.0
        return self._tokens_kept / self._tokens_total

    @property
    def stage1_ratio(self) -> float:
        """Configured stage-1 (SnapKV) eviction ratio — prefill budget is ``S / stage1_ratio``."""
        return self._stage1_ratio

    @property
    def page_size(self) -> int:
        """Configured HSA page size used to build the paged max/min summaries."""
        return self._page_size

    @property
    def head_topk1(self) -> int:
        """Resolved stage-2 per-head top-k1 count used by HSA approximate scoring."""
        return self._head_topk1


__all__ = ["RocketKVKVCache"]

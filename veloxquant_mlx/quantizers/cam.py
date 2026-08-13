"""CaM-adapted eviction primitives — Cache Merging instead of dropping.

Inspired by "CaM: Cache Merging for Memory-efficient LLMs Inference" (Zhang, Du,
Luo, Zhong, Zhang, Liu & Ji, ICML 2024, PMLR 235:58840-58850). Documented as
"CaM-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Every other eviction configuration in the repo (SnapKV, StreamingLLM, H2O, TOVA,
PyramidKV, SqueezeAttention, ChunkKV) **permanently discards** the tokens it
evicts. CaM's insight is that cache eviction *invariably perturbs the output* —
the dropped token still carried mass. So instead of dropping the loser, CaM
**merges** its key/value into the surviving token it most resembles (a weighted
blend), then removes only the now-redundant slot. The information is compressed
into a neighbour rather than thrown away.

This module holds two things:
  1. ``most_similar_survivor`` / ``merge_pair`` — the pure merge machinery:
     pick the retained non-sink token whose key is closest (cosine) to the
     evicted one, and blend the two K/V rows by their cosine similarity.
  2. ``CaMState`` + ``cam_update`` — the per-head loop. It reuses H2O's
     key-as-query cumulative-attention-mass scorer and sink protection verbatim;
     the *only* change is the over-budget step, which merges the lowest-score
     non-sink token into a survivor (``merge`` modes) rather than dropping it.

Relationship to H2O:
  CaM-adapted reuses H2O's scorer, sink protection, and eviction *choice*
  verbatim — it evicts exactly the token H2O would. With ``merge_mode="drop"``
  the merge weight is zero, the survivor is left untouched, and the loser is
  simply removed, so CaM-adapted reduces **bit-for-bit** to H2O-adapted. This is
  the analogue of "``chunk_size=1`` == H2O" (ChunkKV) and "``strength=0`` == H2O"
  (SqueezeAttention), and is asserted by a dedicated equivalence test.

Why not attention-mass weighting:
  CaM's paper weights the merge by the discarded token's attention prominence.
  At the streaming eviction boundary the evicted token is frequently the token
  just appended (score 0, before it accumulates any mass), so an attention-mass
  weight would make the merge a no-op. We therefore weight by **key cosine
  similarity** — always meaningful, cache-observable, and faithful to CaM's
  intent (fold a token into the neighbour it most resembles). Documented, not a
  faithful port.

Merge modes:
  - ``"sim_weighted"`` (default) — blend by the cosine similarity between the
    evicted key and its survivor: ``w = clip(cos(k_e, k_a), 0, 1)`` and
    ``x_merged = (1-w)·x_a + w·x_e``. A token that closely resembles its
    survivor is folded in strongly; a dissimilar one barely perturbs it. This is
    always meaningful (unlike a pure attention-mass weighting, which is zero for a
    just-appended token that overflows before accumulating any mass — the common
    case at the streaming eviction boundary). The survivor inherits the summed
    attention-mass score.
  - ``"mean"`` — unweighted average of the two rows (ablation baseline); the
    survivor still inherits the summed score.
  - ``"drop"`` — no blend; reduces to H2O.

Values are always merged (CaM's core: value merging is what mitigates the output
perturbation). Keys are merged only when ``merge_keys=True``; by default the
survivor keeps its own key (merging keys shifts the attention geometry, which the
paper treats as optional). This is documented, not hidden.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (same as H2O-adapted): both the importance score and the
    merge-similarity are computed from the key vectors the cache holds, not the
    true query / attention maps the paper reads.
  - Single most-similar-survivor merge (no multi-target soft assignment / the
    paper's sampling over discarded locations); nearest neighbour by cosine.
  - No RoPE position-ID remapping after merge.
  - Uniform budget across heads within a layer.

Public API
----------
most_similar_survivor — index of the retained non-sink key closest to a given key
merge_pair            — blend two (k, v) rows by a merge mode + weights
CaMState              — immutable per-head eviction/merge state
init_cam_state        — construct empty state for a layer's budget
cam_update            — absorb S new tokens, merge lowest-score token if over budget
cam_get_kv            — extract current (keys, values) arrays
cam_fp16_bytes        — bytes stored in current state
full_cam_fp16_bytes   — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


def most_similar_survivor(
    evicted_key: mx.array,
    keys: mx.array,
    exclude_idx: int,
    n_sink_eff: int,
) -> int:
    """Index of the retained non-sink key most similar (cosine) to ``evicted_key``.

    The merge target is the surviving token whose key points most nearly in the
    same direction as the evicted token's key — the neighbour that can best absorb
    its mass. Sink positions (``[0, n_sink_eff)``) and the evicted slot itself are
    never chosen.

    Args:
        evicted_key: ``[D]`` key row of the token being evicted.
        keys:        ``[n, D]`` all currently stored key rows.
        exclude_idx: Index of the evicted token (never returned).
        n_sink_eff:  Number of leading sink positions (never returned).

    Returns:
        Index into ``keys`` of the merge target, or ``-1`` when there is no
        eligible survivor (all remaining tokens are sinks or the evicted slot).
    """
    n = int(keys.shape[0])
    k = keys.astype(mx.float32)
    e = evicted_key.astype(mx.float32)
    e_norm = e / (mx.sqrt(mx.sum(e * e)) + 1e-8)  # [D]
    row_norms = mx.sqrt(mx.sum(k * k, axis=-1)) + 1e-8  # [n]
    cos = (k @ e_norm) / row_norms  # [n]

    # Mask out sinks and the evicted slot with -inf so argmax skips them.
    neg_inf = mx.full((n,), float("-inf"), dtype=mx.float32)
    idx = mx.arange(n)
    eligible = (idx >= n_sink_eff) & (idx != exclude_idx)
    masked = mx.where(eligible, cos, neg_inf)

    if not bool(mx.any(eligible).item()):
        return -1
    return int(mx.argmax(masked).item())


def merge_pair(
    k_survivor: mx.array,
    v_survivor: mx.array,
    k_evicted: mx.array,
    v_evicted: mx.array,
    merge_mode: str,
    merge_keys: bool,
) -> tuple[mx.array, mx.array]:
    """Blend an evicted token's (k, v) into a survivor's, returning the new rows.

    The blend weight ``w`` is the share of the evicted token folded into the
    survivor: ``x_new = (1 - w)·x_survivor + w·x_evicted``.

    - ``"sim_weighted"``: ``w = clip(cos(k_evicted, k_survivor), 0, 1)`` — a
      similar loser is absorbed strongly, a dissimilar one barely perturbs the
      survivor. Always meaningful regardless of accumulated attention mass.
    - ``"mean"``: ``w = 0.5`` (unweighted average).
    - ``"drop"``: ``w = 0`` — survivor returned unchanged (reduces to H2O).

    Args:
        k_survivor, v_survivor: ``[D]`` survivor rows.
        k_evicted, v_evicted:   ``[D]`` evicted rows.
        merge_mode: ``"sim_weighted"`` | ``"mean"`` | ``"drop"``.
        merge_keys: If False (default), the survivor keeps its own key (values are
            always merged). If True, keys are blended by the same weight.

    Returns:
        ``(k_new, v_new)`` fp16 rows for the survivor after absorbing the loser.
    """
    if merge_mode == "drop":
        return k_survivor, v_survivor

    ks = k_survivor.astype(mx.float32)
    vs = v_survivor.astype(mx.float32)
    ke = k_evicted.astype(mx.float32)
    ve = v_evicted.astype(mx.float32)

    if merge_mode == "mean":
        w = 0.5
    else:  # sim_weighted
        denom = (mx.sqrt(mx.sum(ks * ks)) * mx.sqrt(mx.sum(ke * ke))) + 1e-8
        cos = float((mx.sum(ks * ke) / denom).item())
        w = min(max(cos, 0.0), 1.0)  # clip negatives → 0 (no anti-merge)

    v_new = ((1.0 - w) * vs + w * ve).astype(mx.float16)
    if merge_keys:
        k_new = ((1.0 - w) * ks + w * ke).astype(mx.float16)
    else:
        k_new = k_survivor
    return k_new, v_new


@dataclass
class CaMState:
    """Per-head CaM-adapted eviction/merge state for one layer.

    Identical fields to H2OState plus the merge configuration — CaM reuses H2O's
    cumulative-mass scorer; only the over-budget disposition (merge vs drop)
    differs.

    Attributes:
        keys:       [n_kept, D] fp16 stored key rows, or None before first update.
        values:     [n_kept, D] fp16 stored value rows, or None before first update.
        scores:     [n_kept] cumulative softmax attention mass (float32), or None.
        n_sink:     Number of leading sink positions — never evicted or merged.
        budget:     Maximum tokens to keep at any time (including sinks).
        merge_mode: ``"sim_weighted"`` | ``"mean"`` | ``"drop"`` (drop == H2O).
        merge_keys: Whether keys are merged too (values always are).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int
    merge_mode: str
    merge_keys: bool


def init_cam_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001
    merge_mode: str = "sim_weighted",
    merge_keys: bool = False,
) -> CaMState:
    """Create an empty CaMState before any tokens arrive.

    Args:
        n_sink:     Number of initial sink positions to protect.
        budget:     Maximum total tokens kept (sinks + non-sinks).
        head_dim:   Head dimension D (unused here; accepted for API symmetry).
        merge_mode: ``"sim_weighted"`` (default), ``"mean"``, or ``"drop"``.
        merge_keys: Merge keys as well as values (default False → values only).

    Raises:
        ValueError: if ``merge_mode`` is unknown, or if there are sink
            positions to protect but they leave no evictable/mergeable room
            within ``budget`` (``n_sink=0, budget=0`` remains a valid
            "disabled cache" configuration).
    """
    if merge_mode not in ("sim_weighted", "mean", "drop"):
        raise ValueError(
            f"init_cam_state: merge_mode must be 'sim_weighted', 'mean', or "
            f"'drop', got {merge_mode!r}."
        )
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"cam: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable/mergeable positions remain, so sinks would be "
            "merged away once the cache fills"
        )
    return CaMState(
        keys=None,
        values=None,
        scores=None,
        n_sink=n_sink,
        budget=budget,
        merge_mode=merge_mode,
        merge_keys=bool(merge_keys),
    )


def _attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights of query_proxy against each key row.

    Args:
        query_proxy: [D] — used as a stand-in for the true query.
        keys:        [n, D] — existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def cam_update(
    state: CaMState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> CaMState:
    """Absorb S new tokens, merging the lowest-score token into a survivor if over budget.

    For each of the S incoming tokens:
      1. Accumulate the new key's attention weight (as proxy query) over all stored
         keys into the per-token scores (exactly like H2O).
      2. Append the new token with score 0.
      3. If over budget: pick the lowest-score non-sink token (H2O's ``argmin`` with
         sinks masked to +inf); find its most-similar surviving non-sink neighbour;
         **merge** the loser's K/V into that neighbour by cosine similarity
         (``merge_pair``), transfer the loser's accumulated score to the neighbour,
         then remove the loser's slot. With ``merge_mode="drop"`` the neighbour is
         untouched and this is exactly H2O.

    Args:
        state:      Current CaMState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated CaMState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = CaMState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                scores=mx.ones((1,), dtype=mx.float32),
                n_sink=state.n_sink,
                budget=state.budget,
                merge_mode=state.merge_mode,
                merge_keys=state.merge_keys,
            )
            continue

        # --- score update (identical to H2O) -------------------------------
        attn = _attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score = 0) ----------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            # Identify the lowest-score non-sink token (H2O eviction choice).
            n_sink_eff = min(state.n_sink, n_total)
            if n_sink_eff > 0:
                inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([inf_block, scores_cat[n_sink_eff:]], axis=0)
            else:
                protected = scores_cat
            evict_idx = int(mx.argmin(protected).item())

            # Merge the loser into its most-similar survivor (unless drop mode).
            if state.merge_mode != "drop":
                tgt = most_similar_survivor(keys_cat[evict_idx], keys_cat, evict_idx, n_sink_eff)
                if tgt >= 0:
                    k_new, v_new = merge_pair(
                        keys_cat[tgt],
                        values_cat[tgt],
                        keys_cat[evict_idx],
                        values_cat[evict_idx],
                        state.merge_mode,
                        state.merge_keys,
                    )
                    # Write the merged rows back into the survivor slot.
                    keys_cat = mx.concatenate(
                        [keys_cat[:tgt], k_new[None], keys_cat[tgt + 1 :]], axis=0
                    )
                    values_cat = mx.concatenate(
                        [values_cat[:tgt], v_new[None], values_cat[tgt + 1 :]], axis=0
                    )
                    # Survivor inherits the loser's mass.
                    merged_score = scores_cat[tgt] + scores_cat[evict_idx]
                    scores_cat = mx.concatenate(
                        [scores_cat[:tgt], merged_score[None], scores_cat[tgt + 1 :]],
                        axis=0,
                    )

            # Remove the loser's slot.
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]
            scores_cat = scores_cat[keep_indices]

        state = CaMState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            merge_mode=state.merge_mode,
            merge_keys=state.merge_keys,
        )

    return state


def cam_get_kv(state: CaMState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def cam_fp16_bytes(state: CaMState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_cam_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "most_similar_survivor",
    "merge_pair",
    "CaMState",
    "init_cam_state",
    "cam_update",
    "cam_get_kv",
    "cam_fp16_bytes",
    "full_cam_fp16_bytes",
]

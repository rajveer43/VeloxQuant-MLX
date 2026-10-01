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

This module holds three things:
  1. ``most_similar_survivor`` / ``merge_pair`` — the pure merge machinery:
     pick the retained non-sink token whose key is closest (cosine) to the
     evicted one, and blend the two K/V rows by their cosine similarity.
  2. ``merge_gate_probability`` — the paper's Eq. 14 sampling gate: whether to
     merge at all, not just how strongly.
  3. ``CaMState`` + ``cam_update`` — the per-head loop. It reuses H2O's
     key-as-query cumulative-attention-mass scorer (via the shared
     ``_eviction_common.attention_scores`` helper) and sink protection
     verbatim; the *only* change is the over-budget step, which merges the
     lowest-score non-sink token into a survivor (``merge`` modes) rather than
     dropping it.

Merge gate (Eq. 14 / Algorithm 1 line 6):
  The paper does not merge unconditionally. It first samples a binary mask
  ``M = Bernoulli(clamp(Ā_i / avg(Ā_j:j+m), 0, 1))`` — the probability of
  merging scales with how much accumulated attention mass the loser (``i``)
  carries relative to its merge target(s) (``j:j+m``). A loser with little
  mass relative to its target is *not* merged (behaves like plain eviction);
  a loser with comparable or greater mass is merged with high probability.
  The paper's own ablation (Table 2, "w.o. Merge Mask": unconditional merging)
  shows this gate is load-bearing — removing it drops performance *below* the
  no-merge baseline, because indiscriminately merging low-signal losers can
  perturb the survivor more than dropping the loser would have (Theorem 3.3).
  Earlier revisions of this module always merged (gate probability implicitly
  1.0), which is exactly this ablated, non-recommended configuration. The gate
  is now implemented and on by default (``cam_merge_gate=True``); it can be
  disabled to reproduce the old unconditional-merge behaviour or to isolate the
  gate's effect, mirroring the paper's own ablation.

Relationship to H2O:
  CaM-adapted reuses H2O's scorer, sink protection, and eviction *choice*
  verbatim — it evicts exactly the token H2O would. With ``merge_mode="drop"``
  the merge weight is zero, the survivor is left untouched, and the loser is
  simply removed, so CaM-adapted reduces **bit-for-bit** to H2O-adapted. This is
  the analogue of "``chunk_size=1`` == H2O" (ChunkKV) and "``strength=0`` == H2O"
  (SqueezeAttention), and is asserted by a dedicated equivalence test.

Why not attention-mass weighting for the blend:
  CaM's paper weights the *blend* by the discarded token's attention
  prominence. At the streaming eviction boundary the evicted token is
  frequently the token just appended (score 0, before it accumulates any
  mass), so an attention-mass blend weight would make the merge a no-op. We
  therefore weight the blend by **key cosine similarity** — always
  meaningful, cache-observable, and faithful to CaM's intent (fold a token
  into the neighbour it most resembles). Documented, not a faithful port.
  Note this is a different quantity from the merge *gate* above: a
  just-appended, zero-score loser correctly gets gate probability ≈0 (drop
  it, it hasn't proven any importance yet) — that is the paper's intended
  behaviour, not the failure mode the cosine substitution was built to avoid.

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
  - Single most-similar-survivor merge target (nearest neighbour by cosine),
    not the paper's local window of ``m`` contiguous tokens (``j:j+m``); the
    gate's ``survivor_score`` is this one neighbour's score, standing in for
    the paper's ``avg(Ā_j:j+m)``.
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

from veloxquant_mlx.quantizers._eviction_common import (
    attention_scores,
    fp16_kv_bytes,
    full_fp16_kv_bytes,
    get_kv,
)


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
    k_new = ((1.0 - w) * ks + w * ke).astype(mx.float16) if merge_keys else k_survivor
    return k_new, v_new


def merge_gate_probability(evicted_score: float, survivor_score: float) -> float:
    """Probability of merging the loser at all — the paper's Eq. 14 gate.

    ``p = clamp(evicted_score / survivor_score, 0, 1)``. A loser with little
    accumulated attention mass relative to its merge target is unlikely to be
    merged (it behaves like plain eviction); a loser with comparable or
    greater mass is merged with high (up to certain) probability.

    Args:
        evicted_score: cumulative attention mass of the loser (``Ā_i``).
        survivor_score: cumulative attention mass of the merge target
            (``avg(Ā_j:j+m)`` in the paper; here the single nearest
            survivor's score, consistent with this module's single-target
            adaptation of the paper's local-window average).

    Returns:
        Merge probability in ``[0, 1]``.
    """
    if survivor_score <= 0.0:
        return 1.0 if evicted_score > 0.0 else 0.0
    return min(max(evicted_score / survivor_score, 0.0), 1.0)


def sample_merge_gate(probability: float, seed: int, draw_id: int) -> bool:
    """Deterministic Bernoulli draw for the merge gate, reproducible by (seed, draw_id).

    Mirrors the Keyformer-adapted module's seeded-Gumbel pattern: the same
    ``(seed, draw_id)`` always yields the same draw, so results are
    reproducible without threading RNG state through ``CaMState``.

    Args:
        probability: merge probability from ``merge_gate_probability``.
        seed: base seed (``KVCacheConfig.seed``).
        draw_id: monotonically increasing counter, unique per gate draw within
            a run (e.g. total tokens absorbed so far).

    Returns:
        True → proceed with the merge; False → drop the loser instead (no blend).
    """
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    key = mx.random.key(seed * 1_000_003 + draw_id)
    u = float(mx.random.uniform(low=0.0, high=1.0, key=key).item())
    return u < probability


def _draw_uniform_from_key(key: mx.array) -> mx.array:
    return mx.random.uniform(low=0.0, high=1.0, key=key)


def _sample_merge_gate_batched(probability: mx.array, seeds: list[int], draw_id: int) -> mx.array:
    """Batched-``[BH]`` equivalent of calling :func:`sample_merge_gate` once per row.

    Same ``mx.random.key``-has-no-batched-key-overload workaround as
    :func:`veloxquant_mlx.quantizers.keyformer._gumbel_at_batched`: one cheap
    Python-level ``mx.random.key`` call per row (only once per token *step*,
    not once per ``(head, step)`` pair), then one batched ``mx.vmap`` draw for
    all ``BH`` rows. Verified bit-for-bit equal to stacking
    :func:`sample_merge_gate`'s draws (same ``seed * 1_000_003 + draw_id`` key
    derivation, same comparison). The ``probability <= 0`` / ``>= 1`` shortcuts
    in the scalar version are folded into the comparison itself here (a draw
    is always made, but ``u < 0`` is always False and ``u < 1`` is always True
    for ``u`` drawn from ``[0, 1)``, so the result is identical).
    """
    keys = mx.stack([mx.random.key(s * 1_000_003 + draw_id) for s in seeds])
    u = mx.vmap(_draw_uniform_from_key)(keys)
    return u < probability


def _most_similar_survivor_batched(
    evicted_key: mx.array,  # [BH, D]
    keys: mx.array,  # [BH, n, D]
    evict_idx: mx.array,  # [BH, 1] int32
    n_sink_eff: int,
) -> mx.array:
    """Batched-``[BH]`` equivalent of calling :func:`most_similar_survivor` once per row.

    Returns ``[BH]`` int32 indices into ``keys``' ``n`` axis, or ``-1`` where
    no eligible survivor exists (mirrors the per-row scalar function).
    """
    bh, n, _ = keys.shape
    k = keys.astype(mx.float32)
    e = evicted_key.astype(mx.float32)
    e_norm = e / (mx.sqrt(mx.sum(e * e, axis=-1, keepdims=True)) + 1e-8)  # [BH, D]
    row_norms = mx.sqrt(mx.sum(k * k, axis=-1)) + 1e-8  # [BH, n]
    cos = mx.sum(k * e_norm[:, None, :], axis=-1) / row_norms  # [BH, n]

    idx = mx.arange(n)[None, :]  # [1, n]
    eligible = (idx >= n_sink_eff) & (idx != evict_idx)  # [BH, n]
    neg_inf = mx.full((bh, n), float("-inf"), dtype=mx.float32)
    masked = mx.where(eligible, cos, neg_inf)

    has_eligible = mx.any(eligible, axis=-1)  # [BH]
    result = mx.argmax(masked, axis=-1).astype(mx.int32)  # [BH]
    return mx.where(has_eligible, result, mx.array(-1, dtype=mx.int32))


def _merge_pair_batched(
    k_survivor: mx.array,  # [BH, D]
    v_survivor: mx.array,  # [BH, D]
    k_evicted: mx.array,  # [BH, D]
    v_evicted: mx.array,  # [BH, D]
    merge_mode: str,
    merge_keys: bool,
) -> tuple[mx.array, mx.array]:
    """Batched-``[BH]`` equivalent of calling :func:`merge_pair` once per row.

    ``merge_mode``/``merge_keys`` are uniform across rows (one config per
    cache instance — see :class:`CaMKVCache`), so only the per-row blend
    weight (``"sim_weighted"``) needs vectorizing.
    """
    if merge_mode == "drop":
        return k_survivor, v_survivor

    ks = k_survivor.astype(mx.float32)
    vs = v_survivor.astype(mx.float32)
    ke = k_evicted.astype(mx.float32)
    ve = v_evicted.astype(mx.float32)

    if merge_mode == "mean":
        w = mx.full((k_survivor.shape[0],), 0.5, dtype=mx.float32)
    else:  # sim_weighted
        denom = (
            mx.sqrt(mx.sum(ks * ks, axis=-1)) * mx.sqrt(mx.sum(ke * ke, axis=-1))
        ) + 1e-8  # [BH]
        cos = mx.sum(ks * ke, axis=-1) / denom  # [BH]
        w = mx.clip(cos, 0.0, 1.0)

    w_col = w[:, None]
    v_new = ((1.0 - w_col) * vs + w_col * ve).astype(v_survivor.dtype)
    k_new = ((1.0 - w_col) * ks + w_col * ke).astype(k_survivor.dtype) if merge_keys else k_survivor
    return k_new, v_new


@dataclass
class CaMState:
    """Per-head CaM-adapted eviction/merge state for one layer.

    Identical fields to H2OState plus the merge configuration — CaM reuses H2O's
    cumulative-mass scorer; only the over-budget disposition (merge vs drop)
    differs.

    Attributes:
        keys:        [n_kept, D] fp16 stored key rows, or None before first update.
        values:      [n_kept, D] fp16 stored value rows, or None before first update.
        scores:      [n_kept] cumulative softmax attention mass (float32), or None.
        n_sink:      Number of leading sink positions — never evicted or merged.
        budget:      Maximum tokens to keep at any time (including sinks).
        merge_mode:  ``"sim_weighted"`` | ``"mean"`` | ``"drop"`` (drop == H2O).
        merge_keys:  Whether keys are merged too (values always are).
        merge_gate:  Whether the Eq. 14 Bernoulli gate decides *whether* to
            merge (True, default) or every over-budget loser is unconditionally
            merged (False — the paper's ablated, non-recommended configuration).
        seed:        Base seed for the gate's deterministic Bernoulli draws.
        draw_count:  Running count of gate draws made so far (advances the
            deterministic RNG stream; not user-facing).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int
    merge_mode: str
    merge_keys: bool
    merge_gate: bool
    seed: int
    draw_count: int


def init_cam_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001
    merge_mode: str = "sim_weighted",
    merge_keys: bool = False,
    merge_gate: bool = True,
    seed: int = 0,
) -> CaMState:
    """Create an empty CaMState before any tokens arrive.

    Args:
        n_sink:     Number of initial sink positions to protect.
        budget:     Maximum total tokens kept (sinks + non-sinks).
        head_dim:   Head dimension D (unused here; accepted for API symmetry).
        merge_mode: ``"sim_weighted"`` (default), ``"mean"``, or ``"drop"``.
        merge_keys: Merge keys as well as values (default False → values only).
        merge_gate: Apply the Eq. 14 Bernoulli merge gate (default True — the
            paper's recommended configuration). False unconditionally merges
            every over-budget loser (the paper's ablated "w.o. Merge Mask"
            configuration, which the paper's own Table 2 shows underperforms
            plain eviction).
        seed:       Base seed for the gate's deterministic Bernoulli draws.

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
        merge_gate=bool(merge_gate),
        seed=int(seed),
        draw_count=0,
    )


def cam_update(
    state: CaMState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
    positions: mx.array | None = None,  # [n] int32, parallel to state.keys
    new_positions: mx.array | None = None,  # [S] int32, parallel to new_keys
) -> CaMState | tuple[CaMState, mx.array | None]:
    """Absorb S new tokens, merging the lowest-score token into a survivor if over budget.

    For each of the S incoming tokens:
      1. Accumulate the new key's attention weight (as proxy query) over all stored
         keys into the per-token scores (exactly like H2O).
      2. Append the new token with score 0.
      3. If over budget: pick the lowest-score non-sink token (H2O's ``argmin`` with
         sinks masked to +inf); find its most-similar surviving non-sink neighbour;
         if ``state.merge_gate`` is set, sample the Eq. 14 Bernoulli gate from the
         loser's score relative to the survivor's — on failure the loser is simply
         dropped (no blend), on success (or when the gate is disabled) it is
         **merged** into that neighbour by cosine similarity (``merge_pair``),
         transferring the loser's accumulated score to the neighbour. Either way
         the loser's slot is then removed. With ``merge_mode="drop"`` the neighbour
         is never touched and this is exactly H2O regardless of the gate.

    Args:
        state:      Current CaMState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.
        positions:  Optional ``[n]`` int32 true absolute positions parallel to
            ``state.keys``. Must be ``None`` iff ``state.keys`` is ``None``
            (mirrors K/V's own bootstrap contract) while ``new_positions``
            is given. See VeloxQuant-MLX#370 — used by ``CaMKVCache`` to
            build an explicit attention mask. On a merge, the survivor's
            position becomes ``max(survivor_position, loser_position)``:
            safe because a mask that is too PERMISSIVE toward a row already
            causally visible to a query can't newly violate causality — the
            target position was already <= any future query it's visible
            to, and folding in a not-older token's mass can only keep that
            true, never make a stale position look newer than it is.
        new_positions: Optional ``[S]`` int32 true absolute positions
            parallel to ``new_keys``. ``None`` disables position tracking
            entirely (the default).

    Returns:
        Updated ``CaMState`` with at most ``state.budget`` tokens, or, when
        ``new_positions`` is given, ``(state, positions_out)`` where
        ``positions_out`` mirrors the final surviving rows' true positions.
    """
    S = new_keys.shape[0]
    track = new_positions is not None
    if track and (positions is None) != (state.keys is None):
        raise ValueError("cam: positions must be given iff state.keys is given")

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]
        p_i = new_positions[i : i + 1] if track else None

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
                merge_gate=state.merge_gate,
                seed=state.seed,
                draw_count=state.draw_count,
            )
            if track:
                positions = p_i
            continue

        # --- score update (identical to H2O) -------------------------------
        attn = attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score = 0) ----------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)
        if track:
            positions_cat = mx.concatenate([positions, p_i], axis=0)

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
            draw_count = state.draw_count

            # Merge the loser into its most-similar survivor (unless drop mode).
            if state.merge_mode != "drop":
                tgt = most_similar_survivor(keys_cat[evict_idx], keys_cat, evict_idx, n_sink_eff)
                if tgt >= 0:
                    do_merge = True
                    if state.merge_gate:
                        p = merge_gate_probability(
                            float(scores_cat[evict_idx].item()), float(scores_cat[tgt].item())
                        )
                        do_merge = sample_merge_gate(p, state.seed, draw_count)
                        draw_count += 1
                    if do_merge:
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
                        if track:
                            # Survivor's position becomes the more-recent of
                            # the two folded-in positions (see this
                            # function's docstring for why this is safe).
                            merged_pos = mx.maximum(positions_cat[tgt], positions_cat[evict_idx])
                            positions_cat = mx.concatenate(
                                [positions_cat[:tgt], merged_pos[None], positions_cat[tgt + 1 :]],
                                axis=0,
                            )

            # Remove the loser's slot.
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]
            scores_cat = scores_cat[keep_indices]
            if track:
                positions_cat = positions_cat[keep_indices]
        else:
            draw_count = state.draw_count

        if track:
            positions = positions_cat
        state = CaMState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            merge_mode=state.merge_mode,
            merge_keys=state.merge_keys,
            merge_gate=state.merge_gate,
            seed=state.seed,
            draw_count=draw_count,
        )

    if track:
        return state, positions
    return state


# How often (in loop iterations) cam_update_batched forces graph
# materialization during the over-budget eviction loop. Same rationale and
# same interval as h2o.py's identically-named constant: without this, a long
# prefill whose budget is exceeded almost immediately queues one eviction's
# (plus merge's) worth of unevaluated graph nodes per token, risking MLX's
# Metal resource/command-buffer tracking limit before generation finishes.
_EVAL_FLUSH_INTERVAL = 32


def cam_update_batched(
    keys: mx.array | None,  # [BH, n, D] fp16 or None
    values: mx.array | None,  # [BH, n, D] fp16 or None
    scores: mx.array | None,  # [BH, n] fp32 or None
    positions: mx.array | None,  # [BH, n] int32 or None
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    n_sink: int,
    budget: int,
    merge_mode: str,
    merge_keys: bool,
    merge_gate: bool,
    seeds: list[int],
    next_pos: int,
    draw_count: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int]:
    """Vectorized-over-``BH`` equivalent of calling :func:`cam_update` once
    per ``(batch, head)`` pair with identical per-row state.

    All ``BH`` rows share ``n_sink``/``budget``/``merge_mode``/``merge_keys``/
    ``merge_gate`` (true for every real caller: :class:`CaMKVCache` applies
    one uniform config to every head — only the gate's RNG stream differs per
    row, via ``seeds``), so the per-token score/append/evict/merge math —
    otherwise identical for every row — can run as one batched MLX call per
    step instead of ``BH`` separate Python-level calls into :func:`cam_update`.
    Mirrors :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`'s
    ``[BH,S,D]`` design; the per-token loop over ``S`` itself remains a
    genuine recurrence (each token's merge decision depends on the previous
    token's state) and is untouched, exactly as in the function it replaces.

    Every step removes exactly one row when over budget (either the merged
    survivor slot is rewritten in place and the loser's slot dropped, or — in
    ``"drop"`` mode / a failed gate draw — the loser's slot is dropped
    outright), so ``keys``/``values``/``scores``/``positions`` stay
    rectangular across rows at every step, the same shape invariant
    :func:`h2o_update_batched` relies on for its ``take_along_axis``
    compaction.

    Numerically identical to the per-head loop it replaces: every op below is
    the same formula as :func:`cam_update`'s bootstrap/score-update/evict/
    merge branches, applied over a leading ``BH`` axis instead of a Python
    loop. Keys, scores, and positions are bit-for-bit equivalent (the
    eviction argmin and survivor argmax are computed identically either
    way); merged values in ``"sim_weighted"``/``"mean"`` mode carry an
    inherent, unavoidable float32 reduction-order difference from
    ``merge_pair``'s cosine-similarity weight (MLX's batched ``[BH,D]`` sum
    accumulates in a different order than a per-row ``[D]`` sum for the same
    data — confirmed to never exceed one fp16 representable step); ``"drop"``
    mode has no such reduction and stays bit-for-bit exact throughout. See
    ``veloxquant_mlx/tests/quantizers/test_cam_batched.py``.

    Args:
        seeds: ``[BH]`` per-row base seeds for the merge gate's deterministic
            draws (``CaMKVCache`` uses ``self._seed + head_idx``, mirroring
            the per-head loop's distinct draw stream per row).
        draw_count: Running count of gate draws made so far, shared across
            rows (every row draws at most once per step, so one counter
            advances once per step exactly as :func:`h2o_update_batched`
            advances ``next_pos`` once per step).

    Returns:
        ``(keys, values, scores, positions, next_pos, draw_count)`` — the
        first four ``[BH, n_kept, D]``/``[BH, n_kept]``.
    """
    bh, s, _d = new_keys.shape
    if s == 0:
        return keys, values, scores, positions, next_pos, draw_count
    if n_sink >= budget:
        raise ValueError("cam: sinks must leave at least one evictable position")

    k_dtype = new_keys.dtype
    v_dtype = new_values.dtype

    for i in range(s):
        k_i = new_keys[:, i].astype(mx.float32)  # [BH, D]
        v_i = new_values[:, i].astype(v_dtype)  # [BH, D]
        cur_pos = next_pos

        if keys is None:
            keys = new_keys[:, i : i + 1].astype(k_dtype)  # [BH, 1, D]
            values = v_i[:, None, :]
            scores = mx.ones((bh, 1), dtype=mx.float32)
            positions = mx.full((bh, 1), cur_pos, dtype=mx.int32)
            next_pos = cur_pos + 1
            continue

        attn = _attention_scores_batched(k_i, keys.astype(mx.float32))  # [BH, n]
        updated_scores = scores + attn

        keys_cat = mx.concatenate([keys, new_keys[:, i : i + 1].astype(k_dtype)], axis=1)
        values_cat = mx.concatenate([values, v_i[:, None, :]], axis=1)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((bh, 1), dtype=mx.float32)], axis=1)
        positions_cat = mx.concatenate(
            [positions, mx.full((bh, 1), cur_pos, dtype=mx.int32)], axis=1
        )

        n_total = keys_cat.shape[1]

        if n_total > budget:
            n_sink_eff = min(n_sink, n_total)
            if n_sink_eff > 0:
                sink_inf = mx.full((bh, n_sink_eff), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([sink_inf, scores_cat[:, n_sink_eff:]], axis=1)
            else:
                protected = scores_cat
            evict_idx = mx.argmin(protected, axis=-1, keepdims=True).astype(mx.int32)  # [BH, 1]

            if merge_mode != "drop":
                evicted_key = mx.take_along_axis(keys_cat, evict_idx[..., None], axis=1)[
                    :, 0
                ].astype(mx.float32)
                tgt = _most_similar_survivor_batched(evicted_key, keys_cat, evict_idx, n_sink_eff)
                has_tgt = tgt >= 0
                tgt_safe = mx.where(has_tgt, tgt, mx.array(0, dtype=mx.int32))[:, None]  # [BH, 1]

                do_merge = has_tgt
                if merge_gate:
                    evicted_score = mx.take_along_axis(scores_cat, evict_idx, axis=1)[:, 0]
                    survivor_score = mx.take_along_axis(scores_cat, tgt_safe, axis=1)[:, 0]
                    p = mx.where(
                        survivor_score <= 0.0,
                        mx.where(evicted_score > 0.0, mx.array(1.0), mx.array(0.0)),
                        mx.clip(evicted_score / mx.maximum(survivor_score, 1e-30), 0.0, 1.0),
                    )
                    gate_draw = _sample_merge_gate_batched(p, seeds, draw_count)
                    draw_count += 1
                    do_merge = do_merge & gate_draw

                k_tgt = mx.take_along_axis(keys_cat, tgt_safe[..., None], axis=1)[:, 0]
                v_tgt = mx.take_along_axis(values_cat, tgt_safe[..., None], axis=1)[:, 0]
                k_evicted_row = evicted_key.astype(k_dtype)
                v_evicted_row = mx.take_along_axis(values_cat, evict_idx[..., None], axis=1)[:, 0]

                k_merged, v_merged = _merge_pair_batched(
                    k_tgt, v_tgt, k_evicted_row, v_evicted_row, merge_mode, merge_keys
                )
                do_merge_col = do_merge[:, None]
                k_new_tgt = mx.where(do_merge_col, k_merged, k_tgt)
                v_new_tgt = mx.where(do_merge_col, v_merged, v_tgt)

                merged_score = (
                    mx.take_along_axis(scores_cat, evict_idx, axis=1)[:, 0]
                    + mx.take_along_axis(scores_cat, tgt_safe, axis=1)[:, 0]
                )
                survivor_score_cur = mx.take_along_axis(scores_cat, tgt_safe, axis=1)[:, 0]
                new_score_tgt = mx.where(do_merge, merged_score, survivor_score_cur)

                tgt_onehot = mx.arange(n_total)[None, :] == tgt_safe
                keys_cat = mx.where(tgt_onehot[..., None], k_new_tgt[:, None, :], keys_cat)
                values_cat = mx.where(tgt_onehot[..., None], v_new_tgt[:, None, :], values_cat)
                scores_cat = mx.where(tgt_onehot, new_score_tgt[:, None], scores_cat)

                if positions is not None:
                    evicted_pos = mx.take_along_axis(positions_cat, evict_idx, axis=1)[:, 0]
                    tgt_pos = mx.take_along_axis(positions_cat, tgt_safe, axis=1)[:, 0]
                    merged_pos = mx.maximum(tgt_pos, evicted_pos)
                    new_pos_tgt = mx.where(do_merge, merged_pos, tgt_pos)
                    positions_cat = mx.where(tgt_onehot, new_pos_tgt[:, None], positions_cat)

            # Remove the loser's slot (batched compaction — same
            # take_along_axis trick as h2o_update_batched's evict path).
            rows = mx.arange(n_total - 1)[None]  # [1, n_total-1]
            source = rows + (rows >= evict_idx)  # [BH, n_total-1]
            keys_cat = mx.take_along_axis(keys_cat, source[..., None], axis=1)
            values_cat = mx.take_along_axis(values_cat, source[..., None], axis=1)
            scores_cat = mx.take_along_axis(scores_cat, source, axis=1)
            positions_cat = mx.take_along_axis(positions_cat, source, axis=1)

        keys, values, scores, positions = keys_cat, values_cat, scores_cat, positions_cat
        next_pos = cur_pos + 1

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, scores, positions)

    return keys, values, scores, positions, next_pos, draw_count


def _attention_scores_batched(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights, batched over a leading ``[BH]`` axis.

    Args:
        query_proxy: ``[BH, D]``.
        keys:        ``[BH, n, D]``.

    Returns:
        ``[BH, n]`` softmax weights, each row summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy[..., None])[..., 0] * scale  # [BH, n]
    return mx.softmax(logits, axis=-1)


def cam_get_kv(state: CaMState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    return get_kv(state.keys, state.values)


def cam_fp16_bytes(state: CaMState) -> int:
    """Bytes currently stored for K + V in fp16."""
    return fp16_kv_bytes(state.keys)


def full_cam_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return full_fp16_kv_bytes(tokens_seen, head_dim)


__all__ = [
    "most_similar_survivor",
    "merge_pair",
    "merge_gate_probability",
    "sample_merge_gate",
    "CaMState",
    "init_cam_state",
    "cam_update",
    "cam_update_batched",
    "cam_get_kv",
    "cam_fp16_bytes",
    "full_cam_fp16_bytes",
]

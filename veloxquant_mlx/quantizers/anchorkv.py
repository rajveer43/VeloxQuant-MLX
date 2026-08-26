"""AnchorKV-adapted quantizer primitives — anchor-residual KV cache compression.

Inspired by "AnchorKV: Anchor-Residual KV Cache Compression" (Khalaf,
Shamshoum, Hodos, Sieradzki, Schuster; Technion; arXiv:2608.02901v1,
2026-08-03). Documented as "AnchorKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port. **No verified peer-reviewed venue
as of 2026-08-20** — a one-time, user-directed exception to this repo's
venue-verification rule, the same exception previously granted to NestedKV.
See ``paper/research/surveys/NEW_METHOD_SURVEY_V22.md``.

What AnchorKV adds that the repo did not have: a compression axis that is
neither eviction nor uniform per-token quantization. Every token is kept —
no position ever leaves the softmax — but most tokens are represented as a
projection onto a small set of exactly-stored "anchor" tokens (one anchor
index + one scalar coefficient) instead of a full K/V vector. Whatever byte
budget remains after anchors and per-token metadata buys quantized residuals
for the tokens whose approximation error costs the most attention-output
error, ranked by a first-order utility estimate (paper Eq. 6).

Adaptation decisions (documented, never hidden):
  1. **Key-as-query proxy**, same convention as ``quantizers/snapkv.py``.
     The paper's anchor scoring and utility estimate (Eq. 6) both use the
     true observation-window *query* vectors, which the cache wrapper does
     not see (only K/V are visible at ``update_and_fetch``). We substitute
     the trailing ``window`` *key* rows as proxy queries throughout —
     for anchor selection AND for utility scoring.
  2. **No fused decode kernel.** The paper fuses reconstruction into a
     FlashAttention-style tiled kernel so the dense cache is never
     materialized. This module (and its cache wrapper) reconstructs the
     dense fp16 K/V eagerly in MLX ops once per ``update_and_fetch`` call —
     correct, but not the paper's steady-state memory story. Stated as a
     limitation, not hidden.
  3. **One-shot prefill compression**, same convention as SnapKV-adapted /
     NestedKV-adapted. The paper's byte accounting and anchor/residual
     allocation run once, at the end of prefill, on a frozen set of
     positions. Decode tokens are appended exactly (fp16, no anchor
     projection) — never retroactively re-anchored — consistent with every
     other one-shot prefill method already in this repo.
  4. **Residual codec.** Randomized-Hadamard rotation (reusing
     ``HadamardPreconditioner``; falls back to the identity when
     ``head_dim`` fails ``is_hadamard_compatible`` — stated, not hidden),
     per-token absmax normalization, and the existing 4-level Lloyd-Max
     Gaussian codebook (``CodebookFactory.create("gaussian", ...)``) in
     place of hand-deriving new centroids — the paper's own codebook is
     also a 4-level Lloyd-Max fit for a unit-Gaussian source, so this reuse
     is exact in spirit, not an approximation of it.
  5. **Uniform anchor budget and window across heads/layers**, matching
     every other method in this repo — the paper also holds these fixed
     across all its experiments.

Byte accounting mirrors the paper's own budget equation (Eq. 9): anchors and
per-token metadata are charged first; residuals get whatever remains.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook
from veloxquant_mlx.math.rotation import is_hadamard_compatible, make_hadamard_diagonal
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner
from veloxquant_mlx.quantizers.snapkv import obs_window_attention_scores

# Residuals are stored at a fixed 2 bits/coordinate (paper §3.1): a 4-level
# Lloyd-Max codebook fit once for a unit-Gaussian source and reused across
# every call — the codebook does not depend on head_dim or the data, so
# fitting it per-tensor would be wasted work.
RESIDUAL_BITS = 2
_RESIDUAL_CODEBOOK_CACHE: dict[int, ScalarCodebook] = {}


def _residual_codebook(bits: int = RESIDUAL_BITS) -> ScalarCodebook:
    """4-level (or ``2**bits``-level) Lloyd-Max codebook for a unit-Gaussian source.

    Cached by bit-width: the codebook only depends on ``bits``, not on
    ``head_dim`` or any per-call data (residual coordinates are normalized
    to unit scale by the per-token absmax before quantizing), so refitting
    it on every call would be pure waste.
    """
    cb = _RESIDUAL_CODEBOOK_CACHE.get(bits)
    if cb is None:
        from veloxquant_mlx.codebooks.base import CodebookFactory

        # d is irrelevant for the "gaussian" strategy beyond deciding the
        # gaussian/beta split inside CodebookFactory (>= 64 picks gaussian);
        # pass a fixed d=128 so the request always resolves to the unit
        # N(0,1)-shaped Lloyd-Max fit the paper specifies, independent of
        # any real head_dim this module is called with.
        cb = CodebookFactory.create("gaussian", b=bits, d=128)
        _RESIDUAL_CODEBOOK_CACHE[bits] = cb
    return cb


class AnchorAssignment(NamedTuple):
    """Per-token anchor assignment and projection, for one side (K or V) of one head.

    Attributes:
        anchor_positions: ``[n_anchor]`` int32 — original token indices stored exactly.
        assign_idx: ``[S]`` int32 — for each token, index into ``anchor_positions``
            of its nearest anchor (anchors point to themselves).
        gamma: ``[S]`` fp32 — projection coefficient (Eq. 2); 1.0 on anchors.
        residual: ``[S, D]`` fp32 — ``x - gamma * anchor_x`` (Eq. 2); zero on anchors.
    """

    anchor_positions: Any  # noqa: F821
    assign_idx: Any  # noqa: F821
    gamma: Any  # noqa: F821
    residual: Any  # noqa: F821


def select_anchors(
    scoring_keys: Any,  # noqa: F821
    k: int,
    window: int,
    rho: float,
    seed: int,
) -> Any:  # noqa: F821
    """Select ``k`` anchor positions for one head (paper §3.2).

    The trailing ``window`` positions are always anchors (recency window +
    proxy-query source + anchors that cost no extra selection). Of the
    remaining ``k - window`` slots, a fraction ``rho`` goes to the
    highest-scoring earlier positions under SnapKV-style observation-window
    attention (reusing ``obs_window_attention_scores`` — key-as-query proxy,
    see module docstring point 1); the rest is sampled uniformly at random
    from the remaining positions, improving directional coverage for tokens
    that attention-based selection alone would not anchor well (paper's own
    stated rationale for the uniform share).

    Args:
        scoring_keys: ``[S, D]`` keys used only to rank earlier positions
            (should be post-RoPE, matching what the model actually attends
            with — see ``anchorkv_key_projection`` for why the *stored*
            projection basis is pre-RoPE while scoring stays post-RoPE).
        k: Total anchor budget (includes the window).
        window: Trailing positions always kept as anchors.
        rho: Fraction of the non-window budget filled by attention score
            (remainder filled by uniform sampling).
        seed: RNG seed for the uniform share (determinism).

    Returns:
        ``[n_anchor]`` int32 array of anchor positions, ascending, deduplicated,
        where ``n_anchor = min(k, S)``.
    """
    import mlx.core as mx

    S = int(scoring_keys.shape[0])
    k = min(max(k, 1), S)
    w = min(max(window, 0), k)

    window_start = S - w
    window_set = set(range(window_start, S))
    remaining_slots = k - w

    if remaining_slots <= 0 or len(window_set) >= S:
        return mx.array(sorted(window_set), dtype=mx.int32)

    candidates = [i for i in range(S) if i not in window_set]
    n_scored = min(int(round(rho * remaining_slots)), len(candidates))
    n_scored = max(n_scored, 0)

    chosen: set[int] = set()
    if n_scored > 0 and w > 0:
        scores = obs_window_attention_scores(scoring_keys, w)
        score_list = scores.tolist()
        ranked = sorted(candidates, key=lambda i: score_list[i], reverse=True)
        chosen.update(ranked[:n_scored])

    n_uniform = remaining_slots - len(chosen)
    if n_uniform > 0:
        rng = np.random.default_rng(seed)
        pool = [i for i in candidates if i not in chosen]
        if pool:
            n_uniform = min(n_uniform, len(pool))
            picked = rng.choice(np.array(pool), size=n_uniform, replace=False)
            chosen.update(int(p) for p in picked.tolist())

    kept = sorted(window_set | chosen)
    return mx.array(kept, dtype=mx.int32)


def assign_and_project(
    x: Any,  # noqa: F821
    anchor_positions: Any,  # noqa: F821
) -> AnchorAssignment:
    """Assign every token to its nearest anchor and project onto it (paper Eqs. 1-2).

    Args:
        x: ``[S, D]`` fp16/fp32 K or V matrix for one head.
        anchor_positions: ``[n_anchor]`` int32 anchor positions (from
            ``select_anchors``; shared across K and V, but assignment and
            projection are computed independently per side per the paper).

    Returns:
        :class:`AnchorAssignment` with the per-token assignment, coefficient,
        and residual (fp32).
    """
    import mlx.core as mx

    S, D = x.shape
    x32 = x.astype(mx.float32)
    anchors = x32[anchor_positions]  # [n_anchor, D]

    anchor_norms = mx.sqrt(mx.sum(anchors * anchors, axis=-1))  # [n_anchor]
    x_norms = mx.sqrt(mx.sum(x32 * x32, axis=-1))  # [S]

    dots = x32 @ anchors.T  # [S, n_anchor]
    denom = (x_norms[:, None] * anchor_norms[None, :]) + 1e-12
    cos = mx.abs(dots) / denom  # Eq. 1 uses |<xi, xa>|

    assign_idx = mx.argmax(cos, axis=-1).astype(mx.int32)  # [S] -> index into anchors

    anchor_norm_sq = (anchor_norms * anchor_norms) + 1e-12  # [n_anchor]
    gamma_all = dots / anchor_norm_sq[None, :]  # [S, n_anchor], Eq. 2 numerator/denominator
    gamma = mx.take_along_axis(gamma_all, assign_idx[:, None], axis=-1)[:, 0]  # [S]

    chosen_anchor = anchors[assign_idx]  # [S, D]
    x_tilde = gamma[:, None] * chosen_anchor  # [S, D]
    residual = x32 - x_tilde  # [S, D]

    return AnchorAssignment(
        anchor_positions=anchor_positions,
        assign_idx=assign_idx,
        gamma=gamma,
        residual=residual,
    )


class ResidualCodec:
    """Rotate -> per-token absmax normalize -> 4-level Lloyd-Max quantize.

    Reuses ``HadamardPreconditioner`` for the randomized rotation (paper's
    Eq. 10) and the repo's existing Lloyd-Max Gaussian codebook. Falls back
    to an identity rotation when ``head_dim`` is not Hadamard-compatible
    (see ``is_hadamard_compatible``) — the codebook is still valid (its fit
    does not depend on the rotation succeeding), just less effective at
    spreading outlier energy, which is stated as a limitation rather than
    silently degrading correctness.
    """

    def __init__(self, head_dim: int, seed: int, bits: int = RESIDUAL_BITS) -> None:
        import mlx.core as mx

        self._d = head_dim
        self._bits = bits
        self._codebook = _residual_codebook(bits)
        if is_hadamard_compatible(head_dim):
            D_np = make_hadamard_diagonal(head_dim, seed=seed)
            self._rotation: HadamardPreconditioner | None = HadamardPreconditioner(mx.array(D_np))
        else:
            self._rotation = None

    def encode(self, residual: Any) -> tuple[Any, Any]:  # noqa: F821
        """Encode ``[n, D]`` fp32 residuals -> (``[n, D]`` uint8 codes, ``[n]`` fp32 scales)."""
        import mlx.core as mx

        rotated = self._rotation.apply(residual) if self._rotation is not None else residual
        scale = mx.max(mx.abs(rotated), axis=-1) + 1e-8  # [n]
        normalized = rotated / scale[:, None]
        codes = self._codebook.quantize(normalized.astype(mx.float16))
        return codes, scale

    def decode(self, codes: Any, scale: Any) -> Any:  # noqa: F821
        """Decode ``([n, D]`` uint8 codes, ``[n]`` fp32 scales) -> ``[n, D]`` fp32 residuals."""
        import mlx.core as mx

        centroids = self._codebook.dequantize(codes).astype(mx.float32)
        rotated = centroids * scale[:, None]
        return self._rotation.apply_inverse(rotated) if self._rotation is not None else rotated

    @property
    def bytes_per_residual(self) -> int:
        """Packed-code bytes + one fp16 scale for a single residual vector."""
        codes_bytes = math.ceil(self._d * self._bits / 8)
        scale_bytes = 2  # fp16
        return codes_bytes + scale_bytes


def pack_codes(codes: np.ndarray, bits: int = RESIDUAL_BITS) -> np.ndarray:
    """Pack ``[n, D]`` small-integer codes into ``[n, ceil(D*bits/8)]`` bytes.

    ``bits``-wide codes are packed most-significant-first within each byte,
    ``8 // bits`` codes per byte (4 codes/byte at the default 2 bits),
    zero-padded at the end of each row if ``D`` does not divide evenly.
    """
    n, d = codes.shape
    per_byte = 8 // bits
    n_bytes = math.ceil(d / per_byte)
    out = np.zeros((n, n_bytes), dtype=np.uint8)
    codes = codes.astype(np.uint8)
    for j in range(d):
        byte_idx = j // per_byte
        shift = 8 - bits * ((j % per_byte) + 1)
        out[:, byte_idx] |= (codes[:, j] & ((1 << bits) - 1)) << shift
    return out


def unpack_codes(packed: np.ndarray, d: int, bits: int = RESIDUAL_BITS) -> np.ndarray:
    """Exact inverse of :func:`pack_codes` for ``d`` codes per row."""
    n = packed.shape[0]
    per_byte = 8 // bits
    out = np.zeros((n, d), dtype=np.uint8)
    mask = (1 << bits) - 1
    for j in range(d):
        byte_idx = j // per_byte
        shift = 8 - bits * ((j % per_byte) + 1)
        out[:, j] = (packed[:, byte_idx] >> shift) & mask
    return out


def key_value_utility(
    proxy_queries: Any,  # noqa: F821
    keys: Any,  # noqa: F821
    values: Any,  # noqa: F821
    key_residual: Any,  # noqa: F821
    value_residual: Any,  # noqa: F821
) -> tuple[Any, Any]:  # noqa: F821
    """First-order attention-output-error utility per token (paper Eq. 6).

    Uses the trailing rows of ``keys`` as proxy observation queries (same
    key-as-query convention as ``quantizers/snapkv.py``; the paper uses the
    prompt's true trailing queries, not visible to a cache wrapper).

    Args:
        proxy_queries: ``[m, D]`` fp32 proxy query vectors (trailing keys).
        keys: ``[S, D]`` fp32 exact keys for one head (pre-residual-drop).
        values: ``[S, D]`` fp32 exact values for one head.
        key_residual: ``[S, D]`` fp32 residual a token would lose if its key
            residual were NOT stored (``x - x_tilde`` from ``assign_and_project``).
        value_residual: ``[S, D]`` fp32 same, for values.

    Returns:
        ``(u_key, u_value)``, each ``[S]`` fp32 — larger means storing that
        token's residual recovers more attention-output error.
    """
    import mlx.core as mx

    S, D = keys.shape
    scale = math.sqrt(D)

    logits = (proxy_queries @ keys.T) / scale  # [m, S]
    attn = mx.softmax(logits, axis=-1)  # [m, S]
    outputs = attn @ values  # [m, D] — exact per-proxy-query output

    # u_key_t = mean_w [ alpha_t^2 * (q_w . Rt(r_t^K) / sqrt(D))^2 * ||V_t - y_w||^2 ]
    delta_s = (proxy_queries @ key_residual.T) / scale  # [m, S]
    diff_norm_sq = mx.sum((values[None, :, :] - outputs[:, None, :]) ** 2, axis=-1)  # [m, S]
    u_key = mx.mean((attn**2) * (delta_s**2) * diff_norm_sq, axis=0)  # [S]

    # u_value_t = mean_w [ alpha_t^2 * ||r_t^V||^2 ]
    value_residual_norm_sq = mx.sum(value_residual * value_residual, axis=-1)  # [S]
    u_value = mx.mean((attn**2), axis=0) * value_residual_norm_sq  # [S]

    return u_key, u_value


def allocate_residual_budget(
    utilities: list[Any],  # noqa: F821
    n_slots: int,
) -> list[Any]:  # noqa: F821
    """Pick the globally top-``n_slots`` utilities pooled across heads (paper §3.4).

    Args:
        utilities: Per-head ``[S_h]`` fp32 utility arrays (S_h may differ per
            head only in the sense of anchor exclusion — callers pass 0 or
            ``-inf`` for anchor positions, which never receive a residual).
        n_slots: Total number of residual slots available for this side
            (K or V) across the whole layer.

    Returns:
        Per-head boolean masks (one ``[S_h]`` mx.array per head, matching
        the input shapes) marking which positions receive a residual.
    """
    import mlx.core as mx

    if n_slots <= 0:
        return [mx.zeros(u.shape, dtype=mx.bool_) for u in utilities]

    flat_scores = []
    owners = []
    for h, u in enumerate(utilities):
        vals = u.tolist()
        for i, v in enumerate(vals):
            flat_scores.append(v)
            owners.append((h, i))

    total = len(flat_scores)
    n_slots = min(n_slots, total)
    if n_slots <= 0:
        return [mx.zeros(u.shape, dtype=mx.bool_) for u in utilities]

    order = sorted(range(total), key=lambda i: flat_scores[i], reverse=True)
    top = order[:n_slots]

    masks_np = [np.zeros(u.shape[0], dtype=bool) for u in utilities]
    for flat_i in top:
        h, i = owners[flat_i]
        masks_np[h][i] = True

    return [mx.array(m) for m in masks_np]


def anchorkv_budget_slots(
    seq_len: int,
    head_dim: int,
    n_anchor: int,
    theta: float,
    residual_codec_bytes: int,
) -> int:
    """Turn the retained fraction ``theta`` into a residual-slot count (paper Eq. 9).

    Charges the full uncompressed layer cost against: anchors stored exactly
    (fp16 K + V) and per-token metadata (one int32 anchor index + one fp32
    coefficient per side, per non-anchor token) for BOTH K and V. Whatever
    remains buys ``residual_codec_bytes``-sized residual slots, split evenly
    between K and V by the caller (matching the paper's ``floor(N/2)`` /
    remainder split — see ``AnchorKVKVCache``).

    Args:
        seq_len: Total prefill token count S.
        head_dim: Per-head dimension D.
        n_anchor: Number of anchor positions (shared by K and V).
        theta: Fraction of the uncompressed fp16 K+V cache to retain (0, 1].
        residual_codec_bytes: Bytes for one quantized residual (codes + scale).

    Returns:
        Total number of residual slots ``N`` (>= 0), to be split across K and V.
    """
    full_bytes = seq_len * head_dim * 2 * 2  # fp16 K + V

    n_non_anchor = max(seq_len - n_anchor, 0)
    anchor_bytes = n_anchor * head_dim * 2 * 2  # exact fp16 K + V
    # Per non-anchor token, per side: int32 index (4B) + fp32 coefficient (4B).
    metadata_bytes = n_non_anchor * 2 * (4 + 4)
    base_bytes = anchor_bytes + metadata_bytes

    budget_bytes = theta * full_bytes
    remaining = budget_bytes - base_bytes
    if remaining <= 0:
        return 0
    return int(remaining // residual_codec_bytes)


__all__ = [
    "RESIDUAL_BITS",
    "AnchorAssignment",
    "ResidualCodec",
    "select_anchors",
    "assign_and_project",
    "pack_codes",
    "unpack_codes",
    "key_value_utility",
    "allocate_residual_budget",
    "anchorkv_budget_slots",
]

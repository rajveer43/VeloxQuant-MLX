"""Q-Filters offline calibration — the paper's query-SVD filter estimation.

Implements §3.2 step 1 of "Q-Filters: Leveraging QK Geometry for Efficient KV
Cache Compression" (arXiv:2503.02812, **preprint**) *as specified*, closing
the deviation documented as "THE HONESTY CRUX" in
:mod:`veloxquant_mlx.quantizers.qfilters`.

Why this module has to exist
----------------------------
The cache-side estimator in :func:`~veloxquant_mlx.quantizers.qfilters.estimate_filter_dir`
derives the filter from the SVD of observed **keys**, because a cache never
sees queries. That recovers the dominant axis but *not its sign*, and the
sign is not a detail — it decides whether "high projection" means "attended"
or "ignored".

The paper's Theorem 3.3 is precisely what supplies the sign::

    E_{Q^h_i}(<Q^h_i, K^h_j>) ~= kappa^h * <K^h_j, u^h>,
    with  kappa^h = E_{Q^h_i}(<Q^h_i, u^h>) > 0        (Appendix A)

``kappa^h > 0`` is what makes "larger projection => larger expected attention
logit" a valid ranking rule. That positivity is a statement about the
**query** distribution (Observation 3.1), so it is recoverable only from
query activations. Estimating from keys discards exactly the quantity that
disambiguates the direction — hence the sign ambiguity the key-only path
carries as a permanent caveat.

What this module computes (paper §3.2, step 1)
----------------------------------------------
  (a) Gather ``Q^h`` activations by passing calibration samples through the
      model (the caller does this; see :func:`collect_query_activations`).
  (b) Compute the SVD of the gathered representations per layer and head::

          Q^h = U S V^T,  V = (v_1, v_2, ..., v_dH)      (paper Eq. 1)

  (c) Take the *positive* right vector, the Q-Filter::

          v_1^+ = sgn(1^T u_1) v_1                       (paper §3.2, 1c)

Two details this module follows the paper on, both easy to get wrong:

* **No mean-centering.** The paper's object is the anisotropic *drift* of the
  query cloud away from the origin (Observation 3.1: ``E<Q_i, u^h> > 0``).
  Centering the data would subtract that drift — the very signal being
  measured — and return the top *variance* axis instead. This is a real
  behavioral difference from the key-side estimator, which does center.
* **GQA group-averaging** (paper §3.2): with Grouped-Query Attention the
  filters of each group of query heads are averaged down to one filter per
  KV head, then renormalized.

Cost, per the paper's §4.2: 20 samples of length 2048 with the SVD taken over
3k sampled representations per head suffices; filters are ``l * n_H * d_H``
parameters (~36000x smaller than Llama-3.2-1B's weights) and calibration is a
one-time, pre-deployment pass — never in the per-token hot path.

Public API
----------
collect_query_activations   — hook a model and capture per-layer Q activations
compute_qfilters            — Eq. (1) + §3.2(1c) over gathered activations
average_gqa_filters         — collapse query-head filters onto KV heads
save_qfilters / load_qfilters — persist calibrated filters as an .npz artifact
QFiltersCalibration         — the loaded artifact (per-layer [H_kv, D] filters)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# Artifact format version — bumped if the on-disk layout changes so a stale
# file fails loudly instead of being silently misinterpreted.
QFILTERS_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class QFiltersCalibration:
    """Calibrated per-layer Q-Filters, ready to load into a cache.

    Attributes:
        filters: One ``[H_kv, D]`` float32 array per layer, each row a
            unit-norm Q-Filter ``v_1^+`` for that KV head.
        model_id: Model the filters were calibrated on. Filters are
            model-specific; applying them to another model is meaningless,
            so this is recorded and checked on load.
        n_samples: Number of query vectors the SVD consumed, per head.
        dataset: Free-form provenance string for the calibration corpus.
    """

    filters: list[mx.array]
    model_id: str
    n_samples: int
    dataset: str

    @property
    def n_layers(self) -> int:
        """Number of calibrated layers."""
        return len(self.filters)

    def head_dim(self) -> int:
        """Per-head dimension ``D`` of the calibrated filters."""
        if not self.filters:
            raise ValueError("qfilters calibration: no layers present")
        return int(self.filters[0].shape[1])


def compute_qfilters(queries: mx.array, max_svd_samples: int = 3000) -> mx.array:
    """Compute per-head Q-Filters from gathered query activations (paper Eq. 1).

    For each head independently, takes the SVD of the ``[N, D]`` query matrix
    and returns the top right-singular vector, sign-fixed per §3.2 step 1c as
    ``v_1^+ = sgn(1^T u_1) v_1``.

    The data is **not** mean-centered: the paper's Observation 3.1 is about
    the query cloud's drift away from the origin, which centering would
    remove. See the module docstring.

    Args:
        queries: ``[H, N, D]`` gathered query activations for one layer — ``H``
            heads, ``N`` sampled token positions, head dimension ``D``.
        max_svd_samples: Cap on rows fed to the SVD per head (paper §4.2 uses
            3k). Rows are subsampled deterministically (evenly strided) when
            ``N`` exceeds this, so results are reproducible.

    Returns:
        ``[H, D]`` float32 array of unit-norm Q-Filters, one per head.

    Raises:
        ValueError: If ``queries`` is not 3-D, or a head has fewer than 2
            sampled rows (an SVD direction is not defined).
    """
    if queries.ndim != 3:
        raise ValueError(f"compute_qfilters: expected [H, N, D], got shape {queries.shape}")
    h, n, d = (int(x) for x in queries.shape)
    if n < 2:
        raise ValueError(f"compute_qfilters: need N >= 2 query samples per head, got {n}")
    if max_svd_samples < 2:
        raise ValueError(f"compute_qfilters: max_svd_samples must be >= 2, got {max_svd_samples}")

    q = np.array(queries.astype(mx.float32), copy=False)

    # Deterministic even-stride subsample (never random) so recalibrating the
    # same activations reproduces the same filters bit-for-bit.
    if n > max_svd_samples:
        idx = np.linspace(0, n - 1, num=max_svd_samples).astype(np.int64)
        q = q[:, idx, :]

    out = np.empty((h, d), dtype=np.float32)
    for head in range(h):
        mat = q[head].astype(np.float64)  # [N, D], NOT centered (see docstring)
        # full_matrices=False: we only need the leading singular triplet.
        u, _s, vt = np.linalg.svd(mat, full_matrices=False)
        v1 = vt[0]  # top right-singular vector, [D]
        u1 = u[:, 0]  # matching left vector, [N]

        # Paper §3.2 step 1c: v_1^+ = sgn(1^T u_1) v_1. The SVD's sign is
        # arbitrary (u_1, v_1) and (-u_1, -v_1) are both valid; anchoring on
        # the left vector's sum makes the filter point along the direction the
        # queries actually drift toward, which is what Theorem 3.3's
        # kappa^h > 0 requires.
        s = float(np.sum(u1))
        # sgn(0) == 0 would zero the filter; fall back to +1 on an exact tie.
        v1 = v1 if s >= 0.0 else -v1

        norm = float(np.linalg.norm(v1))
        out[head] = (v1 / max(norm, 1e-12)).astype(np.float32)

    return mx.array(out)


def average_gqa_filters(filters: mx.array, n_kv_heads: int) -> mx.array:
    """Average query-head filters within each GQA group (paper §3.2).

    "In the case of Grouped-Query Attention or GQA, we simply average the
    Q-Filters for each group of Query representations." The averaged vector is
    renormalized so every stored filter stays unit-norm — projections are
    compared across heads, so a non-unit filter would silently reweight one
    head against another.

    Args:
        filters: ``[H_q, D]`` per-query-head filters from :func:`compute_qfilters`.
        n_kv_heads: Number of KV heads. Must divide ``H_q``.

    Returns:
        ``[n_kv_heads, D]`` float32 unit-norm filters, one per KV head. Returned
        unchanged (up to renormalization) when ``H_q == n_kv_heads`` (MHA).

    Raises:
        ValueError: If ``filters`` is not 2-D, ``n_kv_heads`` is not positive,
            or does not divide the query-head count.
    """
    if filters.ndim != 2:
        raise ValueError(f"average_gqa_filters: expected [H_q, D], got shape {filters.shape}")
    h_q, d = (int(x) for x in filters.shape)
    if n_kv_heads <= 0:
        raise ValueError(f"average_gqa_filters: n_kv_heads must be > 0, got {n_kv_heads}")
    if h_q % n_kv_heads != 0:
        raise ValueError(
            f"average_gqa_filters: n_kv_heads ({n_kv_heads}) must divide the "
            f"query-head count ({h_q})"
        )

    group = h_q // n_kv_heads
    grouped = filters.astype(mx.float32).reshape(n_kv_heads, group, d)
    averaged = mx.mean(grouped, axis=1)  # [n_kv_heads, D]

    norms = mx.sqrt(mx.sum(averaged**2, axis=1, keepdims=True))
    return averaged / mx.maximum(norms, mx.array(1e-12, dtype=mx.float32))


def collect_query_activations(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
    max_samples_per_head: int = 3000,
    seed: int = 0,
) -> list[mx.array]:
    """Run calibration text through a model and capture per-layer Q activations.

    Wraps each attention module's query projection to record its output, runs
    the calibration corpus forward under ``mx.no_grad``-equivalent evaluation,
    then restores the original modules. Nothing about the model is mutated
    once this returns.

    Paper §4.2 found 20 samples of length 2048 sufficient, with the SVD taken
    over 3k sampled representations per head.

    Args:
        model: An ``mlx_lm`` model exposing ``model.layers`` with
            ``self_attn.q_proj`` modules.
        tokenizer: Tokenizer with an ``encode`` method.
        texts: Calibration documents (paper uses a subset of the Pile).
        max_length: Truncate each document to this many tokens.
        max_samples_per_head: Reservoir cap on retained query vectors per
            layer, keeping memory bounded on long corpora.
        seed: Seed for the reservoir subsampling, for reproducibility.

    Returns:
        One ``[H_q, N, D]`` float32 array per layer of gathered query
        activations, ready for :func:`compute_qfilters`.

    Raises:
        ValueError: If ``texts`` is empty or the model exposes no layers.
    """
    if not texts:
        raise ValueError("collect_query_activations: texts must be non-empty")

    layers = getattr(getattr(model, "model", model), "layers", None)
    if not layers:
        raise ValueError("collect_query_activations: model exposes no .layers to hook")

    rng = np.random.default_rng(seed)
    captured: list[list[np.ndarray]] = [[] for _ in range(len(layers))]

    # --- install capture wrappers -------------------------------------
    originals = []
    for li, layer in enumerate(layers):
        attn = layer.self_attn
        q_proj = attn.q_proj
        originals.append((attn, q_proj))

        def make_hook(inner, store):
            def hooked(x):
                out = inner(x)
                store.append(np.array(out.astype(mx.float32), copy=True))
                return out

            return hooked

        attn.q_proj = make_hook(q_proj, captured[li])

    try:
        for text in texts:
            ids = tokenizer.encode(text)[:max_length]
            if len(ids) < 2:
                continue
            out = model(mx.array([ids]))
            # Force the lazy graph so the hooks actually run before we move on.
            mx.eval(out)
    finally:
        # --- always restore, even if a forward pass raised -------------
        for attn, q_proj in originals:
            attn.q_proj = q_proj

    n_heads = _infer_query_heads(model)
    result: list[mx.array] = []
    for li, chunks in enumerate(captured):
        if not chunks:
            raise ValueError(
                f"collect_query_activations: layer {li} captured no activations — "
                "the calibration texts may all have been too short"
            )
        # Each chunk is [B, S, H*D] (or [B, S, H, D]); flatten to [tokens, H*D].
        flat = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chunks], axis=0)
        n_tokens, hidden = flat.shape
        d_head = hidden // n_heads
        per_head = flat.reshape(n_tokens, n_heads, d_head).transpose(1, 0, 2)  # [H, N, D]

        if n_tokens > max_samples_per_head:
            idx = rng.choice(n_tokens, size=max_samples_per_head, replace=False)
            idx.sort()
            per_head = per_head[:, idx, :]

        result.append(mx.array(np.ascontiguousarray(per_head, dtype=np.float32)))

    return result


def _infer_query_heads(model) -> int:
    """Best-effort lookup of the query-head count from a model's config."""
    for holder in (getattr(model, "args", None), getattr(model, "config", None), model):
        for attr in ("num_attention_heads", "n_heads", "num_heads"):
            value = getattr(holder, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError(
        "collect_query_activations: could not infer the query-head count from "
        "the model config; pass pre-shaped [H, N, D] activations to "
        "compute_qfilters() directly instead"
    )


def save_qfilters(calibration: QFiltersCalibration, path: str | Path) -> Path:
    """Persist calibrated filters to a ``.npz`` artifact.

    Args:
        calibration: The calibration to write.
        path: Destination path; the parent directory is created if missing.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"layer_{i}": np.array(f.astype(mx.float32), copy=False)
        for i, f in enumerate(calibration.filters)
    }
    meta = json.dumps(
        {
            "version": QFILTERS_ARTIFACT_VERSION,
            "model_id": calibration.model_id,
            "n_layers": calibration.n_layers,
            "n_samples": calibration.n_samples,
            "dataset": calibration.dataset,
        }
    )
    np.savez(path, __meta__=np.array(meta), **arrays)
    return path


def load_qfilters(path: str | Path, expect_model_id: str | None = None) -> QFiltersCalibration:
    """Load calibrated filters from a ``.npz`` artifact.

    Args:
        path: Artifact written by :func:`save_qfilters`.
        expect_model_id: If given, the artifact's recorded model id must match.
            Filters are model-specific — silently using another model's
            filters would degrade quality with no visible error.

    Returns:
        The loaded :class:`QFiltersCalibration`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On a version mismatch, a missing layer, or a model-id
            mismatch against ``expect_model_id``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"load_qfilters: no calibration artifact at {path}")

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        version = int(meta.get("version", -1))
        if version != QFILTERS_ARTIFACT_VERSION:
            raise ValueError(
                f"load_qfilters: artifact version {version} != expected "
                f"{QFILTERS_ARTIFACT_VERSION} ({path}) — recalibrate"
            )
        model_id = str(meta.get("model_id", ""))
        if expect_model_id is not None and model_id != expect_model_id:
            raise ValueError(
                f"load_qfilters: artifact was calibrated on {model_id!r} but "
                f"{expect_model_id!r} was expected — Q-Filters are model-specific"
            )

        n_layers = int(meta["n_layers"])
        filters = []
        for i in range(n_layers):
            key = f"layer_{i}"
            if key not in data:
                raise ValueError(f"load_qfilters: artifact {path} is missing {key}")
            filters.append(mx.array(data[key].astype(np.float32)))

    return QFiltersCalibration(
        filters=filters,
        model_id=model_id,
        n_samples=int(meta.get("n_samples", 0)),
        dataset=str(meta.get("dataset", "")),
    )


__all__ = [
    "QFILTERS_ARTIFACT_VERSION",
    "QFiltersCalibration",
    "average_gqa_filters",
    "collect_query_activations",
    "compute_qfilters",
    "load_qfilters",
    "save_qfilters",
]

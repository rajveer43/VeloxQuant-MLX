"""Calibration for SpectralQuant.

Collects KV vectors from a real mlx-lm model by running calibration text
through it, then computes per-layer eigenvector rotation matrices via SVD.

The paper uses 100 calibration sequences from WikiText-2 (~15 seconds on
one GPU). We support both real-model calibration and a fast synthetic path
for unit tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from veloxquant_mlx.spectral.participation_ratio import (
    compute_participation_ratio,
    compute_spectral_gap,
)

_CACHE_ROOT = Path(os.environ.get("VELOXQUANT_CACHE_DIR", Path.home() / ".cache" / "veloxquant"))


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------


def _rotation_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_").replace("\\", "_")
    return _CACHE_ROOT / "spectral" / safe / "rotations.npz"


def load_cached_rotations(model_name: str) -> dict | None:
    """Load cached rotation matrices if they exist.

    Returns:
        Dict mapping layer_idx -> (key_U, val_U, key_eigenvalues,
        val_eigenvalues, key_d_s, val_d_s) or None.
    """
    path = _rotation_path(model_name)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    result: dict[int, tuple] = {}
    keys_in_file = set(data.files)
    layer_ids: set[int] = set()
    for k in keys_in_file:
        parts = k.split("_")
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                layer_ids.add(int(parts[1]))
            except ValueError:
                pass
    for layer_idx in layer_ids:
        key_U = data[f"layer_{layer_idx}_key_U"]
        val_U = data[f"layer_{layer_idx}_val_U"]
        key_ev = data.get(f"layer_{layer_idx}_key_ev", np.ones(key_U.shape[0]))
        val_ev = data.get(f"layer_{layer_idx}_val_ev", np.ones(val_U.shape[0]))
        key_ds = int(data.get(f"layer_{layer_idx}_key_ds", np.array(4)))
        val_ds = int(data.get(f"layer_{layer_idx}_val_ds", np.array(50)))
        result[layer_idx] = (key_U, val_U, key_ev, val_ev, key_ds, val_ds)
    return result


def save_rotations(
    model_name: str,
    rotations: dict[int, tuple],
) -> None:
    """Persist rotation matrices and eigenvalues to disk.

    Args:
        model_name: Model identifier (used as cache key).
        rotations: Dict mapping layer_idx -> (key_U, val_U, key_ev, val_ev,
            key_ds, val_ds).
    """
    path = _rotation_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for layer_idx, entry in rotations.items():
        key_U, val_U, key_ev, val_ev, key_ds, val_ds = entry
        arrays[f"layer_{layer_idx}_key_U"] = key_U.astype(np.float32)
        arrays[f"layer_{layer_idx}_val_U"] = val_U.astype(np.float32)
        arrays[f"layer_{layer_idx}_key_ev"] = key_ev.astype(np.float64)
        arrays[f"layer_{layer_idx}_val_ev"] = val_ev.astype(np.float64)
        arrays[f"layer_{layer_idx}_key_ds"] = np.array(key_ds, dtype=np.int32)
        arrays[f"layer_{layer_idx}_val_ds"] = np.array(val_ds, dtype=np.int32)
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------


def _svd_rotation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA via SVD: returns (U, eigenvalues_descending, d_s).

    U is (d, d) with columns = eigenvectors sorted by descending eigenvalue.
    eigenvalues are the variance along each principal direction.
    d_s is ceil(participation_ratio).
    """
    X = X.astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    n = max(len(X) - 1, 1)
    # SVD of data matrix: X = U_data S V^T → Cov = V diag(S²/n) V^T
    # We want columns of V sorted by descending S² (eigenvalue).
    _, s, Vt = np.linalg.svd(X, full_matrices=True)
    eigenvalues = (s**2 / n).astype(np.float64)
    # Pad eigenvalues to d if X has fewer samples than dims
    d = Vt.shape[0]
    if len(eigenvalues) < d:
        eigenvalues = np.pad(eigenvalues, (0, d - len(eigenvalues)))
    # U = V^T transposed = Vt.T: columns are eigenvectors in descending order
    U = Vt.T.astype(np.float32)  # (d, d), columns = eigenvectors

    d_s = max(1, int(np.ceil(compute_participation_ratio(X))))
    d_s = min(d_s, d)
    return U, eigenvalues, d_s


def collect_kv_vectors_mlx(
    model: Any,
    calibration_tokens: Any,
    n_tokens_per_run: int = 512,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Run calibration tokens through model and collect KV vectors per layer.

    Uses mlx-lm's cache protocol: we pass a list of collector objects as the
    `cache` argument. Each collector's `update_and_fetch` method is called
    by the attention layer with raw (pre-compression) key and value tensors.

    Args:
        model: mlx_lm model instance.
        calibration_tokens: Token IDs, shape (1, seq_len) or (seq_len,).
        n_tokens_per_run: Maximum tokens to collect per layer per run.

    Returns:
        Tuple (key_vecs, val_vecs): dicts mapping layer_idx -> np.ndarray
        of shape (N, head_dim) containing collected KV vectors (all heads
        concatenated).
    """
    import mlx.core as mx

    key_vecs: dict[int, list[np.ndarray]] = {}
    val_vecs: dict[int, list[np.ndarray]] = {}

    layers = getattr(model, "layers", None) or model.model.layers
    n_layers = len(layers)

    class _CollectingCache:
        """Wraps an existing mlx-lm cache and intercepts update_and_fetch to record KV tensors."""

        def __init__(self, layer_idx: int, inner: Any | None = None):
            self.layer_idx = layer_idx
            self._inner = inner  # real cache (e.g. RotatingKVCache) or None
            self.keys = None
            self.values = None
            self.offset = 0
            self.is_empty = True

        def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
            # Cast to float32 first — bfloat16 PEP 3118 buffer is incompatible with numpy
            k_np = np.array(keys.astype(mx.float32))
            v_np = np.array(values.astype(mx.float32))
            # Collapse batch × heads × seq into (N, head_dim)
            k_np = k_np.reshape(-1, k_np.shape[-1])
            v_np = v_np.reshape(-1, v_np.shape[-1])
            key_vecs.setdefault(self.layer_idx, []).append(k_np[:n_tokens_per_run])
            val_vecs.setdefault(self.layer_idx, []).append(v_np[:n_tokens_per_run])
            self.offset += keys.shape[-2]
            self.is_empty = False
            # Delegate to the real cache (e.g. RotatingKVCache) if present,
            # otherwise just pass through so attention can proceed normally.
            if self._inner is not None:
                result = self._inner.update_and_fetch(keys, values)
                self.keys, self.values = result
                return result
            self.keys = keys
            self.values = values
            return keys, values

        # Proxy attribute access to inner cache for any other attrs the model reads
        def __getattr__(self, name: str) -> Any:
            if name.startswith("_") or self._inner is None:
                raise AttributeError(name)
            return getattr(self._inner, name)

    # Use model.make_cache() if available — some models (e.g. Gemma 4 with
    # sliding-window attention) produce fewer caches than n_layers, or use
    # specialised cache types (RotatingKVCache). We wrap each real cache so
    # the model's attention logic stays intact while we intercept KV tensors.
    if hasattr(model, "make_cache"):
        real_caches = model.make_cache()
        caches = [_CollectingCache(i, inner=real_caches[i]) for i in range(len(real_caches))]
        # Map cache index → layer index. For standard models make_cache returns
        # one cache per layer; for sliding-window models it may return fewer —
        # in that case we still want per-layer indices from the wrapper's
        # layer_idx (same as cache index, which is what the model iterates).
    else:
        caches = [_CollectingCache(i) for i in range(n_layers)]

    tokens = calibration_tokens
    if hasattr(tokens, "tolist"):
        tokens = mx.array(np.array(tokens))
    if tokens.ndim == 1:
        tokens = tokens[None]

    # Run model with collecting caches
    logits = model(tokens, cache=caches)
    mx.eval(logits)  # force evaluation so all cache calls have fired

    # Consolidate — iterate over whatever indices were actually collected
    key_out: dict[int, np.ndarray] = {}
    val_out: dict[int, np.ndarray] = {}
    for i in key_vecs:
        if key_vecs[i]:
            key_out[i] = np.concatenate(key_vecs[i], axis=0)
    for i in val_vecs:
        if val_vecs[i]:
            val_out[i] = np.concatenate(val_vecs[i], axis=0)

    return key_out, val_out


def calibrate_spectral_rotation(
    model: Any,
    calibration_tokens: Any,
    n_tokens: int = 512,
    model_name: str = "model",
    force_recompute: bool = False,
) -> dict[int, tuple]:
    """Compute per-layer spectral rotation matrices from calibration data.

    Paper §3.1: "Collect n_cal = 100 sequences; extract KV vectors.
    Compute Σ = (1/N) Σ h_t h_t^T. U, Λ = torch.linalg.eigh(Σ), sorted
    descending. d_s = ⌈PR(Σ)⌉. This takes ≈ 15 seconds."

    Args:
        model: Loaded mlx_lm model instance.
        calibration_tokens: Token IDs, shape (seq_len,) or (1, seq_len).
        n_tokens: Maximum KV vectors to collect per layer.
        model_name: String identifier for on-disk cache.
        force_recompute: If True, ignore cached rotations.

    Returns:
        Dict mapping layer_idx -> (key_U, val_U, key_eigenvalues,
        val_eigenvalues, key_d_s, val_d_s) where:
          key_U / val_U:  (d, d) float32, columns = eigenvectors descending.
          key_eigenvalues / val_eigenvalues: (d,) float64 descending.
          key_d_s / val_d_s: int, ⌈d_eff⌉ signal dimensions.
    """
    if not force_recompute:
        cached = load_cached_rotations(model_name)
        if cached is not None:
            return cached

    key_vecs, val_vecs = collect_kv_vectors_mlx(
        model, calibration_tokens, n_tokens_per_run=n_tokens
    )

    layers = getattr(model, "layers", None) or model.model.layers
    rotations: dict[int, tuple] = {}

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue

        # Determine head_dim
        hd = getattr(attn, "head_dim", None)
        if hd is None:
            args = getattr(model, "args", None)
            if args is not None:
                hd = getattr(args, "head_dim", None)
                if hd is None and hasattr(args, "hidden_size"):
                    hd = args.hidden_size // args.num_attention_heads
        if hd is None:
            continue

        # Keys
        if i in key_vecs and len(key_vecs[i]) >= 4:
            K = key_vecs[i][:n_tokens].astype(np.float32)
            # Ensure shape matches head_dim
            if K.shape[1] != hd:
                K = K.reshape(-1, hd)[:n_tokens]
            key_U, key_ev, key_ds = _svd_rotation(K)
        else:
            # Fallback: identity-like random orthogonal (equivalent to random rotation)
            rng = np.random.default_rng(i)
            key_U, _ = np.linalg.qr(rng.standard_normal((hd, hd)).astype(np.float32))
            key_ev = np.ones(hd, dtype=np.float64)
            key_ds = 4  # paper default

        # Values
        if i in val_vecs and len(val_vecs[i]) >= 4:
            V = val_vecs[i][:n_tokens].astype(np.float32)
            if V.shape[1] != hd:
                V = V.reshape(-1, hd)[:n_tokens]
            val_U, val_ev, val_ds = _svd_rotation(V)
        else:
            rng = np.random.default_rng(i + 10000)
            val_U, _ = np.linalg.qr(rng.standard_normal((hd, hd)).astype(np.float32))
            val_ev = np.ones(hd, dtype=np.float64)
            val_ds = 50  # paper default

        rotations[i] = (key_U, val_U, key_ev, val_ev, key_ds, val_ds)

    save_rotations(model_name, rotations)
    return rotations


def calibrate_from_vectors(
    key_vectors: dict[int, np.ndarray],
    val_vectors: dict[int, np.ndarray],
    model_name: str = "synthetic",
) -> dict[int, tuple]:
    """Build calibration result directly from pre-collected KV arrays.

    Useful for testing and for cases where you already have KV vectors
    collected outside the mlx-lm forward pass.

    Args:
        key_vectors: Dict layer_idx -> (N, d) float32.
        val_vectors: Dict layer_idx -> (N, d) float32.
        model_name: Cache key for saving to disk.

    Returns:
        Same format as calibrate_spectral_rotation().
    """
    rotations: dict[int, tuple] = {}
    all_layers = set(key_vectors.keys()) | set(val_vectors.keys())
    for i in sorted(all_layers):
        K = key_vectors.get(i)
        V = val_vectors.get(i)
        if K is not None and len(K) >= 4:
            key_U, key_ev, key_ds = _svd_rotation(K)
        else:
            d = V.shape[1] if V is not None else 128
            rng = np.random.default_rng(i)
            key_U, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
            key_ev = np.ones(d, dtype=np.float64)
            key_ds = 4
        if V is not None and len(V) >= 4:
            val_U, val_ev, val_ds = _svd_rotation(V)
        else:
            d = K.shape[1] if K is not None else 128
            rng = np.random.default_rng(i + 10000)
            val_U, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
            val_ev = np.ones(d, dtype=np.float64)
            val_ds = 50
        rotations[i] = (key_U, val_U, key_ev, val_ev, key_ds, val_ds)
    save_rotations(model_name, rotations)
    return rotations

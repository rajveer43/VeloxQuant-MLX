"""Mapper data structures and on-disk format for cross-model KV transfer.

A fitted mapper is, concretely, one ridge weight matrix and bias per
(target layer, K|V) plus the list of source layers that feed it. The paper
fits *per head*; we store each target layer's heads stacked into a single
``[n_kv_heads, in_dim, head_dim]`` tensor so the apply path is one batched
matmul per layer rather than a Python loop over heads.

Size matters here. The paper reports 1.01–3.36 B mapper parameters (4–12 GB)
per pair. Even at the small end of a family this is large enough that holding
every layer resident defeats the purpose, so :class:`CrossModelMapper` loads
layer weights **lazily** from a safetensors file and can drop them again.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

__all__ = [
    "MatchedKVError",
    "ModelKVSpec",
    "MapperConfig",
    "LayerMap",
    "CrossModelMapper",
]


class MatchedKVError(ValueError):
    """Raised when a source/target pair violates the matched-KV precondition.

    The paper only evaluates pairs where source and target share KV head count
    and per-head dimension, and explicitly leaves mismatched-KV to future work.
    Fitting anyway would produce a mapper with no evidence behind it, so this
    is an error rather than a warning.
    """


@dataclass(frozen=True)
class ModelKVSpec:
    """The KV-relevant shape of one model, read off its config.

    Attributes:
        n_layers: Number of transformer layers.
        n_kv_heads: Key/value head count (post-GQA, not query heads).
        head_dim: Per-head dimension.
        rope_theta: RoPE frequency base.
        rope_traditional: Whether the model uses interleaved-pair RoPE.
        rope_scaling: Model's rope_scaling config, if any.
        model_type: e.g. ``"qwen3"`` — used to warn on cross-family fits.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rope_traditional: bool = False
    rope_scaling: dict[str, Any] | None = None
    model_type: str = "unknown"

    @classmethod
    def from_model(cls, model: Any) -> ModelKVSpec:
        """Derive a spec from an ``mlx_lm`` model.

        Reads ``model.args`` (the ``ModelArgs`` dataclass every ``mlx_lm``
        architecture carries), falling back to ``model.config`` for wrappers
        that expose a dict instead.
        """
        args = getattr(model, "args", None) or getattr(model, "config", None)
        if args is None:
            raise ValueError(
                "Cannot read model shape: expected an mlx_lm model exposing `.args` or `.config`."
            )
        get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)

        n_heads = get("num_attention_heads")
        n_kv_heads = get("num_key_value_heads") or n_heads
        hidden = get("hidden_size")
        head_dim = get("head_dim") or (hidden // n_heads if hidden and n_heads else None)
        if not (n_kv_heads and head_dim):
            raise ValueError("Model config is missing num_key_value_heads / head_dim.")

        return cls(
            n_layers=get("num_hidden_layers"),
            n_kv_heads=int(n_kv_heads),
            head_dim=int(head_dim),
            rope_theta=float(get("rope_theta", 10000.0) or 10000.0),
            rope_traditional=bool(get("rope_traditional", False)),
            rope_scaling=get("rope_scaling", None),
            model_type=str(get("model_type", "unknown")),
        )

    def validate_transfer_to(self, target: ModelKVSpec) -> None:
        """Check the matched-KV precondition and unsupported RoPE variants.

        Raises:
            MatchedKVError: If KV head count or head dim differ, or if either
                model uses a RoPE variant this implementation does not handle.
        """
        if self.n_kv_heads != target.n_kv_heads or self.head_dim != target.head_dim:
            raise MatchedKVError(
                "Cross-model KV transfer requires matched KV: source has "
                f"{self.n_kv_heads} KV heads of dim {self.head_dim}, target has "
                f"{target.n_kv_heads} of dim {target.head_dim}. The paper only "
                "evaluates matched-KV pairs and leaves mismatched-KV to future "
                "work, so this is refused rather than approximated."
            )
        for name, spec in (("source", self), ("target", target)):
            if spec.rope_traditional:
                raise MatchedKVError(
                    f"The {name} model uses traditional (interleaved-pair) RoPE, "
                    "which the content-space mapping in "
                    "veloxquant_mlx.transfer.rope does not implement (it assumes "
                    "NeoX-style half-split, MLX's default)."
                )
            if spec.rope_scaling:
                raise MatchedKVError(
                    f"The {name} model uses rope_scaling ({spec.rope_scaling!r}). "
                    "Stripping RoPE with the plain formula would not invert the "
                    "scaled rotation, silently corrupting the mapped keys. "
                    "Refused — see issue #148 for the same gap in the eviction caches."
                )


@dataclass
class MapperConfig:
    """Hyperparameters of a fit.

    Attributes:
        k: Source layers concatenated per target layer (paper §3.2). The paper
            sweeps ``{1,2,4,6,8,...}`` and selects per pair; ``k=1`` is shown
            to be uniformly insufficient.
        ridge_lambda: Tikhonov term. The paper uses 0.01 and reports a wide
            flat region, collapsing only at ``λ=1``.
        n_calib_sequences: Calibration sequences (paper: 500).
        calib_seq_len: Tokens per calibration sequence (paper: 1024).
        stride: Token subsampling stride (paper: 4).
        content_space: Strip/re-apply RoPE around the map (paper §3.3).
        dtype: Storage dtype for fitted weights.
    """

    k: int = 8
    ridge_lambda: float = 0.01
    n_calib_sequences: int = 500
    calib_seq_len: int = 1024
    stride: int = 4
    content_space: bool = True
    dtype: str = "float16"


@dataclass
class LayerMap:
    """The fitted map for one target layer.

    Attributes:
        target_layer: Index of the target layer this map produces.
        source_layers: Source layer indices whose KV are concatenated as input,
            in the order they were concatenated at fit time. Order is part of
            the artifact: applying with a permuted order silently misaligns
            the weight blocks.
        w_k: ``[n_kv_heads, in_dim, head_dim]`` key weights, or None if unloaded.
        b_k: ``[n_kv_heads, head_dim]`` key biases, or None if unloaded.
        w_v: Value weights, same shape convention as ``w_k``.
        b_v: Value biases.
        r2_k: In-sample key R² at fit time, for diagnostics only. The paper
            (§4.5) shows R² does *not* predict downstream retention across
            pairs — attention-output cosine does — so this is recorded for
            source-layer selection and debugging, not as a quality promise.
        r2_v: In-sample value R².
    """

    target_layer: int
    source_layers: list[int]
    w_k: mx.array | None = None
    b_k: mx.array | None = None
    w_v: mx.array | None = None
    b_v: mx.array | None = None
    r2_k: float = float("nan")
    r2_v: float = float("nan")

    @property
    def is_loaded(self) -> bool:
        return self.w_k is not None and self.w_v is not None

    def unload(self) -> None:
        """Drop weight tensors, keeping the metadata."""
        self.w_k = self.b_k = self.w_v = self.b_v = None


@dataclass
class CrossModelMapper:
    """A fitted source→target KV mapper.

    Weights live in a safetensors file alongside a JSON manifest; layers are
    loaded on demand so a multi-GB mapper never needs to be fully resident.
    """

    source: ModelKVSpec
    target: ModelKVSpec
    config: MapperConfig
    layers: list[LayerMap] = field(default_factory=list)
    _weights_path: Path | None = None
    _weights: dict[str, mx.array] | None = None

    # -- introspection ---------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Total fitted parameters, from shapes rather than resident tensors."""
        in_dim = self.config.k * self.source.n_kv_heads * self.source.head_dim
        per_layer = 2 * self.target.n_kv_heads * (in_dim + 1) * self.target.head_dim
        return per_layer * len(self.layers)

    def describe(self) -> str:
        gb = self.n_parameters * (2 if self.config.dtype == "float16" else 4) / 1e9
        return (
            f"CrossModelMapper({self.source.model_type} L{self.source.n_layers} → "
            f"{self.target.model_type} L{self.target.n_layers}, k={self.config.k}, "
            f"{self.n_parameters / 1e9:.2f}B params, ~{gb:.1f} GB)"
        )

    # -- lazy weight access ----------------------------------------------

    def load_layer(self, target_layer: int) -> LayerMap:
        """Return a layer's map, reading its weights from disk if needed."""
        lm = self.layers[target_layer]
        if lm.is_loaded:
            return lm
        if self._weights_path is None:
            raise RuntimeError(
                f"Layer {target_layer} has no weights and the mapper has no "
                "backing file to load them from."
            )
        if self._weights is None:
            # mx.load on safetensors is lazy per-key on access.
            self._weights = mx.load(str(self._weights_path))
        w = self._weights
        lm.w_k, lm.b_k = w[f"{target_layer}.w_k"], w[f"{target_layer}.b_k"]
        lm.w_v, lm.b_v = w[f"{target_layer}.w_v"], w[f"{target_layer}.b_v"]
        return lm

    def unload_all(self) -> None:
        """Drop every layer's weights, releasing memory."""
        for lm in self.layers:
            lm.unload()
        self._weights = None

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the mapper to ``path`` as ``mapper.safetensors`` + ``manifest.json``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        flat: dict[str, mx.array] = {}
        for lm in self.layers:
            if not lm.is_loaded:
                raise RuntimeError(
                    f"Cannot save: target layer {lm.target_layer} has no weights "
                    "loaded. Saving a partially-unloaded mapper would silently "
                    "write an incomplete artifact."
                )
            flat[f"{lm.target_layer}.w_k"] = lm.w_k
            flat[f"{lm.target_layer}.b_k"] = lm.b_k
            flat[f"{lm.target_layer}.w_v"] = lm.w_v
            flat[f"{lm.target_layer}.b_v"] = lm.b_v

        mx.save_safetensors(str(out / "mapper.safetensors"), flat)
        manifest = {
            "format_version": 1,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "config": asdict(self.config),
            "layers": [
                {
                    "target_layer": lm.target_layer,
                    "source_layers": lm.source_layers,
                    "r2_k": lm.r2_k,
                    "r2_v": lm.r2_v,
                }
                for lm in self.layers
            ],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return out

    @classmethod
    def load(cls, path: str | Path) -> CrossModelMapper:
        """Read a mapper written by :meth:`save`. Weights stay on disk until used."""
        src = Path(path)
        manifest = json.loads((src / "manifest.json").read_text())
        if manifest.get("format_version") != 1:
            raise ValueError(
                f"Unsupported mapper format_version {manifest.get('format_version')!r}; "
                "this build reads version 1."
            )
        mapper = cls(
            source=ModelKVSpec(**manifest["source"]),
            target=ModelKVSpec(**manifest["target"]),
            config=MapperConfig(**manifest["config"]),
            layers=[
                LayerMap(
                    target_layer=entry["target_layer"],
                    source_layers=entry["source_layers"],
                    r2_k=entry.get("r2_k", float("nan")),
                    r2_v=entry.get("r2_v", float("nan")),
                )
                for entry in manifest["layers"]
            ],
        )
        mapper._weights_path = src / "mapper.safetensors"
        return mapper

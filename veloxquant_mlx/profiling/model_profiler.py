"""Model-architecture extraction for the auto-selection pipeline.

Profiles a language model's attention geometry *without* loading weights: from
a HuggingFace-style ``config.json`` (a dict), a loaded HF/MX ``PretrainedConfig``
(object with attribute access), or a loaded mlx_lm model. The derived
:class:`ModelProfile` is what the candidate filter and memory estimator feed on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ModelProfile",
    "profile_model_from_config",
    "attention_type_from_heads",
    "architecture_alias",
]

_ARCHITECTURE_ALIASES: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "Qwen2ForCausalLM": "qwen",
    "Qwen2MoeForCausalLM": "qwen",
    "Qwen3ForCausalLM": "qwen",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
    "Phi3ForCausalLM": "phi",
    "PhiForCausalLM": "phi",
    "OLMoForCausalLM": "olmo",
    "CohereForCausalLM": "command-r",
}

#: model_type -> canonical architecture for assets without `architectures`.
_MODEL_TYPE_ALIASES: dict[str, str] = {
    "llama": "llama",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "qwen2": "qwen",
    "qwen3": "qwen",
    "gemma": "gemma",
    "gemma2": "gemma",
    "phi3": "phi",
    "phi": "phi",
}


def architecture_alias(raw_name: str) -> str:
    """Map an HF class name to a short architecture slug, else pass through."""
    raw = raw_name.strip()
    if raw in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[raw]
    low = raw.lower()
    for key in ("llama", "mistral", "mixtral", "qwen", "gemma", "phi", "olmo"):
        if key in low:
            return _MODEL_TYPE_ALIASES.get(key, key)
    return raw


def attention_type_from_heads(num_query_heads: int, num_kv_heads: int) -> str:
    """Classify MHA / GQA / MQA from the head counts."""
    if num_kv_heads == num_query_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    return "gqa"


@dataclass(frozen=True)
class ModelProfile:
    """Normalized model-architecture description used by the planner.

    Attributes:
        model_id: Model identifier (any string; the caller's ``model_id``).
        architecture: Short architecture slug (``"llama"``, ``"qwen"``, ...).
        num_layers: Number of attention-bearing layers.
        num_query_heads: Attention heads per layer.
        num_kv_heads: Key/value heads per layer (``== num_query_heads`` for MHA).
        head_dim: Attention head dimension.
        attention_type: ``"mha"``, ``"gqa"``, or ``"mqa"``.
        dtype: Expected compute dtype (``"float16"``, ...).
        parameter_count: Total weight count, when the config reports one.
        hidden_size: Model hidden width, when known (informational).
    """

    model_id: str = "unknown"
    architecture: str = "unknown"
    num_layers: int = 0
    num_query_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    attention_type: str = "gqa"
    dtype: str = "float16"
    parameter_count: int | None = None
    hidden_size: int | None = None
    _extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def baseline_kv_bytes_per_token(self) -> int:
        """Full K+V footprint in bytes for one token, one layer (fp16)."""
        return 2 * self.num_kv_heads * self.head_dim * 2

    def baseline_kv_bytes(self, num_layers: int | None = None, batch: int = 1) -> int:
        """Total fp16 K+V footprint across layers/batch for one token."""
        return self.baseline_kv_bytes_per_token * (num_layers or self.num_layers) * batch

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain dict."""
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "attention_type": self.attention_type,
            "dtype": self.dtype,
            "parameter_count": self.parameter_count,
            "hidden_size": self.hidden_size,
        }


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            value = obj.get(name) if isinstance(obj, dict) else None
        if value not in (None, ""):
            return value
    return default


def profile_model_from_config(
    config: dict[str, Any] | Any | None = None,
    model_id: str | None = None,
    *,
    num_layers: int | None = None,
    num_query_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    dtype: str | None = None,
    parameter_count: int | None = None,
    hidden_size: int | None = None,
    architecture: str | None = None,
) -> ModelProfile:
    """Build a :class:`ModelProfile` from a HF-style config without loading weights.

    **Process**:
    1. Extract num_layers from config (keys: num_hidden_layers, num_layers)
    2. Extract num_query_heads (keys: num_attention_heads, num_heads)
    3. Extract head_dim (key: head_dim) OR infer from hidden_size / num_query_heads
    4. Extract num_kv_heads (key: num_key_value_heads); default to num_query_heads (MHA)
    5. Detect attention type from head ratio (MHA/GQA/MQA)
    6. Extract dtype, parameter_count, architecture
    7. Override all extracted values with keyword arguments (if provided)

    **Config formats supported**:
    - ``dict``: HuggingFace ``config.json`` as dictionary
    - ``object``: Loaded ``PretrainedConfig`` (attribute access)
    - ``None``: Allowed if all critical kwargs are provided

    **Keyword overrides**: All kwargs override config values (even if config present)
    Useful for partial configs or CLI inputs (--num-layers, --model-class).

    **Model families**:
    - Llama, Qwen, Mistral, Mixtral, Phi, Gemma, OLMo, Command-R
    Auto-detected via architecture name or model_type field

    **Attention types**:
    - **MHA** (Multi-Head Attention): num_kv_heads == num_query_heads
    - **GQA** (Grouped Query Attention): 1 < num_kv_heads < num_query_heads
    - **MQA** (Multi-Query Attention): num_kv_heads == 1

    **Arguments**:
        config: HF config dict/object (optional; can be None if overrides given)
        model_id: Model identifier (e.g., "Qwen/Qwen2.5-7B"). Extracted from
            config._name_or_path if not provided.
        num_layers: Number of transformer layers (override)
        num_query_heads: Number of query attention heads (override)
        num_kv_heads: Number of key/value heads (override)
        head_dim: Attention head dimension (override)
        dtype: Compute dtype string (override)
        parameter_count: Total model parameters (override)
        hidden_size: Model hidden dimension (override)
        architecture: Architecture slug (override; auto-detected if not given)

    **Returns**:
        ModelProfile with all extracted/derived fields.

    **Raises**:
        ValueError: If critical geometry is missing (num_layers, num_query_heads,
            head_dim cannot be inferred). Keyword overrides help resolve this.

    **Cost**: ~1ms (pure config parsing, no weights loaded)

    **Example**:
        >>> config = {"num_hidden_layers": 32, "hidden_size": 4096,
        ...           "num_attention_heads": 32, "architectures": ["QwenForCausalLM"]}
        >>> profile = profile_model_from_config(config, model_id="Qwen/Qwen2.5-7B")
        >>> print(f"Architecture: {profile.architecture}")  # "qwen"
        >>> print(f"Attention: {profile.attention_type}")  # "gqa" or "mha"
    """
    n_layers = _first_int(num_layers, lambda: _get_attr(config, "num_hidden_layers", "num_layers"))
    n_q = _first_int(num_query_heads, lambda: _get_attr(config, "num_attention_heads", "num_heads"))
    hidden = _first_int(hidden_size, lambda: _get_attr(config, "hidden_size"))
    d = _first_int(head_dim, lambda: _get_attr(config, "head_dim"))

    if d is None and hidden is not None and n_q:
        d = max(1, hidden // n_q)

    owner = _get_attr(config, "architectures")
    architecture_name = architecture or owner
    if isinstance(architecture_name, list):
        architecture_name = architecture_name[0] if architecture_name else None
    arch = (
        architecture_alias(str(architecture_name))
        if architecture_name is not None
        else _get_attr(config, "model_type", default="unknown") or "unknown"
    )

    if n_layers is None or n_q is None or d is None:
        raise ValueError(
            "cannot build ModelProfile: missing attention geometry. Provide a "
            "config.json (num_hidden_layers, num_attention_heads, "
            f"num_key_value_heads, hidden_size or head_dim); got layers={n_layers}, "
            f"query_heads={n_q}, head_dim={d}."
        )

    n_kv = _first_int(
        num_kv_heads, lambda: _get_attr(config, "num_key_value_heads", "num_kv_heads")
    )
    if n_kv is None:
        n_kv = n_q  # MHA default

    dtype_name = dtype or str(_get_attr(config, "torch_dtype", "dtype", default="float16"))
    params = parameter_count
    if params is None:
        params = _get_attr(config, "num_parameters", "_num_parameters", default=None)
    hid = hidden if hidden is not None else _get_attr(config, "hidden_size", default=None)

    return ModelProfile(
        model_id=model_id or str(_get_attr(config, "_name_or_path", default="unknown")),
        architecture=str(arch),
        num_layers=int(n_layers),
        num_query_heads=int(n_q),
        num_kv_heads=int(n_kv),
        head_dim=int(d),
        attention_type=attention_type_from_heads(int(n_q), int(n_kv)),
        dtype=dtype_name if isinstance(dtype_name, str) else str(dtype_name),
        parameter_count=int(params) if params not in (None, "") else None,
        hidden_size=int(hid) if hid not in (None, "") else None,
    )


def _first_int(*candidates: Any) -> int | None:
    """Return the first non-None int-ish value across values and callables."""
    for cand in candidates:
        value = cand() if callable(cand) else cand
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def profile_model_from_model(model: Any, model_id: str | None = None) -> ModelProfile:
    """Profile a loaded mlx_lm/HF model object using ``model.config``.

    ``model`` may be an mlx_lm model with a ``model.config`` attribute or a HF
    ``PreTrainedModel``. Not required for recommending (config-only is enough),
    included for the detailed-profiling path.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model has no `.config` attribute to profile from")
    return profile_model_from_config(
        config,
        model_id=model_id or type(model).__name__,
        dtype=str(getattr(config, "torch_dtype", "float16")),
    )

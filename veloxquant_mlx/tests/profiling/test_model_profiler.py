"""Tests for model-architecture extraction (RFC Phase 2)."""

from __future__ import annotations

import pytest

from veloxquant_mlx.profiling.model_profiler import (
    architecture_alias,
    attention_type_from_heads,
    profile_model_from_config,
    profile_model_from_model,
)

_LLAMA_CONFIG = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_parameters": 8_000_000_000,
    "torch_dtype": "bfloat16",
}


def test_profiles_llama_gqa():
    profile = profile_model_from_config(_LLAMA_CONFIG)
    assert profile.architecture == "llama"
    assert profile.num_layers == 32
    assert profile.num_query_heads == 32
    assert profile.num_kv_heads == 8
    assert profile.head_dim == 128
    assert profile.attention_type == "gqa"
    assert profile.dtype == "bfloat16"
    assert profile.parameter_count == 8_000_000_000
    assert profile.hidden_size == 4096


def test_profiles_mha():
    config = {**_LLAMA_CONFIG, "num_key_value_heads": 32}
    profile = profile_model_from_config(config)
    assert profile.attention_type == "mha"


def test_profiles_mqa():
    config = {**_LLAMA_CONFIG, "num_key_value_heads": 1}
    profile = profile_model_from_config(config)
    assert profile.attention_type == "mqa"


def test_head_dim_inferred_from_hidden_size():
    config = {k: v for k, v in _LLAMA_CONFIG.items() if k != "hidden_size"}
    config["hidden_size"] = 4096
    profile = profile_model_from_config(config)
    assert profile.head_dim == 128


def test_explicit_head_dim_overrides_inference():
    profile = profile_model_from_config(_LLAMA_CONFIG, head_dim=256)
    assert profile.head_dim == 256


def test_overrides_win_over_config():
    profile = profile_model_from_config(_LLAMA_CONFIG, num_layers=8)
    assert profile.num_layers == 8
    assert profile.architecture == "llama"


def test_geometry_overrides_without_config():
    profile = profile_model_from_config(
        None,
        model_id="my-model",
        num_layers=12,
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=64,
        architecture="qwen",
    )
    assert profile.model_id == "my-model"
    assert profile.architecture == "qwen"
    assert profile.attention_type == "mha"


def test_missing_geometry_raises_value_error():
    with pytest.raises(ValueError):
        profile_model_from_config(None, num_layers=4, num_query_heads=16)


def test_attribute_object_config_supported():
    class _Cfg:
        num_hidden_layers = 24
        num_attention_heads = 32
        num_key_value_heads = 4
        hidden_size = 2048

    profile = profile_model_from_config(_Cfg())
    assert profile.attention_type == "gqa"
    assert profile.head_dim == 64


def test_unknown_model_type_becomes_architecture():
    config = {**_LLAMA_CONFIG, "architectures": ["SomethingElseForCausalLM"]}
    profile = profile_model_from_config(config)
    assert profile.architecture == "SomethingElseForCausalLM"


def test_architecture_alias_known_and_unknown():
    assert architecture_alias("LlamaForCausalLM") == "llama"
    assert architecture_alias("Qwen3ForCausalLM") == "qwen"
    assert architecture_alias("weird") == "weird"


def test_model_type_fallback():
    config = {k: v for k, v in _LLAMA_CONFIG.items() if k != "architectures"}
    profile = profile_model_from_config(config)
    assert profile.architecture == "llama"


def test_attention_type_from_heads():
    assert attention_type_from_heads(32, 32) == "mha"
    assert attention_type_from_heads(32, 1) == "mqa"
    assert attention_type_from_heads(32, 8) == "gqa"


def test_baseline_kv_bytes_per_token():
    profile = profile_model_from_config(_LLAMA_CONFIG)
    # 2 tensors (K, V) * 8 kv heads * 128 dim * 2 bytes
    assert profile.baseline_kv_bytes_per_token == 2 * 8 * 128 * 2


def test_baseline_kv_bytes_scales_with_layers_and_batch():
    profile = profile_model_from_config(_LLAMA_CONFIG)
    per_token = profile.baseline_kv_bytes_per_token
    assert profile.baseline_kv_bytes() == per_token * 32
    assert profile.baseline_kv_bytes(num_layers=8, batch=2) == per_token * 8 * 2


def test_to_dict_contains_geometry():
    data = profile_model_from_config(_LLAMA_CONFIG).to_dict()
    assert data["attention_type"] == "gqa"
    assert data["head_dim"] == 128


def test_parameter_count_absent_is_none():
    config = {k: v for k, v in _LLAMA_CONFIG.items() if k != "num_parameters"}
    assert profile_model_from_config(config).parameter_count is None


def test_profile_from_model_object():
    class _Model:
        config = type("C", (), {"torch_dtype": "float16", "num_hidden_layers": 4,
                                "num_attention_heads": 8, "num_key_value_heads": 8,
                                "hidden_size": 1024})()

    profile = profile_model_from_model(_Model())
    assert profile.attention_type == "mha"


def test_profile_from_model_without_config_raises():
    class _NoConfig:
        pass

    with pytest.raises(ValueError):
        profile_model_from_model(_NoConfig())

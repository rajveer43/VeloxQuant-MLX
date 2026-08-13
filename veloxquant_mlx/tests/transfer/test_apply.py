"""Apply-path and persistence tests for cross-model KV transfer.

Two things matter most here and both are silent-corruption risks rather than
crash risks:

  - **Concatenation order.** The fitted weight blocks are positional. Applying
    with source layers in a different order than they were fit produces
    plausible-looking, wrong keys.
  - **Lazy loading.** Mapper artifacts are multi-GB; the apply path must load
    a layer, use it, and release it, and a save/load round-trip must be exact.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.transfer.apply import load_mapper, transfer_cache, transfer_layer
from veloxquant_mlx.transfer.fit import fit_mapper
from veloxquant_mlx.transfer.mapper import CrossModelMapper, MapperConfig, ModelKVSpec
from veloxquant_mlx.transfer.rope import apply_rope, strip_rope

N_TOKENS = 128
HEAD_DIM = 16
N_HEADS = 2
SRC_LAYERS = 4
TGT_LAYERS = 3


@pytest.fixture
def pair():
    """A source cache plus a target that is an exact linear map of source layer 1."""
    mx.random.seed(0)
    pos = mx.arange(N_TOKENS)
    src_spec = ModelKVSpec(SRC_LAYERS, N_HEADS, HEAD_DIM, 10000.0, model_type="src")
    tgt_spec = ModelKVSpec(TGT_LAYERS, N_HEADS, HEAD_DIM, 1000000.0, model_type="tgt")
    source = {
        i: (
            mx.random.normal((N_HEADS, N_TOKENS, HEAD_DIM)),
            mx.random.normal((N_HEADS, N_TOKENS, HEAD_DIM)),
        )
        for i in range(SRC_LAYERS)
    }
    w = mx.random.normal((N_HEADS * HEAD_DIM, HEAD_DIM)) * 0.1
    content = strip_rope(source[1][0], pos, src_spec.rope_theta)
    flat_k = content.transpose(1, 0, 2).reshape(N_TOKENS, -1)
    flat_v = source[1][1].transpose(1, 0, 2).reshape(N_TOKENS, -1)
    target = {
        li: (
            apply_rope(mx.stack([flat_k @ w] * N_HEADS), pos, tgt_spec.rope_theta),
            mx.stack([flat_v @ w] * N_HEADS),
        )
        for li in range(TGT_LAYERS)
    }
    mapper = fit_mapper(source, target, src_spec, tgt_spec, pos, MapperConfig(k=2))
    return source, target, mapper, pos


def test_transfer_reconstructs_target_cache(pair):
    """A perfectly-fittable pair must map back to the target's own KV."""
    source, target, mapper, pos = pair
    out = transfer_cache(source, mapper, pos, unload_after=False)
    assert set(out) == set(target)
    for li in target:
        assert float(mx.max(mx.abs(out[li][0] - target[li][0]))) < 5e-3
        assert float(mx.max(mx.abs(out[li][1] - target[li][1]))) < 5e-3


def test_transfer_output_shapes_and_dtype_match_source(pair):
    source, _, mapper, pos = pair
    keys, values = transfer_layer(mapper, 0, source, pos)
    assert keys.shape == (N_HEADS, N_TOKENS, HEAD_DIM) == values.shape
    assert keys.dtype == source[0][0].dtype


def test_transfer_defaults_positions_to_arange(pair):
    """Omitting positions must behave as a cache prefilled from position 0."""
    source, _, mapper, pos = pair
    explicit = transfer_cache(source, mapper, pos, unload_after=False)
    implied = transfer_cache(source, mapper, None, unload_after=False)
    for li in explicit:
        assert float(mx.max(mx.abs(explicit[li][0] - implied[li][0]))) == 0.0


def test_transfer_applies_target_rope_not_source(pair):
    """Output keys must carry the *target's* rotation.

    Stripping with the target base should recover content space; stripping
    with the source base should not. Without this, a mapper would appear to
    work while feeding the target model keys rotated at the wrong frequency.
    """
    source, target, mapper, pos = pair
    keys, _ = transfer_layer(mapper, 0, source, pos)
    right = strip_rope(keys, pos, mapper.target.rope_theta)
    truth = strip_rope(target[0][0], pos, mapper.target.rope_theta)
    wrong = strip_rope(keys, pos, mapper.source.rope_theta)
    assert float(mx.max(mx.abs(right - truth))) < 5e-3
    assert float(mx.max(mx.abs(wrong - truth))) > 1e-2


def test_transfer_rejects_missing_source_layer(pair):
    source, _, mapper, pos = pair
    needed = mapper.layers[0].source_layers[0]
    with pytest.raises(KeyError, match="missing"):
        transfer_layer(mapper, 0, {k: v for k, v in source.items() if k != needed}, pos)


def test_transfer_rejects_position_count_mismatch(pair):
    source, _, mapper, _ = pair
    with pytest.raises(ValueError, match="positions has"):
        transfer_cache(source, mapper, mx.arange(N_TOKENS - 1))


def test_transfer_rejects_empty_cache(pair):
    _, _, mapper, _ = pair
    with pytest.raises(ValueError, match="empty"):
        transfer_cache({}, mapper)


def test_unload_after_releases_layer_weights(pair):
    """Peak memory must stay at one layer, not the whole mapper."""
    source, _, mapper, pos = pair
    transfer_cache(source, mapper, pos, unload_after=True)
    assert not any(lm.is_loaded for lm in mapper.layers)


# --- persistence -------------------------------------------------------


def test_save_load_round_trip_is_exact(pair, tmp_path):
    source, _, mapper, pos = pair
    before = transfer_cache(source, mapper, pos, unload_after=False)
    mapper.save(tmp_path)

    reloaded = load_mapper(tmp_path)
    assert not reloaded.layers[0].is_loaded, "weights must load lazily"
    after = transfer_cache(source, reloaded, pos, unload_after=False)
    for li in before:
        assert float(mx.max(mx.abs(before[li][0] - after[li][0]))) == 0.0
        assert float(mx.max(mx.abs(before[li][1] - after[li][1]))) == 0.0


def test_save_preserves_metadata(pair, tmp_path):
    _, _, mapper, _ = pair
    mapper.save(tmp_path)
    reloaded = load_mapper(tmp_path)
    assert reloaded.source == mapper.source
    assert reloaded.target == mapper.target
    assert reloaded.config.k == mapper.config.k
    assert [lm.source_layers for lm in reloaded.layers] == [
        lm.source_layers for lm in mapper.layers
    ]


def test_saving_unloaded_mapper_is_refused(pair, tmp_path):
    """Silently writing an incomplete artifact would be worse than failing."""
    _, _, mapper, _ = pair
    mapper.layers[0].unload()
    with pytest.raises(RuntimeError, match="no weights loaded"):
        mapper.save(tmp_path)


def test_load_rejects_unknown_format_version(pair, tmp_path):
    import json

    _, _, mapper, _ = pair
    mapper.save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["format_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version"):
        CrossModelMapper.load(tmp_path)


def test_load_layer_without_backing_file_errors(pair):
    _, _, mapper, _ = pair
    mapper.layers[0].unload()
    with pytest.raises(RuntimeError, match="no backing file"):
        mapper.load_layer(0)


def test_parameter_accounting_matches_stored_shapes(pair):
    """n_parameters is derived from config, so it must match the real tensors."""
    _, _, mapper, _ = pair
    counted = sum(lm.w_k.size + lm.b_k.size + lm.w_v.size + lm.b_v.size for lm in mapper.layers)
    assert mapper.n_parameters == counted
    assert "CrossModelMapper" in mapper.describe()


# --- model spec --------------------------------------------------------


def test_model_spec_reads_mlx_lm_style_config():
    """ModelKVSpec must derive head_dim from hidden_size when not explicit."""

    class Args:
        num_hidden_layers = 4
        num_attention_heads = 8
        num_key_value_heads = 2
        hidden_size = 256
        head_dim = None
        rope_theta = 5000.0
        model_type = "qwen3"

    class Model:
        args = Args()

    spec = ModelKVSpec.from_model(Model())
    assert (spec.n_kv_heads, spec.head_dim, spec.rope_theta) == (2, 32, 5000.0)


def test_model_spec_requires_a_config():
    with pytest.raises(ValueError, match="mlx_lm model"):
        ModelKVSpec.from_model(object())

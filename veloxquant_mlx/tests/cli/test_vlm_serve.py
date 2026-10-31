"""Tests for the MLX-VLM server bridge without loading model weights."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.cli import vlm_serve


def test_installs_veloxquant_hook_once_per_model(monkeypatch):
    class Model:
        pass

    model = Model()
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "native")
    app = SimpleNamespace(
        get_cached_model=lambda *args, **kwargs: (
            model,
            SimpleNamespace(tokenizer=tokenizer),
            "config",
        )
    )
    server_package = SimpleNamespace()
    patched = []
    monkeypatch.setattr(
        vlm_serve,
        "patch_vlm_kv_cache",
        lambda received_model, received_config: patched.append((received_model, received_config)),
    )
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)

    vlm_serve._install_vlm_patch(config, server_package, app)
    first = app.get_cached_model("qwen")
    second = server_package.get_cached_model("qwen")

    assert first == second
    assert patched == [(model, config)]


def test_unhashable_models_are_tracked_without_hashing(monkeypatch):
    """MLX ``nn.Module`` defines ``__eq__`` without ``__hash__``, so it is
    unhashable; the patch tracker must therefore never hash a model."""

    class UnhashableModel:
        def __eq__(self, other):
            return self is other

    with pytest.raises(TypeError, match="unhashable"):
        hash(UnhashableModel())

    model = UnhashableModel()
    processor = SimpleNamespace(tokenizer=object())
    app = SimpleNamespace(
        get_cached_model=lambda *args, **kwargs: (
            model,
            processor,
            "config",
        )
    )
    server_package = SimpleNamespace()
    patched = []
    monkeypatch.setattr(
        vlm_serve,
        "ensure_initial_system_prompt_support",
        lambda *_: False,
    )
    monkeypatch.setattr(
        vlm_serve,
        "patch_vlm_kv_cache",
        lambda received_model, received_config: patched.append((received_model, received_config)),
    )
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)

    vlm_serve._install_vlm_patch(config, server_package, app)
    first = app.get_cached_model("qwen")
    second = server_package.get_cached_model("qwen")

    assert first == second
    assert patched == [(model, config)]


def test_tracker_releases_models_after_collection():
    """A collected model must stop being tracked: CPython can hand its
    ``id()`` to the next model, and a stale entry would skip the patch for an
    entirely different model instance."""

    class UnhashableModel:
        def __eq__(self, other):
            return self is other

    tracker = vlm_serve._PatchedModelTracker()
    model = UnhashableModel()
    aref = weakref.ref(model)

    tracker.mark_patched(model)
    assert tracker.is_patched(model)
    assert len(tracker) == 1

    del model
    gc.collect()

    assert aref() is None
    assert len(tracker) == 0
    # A later model is never treated as already patched, whether or not the
    # allocator reuses the freed model's id.
    assert not tracker.is_patched(UnhashableModel())


def test_parser_defaults_to_servable_method():
    args = vlm_serve.build_parser().parse_args([])
    assert args.method == "turboquant_rvq"
    assert args.bits == 2


def test_validate_method_is_testable_without_process_exit():
    with pytest.raises(vlm_serve.MethodNotServableError, match="cannot be served"):
        vlm_serve._validate_method("turboquant_prod")


def test_vlm_factory_checks_tokenizer_before_wiring_cache(monkeypatch):
    class Model:
        pass

    model = Model()
    processor = SimpleNamespace(tokenizer=object())
    app = SimpleNamespace(get_cached_model=lambda *args, **kwargs: (model, processor, "config"))
    server_package = SimpleNamespace()
    checked = []
    monkeypatch.setattr(
        vlm_serve,
        "ensure_initial_system_prompt_support",
        lambda tokenizer, model_id: checked.append((tokenizer, model_id)) or False,
    )
    monkeypatch.setattr(vlm_serve, "patch_vlm_kv_cache", lambda *_: None)

    vlm_serve._install_vlm_patch(KVCacheConfig(method="turboquant_rvq"), server_package, app)
    app.get_cached_model("mlx-community/Qwen3.5-9B-MLX-4bit")

    assert checked == [(processor.tokenizer, "mlx-community/Qwen3.5-9B-MLX-4bit")]

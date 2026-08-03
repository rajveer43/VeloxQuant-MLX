"""Monkey-patches an mlx-lm model to build its KV cache via VeloxQuant-MLX.

Usage::

    from mlx_lm import load
    from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
    from veloxquant_mlx.cache import KVCacheConfig

    model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    patch_model_kv_cache(model, config)

    # mlx_lm.generate() now builds a quantized cache automatically
    from mlx_lm import generate
    response = generate(model, tokenizer, prompt="...", max_tokens=200)
"""

from __future__ import annotations

from typing import Any

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig


def patch_model_kv_cache(model: Any, config: KVCacheConfig) -> list[Any]:
    """Wire a quantized KV cache into an mlx-lm model's generation path.

    ``mlx_lm.generate()`` builds its prompt cache via
    ``mlx_lm.models.cache.make_prompt_cache()``, which calls
    ``model.make_cache()`` if the model defines one, and falls back to a
    default fp16 cache otherwise. There is no pre-existing ``.cache``
    attribute on attention sub-modules to overwrite — caches are only ever
    constructed lazily, on demand, through that hook.

    This function builds the real per-layer cache list via
    ``KVCacheBuilder.for_model()`` (which already handles VLM wrappers,
    MoE-gate fallback layers, per-layer bit-width lists, and cross-layer
    methods like XQuant/MiniCache/PyramidKV) and overrides
    ``model.make_cache`` so every call builds a *fresh* cache list —
    matching :func:`veloxquant_mlx.integration.mlx_vlm_patch.patch_vlm_kv_cache`.
    A prior version closed over a single cache list built once at patch
    time, so a second ``generate()`` call on the same patched model would
    reuse the first call's leftover KV state instead of starting clean.

    Args:
        model: A loaded mlx_lm model instance.
        config: KVCacheConfig describing the quantization scheme.

    Returns:
        The first freshly built cache list (also validates the config
        eagerly). Subsequent ``generate()`` calls receive their own fresh
        lists via the installed ``make_cache`` hook.

    Example::

        config = KVCacheConfig(
            method="turboquant_rvq",
            bit_width_inlier=1,
            seed=42,
        )
        patch_model_kv_cache(model, config)
    """

    def _make_cache(*_args: Any, **_kwargs: Any) -> list[Any]:
        return KVCacheBuilder.for_model(model, config)

    model.make_cache = _make_cache
    caches = _make_cache()
    print(
        f"[veloxquant_mlx] Wired {len(caches)} layer cache(s) with "
        f"{config.method!r} (fresh per call)."
    )
    return caches

"""Monkey-patches an mlx-vlm vision-language model to use VeloxQuant KV caches.

Usage::

    from mlx_vlm import load, generate
    from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache
    from veloxquant_mlx.cache import KVCacheConfig

    model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)
    patch_vlm_kv_cache(model, config)

    # mlx_vlm.generate() now builds a quantized cache automatically
    output = generate(model, processor, prompt, image)

How it hooks in (verified against mlx-vlm 0.6.5):
``mlx_vlm.generate`` builds its cache via
``mlx_vlm.models.cache.make_prompt_cache(model.language_model, ...)``,
which defers to ``model.language_model.make_cache()`` when defined. This
patch overrides exactly that hook — and *only* on the language model.

The top-level model is deliberately left unpatched: mlx-vlm's batch /
session path (``generate/ar.py:_make_cache``) calls the top-level
``model.make_cache()`` and then converts each returned cache with
``to_batch_cache()``, which raises ``ValueError`` on cache types it does
not know — including VeloxQuant's. Batched generation therefore keeps
mlx-vlm's own caches (its built-in ``kv_bits`` quantization still works
there); single-prompt generation gets the VeloxQuant cache.

Unlike :func:`patch_model_kv_cache` (which returns one persistent cache
list), ``make_cache`` here builds a *fresh* cache list on every call, so
repeated ``generate()`` calls never leak KV state between generations.
"""

from __future__ import annotations

import warnings
from typing import Any, List

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig

# Methods that drop or merge tokens to stay within budget. On multimodal
# prompts the image tokens sit in the prompt prefix and can be evicted
# like any other token — quality on vision inputs is unvalidated.
_EVICTION_METHODS = frozenset(
    {
        "snapkv",
        "streaming_llm",
        "h2o",
        "tova",
        "pyramidkv",
        "chunkkv",
        "cam",
        "keyformer",
        "morphkv",
        "kvzip",
        "squeeze",
        "qfilters",
    }
)


def patch_vlm_kv_cache(model: Any, config: KVCacheConfig) -> List[Any]:
    """Wire a quantized KV cache into an mlx-vlm model's generation path.

    Args:
        model: A loaded mlx-vlm model instance (must expose
               ``.language_model``).
        config: KVCacheConfig describing the quantization scheme.

    Returns:
        The first freshly built cache list (also validates the config
        eagerly). Subsequent ``generate()`` calls receive their own
        fresh lists via the installed ``make_cache`` hook.

    Raises:
        ValueError: If ``model`` has no ``language_model`` attribute —
            for text-only mlx_lm models use
            :func:`veloxquant_mlx.integration.mlx_lm_patch.patch_model_kv_cache`.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise ValueError(
            "patch_vlm_kv_cache: model has no .language_model attribute — "
            "this does not look like an mlx-vlm VLM. For text-only mlx_lm "
            "models use patch_model_kv_cache instead."
        )

    if config.method in _EVICTION_METHODS:
        warnings.warn(
            f"patch_vlm_kv_cache: method {config.method!r} evicts or merges "
            f"tokens and may discard image tokens from the prompt prefix. "
            f"Quality on multimodal prompts is unvalidated — prefer a pure "
            f"quantization method (e.g. 'turboquant_rvq', 'kivi', 'vecinfer').",
            stacklevel=2,
        )

    # KVCacheBuilder.for_model resolves VLM wrappers itself when the
    # wrapper exposes .layers (Qwen2-VL style); otherwise build straight
    # from the language model.
    target = model if getattr(model, "layers", None) is not None else lm

    caches = KVCacheBuilder.for_model(target, config)

    def _make_cache(*_args: Any, **_kwargs: Any) -> List[Any]:
        return KVCacheBuilder.for_model(target, config)

    lm.make_cache = _make_cache
    print(
        f"[veloxquant_mlx] Wired {len(caches)} layer cache(s) with "
        f"{config.method!r} into language_model.make_cache (fresh per call)."
    )
    return caches


__all__ = ["patch_vlm_kv_cache"]

"""Library-level prefix-KV-cache reuse for programmatic ``mlx_lm.generate()``
callers (issue #310, Phase 2).

``veloxquant serve`` already gets prefix-cache reuse for free by handing off
to unmodified ``mlx_lm.server.run()``, which drives an ``LRUPromptCache``
internally. A caller using ``mlx_lm.generate()``/``stream_generate()``
directly gets none of that -- every call rebuilds a cache from scratch and
re-prefills the entire prompt, the same shape of gap described in
https://github.com/ollama/ollama/issues/17829. ``PrefixCache`` wraps
``mlx_lm.models.cache.LRUPromptCache`` to close that gap for library users.

Model-identity keying: by default the cache is keyed on ``id(model)``, which
is only valid for the lifetime of that Python object -- reloading the same
weights produces a new id and a cold cache. Pass ``model_key=`` explicitly if
your process reloads model objects across calls but wants cache reuse to
survive that. Note this has an equivalent, not worse, risk profile to
``mlx_lm.server``'s own ``model_key = (model_path, adapter_path,
draft_model_path)``, which is a plain path tuple, not a content hash --
swapping weights at the same path between requests can serve a stale
prefix cache there too. Fixing that is out of scope here: it is pre-existing
``mlx_lm`` behavior, not something introduced by this module.

Usage::

    from mlx_lm import load
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.integration.prefix_cache import PrefixCache

    model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
    prefix_cache = PrefixCache(config)

    text = prefix_cache.generate(model, tokenizer, "long shared system prompt...")
    # A second call sharing that prefix reuses its KV instead of re-prefilling it.
    text2 = prefix_cache.generate(model, tokenizer, "long shared system prompt...and more")
"""

from __future__ import annotations

from typing import Any, Hashable, List, Optional, Tuple, Union

from mlx_lm.models.cache import LRUPromptCache

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.cache.registry import ServeTier, probe_serve_tier

_NOT_TRIMMABLE_NOTE = (
    "method {method!r} does not support prefix-cache trimming (NOT_TRIMMABLE); "
    "only exact full-prompt repeats will be reused, not partial-prefix overlap."
)


class PrefixCache:
    """Reusable prefix-KV cache for programmatic ``mlx_lm.generate()`` callers.

    Thin wrapper over ``mlx_lm.models.cache.LRUPromptCache`` -- the same
    mechanism ``mlx_lm.server`` already drives internally -- giving library
    users the same reuse mechanic without needing the HTTP server.

    Does not use ``patch_model_kv_cache``: that helper pins one eagerly
    built cache list, which would make every ``.fetch()`` miss share state
    with every other miss. A fresh cache list is built per miss instead, via
    ``KVCacheBuilder.for_model`` directly -- mirroring
    ``veloxquant_mlx.cli.serve.attach_cache``'s reasoning exactly.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        *,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
    ) -> None:
        self._config = config
        self._lru = LRUPromptCache(max_size=max_size, max_bytes=max_bytes)
        if probe_serve_tier(config.method) is ServeTier.NOT_TRIMMABLE:
            print(
                f"[veloxquant_mlx] NOTE: {_NOT_TRIMMABLE_NOTE.format(method=config.method)}"
            )

    def _key_for(self, model: Any, model_key: Optional[Hashable]) -> Hashable:
        return model_key if model_key is not None else id(model)

    def fetch(
        self,
        model: Any,
        prompt: List[int],
        *,
        model_key: Optional[Hashable] = None,
    ) -> Tuple[List[Any], List[int]]:
        """Look up the longest cached prefix of ``prompt``.

        Returns ``(cache, rest)`` where ``rest`` is what must be passed as
        ``generate()``'s ``prompt=`` -- NOT the original full prompt.
        ``mlx_lm.generate()``/``stream_generate()`` do not trim the prompt
        against a supplied ``prompt_cache`` themselves; the caller (this
        method) must have already done so, matching exactly what
        ``mlx_lm.server``'s own request handler does by hand.

        On a full miss, builds and returns a fresh cache list via
        ``KVCacheBuilder.for_model`` (raises ``QuantizerConfigError``
        immediately if ``config.method`` is a standalone method that cannot
        serve -- see ``veloxquant_mlx.cache.base.STANDALONE_METHODS``).
        """
        key = self._key_for(model, model_key)
        cache, rest = self._lru.fetch_nearest_cache(key, prompt)
        if cache is None:
            cache = KVCacheBuilder.for_model(model, self._config)
        return cache, rest

    def insert(
        self,
        model: Any,
        prompt: List[int],
        cache: List[Any],
        *,
        model_key: Optional[Hashable] = None,
        cache_type: str = "assistant",
    ) -> None:
        """Store ``cache`` under ``prompt``.

        ``prompt`` must be the FULL sequence the cache now represents --
        the original prompt plus every generated token -- matching
        ``mlx_lm.server``'s own ``cache_key = prompt[:]; cache_key.append(
        gen.token)`` convention. Passing only the original prompt would
        make later fetches miss the generated continuation.
        """
        key = self._key_for(model, model_key)
        self._lru.insert_cache(key, prompt, cache, cache_type=cache_type)

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: Union[str, List[int]],
        *,
        model_key: Optional[Hashable] = None,
        cache_type: str = "assistant",
        **generate_kwargs: Any,
    ) -> str:
        """Convenience wrapper: fetch -> stream_generate -> insert.

        Uses ``stream_generate`` internally (not the top-level
        ``generate()``) because it needs each generated token id to
        reconstruct ``cache_key`` for the final ``insert()`` call --
        ``generate()``'s plain string return value discards them.
        """
        from mlx_lm.generate import stream_generate

        if isinstance(prompt, str):
            token_ids = tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)

        cache, rest = self.fetch(model, token_ids, model_key=model_key)
        cache_key = list(token_ids)

        text_parts: List[str] = []
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=rest,
            prompt_cache=cache,
            **generate_kwargs,
        ):
            text_parts.append(response.text)
            cache_key.append(response.token)

        self.insert(model, cache_key, cache, model_key=model_key, cache_type=cache_type)
        return "".join(text_parts)

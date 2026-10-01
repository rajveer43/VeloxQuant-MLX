"""``veloxquant vlm-serve`` — serve an MLX VLM with a VeloxQuant KV cache.

``veloxquant serve`` is intentionally a thin wrapper around ``mlx_lm.server``
and is therefore for text-only MLX-LM models.  Qwen3.5 is served by
``mlx_vlm.server`` instead.  This launcher installs VeloxQuant's existing VLM
cache hook at MLX-VLM's model-loader boundary, then delegates the HTTP server
and its OpenAI-compatible API to MLX-VLM unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import weakref
from typing import Any

from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.cache.registry import DEFAULT_SERVE_METHOD, get_method
from veloxquant_mlx.integration.chat_templates import ensure_initial_system_prompt_support
from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache


def build_parser() -> argparse.ArgumentParser:
    """Build the small VeloxQuant-specific portion of the VLM CLI."""
    parser = argparse.ArgumentParser(
        prog="veloxquant vlm-serve",
        description="Serve an MLX VLM using a VeloxQuant-compressed KV cache.",
        add_help=False,
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_SERVE_METHOD,
        help=f"VeloxQuant KV-cache method (default: {DEFAULT_SERVE_METHOD}).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=2,
        help="VeloxQuant inlier bit width (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Quantizer seed (default: 42).")
    return parser


class MethodNotServableError(ValueError):
    """Raised when a requested cache method cannot satisfy the serving API."""


def _validate_method(method: str) -> None:
    """Reject cache methods that do not implement MLX's serving contract."""
    try:
        info = get_method(method)
    except KeyError as exc:
        raise MethodNotServableError(str(exc)) from exc
    if not info.serve_tier.is_servable:
        raise MethodNotServableError(
            f"method {method!r} cannot be served: "
            f"{info.unsupported_reason or 'not serving-compatible'}"
        )


def _install_vlm_patch(config: KVCacheConfig, server_package: Any, app_module: Any) -> None:
    """Patch MLX-VLM's cached-model factory exactly once per model instance.

    MLX-VLM routes both startup preload and OpenAI requests through this
    factory.  Its protocol routers resolve the callable from the package at
    request time, so update both the app module and package re-export.
    """
    original = app_module.get_cached_model
    # Keeping weak object references prevents both id() reuse and retaining a
    # model solely because it was once loaded by the server.
    patched_models: weakref.WeakSet[Any] = weakref.WeakSet()

    def get_cached_model(*args: Any, **kwargs: Any):
        model, processor, model_config = original(*args, **kwargs)
        if model not in patched_models:
            # MLX-VLM returns a processor that owns the text tokenizer. Keep
            # the OpenAI message array intact; only install a compatibility
            # template when its tokenizer cannot render a leading system role.
            tokenizer = getattr(processor, "tokenizer", processor)
            model_name = str(args[0] if args else kwargs.get("model_path", ""))
            if ensure_initial_system_prompt_support(tokenizer, model_name):
                print("[veloxquant vlm-serve] installed Mistral system-message chat template")
            patch_vlm_kv_cache(model, config)
            patched_models.add(model)
        return model, processor, model_config

    app_module.get_cached_model = get_cached_model
    server_package.get_cached_model = get_cached_model


def main(argv: list[str] | None = None) -> None:
    """Install VeloxQuant's VLM hook, then invoke the MLX-VLM server CLI."""
    parser = build_parser()
    own_args, mlx_vlm_args = parser.parse_known_args(argv)
    try:
        _validate_method(own_args.method)
    except MethodNotServableError as exc:
        raise SystemExit(f"error: {exc}") from None
    config = KVCacheConfig(
        method=own_args.method,
        bit_width_inlier=own_args.bits,
        seed=own_args.seed,
    )

    try:
        import mlx_vlm.server as server_package
        from mlx_vlm.server.cli import main as mlx_vlm_main

        # ``mlx_vlm.server`` exports its FastAPI application as ``app``, so
        # a dotted import can bind that object rather than the ``.app``
        # module. Importlib always returns the module containing the loader.
        app_module = importlib.import_module("mlx_vlm.server.app")
    except ImportError:
        raise SystemExit(
            "error: vlm-serve requires mlx-vlm. Install it with "
            "'uv sync --extra vlm' or run 'uv run --extra vlm veloxquant vlm-serve ...'."
        ) from None

    _install_vlm_patch(config, server_package, app_module)
    print(
        f"[veloxquant vlm-serve] using method={own_args.method!r} bits={own_args.bits}; "
        "the first loaded VLM will be patched."
    )

    previous_argv = sys.argv
    try:
        sys.argv = ["mlx_vlm.server", *mlx_vlm_args]
        mlx_vlm_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

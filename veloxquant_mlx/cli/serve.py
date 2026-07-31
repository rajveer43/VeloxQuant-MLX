"""``veloxquant serve`` — OpenAI-compatible server with a VeloxQuant KV cache.

Thin launcher, per option (a) of issue #27: load the model, wire our cache in
via ``patch_model_kv_cache``, hand off to ``mlx_lm.server``. The serving
contract itself is inherited — 35 of 40 caches subclass ``mlx_lm``'s ``KVCache``
and get ``update_and_fetch`` / ``trim`` / ``nbytes`` for free.

What is *not* thin is the honesty layer, and it is the reason this file is
longer than the ten lines #27 predicted:

* An unsupported method must fail before the model loads, not with an
  ``AttributeError`` on the first HTTP request.
* Byte counters are accounting-only. The cache stores dequantized fp16, so
  ``nbytes`` over-reports and no memory is actually saved. A server is exactly
  where users assume otherwise, so the warning is unconditional.
* There is no fp16 fallback, silent or otherwise. A method that cannot serve
  stops the process.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from veloxquant_mlx.cache.registry import DEFAULT_SERVE_METHOD, get_method

#: stdout handshake for the control panel. Printed once the model is loaded
#: and the cache is wired, immediately before mlx_lm.server binds the port.
#: Lets the panel distinguish "starting" from "running" without racing a
#: health poll against a slow model load.
READY_PREFIX = "VELOXQUANT_READY "

ACCOUNTING_WARNING = (
    "compression is accounting-only. Caches store dequantized fp16 tensors, so "
    "reported byte counters measure compression fidelity, not runtime memory "
    "saved. Do not read this server's memory numbers as RSS reduction (issue #27)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veloxquant serve",
        description="Serve an MLX model over an OpenAI-compatible API with a "
                    "VeloxQuant-compressed KV cache.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Hugging Face model id (e.g. mlx-community/...) or local path.",
    )
    parser.add_argument(
        "--method", default=DEFAULT_SERVE_METHOD,
        help=f"KV-cache method (default: {DEFAULT_SERVE_METHOD}). "
             "Run 'veloxquant methods' to list options.",
    )
    parser.add_argument(
        "--bits", type=int, default=2,
        help="Inlier bit width for the cache (default: 2).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000).")
    parser.add_argument("--seed", type=int, default=42, help="Quantizer seed (default: 42).")
    parser.add_argument(
        "--adapter-path", default=None, help="Optional LoRA adapter path passed to mlx_lm.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Default max tokens per completion (default: 512).",
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Default sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Default nucleus sampling p (default: 1.0).",
    )
    return parser


def validate_method(method: str) -> None:
    """Reject unservable methods before the expensive model load.

    Deliberately fatal. #27's crash tier includes ``turboquant_prod`` — the
    *library's* default — so silently substituting a working method here would
    mean users think they benchmarked one thing and served another.
    """
    try:
        info = get_method(method)
    except KeyError as exc:
        raise SystemExit(f"error: {exc}") from None

    if not info.serve_tier.is_servable:
        reason = info.unsupported_reason or "method is not serve-compatible"
        raise SystemExit(
            f"error: method {method!r} cannot be served.\n"
            f"  reason: {reason}\n"
            f"  This is issue #27's crash tier. No fp16 fallback is applied, "
            f"because serving a\n  different method than the one requested would "
            f"misrepresent your results.\n"
            f"  Run 'veloxquant methods --servable-only' to see valid choices "
            f"(default: {DEFAULT_SERVE_METHOD})."
        )


def _warn(message: str) -> None:
    print(f"[veloxquant serve] {message}", file=sys.stderr)


def build_config(args: argparse.Namespace) -> Any:
    from veloxquant_mlx.cache import KVCacheConfig

    return KVCacheConfig(
        method=args.method, bit_width_inlier=args.bits, seed=args.seed,
    )


def attach_cache(model: Any, config: Any) -> int:
    """Point ``model.make_cache`` at a freshly built VeloxQuant cache list.

    Not ``patch_model_kv_cache``: that helper builds one list eagerly and pins
    it, which suits ``generate()`` but would make every concurrent request
    share one cache. Building per call gives each request its own.

    Must run on the thread that will generate — see ``_Provider._load``.
    """
    from veloxquant_mlx.cache.base import KVCacheBuilder

    probe = KVCacheBuilder.for_model(model, config)
    n_layers = len(probe)
    del probe

    model.make_cache = lambda *_a, **_kw: KVCacheBuilder.for_model(model, config)
    return n_layers


def emit_ready(args: argparse.Namespace, n_caches: int) -> None:
    """Print the machine-readable handshake the control panel waits for."""
    base = f"http://{args.host}:{args.port}"
    payload = {
        "schema_version": 1,
        "model": args.model,
        "method": args.method,
        "bits": args.bits,
        "host": args.host,
        "port": args.port,
        "layer_caches": n_caches,
        # Only endpoints mlx_lm.server actually serves. The panel renders this
        # list verbatim, so inventing entries here would put fake copy-buttons
        # in the UI.
        "endpoints": {
            "openai_base_url": f"{base}/v1",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "models": f"{base}/v1/models",
        },
        "accounting_only": True,
        "accounting_note": ACCOUNTING_WARNING,
    }
    print(READY_PREFIX + json.dumps(payload), flush=True)


def _mlx_server_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the arg namespace ``mlx_lm.server`` expects.

    ``ModelProvider`` and ``APIHandler`` read a wide and growing set of fields
    off ``cli_args`` (``draft_model``, ``pipeline``, ``prompt_cache_size``,
    ``decode_concurrency``, …). Hand-listing them means a new upstream option
    breaks us with an ``AttributeError`` at request time, so instead we take
    mlx_lm's *own* parser defaults and override only the flags we expose.
    """
    import mlx_lm.server as _server

    parser = _capture_mlx_parser(_server)
    ns = parser.parse_args([])

    ns.model = args.model
    ns.adapter_path = args.adapter_path
    ns.host = args.host
    ns.port = args.port
    ns.max_tokens = args.max_tokens
    ns.temp = args.temp
    ns.top_p = args.top_p
    return ns


def _capture_mlx_parser(server_module: Any) -> argparse.ArgumentParser:
    """Extract the parser built inside ``mlx_lm.server.main()``.

    Upstream builds it inline in ``main()`` and never exposes it. We run
    ``main()`` with a stub that records the parser and aborts before it can
    bind a port or load a model.

    The stub is installed as a module-level ``argparse`` shim rather than by
    reassigning ``argparse.ArgumentParser`` itself: the subclass would then
    inherit from the patched global and recurse infinitely through its own
    ``__init__``.
    """
    import types

    captured: List[argparse.ArgumentParser] = []

    class _StopParsing(Exception):
        pass

    real_cls = argparse.ArgumentParser

    def _factory(*a: Any, **kw: Any) -> argparse.ArgumentParser:
        parser = real_cls(*a, **kw)
        parser.parse_args = _abort  # type: ignore[method-assign]
        captured.append(parser)
        return parser

    def _abort(*a: Any, **kw: Any):
        raise _StopParsing

    shim = types.SimpleNamespace(**{
        name: getattr(argparse, name) for name in dir(argparse)
        if not name.startswith("__")
    })
    shim.ArgumentParser = _factory

    original = server_module.argparse
    server_module.argparse = shim
    try:
        server_module.main()
    except _StopParsing:
        pass
    except Exception as exc:
        raise SystemExit(
            "error: could not read mlx_lm.server's argument defaults "
            f"({type(exc).__name__}: {exc}). This usually means the installed "
            "mlx_lm version changed its server entry point. Please report this "
            "with your mlx_lm version."
        ) from None
    finally:
        server_module.argparse = original

    if not captured:
        raise SystemExit(
            "error: mlx_lm.server.main() built no argument parser. This usually "
            "means the installed mlx_lm version changed its server entry point. "
            "Please report this with your mlx_lm version."
        )

    parser = captured[0]
    del parser.parse_args  # drop the abort stub; fall back to the real method
    return parser


def run_server(args: argparse.Namespace) -> None:
    """Hand off to mlx_lm's server, wiring our cache in as it loads the model.

    The model is loaded by ``mlx_lm`` itself, inside its generation thread,
    rather than up front on the main thread. MLX arrays are bound to the
    thread that created them, so a main-thread model produces
    "There is no Stream(gpu, N) in current thread." the first time the
    generation thread evaluates cache state. Upstream sidesteps this by
    calling ``load_default()`` from within ``_generate``; hooking ``_load``
    puts our patch on that same thread.
    """
    from mlx_lm.server import ModelProvider, run

    server_args = _mlx_server_args(args)
    config = build_config(args)
    state = {"announced": False}

    class _Provider(ModelProvider):
        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            requested = self._model_map.get(model_path, model_path)
            if requested not in (args.model, "default_model", None):
                # Refusing rather than loading keeps the served model identical
                # to the one whose method/bits the panel is displaying.
                raise ValueError(
                    f"this server is pinned to {args.model!r} with the "
                    f"{args.method!r} KV cache; it will not load {requested!r}. "
                    "Start a second server for a different model."
                )

            _warn(f"loading model {args.model!r} ...")
            super()._load(args.model, args.adapter_path, None)

            n_layers = attach_cache(self.model, config)

            # Recompute after re-patching make_cache: upstream derived this from
            # the stock cache, and our caches decide batchability themselves.
            # Left stale/False, requests take the unbatched _serve_single path,
            # which generates on the HTTP thread with no stream scope.
            from mlx_lm.models.cache import make_prompt_cache

            self.is_batchable = all(
                hasattr(c, "merge") for c in make_prompt_cache(self.model)
            )
            if not self.is_batchable:
                _warn(
                    f"method {args.method!r} has no merge(); serving unbatched "
                    "(slower, one request at a time)."
                )

            _warn(
                f"wired {n_layers} layer cache(s) with "
                f"method={args.method!r} bits={args.bits}"
            )
            if not state["announced"]:
                emit_ready(args, n_layers)
                state["announced"] = True

    _warn(f"listening on http://{args.host}:{args.port}/v1")
    if args.host == "0.0.0.0":
        _warn("bound to 0.0.0.0 — reachable from your local network, with no auth.")

    try:
        run(args.host, args.port, _Provider(server_args))
    except KeyboardInterrupt:
        _warn("shutting down.")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    validate_method(args.method)
    _warn(f"NOTE: {ACCOUNTING_WARNING}")

    # The model loads lazily inside mlx_lm's generation thread, so the READY
    # handshake is emitted from there once the cache is actually wired.
    run_server(args)


if __name__ == "__main__":
    main()

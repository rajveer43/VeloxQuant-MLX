"""``GET /v1/kv/stats`` — read-only cache telemetry for the control panel (#27, #36).

Subclasses ``mlx_lm``'s ``APIHandler`` rather than patching it: ``run()``
already accepts ``handler_class`` as a keyword
(``mlx_lm/server.py:run(..., handler_class=APIHandler)``), so this is the
extension point upstream provides, not a monkeypatch.

Live-cache handle
------------------
``mlx_lm.server.ResponseGenerator`` does not keep a durable reference to "the
cache currently being generated with" — each request either pulls a snapshot
out of its ``LRUPromptCache`` trie or builds a fresh one via
``model.make_cache()``, as a local variable inside ``_generate``'s single
background thread. There is nothing on ``ResponseGenerator`` itself to read.

``KVCacheBuilder.for_model`` is the one place that actually constructs a
model's per-layer cache list (see ``attach_cache`` in ``cli/serve.py``, which
points ``model.make_cache`` at it), so this module records the *most
recently built* cache list there instead of reaching into
``ResponseGenerator``. That list is exactly what's live during whichever
request is currently generating — ``make_cache`` is called once per request,
and a fresh call replaces the previous list.

Aggregation must not allocate or force evaluation: telemetry that perturbs
generation is worse than none. Every property read here
(``compressed_key_bytes`` and friends) is a plain Python int already
accumulated by each cache's own ``update_and_fetch``, not an MLX array — see
each cache's ``@property`` definitions (e.g. ``cachegen_cache.py``).
"""

from __future__ import annotations

import json
from typing import Any

from mlx_lm.server import APIHandler

#: The most recently built per-layer cache list, set by `record_live_caches`
#: (called from `attach_cache` in `cli/serve.py`). Module-level rather than
#: an instance attribute: `APIHandler` instances are constructed fresh per
#: request by `_run_http_server`'s factory closure (`mlx_lm/server.py`), so
#: there is no single long-lived handler instance to hold this on.
_LIVE_CACHES: list[Any] | None = None
_LIVE_METHOD: str | None = None
_LIVE_BITS: int | None = None


def record_live_caches(caches: list[Any], *, method: str, bits: int | None) -> None:
    """Called once per built cache list, from `attach_cache` in `cli/serve.py`."""
    global _LIVE_CACHES, _LIVE_METHOD, _LIVE_BITS
    _LIVE_CACHES = caches
    _LIVE_METHOD = method
    _LIVE_BITS = bits


class TelemetryHandler(APIHandler):
    """Adds `GET /v1/kv/stats`; everything else falls through to `APIHandler`."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/kv/stats":
            self._handle_kv_stats()
            return
        super().do_GET()

    def _handle_kv_stats(self) -> None:
        payload = build_stats_payload(
            caches=_LIVE_CACHES, method=_LIVE_METHOD, bits=_LIVE_BITS
        )
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def build_stats_payload(
    *, caches: list[Any] | None, method: str | None, bits: int | None
) -> dict[str, Any]:
    """Pure function, unit-testable without an HTTP server or a real model.

    Encodes Finding A from docs/control-panel-enhancements.md: never guess a
    whole-cache ratio from partial data, and never render a bare 0 for "not
    reported" — those two are indistinguishable to a user, and #27 is explicit
    that indistinguishable-from-real is the failure mode to avoid.
    """
    from veloxquant_mlx.cache.registry import TelemetryCoverage, get_method

    info = get_method(method) if method else None
    coverage = info.coverage if info is not None else TelemetryCoverage.NONE

    result: dict[str, Any] = {
        "method": method,
        "bits": bits,
        "accounting_only": True,
        "coverage": coverage.value,
        "keys": None,
        "values": None,
        "tokens": None,
        "memory": _memory_block(),
    }

    if caches is None:
        result["coverage"] = TelemetryCoverage.NONE.value
        result["not_reported_reason"] = "no server is generating yet"
        return result

    if coverage is TelemetryCoverage.NONE:
        tokens = _token_counts(caches)
        if tokens is None:
            result["not_reported_reason"] = "this method does not report byte or token counters"
        else:
            result["tokens"] = tokens
        return result

    keys, values = _byte_counts(caches, coverage)
    result["keys"] = keys
    result["values"] = values if coverage is TelemetryCoverage.KEYS_AND_VALUES else None
    return result


def _memory_block() -> dict[str, Any]:
    """Measured, not estimated — mirrors `ui/memory.py`'s honesty contract.

    This runs *inside* the inference server process, unlike `ui/memory.py`'s
    RSS read (which measures the panel's own process via `psutil.Process(pid)`
    from outside). `mx.get_active_memory()` here is finally the number
    `ui/memory.py`'s `_mlx_memory()` withholds as "coming soon" — because it
    would otherwise measure the wrong process (the panel, not the server).
    """
    try:
        import mlx.core as mx

        mlx_active: int | None = mx.get_active_memory()
        mlx_peak: int | None = mx.get_peak_memory()
    except Exception:
        mlx_active = mlx_peak = None

    rss: int | None
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = None

    return {
        "rss_bytes": rss,
        "mlx_active_bytes": mlx_active,
        "mlx_peak_bytes": mlx_peak,
        "source": "measured",
    }


def _byte_counts(
    caches: list[Any], coverage: Any
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Sum per-layer compressed/fp16 byte counters across the cache list.

    Reads plain ints already accumulated by each cache (see module docstring)
    — no array evaluation, no allocation.
    """
    key_compressed = key_fp16 = 0
    value_compressed = value_fp16 = 0
    have_values = False

    for layer_cache in caches:
        key_compressed += int(getattr(layer_cache, "compressed_key_bytes", 0))
        key_fp16 += int(getattr(layer_cache, "fp16_key_bytes", 0))
        if hasattr(layer_cache, "compressed_value_bytes"):
            have_values = True
            value_compressed += int(getattr(layer_cache, "compressed_value_bytes", 0))
            value_fp16 += int(getattr(layer_cache, "fp16_value_bytes", 0))

    keys = {
        "compressed_bytes": key_compressed,
        "fp16_bytes": key_fp16,
        "ratio": round(key_fp16 / key_compressed, 2) if key_compressed else None,
    }
    values = None
    if have_values:
        values = {
            "compressed_bytes": value_compressed,
            "fp16_bytes": value_fp16,
            "ratio": round(value_fp16 / value_compressed, 2) if value_compressed else None,
        }
    return keys, values


def _token_counts(caches: list[Any]) -> dict[str, int] | None:
    """Tokens seen vs. currently retained, summed across layers.

    Eviction caches (h2o, snapkv, tova, streaming_llm, pyramidkv, …) share
    `tokens_seen`/`tokens_kept` properties (see e.g. `h2o_cache.py`). A method
    with neither — i.e. genuinely no telemetry, not just a different name —
    returns `None` so the caller can report "not reported" rather than a
    fabricated `{"seen": 0, "retained": 0}`.
    """
    has_any = any(
        hasattr(c, "tokens_seen") or hasattr(c, "tokens_kept") for c in caches
    )
    if not has_any:
        return None

    seen = sum(int(getattr(c, "tokens_seen", 0)) for c in caches)
    retained = sum(int(getattr(c, "tokens_kept", 0)) for c in caches)
    return {"seen": seen, "retained": retained}

"""Control-plane HTTP server for the VeloxQuant panel (#34).

Serves the static panel and a small JSON API the page drives. Stdlib only —
the panel must not add a runtime dependency to the core package.

Security: this API spawns processes, so it binds loopback unconditionally.
That is deliberately *not* configurable, and is separate from the inference
server's own ``--host``, which the user may set to ``0.0.0.0``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from veloxquant_mlx.ui.config import load_config, save_config
from veloxquant_mlx.ui.supervisor import ServerSupervisor

STATIC_DIR = Path(__file__).parent / "static"

#: The panel's own bind address. Loopback only — see module docstring.
CONTROL_HOST = "127.0.0.1"

#: Host names a legitimate same-machine browser can use to reach the panel.
#: Anything else in the ``Host`` header is a DNS-rebinding attempt.
_LOCAL_HOSTS = ("127.0.0.1", "localhost")

#: Largest request body accepted. The API's payloads are a few hundred bytes.
MAX_BODY_BYTES = 1 << 20


class _BadRequest(Exception):
    """A request the client got wrong; becomes a JSON error with ``status``."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class PanelHandler(BaseHTTPRequestHandler):
    """Request handler serving the panel's static assets and JSON API.

    Bound to a single :class:`~veloxquant_mlx.ui.supervisor.ServerSupervisor`
    instance (injected onto the ``supervisor`` class attribute by
    :func:`serve_panel`, since :class:`~http.server.ThreadingHTTPServer`
    instantiates a fresh handler per request). See :meth:`do_GET` and
    :meth:`do_POST` for the route table.
    """

    supervisor: ServerSupervisor  # injected by serve_panel

    # --- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress the base class's stderr access log; server logs go to the UI instead."""
        pass  # the panel's own access log is noise; server logs go to the UI

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw_length)
        except ValueError:
            raise _BadRequest("invalid Content-Length") from None
        if length < 0:
            raise _BadRequest("invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise _BadRequest("request body too large", status=413)
        if not length:
            return {}

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise _BadRequest("Content-Type must be application/json", status=415)

        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _BadRequest("request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise _BadRequest("request body must be a JSON object")
        return body

    def _check_origin(self) -> None:
        """Refuse requests that did not come from the panel page itself.

        The panel binds loopback, but a browser on this machine will happily
        send a web page's request to ``127.0.0.1``. Two attacks follow:
        a cross-site form/``fetch`` POST (which needs no CORS preflight when
        sent as ``text/plain``) can start a server or rewrite the saved
        config, and DNS rebinding lets a hostile page read the API too.
        Checking ``Host`` stops rebinding; checking ``Origin`` (sent by
        browsers on cross-origin requests) stops the cross-site POST.
        """
        port = self.server.server_address[1]
        allowed_hosts = {f"{h}:{port}" for h in _LOCAL_HOSTS}

        host = (self.headers.get("Host") or "").lower()
        if host not in allowed_hosts:
            raise _BadRequest("forbidden host", status=403)

        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in {f"http://{h}" for h in allowed_hosts}:
            raise _BadRequest("forbidden origin", status=403)

    def _dispatch(self, handler: Any) -> None:
        """Run ``handler`` so every failure becomes a JSON response, never a dropped socket."""
        try:
            self._check_origin()
            handler()
        except _BadRequest as exc:
            self._send_json({"error": str(exc)}, status=exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-response; nothing to send
        except Exception as exc:  # noqa: BLE001 - the UI needs a body, not a closed socket
            self._send_json({"error": f"internal error: {exc}"}, status=500)

    def _send_static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")

        # Contain traversal: normalise, then refuse anything outside STATIC_DIR.
        # os.path (not pathlib) on purpose: this normpath+startswith form is the
        # sanitizer CodeQL's py/path-injection query recognises.
        static_root = os.path.realpath(STATIC_DIR)
        fullpath = os.path.realpath(os.path.normpath(os.path.join(static_root, name)))  # noqa: PTH118
        if not fullpath.startswith(static_root + os.sep) or not os.path.isfile(fullpath):  # noqa: PTH113
            self.send_error(404, "not found")
            return
        target = Path(fullpath)

        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(target.suffix, "application/octet-stream")

        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        """Serve a GET request; see :meth:`_handle_get` for the route table."""
        self._dispatch(self._handle_get)

    def do_POST(self) -> None:  # noqa: N802
        """Serve a POST request; see :meth:`_handle_post` for the route table."""
        self._dispatch(self._handle_post)

    def _handle_get(self) -> None:
        """Serve a JSON API route under ``/api/*``, or fall back to a static file.

        Routes: ``/api/methods`` (available quantization methods),
        ``/api/status`` (supervisor state + version), ``/api/logs``
        (supervised-process output, paginated via ``?since=``),
        ``/api/config`` (persisted panel settings), ``/api/models`` (locally
        cached models), ``/api/models/search`` (Hub search via ``?q=``), and
        ``/api/memory`` (memory usage of the supervised process). Anything
        else is served as a static file from ``STATIC_DIR`` by
        :meth:`_send_static`.
        """
        route = self.path.split("?")[0]

        if route == "/api/methods":
            from veloxquant_mlx.cache.registry import (
                DEFAULT_SERVE_METHOD,
                list_methods,
            )

            self._send_json(
                {
                    "default_serve_method": DEFAULT_SERVE_METHOD,
                    "accounting_only": True,
                    "methods": [i.to_dict() for i in list_methods()],
                }
            )
            return

        if route == "/api/status":
            from veloxquant_mlx import __version__

            status = self.supervisor.status()
            status["version"] = __version__
            self._send_json(status)
            return

        if route == "/api/logs":
            qs = parse_qs(urlparse(self.path).query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except ValueError:
                raise _BadRequest("since must be an integer") from None
            self._send_json(self.supervisor.logs(since=since))
            return

        if route == "/api/config":
            self._send_json(load_config())
            return

        if route == "/api/models":
            from veloxquant_mlx.ui.models import local_models

            self._send_json({"models": local_models()})
            return

        if route == "/api/models/search":
            from veloxquant_mlx.ui.models import search_hub_models

            qs = parse_qs(urlparse(self.path).query)
            query = (qs.get("q", [""])[0]).strip()
            self._send_json({"models": search_hub_models(query) if query else []})
            return

        if route == "/api/memory":
            from veloxquant_mlx.ui.memory import memory_report

            status = self.supervisor.status()
            self._send_json(memory_report(pid=status.get("pid")))
            return

        self._send_static(route)

    def _handle_post(self) -> None:
        """Handle the panel's mutating routes: start/stop the server, or save config.

        Routes: ``/api/start`` (launch ``veloxquant serve`` with the posted
        config, returning 400 on invalid model/method/overrides),
        ``/api/stop`` (terminate the supervised process, returning 500 on
        unexpected failure rather than leaking a raw exception), and
        ``/api/config`` (merge and persist the posted settings, returning
        the full merged config). Any other route is a 404.
        """
        route = self.path.split("?")[0]

        if route == "/api/start":
            config = self._read_json()
            try:
                status = self.supervisor.start(config)
            except (ValueError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            save_config(config)
            self._send_json(status)
            return

        if route == "/api/stop":
            try:
                status = self.supervisor.stop()
            except Exception as exc:  # noqa: BLE001 - never leak a raw 500 to the UI
                self._send_json({"error": str(exc)}, status=500)
                return
            self._send_json(status)
            return

        if route == "/api/config":
            config = self._read_json()
            save_config(config)
            self._send_json(load_config())
            return

        self.send_error(404, "not found")


def serve_panel(port: int = 7860, open_browser: bool = True) -> None:
    """Start the control panel's HTTP server and block until interrupted.

    Binds ``CONTROL_HOST`` (loopback only, unconditionally — see module
    docstring) on ``port``, optionally opens the panel URL in the default
    browser, and serves until ``Ctrl-C``. Always stops any running
    supervised ``veloxquant serve`` child and closes the HTTP server on
    exit, so interrupting the panel never leaves an orphaned inference
    server holding a port.
    """
    supervisor = ServerSupervisor()

    handler = type("BoundPanelHandler", (PanelHandler,), {"supervisor": supervisor})
    httpd = ThreadingHTTPServer((CONTROL_HOST, port), handler)

    url = f"http://{CONTROL_HOST}:{port}/"
    print(f"[veloxquant panel] control panel at {url}")
    print("[veloxquant panel] press Ctrl-C to quit")

    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)

    # The child runs in its own session (so stop() can signal its whole tree),
    # which means a SIGTERM/SIGHUP to the panel -- `kill`, a closed terminal,
    # launchd -- does not reach it, and the default handlers skip the `finally`
    # below. Route both through the same path Ctrl-C takes.
    def _shutdown(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        # signal.signal only works on the main thread; embedders calling
        # serve_panel from a worker thread keep the default behaviour.
        with contextlib.suppress(ValueError):
            previous[sig] = signal.signal(sig, _shutdown)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[veloxquant panel] shutting down ...")
    finally:
        # Never leave an orphaned inference server holding a port.
        supervisor.stop()
        httpd.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

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
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from veloxquant_mlx.ui.config import load_config, save_config
from veloxquant_mlx.ui.supervisor import ServerSupervisor

STATIC_DIR = Path(__file__).parent / "static"

#: The panel's own bind address. Loopback only — see module docstring.
CONTROL_HOST = "127.0.0.1"


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
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def _send_static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / name).resolve()

        # Contain traversal: refuse anything resolving outside STATIC_DIR.
        static_root = STATIC_DIR.resolve()
        if (target != static_root and static_root not in target.parents) or not target.is_file():
            self.send_error(404, "not found")
            return

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
            since = 0
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(self.path).query)
                since = int(qs.get("since", ["0"])[0])
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
            from urllib.parse import parse_qs, urlparse

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

    def do_POST(self) -> None:  # noqa: N802
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

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[veloxquant panel] shutting down ...")
    finally:
        # Never leave an orphaned inference server holding a port.
        supervisor.stop()
        httpd.server_close()

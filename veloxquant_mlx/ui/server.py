"""Control-plane HTTP server for the VeloxQuant panel (#34).

Serves the static panel and a small JSON API the page drives. Stdlib only —
the panel must not add a runtime dependency to the core package.

Security: this API spawns processes, so it binds loopback unconditionally.
That is deliberately *not* configurable, and is separate from the inference
server's own ``--host``, which the user may set to ``0.0.0.0``.
"""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from veloxquant_mlx.ui.config import load_config, save_config
from veloxquant_mlx.ui.supervisor import ServerSupervisor

STATIC_DIR = Path(__file__).parent / "static"

#: The panel's own bind address. Loopback only — see module docstring.
CONTROL_HOST = "127.0.0.1"


class PanelHandler(BaseHTTPRequestHandler):
    supervisor: ServerSupervisor  # injected by serve_panel

    # --- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # the panel's own access log is noise; server logs go to the UI

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
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
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
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
        route = self.path.split("?")[0]

        if route == "/api/methods":
            from veloxquant_mlx.cache.registry import (
                DEFAULT_SERVE_METHOD,
                list_methods,
            )

            self._send_json({
                "default_serve_method": DEFAULT_SERVE_METHOD,
                "accounting_only": True,
                "methods": [i.to_dict() for i in list_methods()],
            })
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

        self._send_static(route)

    def do_POST(self) -> None:  # noqa: N802
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
            self._send_json(self.supervisor.stop())
            return

        if route == "/api/config":
            config = self._read_json()
            save_config(config)
            self._send_json(load_config())
            return

        self.send_error(404, "not found")


def serve_panel(port: int = 7860, open_browser: bool = True) -> None:
    supervisor = ServerSupervisor()

    handler = type("BoundPanelHandler", (PanelHandler,), {"supervisor": supervisor})
    httpd = ThreadingHTTPServer((CONTROL_HOST, port), handler)

    url = f"http://{CONTROL_HOST}:{port}/"
    print(f"[veloxquant panel] control panel at {url}")
    print("[veloxquant panel] press Ctrl-C to quit")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[veloxquant panel] shutting down ...")
    finally:
        # Never leave an orphaned inference server holding a port.
        supervisor.stop()
        httpd.server_close()

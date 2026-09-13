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
    supervisor: ServerSupervisor  # injected by serve_panel

    # --- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
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

    def _fetch_telemetry(self) -> dict[str, Any]:
        """Proxy the child server's ``/v1/kv/stats``.

        The panel and the inference server are different processes on
        different ports — the panel never imports the server, only pipes its
        stdout/stderr (see module docstring) — so this must be an HTTP round
        trip, not an in-process read. ``urllib.request`` is stdlib, matching
        this module's "no runtime dependency" constraint.

        A short timeout keeps a wedged or slow-to-respond child from
        blocking this worker thread for long; ``ThreadingHTTPServer`` means
        that only stalls one request, not the whole panel.
        """
        status = self.supervisor.status()
        ready = status.get("ready")
        if status.get("state") != "running" or not ready:
            return {"available": False, "reason": "no server is running"}

        url = (ready.get("endpoints") or {}).get("kv_stats")
        if not url:
            # A server started by an older veloxquant version, before this
            # endpoint existed, whose handshake predates the "kv_stats" key.
            return {"available": False, "reason": "server predates telemetry"}

        import urllib.error
        import urllib.request
        from urllib.parse import urlparse

        # url is built by our own emit_ready() (cli/serve.py) as
        # "http://<host>:<port>/v1/kv/stats" -- never user input -- but
        # checking the scheme here keeps urlopen from ever being handed a
        # file:// or other unexpected scheme if that assumption ever breaks.
        if urlparse(url).scheme != "http":
            return {"available": False, "reason": "unexpected telemetry URL scheme"}

        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                stats = json.loads(resp.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": str(exc)}

        return {"available": True, **stats}

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
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

        if route == "/api/telemetry":
            self._send_json(self._fetch_telemetry())
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

        if route == "/api/chat":
            self._proxy_chat()
            return

        self.send_error(404, "not found")

    def _proxy_chat(self) -> None:
        """Relay one chat turn to the running child's /v1/chat/completions.

        Byte-for-byte SSE relay, not a reconstruction: the child already
        speaks OpenAI-compatible SSE (mlx_lm.server, per cli/serve.py's
        module docstring), so this forwards its stream rather than parsing
        and re-emitting it -- the lowest-risk way to add this codebase's
        first streaming route.

        Proxied rather than let the browser hit the child directly: the
        child may be bound to 0.0.0.0 with no auth (the panel's own
        host-warning banner covers that choice), and the panel is the one
        place that already knows whether a server is running at all.
        """
        status = self.supervisor.status()
        ready = status.get("ready")
        if status.get("state") != "running" or not ready:
            self._send_json({"error": "no server is running"}, status=409)
            return

        body = self._read_json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json({"error": "messages must be a non-empty list"}, status=400)
            return

        chat_url = (ready.get("endpoints") or {}).get("chat_completions")
        if not chat_url:
            self._send_json({"error": "server predates chat proxying"}, status=409)
            return

        import urllib.error
        import urllib.request
        from urllib.parse import urlparse

        if urlparse(chat_url).scheme != "http":
            self._send_json({"error": "unexpected chat endpoint scheme"}, status=500)
            return

        upstream_payload = json.dumps(
            {
                "model": ready.get("model"),
                "messages": messages,
                "stream": True,
                "max_tokens": body.get("max_tokens", 512),
                "temperature": body.get("temperature", 0.7),
            }
        ).encode()

        req = urllib.request.Request(
            chat_url,
            data=upstream_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            upstream = urllib.request.urlopen(req, timeout=120)  # noqa: S310
        except (OSError, urllib.error.URLError) as exc:
            self._send_json({"error": str(exc)}, status=502)
            return

        with upstream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The browser tab closed or navigated away mid-stream -- the
                # upstream request above still completes server-side (mlx_lm
                # keeps generating), there is simply no one left to write to.
                pass


def serve_panel(port: int = 7860, open_browser: bool = True) -> None:
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

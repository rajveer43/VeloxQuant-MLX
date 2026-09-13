"""Tests for the panel's Slice 2 proxy routes: /api/telemetry and /api/chat.

The panel and the inference server are different processes on different
ports (see veloxquant_mlx/cli/telemetry.py and ui/server.py's module
docstring) -- these routes proxy the child server's /v1/kv/stats and
/v1/chat/completions rather than reading them in-process, so these tests
exercise the proxy logic against the supervisor's reported state without
spinning up a real model or inference server.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from veloxquant_mlx.ui.server import PanelHandler
from veloxquant_mlx.ui.supervisor import ServerSupervisor


@pytest.fixture()
def panel():
    supervisor = ServerSupervisor()
    handler = type("T", (PanelHandler,), {"supervisor": supervisor})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", supervisor
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return json.loads(resp.read())


def _post(base, path, payload, expect_json=True):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if expect_json else body)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, (json.loads(body) if expect_json else body)


def test_telemetry_unavailable_when_stopped(panel):
    base, _ = panel
    body = _get(base, "/api/telemetry")
    assert body == {"available": False, "reason": "no server is running"}


def test_telemetry_unavailable_when_ready_predates_kv_stats(panel):
    """A child started by an older veloxquant version has no "kv_stats" key.

    The proxy must say so rather than crash on a missing dict key or, worse,
    guess a URL that was never advertised (rule 4: endpoints advertised only
    if actually served).
    """
    base, supervisor = panel
    supervisor._state = "running"
    supervisor._ready = {
        "model": "m",
        "method": "turboquant_rvq",
        "endpoints": {"chat_completions": "http://127.0.0.1:9/v1/chat/completions"},
    }

    body = _get(base, "/api/telemetry")
    assert body == {"available": False, "reason": "server predates telemetry"}


def test_telemetry_proxies_a_live_child(panel):
    """With a running child advertising kv_stats, the panel forwards its JSON."""
    from http.server import BaseHTTPRequestHandler

    fake_stats = {"method": "turboquant_rvq", "coverage": "keys_only", "keys": None}

    class _FakeChild(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(fake_stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    child = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChild)
    threading.Thread(target=child.serve_forever, daemon=True).start()
    try:
        base, supervisor = panel
        child_port = child.server_address[1]
        supervisor._state = "running"
        supervisor._ready = {
            "model": "m",
            "method": "turboquant_rvq",
            "endpoints": {"kv_stats": f"http://127.0.0.1:{child_port}/v1/kv/stats"},
        }

        body = _get(base, "/api/telemetry")
        assert body["available"] is True
        assert body["coverage"] == "keys_only"
    finally:
        child.shutdown()
        child.server_close()


def test_telemetry_rejects_non_http_scheme(panel):
    """A kv_stats URL that isn't http:// must be refused, not opened.

    emit_ready() only ever builds http:// URLs, so this only fires if that
    invariant is ever broken elsewhere -- but the proxy must not trust the
    handshake payload's scheme blindly.
    """
    base, supervisor = panel
    supervisor._state = "running"
    supervisor._ready = {
        "model": "m",
        "method": "turboquant_rvq",
        "endpoints": {"kv_stats": "file:///etc/passwd"},
    }

    body = _get(base, "/api/telemetry")
    assert body == {"available": False, "reason": "unexpected telemetry URL scheme"}


# --- /api/chat --------------------------------------------------------------


def test_chat_rejects_when_stopped(panel):
    base, _ = panel
    status, body = _post(base, "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 409
    assert body == {"error": "no server is running"}


def test_chat_rejects_empty_messages(panel):
    base, supervisor = panel
    supervisor._state = "running"
    supervisor._ready = {
        "model": "m",
        "endpoints": {"chat_completions": "http://127.0.0.1:9/v1/chat/completions"},
    }

    status, body = _post(base, "/api/chat", {"messages": []})
    assert status == 400
    assert body == {"error": "messages must be a non-empty list"}


def test_chat_rejects_missing_endpoint(panel):
    """A child predating this feature has no chat_completions key advertised
    under this exact condition -- in practice chat_completions has existed
    since #34, but the proxy must not KeyError if a future handshake ever
    drops it, per rule 4: endpoints advertised only if actually served."""
    base, supervisor = panel
    supervisor._state = "running"
    supervisor._ready = {"model": "m", "endpoints": {}}

    status, body = _post(base, "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 409
    assert body == {"error": "server predates chat proxying"}


def test_chat_relays_sse_bytes_verbatim(panel):
    """The proxy must forward the child's SSE stream byte-for-byte, without
    trying to parse or re-emit it -- see _proxy_chat's docstring on why this
    is a relay, not a reconstruction."""
    from http.server import BaseHTTPRequestHandler

    sse_body = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    class _FakeChild(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)  # drain the request body
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(sse_body)

        def log_message(self, *a):
            pass

    child = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChild)
    threading.Thread(target=child.serve_forever, daemon=True).start()
    try:
        base, supervisor = panel
        child_port = child.server_address[1]
        supervisor._state = "running"
        supervisor._ready = {
            "model": "m",
            "endpoints": {
                "chat_completions": f"http://127.0.0.1:{child_port}/v1/chat/completions"
            },
        }

        status, body = _post(
            base,
            "/api/chat",
            {"messages": [{"role": "user", "content": "hi"}]},
            expect_json=False,
        )
        assert status == 200
        assert body == sse_body
    finally:
        child.shutdown()
        child.server_close()


def test_chat_returns_502_when_child_unreachable(panel):
    base, supervisor = panel
    supervisor._state = "running"
    supervisor._ready = {
        "model": "m",
        # Port 1 is reserved and nothing will be listening there.
        "endpoints": {"chat_completions": "http://127.0.0.1:1/v1/chat/completions"},
    }

    status, body = _post(base, "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 502
    assert "error" in body

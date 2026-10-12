"""Regression tests for the control panel's request guards and shutdown.

Each test pins one bug found in a review of ``veloxquant_mlx/ui``: a cross-site
or DNS-rebinding request being served, malformed input dropping the socket,
the log buffer being iterated while pump threads append, a panel SIGTERM
orphaning the inference server, and torn config writes.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from veloxquant_mlx.ui import config as ui_config
from veloxquant_mlx.ui.server import MAX_BODY_BYTES, PanelHandler
from veloxquant_mlx.ui.supervisor import ServerSupervisor

_JSON = {"Content-Type": "application/json"}


@pytest.fixture()
def panel(tmp_path, monkeypatch):
    """A control plane on an ephemeral port with config redirected to tmp_path."""
    monkeypatch.setattr(ui_config, "CONFIG_PATH", tmp_path / "panel.json")

    supervisor = ServerSupervisor()
    handler = type("T", (PanelHandler,), {"supervisor": supervisor})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        yield port, supervisor
    finally:
        supervisor.stop()
        httpd.shutdown()
        httpd.server_close()


def _request(port, method, path, body=None, headers=None):
    """Raw request so tests can set Host/Origin/Content-Length exactly."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw
    finally:
        conn.close()


# --- cross-site / DNS-rebinding guard -----------------------------------------


def test_cross_origin_post_is_refused_and_has_no_effect(panel):
    port, _ = panel
    status, body = _request(
        port,
        "POST",
        "/api/config",
        json.dumps({"model": "attacker/model", "host": "0.0.0.0"}),
        {"Content-Type": "text/plain", "Origin": "http://evil.example"},
    )
    assert status == 403
    assert "origin" in body["error"]
    assert ui_config.load_config()["model"] == ""


def test_json_cross_origin_post_is_refused(panel):
    port, _ = panel
    status, _ = _request(
        port, "POST", "/api/config", "{}", {**_JSON, "Origin": "http://evil.example"}
    )
    assert status == 403


def test_rebound_host_is_refused(panel):
    port, _ = panel
    status, body = _request(port, "GET", "/api/config", headers={"Host": f"evil.example:{port}"})
    assert status == 403
    assert "host" in body["error"]


def test_same_origin_requests_still_work(panel):
    port, _ = panel
    origin = {"Origin": f"http://127.0.0.1:{port}"}
    status, _ = _request(port, "GET", "/api/status", headers=origin)
    assert status == 200
    status, body = _request(
        port, "POST", "/api/config", json.dumps({"model": "a/b"}), {**_JSON, **origin}
    )
    assert status == 200 and body["model"] == "a/b"
    # `localhost` is how people type it.
    status, _ = _request(port, "GET", "/api/status", headers={"Host": f"localhost:{port}"})
    assert status == 200


def test_post_body_must_be_json_content_type(panel):
    port, _ = panel
    status, _ = _request(port, "POST", "/api/config", "{}", {"Content-Type": "text/plain"})
    assert status == 415


# --- malformed input is answered, never dropped -------------------------------


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/start", "[]"),
        ("/api/start", '"a string"'),
        ("/api/start", "{not json"),
        ("/api/config", "[1]"),
        ("/api/start", json.dumps({"model": 123, "method": "turboquant_rvq"})),
        ("/api/start", json.dumps({"model": "m", "method": 5})),
        ("/api/start", json.dumps({"model": "m", "method": "turboquant_rvq", "overrides": "x"})),
        ("/api/start", json.dumps({"model": "m", "method": "turboquant_rvq", "overrides": [1]})),
    ],
)
def test_malformed_post_bodies_get_a_400(panel, path, body):
    port, _ = panel
    status, payload = _request(port, "POST", path, body, _JSON)
    assert status == 400
    assert payload["error"]


@pytest.mark.parametrize("bits", ["1e999", '"two"', "[2]"])
def test_unconvertible_numeric_settings_get_a_400(panel, bits):
    port, supervisor = panel
    body = f'{{"model": "m", "method": "turboquant_rvq", "bits": {bits}}}'
    status, payload = _request(port, "POST", "/api/start", body, _JSON)
    assert status == 400
    assert supervisor.state == "stopped"


@pytest.mark.parametrize("since", ["abc", "1.5"])
def test_logs_since_must_be_an_integer(panel, since):
    port, _ = panel
    status, body = _request(port, "GET", f"/api/logs?since={since}")
    assert status == 400
    assert "since" in body["error"]


@pytest.mark.parametrize("length", ["abc", "-1"])
def test_bad_content_length_is_rejected_without_hanging(panel, length):
    port, _ = panel
    status, _ = _request(port, "POST", "/api/config", None, {**_JSON, "Content-Length": length})
    assert status == 400


def test_oversized_body_is_rejected(panel):
    port, _ = panel
    status, _ = _request(
        port, "POST", "/api/config", None, {**_JSON, "Content-Length": str(MAX_BODY_BYTES + 1)}
    )
    assert status == 413


# --- log buffer is safe to read while pump threads write ----------------------


def test_logs_and_diagnose_survive_concurrent_appends():
    supervisor = ServerSupervisor()
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            supervisor._log("stderr", "x" * 40)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            supervisor.logs(0)  # RuntimeError: deque mutated during iteration, before the fix
            supervisor._diagnose()
    finally:
        stop.set()
        thread.join()


def test_logs_cursor_never_skips_a_line_under_concurrent_appends():
    supervisor = ServerSupervisor()
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            supervisor._log("stdout", str(i))
            i += 1
            time.sleep(0.0001)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    seen: list[int] = []
    cursor = 0
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            page = supervisor.logs(since=cursor)
            seen.extend(int(line["text"]) for line in page["lines"])
            cursor = page["total"]
    finally:
        stop.set()
        thread.join()
    page = supervisor.logs(since=cursor)
    seen.extend(int(line["text"]) for line in page["lines"])
    assert seen == list(range(len(seen)))


# --- supervisor type validation ------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [[], "x", {"model": 1}, {"model": "m", "method": []}, {"model": "m", "overrides": "x"}],
)
def test_supervisor_start_rejects_wrongly_typed_config(config):
    with pytest.raises(ValueError):
        ServerSupervisor().start(config)


# --- config writes -------------------------------------------------------------


def test_concurrent_config_saves_never_leave_a_torn_file(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    monkeypatch.setattr(ui_config, "CONFIG_PATH", path)

    def saver(n):
        for i in range(25):
            ui_config.save_config({"model": f"m{n}-{i}", "port": 9000 + n})

    threads = [threading.Thread(target=saver, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    json.loads(path.read_text())  # parses: not torn
    assert [p.name for p in tmp_path.iterdir()] == ["panel.json"]  # no stray temp files


# --- shutdown ------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_panel_signal_stops_the_supervised_server(sig, tmp_path):
    """The child lives in its own session, so only the panel's cleanup can stop it."""
    marker = f"vq-orphan-test-{os.getpid()}-{int(sig)}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    code = textwrap.dedent(
        f"""
        import builtins, threading
        from veloxquant_mlx.ui import server
        from veloxquant_mlx.ui.supervisor import ServerSupervisor

        orig = ServerSupervisor.__init__
        def patched(self):
            orig(self)
            builtins.SUP = self
            self._assert_servable = lambda m: None
            self._build_command = lambda c, m, me: ["sleep", "300", "{marker}"]
        ServerSupervisor.__init__ = patched
        threading.Timer(1.0, lambda: builtins.SUP.start({{"model": "m", "method": "x"}})).start()
        server.serve_panel(port={port}, open_browser=False)
        """
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _pids(marker):
            time.sleep(0.2)
        assert _pids(marker), "supervised child never started"

        proc.send_signal(sig)
        proc.wait(timeout=30)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _pids(marker):
            time.sleep(0.2)
        assert not _pids(marker), "supervised server survived the panel's shutdown"
    finally:
        for pid in _pids(marker):
            os.kill(pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()


def _pids(marker):
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]

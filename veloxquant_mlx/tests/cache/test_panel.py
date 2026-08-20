"""Tests for the control panel's supervisor and control API (#34).

Covers everything that does not need model weights: refusal rules, config
persistence, the static-file guard, and failure diagnosis.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from veloxquant_mlx.ui import config as ui_config
from veloxquant_mlx.ui.server import PanelHandler
from veloxquant_mlx.ui.supervisor import LOG_CAPACITY, ServerSupervisor

# Generous enough that a slow/loaded CI runner never turns latency into a
# spurious failure; the panel fixture already warms the one genuinely slow
# call, so a request hitting this ceiling means something is really stuck.
_TIMEOUT = 60


@pytest.fixture()
def panel():
    """A control plane on an ephemeral port, torn down after the test.

    ``/api/methods`` resolves the method registry lazily *inside* the request
    handler thread (deliberate: it keeps panel startup fast). The first
    ``list_methods()`` call is the expensive one — it imports all 40 method
    modules behind ``get_method``'s cache, ~5s here and slower on a cold CI
    runner — while every later call is ~0.04s. That one-off cost landed on
    whichever request came first, pushing it past the client timeout so the
    test failed with a socket TimeoutError rather than any real panel bug.
    Warm the cache here so it is paid during setup, not inside a timed
    request.
    """
    from veloxquant_mlx.cache.registry import list_methods

    list_methods()

    supervisor = ServerSupervisor()
    handler = type("T", (PanelHandler,), {"supervisor": supervisor})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, supervisor
    finally:
        supervisor.stop()
        httpd.shutdown()
        httpd.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=_TIMEOUT) as resp:
        return resp.status, json.loads(resp.read())


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --- supervisor refusal rules ---------------------------------------------


def test_start_requires_a_model():
    with pytest.raises(ValueError, match="model id or local path is required"):
        ServerSupervisor().start({"model": "  ", "method": "turboquant_rvq"})


def test_start_refuses_crash_tier_method():
    """No silent substitution: the panel must refuse what the CLI refuses."""
    with pytest.raises(ValueError, match="cannot be served"):
        ServerSupervisor().start({"model": "x/y", "method": "turboquant_prod"})


def test_start_refuses_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        ServerSupervisor().start({"model": "x/y", "method": "nope"})


def test_initial_state_is_stopped():
    supervisor = ServerSupervisor()
    assert supervisor.state == "stopped"
    assert supervisor.status()["ready"] is None


def test_stop_is_idempotent():
    supervisor = ServerSupervisor()
    assert supervisor.stop()["state"] == "stopped"
    assert supervisor.stop()["state"] == "stopped"


def test_command_carries_every_setting():
    supervisor = ServerSupervisor()
    cmd = supervisor._build_command(
        {
            "bits": 3,
            "host": "0.0.0.0",
            "port": 9001,
            "max_tokens": 64,
            "temp": 0.7,
            "top_p": 0.9,
            "seed": 7,
        },
        "some/model",
        "kivi",
    )
    joined = " ".join(cmd)
    assert "veloxquant_mlx serve" in joined
    for expected in (
        "--method kivi",
        "--bits 3",
        "--host 0.0.0.0",
        "--port 9001",
        "--max-tokens 64",
        "--seed 7",
    ):
        assert expected in joined


def test_log_buffer_is_bounded():
    supervisor = ServerSupervisor()
    for i in range(LOG_CAPACITY + 120):
        supervisor._log("stdout", f"line {i}")
    assert supervisor.logs()["total"] == LOG_CAPACITY


def test_ready_handshake_promotes_to_running():
    from veloxquant_mlx.cli.serve import READY_PREFIX

    supervisor = ServerSupervisor()
    supervisor._state = "starting"
    payload = {
        "method": "kivi",
        "bits": 2,
        "layer_caches": 16,
        "endpoints": {"openai_base_url": "http://127.0.0.1:8000/v1"},
    }
    supervisor._on_ready(READY_PREFIX + json.dumps(payload))

    assert supervisor._state == "running"
    assert supervisor.status()["ready"]["method"] == "kivi"


def test_malformed_handshake_does_not_promote():
    supervisor = ServerSupervisor()
    supervisor._state = "starting"
    supervisor._on_ready("VELOXQUANT_READY {not json")
    assert supervisor._state == "starting"


@pytest.mark.parametrize(
    "stderr_line,expected",
    [
        ("OSError: [Errno 48] Address already in use", "already in use"),
        ("RuntimeError: metal::malloc failed", "GPU memory"),
        ("huggingface_hub.errors.RepositoryNotFoundError: Repository Not Found", "not found"),
    ],
)
def test_diagnose_translates_common_failures(stderr_line, expected):
    """#34: startup failures must read as causes, not stack frames."""
    supervisor = ServerSupervisor()
    supervisor._config = {"port": 8000, "host": "127.0.0.1"}
    supervisor._log("stderr", stderr_line)
    assert expected in supervisor._diagnose()


def test_diagnose_skips_stack_frames():
    supervisor = ServerSupervisor()
    supervisor._config = {"port": 8000}
    supervisor._log("stderr", "Traceback (most recent call last):")
    supervisor._log("stderr", '  File "/x/y.py", line 3, in main')
    supervisor._log("stderr", "ValueError: something specific broke")
    assert supervisor._diagnose() == "ValueError: something specific broke"


# --- control API ----------------------------------------------------------


def test_api_status_reports_version(panel):
    base, _ = panel
    status, body = _get(base, "/api/status")
    assert status == 200
    assert body["state"] == "stopped"
    assert body["version"]


def test_api_methods_matches_registry(panel):
    base, _ = panel
    _, body = _get(base, "/api/methods")
    assert len(body["methods"]) == 41
    assert body["accounting_only"] is True
    assert sum(1 for m in body["methods"] if not m["is_servable"]) == 5


def test_api_start_rejects_bad_input(panel):
    base, _ = panel
    status, body = _post(base, "/api/start", {"model": "", "method": "turboquant_rvq"})
    assert status == 400
    assert "required" in body["error"]

    status, body = _post(base, "/api/start", {"model": "x/y", "method": "polar"})
    assert status == 400
    assert "cannot be served" in body["error"]


def test_api_serves_the_panel(panel):
    base, _ = panel
    with urllib.request.urlopen(base + "/", timeout=_TIMEOUT) as resp:
        html = resp.read().decode()
    assert "VeloxQuant-MLX" in html
    assert 'id="primary-btn"' in html


def test_api_refuses_path_traversal(panel):
    base, _ = panel
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/../config.py", timeout=_TIMEOUT)
    assert exc.value.code == 404


# --- config persistence ---------------------------------------------------


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_config, "CONFIG_PATH", tmp_path / "panel.json")

    ui_config.save_config({"model": "a/b", "port": 9999})
    loaded = ui_config.load_config()
    assert loaded["model"] == "a/b"
    assert loaded["port"] == 9999
    assert loaded["method"] == ui_config.DEFAULTS["method"]


def test_config_ignores_unknown_keys(tmp_path, monkeypatch):
    """Allowlist, so a stray UI field cannot start persisting to disk."""
    monkeypatch.setattr(ui_config, "CONFIG_PATH", tmp_path / "panel.json")

    ui_config.save_config({"model": "a/b", "surprise": "nope"})
    assert "surprise" not in ui_config.load_config()


def test_config_survives_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    path.write_text("{{{ not json")
    monkeypatch.setattr(ui_config, "CONFIG_PATH", path)
    assert ui_config.load_config() == ui_config.DEFAULTS

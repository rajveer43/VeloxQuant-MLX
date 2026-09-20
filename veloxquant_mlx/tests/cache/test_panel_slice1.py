"""Tests for the method browser, knobs, model picker and memory panel.

The presentation rules here exist because telemetry coverage is uneven across
the catalog (14 keys+values / 5 keys-only / 19 none). A UI that assumed uniform
coverage would render zeros for half the methods, which reads as "no
compression" — a claim nobody measured.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from veloxquant_mlx.cache.registry import (
    TelemetryCoverage,
    describe_field,
    get_method,
    list_methods,
    telemetry_coverage,
)
from veloxquant_mlx.cli.serve import parse_overrides
from veloxquant_mlx.ui.memory import memory_report
from veloxquant_mlx.ui.models import _human_size, _looks_servable, local_models
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
        supervisor.stop()
        httpd.shutdown()
        httpd.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return json.loads(resp.read())


# --- telemetry coverage ---------------------------------------------------


def test_coverage_split_is_stable():
    """Locks the measured 14/5/19 split; drift should be a deliberate change.

    age_tiered (issue #256) joined the NONE bucket in the earlier 20/13/5 ->
    the only place a fourth split component (13/5/20) added a servable
    method without a per-K/V ``compressed_key_bytes``/``compressed_value_bytes``
    pair — same accounting shape as amc, which reports NONE for the same
    reason. kvtc (issue #17) moved NONE -> KEYS_AND_VALUES after gaining
    those four properties (it always compressed both K and V; it just never
    exposed the split telemetry.py's probe looks for), giving 14/5/19.
    """
    counts = dict.fromkeys(TelemetryCoverage, 0)
    for info in list_methods(servable_only=True):
        counts[info.coverage] += 1

    assert counts[TelemetryCoverage.KEYS_AND_VALUES] == 14
    assert counts[TelemetryCoverage.KEYS_ONLY] == 5
    assert counts[TelemetryCoverage.NONE] == 19


def test_default_method_is_keys_only():
    """The serve default reports keys only, so no whole-cache ratio exists.

    Guards the most likely UI mistake: showing turboquant_rvq's key ratio as if
    it described the entire cache.
    """
    assert get_method("turboquant_rvq").coverage is TelemetryCoverage.KEYS_ONLY


def test_eviction_methods_report_no_byte_counters():
    """Eviction drops tokens rather than compressing bytes."""
    for name in ("h2o", "snapkv", "tova", "streaming_llm"):
        assert get_method(name).coverage is TelemetryCoverage.NONE


def test_coverage_labels_never_read_as_zero():
    """The "no estimate" label must not be confusable with a measured zero."""
    assert TelemetryCoverage.NONE.label == "no estimate"
    for coverage in TelemetryCoverage:
        assert coverage.label and coverage.label != "0"


def test_crash_tier_methods_report_no_coverage():
    for info in list_methods():
        if not info.serve_tier.is_servable:
            assert info.coverage is TelemetryCoverage.NONE


def test_coverage_is_memoized():
    assert telemetry_coverage("kivi") is telemetry_coverage("kivi")


# --- field schema ---------------------------------------------------------


def test_describe_field_reads_types_from_the_dataclass():
    assert describe_field("kivi_group_size") == {
        "name": "kivi_group_size",
        "type": "int",
        "default": 32,
        "optional": False,
        "help": "Tokens per min/max quantization group.",
    }

    rank = describe_field("svdq_rank")
    assert rank["type"] == "int" and rank["optional"] and rank["default"] is None

    threshold = describe_field("svdq_energy_threshold")
    assert threshold["type"] == "float" and threshold["default"] == 0.95


def test_describe_field_tolerates_unknown_names():
    assert describe_field("not_a_field")["type"] == "unknown"


def test_every_method_schema_is_serializable():
    payload = json.dumps([i.to_dict() for i in list_methods()])
    assert "field_schema" in payload
    for info in list_methods():
        for field in info.field_schema:
            assert set(field) == {"name", "type", "default", "optional", "help"}


# --- overrides ------------------------------------------------------------


def test_parse_overrides_types_values():
    assert parse_overrides(["kivi_group_size=64"]) == {"kivi_group_size": 64}
    assert parse_overrides(["svdq_energy_threshold=0.9"]) == {"svdq_energy_threshold": 0.9}


def test_parse_overrides_blank_clears_optional_field():
    """Blank is meaningful: svdq_rank=None selects the energy-threshold path."""
    assert parse_overrides(["svdq_rank="]) == {"svdq_rank": None}


@pytest.mark.parametrize("bad", ["nope=1", "kivi_group_size=abc", "noequals", "method=kivi"])
def test_parse_overrides_rejects_bad_input(bad):
    with pytest.raises(SystemExit):
        parse_overrides([bad])


def test_supervisor_rejects_knobs_from_another_method():
    supervisor = ServerSupervisor()
    with pytest.raises(ValueError, match="not a setting for"):
        supervisor.start(
            {
                "model": "x/y",
                "method": "kivi",
                "overrides": {"svdq_rank": 8},
            }
        )


def test_supervisor_rejects_unparseable_knob():
    supervisor = ServerSupervisor()
    with pytest.raises(ValueError, match="expects int"):
        supervisor.start(
            {
                "model": "x/y",
                "method": "kivi",
                "overrides": {"kivi_group_size": "wide"},
            }
        )


def test_overrides_reach_the_command_line():
    cmd = ServerSupervisor()._build_command(
        {"port": 8000, "overrides": {"kivi_group_size": 64}}, "m/x", "kivi"
    )
    assert "--set" in cmd
    assert "kivi_group_size=64" in cmd


# --- logs -------------------------------------------------------------


def test_logs_since_returns_only_new_lines():
    """``logs(since=n)`` must return lines *after* index ``n``, not from it.

    Also guards the ``itertools.islice`` rewrite in ``ServerSupervisor.logs``
    (originally ``list(self._logs)[since:]``): both must agree on the exact
    same slice semantics.
    """
    supervisor = ServerSupervisor()
    for i in range(5):
        supervisor._log("panel", f"line {i}")

    first = supervisor.logs(since=0)
    assert first["total"] == 5
    assert [line["text"] for line in first["lines"]] == [f"line {i}" for i in range(5)]

    second = supervisor.logs(since=3)
    assert second["total"] == 5
    assert [line["text"] for line in second["lines"]] == ["line 3", "line 4"]


def test_logs_since_at_or_past_total_returns_empty():
    supervisor = ServerSupervisor()
    supervisor._log("panel", "only line")

    caught_up = supervisor.logs(since=1)
    assert caught_up["lines"] == []
    assert caught_up["total"] == 1

    past_end = supervisor.logs(since=100)
    assert past_end["lines"] == []
    assert past_end["total"] == 1


def test_logs_total_matches_capacity_once_the_buffer_wraps():
    """Once the deque hits ``LOG_CAPACITY`` and evicts old lines, ``total``
    reflects what's actually *retained* (the deque is bounded, not a lifetime
    counter) — this pins the ``itertools.islice`` rewrite to the exact same
    semantics as the original ``list(self._logs)[since:]``.
    """
    from veloxquant_mlx.ui.supervisor import LOG_CAPACITY

    supervisor = ServerSupervisor()
    for i in range(LOG_CAPACITY + 10):
        supervisor._log("panel", f"line {i}")

    result = supervisor.logs(since=0)
    assert result["total"] == LOG_CAPACITY  # capped, not a lifetime count
    assert len(result["lines"]) == LOG_CAPACITY  # only what's retained
    assert result["lines"][0]["text"] == "line 10"  # oldest 10 evicted
    assert result["lines"][-1]["text"] == f"line {LOG_CAPACITY + 9}"


# --- memory ---------------------------------------------------------------


def test_memory_reports_no_server_honestly():
    report = memory_report(pid=None)
    assert report["source"] == "measured"
    assert report["process"]["rss_bytes"] is None
    assert "no server" in report["process"]["unavailable_reason"]


def test_memory_measures_a_real_process():
    import os

    report = memory_report(pid=os.getpid())
    rss = report["process"]["rss_bytes"]
    # psutil is a dev extra; skip rather than fail where it is absent.
    if rss is None:
        pytest.skip(report["process"]["unavailable_reason"])
    assert rss > 0


def test_mlx_memory_is_withheld_with_a_reason():
    """Process-local MLX counters would describe the panel, not the server.

    Reporting them beside the server's RSS under a "measured" tag would be true
    but misleading, so they are withheld until #27's stats endpoint exists.
    """
    mlx = memory_report(pid=None)["mlx"]
    assert mlx["active_bytes"] is None
    assert mlx["peak_bytes"] is None
    assert mlx["unavailable_reason"]


def test_memory_note_disclaims_comparability():
    note = memory_report(pid=None)["note"]
    assert "compression" in note
    assert "drop" in note


def test_absent_memory_is_none_not_zero():
    """A zero here would assert "no memory used" — a claim we did not measure.

    ``None`` forces the UI down its "not reported" branch; ``0`` would render as
    a legitimate measurement.
    """
    report = memory_report(pid=None)
    for value in (
        report["process"]["rss_bytes"],
        report["mlx"]["active_bytes"],
        report["mlx"]["peak_bytes"],
    ):
        assert value is None


def test_memory_report_is_not_aliased_across_calls():
    """The RSS cache must hand back copies, not the cached dict itself.

    ``/api/memory`` is polled every 1s (see static/panel.js), so reads for a
    live pid are cached briefly (see ``ui/memory.py``). If two calls returned
    the *same* dict object, a caller mutating one report (or json-encoding
    frameworks that don't) could corrupt what the next poll reads back.
    """
    import os

    first = memory_report(pid=os.getpid())
    first["process"]["rss_bytes"] = -1  # poison it if it's the live cache entry
    second = memory_report(pid=os.getpid())
    assert second["process"]["rss_bytes"] != -1


def test_memory_cache_does_not_bleed_across_pids():
    """A cached reading for one pid must never be served for a different one.

    Otherwise stopping a server and immediately starting a new one (new pid,
    reusing the panel process) could show the previous process's memory for
    up to the cache TTL.
    """
    import os

    real_pid = os.getpid()
    real_rss = memory_report(pid=real_pid)["process"]["rss_bytes"]  # warms the cache

    # A pid psutil can't find must report "unavailable", never the cached
    # real process's RSS, even though the cache is still fresh.
    bogus_pid = -1
    bogus = memory_report(pid=bogus_pid)["process"]
    assert bogus["rss_bytes"] != real_rss
    assert bogus["rss_bytes"] is None


# --- model picker ---------------------------------------------------------


def test_local_models_never_raises():
    assert isinstance(local_models(), list)


def test_local_models_shape():
    for model in local_models():
        assert set(model) == {"repo_id", "size_bytes", "size_label", "is_mlx"}
        assert model["size_bytes"] >= 0


@pytest.mark.parametrize(
    "repo_id,expected",
    [
        ("mlx-community/Llama-3.2-1B-Instruct-4bit", True),
        ("openai/clip-vit-base-patch32", False),
        ("BAAI/bge-m3", False),
        ("dslim/bert-base-NER", False),
        ("mlx-community/Qwen2-VL-7B-Instruct-bf16", False),
    ],
)
def test_model_filter(repo_id, expected):
    assert _looks_servable(repo_id) is expected


def test_human_size():
    assert _human_size(512) == "512 B"
    assert _human_size(1536) == "1.5 KB"
    assert _human_size(5 * 1024**3) == "5.0 GB"


# --- control API ----------------------------------------------------------


def test_api_models(panel):
    base, _ = panel
    body = _get(base, "/api/models")
    assert isinstance(body["models"], list)


def test_api_memory(panel):
    base, _ = panel
    body = _get(base, "/api/memory")
    assert body["source"] == "measured"
    assert body["process"]["rss_bytes"] is None  # nothing running


def test_methods_cli_states_coverage():
    """The human-readable CLI must carry the same caveats as the panel.

    Without this, `veloxquant methods` reads as if every method reports byte
    counters — the exact confusion rules 8 and 9 exist to prevent.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "veloxquant_mlx", "methods", "--servable-only"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    assert "telemetry: no estimate" in proc.stdout
    assert "telemetry: full estimate" in proc.stdout
    assert "'not reported' is not zero" in proc.stdout
    assert "not a whole-cache ratio" in proc.stdout


def test_api_methods_exposes_schema_and_coverage(panel):
    base, _ = panel
    methods = _get(base, "/api/methods")["methods"]

    kivi = next(m for m in methods if m["name"] == "kivi")
    assert kivi["coverage"] == "keys_and_values"
    assert [f["name"] for f in kivi["field_schema"]] == [
        "bit_width_inlier",
        "kivi_group_size",
    ]

    h2o = next(m for m in methods if m["name"] == "h2o")
    assert h2o["coverage"] == "none"
    assert h2o["coverage_label"] == "no estimate"

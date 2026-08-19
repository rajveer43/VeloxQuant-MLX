"""Tests for the powermetrics sampler -- all run UNPRIVILEGED in CI.

The point of this file is the degradation path: without root the sampler must
report unavailable, hand back ``None``, and never raise. A silent ``0.0``
anywhere here would become a fabricated energy measurement downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from veloxquant_mlx.profiling.power_sampler import (
    PowerSample,
    PowerSampler,
    parse_plist_samples,
    parse_text_samples,
)

FIXTURE = Path(__file__).parent / "fixtures" / "powermetrics_sample.plist"


def test_sampler_reports_unavailable_without_root(monkeypatch):
    """Non-root: unavailable, clean enter/exit, and energy is None not 0.0."""
    monkeypatch.setattr("os.geteuid", lambda: 501)
    sampler = PowerSampler(interval_ms=100)
    assert sampler.available() is False

    with sampler as s:
        assert s is sampler  # context manager still enters

    # The critical assertion: None, never a silent zero.
    assert sampler.energy_joules() is None
    assert sampler.energy_joules() != 0.0

    power = sampler.mean_power_mw()
    assert power["cpu"] is None
    assert power["gpu"] is None
    assert power["package"] is None


def test_sampler_does_not_spawn_subprocess_without_root(monkeypatch):
    """Unprivileged runs must not even attempt the privileged binary."""
    monkeypatch.setattr("os.geteuid", lambda: 501)

    def _boom(*a, **k):
        raise AssertionError("subprocess must not be spawned when unprivileged")

    monkeypatch.setattr("subprocess.Popen", _boom)
    with PowerSampler() as s:
        pass
    assert s.energy_joules() is None


def test_exit_never_raises(monkeypatch):
    """A profiling failure must not fail the run being profiled."""
    monkeypatch.setattr("os.geteuid", lambda: 501)
    sampler = PowerSampler()
    sampler.__enter__()

    # Simulate a broken process handle; __exit__ must swallow it.
    class _Broken:
        def terminate(self):
            raise RuntimeError("boom")

        def kill(self):
            raise RuntimeError("boom")

        def wait(self, timeout=None):
            raise RuntimeError("boom")

    sampler._proc = _Broken()
    sampler.__exit__(None, None, None)  # must not raise


@pytest.mark.parametrize(
    "junk",
    [
        b"",
        b"powermetrics must be invoked as the superuser\n",
        b'<?xml version="1.0"?><plist><dict><key>trunc',  # truncated
        b"\x00\x01\x02\xff\xfe garbage bytes",
        b'<?xml version="1.0"?><plist><dict></dict></plist>',  # valid, no power
    ],
)
def test_sampler_never_raises_on_malformed_output(junk):
    """Truncated/garbage bytes yield no samples and no exception."""
    assert parse_plist_samples(junk) == []


def test_text_parser_never_raises_on_malformed_output():
    assert parse_text_samples(b"") == []
    assert parse_text_samples(b"\x00\xff nonsense") == []


def test_energy_is_none_when_no_samples_parsed():
    """Zero parsed samples is 'unavailable', not 'zero joules'."""
    sampler = PowerSampler()
    assert sampler.energy_joules() is None


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="captured powermetrics plist fixture not present (requires a root capture on real hardware)",
)
def test_plist_parsing_extracts_power_fields():
    """Parse a fixture CAPTURED on real hardware -- never a hand-written one."""
    raw = FIXTURE.read_bytes()
    samples = parse_plist_samples(raw)
    assert samples, "captured fixture produced no samples"
    assert all(isinstance(s, PowerSample) for s in samples)

    # At least one sample must carry real power data.
    with_power = [s for s in samples if s.package_mw is not None]
    assert with_power, "no sample carried package power"
    for s in with_power:
        assert s.package_mw >= 0.0


@pytest.mark.skipif(not FIXTURE.exists(), reason="captured fixture not present")
def test_plist_parsing_handles_nul_separated_documents():
    """powermetrics NUL-terminates each emitted document.

    Regression guard: splitting only on the XML header leaves a leading NUL on
    every document after the first, so plistlib rejects them and all but one
    sample is silently dropped -- understating integrated energy rather than
    raising. The captured fixture holds 3 samples; all 3 must parse.
    """
    raw = FIXTURE.read_bytes()
    assert raw.count(b"<?xml version") == len(parse_plist_samples(raw)), (
        "every emitted plist document must yield a sample"
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason="captured fixture not present")
def test_energy_joules_from_fixture_is_positive():
    """End-to-end: captured samples integrate to a positive joule estimate."""
    sampler = PowerSampler()
    sampler._samples = parse_plist_samples(FIXTURE.read_bytes())
    sampler._t_start = 0.0
    sampler._t_end = 1.0
    energy = sampler.energy_joules()
    assert energy is not None
    assert energy > 0.0


def _mk(pkg_mw: float, elapsed_s: float) -> PowerSample:
    return PowerSample(
        cpu_mw=pkg_mw, gpu_mw=0.0, ane_mw=0.0, package_mw=pkg_mw, elapsed_s=elapsed_s
    )


def test_energy_integrates_over_sampled_window_not_wall_time():
    """Partial sampling must not have energy invented for unobserved time.

    Multiplying a partial-window mean by full wall time fabricates energy for
    intervals nobody measured. Here 2s of samples were collected over a 10s
    run, so energy must reflect 2s (20 J), not 10s (100 J).
    """
    s = PowerSampler()
    s._samples = [_mk(10_000.0, 1.0), _mk(10_000.0, 1.0)]  # 10 W over 2 s
    s._t_start, s._t_end = 0.0, 10.0
    assert s.energy_joules() == pytest.approx(20.0)


def test_coverage_reports_fraction_of_wall_time_sampled():
    s = PowerSampler()
    s._samples = [_mk(10_000.0, 1.0), _mk(10_000.0, 1.0)]
    s._t_start, s._t_end = 0.0, 10.0
    assert s.coverage == pytest.approx(0.2)
    assert s.sampled_s == pytest.approx(2.0)


def test_coverage_is_none_without_samples():
    """No samples is 'unknown coverage', not 'zero coverage'."""
    s = PowerSampler()
    s._t_start, s._t_end = 0.0, 10.0
    assert s.coverage is None
    assert s.energy_joules() is None


def test_full_coverage_matches_wall_time_integration():
    """When nothing is lost, the two integrations agree."""
    s = PowerSampler()
    s._samples = [_mk(10_000.0, 0.5) for _ in range(20)]  # 10 W over 10 s
    s._t_start, s._t_end = 0.0, 10.0
    assert s.coverage == pytest.approx(1.0)
    assert s.energy_joules() == pytest.approx(100.0)

"""Background ``powermetrics`` sampler -- the only module that knows about root.

Isolating the privileged path here is what lets the rest of the harness be
unit-tested unprivileged, which is how CI stays green.

What this measures
------------------
Package, CPU, GPU and ANE power in milliwatts, sampled at a fixed interval by
the macOS ``powermetrics`` binary.

What this CANNOT measure
------------------------
* **Energy, exactly.** ``energy_joules()`` integrates *sampled mean power* over
  wall time (``J = mean_package_W x elapsed_s``). This is a sampled estimate,
  not a hardware energy counter: its resolution is bounded by the sampling
  interval, and power excursions shorter than that interval are averaged away.
* **Per-process attribution.** ``powermetrics`` reports package-level power, so
  every other process on the machine contributes to the total.
* **Memory bandwidth.** ``powermetrics`` reports power and residency, not DRAM
  bytes/s, on Apple Silicon.

Privilege
---------
``powermetrics`` requires root. Without it this class degrades to ``None``-filled
readings: ``available()`` is ``False``, ``energy_joules()`` returns ``None``
(never ``0.0``, which would propagate downstream as a fake measurement), and the
context manager still enters and exits cleanly. A profiling failure must never
fail the run being profiled, so ``__exit__`` never raises.

Output format
-------------
The sampler requests ``-f plist`` and parses it with the stdlib ``plistlib``.
``-f plist`` was **verified accepted** under sudo on an Apple M4 (macOS,
``powermetrics`` at ``/usr/bin/powermetrics``): it exits 0 and emits one
NUL-terminated XML document per sample. A text-output fallback is kept for
binaries that reject plist, since ``powermetrics`` enforces its root check
*before* validating arguments -- an unprivileged probe returns the same
superuser error for every format value, including bogus ones, so the format
cannot be confirmed without root.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

POWERMETRICS_BIN = "/usr/bin/powermetrics"

# plist emits one <dict> per sample, concatenated. Split on the document header
# rather than assuming a single well-formed document.
_PLIST_HEADER = b"<?xml version"

_TEXT_PATTERNS = {
    "cpu_mw": re.compile(rb"CPU Power:\s*([0-9.]+)\s*mW", re.IGNORECASE),
    "gpu_mw": re.compile(rb"GPU Power:\s*([0-9.]+)\s*mW", re.IGNORECASE),
    "ane_mw": re.compile(rb"ANE Power:\s*([0-9.]+)\s*mW", re.IGNORECASE),
    "package_mw": re.compile(
        rb"Combined Power \(CPU \+ GPU \+ ANE\):\s*([0-9.]+)\s*mW", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class PowerSample:
    """One sampling interval. Any field may be ``None`` if not reported."""

    cpu_mw: float | None
    gpu_mw: float | None
    ane_mw: float | None
    package_mw: float | None
    elapsed_s: float


def _f(value) -> float | None:
    """Coerce to float, returning None rather than raising on junk."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def parse_plist_samples(raw: bytes) -> list[PowerSample]:
    """Parse concatenated plist documents into samples. Never raises.

    ``powermetrics -f plist`` emits one document per sample back-to-back, which
    is not a single valid plist, so documents are split on the XML header before
    being handed to ``plistlib`` individually.

    Each emitted document is **NUL-terminated** (`</plist>\\n\\x00<?xml ...`),
    verified against a captured fixture on an M4. The NUL must be stripped or
    every document after the first fails to parse -- which would silently drop
    most samples and understate the integrated energy rather than erroring.
    """
    if not raw:
        return []
    samples: list[PowerSample] = []
    chunks = [c for c in raw.split(_PLIST_HEADER) if c.strip(b"\x00 \t\r\n")]
    for chunk in chunks:
        # Strip the NUL terminator (and any stray whitespace) that separates
        # consecutive documents.
        doc = _PLIST_HEADER + chunk.strip(b"\x00 \t\r\n")
        try:
            parsed = plistlib.loads(doc)
        except Exception:
            continue  # truncated or malformed tail -- skip, do not raise
        if not isinstance(parsed, dict):
            continue
        sample = _sample_from_plist_dict(parsed)
        if sample is not None:
            samples.append(sample)
    return samples


def _sample_from_plist_dict(d: dict) -> PowerSample | None:
    """Extract power fields from one parsed plist sample dict."""
    processor = d.get("processor")
    if not isinstance(processor, dict):
        return None

    cpu = _f(processor.get("cpu_power"))
    gpu = _f(processor.get("gpu_power"))
    ane = _f(processor.get("ane_power"))
    package = _f(processor.get("combined_power"))
    if package is None:
        # Older/!M-series layouts omit combined_power; sum what we have rather
        # than reporting a hole. Returns None if nothing at all was reported.
        parts = [p for p in (cpu, gpu, ane) if p is not None]
        package = sum(parts) if parts else None

    if cpu is None and gpu is None and ane is None and package is None:
        return None

    # elapsed_ns is per-sample; fall back to 0.0 so a missing field cannot
    # silently inflate the integration window.
    elapsed_ns = _f(d.get("elapsed_ns")) or 0.0
    return PowerSample(
        cpu_mw=cpu,
        gpu_mw=gpu,
        ane_mw=ane,
        package_mw=package,
        elapsed_s=elapsed_ns / 1e9,
    )


def parse_text_samples(raw: bytes) -> list[PowerSample]:
    """Parse the default (non-plist) text output. Never raises.

    Fallback for binaries that reject ``-f plist``.
    """
    if not raw:
        return []
    samples: list[PowerSample] = []
    # Each sample block is delimited by the sampler banner.
    blocks = re.split(rb"\*\*\* Sampled system activity", raw)
    for block in blocks:
        found = {}
        for key, pattern in _TEXT_PATTERNS.items():
            m = pattern.search(block)
            if m:
                found[key] = _f(m.group(1))
        if not found:
            continue
        package = found.get("package_mw")
        if package is None:
            parts = [found.get(k) for k in ("cpu_mw", "gpu_mw", "ane_mw")]
            parts = [p for p in parts if p is not None]
            package = sum(parts) if parts else None
        samples.append(
            PowerSample(
                cpu_mw=found.get("cpu_mw"),
                gpu_mw=found.get("gpu_mw"),
                ane_mw=found.get("ane_mw"),
                package_mw=package,
                elapsed_s=0.0,
            )
        )
    return samples


class PowerSampler:
    """Background ``powermetrics`` sampler.

    Yields ``None``-filled readings when unprivileged rather than raising, so
    callers get an honest "no data" instead of a fabricated zero.
    """

    def __init__(
        self,
        interval_ms: int = 100,
        samplers: tuple[str, ...] = ("cpu_power", "gpu_power"),
    ) -> None:
        self.interval_ms = int(interval_ms)
        self.samplers = tuple(samplers)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._raw = bytearray()
        self._lock = threading.Lock()
        self._samples: list[PowerSample] = []
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._used_text_fallback = False

    # -- availability ---------------------------------------------------
    def available(self) -> bool:
        """True only if we are root AND the binary exists.

        Checked explicitly rather than by catching a subprocess failure:
        ``powermetrics`` has been observed printing "must be invoked as the
        superuser" to stdout with a *zero* exit code, so an exit-code check
        is not reliable.
        """
        if os.geteuid() != 0:
            return False
        return shutil.which(POWERMETRICS_BIN) is not None or os.path.exists(POWERMETRICS_BIN)

    # -- context manager ------------------------------------------------
    def __enter__(self) -> PowerSampler:
        self._t_start = time.perf_counter()
        if not self.available():
            return self  # degrade silently; energy_joules() will return None
        cmd = [
            POWERMETRICS_BIN,
            "-i",
            str(self.interval_ms),
            "--samplers",
            ",".join(self.samplers),
            "-f",
            "plist",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except Exception:
            self._proc = None  # unavailable, not fatal
        return self

    def __exit__(self, *exc) -> None:
        """Stop sampling. Never raises -- profiling must not fail the run.

        Ordering matters here. ``powermetrics`` writes a full plist document
        per interval into a pipe the reader thread drains in 64 KiB chunks, so
        at a 100 ms interval over a multi-second run there is always output in
        flight. Killing the process first discards whatever is still buffered,
        and how much is lost depends on thread scheduling -- which produced
        run-to-run energy swings of 3x on identical work.

        So: close stdin and SIGTERM (powermetrics exits on either), then let
        the reader drain to EOF before joining. The join timeout scales with
        the run length because there is proportionally more to drain.
        """
        self._t_end = time.perf_counter()
        try:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            # Drain BEFORE reaping: the reader exits on EOF once powermetrics
            # closes its end, so this collects the tail rather than dropping it.
            if self._thread is not None:
                self._thread.join(timeout=max(5.0, 0.2 * self.elapsed_s))
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
                    except Exception:
                        pass
                try:
                    if self._proc.stdout is not None:
                        self._proc.stdout.close()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                self._parse()
            except Exception:
                pass

    # -- internals ------------------------------------------------------
    def _reader(self) -> None:
        """Drain stdout incrementally on a daemon thread."""
        try:
            assert self._proc is not None and self._proc.stdout is not None
            for chunk in iter(lambda: self._proc.stdout.read(65536), b""):
                if not chunk:
                    break
                with self._lock:
                    self._raw.extend(chunk)
        except Exception:
            pass  # a dead reader means no samples, which reads as unavailable

    def _parse(self) -> None:
        with self._lock:
            raw = bytes(self._raw)
        samples = parse_plist_samples(raw)
        if not samples:
            samples = parse_text_samples(raw)
            self._used_text_fallback = bool(samples)
        self._samples = samples

    # -- results --------------------------------------------------------
    @property
    def samples(self) -> list[PowerSample]:
        return list(self._samples)

    @property
    def elapsed_s(self) -> float:
        if self._t_start is None:
            return 0.0
        end = self._t_end if self._t_end is not None else time.perf_counter()
        return end - self._t_start

    @property
    def sampled_s(self) -> float:
        """Wall time the collected samples actually cover.

        Each plist document reports its own ``elapsed_ns``, so this is the real
        observed window -- which is *not* the same as :attr:`elapsed_s` if any
        output was lost.
        """
        return sum(s.elapsed_s for s in self._samples)

    @property
    def coverage(self) -> float | None:
        """Fraction of wall time the samples cover, or ``None`` if unmeasured.

        1.0 means every interval was captured. Substantially less means output
        was dropped, and the energy figure is correspondingly less trustworthy.
        """
        if not self._samples or self.elapsed_s <= 0:
            return None
        return self.sampled_s / self.elapsed_s

    def energy_joules(self) -> float | None:
        """Sampled energy estimate in joules, or ``None`` if unmeasured.

        ``J = mean_package_W x sampled_s``, integrated over the window the
        samples **actually cover**, not over raw wall time. Those differ
        whenever sampler output is lost, and multiplying a partial-window mean
        by full wall time silently invents energy for unobserved intervals.

        This is a **sampled estimate**, not a hardware energy counter: the
        sampling interval bounds its resolution and short power excursions are
        averaged away. Check :attr:`coverage` before trusting the figure --
        well under 1.0 means intervals were missed, so this under-reports the
        true energy of the run.

        Returns ``None`` (never ``0.0``) when no samples were collected, so a
        missing measurement can never masquerade as "used no energy".
        """
        readings = [s.package_mw for s in self._samples if s.package_mw is not None]
        if not readings:
            return None
        mean_w = (sum(readings) / len(readings)) / 1000.0
        # Prefer the observed window; fall back to wall time only if the
        # per-sample durations are unavailable (e.g. the text fallback).
        window = self.sampled_s if self.sampled_s > 0 else self.elapsed_s
        return mean_w * window

    def mean_power_mw(self) -> dict[str, float | None]:
        """Mean per-domain power in mW; each entry ``None`` when unsampled."""
        out: dict[str, float | None] = {}
        for key, attr in (
            ("cpu", "cpu_mw"),
            ("gpu", "gpu_mw"),
            ("ane", "ane_mw"),
            ("package", "package_mw"),
        ):
            vals = [getattr(s, attr) for s in self._samples]
            vals = [v for v in vals if v is not None]
            out[key] = (sum(vals) / len(vals)) if vals else None
        return out


__all__ = [
    "POWERMETRICS_BIN",
    "PowerSample",
    "PowerSampler",
    "parse_plist_samples",
    "parse_text_samples",
]

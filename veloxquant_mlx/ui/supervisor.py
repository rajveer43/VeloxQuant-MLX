"""Supervises the ``veloxquant serve`` subprocess for the control panel (#34).

The panel never runs inference in its own process. It spawns the same
``veloxquant serve`` command a user would type, watches its stdout for the
``VELOXQUANT_READY`` handshake, and reports what that handshake says — so the
UI cannot advertise an endpoint or a compression mode the backend did not
actually announce.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from veloxquant_mlx.cli.serve import READY_PREFIX

#: Lines of server output kept for the log pane. Bounded so a long-running
#: server cannot grow the panel's memory without limit.
LOG_CAPACITY = 500

#: Interpreter-shutdown fallout that appears *after* the real error and buries
#: it under ~30 lines of frames. Dropping these keeps the actionable line
#: visible; genuine tracebacks are still shown in full.
_NOISE_MARKERS = (
    "cannot schedule new futures after interpreter shutdown",
    "site-packages/tqdm/",
    "concurrent/futures/",
    "Exception ignored in",
    "^^^^^^",
)


def _is_noise(line: str) -> bool:
    return any(marker in line for marker in _NOISE_MARKERS)


@dataclass
class LogLine:
    stream: str  # "stdout" | "stderr" | "panel"
    text: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"stream": self.stream, "text": self.text, "ts": self.ts}


class ServerSupervisor:
    """Owns at most one ``veloxquant serve`` child process.

    States: ``stopped`` → ``starting`` → ``running`` → ``stopped``/``error``.
    ``running`` is entered only on the READY handshake, never on a timer, so
    the UI cannot claim readiness while the model is still loading.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._logs: Deque[LogLine] = deque(maxlen=LOG_CAPACITY)
        self._state = "stopped"
        self._ready: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._config: Dict[str, Any] = {}

    # --- introspection ----------------------------------------------------

    @property
    def state(self) -> str:
        self._reap()
        return self._state

    def status(self) -> Dict[str, Any]:
        state = self.state
        return {
            "state": state,
            "pid": self._proc.pid if self._proc and state != "stopped" else None,
            "ready": self._ready,
            "error": self._error,
            "config": self._config,
        }

    def logs(self, since: int = 0) -> Dict[str, Any]:
        lines = list(self._logs)
        return {
            "lines": [l.to_dict() for l in lines[since:]],
            "total": len(lines),
        }

    # --- lifecycle --------------------------------------------------------

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._state in ("starting", "running"):
                raise RuntimeError("server is already running; stop it first")

            model = (config.get("model") or "").strip()
            if not model:
                # #34: cannot start an empty server.
                raise ValueError("a model id or local path is required")

            method = (config.get("method") or "").strip()
            self._assert_servable(method)
            self._assert_valid_overrides(method, config.get("overrides") or {})

            cmd = self._build_command(config, model, method)

            self._logs.clear()
            self._ready = None
            self._error = None
            self._config = dict(config)
            self._state = "starting"
            self._log("panel", f"$ {' '.join(cmd)}")

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    # New process group so stop() can signal the whole tree;
                    # mlx_lm spawns worker threads that ignore a bare SIGTERM
                    # to the parent alone.
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except OSError as exc:
                self._state = "error"
                self._error = f"could not launch server: {exc}"
                raise RuntimeError(self._error) from None

            threading.Thread(
                target=self._pump, args=(self._proc.stdout, "stdout"), daemon=True
            ).start()
            threading.Thread(
                target=self._pump, args=(self._proc.stderr, "stderr"), daemon=True
            ).start()

        return self.status()

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._state = "stopped"
                self._proc = None
                return self.status()

            self._log("panel", "stopping server ...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._log("panel", "did not exit in time; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=5)

            self._proc = None
            self._state = "stopped"
            self._ready = None
            self._log("panel", "stopped.")

        return self.status()

    # --- internals --------------------------------------------------------

    def _build_command(
        self, config: Dict[str, Any], model: str, method: str
    ) -> List[str]:
        cmd = [
            sys.executable, "-m", "veloxquant_mlx", "serve",
            "--model", model,
            "--method", method,
            "--bits", str(int(config.get("bits", 2))),
            "--host", str(config.get("host", "127.0.0.1")),
            "--port", str(int(config.get("port", 8000))),
            "--max-tokens", str(int(config.get("max_tokens", 512))),
            "--temp", str(float(config.get("temp", 0.0))),
            "--top-p", str(float(config.get("top_p", 1.0))),
        ]
        if config.get("seed") is not None:
            cmd += ["--seed", str(int(config["seed"]))]

        # Method-specific knobs travel as repeated --set FIELD=VALUE, so the
        # panel and the CLI share one validation path rather than two.
        for name, value in sorted((config.get("overrides") or {}).items()):
            cmd += ["--set", f"{name}={'' if value is None else value}"]

        return cmd

    def _assert_servable(self, method: str) -> None:
        """Refuse crash-tier methods here as well as in the CLI.

        The CLI is the real gate, but failing in the panel gives an immediate,
        readable message instead of a subprocess that dies a second later.
        """
        from veloxquant_mlx.cache.registry import get_method

        try:
            info = get_method(method)
        except KeyError as exc:
            raise ValueError(str(exc)) from None

        if not info.serve_tier.is_servable:
            raise ValueError(
                f"{method!r} cannot be served: "
                f"{info.unsupported_reason or 'unsupported'}"
            )

    def _assert_valid_overrides(
        self, method: str, overrides: Dict[str, Any]
    ) -> None:
        """Reject knobs that don't belong to this method, or won't parse.

        Catching it here turns a subprocess that dies a second after Start into
        an immediate, field-named message in the UI.
        """
        if not overrides:
            return

        from veloxquant_mlx.cache.registry import describe_field, get_method

        allowed = set(get_method(method).config_fields)
        for name, value in overrides.items():
            if name not in allowed:
                raise ValueError(
                    f"{name!r} is not a setting for {method!r}; "
                    f"valid: {', '.join(sorted(allowed)) or 'none'}"
                )

            schema = describe_field(name)
            if value is None or value == "":
                if not schema["optional"]:
                    raise ValueError(f"{name!r} requires a value")
                continue

            try:
                if schema["type"] == "int":
                    int(value)
                elif schema["type"] == "float":
                    float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name!r} expects {schema['type']}, got {value!r}"
                ) from None

    def _pump(self, pipe: Any, stream: str) -> None:
        try:
            for raw in iter(pipe.readline, ""):
                line = raw.rstrip("\n")
                if not line:
                    continue

                if stream == "stdout" and line.startswith(READY_PREFIX):
                    self._on_ready(line)
                    continue

                if _is_noise(line):
                    continue

                self._log(stream, line)
        except (ValueError, OSError):
            pass  # pipe closed during shutdown
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _on_ready(self, line: str) -> None:
        try:
            payload = json.loads(line[len(READY_PREFIX):])
        except json.JSONDecodeError:
            self._log("panel", "received a malformed ready handshake")
            return

        self._ready = payload
        self._state = "running"
        self._log(
            "panel",
            f"ready — {payload.get('layer_caches')} layer caches, "
            f"method={payload.get('method')} bits={payload.get('bits')}",
        )

    def _log(self, stream: str, text: str) -> None:
        self._logs.append(LogLine(stream=stream, text=text))

    def _reap(self) -> None:
        """Detect a child that died on its own (bad port, OOM, external kill)."""
        proc = self._proc
        if proc is None or self._state == "stopped":
            return

        code = proc.poll()
        if code is None:
            return

        self._proc = None
        if self._state == "starting":
            reason = self._diagnose()
            self._error = (
                f"server exited with code {code} before becoming ready"
                + (f": {reason}" if reason else ".")
            )
            self._state = "error"
        else:
            self._error = f"server exited unexpectedly with code {code}"
            self._state = "error"
        self._ready = None
        self._log("panel", self._error)

    def _diagnose(self) -> Optional[str]:
        """Turn the child's stderr into one line a user can act on.

        A raw traceback tail is nearly useless in the UI — the last three lines
        of a Python stack say nothing about what went wrong. Match the failures
        #34 calls out by name first, and fall back to the exception line rather
        than to interior frames.
        """
        stderr = [l.text for l in self._logs if l.stream == "stderr"]
        blob = "\n".join(stderr)

        port = self._config.get("port")
        signatures = [
            ("address already in use", f"port {port} is already in use"),
            ("errno 48", f"port {port} is already in use"),
            ("can't assign requested address",
             f"cannot bind {self._config.get('host')!r}"),
            ("cannot be served", "the selected method cannot be served"),
            ("repository not found", "model not found on Hugging Face"),
            ("does not appear to have a file named",
             "the model path is missing required files"),
            ("no such file or directory", "model path does not exist"),
            ("out of memory", "ran out of memory loading the model"),
            ("metal::malloc", "ran out of GPU memory loading the model"),
        ]
        lowered = blob.lower()
        for needle, message in signatures:
            if needle in lowered:
                return message

        # Prefer a real exception line ("ValueError: ...") over stack frames.
        for line in reversed(stderr):
            text = line.strip()
            if text.startswith(("File \"", "Traceback", "  ")) or not text:
                continue
            return text[:200]
        return None

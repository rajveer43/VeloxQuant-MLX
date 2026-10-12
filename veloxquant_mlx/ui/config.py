"""Persisted panel settings (#34: remember network + last-used model)."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".veloxquant" / "panel.json"

DEFAULTS: dict[str, Any] = {
    "model": "",
    "method": "turboquant_rvq",
    "bits": 2,
    "host": "127.0.0.1",
    "port": 8000,
    "max_tokens": 512,
    "temp": 0.0,
    "top_p": 1.0,
    # Method-specific KVCacheConfig knobs, e.g. {"kivi_group_size": 64}.
    "overrides": {},
}

#: Only these keys are persisted. An allowlist rather than a blocklist so a
#: field added to the UI cannot silently start being written to disk.
_PERSISTED = set(DEFAULTS)

#: ThreadingHTTPServer runs each request on its own thread, so two saves can
#: interleave their load-merge-write and lose one update.
_save_lock = threading.Lock()


def load_config() -> dict[str, Any]:
    """Load persisted panel settings from ``CONFIG_PATH``, falling back to ``DEFAULTS``.

    Missing file, unreadable file, or malformed JSON all fall back silently
    to :data:`DEFAULTS` (a fresh panel with no saved state shouldn't error).
    Only keys present in :data:`DEFAULTS` are read from the stored file, so
    a stale or hand-edited key on disk cannot inject an unexpected setting.
    """
    config = dict(DEFAULTS)
    try:
        stored = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return config

    if isinstance(stored, dict):
        config.update({k: v for k, v in stored.items() if k in _PERSISTED})
    return config


def save_config(config: dict[str, Any]) -> None:
    """Merge ``config`` onto the currently persisted settings and write them to disk.

    Merges rather than overwrites, so a partial ``config`` (e.g. from a
    single settings-form field) doesn't wipe out other previously-saved
    keys. Only keys present in :data:`DEFAULTS` are persisted. Silently
    does nothing if the config directory can't be created or written (e.g.
    a read-only home directory) — a persistence failure should not block
    starting the server.
    """
    with _save_lock:
        merged = load_config()
        merged.update({k: v for k, v in config.items() if k in _PERSISTED})

        tmp_name = None
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash or concurrent reader never sees a
            # torn file (which load_config would silently treat as "defaults").
            fd, tmp_name = tempfile.mkstemp(dir=CONFIG_PATH.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(merged, indent=2))
            Path(tmp_name).replace(CONFIG_PATH)
        except OSError:
            # a read-only home should not break Start
            if tmp_name:
                with contextlib.suppress(OSError):
                    Path(tmp_name).unlink()

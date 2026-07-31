"""Persisted panel settings (#34: remember network + last-used model)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path.home() / ".veloxquant" / "panel.json"

DEFAULTS: Dict[str, Any] = {
    "model": "",
    "method": "turboquant_rvq",
    "bits": 2,
    "host": "127.0.0.1",
    "port": 8000,
    "max_tokens": 512,
    "temp": 0.0,
    "top_p": 1.0,
}

#: Only these keys are persisted. An allowlist rather than a blocklist so a
#: field added to the UI cannot silently start being written to disk.
_PERSISTED = set(DEFAULTS)


def load_config() -> Dict[str, Any]:
    config = dict(DEFAULTS)
    try:
        stored = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return config

    if isinstance(stored, dict):
        config.update({k: v for k, v in stored.items() if k in _PERSISTED})
    return config


def save_config(config: Dict[str, Any]) -> None:
    merged = load_config()
    merged.update({k: v for k, v in config.items() if k in _PERSISTED})

    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(merged, indent=2))
    except OSError:
        pass  # a read-only home should not break Start

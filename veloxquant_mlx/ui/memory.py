"""Measured memory reporting for the control panel.

#27 forbids *claiming* memory savings the backend cannot prove. It does not
require staying silent about memory — and showing genuinely measured numbers
next to the accounting-only estimate is strictly more honest than showing
neither, because the gap between them becomes visible rather than asserted.

Every value carries ``source: "measured"``. Anything unavailable is reported as
``None`` with a reason, never as ``0`` — a zero in a memory panel reads as
"nothing used", which would be a fresh lie in place of a missing number.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: psutil ships in the `dev` extra, not the runtime deps, so a normal
#: `pip install VeloxQuant-MLX` will not have it. Degrade, never fabricate.
try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - depends on install extras
    _psutil = None


def _mlx_memory() -> Dict[str, Any]:
    """MLX memory — deliberately *not* reported by the panel.

    ``mx.get_active_memory()`` is process-local, so calling it here measures the
    panel's own MLX usage (effectively zero), not the server's. Showing that
    beside the server's RSS under a "measured" tag would be technically true and
    completely misleading — the reader would take it as the server's GPU memory.

    Reporting it requires asking the server process, which needs the
    ``/v1/kv/stats`` endpoint from #27. Until then this states why it is absent
    rather than printing a number that describes the wrong process.
    """
    return {
        "active_bytes": None,
        "peak_bytes": None,
        "unavailable_reason": "coming soon",
    }


def _process_memory(pid: Optional[int]) -> Dict[str, Any]:
    """RSS of the *server* process, not the panel's own."""
    if pid is None:
        return {"rss_bytes": None, "unavailable_reason": "no server is running"}

    if _psutil is None:
        return {
            "rss_bytes": None,
            "unavailable_reason": "not available in this install",
        }

    try:
        return {
            "rss_bytes": int(_psutil.Process(pid).memory_info().rss),
            "unavailable_reason": None,
        }
    except Exception:
        return {"rss_bytes": None, "unavailable_reason": "not available right now"}


def memory_report(pid: Optional[int] = None) -> Dict[str, Any]:
    """Everything the panel's memory card needs.

    ``source`` is part of the payload rather than UI copy, so a consumer cannot
    render these numbers without also knowing they are measurements — and,
    by contrast, that the cache byte counters beside them are not.
    """
    process = _process_memory(pid)
    mlx = _mlx_memory()

    return {
        "source": "measured",
        "process": process,
        "mlx": mlx,
        "note": (
            "This is the real memory your Mac is using right now. The "
            "compression numbers shown elsewhere describe your conversation "
            "data, not your Mac's memory — so don't expect this number to "
            "drop when you pick a higher compression method yet."
        ),
    }

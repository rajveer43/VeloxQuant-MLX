"""Local web control panel for VeloxQuant-MLX (#33 / #34)."""
from __future__ import annotations

__all__ = ["serve_panel"]


def serve_panel(*args, **kwargs):
    from veloxquant_mlx.ui.server import serve_panel as _serve

    return _serve(*args, **kwargs)

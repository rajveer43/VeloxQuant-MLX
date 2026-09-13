"""Command implementations backing the ``python -m veloxquant_mlx`` CLI.

Each sibling module implements one subcommand (``precompute``, ``benchmark``,
``serve``, ``profile``, ``recommend``, ``methods``, ``auto_config``,
``worker``, ``panel``); this package holds no re-exports of its own, so
callers import each command module directly.
"""

from __future__ import annotations

"""Re-exports the :class:`ArtifactStore` ABC for the ``artifacts`` subpackage.

Lets callers write ``from veloxquant_mlx.artifacts.base import ArtifactStore``
alongside the concrete backends in this package instead of reaching into
``core.abstractions`` directly.
"""

from __future__ import annotations

from veloxquant_mlx.core.abstractions import ArtifactStore

__all__ = ["ArtifactStore"]

"""Shared pytest fixtures for veloxquant_mlx tests."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def d() -> int:
    """Default head dimension."""
    return 128


@pytest.fixture(scope="session")
def seed() -> int:
    return 42

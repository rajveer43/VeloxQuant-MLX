from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Type


class _BaseRegistry:
    """Thread-safe singleton registry backing class-decorator registration."""

    _lock: threading.Lock
    _registry: Dict[str, type]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}
        cls._lock = threading.Lock()

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers a concrete class under name.

        Args:
            name: Registry key (e.g. ``"qjl"``, ``"turboquant_prod"``).

        Returns:
            Decorator that registers the class and returns it unchanged.

        Example::

            @QuantizerRegistry.register("qjl")
            class QJLQuantizer(Quantizer): ...
        """

        def decorator(target_cls: type) -> type:
            with cls._lock:
                if name in cls._registry:
                    raise KeyError(
                        f"{cls.__name__}: '{name}' is already registered by "
                        f"{cls._registry[name].__name__}. "
                        f"Cannot re-register with {target_cls.__name__}."
                    )
                cls._registry[name] = target_cls
            return target_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a registered class by name.

        Args:
            name: Registry key.

        Returns:
            The registered class.

        Raises:
            KeyError: If name is not registered.
        """
        with cls._lock:
            if name not in cls._registry:
                available = sorted(cls._registry.keys())
                raise KeyError(f"{cls.__name__}: '{name}' not registered. Available: {available}")
            return cls._registry[name]

    @classmethod
    def list_names(cls) -> list[str]:
        """Return all registered names sorted alphabetically."""
        with cls._lock:
            return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if name is registered."""
        with cls._lock:
            return name in cls._registry


class QuantizerRegistry(_BaseRegistry):
    """Registry for Quantizer concrete classes."""


class CodebookRegistry(_BaseRegistry):
    """Registry for CodebookStrategy concrete classes."""


class PreconditionerRegistry(_BaseRegistry):
    """Registry for Preconditioner concrete classes."""

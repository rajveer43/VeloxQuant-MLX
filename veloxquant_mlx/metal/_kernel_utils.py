"""Shared kernel-source-loading and compile-once-cache helpers.

Every ``metal/_*.py`` wrapper reads its ``.metal`` source from ``metal/src/``
once at import time and lazily compiles ``mx.fast.metal_kernel`` instances
keyed by the template parameters that vary per call (head dim, bit width,
SIMD-group count, ...), since recompiling on every call would be far too
slow. This module is the one place that logic lives, so a future fix (a
race condition, LRU eviction, path resolution under a frozen/zipped
package, ...) only needs to be applied once instead of hand-copied across
every wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")


def read_kernel_source(module_file: str, filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/.

    Args:
        module_file: The calling module's ``__file__`` (its ``src/``
            sibling directory is where ``filename`` is looked up).
        filename: The ``.metal`` file's name, e.g. ``"qjl_encode.metal"``.
    """
    return (Path(module_file).parent / "src" / filename).read_text()


class KernelCache(dict):
    """Compile-once memoization for ``mx.fast.metal_kernel`` instances.

    A dict keyed by each kernel factory's own template-parameter tuple
    (e.g. ``(D, nsg, heads_per_kv)``) so a kernel is only jit-compiled the
    first time a given parameter combination is requested. Subclasses dict
    (rather than wrapping one) so existing white-box tests that inspect a
    module's cache directly — ``key in cache``, ``len(cache)``,
    ``dict(cache)`` — keep working unchanged.
    """

    def get_or_create(self, key: object, factory: Callable[[], _T]) -> _T:
        """Return the cached kernel for key, building it via factory if absent."""
        if key not in self:
            self[key] = factory()
        return self[key]

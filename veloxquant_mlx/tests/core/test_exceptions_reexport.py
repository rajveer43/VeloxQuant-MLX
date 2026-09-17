"""Regression for #403: core/__init__.py must re-export every exception type.

core/__init__.py's docstring claims to re-export "the package's exception
hierarchy" in full. BlockPoolExhaustedError and OwnerAlreadyActiveError were
defined in core/exceptions.py but missing from that re-export list even
though they're used by veloxquant_mlx.memory.block_pool — this test catches
the next exception class that's added to core/exceptions.py but forgotten
here.
"""

from __future__ import annotations

import inspect

import veloxquant_mlx.core as core_pkg
import veloxquant_mlx.core.exceptions as exceptions_mod


def _defined_exception_names() -> set[str]:
    return {
        name
        for name, obj in vars(exceptions_mod).items()
        if inspect.isclass(obj)
        and issubclass(obj, Exception)
        and obj.__module__ == exceptions_mod.__name__
    }


def test_every_exception_is_reexported_from_core():
    defined = _defined_exception_names()
    assert defined, "sanity check: exceptions module should define at least one exception"
    missing = defined - set(core_pkg.__all__)
    assert not missing, f"core/__init__.py is missing these exceptions from __all__: {missing}"
    for name in defined:
        assert getattr(core_pkg, name, None) is getattr(exceptions_mod, name), (
            f"core.{name} does not match core.exceptions.{name}"
        )

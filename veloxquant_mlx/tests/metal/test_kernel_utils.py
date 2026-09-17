"""Tests for #407: shared kernel-source-loading + caching helpers.

Before this, 20 of the 27 files in metal/*.py independently defined an
identical ``_read_kernel_source`` helper plus their own ``_cache: dict = {}``
memoization block. These pin the extracted ``metal/_kernel_utils.py``
module's own behavior, and guard against the 20 call sites drifting back to
a locally-defined copy.
"""

from __future__ import annotations

import ast
from pathlib import Path

from veloxquant_mlx.metal._kernel_utils import KernelCache, read_kernel_source

# The 20 files named in issue #407 (all of metal/*.py except the three that
# use functools.lru_cache instead: _pyramidkv_evict.py, _snapkv_select.py,
# _tova_evict.py) plus fused_sdpa.py, which was not named in the issue's
# list but has the identical pattern.
_EXPECTED_MIGRATED_MODULES = {
    "_bit_packing",
    "_comm_vq",
    "_crosskv_rope",
    "_experimental_streaming_prefill",
    "_flash_prefill",
    "_h2o_evict",
    "_keyformer_evict",
    "_kivi_quant",
    "_qfilters_evict",
    "_qjl",
    "_rabitq",
    "_rabitq_attend",
    "_rabitq_encode",
    "_rabitq_prefill",
    "_rabitq_values",
    "_rvq_attend",
    "_rvq_quant_pack",
    "_scalar_attend",
    "_scalar_quant",
    "_vecinfer",
    "fused_sdpa",
}


def test_read_kernel_source_reads_from_sibling_src_dir(tmp_path):
    module_dir = tmp_path / "some_module_dir"
    (module_dir / "src").mkdir(parents=True)
    (module_dir / "src" / "example.metal").write_text("kernel body")
    fake_module_file = str(module_dir / "wrapper.py")

    assert read_kernel_source(fake_module_file, "example.metal") == "kernel body"


def test_kernel_cache_builds_once_per_key():
    cache = KernelCache()
    calls = []

    def factory():
        calls.append(1)
        return object()

    first = cache.get_or_create("k", factory)
    second = cache.get_or_create("k", factory)

    assert first is second
    assert len(calls) == 1


def test_kernel_cache_builds_separately_per_distinct_key():
    cache = KernelCache()
    built = cache.get_or_create(("a", 1), lambda: "a1")
    built2 = cache.get_or_create(("a", 2), lambda: "a2")

    assert built == "a1"
    assert built2 == "a2"
    assert len(cache) == 2


def test_kernel_cache_supports_dict_protocol_for_whitebox_tests():
    """Existing tests reach into a module's _cache directly (`key in _cache`,
    `len(_cache)`, `dict(_cache)`) — KernelCache must keep behaving like a
    dict, not just expose get_or_create.
    """
    cache = KernelCache()
    cache.get_or_create("k1", lambda: "v1")

    assert "k1" in cache
    assert len(cache) == 1
    assert dict(cache) == {"k1": "v1"}
    cache.clear()
    assert len(cache) == 0


def test_every_expected_metal_module_imports_shared_kernel_utils():
    """Regression for #407: a file that reads .metal source + memoizes a
    kernel must import _read_kernel_source/KernelCache from _kernel_utils
    rather than redefining its own copy — this is what actually eliminates
    the 20-file hand-sync burden the issue describes.
    """
    metal_dir = Path(__file__).parent.parent.parent / "metal"
    missing = []
    for name in sorted(_EXPECTED_MIGRATED_MODULES):
        path = metal_dir / f"{name}.py"
        assert path.exists(), f"expected module file not found: {path}"
        tree = ast.parse(path.read_text())
        imports_shared = any(
            isinstance(node, ast.ImportFrom) and node.module == "veloxquant_mlx.metal._kernel_utils"
            for node in ast.walk(tree)
        )
        if not imports_shared:
            missing.append(name)
    assert not missing, f"modules missing the shared _kernel_utils import: {missing}"


def test_no_migrated_module_redefines_its_own_source_reader_body():
    """A module may still keep a local `_read_kernel_source` name (some
    callers use it without the module-file argument), but its body must
    delegate to the shared helper rather than reimplementing
    Path(__file__).parent / "src" / filename locally.
    """
    metal_dir = Path(__file__).parent.parent.parent / "metal"
    offenders = []
    for name in sorted(_EXPECTED_MIGRATED_MODULES):
        src = (metal_dir / f"{name}.py").read_text()
        if "Path(__file__)" in src:
            offenders.append(name)
    assert not offenders, f"modules still hand-rolling Path(__file__): {offenders}"

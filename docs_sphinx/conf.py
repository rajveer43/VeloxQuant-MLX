"""Sphinx configuration for the VeloxQuant-MLX API reference.

This builds an autodoc-generated API reference from the veloxquant_mlx
package's docstrings, hosted on Read the Docs. It is separate from:
- docs/        -- research findings and design notes (hand-written)
- docs-site/   -- the Docusaurus user guide at veloxquant.dev

``mlx``/``mlx_lm`` are macOS/Apple-Silicon-only and have no Linux wheels,
so Read the Docs' Ubuntu build image can't install them. They're mocked
via autodoc_mock_imports below instead -- this is why Read the Docs must
install the package with ``pip install --no-deps -e .`` rather than a
plain ``pip install -e .[docs]`` (see .readthedocs.yaml).
"""

from __future__ import annotations

import importlib.metadata

project = "VeloxQuant-MLX"
copyright = "2026, Rajveer Rathod"
author = "Rajveer Rathod"
release = importlib.metadata.version("VeloxQuant-MLX")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# autosummary_generate is left off: the top-level `veloxquant_mlx` autosummary
# table (docs_sphinx/api/veloxquant_mlx.rst) only links to definitions that
# already have their own automodule-generated page in the same api/ tree --
# generating separate stub pages for these re-exported names would document
# each of them twice (see docs_sphinx/api/veloxquant_mlx.rst for why).
autosummary_generate = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- autodoc / napoleon -----------------------------------------------------
# mlx and mlx_lm have no Linux wheels (macOS/Apple-Silicon-only), so they
# can't be installed on the Read the Docs build image. autodoc imports every
# module it documents, so anything transitively importing these must be
# mocked rather than actually installed.
autodoc_mock_imports = ["mlx", "mlx_lm"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Docstrings across the package are Google-style (Args:/Returns:), confirmed
# by inspecting veloxquant_mlx/allocators/*.py.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
# Renders an "Attributes:" docstring section as :ivar:/:vartype: field-list
# entries instead of separate `.. attribute::` blocks -- without this,
# autodoc's own dataclass-field introspection documents each annotated field
# a second time, producing hundreds of "duplicate object description"
# warnings across the many dataclasses in this codebase (BenchmarkRecord,
# *State classes, etc).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# --- HTML output -------------------------------------------------------------
# Furo chosen over sphinx-rtd-theme for a more modern look; RTD supports any
# theme, it just defaults its own if none is set.
html_theme = "furo"
html_title = f"{project} {version}"
html_static_path = []

"""Sphinx configuration for the PePsY documentation.

The hand-written Markdown pages are the user guide. ``sphinx-autoapi`` adds a
generated reference from the canonical source packages without importing the
package during discovery, which keeps optional integrations out of the docs
build's critical path.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tomllib
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "PePsY"
copyright = "2026, Quantinuum"
author = "Quantinuum"
release = tomllib.loads(
    (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
]

root_doc = "index"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# AutoAPI parses the source tree instead of importing it. Restricting the
# options to public members keeps private implementation details out of the
# user-facing reference while respecting each module's ``__all__``.
autoapi_dirs = [str(PROJECT_ROOT / "src")]
autoapi_root = "api/reference"
autoapi_add_toctree_entry = True
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_keep_files = True
autoapi_ignore = [
    "*/_api.py",
    "*/_version.py",
    "*/_internal/*",
]
# Lazy re-exports and optional backends are intentionally not importable from
# every implementation module during AutoAPI's static name resolution.
suppress_warnings = ["autoapi.python_import_resolution"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "quimb": ("https://quimb.readthedocs.io/en/latest", None),
}

html_theme = "furo"
html_title = "PePsY documentation"
docs_ref = os.environ.get(
    "PEPSY_DOCS_REF",
    os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "develop"),
)
html_theme_options = {
    "source_repository": "https://github.com/quantinuum-dev/pepsy",
    "source_branch": docs_ref,
    "source_directory": "docs/",
}


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return a GitHub URL pointing at one documented Python object."""

    if domain != "py" or not info.get("module"):
        return None

    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info.get("fullname", "").split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        source_file = Path(inspect.getsourcefile(obj) or "").resolve()
        start_line = inspect.getsourcelines(obj)[1]
        end_line = start_line + len(inspect.getsource(obj).splitlines()) - 1
        relative_path = source_file.relative_to(PROJECT_ROOT).as_posix()
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        # A source link is helpful but must not make an optional-domain page
        # fail when its backend is not installed in the docs environment.
        return None

    ref = docs_ref
    return (
        "https://github.com/quantinuum-dev/pepsy/blob/"
        f"{ref}/{relative_path}#L{start_line}-L{end_line}"
    )

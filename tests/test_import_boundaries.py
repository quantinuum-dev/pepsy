"""Import-weight contracts for the core package facade."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

_OPTIONAL_ROOTS = (
    "flax",
    "guppy",
    "jax",
    "netket",
    "nevergrad",
    "nlopt",
    "scipy",
    "stim",
    "symmray",
    "torch",
)

_CORE_MODULES = (
    "pepsy.backends",
    "pepsy.boundary",
    "pepsy.fitting",
    "pepsy.interop",
    "pepsy.operators",
    "pepsy.sampling",
    "pepsy.solvers",
    "pepsy.tensors",
)


def _run_clean_import(script: str) -> set[str]:
    """Run an import probe with only Pepsy's source tree added."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split())


def test_import_pepsy_does_not_load_optional_backends():
    """The root facade must stay usable without advanced backend imports."""
    roots = repr(_OPTIONAL_ROOTS)
    loaded = _run_clean_import(
        f"""
import sys
import pepsy

roots = {roots}
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_core_namespaces_do_not_load_optional_backends():
    """Core namespace discovery must remain independent of optional stacks."""
    modules = repr(_CORE_MODULES)
    roots = repr(_OPTIONAL_ROOTS)
    loaded = _run_clean_import(
        f"""
import importlib
import sys

for module_name in {modules}:
    importlib.import_module(module_name)

roots = {roots}
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_experimental_namespace_is_lazy():
    """Discovering advanced domains must not import their implementations."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.experimental

print(*sorted(
    name for name in sys.modules
    if name == 'pepsy.vmc'
    or name.startswith('pepsy.vmc.')
    or name == 'pepsy.bp'
    or name.startswith('pepsy.bp.')
    or name == 'pepsy.optimizers.qmera'
    or name.startswith('pepsy.optimizers.qmera.')
))
"""
    )
    assert not loaded

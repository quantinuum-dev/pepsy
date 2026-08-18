"""Compatibility namespace for the former qMERA package name.

Use :mod:`pepsy.optimizers.qmera` for all new code. This module only aliases
the canonical qMERA implementation so existing imports keep working during
the namespace migration.
"""

from importlib import import_module
import sys
import warnings


warnings.warn(
    "pepsy.optimizers.mera is a compatibility alias; use "
    "pepsy.optimizers.qmera instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..qmera import *  # noqa: F401,F403
from ..qmera import __all__ as __all__


for _module_name in (
    "builders",
    "cache",
    "compiled",
    "fermions",
    "gates",
    "geometry",
    "layout",
    "lightcones",
    "parametric",
    "prototype",
    "schedules",
    "schematics",
    "terms",
):
    sys.modules.setdefault(
        f"{__name__}.{_module_name}",
        import_module(f"..qmera.{_module_name}", __name__),
    )

del import_module, sys, warnings, _module_name

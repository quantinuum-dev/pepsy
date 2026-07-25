"""Compatibility helpers for deprecated Pepsy module paths."""

from __future__ import annotations

import warnings
from importlib import import_module


def install_deprecated_module(namespace, target: str, replacement: str) -> None:
    """Populate a legacy module facade and emit one import-time warning."""
    warnings.warn(
        f"{namespace['__name__']} is deprecated; import from {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    module = import_module(target)
    namespace["_legacy_module"] = module
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = (name for name in dir(module) if not name.startswith("_"))
    namespace["__all__"] = tuple(exported)

    def __getattr__(name):
        return getattr(module, name)

    def __dir__():
        return sorted(set(namespace) | set(dir(module)))

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__

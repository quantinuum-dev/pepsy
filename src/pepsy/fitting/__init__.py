"""Local fitting utilities."""

from importlib import import_module

from .local import FIT, internal_inds

__all__ = ["FIT", "internal_inds", "local"]


def __getattr__(name):
    if name == "local":
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

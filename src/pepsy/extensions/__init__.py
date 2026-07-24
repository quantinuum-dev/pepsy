"""Optional and advanced Pepsy extension namespaces.

These modules make the heavier or more experimental domains explicit without
breaking their established import paths. Import an extension only when it is
needed, for example ``from pepsy.extensions.vmc import TorchVMCDriver``.
"""

from importlib import import_module

__all__ = ["bp", "mera", "stabilizer", "vmc"]


def __getattr__(name):
    if name in __all__:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


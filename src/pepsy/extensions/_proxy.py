"""Small helper for compatibility proxies under :mod:`pepsy.extensions`."""

from importlib import import_module


def make_proxy(module_name):
    """Return module-level ``__getattr__`` and ``__dir__`` implementations."""

    def __getattr__(name):
        return getattr(import_module(module_name), name)

    def __dir__():
        return sorted(set(globals()) | set(dir(import_module(module_name))))

    return __getattr__, __dir__


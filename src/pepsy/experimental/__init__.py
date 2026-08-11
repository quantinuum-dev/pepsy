"""Explicit namespace for advanced and research-stage Pepsy domains.

The stable user surface is the core package modules documented in
``docs/stability.md``. Advanced domains remain available here without making
their implementation details part of the default API contract.
"""

from importlib import import_module

_MODULES = {
    "bp": "pepsy.bp",
    "mera": "pepsy.optimizers.qmera",
    "mps_fit": "pepsy.experimental.mps_fit",
    "qmera": "pepsy.optimizers.qmera",
    "stabilizer": "pepsy.optimizers.stabilizer_tn",
    "symmetry": "pepsy.tensors.symmetric",
    "tree": "pepsy.optimizers.tree",
    "tree_stabilizer": "pepsy.optimizers.tree_stabilizer",
    "vmc": "pepsy.vmc",
}

__all__ = tuple(_MODULES)


def __getattr__(name):
    """Load an advanced domain only when explicitly requested."""
    target = _MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target)
    globals()[name] = module
    return module

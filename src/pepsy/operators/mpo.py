"""Compatibility facade for semantic and history-aware MPO construction.

The implementation now lives in :mod:`pepsy.operators.mpo_semantic`.  This
module preserves the original import path while the public family facades
point at the explicit semantic and basis owners.
"""

from . import mpo_basis as _basis
from . import mpo_semantic as _semantic

__all__ = [*_semantic.__all__, *_basis.__all__]


def __getattr__(name):
    try:
        return getattr(_semantic, name)
    except AttributeError:
        return getattr(_basis, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_semantic)) | set(dir(_basis)))

"""Compatibility facade for dense PEPO cluster construction.

The implementation now lives in :mod:`pepsy.operators.pepo_dense`.  The
historical ``cluster`` module remains importable so existing callers and
private compatibility paths do not need to change at once.
"""

from . import pepo_active as _active
from . import pepo_basis as _basis
from . import pepo_dense as _dense
from . import pepo_product as _product

__all__ = [
    *_dense.__all__,
    *_active.__all__,
    *_basis.__all__,
    *_product.__all__,
]


def __getattr__(name):
    for module in (_dense, _active, _basis, _product):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(dir(_dense))
        | set(dir(_active))
        | set(dir(_basis))
        | set(dir(_product))
    )

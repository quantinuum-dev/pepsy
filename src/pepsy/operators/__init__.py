"""Gate and Hamiltonian operators."""

from importlib import import_module

from . import gates as _gates
from .hamiltonians import ham_tn

for _name in _gates.__all__:
    globals()[_name] = getattr(_gates, _name)

__all__ = [*_gates.__all__, "ham_tn", "gate_apply", "gate_builders", "gates", "hamiltonians"]

del _name


def __getattr__(name):
    if name in {"gate_apply", "gate_builders", "gates", "hamiltonians"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

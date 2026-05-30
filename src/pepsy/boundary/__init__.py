"""Boundary-MPS contraction tools for PEPS-like tensor networks."""

from importlib import import_module

from .metrics import (
    BoundaryContractResult,
    build_bra_ket,
    contract_boundary,
    infidelity,
    normalize,
)
from .states import BdyMPS, make_numpy_array_caster
from .sweeps import CompBdy

__all__ = [
    "BdyMPS",
    "BoundaryContractResult",
    "CompBdy",
    "build_bra_ket",
    "contract_boundary",
    "infidelity",
    "make_numpy_array_caster",
    "normalize",
    "metrics",
    "states",
    "sweeps",
]


def __getattr__(name):
    if name in {"metrics", "states", "sweeps"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

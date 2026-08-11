"""Boundary-MPS contraction tools for PEPS-like tensor networks."""

from importlib import import_module

from .metrics import (
    BoundaryContractResult,
    boundary_norm,
    build_bra_ket,
    contract_boundary,
    contract_flat,
    infidelity,
    normalize,
    peps_fidelity,
    peps_infidelity,
    peps_norm,
    peps_normalize,
    quimb_ctmrg_projector_compat,
)
from .states import BdyMPS, make_numpy_array_caster
from .sweeps import BoundaryFitDiagnostic, CompBdy

__all__ = [
    "BdyMPS",
    "BoundaryContractResult",
    "BoundaryFitDiagnostic",
    "CompBdy",
    "boundary_norm",
    "build_bra_ket",
    "contract_boundary",
    "contract_flat",
    "infidelity",
    "make_numpy_array_caster",
    "normalize",
    "peps_fidelity",
    "peps_infidelity",
    "peps_norm",
    "peps_normalize",
    "quimb_ctmrg_projector_compat",
    "metrics",
    "states",
    "sweeps",
]


def __getattr__(name):
    if name in {"metrics", "states", "sweeps"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

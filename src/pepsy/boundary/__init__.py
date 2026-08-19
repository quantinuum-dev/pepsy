"""Lazy boundary-MPS contraction tools for PEPS-like tensor networks."""

from importlib import import_module
import warnings


_SYMBOL_MODULES = {
    "BoundaryContractResult": ".metrics",
    "boundary_norm": ".metrics",
    "build_bra_ket": ".metrics",
    "contract_boundary": ".metrics",
    "contract_flat": ".metrics",
    "infidelity": ".metrics",
    "normalize": ".metrics",
    "peps_fidelity": ".metrics",
    "peps_infidelity": ".metrics",
    "peps_norm": ".metrics",
    "peps_normalize": ".metrics",
    "quimb_ctmrg_projector_compat": ".metrics",
    "BdyMPS": ".states",
    "make_numpy_array_caster": ".states",
    "BoundaryFitDiagnostic": ".sweeps",
    "CompBdy": ".sweeps",
}

__all__ = [*_SYMBOL_MODULES, "metrics", "states", "sweeps"]

_DEPRECATED_ALIASES = {
    "normalize": "peps_normalize",
    "infidelity": "peps_infidelity",
}


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        canonical = _DEPRECATED_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"pepsy.boundary.{name} is a compatibility alias; use "
                f"pepsy.boundary.{canonical} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in {"metrics", "states", "sweeps"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

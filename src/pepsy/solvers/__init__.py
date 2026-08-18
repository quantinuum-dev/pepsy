"""Lazy parameter optimization solvers."""

from importlib import import_module

_SYMBOL_MODULES = {
    "FDSolver": ".gradient",
    "GradSolverResult": ".gradient",
    "GradientOptimizer": ".gradient",
    "SUPPORTED_SOLVERS": ".gradient",
    "optimize_packed_params": ".gradient",
}

__all__ = [*_SYMBOL_MODULES, "finite_difference", "gradient"]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in {"finite_difference", "gradient"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

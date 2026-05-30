"""Parameter optimization solvers."""

from importlib import import_module

from .gradient import (
    FDSolver,
    GradSolverResult,
    GradientOptimizer,
    SUPPORTED_SOLVERS,
    optimize_packed_params,
)

__all__ = [
    "FDSolver",
    "GradSolverResult",
    "GradientOptimizer",
    "SUPPORTED_SOLVERS",
    "optimize_packed_params",
    "finite_difference",
    "gradient",
]


def __getattr__(name):
    if name in {"finite_difference", "gradient"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Tree-tensor-network stabilizer simulator."""

import warnings

from ..stabilizer_tn.records import StabilizerTreeRunResult
from .optimizer import (
    StabilizerTreeSimulator,
    run_stabilizer_tree_stream,
)

_DEPRECATED_ALIASES = {"TreeStabOptimizer": "StabilizerTreeSimulator"}

__all__ = [
    "StabilizerTreeRunResult",
    "StabilizerTreeSimulator",
    "TreeStabOptimizer",
    "run_stabilizer_tree_stream",
]


def __getattr__(name):
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pepsy.optimizers.tree_stabilizer.{name} is a compatibility alias; "
        f"use pepsy.optimizers.tree_stabilizer.{canonical} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = globals()[canonical]
    globals()[name] = value
    return value

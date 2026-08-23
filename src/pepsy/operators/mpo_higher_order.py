"""Public facade for the paper-style higher-order MPO construction.

This module names the SciPost higher-order family explicitly.  It owns the
history-aware construction of ``exp(step * H)`` on a finite chain; it does not
own spatial MPO cluster residuals or ordered multi-factor products.  The
implementation currently remains in :mod:`pepsy.operators.mpo` during the
extraction, so these exports are object-identical compatibility facades.

Use :mod:`pepsy.operators.mpo_cluster` for connected interval/graph cluster
expansions and their joint local ``exp(A) @ exp(B) @ ...`` construction.
"""

from .mpo import (
    CompiledMPOExp,
    FirstDegreeMPO,
    MPODifferentiableCompressionReport,
    MPOLocalOperatorTerm,
    MPONumericalCompressionReport,
    MPOBasis,
    MPOCompressionReport,
    MPOLevel,
    MPOLevelToken,
    MPOParameter,
    MPOProductTerm,
    exp_mpo,
)
from .mpo_space import MPOBraiding, MPOPhysicalSpace

__all__ = [
    "MPOParameter",
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOLocalOperatorTerm",
    "MPOBraiding",
    "MPOPhysicalSpace",
    "MPOCompressionReport",
    "MPONumericalCompressionReport",
    "MPODifferentiableCompressionReport",
    "FirstDegreeMPO",
    "CompiledMPOExp",
    "MPOBasis",
    "exp_mpo",
]

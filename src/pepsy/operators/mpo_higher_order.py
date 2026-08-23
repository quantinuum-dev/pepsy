"""Public facade for the paper-style higher-order MPO construction.

This module names the SciPost higher-order family explicitly.  It owns the
history-aware construction of ``exp(step * H)`` on a finite chain; it does not
own spatial MPO cluster residuals or ordered multi-factor products. The
semantic history implementation remains in
:mod:`pepsy.operators.mpo_semantic`, while
coefficient bases and compiled evaluators live in
:mod:`pepsy.operators.mpo_basis`.

Use :mod:`pepsy.operators.mpo_product` for connected interval/graph cluster
expansions and their joint local ``exp(A) @ exp(B) @ ...`` construction.
The historical :mod:`pepsy.operators.mpo_cluster` import remains a
compatibility facade.
"""

from .mpo_semantic import (
    FirstDegreeMPO,
    MPODifferentiableCompressionReport,
    MPOLocalOperatorTerm,
    MPONumericalCompressionReport,
    MPOCompressionReport,
    MPOLevel,
    MPOLevelToken,
    MPOParameter,
    MPOProductTerm,
)
from .mpo_basis import CompiledMPOExp, MPOBasis, exp_mpo
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

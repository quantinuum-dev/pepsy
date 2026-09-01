"""Compatibility facade for connected MPO cluster products.

The implementation lives in :mod:`pepsy.operators.mpo_product`.  This
module remains importable because ``mpo_cluster`` was the original public
owner of the interval and graph cluster API.  New code should use the
explicit product names when constructing ordered ``exp(A) @ exp(B) @ ...``
expansions.
"""

from .mpo_product import (
    ClusterBasisExpansion,
    ClusterExpBasis,
    ClusterExpansionBasis,
    CompiledMPOClusterExp,
    CompiledMPOClusterProduct,
    MPOClusterBasisExpansion,
    MPOClusterExpansion,
    MPOClusterExpansionReport,
    MPOClusterFactor,
    MPOClusterProductExpansion,
    MPOGraphClusterProductExpansion,
    MPOGraphClusterBasisExpansion,
    exp_mpo_cluster,
    _graph_lattice_for_basis,  # noqa: F401 - compatibility helper
    _graph_lattice_from_input,  # noqa: F401 - compatibility helper
)

__all__ = [
    "MPOClusterFactor",
    "MPOClusterExpansionReport",
    "MPOClusterProductExpansion",
    "CompiledMPOClusterProduct",
    "MPOGraphClusterProductExpansion",
    "MPOClusterBasisExpansion",
    "MPOGraphClusterBasisExpansion",
    "CompiledMPOClusterExp",
    "ClusterBasisExpansion",
    "ClusterExpansionBasis",
    "ClusterExpBasis",
    "MPOClusterExpansion",
    "exp_mpo_cluster",
]

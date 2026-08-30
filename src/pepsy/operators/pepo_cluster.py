"""Public facade for PEPO connected-cluster constructions.

The PEPO family covers square-lattice and graph geometry, fixed-channel
Pauli/autodiff bases, dense connected-cluster plans, and joint ordered local
products.  In particular, ``exp(A) @ exp(B) @ exp(C)`` is assembled from local
cluster targets and connected residuals in one PEPO topology; separately
materialized full-lattice factor PEPOs are not the primary algorithm.

The dense/geometry implementation lives in :mod:`pepsy.operators.pepo_dense`,
active blocks in :mod:`pepsy.operators.pepo_active`, fixed channels in
:mod:`pepsy.operators.pepo_basis`, and joint products in
:mod:`pepsy.operators.pepo_product`. This facade makes the intended public
owners explicit without breaking the legacy ``operators.cluster`` path.
"""

from .pepo_dense import (
    ClusterExpansionPlan,
    ClusterExpansionReport,
    ClusterInternalSymmetry,
    ClusterLattice,
    ClusterModelAdapter,
    ConnectedClusterShape,
    GraphClusterExpansionPlan,
    GraphConnectedClusterShape,
    adapt_cluster_model,
    build_cluster_expansion_pepo,
    build_graph_cluster_expansion_pepo,
    build_itf_cluster_expansion_pepo,
    build_model_cluster_expansion_pepo,
    build_real_time_cluster_expansion_pepo,
    compose_cluster_expansion_pepo,
    compose_pepo_layers,
    generate_connected_cluster_shapes,
)
from .pepo_active import ActivePEPOBlocks, GraphActivePEPOBlocks
from .pepo_basis import CompiledPEPOExp, PauliPEPOBasis, PauliPEPOTerm
from .pepo_product import (
    CompiledPEPOClusterProduct,
    PEPOClusterFactor,
    PEPOClusterProductExpansion,
)

__all__ = [
    "ActivePEPOBlocks",
    "GraphActivePEPOBlocks",
    "GraphClusterExpansionPlan",
    "ClusterInternalSymmetry",
    "ClusterLattice",
    "ConnectedClusterShape",
    "GraphConnectedClusterShape",
    "ClusterExpansionReport",
    "ClusterExpansionPlan",
    "ClusterModelAdapter",
    "adapt_cluster_model",
    "PauliPEPOTerm",
    "PauliPEPOBasis",
    "CompiledPEPOExp",
    "PEPOClusterFactor",
    "PEPOClusterProductExpansion",
    "CompiledPEPOClusterProduct",
    "compose_pepo_layers",
    "compose_cluster_expansion_pepo",
    "generate_connected_cluster_shapes",
    "build_graph_cluster_expansion_pepo",
    "build_cluster_expansion_pepo",
    "build_model_cluster_expansion_pepo",
    "build_itf_cluster_expansion_pepo",
    "build_real_time_cluster_expansion_pepo",
]

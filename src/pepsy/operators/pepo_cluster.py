"""Public facade for PEPO connected-cluster constructions.

The PEPO family covers square-lattice and graph geometry, fixed-channel
Pauli/autodiff bases, dense connected-cluster plans, and joint ordered local
products.  In particular, ``exp(A) @ exp(B) @ exp(C)`` is assembled from local
cluster targets and connected residuals in one PEPO topology; separately
materialized full-lattice factor PEPOs are not the primary algorithm.

The geometry, active-block, and fixed-channel implementation is still in
:mod:`pepsy.operators.cluster`, while the joint product implementation now
lives in :mod:`pepsy.operators.pepo_product`. This facade makes the intended
public owner explicit without breaking the legacy ``operators.cluster`` import
path.
"""

from .cluster import (
    ActivePEPOBlocks,
    ClusterExpansionPlan,
    ClusterExpansionReport,
    ClusterInternalSymmetry,
    ClusterLattice,
    ClusterModelAdapter,
    CompiledPEPOExp,
    ConnectedClusterShape,
    GraphActivePEPOBlocks,
    GraphClusterExpansionPlan,
    GraphConnectedClusterShape,
    PauliPEPOBasis,
    PauliPEPOTerm,
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

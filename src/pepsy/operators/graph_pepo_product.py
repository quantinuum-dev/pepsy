"""Ordered, nonuniform interaction-graph products in a graph PEPO container."""

import autoray as ar

from .cluster_plan import ClusterPlan
from .mpo_product import MPOClusterProductExpansion
from .pepo_dense import (
    GraphActivePEPOBlocks,
    _SectorAllocator,
    _add_graph_tree_factor_blocks,
    _graph_tree_factorize_operator,
    _validate_shape,
)


class GraphPEPOClusterProductExpansion:
    """Use the same located factors and cluster plan as the MPO engine.

    Factors are in algebraic order: ``[A, B]`` means ``exp(step*A) @
    exp(step*B)``. Each may contain different couplings/operators on arbitrary
    supports. The graph PEPO materializer currently uses NumPy tree SVDs;
    scalar ``trace_exp`` and local ``residuals`` preserve backend gradients.
    With a rank cap, the materialized operator can differ from those scalars.
    """

    def __init__(self, plan, factors, *, phys_dim=None, max_tree_rank=None):
        if not isinstance(plan, ClusterPlan):
            raise TypeError("plan must be a ClusterPlan.")
        self.cluster_plan = plan
        self.cluster_size = plan.cluster_size
        self.max_tree_rank = (
            None if max_tree_rank is None else _validate_shape(max_tree_rank, "max_tree_rank")
        )
        self._local = MPOClusterProductExpansion.from_plan(
            plan,
            factors,
            phys_dim=phys_dim,
            cutoff=0.0,
        )
        self.factors = self._local.factors

    @classmethod
    def from_plan(cls, plan, factors, **kwargs):
        """Compile located factors; term sites index ``plan.sites`` order."""
        return cls(plan, factors, **kwargs)

    def compile_exp(self):
        return self

    @property
    def cache_info(self):
        return {**self._local.cache_info, "representation": "graph-pepo"}

    def residuals(self, step=1.0, parameters=None, *, coefficients=None):
        return self._local.residuals(step, parameters=parameters, coefficients=coefficients)

    def trace_exp(
        self, step=1.0, parameters=None, *, coefficients=None, normalized=False, state_budget=100000
    ):
        """Trace the complete partition expansion without graph PEPO compression."""
        return self._local.trace_exp(
            step,
            parameters,
            coefficients=coefficients,
            normalized=normalized,
            state_budget=state_budget,
        )

    def exp(
        self,
        step=1.0,
        parameters=None,
        *,
        coefficients=None,
        materialize=False,
        return_report=False,
    ):
        """Return active graph blocks, or a Quimb graph tensor network explicitly."""
        residuals = self.residuals(step, parameters, coefficients=coefficients)
        if any(
            ar.infer_backend(value) not in {"numpy", "builtins"} for value in residuals.values()
        ):
            raise TypeError(
                "graph PEPO materialization currently requires NumPy values; "
                "use residuals/trace_exp for differentiable backend values."
            )
        plan = self.cluster_plan
        lattice = plan.lattice
        dimension = self._local.phys_dim
        directions = {
            site: tuple(i for i, edge in enumerate(lattice.edges) if site in edge)
            for site in plan.sites
        }
        blocks = {
            site: {(0,) * len(directions[site]): residuals[(i,)]}
            for i, site in enumerate(plan.sites)
        }
        allocator = _SectorAllocator()
        diagnostics = []
        for cluster, shape in zip(plan.index_clusters, plan.graph_shapes):
            if len(cluster) == 1:
                continue
            (tensors, parent, parent_edge, _children, ranks, rank, error, norm) = (
                _graph_tree_factorize_operator(
                    residuals[cluster], shape.edges, len(cluster), dimension, self.max_tree_rank
                )
            )
            if tensors is None:
                continue
            sectors = {parent_edge[site]: allocator.allocate(r) for site, r in ranks.items()}
            _add_graph_tree_factor_blocks(
                blocks,
                directions,
                shape.sites,
                shape.edges,
                tensors,
                parent,
                parent_edge,
                sectors,
                dimension,
            )
            diagnostics.append((cluster, rank, error, norm))
        active = GraphActivePEPOBlocks(
            sites=plan.sites,
            edges=lattice.edges,
            bond_dim=allocator.next_sector,
            physical_dim=dimension,
            site_directions=directions,
            blocks=blocks,
        )
        result = active.to_tensor_network() if materialize else active
        report = dict(
            cluster_size=self.cluster_size,
            cluster_counts=plan.counts,
            factorization_residuals=tuple(diagnostics),
            max_tree_rank=self.max_tree_rank,
        )
        return (result, report) if return_report else result

    __call__ = exp
    evaluate = exp

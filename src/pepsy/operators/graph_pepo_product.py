"""Ordered, nonuniform interaction-graph products in a graph PEPO container."""

import autoray as ar

from .cluster_plan import ClusterPlan
from ._cluster_factorization import normalize_factorization
from .mpo_product import MPOClusterProductExpansion
from .pepo_dense import (
    GraphActivePEPOBlocks,
    _SectorAllocator,
    _add_graph_tree_factor_blocks,
    _graph_tree_factorize_operator,
    _tree_factorize_operator_backend,
    _validate_shape,
)


class GraphPEPOClusterProductExpansion:
    """Use the same located factors and cluster plan as the MPO engine.

    Factors are in algebraic order: ``[A, B]`` means ``exp(step*A) @
    exp(step*B)``. Each may contain different couplings/operators on arbitrary
    supports. ``factorization="fixed"`` uses backend-native exact splits,
    including at zero coefficients, with no numerical rank decisions. Auto
    also uses fixed splits for uncapped tensor backends; NumPy keeps its SVD
    compression, and capped tensor backends use a static-rank SVD.
    ``trace_exp`` constructs and traces that operator. ``partition_trace_exp``
    retains the separate scalar closure before numerical rank truncation.
    """

    def __init__(self, plan, factors, *, phys_dim=None, max_tree_rank=None,
                 factorization="auto", spatial_reuse=True):
        if not isinstance(plan, ClusterPlan):
            raise TypeError("plan must be a ClusterPlan.")
        self.cluster_plan = plan
        self.cluster_size = plan.cluster_size
        self.max_tree_rank = (
            None if max_tree_rank is None else _validate_shape(max_tree_rank, "max_tree_rank")
        )
        self.factorization = normalize_factorization(factorization)
        if self.factorization == "fixed" and self.max_tree_rank is not None:
            raise ValueError("factorization='fixed' requires max_tree_rank=None.")
        self._local = MPOClusterProductExpansion.from_plan(
            plan,
            factors,
            phys_dim=phys_dim,
            cutoff=0.0,
            spatial_reuse=spatial_reuse,
        )
        self.factors = self._local.factors

    @classmethod
    def from_plan(cls, plan, factors, **kwargs):
        """Compile located factors; term sites index ``plan.sites`` order."""
        return cls(plan, factors, **kwargs)

    def compile_exp(self):
        return self

    def prepare_compression(self, step=1.0, parameters=None, *, coefficients=None,
                            max_tree_rank):
        """Choose frozen tree subspaces at an explicit reference evaluation.

        Run outside autodiff/JIT. Replay differentiates the compressed
        operator with these fixed subspaces, not the subspace selection.
        """
        from .cluster_compression import ClusterCompressionPlan

        if self.max_tree_rank is not None:
            raise ValueError("compression supplies its own ranks; use max_tree_rank=None on the builder.")
        return ClusterCompressionPlan.prepare(
            self.cluster_plan, self.residuals(step, parameters, coefficients=coefficients),
            self._local.phys_dim, max_tree_rank)

    @property
    def cache_info(self):
        return {**self._local.cache_info, "representation": "graph-pepo",
                "factorization": self.factorization}

    def residuals(self, step=1.0, parameters=None, *, coefficients=None):
        return self._local.residuals(step, parameters=parameters, coefficients=coefficients)

    def trace_exp(
        self, step=1.0, parameters=None, *, coefficients=None, normalized=False, state_budget=100000,
        compression=None, contract_opts=None,
    ):
        """Construct the graph PEPO and contract its physical trace."""
        from .pepo_trace import _trace_options, trace_pepo

        materialize = contract_opts is not None
        trace_options = _trace_options(state_budget, contract_opts, materialized=materialize)
        pepo = self.exp(step, parameters, coefficients=coefficients, compression=compression,
                        materialize=materialize)
        return trace_pepo(pepo, normalized=normalized, **trace_options)

    def partition_trace_exp(
        self, step=1.0, parameters=None, *, coefficients=None, normalized=False, state_budget=100000
    ):
        """Explicit scalar partition closure, independent of PEPO compression."""
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
        compression=None,
    ):
        """Return active graph blocks, or a Quimb graph tensor network explicitly."""
        if compression is not None:
            from .cluster_compression import ClusterCompressionPlan

            if not isinstance(compression, ClusterCompressionPlan):
                raise TypeError("compression must be a ClusterCompressionPlan.")
            compression.validate(self.cluster_plan, self._local.phys_dim)
            if self.max_tree_rank is not None:
                raise ValueError("compression supplies its own ranks; use max_tree_rank=None on the builder.")
        projections = {} if compression is None else dict(compression.projectors)
        from ._cluster_stats import MaterializationStats

        stats = MaterializationStats() if return_report else None
        residuals = self._local.residuals(
            step, parameters=parameters, coefficients=coefficients, _stats=stats)
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
            residual = residuals[cluster]
            if compression is not None:
                tensors, parent, parent_edge, _children, ranks = _tree_factorize_operator_backend(
                    residual, shape.edges, len(cluster), dimension, graph=True,
                    projectors=dict(projections[cluster]))
                rank, error, norm = max(ranks.values(), default=0), None, None
            elif self.factorization == "auto" and ar.infer_backend(residual) in {"numpy", "builtins"}:
                tensors, parent, parent_edge, _children, ranks, rank, error, norm = _graph_tree_factorize_operator(
                    residuals[cluster], shape.edges, len(cluster), dimension, self.max_tree_rank
                )
            else:
                tensors, parent, parent_edge, _children, ranks = _tree_factorize_operator_backend(
                    residual, shape.edges, len(cluster), dimension, self.max_tree_rank,
                    factorization="fixed" if self.max_tree_rank is None else self.factorization,
                    graph=True,
                )
                rank = max(ranks.values(), default=0)
                # Numerical reconstruction diagnostics are not evaluated on
                # backend graphs; None does not imply zero truncation error.
                error, norm = None, None
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
            layout="graph",
            cluster_size=self.cluster_size,
            cluster_counts=plan.counts,
            factorization_residuals=tuple(diagnostics),
            max_tree_rank=self.max_tree_rank,
            factorization=self.factorization,
            compression=None if compression is None else {
                "method": "frozen-projectors", "max_tree_rank": compression.max_tree_rank,
                "reference_errors": compression.reference_errors,
                "ranks": compression.ranks, "rank_selection_differentiated": False,
            },
        )
        if stats is not None:
            report["materialization"] = stats.report(active)
        return (result, report) if return_report else result

    __call__ = exp
    evaluate = exp

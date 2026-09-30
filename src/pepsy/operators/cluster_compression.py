"""Explicit reference subspaces for differentiable local PEPO compression."""

from dataclasses import dataclass

import autoray as ar
import numpy as np


@dataclass(frozen=True)
class ClusterCompressionPlan:
    """Frozen, reference-dependent tree subspaces; ranks are not differentiated.

    Construct with a PEPO builder's ``prepare_compression`` outside the
    objective. Reuse with ``exp(..., compression=plan)``. The objective then
    differentiates the projected expansion, including at zero coefficients.
    Refresh explicitly when its approximation is no longer adequate.
    """

    signature: tuple
    projectors: tuple
    max_tree_rank: int
    reference_errors: tuple

    @staticmethod
    def _signature(plan, dimension):
        return (plan.sites, plan.index_edges, plan.index_clusters, dimension)

    def validate(self, plan, dimension):
        if self.signature != self._signature(plan, dimension):
            raise ValueError("compression plan belongs to different cluster geometry or physical dimension.")

    @property
    def ranks(self):
        return tuple((cluster, tuple((site, matrix.shape[1]) for site, matrix in matrices))
                     for cluster, matrices in self.projectors)

    @classmethod
    def prepare(cls, plan, residuals, dimension, max_tree_rank):
        from .pepo_dense import (
            _tree_factorize_operator_backend, _validate_shape,
            _contract_tree_factor_coefficients, _operator_tensor,
        )

        if isinstance(max_tree_rank, (bool, np.bool_)):
            raise TypeError("max_tree_rank must be a positive integer.")
        cap = _validate_shape(max_tree_rank, "max_tree_rank")
        projectors, errors = [], []
        for cluster, shape in zip(plan.index_clusters, plan.graph_shapes):
            if len(cluster) == 1:
                continue
            # Preparation is an explicit host computation, never a hidden
            # detach during exp(). Callers must keep it outside JIT/grad.
            residual = np.array(ar.to_numpy(residuals[cluster]), copy=True)
            tensors, parent, _, children, ranks = _tree_factorize_operator_backend(
                residual, shape.edges, len(cluster), dimension, cap, graph=True)
            matrices = []
            for site, rank in ranks.items():
                matrix = np.array(tensors[site][1].reshape(-1, rank), copy=True)
                matrix.setflags(write=False)
                matrices.append((site, matrix))
            projectors.append((cluster, tuple(matrices)))
            norm = np.linalg.norm(residual)
            reconstructed = _contract_tree_factor_coefficients(tensors, parent, children, len(cluster))
            error = np.linalg.norm(_operator_tensor(residual, len(cluster), dimension) - reconstructed)
            errors.append((cluster, float(error), float(norm)))
        return cls(cls._signature(plan, dimension), tuple(projectors), cap, tuple(errors))

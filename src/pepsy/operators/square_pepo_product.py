"""Shared finite plans adapted to the existing square Pauli PEPO builder."""

from itertools import product

import numpy as np

from .cluster_plan import ClusterPlan
from .mpo_product import MPOClusterProductExpansion
from .mpo_semantic import MPOProductTerm


class _UnsupportedSquare(ValueError):
    pass


def _square_plan_geometry(plan):
    """Require the same sites and connected family as the square backend."""
    from .pepo_dense import ClusterLattice

    if not isinstance(plan, ClusterPlan):
        raise TypeError("plan must be a ClusterPlan.")
    if plan.shape is None or len(plan.shape) != 2:
        raise _UnsupportedSquare(
            "square layout requires an explicit or inferred two-dimensional shape."
        )
    coordinates = tuple(product(*(range(n) for n in plan.shape)))
    coordinate_labels = all(
        isinstance(site, tuple)
        and len(site) == 2
        and all(isinstance(value, (int, np.integer)) for value in site)
        for site in plan.sites
    )
    if coordinate_labels and plan.sites != coordinates:
        raise _UnsupportedSquare("square coordinate sites must match shape in row-major order.")
    if any(periodic and size == 1 for periodic, size in zip(plan.cyclic, plan.shape)):
        raise _UnsupportedSquare("a periodic square axis must contain at least two sites.")
    lattice = ClusterLattice.square(*plan.shape, cyclic=plan.cyclic)
    positions = {site: i for i, site in enumerate(coordinates)}
    edges = {frozenset((positions[a], positions[b])) for a, b in lattice.edges}
    if edges != set(map(frozenset, plan.index_edges)):
        raise _UnsupportedSquare(
            "square layout requires the complete nearest-neighbor graph for its shape and boundaries."
        )
    if plan.cluster_size > 9:
        raise _UnsupportedSquare(
            "the square Pauli PEPO backend supports cluster_size through nine."
        )
    return coordinates


def _square_terms(plan, factors, phys_dim=None):
    from .pepo_basis import PauliPEPOTerm

    coordinates = _square_plan_geometry(plan)
    if phys_dim is not None and (
        isinstance(phys_dim, bool) or not isinstance(phys_dim, (int, np.integer)) or phys_dim != 2
    ):
        raise _UnsupportedSquare("square Pauli layout requires phys_dim=2.")
    paulis = (
        ("I", np.eye(2)),
        ("X", np.array([[0.0, 1.0], [1.0, 0.0]])),
        ("Y", np.array([[0.0, -1j], [1j, 0.0]])),
        ("Z", np.diag([1.0, -1.0])),
    )
    directions = (("u", (1, 0)), ("r", (0, 1)), ("d", (-1, 0)), ("l", (0, -1)))
    converted = []
    for factor in factors:
        terms = []
        for term in factor.terms:
            if not isinstance(term, MPOProductTerm) or len(term.sites) > 2:
                raise _UnsupportedSquare(
                    "square layout currently requires one- or two-site Pauli product terms."
                )
            if (
                term.string_operators is not None
                or term.charge is not None
                or term.braiding.fermionic
                or any(term.parities)
            ):
                raise _UnsupportedSquare(
                    "square layout does not convert strings or charged/fermionic term metadata."
                )
            if any(site < 0 or site >= len(coordinates) for site in term.sites):
                raise ValueError("a cluster term site is outside the plan.")
            if len(term.sites) > plan.cluster_size:
                raise ValueError("term support exceeds the plan cluster_size.")
            word = []
            for operator in term.operators:
                label = next(
                    (
                        label
                        for label, matrix in paulis
                        if isinstance(operator, np.ndarray) and np.array_equal(operator, matrix)
                    ),
                    None,
                )
                if label is None:
                    raise _UnsupportedSquare(
                        "square layout requires fixed NumPy I/X/Y/Z matrices; put scalar amplitudes in coefficients."
                    )
                word.append(label)
            sites = tuple(coordinates[i] for i in term.sites)
            if len(sites) == 1:
                terms.append(
                    PauliPEPOTerm("onsite", "".join(word), term.coefficient, where=sites[0])
                )
                continue
            source, target = sites
            direction = None
            for name, delta in directions:
                neighbor = tuple(source[a] + delta[a] for a in range(2))
                if any(
                    not plan.cyclic[a] and not 0 <= neighbor[a] < plan.shape[a] for a in range(2)
                ):
                    continue
                neighbor = tuple(neighbor[a] % plan.shape[a] for a in range(2))
                if neighbor == target:
                    direction = name
                    break
            if direction is None:
                raise _UnsupportedSquare("square layout requires nearest-neighbor term supports.")
            # Length-two periodic axes offer two routes to the same endpoint.
            # Choose one virtual route per supplied term; never duplicate it.
            terms.append(
                PauliPEPOTerm(
                    "edge", "".join(word), term.coefficient, where=sites, direction=direction
                )
            )
        converted.append(tuple(terms))
    return tuple(converted)


def from_interaction_plan(plan, factors, *, layout="auto", **kwargs):
    """Select square construction only when its exact input contract matches."""
    from .graph_pepo_product import GraphPEPOClusterProductExpansion

    if not isinstance(plan, ClusterPlan):
        raise TypeError("plan must be a ClusterPlan.")
    if layout not in ("auto", "square", "graph"):
        raise ValueError("layout must be 'auto', 'square', or 'graph'.")
    factors = tuple(MPOClusterProductExpansion._normalize_factor(factor) for factor in factors)
    if not factors:
        raise ValueError("at least one PEPO cluster factor is required.")
    if layout != "graph":
        try:
            return SquarePEPOClusterProductExpansion(plan, factors, **kwargs)
        except _UnsupportedSquare:
            if layout == "square":
                raise
    return GraphPEPOClusterProductExpansion.from_plan(plan, factors, **kwargs)


class SquarePEPOClusterProductExpansion:
    """Located Pauli products with shared planning and square PEPO output.

    Term slots, coefficients and factor order are unchanged by conversion.
    ``exp`` defaults to active square blocks; ``materialize=True`` returns a
    Quimb PEPO. ``factorization='fixed'`` supports differentiable construction
    without numerical rank selection. Traces precede materialization and
    remain independent of rank caps. Numerical values are never cached.
    """

    def __init__(
        self,
        plan,
        factors,
        *,
        phys_dim=None,
        max_tree_rank=None,
        factorization="auto",
        spatial_reuse=True,
    ):
        from .pepo_basis import PauliPEPOBasis
        from .pepo_product import PEPOClusterFactor, PEPOClusterProductExpansion

        factors = tuple(MPOClusterProductExpansion._normalize_factor(f) for f in factors)
        if not factors:
            raise ValueError("at least one PEPO cluster factor is required.")
        terms = _square_terms(plan, factors, phys_dim)
        self.cluster_plan = plan
        self.cluster_size = plan.cluster_size
        self.factors = factors
        self.lx, self.ly = plan.shape
        self.cyclic = plan.cyclic
        self.max_tree_rank = max_tree_rank
        self._square = PEPOClusterProductExpansion(
            tuple(
                PEPOClusterFactor(
                    PauliPEPOBasis(
                        self.lx,
                        self.ly,
                        local_terms,
                        cluster_size=plan.cluster_size,
                        cyclic=plan.cyclic,
                        max_tree_rank=max_tree_rank,
                        factorization=factorization,
                        spatial_reuse=spatial_reuse,
                        cluster_plan=plan,
                    ),
                    factor.coefficient,
                )
                for factor, local_terms in zip(factors, terms)
            )
        )
        self._local = None

    @classmethod
    def from_plan(cls, plan, factors, **kwargs):
        return cls(plan, factors, **kwargs)

    def compile_exp(self):
        self._square.compile_exp()
        return self

    @property
    def cluster_inventory(self):
        return self.cluster_plan.graph_shapes

    @property
    def cache_info(self):
        return {
            **self._square.cache_info,
            "representation": "square-pepo",
            "local_residuals_compiled": self._local is not None,
        }

    def residuals(self, step=1.0, parameters=None, *, coefficients=None):
        if self._local is None:
            self._local = MPOClusterProductExpansion.from_plan(
                self.cluster_plan,
                self.factors,
                cutoff=0.0,
            )
        return self._local.residuals(step, parameters=parameters, coefficients=coefficients)

    def trace_exp(
        self, step=1.0, parameters=None, *, coefficients=None, normalized=False, state_budget=100000
    ):
        return self._square.trace_exp(
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
        result = self._square.exp(
            step, parameters, coefficients=coefficients, materialize=materialize
        )
        report = dict(
            layout="square",
            cluster_size=self.cluster_size,
            cluster_counts=self.cluster_plan.counts,
            max_tree_rank=self.max_tree_rank,
            factorization=self._square.cache_info["factorization"],
        )
        return (result, report) if return_report else result

    __call__ = exp
    evaluate = exp

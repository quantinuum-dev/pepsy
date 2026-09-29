"""Joint ordered PEPO cluster products.

This module owns the product-specific part of the PEPO cluster API.  Given
factors ``A``, ``B``, and ``C``, it forms the ordered local targets
``exp(A_S) @ exp(B_S) @ exp(C_S)`` on each connected cluster ``S`` and asks the
first factor's PEPO basis to assemble the resulting connected residuals into
one active topology.

The module intentionally does not materialize a full-lattice PEPO for each
factor.  Ordinary Quimb multiplication remains an execution/interoperability
operation, not the algorithm used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import autoray as ar

from .mpo_automaton import _as_backend, _backend_reference

if TYPE_CHECKING:
    from .pepo_basis import PauliPEPOBasis

__all__ = [
    "PEPOClusterFactor",
    "PEPOClusterProductExpansion",
    "CompiledPEPOClusterProduct",
]


def _pauli_pepo_basis_type():
    """Resolve the basis type lazily to keep the compatibility facade acyclic."""
    from .pepo_basis import PauliPEPOBasis

    return PauliPEPOBasis


def _resolve_pepo_factor_value(value, parameters):
    """Resolve a product-level scalar without caching backend values."""
    if hasattr(value, "resolve"):
        return value.resolve(parameters)
    if callable(value):
        if parameters is None:
            raise KeyError("callable PEPO factor coefficients require parameters.")
        return value(parameters)
    return value


def _as_backend_dtype(value, *, like):
    """Convert a scalar to ``like``'s backend without losing its dtype."""
    if ar.infer_backend(like) == "torch" and ar.infer_backend(value) == "torch":
        # Native tensor arithmetic keeps dtype promotion inside the captured graph.
        return value.to(device=like.device) + like.new_zeros(())
    target_dtype = ar.do("result_type", like, value)
    value = _as_backend(value, like=like, dtype=target_dtype)
    if target_dtype is not None and getattr(value, "dtype", None) != target_dtype:
        if hasattr(value, "astype"):
            value = ar.do("astype", value, target_dtype)
        else:
            value = ar.do("array", value, like=like, dtype=target_dtype)
    return value


@dataclass(frozen=True)
class PEPOClusterFactor:
    """One local Hamiltonian factor in a joint ordered cluster expansion."""

    basis: PauliPEPOBasis
    coefficient: object = 1.0

    def __post_init__(self):
        if not isinstance(self.basis, _pauli_pepo_basis_type()):
            raise TypeError("PEPOClusterFactor.basis must be a PauliPEPOBasis.")


class CompiledPEPOClusterProduct:
    """Reusable joint ordered PEPO cluster-expansion evaluator."""

    def __init__(self, expansion):
        if not isinstance(expansion, PEPOClusterProductExpansion):
            raise TypeError(
                "expansion must be a PEPOClusterProductExpansion."
            )
        self.expansion = expansion
        self.cluster_size = expansion.cluster_size
        localized = any(factor.basis.inhomogeneous for factor in expansion.factors)
        for factor in expansion.factors:
            factor.basis._prepare_exp_plan(localized=localized)
        expansion.factors[0].basis._prepare_spatial_plans(
            tuple(factor.basis for factor in expansion.factors), localized=localized)

    @property
    def cache_info(self):
        """Return topology and evaluation diagnostics."""
        return self.expansion.cache_info

    @property
    def cluster_inventory(self):
        """Return square-lattice shape counts for the shared spatial cutoff."""
        return self.expansion.cluster_inventory

    def exp(
        self,
        step,
        parameters=None,
        *,
        coefficients=None,
        materialize=True,
        compress=False,
        **compress_opts,
    ):
        """Evaluate the ordered product ``exp(A) exp(B) ...``.

        Set ``materialize=False`` to return the sparse active blocks before
        allocating Quimb PEPO site tensors.
        """
        return self.expansion.exp(
            step,
            parameters,
            coefficients=coefficients,
            materialize=materialize,
            compress=compress,
            **compress_opts,
        )

    def trace_exp(self, step, parameters=None, *, coefficients=None,
                  normalized=False, state_budget=100000):
        """Evaluate the complete selected-order trace without building a PEPO."""
        return self.expansion.trace_exp(
            step, parameters, coefficients=coefficients, normalized=normalized,
            state_budget=state_budget,
        )

    evaluate = exp
    __call__ = exp


class PEPOClusterProductExpansion:
    """Build one joint Guppy-style PEPO cluster expansion.

    ``factors`` are specified in algebraic order, so ``(A, B, C)`` means
    ``exp(A) @ exp(B) @ exp(C)``. For every connected spatial cluster ``S``,
    the local dense target is formed as
    ``exp(A_S) @ exp(B_S) @ exp(C_S)``. Lower connected partitions are then
    subtracted and the resulting residual channels are assembled once into
    one PEPO. No full-lattice PEPO is built for an individual factor.

    All factors must use the same lattice, symmetry policy, and cluster order.
    The order is a joint local-cluster cutoff, not a factor label: use one
    order-2 expansion for ``A, B, C`` rather than multiplying an order-2 PEPO
    by an order-3 PEPO.
    """

    def __init__(self, factors):
        try:
            factors = tuple(self._normalize_factor(factor) for factor in factors)
        except TypeError as exc:
            raise TypeError(
                "factors must be an iterable of PEPO cluster factors."
            ) from exc
        if not factors:
            raise ValueError("at least one PEPO cluster factor is required.")
        reference = factors[0].basis
        geometry = (reference.lx, reference.ly, reference.cyclic)
        for factor in factors[1:]:
            basis = factor.basis
            if (basis.lx, basis.ly, basis.cyclic) != geometry:
                raise ValueError(
                    "all PEPO cluster factors must have matching lattice "
                    "shape and periodicity."
                )
            if basis.order != reference.order:
                raise ValueError(
                    "all PEPO cluster factors must use the same joint order."
                )
            if basis.symmetry != reference.symmetry:
                raise ValueError(
                    "all PEPO cluster factors must use the same symmetry policy."
                )
            if basis.spatial_reuse != reference.spatial_reuse:
                raise ValueError("all PEPO cluster factors must use the same spatial_reuse policy.")
            if basis.factorization != reference.factorization:
                raise ValueError("all PEPO cluster factors must use the same factorization policy.")
            if basis.max_tree_rank != reference.max_tree_rank:
                raise ValueError(
                    "all PEPO cluster factors must use the same max_tree_rank."
                )
        self.factors = factors
        self.lx, self.ly, self.cyclic = geometry
        self.cluster_size = reference.cluster_size
        self._build_count = 0
        self._compiled_exp = None
        self._trace_plans = {}

    @classmethod
    def from_plan(cls, plan, factors, **kwargs):
        """Compile located graph factors in algebraic order.

        Returns GraphPEPOClusterProductExpansion, using MPOClusterFactor's
        term schema and graph tensor-network materialization.
        """
        from .graph_pepo_product import GraphPEPOClusterProductExpansion

        return GraphPEPOClusterProductExpansion.from_plan(plan, factors, **kwargs)

    @staticmethod
    def _normalize_factor(factor):
        if isinstance(factor, PEPOClusterFactor):
            return factor
        if isinstance(factor, _pauli_pepo_basis_type()):
            return PEPOClusterFactor(factor)
        if isinstance(factor, Mapping):
            basis = factor.get("basis")
            if basis is None:
                raise ValueError("PEPO factor mappings require a 'basis'.")
            return PEPOClusterFactor(basis, factor.get("coefficient", 1.0))
        if isinstance(factor, (tuple, list)) and len(factor) == 2:
            return PEPOClusterFactor(factor[0], factor[1])
        raise TypeError(
            "PEPO factors must be PauliPEPOBasis values, PEPOClusterFactor "
            "values, mappings, or (basis, coefficient) pairs."
        )

    @classmethod
    def from_bases(cls, bases, *, coefficients=None):
        """Construct an ordered product from compiled PEPO bases."""
        bases = tuple(bases)
        if not bases:
            raise ValueError("bases must contain at least one PauliPEPOBasis.")
        if coefficients is None:
            coefficients = (1.0,) * len(bases)
        else:
            coefficients = tuple(coefficients)
            if len(coefficients) != len(bases):
                raise ValueError("coefficients must align with bases.")
        return cls(
            PEPOClusterFactor(basis, coefficient)
            for basis, coefficient in zip(bases, coefficients)
        )

    @property
    def cluster_inventory(self):
        """Return shape counts once for the shared lattice and spatial cutoff.

        Factor count does not multiply the connected spatial cluster inventory.
        """
        return self.factors[0].basis.cluster_inventory

    @property
    def cache_info(self):
        """Return topology-only product diagnostics."""
        return {
            "compiled": True,
            "builds": self._build_count,
            "factor_count": len(self.factors),
            "lattice_shape": (self.lx, self.ly),
            "cyclic": self.cyclic,
            "cluster_size": self.cluster_size,
            "factor_orders": tuple(factor.basis.order for factor in self.factors),
            "factor_cache_info": tuple(factor.basis.cache_info for factor in self.factors),
            "max_tree_rank": self.factors[0].basis.max_tree_rank,
            "factorization": self.factors[0].basis.factorization,
            "joint_cluster_residual": True,
            "compiled_exp": self._compiled_exp is not None,
        }

    def compile_exp(self):
        """Return a reusable ordered product evaluator."""
        if self._compiled_exp is None:
            self._compiled_exp = CompiledPEPOClusterProduct(self)
        return self._compiled_exp

    def _factor_coefficients(self, coefficients, parameters=None):
        if coefficients is not None and parameters is not None:
            raise ValueError("parameters and coefficients are mutually exclusive.")
        if coefficients is None:
            return (None,) * len(self.factors)
        if len(self.factors) == 1:
            return (coefficients,)
        try:
            values = tuple(coefficients)
        except TypeError as exc:
            raise TypeError(
                "coefficients must contain one coefficient vector per factor."
            ) from exc
        if len(values) != len(self.factors):
            raise ValueError(
                "coefficients must contain one vector per PEPO cluster factor."
            )
        return values

    def trace_exp(self, step, parameters=None, *, coefficients=None,
                  normalized=False, state_budget=100000):
        """Trace the complete chosen-order ordered product using scalar clusters.

        Connected scalar residuals and a subset DP include every compatible
        placement. This is independent of PEPO tree-rank and compression
        settings: a rank-capped materialized PEPO may have a different trace.
        """
        factor_coefficients = self._factor_coefficients(coefficients, parameters)
        from ._cluster_trace import compile_trace_plan, evaluate_trace_plan

        reference_basis = self.factors[0].basis
        records_by_size = reference_basis._localized_cluster_records()
        if state_budget not in self._trace_plans:
            clusters = tuple(
                record.site_indices
                for records in records_by_size.values()
                for record in records
            )
            self._trace_plans[state_budget] = compile_trace_plan(
                self.lx * self.ly, clusters, state_budget,
            )
        ordered, plan = self._trace_plans[state_budget]
        localized = []
        for factor, term_coefficients in zip(self.factors, factor_coefficients):
            coefficient = _resolve_pepo_factor_value(factor.coefficient, parameters)
            beta = ar.do("multiply", -step, coefficient)
            values = factor.basis._coefficient_values(
                parameters if term_coefficients is None else None,
                term_coefficients,
            )
            local_reference = _backend_reference((beta, *values))
            beta = _as_backend_dtype(beta, like=local_reference)
            values = tuple(
                _as_backend_dtype(value, like=local_reference) for value in values
            )
            site_components, edge_components = (
                factor.basis._localized_hamiltonian_components(values)
            )
            localized.append((factor.basis, beta, site_components, edge_components))
        backend_reference = _backend_reference(tuple(
            value for _basis, beta, sites, edges in localized
            for value in (beta, sites, edges)
        ))
        localized = tuple(
            (basis, _as_backend(beta, like=backend_reference),
             _as_backend(sites, like=backend_reference),
             _as_backend(edges, like=backend_reference))
            for basis, beta, sites, edges in localized
        )
        defaults = tuple(value is None for value in factor_coefficients)
        bases = tuple(factor.basis for factor in self.factors)
        local_traces = {}
        evaluated = 0
        for records in records_by_size.values():
            if reference_basis.spatial_reuse:
                spatial_plan = reference_basis._localized_spatial_plan(
                    bases, defaults, records,
                )
                sources = tuple(
                    index for index, (source, _axes) in spatial_plan.entries.items()
                    if index == source
                )
            else:
                spatial_plan = None
                sources = tuple(range(len(records)))
            products = reference_basis._localized_ordered_products(
                localized, tuple(records[index] for index in sources),
                like=backend_reference,
            )
            evaluated += len(sources)
            source_traces = {
                source: ar.do("trace", product) / 2 ** len(records[source].sites)
                for source, product in zip(sources, products)
            }
            for index, record in enumerate(records):
                source = index if spatial_plan is None else spatial_plan.entries[index][0]
                local_traces[record.site_indices] = source_traces[source]
        reference_basis._last_spatial_evaluations = evaluated
        return evaluate_trace_plan(
            ordered, plan, local_traces, phys_dim=2, normalized=normalized,
        )

    def exp(
        self,
        step,
        parameters=None,
        *,
        coefficients=None,
        materialize=True,
        compress=False,
        **compress_opts,
    ):
        """Build one PEPO for ``exp(A) @ exp(B) @ ...``.

        The exponentials are multiplied only on each small connected cluster.
        Their connected residuals are combined into a single PEPO topology;
        independent full-lattice factor PEPOs are never materialized. Set
        ``materialize=False`` to return those active blocks directly;
        compression requires a materialized Quimb PEPO.
        """
        factor_coefficients = self._factor_coefficients(coefficients, parameters)
        if compress and self.factors[0].basis.factorization == "fixed":
            raise ValueError("factorization='fixed' requires compress=False; compress the result separately.")
        localized = any(factor.basis.inhomogeneous for factor in self.factors)
        factor_data = []
        factor_sources = []
        for factor, term_coefficients in zip(self.factors, factor_coefficients):
            coefficient = _resolve_pepo_factor_value(
                factor.coefficient,
                parameters,
            )
            factor_beta = ar.do("multiply", -step, coefficient)
            values = factor.basis._coefficient_values(
                parameters if term_coefficients is None else None,
                term_coefficients,
            )
            reference = _backend_reference((factor_beta, *values))
            factor_beta = _as_backend_dtype(factor_beta, like=reference)
            values = tuple(
                _as_backend_dtype(value, like=reference)
                for value in values
            )
            if localized:
                factor_sources.append((factor.basis, factor_beta, values))
            else:
                onsite_components, edge_components = factor.basis._hamiltonian_components(
                    values,
                    factor_beta,
                )
                factor_data.append(
                    (
                        factor.basis,
                        factor_beta,
                        onsite_components,
                        edge_components,
                    )
                )

        if localized:
            active = self.factors[0].basis._build_inhomogeneous_active(
                factor_sources,
                defaults=tuple(value is None for value in factor_coefficients),
            )
        else:
            active = self.factors[0].basis._build_active(
                None,
                None,
                factor_data=factor_data,
            )
        if compress and not materialize:
            raise ValueError("compress=True requires materialize=True.")
        if not materialize:
            self._build_count += 1
            return active
        result = active.to_pepo()
        if compress:
            result.compress(**compress_opts)
        self._build_count += 1
        return result

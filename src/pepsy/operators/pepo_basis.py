"""Fixed-channel Pauli PEPO basis and compiled exponential evaluator.

This module owns the value-independent Pauli-slot topology and the
coefficient-only evaluation boundary. Geometry-specific residual helpers
remain in the legacy cluster implementation for now and are resolved
lazily to avoid coupling the basis to the planner at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import autoray as ar
import numpy as np
from itertools import product

from .mpo_automaton import _as_backend, _backend_reference
from .pepo_active import ActivePEPOBlocks

__all__ = ["PauliPEPOTerm", "CompiledPEPOExp", "PauliPEPOBasis"]

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_BASIS_CACHE = {}
_POSITIVE_DIRECTIONS = frozenset(("u", "r"))
_OPPOSITE_DIRECTION = {"u": "d", "r": "l", "d": "u", "l": "r"}


def _cluster_helper(name):
    """Resolve a planner helper only when a basis operation needs it."""
    from . import cluster

    return getattr(cluster, name)


def _lazy_cluster_proxy(name):
    def proxy(*args, **kwargs):
        return _cluster_helper(name)(*args, **kwargs)

    proxy.__name__ = name
    return proxy


def _add_generic_active_levels(*args, **kwargs):
    return _cluster_helper("_add_generic_active_levels")(*args, **kwargs)

def _add_pair_block_at_site(*args, **kwargs):
    return _cluster_helper("_add_pair_block_at_site")(*args, **kwargs)

def _add_pair_blocks(*args, **kwargs):
    return _cluster_helper("_add_pair_blocks")(*args, **kwargs)

def _add_positive_edge_channels(*args, **kwargs):
    return _cluster_helper("_add_positive_edge_channels")(*args, **kwargs)

def _add_single_direction_blocks(*args, **kwargs):
    return _cluster_helper("_add_single_direction_blocks")(*args, **kwargs)

def _add_tree_factor_blocks_backend(*args, **kwargs):
    return _cluster_helper("_add_tree_factor_blocks_backend")(*args, **kwargs)

def _add_triple_blocks(*args, **kwargs):
    return _cluster_helper("_add_triple_blocks")(*args, **kwargs)

def _all_direction_pairs(*args, **kwargs):
    return _cluster_helper("_all_direction_pairs")(*args, **kwargs)

def _backend_embed_operator(*args, **kwargs):
    return _cluster_helper("_backend_embed_operator")(*args, **kwargs)

def _backend_expm(*args, **kwargs):
    return _cluster_helper("_backend_expm")(*args, **kwargs)

def _backend_operator_product(*args, **kwargs):
    return _cluster_helper("_backend_operator_product")(*args, **kwargs)

def _backend_pauli_basis(*args, **kwargs):
    value = _cluster_helper("_backend_pauli_basis")(*args, **kwargs)
    nsites = args[0] if args else kwargs.get("nsites")
    if nsites is not None:
        _PAULI_BASIS_CACHE[nsites] = value
    return value

def _backend_pauli_expand(*args, **kwargs):
    return _cluster_helper("_backend_pauli_expand")(*args, **kwargs)

def _backend_stack(*args, **kwargs):
    return _cluster_helper("_backend_stack")(*args, **kwargs)

def _backend_swap_two_site(*args, **kwargs):
    return _cluster_helper("_backend_swap_two_site")(*args, **kwargs)

def _cluster_shape_c4_orbit(*args, **kwargs):
    return _cluster_helper("_cluster_shape_c4_orbit")(*args, **kwargs)

def _cluster_shape_embeddings(*args, **kwargs):
    return _cluster_helper("_cluster_shape_embeddings")(*args, **kwargs)

def _complexify_backend(*args, **kwargs):
    return _cluster_helper("_complexify_backend")(*args, **kwargs)

def _contract_active_support_backend(*args, **kwargs):
    return _cluster_helper("_contract_active_support_backend")(*args, **kwargs)

def _initialize_blocks(*args, **kwargs):
    return _cluster_helper("_initialize_blocks")(*args, **kwargs)

def _normalize_pauli_term(*args, **kwargs):
    return _cluster_helper("_normalize_pauli_term")(*args, **kwargs)

def _normalize_paulis(*args, **kwargs):
    return _cluster_helper("_normalize_paulis")(*args, **kwargs)

def _normalize_pauli_support(*args, **kwargs):
    return _cluster_helper("_normalize_pauli_support")(*args, **kwargs)

def _pair_orbits(*args, **kwargs):
    return _cluster_helper("_pair_orbits")(*args, **kwargs)

def _path_orbits(*args, **kwargs):
    return _cluster_helper("_path_orbits")(*args, **kwargs)

def _path_start_sites(*args, **kwargs):
    return _cluster_helper("_path_start_sites")(*args, **kwargs)

def _plaquette_edges(*args, **kwargs):
    return _cluster_helper("_plaquette_edges")(*args, **kwargs)

def _plaquette_starts(*args, **kwargs):
    return _cluster_helper("_plaquette_starts")(*args, **kwargs)

def _rotate_direction_tensor(*args, **kwargs):
    return _cluster_helper("_rotate_direction_tensor")(*args, **kwargs)

def _shape_rotation_map(*args, **kwargs):
    return _cluster_helper("_shape_rotation_map")(*args, **kwargs)

def _site_after(*args, **kwargs):
    return _cluster_helper("_site_after")(*args, **kwargs)

def _site_directions(*args, **kwargs):
    return _cluster_helper("_site_directions")(*args, **kwargs)

def _subset_orbits(*args, **kwargs):
    return _cluster_helper("_subset_orbits")(*args, **kwargs)

def _swap_two_site_operator(*args, **kwargs):
    return _cluster_helper("_swap_two_site_operator")(*args, **kwargs)

def _three_subtrees(*args, **kwargs):
    return _cluster_helper("_three_subtrees")(*args, **kwargs)

def _transform_tree_factorization(*args, **kwargs):
    return _cluster_helper("_transform_tree_factorization")(*args, **kwargs)

def _tree_factorize_operator_backend(*args, **kwargs):
    return _cluster_helper("_tree_factorize_operator_backend")(*args, **kwargs)

def _validate_cyclic(*args, **kwargs):
    return _cluster_helper("_validate_cyclic")(*args, **kwargs)

def _validate_shape(*args, **kwargs):
    return _cluster_helper("_validate_shape")(*args, **kwargs)

def _SectorAllocator(*args, **kwargs):
    return _cluster_helper("_SectorAllocator")(*args, **kwargs)

def generate_connected_cluster_shapes(*args, **kwargs):
    return _cluster_helper("generate_connected_cluster_shapes")(*args, **kwargs)



@dataclass(frozen=True)
class PauliPEPOTerm:
    """One translation-invariant Pauli slot in a square-lattice PEPO basis.

    ``support="onsite"`` contributes the same one-site Pauli operator to
    every lattice site. ``support="edge"`` contributes the same ordered
    two-site Pauli operator to every positive (``u`` and ``r``) lattice edge.
    The scalar ``coefficient`` may be a Python number, a Torch/JAX scalar, or
    a callable accepting the parameter container passed to
    :meth:`PauliPEPOBasis.exp`.
    """

    support: str
    paulis: object
    coefficient: object = 1.0

    def __post_init__(self):
        support = _normalize_pauli_support(self.support)
        labels = _normalize_paulis(self.paulis, support=support)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "paulis", labels)

    @classmethod
    def from_pauli(cls, support, paulis, *, coefficient=1.0):
        """Construct a term from ``"X"`` or ``"ZZ"`` labels."""
        return cls(support, paulis, coefficient)


class CompiledPEPOExp:
    """Cached callable for repeated fixed-topology PEPO exponentials.

    The callable owns no coefficient or autodiff values. It only points to a
    :class:`PauliPEPOBasis` whose lattice, Pauli channels, C4 orbits, and
    active-sector layout were compiled once. Each :meth:`exp` call returns
    fresh :class:`ActivePEPOBlocks` unless ``materialize=True`` is requested.
    """

    def __init__(self, basis):
        if not isinstance(basis, PauliPEPOBasis):
            raise TypeError("basis must be a PauliPEPOBasis.")
        self.basis = basis
        # Compile only value-independent cluster embeddings here. Matrix
        # exponentials and coefficient contractions still happen per call.
        basis._prepare_exp_plan()

    def exp(
        self,
        step,
        parameters=None,
        *,
        coefficients=None,
        materialize=False,
    ):
        """Evaluate ``exp(step * H)`` with fresh backend values.

        ``step=-1j * tau`` is real-time evolution. ``parameters`` resolves
        callable/parameterized slots; ``coefficients`` is a one-dimensional
        batch in the basis term order, and the two inputs are mutually
        exclusive.
        """
        return self.basis.exp(
            step,
            parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    evaluate = exp
    __call__ = exp


class PauliPEPOBasis:
    """Compiled fixed-channel Pauli basis for square-lattice evolution.

    This is the PEPO analogue of :class:`~pepsy.operators.mpo.MPOBasis`.
    The lattice, cluster shapes, Pauli channels, and active-sector topology
    are compiled once. Each evaluation only assembles local cluster
    exponentials and fills those fixed channels with the current coefficient
    and time-step values, so backend scalar graphs are not cached or copied.

    The supported Hamiltonian family is

    ``H(theta) = sum_i h(theta)_i + sum_<ij> v(theta)_{ij}``,

    where ``h`` and ``v`` are linear combinations of the supplied onsite and
    edge Pauli slots. The returned representation is normally kept as
    :class:`ActivePEPOBlocks`; dense Quimb materialization is intended for
    small validation lattices because a fixed Pauli channel basis is much
    larger than an SVD-compressed numerical basis. Orders five through nine
    use the generic connected-shape inventory and a backend-native
    spanning-tree factorization. Loop edges are included when forming the
    local residual, so the local loop correction remains exact even though
    its PEPO channel uses a tree representation of that tensor.
    """

    def __init__(
        self,
        lx,
        ly,
        terms,
        *,
        order=4,
        cyclic=False,
        symmetry=None,
        max_tree_rank=None,
    ):
        self.lx = _validate_shape(lx, "lx")
        self.ly = _validate_shape(ly, "ly")
        self.cyclic = _validate_cyclic(cyclic, self.lx, self.ly)
        if not isinstance(order, Integral):
            raise TypeError("order must be an integer.")
        self.order = int(order)
        if self.order < 1 or self.order > 9:
            raise ValueError("PauliPEPOBasis currently supports orders 1 through 9.")
        if symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")
        if max_tree_rank is not None:
            if not isinstance(max_tree_rank, Integral):
                raise TypeError("max_tree_rank must be an integer or None.")
            if int(max_tree_rank) < 1:
                raise ValueError("max_tree_rank must be >= 1 or None.")
            max_tree_rank = int(max_tree_rank)
        self.symmetry = symmetry
        self.max_tree_rank = max_tree_rank
        self._terms = tuple(_normalize_pauli_term(term) for term in terms)
        if not self._terms:
            raise ValueError("terms must contain at least one Pauli slot.")
        # Static one-hot maps let each evaluation fuse all coefficient slots
        # into onsite and edge Pauli components in two backend contractions.
        # They contain topology only, so they are safe to retain across
        # Torch/JAX autodiff calls.
        self._onsite_term_map = np.zeros((len(self._terms), 4), dtype=float)
        self._edge_term_map = np.zeros((len(self._terms), 16), dtype=float)
        for term_index, term in enumerate(self._terms):
            labels = tuple(_PAULI_LABELS.index(label) for label in term.paulis)
            if term.support == "onsite":
                self._onsite_term_map[term_index, labels[0]] = 1.0
            else:
                self._edge_term_map[term_index, labels[0] * 4 + labels[1]] = 1.0
        self._cluster_embedding_cache = {}
        self._generic_cluster_cache = {}
        self.site_directions = {
            (i, j): _site_directions(i, j, self.lx, self.ly, *self.cyclic)
            for i in range(self.lx)
            for j in range(self.ly)
        }
        self.plaquette_starts = _plaquette_starts(self.lx, self.ly, self.cyclic)
        self.pair_orbits = _pair_orbits() if symmetry == "C4" else tuple(
            (pair, (pair,)) for pair in _all_direction_pairs()
        )
        self.triple_orbits = _subset_orbits(3, symmetry)
        self.path_orbits = _path_orbits(symmetry)
        self._build_count = 0
        self._compiled_exp = None

    @classmethod
    def compile(cls, lx, ly, terms, **kwargs):
        """Compile fixed lattice and Pauli topology for repeated evaluations.

        This constructor does not evaluate an exponential and does not cache
        backend coefficient values. Use :meth:`exp` for a one-off call or
        :meth:`compile_exp` when the exponential policy should be reused.
        """
        return cls(lx, ly, terms, **kwargs)

    @property
    def terms(self):
        """Read-only translation-invariant Pauli slots."""
        return self._terms

    @property
    def num_terms(self):
        """Number of coefficient slots."""
        return len(self._terms)

    @property
    def cache_info(self):
        """Return topology-only compilation diagnostics."""
        return {
            "compiled": True,
            "builds": self._build_count,
            "terms": self.num_terms,
            "order": self.order,
            "pair_orbits": len(self.pair_orbits),
            "tree_orbits": len(self.triple_orbits) + len(self.path_orbits),
            "plaquettes": len(self.plaquette_starts),
            "cluster_embedding_plans": len(self._cluster_embedding_cache),
            "generic_cluster_levels": len(self._generic_cluster_cache),
            "generic_cluster_shapes": sum(
                len(level) for level in self._generic_cluster_cache.values()
            ),
            "generic_translated_clusters": sum(
                sum(len(record[1]) for record in level)
                for level in self._generic_cluster_cache.values()
            ),
            "fused_pauli_slots": int(
                np.count_nonzero(self._onsite_term_map)
                + np.count_nonzero(self._edge_term_map)
            ),
            "cyclic": self.cyclic,
            "symmetry": self.symmetry,
            "max_tree_rank": self.max_tree_rank,
            "compiled_exp": self._compiled_exp is not None,
        }

    def compile_exp(self):
        """Return the cached fixed-topology :class:`CompiledPEPOExp`.

        Only geometry and channel structure are cached. Coefficients and the
        exponential step are supplied afresh to every call so Torch/JAX
        autodiff graphs cannot become stale.
        """
        if self._compiled_exp is None:
            self._compiled_exp = CompiledPEPOExp(self)
        return self._compiled_exp

    def _prepare_exp_plan(self):
        """Precompute all small cluster embedding maps for this basis."""
        # Populate the process-wide physical basis cache before building the
        # per-basis cluster maps.
        _backend_pauli_basis(1)
        _backend_pauli_basis(2)
        # The joint ordered-product path always evaluates the one-site
        # background and positive reference edge, including at order two.
        self._cluster_embedding_plan(1, ())
        self._cluster_embedding_plan(2, ((0, 1, "r"),))
        if self.order < 3:
            return self
        representatives = []
        for representative, _orbit in self.pair_orbits:
            representatives.append(
                tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
            )
        if self.order >= 4:
            for representative, _orbit in self.triple_orbits:
                representatives.append(
                    tuple(
                        (0, index + 1, direction)
                        for index, direction in enumerate(representative)
                    )
                )
            for representative, _orbit in self.path_orbits:
                representatives.append(
                    tuple(
                        (index, index + 1, direction)
                        for index, direction in enumerate(representative)
                    )
                )
        for edges in representatives:
            nsites = max(
                max(source, target) for source, target, _direction in edges
            ) + 1
            self._cluster_embedding_plan(nsites, edges)
            if nsites == 4:
                for center, branches in _three_subtrees(edges):
                    three_edges = tuple(
                        (0, index + 1, direction)
                        for index, (_endpoint, direction) in enumerate(branches)
                    )
                    self._cluster_embedding_plan(3, three_edges)
        if self.order >= 4 and self.plaquette_starts:
            loop_edges = _plaquette_edges()
            self._cluster_embedding_plan(4, loop_edges)
            for _center, branches in _three_subtrees(loop_edges):
                three_edges = tuple(
                    (0, index + 1, direction)
                    for index, (_endpoint, direction) in enumerate(branches)
                )
                self._cluster_embedding_plan(3, three_edges)
        return self

    def _generic_cluster_records(self, order):
        """Return valid translated generic shapes, cached by lattice level.

        The cache contains geometry only.  A record is
        ``(source_shape, source_embeddings, variants)`` where ``variants``
        holds ``(shape, embeddings)`` pairs.  With C4 reduction, the source
        residual and its tree factorization are transported to the variants;
        without C4 every oriented shape is its own source.  Invalid shapes
        are filtered before any local exponential is evaluated, which is
        important on small periodic tori where many abstract polyominoes
        self-overlap after wrapping.
        """
        try:
            return self._generic_cluster_cache[order]
        except KeyError:
            pass
        shapes = generate_connected_cluster_shapes(
            order,
            min_sites=order,
            quotient_rotations=self.symmetry == "C4",
        )
        records = []
        for shape in shapes:
            if self.symmetry == "C4":
                candidates = tuple(
                    (variant, _cluster_shape_embeddings(
                        variant,
                        self.lx,
                        self.ly,
                        self.cyclic,
                    ))
                    for variant, _source_to_target, _turns in _cluster_shape_c4_orbit(
                        shape
                    )
                )
            else:
                candidates = (
                    (
                        shape,
                        _cluster_shape_embeddings(
                            shape,
                            self.lx,
                            self.ly,
                            self.cyclic,
                        ),
                    ),
                )
            variants = tuple(
                (variant, embeddings)
                for variant, embeddings in candidates
                if embeddings
            )
            if not variants:
                continue
            source_shape, source_embeddings = variants[0]
            records.append((source_shape, source_embeddings, variants))
        records = tuple(records)
        self._generic_cluster_cache[order] = records
        return records

    def _coefficient_values(self, parameters, coefficients):
        if coefficients is not None and parameters is not None:
            raise ValueError("parameters and coefficients are mutually exclusive.")
        if coefficients is None:
            values = []
            for term in self._terms:
                value = term.coefficient
                if hasattr(value, "resolve"):
                    value = value.resolve(parameters)
                elif callable(value):
                    if parameters is None:
                        raise KeyError(
                            "callable Pauli coefficients require parameters."
                        )
                    value = value(parameters)
                values.append(value)
        else:
            shape = getattr(coefficients, "shape", None)
            if shape is not None:
                shape = tuple(shape)
                if not shape:
                    if self.num_terms != 1:
                        raise ValueError(
                            "a scalar coefficient batch is valid only for one term."
                        )
                    values = [coefficients]
                elif len(shape) == 1:
                    if int(shape[0]) != self.num_terms:
                        raise ValueError(
                            f"coefficients must have length {self.num_terms}, "
                            f"got {shape[0]}."
                        )
                    values = [coefficients[index] for index in range(self.num_terms)]
                else:
                    raise ValueError("coefficients must be one-dimensional.")
            else:
                try:
                    values = list(coefficients)
                except TypeError as exc:
                    raise TypeError("coefficients must be one-dimensional.") from exc
                if len(values) != self.num_terms:
                    raise ValueError(
                        f"coefficients must have length {self.num_terms}, "
                        f"got {len(values)}."
                    )
        for index, value in enumerate(values):
            ndim = getattr(value, "ndim", None)
            if ndim is None:
                ndim = np.ndim(value)
            if ndim != 0:
                raise TypeError(f"coefficient[{index}] must be scalar.")
        reference = _backend_reference(values)
        return tuple(_as_backend(value, like=reference) for value in values)

    def coefficients(self, parameters=None):
        """Evaluate the coefficient slots as one backend-native vector."""
        values = self._coefficient_values(parameters, None)
        return _backend_stack(values)

    def _hamiltonian_components(self, values, beta):
        """Fuse coefficient slots into onsite and edge Pauli components."""
        reference = _backend_reference((*values, beta))
        coefficient_batch = _backend_stack(values)
        onsite_map = _as_backend(self._onsite_term_map, like=reference)
        edge_map = _as_backend(self._edge_term_map, like=reference)
        return (
            ar.do(
                "tensordot",
                coefficient_batch,
                onsite_map,
                axes=([0], [0]),
            ),
            ar.do(
                "tensordot",
                coefficient_batch,
                edge_map,
                axes=([0], [0]),
            ),
        )

    @staticmethod
    def _components_to_operators(onsite_components, edge_components):
        """Convert Pauli components into local Hamiltonian matrices."""
        onsite_components = _complexify_backend(onsite_components)
        edge_components = _complexify_backend(edge_components)
        onsite_basis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=onsite_components),
            axis=0,
        )
        edge_basis = ar.do(
            "stack",
            _backend_pauli_basis(2, like=edge_components),
            axis=0,
        )
        return (
            ar.do("tensordot", onsite_components, onsite_basis, axes=([0], [0])),
            ar.do("tensordot", edge_components, edge_basis, axes=([0], [0])),
        )

    def _hamiltonian(self, values, beta):
        """Return local Hamiltonian matrices after fused slot assembly."""
        components = self._hamiltonian_components(values, beta)
        return self._components_to_operators(*components)

    @staticmethod
    def _oriented_edge(operator, direction):
        return (
            operator
            if direction in _POSITIVE_DIRECTIONS
            else _backend_swap_two_site(operator, 2)
        )

    def _cluster_embedding_plan(self, nsites, edges):
        """Cache linear maps from local Pauli components to a cluster matrix."""
        _backend_pauli_basis(1)
        _backend_pauli_basis(2)
        key = (nsites, tuple(edges))
        try:
            return self._cluster_embedding_cache[key]
        except KeyError:
            pass
        dimension = 2**nsites
        onsite_basis = np.stack(
            [
                sum(
                    (
                        np.asarray(
                            _backend_embed_operator(
                                matrix,
                                (site,),
                                nsites,
                                2,
                            )
                        )
                        for site in range(nsites)
                    ),
                    start=np.zeros((dimension, dimension), dtype=complex),
                )
                for matrix in _PAULI_BASIS_CACHE[1]
            ],
            axis=0,
        )
        edge_basis = np.zeros((16, dimension, dimension), dtype=complex)
        for component, matrix in enumerate(_PAULI_BASIS_CACHE[2]):
            for source, target, direction in edges:
                oriented = (
                    matrix
                    if direction in _POSITIVE_DIRECTIONS
                    else _swap_two_site_operator(matrix, 2)
                )
                edge_basis[component] += np.asarray(
                    _backend_embed_operator(
                        oriented,
                        (source, target),
                        nsites,
                        2,
                    )
                )
        plan = {"onsite": onsite_basis, "edge": edge_basis}
        self._cluster_embedding_cache[key] = plan
        return plan

    def _cluster_hamiltonian(
        self,
        nsites,
        edges,
        onsite_components,
        edge_components,
        *,
        like,
    ):
        """Assemble a cluster Hamiltonian from cached component embeddings."""
        onsite_components = _complexify_backend(onsite_components)
        edge_components = _complexify_backend(edge_components)
        plan = self._cluster_embedding_plan(nsites, edges)
        onsite_basis = _as_backend(plan["onsite"], like=like)
        edge_basis = _as_backend(plan["edge"], like=like)
        onsite_part = ar.do(
            "tensordot",
            onsite_components,
            onsite_basis,
            axes=([0], [0]),
        )
        edge_part = ar.do(
            "tensordot",
            edge_components,
            edge_basis,
            axes=([0], [0]),
        )
        return ar.do("add", onsite_part, edge_part)

    @staticmethod
    def _ordered_cluster_product(factor_data, nsites, edges):
        """Evaluate one local ordered product on a connected cluster.

        ``factor_data`` contains the local Hamiltonian components for every
        exponential factor.  This is deliberately a local dense operation:
        no intermediate full-lattice MPO or PEPO is constructed.  Keeping the
        matrix products here is what makes the PEPO route a joint
        Guppy-style cluster expansion rather than a product of independent
        global approximations.
        """
        reference = _backend_reference(
            tuple(
                value
                for _basis, beta, onsite_components, edge_components in factor_data
                for value in (beta, onsite_components, edge_components)
            )
        )
        result = None
        for basis, beta, onsite_components, edge_components in factor_data:
            hamiltonian = basis._cluster_hamiltonian(
                nsites,
                edges,
                onsite_components,
                edge_components,
                like=reference,
            )
            local_exp = _backend_expm(
                ar.do("multiply", -beta, hamiltonian)
            )
            result = (
                local_exp
                if result is None
                else ar.do("matmul", result, local_exp)
            )
        return result

    @staticmethod
    def _ordered_cluster_product_batch(
        factor_data,
        nsites,
        edge_batches,
        *,
        batch_size=8,
    ):
        """Evaluate several local ordered products in backend batches.

        A translated cluster has the same local ``W_S`` as its source
        embedding on a homogeneous square lattice.  This helper therefore
        batches the source shapes at one cluster level, while retaining the
        factor order inside every batch.  ``batch_size`` bounds the temporary
        ``(batch, 2**p, 2**p)`` allocation for order-nine clusters.
        """
        edge_batches = tuple(tuple(edges) for edges in edge_batches)
        if not edge_batches:
            return ()
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        reference = _backend_reference(
            tuple(
                value
                for _basis, beta, onsite_components, edge_components in factor_data
                for value in (beta, onsite_components, edge_components)
            )
        )
        results = []
        for start in range(0, len(edge_batches), batch_size):
            chunk = edge_batches[start : start + batch_size]
            product_batch = None
            for basis, beta, onsite_components, edge_components in factor_data:
                hamiltonians = tuple(
                    basis._cluster_hamiltonian(
                        nsites,
                        edges,
                        onsite_components,
                        edge_components,
                        like=reference,
                    )
                    for edges in chunk
                )
                hamiltonian_batch = ar.do("stack", hamiltonians, axis=0)
                local_exp = _backend_expm(
                    ar.do("multiply", -beta, hamiltonian_batch)
                )
                product_batch = (
                    local_exp
                    if product_batch is None
                    else ar.do("matmul", product_batch, local_exp)
                )
            results.extend(product_batch[index] for index in range(len(chunk)))
        return tuple(results)

    @staticmethod
    def _embed_with_background(operator, positions, nsites, background):
        """Embed a connected correction and dress untouched sites by ``E1``."""
        result = _backend_embed_operator(operator, positions, nsites, 2)
        positions = set(positions)
        for site in range(nsites):
            if site in positions:
                continue
            result = ar.do(
                "matmul",
                result,
                _backend_embed_operator(background, (site,), nsites, 2),
            )
        return result

    def _connected_residual(
        self,
        nsites,
        edges,
        factor_data,
        one_exp,
        edge_residual,
    ):
        """Evaluate a joint connected residual by partition subtraction."""
        exact = self._ordered_cluster_product(factor_data, nsites, edges)
        residual = ar.do(
            "subtract",
            exact,
            _backend_operator_product([one_exp] * nsites),
        )
        for source, target, direction in edges:
            lower = self._oriented_edge(edge_residual, direction)
            residual = ar.do(
                "subtract",
                residual,
                self._embed_with_background(
                    lower,
                    (source, target),
                    nsites,
                    one_exp,
                ),
            )
        if nsites == 4:
            for first_index, first in enumerate(edges):
                first_sites = {first[0], first[1]}
                for second in edges[first_index + 1 :]:
                    if first_sites & {second[0], second[1]}:
                        continue
                    first_lower = _backend_embed_operator(
                        self._oriented_edge(edge_residual, first[2]),
                        (first[0], first[1]),
                        nsites,
                        2,
                    )
                    second_lower = _backend_embed_operator(
                        self._oriented_edge(edge_residual, second[2]),
                        (second[0], second[1]),
                        nsites,
                        2,
                    )
                    residual = ar.do(
                        "subtract",
                        residual,
                        ar.do("matmul", first_lower, second_lower),
                    )
            for center, branches in _three_subtrees(edges):
                endpoints = tuple(endpoint for endpoint, _direction in branches)
                three_edges = tuple(
                    (0, index + 1, direction)
                    for index, (_endpoint, direction) in enumerate(branches)
                )
                lower = self._connected_residual(
                    3,
                    three_edges,
                    factor_data,
                    one_exp,
                    edge_residual,
                )
                residual = ar.do(
                    "subtract",
                    residual,
                    self._embed_with_background(
                        lower,
                        (center, *endpoints),
                        nsites,
                        one_exp,
                    ),
                )
        return residual

    @staticmethod
    def _center_tensor(coefficients):
        """Convert ``(physical, active...)`` Pauli coefficients to blocks."""
        basis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=coefficients),
            axis=0,
        )
        # One contraction replaces one Python/backend operation per active
        # sector. The result already has ``active_shape + (2, 2)`` layout.
        return ar.do("tensordot", coefficients, basis, axes=([0], [0]))

    @staticmethod
    def _path_tensors(coefficients):
        """Build a fixed-rank two-block factorization of a four-site path."""
        paulis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=coefficients),
            axis=0,
        )
        coefficient_view = ar.do(
            "reshape",
            coefficients,
            (4, 4, 4, 4, 1, 1),
        )
        p1_view = ar.do("reshape", paulis, (1, 4, 1, 1, 2, 2))
        left = ar.do("multiply", coefficient_view, p1_view)
        left = ar.do("reshape", left, (4, 64, 2, 2))

        # The right factor is a fixed selector tensor: its last virtual index
        # must equal the final physical Pauli label. Constructing it by
        # broadcasting avoids the old 256-entry Python loop on every call.
        selector = _as_backend(np.eye(4), like=coefficients)
        selector = ar.do("reshape", selector, (1, 1, 4, 4, 1, 1))
        p2_view = ar.do("reshape", paulis, (1, 4, 1, 1, 2, 2))
        right = ar.do("multiply", p2_view, selector)
        right = ar.do(
            "multiply",
            right,
            _as_backend(np.ones((4, 1, 1, 1, 1, 1)), like=coefficients),
        )
        right = ar.do("reshape", right, (64, 4, 2, 2))
        return (
            left,
            right,
        )

    @staticmethod
    def _loop_tensors(coefficients):
        """Build fixed-rank corner tensors for a four-site plaquette loop.

        The physical coefficient tensor is ordered around the cycle as
        ``(lower-left, upper-left, upper-right, lower-right)``. Every loop
        bond carries a fixed 16-state pair history. The corner tensors pass
        that history around the cycle, so this is an exact tensor-ring
        factorization with no coefficient-dependent SVD.
        """
        loop_coefficients = ar.do("transpose", coefficients, (0, 1, 3, 2))
        loop_coefficients = ar.do("reshape", loop_coefficients, (4, 4, 16))
        pair_labels = np.arange(16)
        # Keep these labels as static NumPy indices.  The resulting Pauli
        # banks are constants, while ``loop_coefficients`` remains a backend
        # array and therefore stays on the autodiff graph.
        pauli_values = np.stack(_backend_pauli_basis(1), axis=0)
        first_paulis = _as_backend(
            pauli_values[pair_labels // 4],
            like=coefficients,
        )
        second_paulis = _as_backend(
            pauli_values[pair_labels % 4],
            like=coefficients,
        )
        diagonal = _as_backend(np.eye(16), like=coefficients)
        ones = _as_backend(np.ones((1, 16, 1, 1)), like=coefficients)
        first_corner = ar.do(
            "multiply",
            ar.do("reshape", first_paulis, (1, 16, 2, 2)),
            ar.do("reshape", diagonal, (16, 16, 1, 1)),
        )
        second_corner = ar.do(
            "multiply",
            ar.do("reshape", loop_coefficients, (16, 16, 1, 1)),
            ar.do("reshape", second_paulis, (16, 1, 2, 2)),
        )
        third_corner = ar.do(
            "multiply",
            ar.do("reshape", first_paulis, (1, 16, 2, 2)),
            ar.do("reshape", diagonal, (16, 16, 1, 1)),
        )
        fourth_corner = ar.do(
            "multiply",
            ar.do("reshape", second_paulis, (16, 1, 2, 2)),
            ones,
        )
        return first_corner, second_corner, third_corner, fourth_corner

    def _add_generic_active_levels(
        self,
        blocks,
        allocator,
        factor_data,
        one_exp,
    ):
        """Add backend-native connected corrections for orders five to nine.

        ``W_S`` is evaluated once for every source shape at a given level and
        in small batches.  Its translated copies share the same local
        residual and PEPO factorization.  For C4-symmetric bases the rotated
        copies are transported rather than recomputed.  The full shape graph
        is used in the local ordered product and in lower-order subtraction;
        only the final tensor factorization chooses a spanning tree.
        """
        for cluster_order in range(5, self.order + 1):
            records = self._generic_cluster_records(cluster_order)
            if not records:
                continue
            lower_active = ActivePEPOBlocks(
                lx=self.lx,
                ly=self.ly,
                cyclic=self.cyclic,
                bond_dim=allocator.next_sector,
                physical_dim=one_exp.shape[0],
                site_directions=self.site_directions,
                blocks={
                    site: dict(site_blocks)
                    for site, site_blocks in blocks.items()
                },
            )
            source_products = self._ordered_cluster_product_batch(
                factor_data,
                cluster_order,
                [record[0].edges for record in records],
            )
            for (source_shape, source_embeddings, variants), exact in zip(
                records,
                source_products,
            ):
                lower = _contract_active_support_backend(
                    lower_active,
                    source_embeddings[0],
                    source_shape.edges,
                )
                source_residual = ar.do("subtract", exact, lower)
                factorized = _tree_factorize_operator_backend(
                    source_residual,
                    source_shape.edges,
                    source_shape.nsites,
                    one_exp.shape[0],
                    self.max_tree_rank,
                )
                if factorized is None:
                    continue
                (
                    source_tensors,
                    source_parent,
                    source_parent_direction,
                    source_children,
                    source_ranks,
                ) = factorized

                for variant, embeddings in variants:
                    source_to_target, turns = _shape_rotation_map(
                        source_shape,
                        variant,
                    )
                    if source_to_target == tuple(range(source_shape.nsites)):
                        local_tensors = source_tensors
                        parent = source_parent
                        parent_direction = source_parent_direction
                        children = source_children
                    else:
                        local_tensors, parent, parent_direction, children = (
                            _transform_tree_factorization(
                                source_tensors,
                                source_parent,
                                source_parent_direction,
                                source_children,
                                source_to_target,
                                turns,
                            )
                        )
                    ranks = {
                        source_to_target[site]: rank
                        for site, rank in source_ranks.items()
                    }
                    tree_directions = {}
                    sectors = {}
                    for child, parent_site in parent.items():
                        if parent_site is None:
                            continue
                        direction = parent_direction[child]
                        tree_directions[(parent_site, child)] = direction
                        tree_directions[(child, parent_site)] = _OPPOSITE_DIRECTION[
                            direction
                        ]
                        sectors[child] = allocator.allocate(ranks[child])
                    _add_tree_factor_blocks_backend(
                        blocks,
                        self.site_directions,
                        embeddings,
                        local_tensors,
                        parent,
                        tree_directions,
                        sectors,
                        one_exp.shape[0],
                    )

    def _build_active(self, beta, values, *, factor_data=None):
        """Build one PEPO from local joint cluster products.

        A single :class:`PauliPEPOBasis` supplies one factor in the common
        case.  ``factor_data`` is used by ordered products and contains all
        factors at once; the same connected residual hierarchy is then used
        for ``exp(A_C) @ exp(B_C) @ ...`` on every local cluster ``C``.
        """
        if factor_data is None:
            onsite_components, edge_components = self._hamiltonian_components(
                values,
                beta,
            )
            reference = _backend_reference(
                (beta, onsite_components, edge_components)
            )
            beta = _as_backend(beta, like=reference)
            onsite_components = _as_backend(onsite_components, like=reference)
            edge_components = _as_backend(edge_components, like=reference)
            factor_data = (
                (self, beta, onsite_components, edge_components),
            )
        else:
            factor_data = tuple(factor_data)
            if not factor_data:
                raise ValueError("factor_data must contain at least one factor.")

        one_exp = self._ordered_cluster_product(factor_data, 1, ())
        edge_exact = self._ordered_cluster_product(
            factor_data,
            2,
            ((0, 1, "r"),),
        )
        edge_residual = ar.do(
            "subtract",
            edge_exact,
            _backend_operator_product([one_exp, one_exp]),
        )
        reference = _backend_reference((one_exp, edge_residual))
        paulis = _backend_pauli_basis(1, like=reference)
        blocks = _initialize_blocks(self.lx, self.ly, one_exp, self.site_directions)
        allocator = _SectorAllocator()

        if self.order >= 2:
            edge_coefficients = _backend_pauli_expand(edge_residual, 2)
            channels = tuple(product(range(4), repeat=2))
            source = _backend_stack([paulis[first] for first, _ in channels])
            target = _backend_stack(
                [
                    ar.do("multiply", edge_coefficients[first, second], paulis[second])
                    for first, second in channels
                ]
            )
            sectors = allocator.allocate(len(channels))
            _add_positive_edge_channels(
                blocks,
                self.site_directions,
                source,
                target,
                sectors,
            )

        if self.order >= 3:
            for representative, orbit in self.pair_orbits:
                if not any(
                    all(direction in directions for direction in pair)
                    for pair in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                edges = tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    3,
                    edges,
                    factor_data,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 3)
                sectors = (
                    allocator.allocate(4),
                    allocator.allocate(4),
                )
                center = self._center_tensor(coefficients)
                for pair in orbit:
                    pair_center = (
                        center
                        if pair == representative
                        else _rotate_direction_tensor(representative, pair, center)
                    )
                    for axis, direction in enumerate(pair):
                        _add_single_direction_blocks(
                            blocks,
                            self.site_directions,
                            self.lx,
                            self.ly,
                            _OPPOSITE_DIRECTION[direction],
                            sectors[axis],
                            paulis,
                            source=False,
                            cyclic=self.cyclic,
                        )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        pair,
                        pair_center,
                        sectors,
                    )

        if self.order >= 4:
            for representative, orbit in self.triple_orbits:
                if not any(
                    all(direction in directions for direction in star)
                    for star in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                edges = tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    4,
                    edges,
                    factor_data,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 4)
                sectors = tuple(allocator.allocate(4) for _ in range(3))
                center = self._center_tensor(coefficients)
                for star in orbit:
                    star_center = (
                        center
                        if star == representative
                        else _rotate_direction_tensor(representative, star, center)
                    )
                    for axis, direction in enumerate(star):
                        _add_single_direction_blocks(
                            blocks,
                            self.site_directions,
                            self.lx,
                            self.ly,
                            _OPPOSITE_DIRECTION[direction],
                            sectors[axis],
                            paulis,
                            source=False,
                            cyclic=self.cyclic,
                        )
                    _add_triple_blocks(
                        blocks,
                        self.site_directions,
                        star,
                        star_center,
                        sectors[0],
                    )

            for representative, orbit in self.path_orbits:
                if not any(
                    _path_start_sites(steps, self.lx, self.ly, self.cyclic)
                    for steps in orbit
                ):
                    continue
                edges = tuple(
                    (index, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    4,
                    edges,
                    factor_data,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 4)
                left, right = self._path_tensors(coefficients)
                for steps in orbit:
                    first_sectors = allocator.allocate(4)
                    middle_sectors = allocator.allocate(64)
                    last_sectors = allocator.allocate(4)
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        steps[0],
                        first_sectors,
                        paulis,
                        source=True,
                        cyclic=self.cyclic,
                    )
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        _OPPOSITE_DIRECTION[steps[2]],
                        last_sectors,
                        paulis,
                        source=False,
                        cyclic=self.cyclic,
                    )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        (_OPPOSITE_DIRECTION[steps[0]], steps[1]),
                        left,
                        (first_sectors, middle_sectors),
                    )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        (_OPPOSITE_DIRECTION[steps[1]], steps[2]),
                        right,
                        (middle_sectors, last_sectors),
                    )

        if self.order >= 4 and self.plaquette_starts:
            loop_edges = _plaquette_edges()
            loop_residual = self._connected_residual(
                4,
                loop_edges,
                factor_data,
                one_exp,
                edge_residual,
            )
            loop_coefficients = _backend_pauli_expand(loop_residual, 4)
            loop_tensors = self._loop_tensors(loop_coefficients)
            for start in self.plaquette_starts:
                upper = _site_after(start, "u", self.lx, self.ly, self.cyclic)
                right = _site_after(start, "r", self.lx, self.ly, self.cyclic)
                diagonal = _site_after(upper, "r", self.lx, self.ly, self.cyclic)
                loop_sites = (start, upper, diagonal, right)
                lower_bond = allocator.allocate(16)
                right_bond = allocator.allocate(16)
                upper_bond = allocator.allocate(16)
                left_bond = allocator.allocate(16)
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[0],
                    ("r", "u"),
                    loop_tensors[0],
                    (left_bond, lower_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[1],
                    ("d", "r"),
                    loop_tensors[1],
                    (lower_bond, right_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[2],
                    ("l", "d"),
                    loop_tensors[2],
                    (right_bond, upper_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[3],
                    ("u", "l"),
                    loop_tensors[3],
                    (upper_bond, left_bond),
                )

        if self.order >= 5:
            self._add_generic_active_levels(
                blocks,
                allocator,
                factor_data,
                one_exp,
            )

        self._build_count += 1
        return ActivePEPOBlocks(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=2,
            site_directions=self.site_directions,
            blocks=blocks,
        )

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        tau=None,
        beta=None,
        materialize=False,
    ):
        """Evaluate ``exp(step * H(coefficients))`` as fixed-channel blocks.

        Parameters
        ----------
        step : scalar, optional
            Actual scalar in the exponential. Real time is
            ``step=-1j * tau``; imaginary time can use ``step=-beta``.
        parameters : mapping or sequence, optional
            Values used by parameterized/callable Pauli slots.
        coefficients : one-dimensional array-like, optional
            Values in ``basis.terms`` order. Mutually exclusive with
            ``parameters``.
        tau, beta : scalar, optional
            Compatibility shorthands. ``tau`` maps to ``-1j * tau`` and
            ``beta`` maps to ``-beta``.
        materialize : bool, optional
            If false (default), return sparse :class:`ActivePEPOBlocks`; if
            true, return a dense Quimb ``PEPO``.
        """
        if step is not None and (tau is not None or beta is not None):
            raise ValueError("step cannot be combined with tau or beta.")
        if tau is not None and beta is not None:
            raise ValueError("tau and beta are mutually exclusive.")
        if step is None:
            if tau is not None:
                step = -1j * tau
            elif beta is not None:
                step = -beta
            else:
                raise TypeError("exp requires step, tau, or beta.")
        values = self._coefficient_values(parameters, coefficients)
        active = self._build_active(-step, values)
        return active.to_pepo() if materialize else active

    def evaluate(
        self,
        tau=None,
        parameters=None,
        *,
        coefficients=None,
        step=None,
        beta=None,
        materialize=False,
    ):
        """Compatibility wrapper for the former ``evaluate(tau=...)`` API.

        New code should use :meth:`exp` so the scalar convention is explicit.
        """
        if step is not None or beta is not None:
            return self.exp(
                step,
                parameters,
                coefficients=coefficients,
                beta=beta,
                materialize=materialize,
            )
        return self.exp(
            tau=tau,
            parameters=parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    def time_evolution(
        self,
        tau,
        parameters=None,
        *,
        coefficients=None,
        materialize=False,
    ):
        """Compatibility alias for ``exp(step=-1j * tau)``."""
        return self.exp(
            -1j * tau,
            parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    def build(self, parameters=None, *, coefficients=None, tau=None, beta=None, materialize=False):
        """Compatibility alias for :meth:`evaluate`."""
        return self.evaluate(
            tau=tau,
            parameters=parameters,
            coefficients=coefficients,
            beta=beta,
            materialize=materialize,
        )


# Compatibility re-export; the implementation lives in ``pepo_product``.

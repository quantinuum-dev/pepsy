"""Fixed-channel Pauli PEPO basis and compiled exponential evaluator.

This module owns the value-independent Pauli-slot topology and the
coefficient-only evaluation boundary. Geometry-specific residual helpers are
resolved lazily through the shared PEPO geometry/planner boundary so this
basis does not depend on a compatibility facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import autoray as ar
import numpy as np
from itertools import product

from .mpo_automaton import _as_backend, _backend_reference
from .pepo_active import ActivePEPOBlocks
from ._cluster_api import resolve_cluster_size
from ._cluster_factorization import normalize_factorization
from ._cluster_symmetry import normalize_spatial_symmetries, verify_spatial_symmetries
from ._cluster_symmetry import (
    ClusterReusePlan, SpatialProductData, coefficient_key, permute_operator,
)

__all__ = ["PauliPEPOTerm", "CompiledPEPOExp", "PauliPEPOBasis"]

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_BASIS_CACHE = {}
_DIRECTIONS = ("u", "r", "d", "l")
_POSITIVE_DIRECTIONS = frozenset(("u", "r"))
_OPPOSITE_DIRECTION = {"u": "d", "r": "l", "d": "u", "l": "r"}


def _cluster_helper(name):
    """Resolve a planner helper only when a basis operation needs it."""
    from . import pepo_geometry

    return getattr(pepo_geometry, name)


def _lazy_cluster_proxy(name):
    def proxy(*args, **kwargs):
        return _cluster_helper(name)(*args, **kwargs)

    proxy.__name__ = name
    return proxy


def _add_generic_active_levels(*args, **kwargs):
    return _cluster_helper("_add_generic_active_levels")(*args, **kwargs)

def _add_block(*args, **kwargs):
    return _cluster_helper("_add_block")(*args, **kwargs)

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

def _normalize_pauli_where(*args, **kwargs):
    return _cluster_helper("_normalize_pauli_where")(*args, **kwargs)

def _SectorAllocator(*args, **kwargs):
    return _cluster_helper("_SectorAllocator")(*args, **kwargs)

def generate_connected_cluster_shapes(*args, **kwargs):
    return _cluster_helper("generate_connected_cluster_shapes")(*args, **kwargs)



@dataclass(frozen=True)
class PauliPEPOTerm:
    """One homogeneous or explicitly located Pauli slot in a PEPO basis.

    ``support="onsite"`` contributes the same one-site Pauli operator to
    every lattice site. ``support="edge"`` contributes the same ordered
    two-site Pauli operator to every positive (``u`` and ``r``) lattice edge.
    Set ``where=(i, j)`` for one site or
    ``where=((i0, j0), (i1, j1))`` for one nearest-neighbour edge. On a
    periodic dimension of length two, two distinct bonds have the same
    endpoints; set ``direction="u"``/``"r"``/``"d"``/``"l"`` to select the
    directed occurrence from the first endpoint. Explicitly located slots use
    a finite-lattice connected-subset builder at every supported order.
    The scalar ``coefficient`` may be a Python number, a Torch/JAX scalar, or
    a callable accepting the parameter container passed to
    :meth:`PauliPEPOBasis.exp`.
    """

    support: str
    paulis: object
    coefficient: object = 1.0
    where: object = None
    direction: str | None = None

    def __post_init__(self):
        support = _normalize_pauli_support(self.support)
        labels = _normalize_paulis(self.paulis, support=support)
        where = _normalize_pauli_where(self.where, support=support)
        direction = self.direction
        if direction is not None:
            direction = str(direction).strip().lower()
            if direction not in _DIRECTIONS:
                raise ValueError("direction must be 'u', 'r', 'd', 'l', or None.")
            if support != "edge" or where is None:
                raise ValueError(
                    "direction is valid only for an explicitly located edge."
                )
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "paulis", labels)
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "direction", direction)

    @classmethod
    def from_pauli(
        cls,
        support,
        paulis,
        *,
        coefficient=1.0,
        where=None,
        direction=None,
    ):
        """Construct a term from ``"X"`` or ``"ZZ"`` labels."""
        return cls(support, paulis, coefficient, where, direction)


@dataclass(frozen=True)
class _LocalizedClusterRecord:
    """One connected finite-lattice site subset and its bond occurrences."""

    sites: tuple[tuple[int, int], ...]
    site_indices: tuple[int, ...]
    edges: tuple[tuple[int, int, str], ...]
    edge_indices: tuple[int, ...]


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
        self.cluster_size = basis.cluster_size
        # Compile only value-independent cluster embeddings here. Matrix
        # exponentials and coefficient contractions still happen per call.
        basis._prepare_exp_plan()
        basis._prepare_spatial_plans((basis,), localized=basis.inhomogeneous)

    @property
    def cache_info(self):
        """Return current compilation and evaluation diagnostics."""
        return self.basis.cache_info

    @property
    def cluster_inventory(self):
        """Return independent per-size square-lattice shape counts."""
        return self.basis.cluster_inventory

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

    def trace_exp(self, step, parameters=None, *, coefficients=None,
                  normalized=False, state_budget=100000):
        """Evaluate the complete selected-order trace without building a PEPO."""
        return self.basis.trace_exp(
            step, parameters, coefficients=coefficients, normalized=normalized,
            state_budget=state_budget,
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

    ``cluster_size`` is the maximum connected site count (default 4).
    ``order`` remains an equivalent spelling; supplying both requires agreement.

    ``spatial_reuse=True`` shares exact local targets under verified site
    relabelings, independently of the existing ``symmetry="C4"`` block
    transport. Disable it for unreduced comparisons. Structural plans retain
    coefficient identities, while numerical targets live for one call only.

    ``factorization="fixed"`` guarantees SVD-free construction, including
    generic trees, and requires max_tree_rank=None. ``spatial_symmetries``
    optionally declares finite site permutations in lattice coordinates;
    declarations are validated against geometry and term bindings.

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
        order=None,
        cluster_size=None,
        cyclic=False,
        symmetry=None,
        max_tree_rank=None,
        spatial_reuse=True,
        spatial_symmetries=(),
        factorization="auto",
        cluster_plan=None,
    ):
        self.lx = _validate_shape(lx, "lx")
        self.ly = _validate_shape(ly, "ly")
        self.cyclic = _validate_cyclic(cyclic, self.lx, self.ly)
        self.order = resolve_cluster_size(order, cluster_size, default=4)
        self.cluster_size = self.order
        self.cluster_plan = cluster_plan
        if cluster_plan is not None:
            from .square_pepo_product import _square_plan_geometry

            _square_plan_geometry(cluster_plan)
            if (cluster_plan.shape != (self.lx, self.ly)
                    or cluster_plan.cyclic != self.cyclic
                    or cluster_plan.cluster_size != min(self.order, self.lx * self.ly)):
                raise ValueError("shape, cyclic and cluster_size must match cluster_plan.")
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
        if not isinstance(spatial_reuse, bool):
            raise TypeError("spatial_reuse must be a bool.")
        self.spatial_reuse = spatial_reuse
        self.symmetry = symmetry
        self.max_tree_rank = max_tree_rank
        self.factorization = normalize_factorization(factorization)
        if self.factorization == "fixed" and max_tree_rank is not None:
            raise ValueError("factorization='fixed' requires max_tree_rank=None.")
        self._terms = tuple(_normalize_pauli_term(term) for term in terms)
        if not self._terms:
            raise ValueError("terms must contain at least one Pauli slot.")
        if self.factorization == "fixed" and symmetry == "C4" and any(
            term.support == "edge" and term.paulis != term.paulis[::-1]
            for term in self._terms
        ):
            raise ValueError(
                "fixed C4 block transport requires each edge slot to be invariant "
                "under endpoint reversal; use symmetry=None for term-aware "
                "automatic reuse or validated spatial_symmetries."
            )
        self.inhomogeneous = any(term.where is not None for term in self._terms)
        if self.inhomogeneous:
            if self.symmetry is not None:
                raise ValueError(
                    "explicit Pauli PEPO locations cannot use geometric symmetry."
                )
        self.site_directions = {
            (i, j): _site_directions(i, j, self.lx, self.ly, *self.cyclic)
            for i in range(self.lx)
            for j in range(self.ly)
        }
        self._sites = tuple(self.site_directions)
        self.spatial_symmetries = normalize_spatial_symmetries(self._sites, spatial_symmetries)
        if self.spatial_symmetries and not self.spatial_reuse:
            raise ValueError("spatial_symmetries requires spatial_reuse=True.")
        self._site_indices = {
            site: index for index, site in enumerate(self._sites)
        }
        self._positive_edges = tuple(
            (
                site,
                _site_after(site, direction, self.lx, self.ly, self.cyclic),
                direction,
            )
            for site, directions in self.site_directions.items()
            for direction in directions
            if direction in _POSITIVE_DIRECTIONS
        )
        self._edge_indices = {
            (source, target, direction): index
            for index, (source, target, direction) in enumerate(
                self._positive_edges
            )
        }
        # Static one-hot maps let each evaluation fuse all coefficient slots
        # into onsite and edge Pauli components in two backend contractions.
        # They contain topology only, so they are safe to retain across
        # Torch/JAX autodiff calls.
        self._onsite_term_map = np.zeros((len(self._terms), 4), dtype=float)
        self._edge_term_map = np.zeros((len(self._terms), 16), dtype=float)
        self._site_term_map = np.zeros(
            (len(self._terms), len(self._sites), 4),
            dtype=float,
        )
        self._lattice_edge_term_map = np.zeros(
            (len(self._terms), len(self._positive_edges), 16),
            dtype=float,
        )
        for term_index, term in enumerate(self._terms):
            labels = tuple(_PAULI_LABELS.index(label) for label in term.paulis)
            if term.support == "onsite":
                self._onsite_term_map[term_index, labels[0]] = 1.0
                if term.where is None:
                    self._site_term_map[term_index, :, labels[0]] = 1.0
                else:
                    try:
                        site_index = self._site_indices[term.where]
                    except KeyError as exc:
                        raise ValueError(
                            f"onsite Pauli location {term.where!r} is outside "
                            f"the {self.lx}x{self.ly} lattice."
                        ) from exc
                    self._site_term_map[term_index, site_index, labels[0]] = 1.0
            else:
                self._edge_term_map[term_index, labels[0] * 4 + labels[1]] = 1.0
                if term.where is None:
                    self._lattice_edge_term_map[
                        term_index, :, labels[0] * 4 + labels[1]
                    ] = 1.0
                else:
                    source, target = term.where
                    candidates = []
                    if term.direction is not None:
                        if (
                            _site_after(
                                source,
                                term.direction,
                                self.lx,
                                self.ly,
                                self.cyclic,
                            )
                            != target
                        ):
                            raise ValueError(
                                f"edge direction {term.direction!r} does not lead "
                                f"from {source!r} to {target!r}."
                            )
                        if term.direction in _POSITIVE_DIRECTIONS:
                            key = (source, target, term.direction)
                            candidates.append((self._edge_indices.get(key), labels))
                        else:
                            positive = _OPPOSITE_DIRECTION[term.direction]
                            key = (target, source, positive)
                            candidates.append(
                                (self._edge_indices.get(key), labels[::-1])
                            )
                    else:
                        for edge_index, (edge_source, edge_target, _direction) in enumerate(
                            self._positive_edges
                        ):
                            if (source, target) == (edge_source, edge_target):
                                candidates.append((edge_index, labels))
                            elif (source, target) == (edge_target, edge_source):
                                candidates.append((edge_index, labels[::-1]))
                    candidates = [
                        candidate
                        for candidate in candidates
                        if candidate[0] is not None
                    ]
                    if not candidates:
                        raise ValueError(
                            f"edge Pauli location {term.where!r} is not a nearest-"
                            "neighbour lattice bond occurrence."
                        )
                    if len(candidates) > 1:
                        raise ValueError(
                            f"edge Pauli location {term.where!r} is ambiguous on "
                            "this periodic lattice; pass direction explicitly."
                        )
                    edge_index, labels = candidates[0]
                    component = labels[0] * 4 + labels[1]
                    self._lattice_edge_term_map[
                        term_index, edge_index, component
                    ] = 1.0
        self._cluster_embedding_cache = {}
        self._generic_cluster_cache = {}
        self._localized_cluster_cache = None
        self._localized_embedding_cache = {}
        self._localized_topology_cache = {}
        self._prepared_exp_modes = set()
        self._spatial_plans = {}
        self._spatial_slot_cache = None
        self._last_spatial_evaluations = 0
        self.plaquette_starts = _plaquette_starts(self.lx, self.ly, self.cyclic)
        self.pair_orbits = _pair_orbits() if symmetry == "C4" else tuple(
            (pair, (pair,)) for pair in _all_direction_pairs()
        )
        self.triple_orbits = _subset_orbits(3, symmetry)
        self.path_orbits = _path_orbits(symmetry)
        self._build_count = 0
        self._compiled_exp = None
        self._trace_product = None
        self._verify_declared_spatial_symmetries()

    def _verify_declared_spatial_symmetries(self):
        if not self.spatial_symmetries:
            return
        plan = ClusterReusePlan()
        keys = tuple(coefficient_key(term.coefficient) for term in self._terms)
        if any(key is None for key in keys):
            raise ValueError("cannot verify spatial_symmetries for opaque coefficient bindings.")
        record = _LocalizedClusterRecord(
            sites=self._sites, site_indices=tuple(range(len(self._sites))),
            edges=tuple((self._site_indices[a], self._site_indices[b], direction)
                        for a, b, direction in self._positive_edges),
            edge_indices=tuple(range(len(self._positive_edges))),
        )
        terms = self._spatial_factor_terms(
            plan, self, len(self._sites), record.edges, record,
            slot_labels=tuple(plan.label(key) for key in keys),
        )
        verify_spatial_symmetries(
            self.spatial_symmetries, len(self._sites),
            tuple((a, b) for a, b, _ in record.edges), (terms,),
        )

    @classmethod
    def from_plan(cls, plan, terms, **kwargs):
        """Compile located terms with automatic square/graph PEPO selection.

        Terms use MPO's ``(sites, paulis, coefficient)`` schema with integer
        indices in ``plan.sites`` order. ``layout="auto"`` selects the square
        builder when compatible; ``layout="graph"`` keeps generic output.
        """
        from .pepo_product import PEPOClusterProductExpansion
        from .mpo_product import MPOClusterFactor

        return PEPOClusterProductExpansion.from_plan(plan, (MPOClusterFactor(terms),), **kwargs)

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
        """Read-only homogeneous or explicitly located Pauli slots."""
        return self._terms

    @property
    def num_terms(self):
        """Number of coefficient slots."""
        return len(self._terms)

    @property
    def cluster_inventory(self):
        """Return square-lattice shape counts by size, including trees and loops.

        ``c4_*`` identifies rotations as well as translations. Counts describe
        geometry, independently of symmetry reuse, finite placements or ranks.
        """
        return _cluster_helper("_cluster_shape_inventory")(self.order)

    @property
    def cache_info(self):
        """Return topology-only compilation diagnostics."""
        return {
            "compiled": True,
            "builds": self._build_count,
            "terms": self.num_terms,
            "order": self.order,
            "cluster_size": self.cluster_size,
            "pair_orbits": len(self.pair_orbits),
            "tree_orbits": len(self.triple_orbits) + len(self.path_orbits),
            "plaquettes": len(self.plaquette_starts),
            "cluster_embedding_plans": len(self._cluster_embedding_cache),
            "localized_embedding_plans": len(self._localized_embedding_cache),
            "localized_tree_plans": len(self._localized_topology_cache),
            "prepared_exp_modes": tuple(sorted(self._prepared_exp_modes)),
            "generic_cluster_levels": len(self._generic_cluster_cache),
            "generic_cluster_shapes": sum(
                len(level) for level in self._generic_cluster_cache.values()
            ),
            "generic_translated_clusters": sum(
                sum(len(record[1]) for record in level)
                for level in self._generic_cluster_cache.values()
            ),
            "localized_cluster_counts": (
                {}
                if self._localized_cluster_cache is None
                else {
                    order: len(records)
                    for order, records in self._localized_cluster_cache.items()
                }
            ),
            "fused_pauli_slots": int(
                np.count_nonzero(self._onsite_term_map)
                + np.count_nonzero(self._edge_term_map)
            ),
            "inhomogeneous": self.inhomogeneous,
            "cyclic": self.cyclic,
            "symmetry": self.symmetry,
            "max_tree_rank": self.max_tree_rank,
            "factorization": self.factorization,
            "compiled_exp": self._compiled_exp is not None,
            "spatial_reuse": self.spatial_reuse,
            "declared_spatial_symmetry_count": len(self.spatial_symmetries),
            "spatial_plans": tuple(dict(plan.info) for plan in self._spatial_plans.values()),
            "last_local_targets_evaluated": self._last_spatial_evaluations,
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

    def trace_exp(self, step, parameters=None, *, coefficients=None,
                  normalized=False, state_budget=100000):
        """Trace this basis's complete cluster expansion without PEPO bonds."""
        if self._trace_product is None:
            from .pepo_product import PEPOClusterProductExpansion
            self._trace_product = PEPOClusterProductExpansion((self,))
        return self._trace_product.trace_exp(
            step, parameters, coefficients=coefficients,
            normalized=normalized, state_budget=state_budget,
        )

    def _prepare_exp_plan(self, *, localized=False):
        """Prepare geometry and static operator maps for the evaluation route."""
        localized = localized or self.inhomogeneous
        mode = "localized" if localized else "homogeneous"
        if mode in self._prepared_exp_modes:
            return self
        # Populate the process-wide physical basis cache before building the
        # per-basis cluster maps.
        _backend_pauli_basis(1)
        _backend_pauli_basis(2)
        if localized:
            for records in self._localized_cluster_records().values():
                for record in records:
                    self._localized_embedding_plan(record)
                    if len(record.sites) > 1:
                        self._localized_topology_plan(record)
            self._prepared_exp_modes.add(mode)
            return self
        # The joint ordered-product path always evaluates the one-site
        # background and positive reference edge, including at order two.
        self._cluster_embedding_plan(1, ())
        self._cluster_embedding_plan(2, ((0, 1, "r"),))
        if self.order < 3:
            self._prepared_exp_modes.add(mode)
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
        for order in range(5, self.order + 1):
            for shape, _embeddings, _variants in self._generic_cluster_records(order):
                self._cluster_embedding_plan(shape.nsites, shape.edges)
                _cluster_helper("_backend_tree_topology")(shape.nsites, tuple(shape.edges))
        self._prepared_exp_modes.add(mode)
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
        # Keep Python constants at the precision of the trainable slots.
        # Converting a constant through Torch's default float32 first loses
        # information even when stack() later promotes the batch to float64.
        from .pepo_product import _as_backend_dtype
        return tuple(_as_backend_dtype(value, like=reference) for value in values)

    def coefficients(self, parameters=None):
        """Evaluate the coefficient slots as one backend-native vector."""
        values = self._coefficient_values(parameters, None)
        return _backend_stack(values)

    def _hamiltonian_components(self, values, beta):
        """Fuse coefficient slots into onsite and edge Pauli components."""
        reference = _backend_reference((*values, beta))
        coefficient_batch = _backend_stack(values)
        # Torch's tensordot deliberately requires equal dtypes.  A complex
        # trainable coefficient therefore also needs complex static channel
        # maps; backend conversion alone would retain their host float dtype.
        coefficient_dtype = getattr(coefficient_batch, "dtype", None)
        onsite_map = _as_backend(
            self._onsite_term_map,
            like=reference,
            dtype=coefficient_dtype,
        )
        edge_map = _as_backend(
            self._edge_term_map,
            like=reference,
            dtype=coefficient_dtype,
        )
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

    def _localized_hamiltonian_components(self, values):
        """Fuse slots into one Pauli-component vector per site and edge."""
        reference = _backend_reference(values)
        coefficient_batch = _backend_stack(values)
        coefficient_dtype = getattr(coefficient_batch, "dtype", None)
        site_map = _as_backend(
            self._site_term_map,
            like=reference,
            dtype=coefficient_dtype,
        )
        edge_map = _as_backend(
            self._lattice_edge_term_map,
            like=reference,
            dtype=coefficient_dtype,
        )
        return (
            ar.do("tensordot", coefficient_batch, site_map, axes=([0], [0])),
            ar.do("tensordot", coefficient_batch, edge_map, axes=([0], [0])),
        )

    def _localized_cluster_records(self):
        """Enumerate connected finite-lattice site subsets once.

        The connectivity graph deduplicates endpoint pairs, while each record
        retains every oriented PEPO bond occurrence inside its site set. This
        distinction is required on length-two periodic dimensions, where two
        physical virtual bonds connect the same pair of sites.
        """
        if self._localized_cluster_cache is not None:
            return self._localized_cluster_cache

        if self.cluster_plan is not None:
            levels = {}
            for cluster in self.cluster_plan.index_clusters:
                levels.setdefault(len(cluster), set()).add(frozenset(cluster))
        else:
            adjacency = {index: set() for index in range(len(self._sites))}
            for source, target, _direction in self._positive_edges:
                source_index = self._site_indices[source]
                target_index = self._site_indices[target]
                adjacency[source_index].add(target_index)
                adjacency[target_index].add(source_index)

            levels = {
                1: {
                    frozenset((site_index,))
                    for site_index in range(len(self._sites))
                }
            }
            for size in range(2, min(self.order, len(self._sites)) + 1):
                candidates = set()
                for selected in levels[size - 1]:
                    frontier = {
                        neighbor
                        for site_index in selected
                        for neighbor in adjacency[site_index]
                        if neighbor not in selected
                    }
                    candidates.update(
                        selected | frozenset((neighbor,))
                        for neighbor in frontier
                    )
                levels[size] = candidates

        records = {}
        for size, selected_sets in levels.items():
            level = []
            for selected in sorted(
                selected_sets,
                key=lambda value: tuple(sorted(value)),
            ):
                site_indices = tuple(sorted(selected))
                sites = tuple(self._sites[index] for index in site_indices)
                local_index = {
                    site: index for index, site in enumerate(sites)
                }
                internal = tuple(
                    (
                        local_index[source],
                        local_index[target],
                        direction,
                        edge_index,
                    )
                    for edge_index, (source, target, direction) in enumerate(
                        self._positive_edges
                    )
                    if source in local_index and target in local_index
                )
                level.append(
                    _LocalizedClusterRecord(
                        sites=sites,
                        site_indices=site_indices,
                        edges=tuple(edge[:3] for edge in internal),
                        edge_indices=tuple(edge[3] for edge in internal),
                    )
                )
            records[size] = tuple(level)
        self._localized_cluster_cache = records
        return records

    def _localized_embedding_plan(self, record):
        """Share static Pauli embeddings by ordered local graph structure.

        Global site/edge indices select coefficients later; they do not change
        these local matrices. Keep edge ordering, orientation and repeated
        endpoint pairs in the key so directed and parallel bonds stay distinct.
        """
        key = (len(record.sites), tuple((s, t) for s, t, _ in record.edges))
        try:
            return self._localized_embedding_cache[key]
        except KeyError:
            pass

        nsites = len(record.sites)
        onsite_basis = np.stack(
            [
                np.stack(
                    [
                        np.asarray(
                            _backend_embed_operator(
                                matrix,
                                (local_site,),
                                nsites,
                                2,
                            )
                        )
                        for matrix in _backend_pauli_basis(1)
                    ],
                    axis=0,
                )
                for local_site in range(nsites)
            ],
            axis=0,
        )
        edge_basis = np.stack(
            [
                np.stack(
                    [
                        np.asarray(
                            _backend_embed_operator(
                                matrix,
                                (source, target),
                                nsites,
                                2,
                            )
                        )
                        for matrix in _backend_pauli_basis(2)
                    ],
                    axis=0,
                )
                for source, target, _direction in record.edges
            ],
            axis=0,
        ) if record.edges else None
        plan = (onsite_basis, edge_basis)
        self._localized_embedding_cache[key] = plan
        return plan

    def _localized_cluster_hamiltonian(
        self,
        record,
        site_components,
        edge_components,
        *,
        like,
    ):
        """Assemble one occurrence-specific cluster Hamiltonian."""
        onsite_basis, edge_basis = self._localized_embedding_plan(record)
        site_values = _complexify_backend(
            _backend_stack(
                [site_components[index] for index in record.site_indices]
            )
        )
        onsite_basis = _as_backend(
            onsite_basis,
            like=like,
            dtype=getattr(site_values, "dtype", None),
        )
        hamiltonian = ar.do(
            "tensordot",
            site_values,
            onsite_basis,
            axes=([0, 1], [0, 1]),
        )
        if record.edge_indices:
            edge_values = _complexify_backend(
                _backend_stack(
                    [edge_components[index] for index in record.edge_indices]
                )
            )
            edge_basis = _as_backend(
                edge_basis,
                like=like,
                dtype=getattr(edge_values, "dtype", None),
            )
            hamiltonian = ar.do(
                "add",
                hamiltonian,
                ar.do(
                    "tensordot",
                    edge_values,
                    edge_basis,
                    axes=([0, 1], [0, 1]),
                ),
            )
        return hamiltonian

    def _localized_ordered_product(self, localized, record, *, like):
        """Evaluate all ordered factors on one actual finite-lattice subset."""
        result = None
        for basis, beta, site_components, edge_components in localized:
            hamiltonian = basis._localized_cluster_hamiltonian(
                record,
                site_components,
                edge_components,
                like=like,
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

    def _prepare_spatial_plans(self, bases, *, localized):
        """Compile common binding modes before the first exponential call."""
        if not self.spatial_reuse:
            return
        bases = tuple(bases)
        if localized:
            for defaults in ((True,) * len(bases), (False,) * len(bases)):
                for records in self._localized_cluster_records().values():
                    self._localized_spatial_plan(bases, defaults, records)
        else:
            key = ("uniform", bases)
            plan = self._spatial_plans.setdefault(key, ClusterReusePlan())
            data = SpatialProductData(tuple((basis, None, None, None) for basis in bases), plan)
            for nsites, edges in self._cluster_embedding_cache:
                self._uniform_spatial_entry(data, nsites, edges)

    def _spatial_slot_descriptions(self):
        """Sparse, value-independent slot support; shared by binding modes."""
        if self._spatial_slot_cache is None:
            sites = [[] for _ in self._sites]
            edges = [[] for _ in self._positive_edges]
            for slot, site, pauli in zip(*np.nonzero(self._site_term_map)):
                sites[site].append((int(slot), int(pauli)))
            for slot, edge, pauli in zip(*np.nonzero(self._lattice_edge_term_map)):
                a, b = divmod(int(pauli), 4)
                edges[edge].append((int(slot), a, b))
            self._spatial_slot_cache = (tuple(map(tuple, sites)), tuple(map(tuple, edges)))
        return self._spatial_slot_cache

    @staticmethod
    def _spatial_factor_terms(plan, basis, nsites, edges, record=None, *, slot_labels=None):
        """Describe the same Pauli sum as the cached embedding maps."""
        terms = []
        if record is not None:
            site_slots, edge_slots = basis._spatial_slot_descriptions()
            for site, physical in enumerate(record.site_indices):
                terms.extend((slot_labels[slot], ((site, pauli),))
                             for slot, pauli in site_slots[physical])
            for (source, target, _), physical in zip(record.edges, record.edge_indices):
                terms.extend((slot_labels[slot], ((source, a), (target, b)))
                             for slot, a, b in edge_slots[physical])
        else:
            for slot, term in enumerate(basis._terms):
                # Uniform coefficient vectors may vary independently by slot.
                label = plan.label(("slot", slot))
                paulis = tuple(_PAULI_LABELS.index(p) for p in term.paulis)
                if term.support == "onsite":
                    terms.extend((label, ((site, paulis[0]),)) for site in range(nsites))
                else:
                    for source, target, direction in edges:
                        a, b = paulis if direction in _POSITIVE_DIRECTIONS else paulis[::-1]
                        terms.append((label, ((source, a), (target, b))))
        return tuple(terms)

    @classmethod
    def _uniform_spatial_entry(cls, data, nsites, edges):
        key = (nsites, tuple(edges))
        if key not in data.plan.entries:
            factors = tuple(cls._spatial_factor_terms(data.plan, basis, nsites, edges)
                            for basis, *_ in data)
            data.plan.add(key, nsites, tuple((a, b) for a, b, _ in edges), factors)
        return data.plan.entries[key]

    def _localized_spatial_plan(self, bases, defaults, records):
        nsites = len(records[0].sites)
        key = ("localized", bases, defaults, nsites)
        if key not in self._spatial_plans:
            plan = ClusterReusePlan()
            labels = []
            for basis, use_defaults in zip(bases, defaults):
                slots = []
                for slot, term in enumerate(basis._terms):
                    coefficient = coefficient_key(term.coefficient) if use_defaults else None
                    # Override vectors and opaque coefficients retain slot
                    # identity even when their current numbers happen to agree.
                    slots.append(plan.label(("slot", slot) if coefficient is None else coefficient))
                labels.append(tuple(slots))
            for index, record in enumerate(records):
                factors = tuple(self._spatial_factor_terms(
                    plan, basis, nsites, record.edges, record, slot_labels=slot_labels)
                    for basis, slot_labels in zip(bases, labels))
                plan.add(index, nsites, tuple((a, b) for a, b, _ in record.edges), factors)
            self._spatial_plans[key] = plan
        return self._spatial_plans[key]

    def _localized_reused_products(self, localized, records, *, like, defaults):
        if not self.spatial_reuse:
            self._last_spatial_evaluations += len(records)
            return self._localized_ordered_products(localized, records, like=like)
        bases = tuple(basis for basis, *_ in localized)
        plan = self._localized_spatial_plan(bases, defaults, records)
        sources = tuple(i for i, (source, _) in plan.entries.items() if i == source)
        products = dict(zip(sources, self._localized_ordered_products(
            localized, tuple(records[i] for i in sources), like=like)))
        self._last_spatial_evaluations += len(sources)
        return tuple(permute_operator(products[source], axes, 2)
                     for source, axes in plan.entries.values())

    def _localized_ordered_products(self, localized, records, *, like, batch_size=8):
        """Batch equal-size finite-cluster targets, preserving factor order.

        Actual finite embeddings share matrix size, but not coefficients.
        Batching avoids per-cluster backend dispatch and uses Torch's batched
        exponential path, which is more accurate than its low-degree scalar
        shortcut for small onsite matrices in tested Torch versions.
        """
        results = []
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            result = None
            for basis, beta, site_components, edge_components in localized:
                hamiltonians = ar.do("stack", tuple(
                    basis._localized_cluster_hamiltonian(
                        record, site_components, edge_components, like=like)
                    for record in chunk), axis=0)
                local_exp = _backend_expm(ar.do("multiply", -beta, hamiltonians))
                result = local_exp if result is None else ar.do("matmul", result, local_exp)
            results.extend(result[i] for i in range(len(chunk)))
        return tuple(results)

    def _localized_topology_plan(self, record):
        """Reuse deterministic tree metadata for identical directed graphs."""
        key = (len(record.sites), record.edges)
        if key not in self._localized_topology_cache:
            self._localized_topology_cache[key] = self._localized_tree_topology(
                record.edges, len(record.sites)
            )
        return self._localized_topology_cache[key]

    @staticmethod
    def _localized_tree_topology(edges, nsites):
        """Choose a deterministic spanning tree and a low-width root."""
        adjacency = [[] for _ in range(nsites)]
        for source, target, direction in edges:
            adjacency[source].append((target, direction))
            adjacency[target].append(
                (source, _OPPOSITE_DIRECTION[direction])
            )

        tree_adjacency = [[] for _ in range(nsites)]
        visited = {0}
        queue = [0]
        for source in queue:
            for target, direction in adjacency[source]:
                if target in visited:
                    continue
                visited.add(target)
                queue.append(target)
                tree_adjacency[source].append((target, direction))
                tree_adjacency[target].append(
                    (source, _OPPOSITE_DIRECTION[direction])
                )
        if len(visited) != nsites:
            raise ValueError("localized cluster graph must be connected.")

        def rooted(root):
            parent = {root: None}
            parent_direction = {}
            traversal = [root]
            for source in traversal:
                for target, direction in sorted(tree_adjacency[source]):
                    if target in parent:
                        continue
                    parent[target] = source
                    parent_direction[target] = direction
                    traversal.append(target)
            children = {site: [] for site in range(nsites)}
            subtree_sizes = {site: 1 for site in range(nsites)}
            for site in traversal[1:]:
                children[parent[site]].append(site)
            for site in reversed(traversal[1:]):
                subtree_sizes[parent[site]] += subtree_sizes[site]
            width = max(
                (subtree_sizes[child] for child in children[root]),
                default=0,
            )
            return width, root, parent, parent_direction, children, traversal

        _, _, parent, parent_direction, children, traversal = min(
            (rooted(root) for root in range(nsites)),
            key=lambda item: item[:2],
        )
        children = {
            site: tuple(sorted(site_children))
            for site, site_children in children.items()
        }
        subtree_sites = {}
        for site in reversed(traversal):
            descendants = [site]
            for child in children[site]:
                descendants.extend(subtree_sites[child])
            subtree_sites[site] = tuple(sorted(descendants))
        ranks = {
            site: 4 ** len(subtree_sites[site])
            for site in range(nsites)
            if parent[site] is not None
        }
        return parent, parent_direction, children, subtree_sites, ranks

    def _add_localized_pauli_tree(
        self,
        blocks,
        allocator,
        record,
        residual,
    ):
        """Insert one exact fixed-history Pauli tree correction.

        Only the root carries coefficient-dependent data. Other tensors are
        deterministic history selectors, so the default localized route has
        no coefficient-dependent SVD gauge and retains stable autodiff.
        """
        nsites = len(record.sites)
        topology = self._localized_topology_plan(record)
        parent, parent_direction, children, subtree_sites, ranks = topology
        required_rank = max(ranks.values(), default=1)
        if self.max_tree_rank is not None and self.max_tree_rank < required_rank:
            factorized = _tree_factorize_operator_backend(
                residual,
                record.edges,
                nsites,
                2,
                self.max_tree_rank,
                factorization=self.factorization,
            )
            if factorized is None:
                return
            local_tensors, parent, parent_direction, children, ranks = factorized
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
                (record.sites,),
                local_tensors,
                parent,
                tree_directions,
                sectors,
                2,
            )
            return

        coefficients = _backend_pauli_expand(residual, nsites)
        paulis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=coefficients),
            axis=0,
        )
        root = next(site for site, value in parent.items() if value is None)
        sectors = {
            site: allocator.allocate(rank) for site, rank in ranks.items()
        }
        tree_directions = {}
        for child, parent_site in parent.items():
            if parent_site is None:
                continue
            direction = parent_direction[child]
            tree_directions[(parent_site, child)] = direction
            tree_directions[(child, parent_site)] = _OPPOSITE_DIRECTION[
                direction
            ]

        root_children = children[root]
        root_axis_order = tuple(
            descendant
            for child in root_children
            for descendant in subtree_sites[child]
        ) + (root,)
        root_coefficients = ar.do(
            "transpose",
            coefficients,
            root_axis_order,
        )
        root_shape = tuple(ranks[child] for child in root_children) + (4,)
        root_coefficients = ar.do("reshape", root_coefficients, root_shape)
        root_blocks = ar.do(
            "tensordot",
            root_coefficients,
            paulis,
            axes=([len(root_children)], [0]),
        )
        for child_histories in np.ndindex(
            tuple(ranks[child] for child in root_children)
        ):
            _add_block(
                blocks,
                self.site_directions,
                record.sites[root],
                {
                    tree_directions[(root, child)]: sectors[child][history]
                    for child, history in zip(root_children, child_histories)
                },
                root_blocks[child_histories],
            )

        for site in range(nsites):
            if site == root:
                continue
            site_children = children[site]
            child_shapes = tuple(ranks[child] for child in site_children)
            for child_histories in np.ndindex(child_shapes):
                assignments = {}
                for child, history in zip(site_children, child_histories):
                    labels = np.unravel_index(
                        history,
                        (4,) * len(subtree_sites[child]),
                    )
                    assignments.update(zip(subtree_sites[child], labels))
                for pauli_index in range(4):
                    assignments[site] = pauli_index
                    parent_history = np.ravel_multi_index(
                        tuple(
                            assignments[descendant]
                            for descendant in subtree_sites[site]
                        ),
                        (4,) * len(subtree_sites[site]),
                    )
                    sector_by_direction = {
                        tree_directions[(site, child)]: sectors[child][history]
                        for child, history in zip(
                            site_children,
                            child_histories,
                        )
                    }
                    sector_by_direction[
                        tree_directions[(site, parent[site])]
                    ] = sectors[site][parent_history]
                    _add_block(
                        blocks,
                        self.site_directions,
                        record.sites[site],
                        sector_by_direction,
                        paulis[pauli_index],
                    )
        # Tracing a nonroot selector kills X/Y/Z. Inductively, the subtree
        # history on every parent leg must be all-I (index zero). At the root
        # this selects the identity coefficient. This certificate applies to
        # the closed trace only, never to lower-support operator subtraction.
        return {sector[0] for sector in sectors.values()}

    def _build_inhomogeneous_active(self, factor_sources, *, defaults=None):
        """Build an occurrence-aware finite-lattice connected-cluster PEPO."""
        factor_sources = tuple(factor_sources)
        if not factor_sources:
            raise ValueError("factor_sources must contain at least one factor.")
        if defaults is None:
            defaults = (False,) * len(factor_sources)
        self._last_spatial_evaluations = 0
        localized = []
        for basis, beta, values in factor_sources:
            if (basis.lx, basis.ly, basis.cyclic) != (
                self.lx,
                self.ly,
                self.cyclic,
            ):
                raise ValueError("inhomogeneous PEPO factors must share one lattice.")
            site_components, edge_components = (
                basis._localized_hamiltonian_components(values)
            )
            localized.append(
                (basis, beta, site_components, edge_components)
            )

        reference = _backend_reference(
            tuple(
                value
                for _basis, beta, site_components, edge_components in localized
                for value in (beta, site_components, edge_components)
            )
        )
        localized = [
            (
                basis,
                _as_backend(beta, like=reference),
                _as_backend(site_components, like=reference),
                _as_backend(edge_components, like=reference),
            )
            for basis, beta, site_components, edge_components in localized
        ]
        cluster_records = self._localized_cluster_records()
        one_exps = self._localized_reused_products(
            localized, cluster_records[1], like=reference, defaults=defaults)
        blocks = {
            site: {
                (0,) * len(self.site_directions[site]): one_exps[site_index]
            }
            for site_index, site in enumerate(self._sites)
        }
        allocator = _SectorAllocator()
        trace_sectors = {0}
        for cluster_order in range(2, min(self.order, len(self._sites)) + 1):
            # Residuals at one order subtract the completed lower-order PEPO,
            # never another correction from the same level.
            lower_active = ActivePEPOBlocks(
                lx=self.lx,
                ly=self.ly,
                cyclic=self.cyclic,
                bond_dim=allocator.next_sector,
                physical_dim=2,
                site_directions=self.site_directions,
                blocks={
                    site: dict(site_blocks)
                    for site, site_blocks in blocks.items()
                },
            )
            records = cluster_records[cluster_order]
            exact_products = self._localized_reused_products(
                localized, records, like=reference, defaults=defaults)
            lower_plan = (
                self._localized_spatial_plan(
                    tuple(basis for basis, *_ in localized), defaults, records
                )
                if self.spatial_reuse else None
            )
            lower_cache = {}
            for index, (record, exact) in enumerate(zip(records, exact_products)):
                if lower_plan is None:
                    lower = _contract_active_support_backend(
                        lower_active, record.sites, record.edges
                    )
                else:
                    # The completed lower-order expansion is equivariant under
                    # the same verified term and geometry permutation as the
                    # exact local target. Keep values only for this level/call.
                    source, axes = lower_plan.entries[index]
                    if source not in lower_cache:
                        source_record = records[source]
                        lower_cache[source] = _contract_active_support_backend(
                            lower_active, source_record.sites, source_record.edges
                        )
                    lower = permute_operator(lower_cache[source], axes, 2)
                residual = ar.do("subtract", exact, lower)
                allowed = self._add_localized_pauli_tree(
                    blocks,
                    allocator,
                    record,
                    residual,
                )
                if allowed is None:
                    # A rank-capped SVD does not have fixed Pauli selectors.
                    trace_sectors = None
                elif trace_sectors is not None:
                    trace_sectors.update(allowed)

        self._build_count += 1
        return ActivePEPOBlocks(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=2,
            site_directions=self.site_directions,
            blocks=blocks,
            trace_sectors=None if trace_sectors is None else frozenset(trace_sectors),
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
        if isinstance(factor_data, SpatialProductData):
            source, axes = PauliPEPOBasis._uniform_spatial_entry(factor_data, nsites, edges)
            if source not in factor_data.products:
                source_nsites, source_edges = source
                factor_data.products[source] = PauliPEPOBasis._ordered_cluster_product(
                    tuple(factor_data), source_nsites, source_edges)
            return permute_operator(factor_data.products[source], axes, 2)
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
        if isinstance(factor_data, SpatialProductData):
            entries = tuple(PauliPEPOBasis._uniform_spatial_entry(factor_data, nsites, edges)
                            for edges in edge_batches)
            missing = tuple(dict.fromkeys(source for source, _ in entries
                                          if source not in factor_data.products))
            if missing:
                values = PauliPEPOBasis._ordered_cluster_product_batch(
                    tuple(factor_data), nsites, tuple(edges for _, edges in missing),
                    batch_size=batch_size)
                factor_data.products.update(zip(missing, values))
            return tuple(permute_operator(factor_data.products[source], axes, 2)
                         for source, axes in entries)
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
                    factorization=self.factorization,
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

        if self.spatial_reuse:
            key = ("uniform", tuple(basis for basis, *_ in factor_data))
            plan = self._spatial_plans.setdefault(key, ClusterReusePlan())
            factor_data = SpatialProductData(factor_data, plan)
        self._last_spatial_evaluations = None
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

        if isinstance(factor_data, SpatialProductData):
            self._last_spatial_evaluations = len(factor_data.products)
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
        beta = -step
        if self.inhomogeneous:
            reference = _backend_reference((beta, *values))
            beta = _as_backend(beta, like=reference)
            active = self._build_inhomogeneous_active(
                ((self, beta, values),), defaults=(coefficients is None,))
        else:
            active = self._build_active(beta, values)
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

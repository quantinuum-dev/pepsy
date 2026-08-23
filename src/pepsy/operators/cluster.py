"""Dense square-lattice PEPO cluster-expansion builders.

This module contains the first dense Pepsy implementation of the connected
cluster construction from Vanhecke, Vanderstraeten, and Verstraete.  It is
intentionally separate from the snake-MPO Taylor path: the local residuals
are factorized into PEPO virtual channels, so the approximation is extensive
in the lattice size.

The dense implementation supports specialized tree and plaquette-loop
clusters through order four, followed by a generic connected-cluster path
through order nine. It also has a fixed-channel Pauli/autodiff path and an
explicit Symmray conversion boundary for homogeneous operator-charge
sectors; orders above nine, native charge-block solving, and mixed-charge
component splitting remain separate extension points. The joint ordered PEPO
product implementation is extracted to :mod:`pepsy.operators.pepo_product`;
this module re-exports it for compatibility.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from numbers import Integral

import autoray as ar
import numpy as np
import quimb
import quimb.tensor as qtn
from quimb.tensor.fitting import tensor_network_distance

from .mpo_automaton import _as_backend
from .diagnostics import OperatorReportInfo
from .mpo import _fixed_rank_svd
from .pepo_active import (
    ActivePEPOBlocks,
    GraphActivePEPOBlocks,
    _site_after,
)
from .pepo_basis import CompiledPEPOExp, PauliPEPOTerm, PauliPEPOBasis

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


_DIRECTIONS = ("u", "r", "d", "l")
_POSITIVE_DIRECTIONS = frozenset(("u", "r"))
_C4_ROTATION = {"u": "r", "r": "d", "d": "l", "l": "u"}
_OPPOSITE_DIRECTION = {"u": "d", "r": "l", "d": "u", "l": "r"}
_DIRECTION_VECTORS = {"u": (1, 0), "r": (0, 1), "d": (-1, 0), "l": (0, -1)}

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_BASIS_CACHE = {}


@dataclass(frozen=True)
class ConnectedClusterShape:
    """Translation-canonical connected square-lattice cluster metadata.

    ``sites`` contains integer ``(x, y)`` coordinates, sorted after removing
    a common translation. ``edges`` stores the nearest-neighbour graph as
    ``(source, target, direction)`` entries, where the first two values index
    ``sites`` and ``direction`` is measured from the source site. ``loops``
    is the graph cyclomatic number, so it is zero for trees and one for the
    elementary plaquette.

    The metadata is deliberately independent of tensor values and of BP. It
    is the geometry inventory needed by a generic connected-cluster residual
    solver; the dense PEPO builder consumes the five- through nine-site slices
    of this inventory in its generic higher-order path.
    """

    sites: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int, str], ...]
    diagonal_edges: tuple[tuple[int, int], ...]
    loops: int

    @property
    def nsites(self):
        """Return the number of sites in the cluster."""
        return len(self.sites)

    @property
    def is_tree(self):
        """Whether the nearest-neighbour cluster graph is a tree."""
        return self.loops == 0


@dataclass(frozen=True)
class GraphConnectedClusterShape:
    """Connected cluster metadata for a finite arbitrary graph.

    ``sites`` contains the labels from the parent :class:`ClusterLattice`.
    ``edges`` stores ``(source, target, lattice_edge)`` entries, where the
    first two values index ``sites`` and ``lattice_edge`` identifies the
    corresponding edge in the parent lattice.  Unlike
    :class:`ConnectedClusterShape`, this object makes no assumption about
    coordinates, translation, or a fixed coordination number.
    """

    sites: tuple[Hashable, ...]
    edges: tuple[tuple[int, int, int], ...]
    loops: int

    @property
    def nsites(self):
        """Return the number of sites in the cluster."""
        return len(self.sites)

    @property
    def is_tree(self):
        """Whether the induced cluster graph is a tree."""
        return self.loops == 0


@dataclass(frozen=True)
class ClusterLattice:
    """Finite graph geometry for a graph-native cluster expansion.

    ``edges`` are undirected lattice bonds represented as ``(source, target)``
    pairs.  Their order is stable and becomes the virtual-bond order in the
    graph PEPO representation.  The source/target order is also the order in
    which an asymmetric ``twosite_op`` is applied.

    This object is deliberately independent of Quimb's square-only ``PEPO``
    container.  It supplies finite connected-cluster inventory and the bond
    slots needed by :func:`build_graph_cluster_expansion_pepo`.
    """

    sites: tuple[Hashable, ...] | int
    edges: tuple[tuple[Hashable, Hashable], ...]
    name: str = "graph"

    def __post_init__(self):
        sites = (
            tuple(range(self.sites))
            if isinstance(self.sites, Integral)
            else tuple(self.sites)
        )
        if not sites:
            raise ValueError("ClusterLattice needs at least one site.")
        if any(not isinstance(site, Hashable) for site in sites):
            raise TypeError("ClusterLattice site labels must be hashable.")
        if len(set(sites)) != len(sites):
            raise ValueError("ClusterLattice site labels must be distinct.")

        normalized_edges = []
        seen_edges = set()
        site_set = set(sites)
        for edge in self.edges:
            try:
                source, target = edge
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "ClusterLattice edges must be two-item (source, target) pairs."
                ) from exc
            if source not in site_set or target not in site_set:
                raise ValueError("every ClusterLattice edge endpoint must be a site.")
            if source == target:
                raise ValueError("ClusterLattice does not support self-loops.")
            edge_key = frozenset((source, target))
            if edge_key in seen_edges:
                raise ValueError("ClusterLattice cannot contain duplicate undirected edges.")
            seen_edges.add(edge_key)
            normalized_edges.append((source, target))

        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "edges", tuple(normalized_edges))
        object.__setattr__(self, "name", str(self.name))

    @classmethod
    def from_edges(cls, sites, edges, *, name="graph"):
        """Construct a graph lattice from site labels and undirected edges."""
        return cls(sites, tuple(edges), name=name)

    @classmethod
    def square(cls, lx, ly, *, cyclic=False):
        """Construct the finite square graph used by the legacy PEPO path."""
        lx = _validate_shape(lx, "lx")
        ly = _validate_shape(ly, "ly")
        cyclic_x, cyclic_y = _validate_cyclic(cyclic, lx, ly)
        sites = tuple((i, j) for i in range(lx) for j in range(ly))
        edges = []
        for site in sites:
            i, j = site
            for direction in ("u", "r"):
                target = _site_after(site, direction, lx, ly, (cyclic_x, cyclic_y))
                if target is None:
                    continue
                if (site, target) not in edges and (target, site) not in edges:
                    edges.append((site, target))
        return cls(sites, tuple(edges), name="square")

    @property
    def adjacency(self):
        """Return ``site -> ((neighbor, edge_index), ...)`` adjacency."""
        adjacency = {site: [] for site in self.sites}
        for edge_index, (source, target) in enumerate(self.edges):
            adjacency[source].append((target, edge_index))
            adjacency[target].append((source, edge_index))
        return {site: tuple(links) for site, links in adjacency.items()}

    def connected_cluster_shapes(self, max_sites, *, min_sites=1):
        """Enumerate connected induced subgraphs up to ``max_sites`` sites."""
        max_sites = _validate_shape(max_sites, "max_sites")
        min_sites = _validate_shape(min_sites, "min_sites")
        if min_sites > max_sites:
            raise ValueError("min_sites must be <= max_sites.")

        site_index = {site: index for index, site in enumerate(self.sites)}
        adjacency = {
            site_index[site]: tuple(site_index[neighbor] for neighbor, _ in links)
            for site, links in self.adjacency.items()
        }
        shapes = []
        upper = min(max_sites, len(self.sites))
        levels = {
            1: {frozenset((site,)) for site in range(len(self.sites))}
        }
        for size in range(2, upper + 1):
            candidates = set()
            for selected in levels[size - 1]:
                frontier = {
                    neighbor
                    for site in selected
                    for neighbor in adjacency[site]
                    if neighbor not in selected
                }
                candidates.update(
                    selected | frozenset((neighbor,))
                    for neighbor in frontier
                )
            levels[size] = candidates

        for size in range(min_sites, upper + 1):
            for selected in sorted(levels[size], key=lambda sites: tuple(sorted(sites))):
                selected = tuple(sorted(selected))
                local_sites = tuple(self.sites[index] for index in selected)
                selected_set = set(local_sites)
                local_index = {site: index for index, site in enumerate(local_sites)}
                local_edges = tuple(
                    (
                        local_index[source],
                        local_index[target],
                        edge_index,
                    )
                    for edge_index, (source, target) in enumerate(self.edges)
                    if source in selected_set and target in selected_set
                )
                shapes.append(
                    GraphConnectedClusterShape(
                        sites=local_sites,
                        edges=local_edges,
                        loops=len(local_edges) - size + 1,
                    )
                )
        return tuple(shapes)


@dataclass(frozen=True)
class ClusterInternalSymmetry:
    """Internal charge-conservation contract for cluster Hamiltonians.

    ``name`` supports ``"U1"``, ``"Z2"``, ``"U1U1"``, and ``"Z2Z2"``.
    With ``physical_sectors`` the dense basis is assumed to be ordered by
    charge sectors and neutrality is checked blockwise.  ``generators`` can
    instead validate a symmetry in an arbitrary dense basis: U(1) entries are
    additive generators, while Z2 entries are local representation matrices.

    This contract validates and records symmetry; it does not pretend that a
    dense SVD has produced native block-sparse factors.  Native Symmray
    conversion still requires compatible physical sectors and explicit
    virtual-sector charges.
    """

    name: str
    physical_sectors: Mapping | None = None
    generators: tuple[object, ...] | None = None
    tolerance: float = 1e-10
    fermionic: bool = False

    def __post_init__(self):
        name = str(self.name).upper().replace("-", "")
        if name not in {"U1", "Z2", "U1U1", "Z2Z2"}:
            raise ValueError(
                "internal symmetry must be one of 'U1', 'Z2', 'U1U1', or 'Z2Z2'."
            )
        components = 2 if len(name) == 4 else 1
        generators = None
        if self.generators is not None:
            generators = tuple(np.asarray(generator) for generator in self.generators)
            if len(generators) != components:
                raise ValueError(
                    f"{name} requires {components} local generator/representation matrix."
                )
        if self.tolerance <= 0:
            raise ValueError("internal-symmetry tolerance must be > 0.")
        if not isinstance(self.fermionic, (bool, np.bool_)):
            raise TypeError("fermionic must be a bool.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "physical_sectors", (
            None if self.physical_sectors is None else dict(self.physical_sectors)
        ))
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "fermionic", bool(self.fermionic))

    @property
    def is_continuous(self):
        """Whether the symmetry uses additive U(1) generators/charges."""
        return self.name.startswith("U1")

    def resolved_physical_sectors(self, local_dim):
        """Return physical sectors when a native basis description is known."""
        if self.physical_sectors is not None:
            sectors = dict(self.physical_sectors)
        else:
            if self.generators is not None:
                # A generator validates an arbitrary dense basis, but does
                # not by itself tell Symmray how that basis is ordered into
                # charge sectors.
                return None
            from pepsy.tensors.symmetric import default_physical_sectors

            try:
                sectors = default_physical_sectors(self.name, local_dim)
            except ValueError:
                return None
        if sum(int(size) for size in sectors.values()) != local_dim:
            raise ValueError(
                "physical_sectors must describe exactly the local Hilbert-space dimension."
            )
        if any(int(size) < 1 for size in sectors.values()):
            raise ValueError("physical sector sizes must be positive integers.")
        return sectors

    def validate(self, twosite_op, onesite_op):
        """Validate that both local terms are neutral under this symmetry."""
        onesite_op = np.asarray(onesite_op)
        twosite_op = np.asarray(twosite_op)
        if onesite_op.ndim != 2 or onesite_op.shape[0] != onesite_op.shape[1]:
            raise ValueError("onesite_op must be a square rank-2 matrix.")
        if twosite_op.ndim != 2 or twosite_op.shape[0] != twosite_op.shape[1]:
            raise ValueError("twosite_op must be a square rank-2 matrix.")
        local_dim = onesite_op.shape[0]
        if twosite_op.shape != (local_dim**2, local_dim**2):
            raise ValueError(
                "twosite_op must have shape (local_dim**2, local_dim**2)."
            )
        if self.generators is not None:
            identity = np.eye(local_dim, dtype=np.result_type(onesite_op, complex))
            for generator in self.generators:
                generator = np.asarray(generator)
                if generator.shape != (local_dim, local_dim):
                    raise ValueError("each internal-symmetry generator must be local_dim x local_dim.")
                if self.is_continuous:
                    onsite_error = np.linalg.norm(onesite_op @ generator - generator @ onesite_op)
                    edge_generator = np.kron(generator, identity) + np.kron(identity, generator)
                else:
                    onsite_error = np.linalg.norm(onesite_op @ generator - generator @ onesite_op)
                    edge_generator = np.kron(generator, generator)
                edge_error = np.linalg.norm(twosite_op @ edge_generator - edge_generator @ twosite_op)
                if max(onsite_error, edge_error) > self.tolerance:
                    raise ValueError(
                        f"local terms are not neutral under internal symmetry {self.name}."
                    )
            return self

        sectors = self.resolved_physical_sectors(local_dim)
        if sectors is None:
            raise ValueError(
                "internal symmetry needs physical_sectors or generators for this local dimension."
            )
        charges = []
        for charge, size in sectors.items():
            charges.extend([charge] * int(size))

        def combine(left, right):
            if isinstance(left, tuple) or isinstance(right, tuple):
                left = left if isinstance(left, tuple) else (left,)
                right = right if isinstance(right, tuple) else (right,)
                return tuple(
                    _combine_internal_charge(a, b, self.name)
                    for a, b in zip(left, right)
                )
            return _combine_internal_charge(left, right, self.name)

        nonzero_onsite = np.argwhere(np.abs(onesite_op) > self.tolerance)
        for row, column in nonzero_onsite:
            if charges[row] != charges[column]:
                raise ValueError(
                    f"onesite_op changes {self.name} charge from "
                    f"{charges[column]!r} to {charges[row]!r}."
                )
        edge_charges = [combine(charges[row], charges[column]) for row, column in product(charges, charges)]
        nonzero_edge = np.argwhere(np.abs(twosite_op) > self.tolerance)
        for row, column in nonzero_edge:
            if edge_charges[row] != edge_charges[column]:
                raise ValueError(f"twosite_op is not neutral under internal symmetry {self.name}.")
        return self


def _combine_internal_charge(left, right, symmetry):
    if symmetry.startswith("Z2"):
        return (int(left) + int(right)) % 2
    return left + right


def _coerce_internal_symmetry(value):
    if value is None or isinstance(value, ClusterInternalSymmetry):
        return value
    if isinstance(value, str):
        return ClusterInternalSymmetry(value)
    raise TypeError(
        "internal_symmetry must be None, a symmetry name, or a "
        "ClusterInternalSymmetry."
    )


def _normalize_cluster_sites(sites):
    """Normalize a finite site set under translations only."""
    sites = tuple((int(x), int(y)) for x, y in sites)
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("a cluster must contain distinct, non-empty sites.")
    min_x = min(x for x, _ in sites)
    min_y = min(y for _, y in sites)
    return tuple(sorted((x - min_x, y - min_y) for x, y in sites))


def _rotate_cluster_sites(sites):
    """Rotate coordinates by 90 degrees around the origin."""
    return tuple((-y, x) for x, y in sites)


def _canonical_cluster_sites(sites, *, quotient_rotations=False):
    """Canonicalize a site set under translations and optionally C4 rotations."""
    candidate = _normalize_cluster_sites(sites)
    if not quotient_rotations:
        return candidate

    rotations = []
    for _ in range(4):
        rotations.append(_normalize_cluster_sites(candidate))
        candidate = _rotate_cluster_sites(candidate)
    return min(rotations)


def _make_cluster_shape(sites):
    """Build graph metadata for one already canonical site set."""
    sites = _normalize_cluster_sites(sites)
    site_indices = {site: index for index, site in enumerate(sites)}
    edges = []
    diagonal_edges = []
    for source, (x, y) in enumerate(sites):
        for direction, (dx, dy) in _DIRECTION_VECTORS.items():
            target = site_indices.get((x + dx, y + dy))
            if target is not None and source < target:
                edges.append((source, target, direction))
        for target in range(source + 1, len(sites)):
            tx, ty = sites[target]
            if abs(tx - x) == 1 and abs(ty - y) == 1:
                diagonal_edges.append((source, target))

    loops = len(edges) - len(sites) + 1
    return ConnectedClusterShape(
        sites=sites,
        edges=tuple(edges),
        diagonal_edges=tuple(diagonal_edges),
        loops=loops,
    )


@lru_cache(maxsize=8)
def _connected_cluster_shapes_cached(max_sites, quotient_rotations):
    """Build the immutable connected-shape inventory for one C4 policy."""
    levels = {1: {((0, 0),)}}
    for size in range(2, max_sites + 1):
        candidates = set()
        for sites in levels[size - 1]:
            occupied = set(sites)
            for x, y in sites:
                for dx, dy in _DIRECTION_VECTORS.values():
                    neighbour = (x + dx, y + dy)
                    if neighbour in occupied:
                        continue
                    candidates.add(
                        _canonical_cluster_sites(
                            (*sites, neighbour),
                            quotient_rotations=quotient_rotations,
                        )
                    )
        levels[size] = candidates

    return tuple(
        _make_cluster_shape(sites)
        for size in range(1, max_sites + 1)
        for sites in sorted(levels[size])
    )


def generate_connected_cluster_shapes(
    max_sites,
    *,
    min_sites=1,
    quotient_rotations=False,
):
    """Enumerate connected square-lattice cluster shapes.

    Shapes are generated recursively by adding a nearest neighbour to a
    smaller connected shape. They are canonicalized under translations, so
    the result is finite and independent of the lattice dimensions. By
    default rotations remain distinct, matching the oriented cluster list
    used by the reference cluster-expansion construction. Set
    ``quotient_rotations=True`` to identify the four C4 rotations.

    This function generates geometry only. It does not solve a local
    exponential, construct a PEPO, or call the BP loop-cluster expansion.

    Parameters
    ----------
    max_sites : int
        Largest cluster size to enumerate.
    min_sites : int, optional
        Smallest cluster size returned, defaulting to one.
    quotient_rotations : bool, optional
        Whether to quotient the shape inventory by square-lattice rotations.

    Returns
    -------
    tuple[ConnectedClusterShape, ...]
        Shapes ordered first by site count and then by canonical coordinates.
    """
    max_sites = _validate_shape(max_sites, "max_sites")
    min_sites = _validate_shape(min_sites, "min_sites")
    if min_sites > max_sites:
        raise ValueError("min_sites must be <= max_sites.")
    if not isinstance(quotient_rotations, (bool, np.bool_)):
        raise TypeError("quotient_rotations must be a bool.")

    shapes = _connected_cluster_shapes_cached(
        max_sites,
        bool(quotient_rotations),
    )
    return tuple(shape for shape in shapes if shape.nsites >= min_sites)


def compose_pepo_layers(layers, *, compress=False, **compress_opts):
    """Compose Quimb ``PEPO`` layers without forming a global matrix.

    The layers are supplied in application order. For example,
    ``compose_pepo_layers((u0, u1))`` returns ``u1 @ u0``. Quimb contracts
    the physical legs at each site and retains the result as a ``PEPO``. By
    default no virtual truncation is performed; pass ``compress=True`` and
    Quimb compression options to truncate after each multiplication.

    Parameters
    ----------
    layers : iterable of :class:`quimb.tensor.PEPO`
        At least one PEPO with matching lattice shape and periodicity.
    compress : bool, optional
        Whether to ask Quimb to compress each intermediate product.
    compress_opts
        Options forwarded to Quimb's PEPO compression method when
        ``compress=True``.

    Returns
    -------
    quimb.tensor.PEPO
        The composed operator, with the same physical layout as the input
        layers.
    """
    try:
        layers = tuple(layers)
    except TypeError as exc:
        raise TypeError("layers must be an iterable of Quimb PEPOs.") from exc
    if not layers:
        raise ValueError("layers must contain at least one Quimb PEPO.")
    if not all(isinstance(layer, qtn.PEPO) for layer in layers):
        raise TypeError("layers must contain only Quimb PEPO objects.")

    reference = layers[0]
    reference_shape = (
        reference.Lx,
        reference.Ly,
        bool(reference.is_cyclic_x()),
        bool(reference.is_cyclic_y()),
    )
    for layer in layers[1:]:
        shape = (
            layer.Lx,
            layer.Ly,
            bool(layer.is_cyclic_x()),
            bool(layer.is_cyclic_y()),
        )
        if shape != reference_shape:
            raise ValueError(
                "all PEPO layers must have matching lattice shape and periodicity."
            )

    result = reference.copy()
    for layer in layers[1:]:
        result = layer.apply(
            result,
            compress=compress,
            contract=True,
            **compress_opts,
        )
    return result


def _yoshida4_coefficients():
    """Return the symmetric triple-jump coefficients for fourth order."""
    cube_root_two = 2.0 ** (1.0 / 3.0)
    first = 1.0 / (2.0 - cube_root_two)
    middle = -cube_root_two / (2.0 - cube_root_two)
    return first, middle, first


def _validate_shape(value, name):
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def _validate_cyclic(cyclic, lx, ly):
    if isinstance(cyclic, (bool, np.bool_)):
        cyclic_x = cyclic_y = bool(cyclic)
    else:
        try:
            cyclic_x, cyclic_y = cyclic
        except (TypeError, ValueError) as exc:
            raise TypeError("cyclic must be a bool or a two-item bool tuple.") from exc
        if not isinstance(cyclic_x, (bool, np.bool_)) or not isinstance(
            cyclic_y, (bool, np.bool_)
        ):
            raise TypeError("cyclic must be a bool or a two-item bool tuple.")
        cyclic_x, cyclic_y = bool(cyclic_x), bool(cyclic_y)

    if cyclic_x and lx == 1:
        raise ValueError("periodic x direction requires Lx >= 2.")
    if cyclic_y and ly == 1:
        raise ValueError("periodic y direction requires Ly >= 2.")
    return cyclic_x, cyclic_y


def _as_square_operator(operator, name, *, dtype=None):
    array = np.asarray(operator, dtype=dtype)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square rank-2 matrix.")
    return array


def _normalize_paulis(paulis, *, support):
    if isinstance(paulis, str):
        labels = tuple(paulis.upper())
    else:
        labels = tuple(str(label).upper() for label in paulis)
    expected = 1 if support == "onsite" else 2
    if len(labels) != expected:
        raise ValueError(
            f"{support} Pauli terms need {expected} label(s), got {len(labels)}."
        )
    if any(label not in _PAULI_LABELS for label in labels):
        raise ValueError(
            "Pauli labels must be drawn from 'I', 'X', 'Y', and 'Z'."
        )
    return labels


def _normalize_pauli_support(support):
    aliases = {
        "site": "onsite",
        "one_site": "onsite",
        "one-site": "onsite",
        "onsite": "onsite",
        "bond": "edge",
        "two_site": "edge",
        "two-site": "edge",
        "edge": "edge",
    }
    try:
        return aliases[str(support).lower()]
    except KeyError as exc:
        raise ValueError("PauliPEPOTerm support must be 'onsite' or 'edge'.") from exc


def _normalize_pauli_term(term):
    if isinstance(term, PauliPEPOTerm):
        return term
    if isinstance(term, Mapping):
        support = term.get("support", term.get("kind", term.get("type")))
        paulis = term.get("paulis", term.get("operators"))
        if support is None or paulis is None:
            raise ValueError(
                "each Pauli PEPO term needs support and paulis entries."
            )
        return PauliPEPOTerm(
            support,
            paulis,
            term.get("coefficient", 1.0),
        )
    if isinstance(term, (tuple, list)) and len(term) in (2, 3):
        return PauliPEPOTerm(
            term[0],
            term[1],
            term[2] if len(term) == 3 else 1.0,
        )
    raise TypeError(
        "Pauli PEPO terms must be PauliPEPOTerm values, mappings, or "
        "(support, paulis[, coefficient]) tuples."
    )


def _backend_kron_all(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = ar.do("kron", result, matrix)
    return result


def _backend_expm(matrix):
    """Evaluate a local exponential while retaining Torch/JAX graphs."""
    backend = ar.infer_backend(matrix)
    if backend in ("builtins", "numpy"):
        matrix = np.asarray(matrix)
        if matrix.ndim == 2:
            return np.asarray(quimb.expm(matrix), dtype=matrix.dtype)
        if matrix.ndim == 3:
            return np.stack(
                [quimb.expm(local_matrix) for local_matrix in matrix],
                axis=0,
            ).astype(matrix.dtype, copy=False)
        raise ValueError("local matrix exponentials must be rank two or three.")
    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        return torch.matrix_exp(matrix)
    if backend == "jax":
        import jax.scipy.linalg as jsp  # pylint: disable=import-outside-toplevel

        return jsp.expm(matrix)
    raise TypeError(
        "PauliPEPOBasis does not have a matrix-exponential rule for backend "
        f"{backend!r}."
    )


def _complexify_backend(value):
    """Promote real backend values before contraction with complex Paulis."""
    return ar.do(
        "multiply",
        value,
        _as_backend(1.0 + 0.0j, like=value),
    )


def _backend_operator_product(factors):
    """Kronecker multiply local operators and return row/column axes grouped."""
    if len(factors) == 1:
        return factors[0]
    local_dim = int(factors[0].shape[0])
    result = _backend_kron_all(*factors)
    nsites = len(factors)
    # A Kronecker product of matrices already has grouped row axes followed
    # by grouped column axes when reshaped this way.
    return ar.do(
        "reshape",
        result,
        (local_dim**nsites, local_dim**nsites),
    )


def _plaquette_edges():
    """Return the canonical four-edge square plaquette topology."""
    return ((0, 1, "u"), (0, 2, "r"), (2, 3, "u"), (1, 3, "r"))


def _backend_embed_operator(operator, positions, nsites, local_dim):
    """Embed a small operator on ordered ``positions`` in a cluster."""
    positions = tuple(positions)
    if len(set(positions)) != len(positions):
        raise ValueError("operator positions must be distinct.")
    other_positions = tuple(site for site in range(nsites) if site not in positions)
    identity = ar.do("eye", local_dim, like=operator)
    factors = [operator, *([identity] * len(other_positions))]
    result = _backend_kron_all(*factors)
    result = ar.do("reshape", result, (local_dim,) * (2 * nsites))
    factor_order = positions + other_positions
    row_axes = tuple(factor_order.index(site) for site in range(nsites))
    axes = row_axes + tuple(nsites + axis for axis in row_axes)
    return ar.do("transpose", result, axes).reshape(
        local_dim**nsites,
        local_dim**nsites,
    )


def _backend_swap_two_site(operator, local_dim):
    reshaped = ar.do(
        "reshape",
        operator,
        (local_dim, local_dim, local_dim, local_dim),
    )
    reshaped = ar.do("transpose", reshaped, (1, 0, 3, 2))
    return ar.do("reshape", reshaped, (local_dim**2, local_dim**2))


def _backend_pauli_basis(nsites, *, like=None):
    """Return the physical Pauli basis, reusing its static NumPy layout."""
    try:
        matrices = _PAULI_BASIS_CACHE[nsites]
    except KeyError:
        matrices = np.stack(
            [
                _backend_operator_product(
                    [_PAULI_MATRICES[label] for label in labels]
                )
                for labels in product(_PAULI_LABELS, repeat=nsites)
            ],
            axis=0,
        )
        _PAULI_BASIS_CACHE[nsites] = matrices
    if like is None:
        return tuple(matrices)
    return tuple(_as_backend(matrix, like=like) for matrix in matrices)


def _backend_pauli_expand(operator, nsites):
    """Expand a local operator in the fixed physical Pauli basis."""
    basis = ar.do(
        "stack",
        _backend_pauli_basis(nsites, like=operator),
        axis=0,
    )
    coefficients = ar.do(
        "tensordot",
        ar.do("conj", basis),
        operator,
        axes=([1, 2], [0, 1]),
    )
    coefficients = ar.do("divide", coefficients, 2**nsites)
    return ar.do("reshape", coefficients, (4,) * nsites)


def _backend_sum_pauli(coefficients, axis):
    """Return ``sum_p coefficients[p] * P_p`` for one physical site."""
    del axis  # retained for the old helper signature
    basis = ar.do(
        "stack",
        _backend_pauli_basis(1, like=coefficients),
        axis=0,
    )
    return ar.do("tensordot", coefficients, basis, axes=([0], [0]))


def _backend_stack(values, *, axis=0):
    if len(values) == 1:
        return ar.do("expand_dims", values[0], axis=axis)
    return ar.do("stack", tuple(values), axis=axis)


def _expm(matrix, dtype):
    """Evaluate a dense exponential without making SciPy a hard dependency."""
    return np.asarray(quimb.expm(np.asarray(matrix, dtype=dtype)), dtype=dtype)


def _kron_all(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def _operator_tensor(operator, nsites, local_dim):
    """View a dense cluster operator as one flattened matrix per site."""
    axes = tuple(axis for site in range(nsites) for axis in (site, nsites + site))
    return operator.reshape((local_dim,) * (2 * nsites)).transpose(axes).reshape(
        (local_dim**2,) * nsites
    )


def _backend_operator_tensor(operator, nsites, local_dim):
    """Backend-preserving counterpart of :func:`_operator_tensor`."""
    axes = tuple(axis for site in range(nsites) for axis in (site, nsites + site))
    reshaped = ar.do("reshape", operator, (local_dim,) * (2 * nsites))
    grouped = ar.do("transpose", reshaped, axes)
    return ar.do("reshape", grouped, (local_dim**2,) * nsites)


def _operator_from_tensor(operator_tensor, nsites, local_dim):
    """Undo :func:`_operator_tensor` for a local operator tensor."""
    pair_axes = tuple(range(2 * nsites))
    row_axes = pair_axes[::2]
    column_axes = pair_axes[1::2]
    return operator_tensor.reshape((local_dim, local_dim) * nsites).transpose(
        row_axes + column_axes
    ).reshape(local_dim**nsites, local_dim**nsites)


def _backend_operator_from_tensor(operator_tensor, nsites, local_dim):
    """Backend-preserving counterpart of :func:`_operator_from_tensor`."""
    pair_axes = tuple(range(2 * nsites))
    row_axes = pair_axes[::2]
    column_axes = pair_axes[1::2]
    local = ar.do(
        "reshape",
        operator_tensor,
        (local_dim, local_dim) * nsites,
    )
    ordered = ar.do("transpose", local, row_axes + column_axes)
    return ar.do("reshape", ordered, (local_dim**nsites, local_dim**nsites))


def _permute_operator_sites(operator, order, local_dim):
    """Return ``operator`` with its site order changed by ``order``."""
    tensor = _operator_tensor(operator, len(order), local_dim)
    tensor = tensor.transpose(order)
    return _operator_from_tensor(tensor, len(order), local_dim)


def _cycle_active_operator(blocks, site_directions, loop_sites, physical_dim):
    """Contract the existing active blocks around one four-site cycle.

    The dense plan shares tree channel ranges so some lower-order products can
    already close around a plaquette.  This helper measures that contribution
    before adding the explicit loop sector; it prevents double counting while
    retaining the existing tree-channel compatibility contract.
    """
    pairs = (("r", "u"), ("d", "r"), ("l", "d"), ("u", "l"))
    active = []
    for site, pair in zip(loop_sites, pairs):
        directions = site_directions[site]
        pair_indices = tuple(directions.index(direction) for direction in pair)
        entries = {}
        for key, block in blocks[site].items():
            if any(
                key[index]
                for index, direction in enumerate(directions)
                if direction not in pair
            ):
                continue
            pair_key = tuple(key[index] for index in pair_indices)
            entries[pair_key] = (
                entries[pair_key] + block
                if pair_key in entries
                else block
            )
        active.append(entries)

    result = np.zeros(
        (physical_dim**4, physical_dim**4),
        dtype=np.result_type(
            *(block for entries in active for block in entries.values())
        ),
    )
    for (right, upper), first in active[0].items():
        for (down, diagonal), second in active[1].items():
            if down != upper:
                continue
            for (upper_again, right_again), third in active[2].items():
                if upper_again != diagonal:
                    continue
                fourth = active[3].get((right_again, right))
                if fourth is None:
                    continue
                result += _kron_all(first, second, third, fourth)
    return result


def _swap_two_site_operator(operator, local_dim):
    return operator.reshape(local_dim, local_dim, local_dim, local_dim).transpose(
        1, 0, 3, 2
    ).reshape(local_dim**2, local_dim**2)


def _oriented_two_site_operator(operator, local_dim, direction):
    if direction in _POSITIVE_DIRECTIONS:
        return operator
    return _swap_two_site_operator(operator, local_dim)


def _embed_one_site_operator(operator, position, nsites, local_dim):
    identity = np.eye(local_dim, dtype=operator.dtype)
    factors = [identity] * nsites
    factors[position] = operator
    return _kron_all(*factors)


def _embed_two_site_operator(operator, positions, nsites, local_dim):
    """Embed a rank-two-site matrix, preserving the supplied site ordering."""
    pos0, pos1 = positions
    dimension = local_dim**nsites
    result = np.zeros((dimension, dimension), dtype=operator.dtype)
    for row in np.ndindex(*(local_dim for _ in range(nsites))):
        row_flat = np.ravel_multi_index(row, (local_dim,) * nsites)
        for col in np.ndindex(*(local_dim for _ in range(nsites))):
            if any(row[pos] != col[pos] for pos in range(nsites) if pos not in positions):
                continue
            col_flat = np.ravel_multi_index(col, (local_dim,) * nsites)
            row_pair = row[pos0] * local_dim + row[pos1]
            col_pair = col[pos0] * local_dim + col[pos1]
            result[row_flat, col_flat] = operator[row_pair, col_pair]
    return result


def _tree_hamiltonian(nsites, edges, twosite_op, onesite_op):
    """Build the dense Hamiltonian on a small embedded tree cluster."""
    local_dim = onesite_op.shape[0]
    result = sum(
        (
            _embed_one_site_operator(onesite_op, position, nsites, local_dim)
            for position in range(nsites)
        ),
        start=np.zeros((local_dim**nsites, local_dim**nsites), dtype=onesite_op.dtype),
    )
    for source, target, direction in edges:
        result += _embed_two_site_operator(
            _oriented_two_site_operator(twosite_op, local_dim, direction),
            (source, target),
            nsites,
            local_dim,
        )
    return result


def _edge_contribution(nsites, edges, one_site_exp, start_factors, end_factors):
    """Evaluate all two-site PEPO channels on a small tree."""
    if not start_factors.shape[0]:
        return np.zeros_like(
            _kron_all(*(one_site_exp for _ in range(nsites)))
        )

    result = np.zeros(
        (one_site_exp.shape[0] ** nsites, one_site_exp.shape[1] ** nsites),
        dtype=np.result_type(one_site_exp, start_factors, end_factors),
    )
    for source, target, direction in edges:
        source_factors = start_factors if direction in _POSITIVE_DIRECTIONS else end_factors
        target_factors = end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for index in range(start_factors.shape[0]):
            factors = [one_site_exp] * nsites
            factors[source] = source_factors[index]
            factors[target] = target_factors[index]
            result += _kron_all(*factors)
    return result


def _three_cluster_contribution(
    nsites,
    center,
    branches,
    pair_tensor,
    start_factors,
    end_factors,
    one_site_exp,
):
    """Evaluate one connected three-site correction inside a larger tree."""
    if not pair_tensor.size:
        return np.zeros_like(_kron_all(*(one_site_exp for _ in range(nsites))))

    result = np.zeros(
        (one_site_exp.shape[0] ** nsites, one_site_exp.shape[1] ** nsites),
        dtype=np.result_type(one_site_exp, pair_tensor),
    )
    directions = tuple(direction for _, direction in branches)
    order = np.argsort([_DIRECTIONS.index(direction) for direction in directions])
    ordered_branches = tuple(branches[index] for index in order)
    for first in range(pair_tensor.shape[0]):
        for second in range(pair_tensor.shape[1]):
            factors = [one_site_exp] * nsites
            factors[center] = pair_tensor[first, second]
            for channel, (endpoint, direction) in enumerate(ordered_branches):
                endpoint_factors = (
                    end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
                )
                factors[endpoint] = endpoint_factors[
                    (first, second)[channel]
                ]
            result += _kron_all(*factors)
    return result


def _three_subtrees(edges):
    """Yield the two-edge connected subtrees of a four-site tree."""
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1 :]:
            first_vertices = {first[0], first[1]}
            second_vertices = {second[0], second[1]}
            common = first_vertices & second_vertices
            if len(common) != 1:
                continue
            center = next(iter(common))
            branches = []
            for source, target, direction in (first, second):
                if source == center:
                    branches.append((target, direction))
                else:
                    branches.append((source, _OPPOSITE_DIRECTION[direction]))
            yield center, tuple(branches)


def _pair_pair_contribution(
    edges,
    pair_tensors,
    start_factors,
    end_factors,
    one_site_exp,
):
    """Evaluate the lower-order product of two adjacent pair-center blocks."""
    adjacency = {index: [] for index in range(4)}
    for source, target, direction in edges:
        adjacency[source].append((target, direction))
        adjacency[target].append((source, _OPPOSITE_DIRECTION[direction]))
    endpoints = [vertex for vertex, links in adjacency.items() if len(links) == 1]
    if len(endpoints) != 2 or any(len(links) > 2 for links in adjacency.values()):
        return np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))

    path = [endpoints[0]]
    steps = []
    previous = None
    while len(path) < 4:
        current = path[-1]
        next_vertex, direction = next(
            link for link in adjacency[current] if link[0] != previous
        )
        steps.append(direction)
        path.append(next_vertex)
        previous = current

    first_factors = start_factors if steps[0] in _POSITIVE_DIRECTIONS else end_factors
    last_factors = end_factors if steps[2] in _POSITIVE_DIRECTIONS else start_factors
    rank = start_factors.shape[0]
    result = np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))

    def pair_entry(first_direction, second_direction, first, second):
        pair = tuple(
            sorted((first_direction, second_direction), key=_DIRECTIONS.index)
        )
        tensor = pair_tensors[pair]
        indices = {
            first_direction: first,
            second_direction: second,
        }
        return tensor[indices[pair[0]], indices[pair[1]]]

    del path  # only the ordered edge directions are needed below
    first_back = _OPPOSITE_DIRECTION[steps[0]]
    first_forward = steps[1]
    second_back = _OPPOSITE_DIRECTION[steps[1]]
    second_forward = steps[2]
    for first in range(rank):
        for middle in range(rank):
            for last in range(rank):
                result += _kron_all(
                    first_factors[first],
                    pair_entry(
                        first_back,
                        first_forward,
                        first,
                        middle,
                    ),
                    pair_entry(
                        second_back,
                        second_forward,
                        middle,
                        last,
                    ),
                    last_factors[last],
                )
    return result


def _disconnected_edge_product(edges, start_factors, end_factors, one_site_exp):
    """Evaluate products of disjoint two-site residual channels."""
    result = np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))
    for first_index, first_edge in enumerate(edges):
        first_vertices = {first_edge[0], first_edge[1]}
        for second_edge in edges[first_index + 1 :]:
            if first_vertices & {second_edge[0], second_edge[1]}:
                continue
            first_source, first_target, first_direction = first_edge
            second_source, second_target, second_direction = second_edge
            first_source_factors = (
                start_factors
                if first_direction in _POSITIVE_DIRECTIONS
                else end_factors
            )
            first_target_factors = (
                end_factors
                if first_direction in _POSITIVE_DIRECTIONS
                else start_factors
            )
            second_source_factors = (
                start_factors
                if second_direction in _POSITIVE_DIRECTIONS
                else end_factors
            )
            second_target_factors = (
                end_factors
                if second_direction in _POSITIVE_DIRECTIONS
                else start_factors
            )
            for first_channel in range(start_factors.shape[0]):
                for second_channel in range(start_factors.shape[0]):
                    factors = [one_site_exp] * 4
                    factors[first_source] = first_source_factors[first_channel]
                    factors[first_target] = first_target_factors[first_channel]
                    factors[second_source] = second_source_factors[second_channel]
                    factors[second_target] = second_target_factors[second_channel]
                    result += _kron_all(*factors)
    return result


def _lower_tree_residual(
    nsites,
    edges,
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors=None,
):
    """Return a tree residual after subtracting all lower connected terms."""
    exact = _expm(
        -beta * _tree_hamiltonian(nsites, edges, twosite_op, onesite_op),
        onesite_op.dtype,
    )
    residual = exact - _kron_all(*(one_site_exp for _ in range(nsites)))
    residual -= _edge_contribution(
        nsites, edges, one_site_exp, start_factors, end_factors
    )
    if pair_tensors:
        for center, branches in _three_subtrees(edges):
            directions = tuple(direction for _, direction in branches)
            pair = tuple(sorted(directions, key=_DIRECTIONS.index))
            residual -= _three_cluster_contribution(
                nsites,
                center,
                branches,
                pair_tensors[pair],
                start_factors,
                end_factors,
                one_site_exp,
            )
        if nsites == 4 and len(edges) == 3:
            residual -= _disconnected_edge_product(
                edges,
                start_factors,
                end_factors,
                one_site_exp,
            )
            residual -= _pair_pair_contribution(
                edges,
                pair_tensors,
                start_factors,
                end_factors,
                one_site_exp,
            )
    return exact, residual


def _lower_loop_residual(
    edges,
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
):
    """Return a four-site plaquette residual after all lower clusters.

    A plaquette has four connected three-site subtrees and two disconnected
    opposite-edge products.  The latter is the only lower-order subtraction
    that differs from the four-site tree residual used by
    :func:`_lower_tree_residual`; no four-site path is embedded on a plaquette
    because the path endpoints are already nearest neighbours.
    """
    exact = _expm(
        -beta * _tree_hamiltonian(4, edges, twosite_op, onesite_op),
        onesite_op.dtype,
    )
    residual = exact - _kron_all(*(one_site_exp for _ in range(4)))
    residual -= _edge_contribution(
        4, edges, one_site_exp, start_factors, end_factors
    )
    for center, branches in _three_subtrees(edges):
        directions = tuple(direction for _, direction in branches)
        pair = tuple(sorted(directions, key=_DIRECTIONS.index))
        residual -= _three_cluster_contribution(
            4,
            center,
            branches,
            pair_tensors[pair],
            start_factors,
            end_factors,
            one_site_exp,
        )
    residual -= _disconnected_edge_product(
        edges,
        start_factors,
        end_factors,
        one_site_exp,
    )
    return exact, residual


def _edge_factors(
    twosite_op,
    onesite_op,
    beta,
    *,
    edge_cutoff,
    max_edge_rank,
    symmetric=False,
):
    """Return the two endpoint factors of the connected two-site residual."""
    local_dim = onesite_op.shape[0]
    identity = np.eye(local_dim, dtype=onesite_op.dtype)
    one_site_exp = _expm(-beta * onesite_op, onesite_op.dtype)
    two_site_hamiltonian = (
        twosite_op
        + np.kron(onesite_op, identity)
        + np.kron(identity, onesite_op)
    )
    residual = _expm(-beta * two_site_hamiltonian, onesite_op.dtype) - np.kron(
        one_site_exp, one_site_exp
    )

    residual = residual.reshape(local_dim, local_dim, local_dim, local_dim).transpose(
        0, 2, 1, 3
    ).reshape(local_dim**2, local_dim**2)
    if symmetric:
        if not np.allclose(residual, residual.T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "C4 cluster reduction requires a site-exchange-symmetric "
                "two-site residual."
            )
        if not np.allclose(residual, residual.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "C4 cluster reduction currently requires a real/Hermitian "
                "reshuffled two-site residual."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        ordering = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[ordering]
        eigenvectors = eigenvectors[:, ordering]
        singular_values = np.abs(eigenvalues)
    else:
        left, singular_values, right = np.linalg.svd(residual, full_matrices=False)
    if not singular_values.size or singular_values[0] == 0:
        empty = np.zeros((0, local_dim, local_dim), dtype=onesite_op.dtype)
        return one_site_exp, empty, empty

    edge_cutoff = float(edge_cutoff)
    threshold = singular_values[0] * (
        edge_cutoff
        if edge_cutoff > 0.0
        else np.finfo(singular_values.dtype).eps * local_dim**2
    )
    keep = np.flatnonzero(singular_values > threshold)
    if max_edge_rank is not None:
        keep = keep[:max_edge_rank]
    singular_values = singular_values[keep]
    if not singular_values.size:
        empty = np.zeros((0, local_dim, local_dim), dtype=onesite_op.dtype)
        return one_site_exp, empty, empty

    if symmetric:
        root = np.sqrt(eigenvalues[keep].astype(np.result_type(onesite_op.dtype, complex)))
        factors = (eigenvectors[:, keep] * root).T.reshape(-1, local_dim, local_dim)
        return one_site_exp, factors, factors.copy()

    root = np.sqrt(singular_values)
    start = (left[:, keep] * root).T.reshape(-1, local_dim, local_dim)
    end = (root[:, None] * right[keep, :]).reshape(-1, local_dim, local_dim)
    return one_site_exp, start, end


def _edge_fit_residual(
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
):
    """Return the unrepresented two-site residual after channel truncation."""
    local_dim = onesite_op.shape[0]
    identity = np.eye(local_dim, dtype=onesite_op.dtype)
    hamiltonian = (
        twosite_op
        + np.kron(onesite_op, identity)
        + np.kron(identity, onesite_op)
    )
    exact = _expm(-beta * hamiltonian, onesite_op.dtype)
    residual = exact - np.kron(one_site_exp, one_site_exp)
    for index in range(start_factors.shape[0]):
        residual -= np.kron(start_factors[index], end_factors[index])
    return residual


def _solve_three_site_pair(
    pair,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    beta,
):
    """Solve the residual for one center with two active virtual directions."""
    local_dim = onesite_op.shape[0]
    rank = start_factors.shape[0]

    edges = ((0, 1, pair[0]), (0, 2, pair[1]))
    _, residual = _lower_tree_residual(
        3,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
    )

    endpoint_factors = [
        end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for direction in pair
    ]
    # Project each endpoint directly onto its operator-Schmidt space.  The
    # old implementation built a ``d**6 x (r**2 * d**2)`` Kronecker design
    # and solved it as one dense system.  That is mathematically equivalent,
    # but it makes a small four-site cluster pay for a much larger LAPACK
    # solve.  These two pseudoinverses are only ``d**2 x r`` and preserve the
    # least-squares behavior when an endpoint channel basis is truncated.
    endpoint_matrices = tuple(
        factors.reshape(rank, local_dim**2).T for factors in endpoint_factors
    )
    endpoint_pseudoinverses = tuple(
        np.linalg.pinv(factors) for factors in endpoint_matrices
    )
    residual_tensor = _operator_tensor(residual, 3, local_dim)
    center_coefficients = np.einsum(
        "ijk,aj,bk->iab",
        residual_tensor,
        endpoint_pseudoinverses[0],
        endpoint_pseudoinverses[1],
    )
    solution = center_coefficients.transpose(1, 2, 0).reshape(
        rank, rank, local_dim, local_dim
    )
    fitted = np.einsum(
        "ia,jab,kb->ijk",
        endpoint_matrices[0],
        center_coefficients,
        endpoint_matrices[1],
    )
    return solution, np.linalg.norm(residual_tensor - fitted), np.linalg.norm(residual_tensor)


def _site_directions(i, j, lx, ly, cyclic_x, cyclic_y):
    return tuple(
        direction
        for direction, present in (
            ("u", cyclic_x or i < lx - 1),
            ("r", cyclic_y or j < ly - 1),
            ("d", cyclic_x or i > 0),
            ("l", cyclic_y or j > 0),
        )
        if present
    )


@dataclass(frozen=True)
class ClusterExpansionReport:
    """Numerical and storage diagnostics from one cluster-expansion build.

    ``residual_norms`` report the largest local residual left after the
    corresponding factorization or least-squares solve. They are local
    operator norms in the dense Frobenius metric, not a global PEPO error
    bound. ``cluster_counts`` count embedded cluster instances on the chosen
    finite lattice; the separate ``*_solved`` entries expose C4 reuse.
    """

    beta: object
    order: int
    local_dim: int
    edge_rank: int
    tree_rank: int
    loop_rank: int
    cluster_counts: dict[str, int]
    residual_norms: dict[str, float]
    relative_residual_norms: dict[str, float]
    active_block_count: int
    active_nbytes: int
    dense_nbytes: int
    generic_loop_rank: int = 0

    @property
    def max_residual_norm(self):
        """Return the largest reported local residual."""
        return max(self.residual_norms.values(), default=0.0)

    @property
    def max_relative_residual(self):
        """Return the largest residual relative to its uncompressed target."""
        return max(self.relative_residual_norms.values(), default=0.0)

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="pepo",
            algorithm="cluster_expansion",
            representation="active_pepo",
            order=self.order,
        )


@dataclass(frozen=True)
class ClusterModelAdapter:
    """Dense local-term adapter for finite cluster-expansion PEPOs.

    The adapter is deliberately small: it translates a translation-invariant
    square-lattice model into the two matrices consumed by
    :class:`ClusterExpansionPlan`.  This mirrors the Julia workflow where a
    model supplies local and nearest-neighbour terms while the cluster engine
    owns geometry, residual subtraction, and PEPO factorization.

    Parameters
    ----------
    twosite_op, onesite_op : array-like
        Dense matrices with shapes ``(d**2, d**2)`` and ``(d, d)``.
    name : str, optional
        Human-readable model name used in diagnostics.
    symmetry : {None, "C4"}, optional
        Finite square-lattice symmetry that is safe for this model.  ``C4``
        requires a site-symmetric two-site term and enables finite orbit
        reuse in the generic cluster solver.

    Notes
    -----
    This is a dense finite adapter.  It intentionally does not carry
    fermionic parity data or native Symmray charge sectors.
    """

    twosite_op: object
    onesite_op: object
    name: str = "custom"
    symmetry: str | None = None
    internal_symmetry: ClusterInternalSymmetry | None = None

    def __post_init__(self):
        twosite_op = _as_square_operator(self.twosite_op, "twosite_op")
        onesite_op = _as_square_operator(self.onesite_op, "onesite_op")
        if twosite_op.shape != (onesite_op.shape[0] ** 2,) * 2:
            raise ValueError(
                "twosite_op must have shape "
                f"({onesite_op.shape[0] ** 2}, {onesite_op.shape[0] ** 2}) "
                f"for local dimension {onesite_op.shape[0]}."
            )
        if self.symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")
        if self.symmetry == "C4":
            swapped = _swap_two_site_operator(twosite_op, onesite_op.shape[0])
            if not np.allclose(twosite_op, swapped, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    "C4 cluster reduction requires a site-symmetric twosite_op."
                )
        object.__setattr__(self, "twosite_op", twosite_op)
        object.__setattr__(self, "onesite_op", onesite_op)
        object.__setattr__(self, "name", str(self.name))
        internal_symmetry = _coerce_internal_symmetry(self.internal_symmetry)
        object.__setattr__(self, "internal_symmetry", internal_symmetry)
        if internal_symmetry is not None:
            internal_symmetry.validate(twosite_op, onesite_op)

    @property
    def local_dim(self):
        """Return the one-site Hilbert-space dimension."""
        return self.onesite_op.shape[0]

    @classmethod
    def custom(
        cls,
        twosite_op,
        onesite_op,
        *,
        name="custom",
        symmetry=None,
        internal_symmetry=None,
    ):
        """Adapt already-assembled dense local and edge matrices."""
        return cls(
            twosite_op,
            onesite_op,
            name=name,
            symmetry=symmetry,
            internal_symmetry=internal_symmetry,
        )

    @classmethod
    def ising(
        cls,
        *,
        J=1.0,
        field=1.0,
        field_axis="x",
        dtype=None,
        symmetry="C4",
    ):
        """Build the transverse-field Ising adapter.

        The convention is ``H = J sum Z_i Z_j + field sum sigma_i`` with
        ``field_axis="x"`` by default, matching Pepsy's ITF helper.
        """
        dtype = _model_dtype(dtype, J, field)
        paulis = _model_paulis(dtype)
        axis = _model_axis(field_axis)
        return cls(
            J * np.kron(paulis["z"], paulis["z"]),
            field * paulis[axis],
            name="ising",
            symmetry=symmetry,
        )

    @classmethod
    def heisenberg(
        cls,
        *,
        J=1.0,
        field=0.0,
        field_axis="z",
        dtype=None,
        symmetry="C4",
    ):
        """Build a spin-1/2 Heisenberg adapter.

        The convention uses ``S^a = sigma^a / 2`` and
        ``H = J sum_a S^a_i S^a_j + field sum S^axis_i``.
        """
        dtype = _model_dtype(dtype, J, field)
        paulis = _model_paulis(dtype)
        spin = {axis: matrix / 2 for axis, matrix in paulis.items()}
        axis = _model_axis(field_axis)
        twosite = J * sum(
            (
                np.kron(spin[component], spin[component])
                for component in ("x", "y", "z")
            ),
            start=np.zeros((4, 4), dtype=dtype),
        )
        return cls(
            twosite,
            field * spin[axis],
            name="heisenberg",
            symmetry=symmetry,
        )

    @classmethod
    def xxz(
        cls,
        *,
        Jxy=1.0,
        Jz=1.0,
        field=0.0,
        field_axis="z",
        dtype=None,
        symmetry="C4",
    ):
        """Build a spin-1/2 XXZ adapter with an optional onsite field."""
        dtype = _model_dtype(dtype, Jxy, Jz, field)
        paulis = _model_paulis(dtype)
        spin = {axis: matrix / 2 for axis, matrix in paulis.items()}
        axis = _model_axis(field_axis)
        twosite = (
            Jxy * (
                np.kron(spin["x"], spin["x"])
                + np.kron(spin["y"], spin["y"])
            )
            + Jz * np.kron(spin["z"], spin["z"])
        )
        return cls(
            twosite,
            field * spin[axis],
            name="xxz",
            symmetry=symmetry,
        )

    @classmethod
    def from_model(
        cls,
        model,
        *,
        name=None,
        symmetry=None,
        internal_symmetry=None,
    ):
        """Adapt a mapping or object exposing local cluster terms.

        Accepted mappings use ``twosite_op``/``onesite_op`` or the aliases
        ``edge_op``/``onsite_op``.  Objects may expose those attributes or a
        zero-argument ``cluster_terms()`` method returning such a mapping.
        """
        if isinstance(model, cls):
            if name is None and symmetry is None and internal_symmetry is None:
                return model
            return cls(
                model.twosite_op,
                model.onesite_op,
                name=model.name if name is None else name,
                symmetry=model.symmetry if symmetry is None else symmetry,
                internal_symmetry=(
                    model.internal_symmetry
                    if internal_symmetry is None
                    else internal_symmetry
                ),
            )
        if hasattr(model, "cluster_terms"):
            model = model.cluster_terms()
        if isinstance(model, Mapping):
            twosite_op = model.get("twosite_op", model.get("edge_op"))
            onesite_op = model.get("onesite_op", model.get("onsite_op"))
            model_name = model.get("name", "custom")
            model_symmetry = model.get("symmetry")
            model_internal_symmetry = model.get("internal_symmetry")
        else:
            twosite_op = getattr(model, "twosite_op", getattr(model, "edge_op", None))
            onesite_op = getattr(model, "onesite_op", getattr(model, "onsite_op", None))
            model_name = getattr(model, "name", "custom")
            model_symmetry = getattr(model, "symmetry", None)
            model_internal_symmetry = getattr(model, "internal_symmetry", None)
        if twosite_op is None or onesite_op is None:
            raise TypeError(
                "model must expose twosite_op/onesite_op (or edge_op/onsite_op), "
                "or return them from cluster_terms()."
            )
        return cls(
            twosite_op,
            onesite_op,
            name=model_name if name is None else name,
            symmetry=model_symmetry if symmetry is None else symmetry,
            internal_symmetry=(
                model_internal_symmetry
                if internal_symmetry is None
                else internal_symmetry
            ),
        )

    def build(self, lx, ly, beta, **kwargs):
        """Build a finite PEPO using this adapter's local terms."""
        kwargs.setdefault("symmetry", self.symmetry)
        kwargs.setdefault("internal_symmetry", self.internal_symmetry)
        return build_cluster_expansion_pepo(
            lx,
            ly,
            beta,
            self.twosite_op,
            self.onesite_op,
            **kwargs,
        )


def _model_dtype(dtype, *values):
    """Choose a safe dense dtype for a model factory."""
    if dtype is None:
        return np.result_type(*values, np.float64)
    return np.result_type(np.dtype(dtype), *values)


def _model_paulis(dtype):
    """Return Pauli matrices in the requested model dtype."""
    return {
        "x": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=dtype),
        "y": np.asarray(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=np.result_type(dtype, np.complex128),
        ),
        "z": np.asarray(np.diag([1.0, -1.0]), dtype=dtype),
    }


def _model_axis(axis):
    axis = str(axis).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("field_axis must be one of 'x', 'y', or 'z'.")
    return axis


def adapt_cluster_model(
    model,
    *,
    name=None,
    symmetry=None,
    internal_symmetry=None,
):
    """Return a :class:`ClusterModelAdapter` for a dense local-term model."""
    return ClusterModelAdapter.from_model(
        model,
        name=name,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
    )


def _assemble_coefficient_terms(terms, name):
    """Assemble ``[(coefficient, operator), ...]`` into one dense matrix."""
    if terms is None:
        raise ValueError(f"{name} must contain at least one local term.")
    try:
        candidate = np.asarray(terms)
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None and candidate.ndim == 2:
        return _as_square_operator(candidate, name)

    if isinstance(terms, Mapping):
        terms = tuple(terms.values())
    elif isinstance(terms, (tuple, list)) and len(terms) == 2:
        try:
            is_single_pair = (
                np.asarray(terms[1]).ndim == 2
                and np.asarray(terms[0]).ndim == 0
            )
        except (TypeError, ValueError):
            is_single_pair = False
        if is_single_pair:
            terms = (terms,)
    else:
        try:
            terms = tuple(terms)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a square matrix or an iterable of "
                "(coefficient, operator) terms."
            ) from exc

    parsed = []
    for term in terms:
        if isinstance(term, Mapping):
            coefficient = term.get("coefficient", term.get("coeff"))
            operator = term.get("operator", term.get("term"))
        else:
            try:
                coefficient, operator = term
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"each {name} entry must be (coefficient, operator)."
                ) from exc
        if coefficient is None or operator is None:
            raise ValueError(
                f"each {name} entry needs 'coefficient' and 'operator' values."
            )
        operator = _as_square_operator(operator, f"{name} operator")
        parsed.append((coefficient, operator))
    if not parsed:
        raise ValueError(f"{name} must contain at least one local term.")
    shape = parsed[0][1].shape
    if any(operator.shape != shape for _, operator in parsed[1:]):
        raise ValueError(f"all {name} operators must have the same shape.")
    dtype = np.result_type(*(operator.dtype for _, operator in parsed), *(
        np.asarray(coefficient).dtype for coefficient, _ in parsed
    ))
    result = np.zeros(shape, dtype=dtype)
    for coefficient, operator in parsed:
        result = result + coefficient * operator
    return result


def build_real_time_cluster_expansion_pepo(
    lx,
    ly,
    time,
    twosite_terms,
    onesite_terms,
    *,
    order=5,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    max_loop_rank=None,
    dtype=None,
    symmetry=None,
    internal_symmetry=None,
    fit_method="quimb",
    fit_steps=100,
    fit_tol=1e-10,
    fit_solver_maxiter=8,
    fit_seed=0,
    adaptive_loop_rank=False,
    loop_rank_start=None,
    loop_rank_step=2,
    fit_warm_start=True,
    materialize=True,
    return_report=False,
):
    """Build a numerical PEPO for ``exp(-1j * time * H(coefficients))``.

    ``twosite_terms`` and ``onesite_terms`` can each be an already assembled
    dense matrix or an iterable of ``(coefficient, operator)`` pairs. Mapping
    entries with ``coefficient``/``operator`` (or ``coeff``/``term``) fields
    are also accepted. The local coefficient sums are formed before the
    cluster exponentials are evaluated, so this represents the exponential
    of the sum rather than a Trotter product of local exponentials.

    This is a dense numerical path. Quimb's tree/ALS fitting is not
    differentiable with respect to the coefficients; use
    :class:`PauliPEPOBasis` for the existing fixed-channel autodiff path.
    """
    twosite_op = _assemble_coefficient_terms(twosite_terms, "twosite_terms")
    onesite_op = _assemble_coefficient_terms(onesite_terms, "onesite_terms")
    if dtype is not None:
        twosite_op = np.asarray(twosite_op, dtype=dtype)
        onesite_op = np.asarray(onesite_op, dtype=dtype)
    return build_cluster_expansion_pepo(
        lx,
        ly,
        1j * time,
        twosite_op,
        onesite_op,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        max_loop_rank=max_loop_rank,
        dtype=dtype,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
        fit_method=fit_method,
        fit_steps=fit_steps,
        fit_tol=fit_tol,
        fit_solver_maxiter=fit_solver_maxiter,
        fit_seed=fit_seed,
        adaptive_loop_rank=adaptive_loop_rank,
        loop_rank_start=loop_rank_start,
        loop_rank_step=loop_rank_step,
        fit_warm_start=fit_warm_start,
        materialize=materialize,
        return_report=return_report,
    )


def _graph_hamiltonian(nsites, edges, twosite_op, onesite_op):
    """Build a dense Hamiltonian for a graph cluster."""
    local_dim = onesite_op.shape[0]
    result = sum(
        (
            _embed_one_site_operator(onesite_op, site, nsites, local_dim)
            for site in range(nsites)
        ),
        start=np.zeros((local_dim**nsites, local_dim**nsites), dtype=onesite_op.dtype),
    )
    for source, target, _edge_index in edges:
        result += _embed_two_site_operator(
            twosite_op,
            (source, target),
            nsites,
            local_dim,
        )
    return result


def _contract_graph_active_support(active, sites, edges):
    """Contract active graph blocks on one connected support."""
    sites = tuple(sites)
    internal_edges = {edge_index for _, _, edge_index in edges}
    local_dim = active.physical_dim
    operands = []
    row_labels = []
    column_labels = []
    next_label = 0
    edge_labels = {edge_index: next_label + index for index, edge_index in enumerate(sorted(internal_edges))}
    next_label += len(edge_labels)
    for site in sites:
        row_label = next_label
        column_label = next_label + 1
        next_label += 2
        row_labels.append(row_label)
        column_labels.append(column_label)
        directions = active.site_directions[site]
        role_directions = tuple(
            edge_index for edge_index in directions if edge_index in internal_edges
        )
        shape = (active.bond_dim,) * len(role_directions) + (local_dim, local_dim)
        reference = next(iter(active.blocks[site].values()))
        factor = np.zeros(shape, dtype=np.asarray(reference).dtype)
        direction_axes = {edge_index: axis for axis, edge_index in enumerate(directions)}
        for key, block in active.blocks[site].items():
            if any(
                key[axis] != 0
                for edge_index, axis in direction_axes.items()
                if edge_index not in internal_edges
            ):
                continue
            sector_indices = tuple(
                key[direction_axes[edge_index]] for edge_index in role_directions
            )
            factor[sector_indices] += block
        labels = [edge_labels[edge_index] for edge_index in role_directions]
        labels.extend((row_label, column_label))
        operands.extend((factor, labels))
    result = np.einsum(*operands, row_labels + column_labels, optimize=True)
    return result.reshape(local_dim ** len(sites), local_dim ** len(sites))


def _graph_tree_factorize_operator(operator, edges, nsites, local_dim, max_rank):
    """Factor a graph-cluster operator over a spanning tree."""
    operator_tensor = _operator_tensor(operator, nsites, local_dim)
    operator_rank = local_dim**2
    adjacency = [[] for _ in range(nsites)]
    for source, target, edge_index in edges:
        adjacency[source].append((target, edge_index))
        adjacency[target].append((source, edge_index))

    parent = {0: None}
    parent_edge = {}
    traversal = [0]
    for site in traversal:
        for neighbour, edge_index in adjacency[site]:
            if neighbour in parent:
                continue
            parent[neighbour] = site
            parent_edge[neighbour] = edge_index
            traversal.append(neighbour)
    if len(parent) != nsites:
        raise ValueError("graph cluster must be connected for tree factorization.")

    children = {site: [] for site in range(nsites)}
    for site in traversal[1:]:
        children[parent[site]].append(site)

    current = operator_tensor
    axes = [("physical", site) for site in range(nsites)]
    local_tensors = {}
    ranks = {}
    for site in reversed(traversal[1:]):
        child_nodes = tuple(children[site])
        row_axes = [("bond", child) for child in child_nodes]
        row_axes.append(("physical", site))
        column_axes = [axis for axis in axes if axis not in row_axes]
        permutation = [axes.index(axis) for axis in row_axes + column_axes]
        matrix = current.transpose(permutation).reshape(
            int(np.prod([current.shape[index] for index in permutation[:len(row_axes)]])),
            -1,
        )
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if not singular_values.size or singular_values[0] == 0.0:
            return None, None, None, None, None, 0, 0.0, 0.0
        threshold = singular_values[0] * np.finfo(singular_values.dtype).eps * max(
            1, matrix.shape[0], matrix.shape[1]
        )
        keep = np.flatnonzero(singular_values > threshold)
        if max_rank is not None:
            keep = keep[:max_rank]
        if not keep.size:
            keep = np.array([0])
        rank = int(keep.size)
        local_tensors[site] = (
            child_nodes,
            left[:, keep].reshape(
                tuple(current.shape[axes.index(("bond", child))] for child in child_nodes)
                + (operator_rank, rank),
            ),
        )
        ranks[site] = rank
        current = (singular_values[keep, None] * right[keep, :]).reshape(
            (rank,) + tuple(current.shape[axes.index(axis)] for axis in column_axes)
        )
        axes = [("bond", site)] + column_axes

    root = 0
    root_axes = [("bond", child) for child in children[root]]
    root_axes.append(("physical", root))
    permutation = [axes.index(axis) for axis in root_axes]
    local_tensors[root] = (
        tuple(children[root]),
        current.transpose(permutation).reshape(
            tuple(current.shape[index] for index in permutation)
        ),
    )
    reconstructed = _contract_tree_factor_coefficients(
        local_tensors,
        parent,
        children,
        nsites,
    )
    return (
        local_tensors,
        parent,
        parent_edge,
        children,
        ranks,
        max(ranks.values(), default=0),
        float(np.linalg.norm(operator_tensor - reconstructed)),
        float(np.linalg.norm(operator_tensor)),
    )


def _add_graph_edge_blocks(blocks, site_directions, edge_index, source, target, start, end, sectors):
    """Insert one graph-edge channel at both endpoint tensors."""
    for site, factor in ((source, start), (target, end)):
        directions = site_directions[site]
        axis = directions.index(edge_index)
        for channel, sector in enumerate(sectors):
            key = [0] * len(directions)
            key[axis] = sector
            key = tuple(key)
            blocks[site][key] = blocks[site].get(key, 0) + factor[channel]


def _add_graph_tree_factor_blocks(
    blocks,
    site_directions,
    sites,
    edges,
    local_tensors,
    parent,
    parent_edge,
    sectors,
    local_dim,
):
    """Insert a spanning-tree factorization into graph PEPO blocks."""
    matrix_units = np.eye(local_dim**2).reshape(local_dim**2, local_dim, local_dim)
    root = next(site for site, value in parent.items() if value is None)
    for site, (child_nodes, tensor) in local_tensors.items():
        physical_axis = len(child_nodes)
        local_blocks = np.tensordot(
            tensor,
            matrix_units,
            axes=([physical_axis], [0]),
        )
        lattice_site = sites[site]
        directions = site_directions[lattice_site]
        for active_indices in np.ndindex(local_blocks.shape[:-2]):
            key = [0] * len(directions)
            for axis, child in enumerate(child_nodes):
                edge_index = parent_edge[child]
                key[directions.index(edge_index)] = sectors[edge_index][
                    active_indices[axis]
                ]
            if site != root:
                edge_index = parent_edge[site]
                key[directions.index(edge_index)] = sectors[edge_index][
                    active_indices[-1]
                ]
            key = tuple(key)
            block = local_blocks[active_indices]
            blocks[lattice_site][key] = blocks[lattice_site].get(key, 0) + block


class GraphClusterExpansionPlan:
    """Reusable arbitrary-graph cluster-expansion construction plan.

    The graph builder uses exact connected-subgraph residual subtraction and a
    spanning-tree factorization for every residual, including clusters whose
    induced graph contains loops.  This keeps the representation valid on
    triangular, honeycomb, Kagome, irregular, and user-supplied finite graphs;
    the tradeoff is that high-order graph residuals can require larger tree
    ranks than the specialized square implementation.
    """

    def __init__(
        self,
        lattice,
        twosite_op,
        onesite_op,
        *,
        order=3,
        edge_cutoff=0.0,
        max_edge_rank=None,
        max_tree_rank=None,
        internal_symmetry=None,
        dtype=None,
    ):
        if not isinstance(lattice, ClusterLattice):
            lattice = ClusterLattice.from_edges(lattice[0], lattice[1])
        self.lattice = lattice
        self.order = _validate_shape(order, "order")
        if self.order > 9:
            raise NotImplementedError(
                "graph cluster-expansion builders currently support orders 1 through 9."
            )
        if edge_cutoff < 0.0:
            raise ValueError("edge_cutoff must be >= 0.")
        self.edge_cutoff = float(edge_cutoff)
        self.max_edge_rank = None if max_edge_rank is None else _validate_shape(max_edge_rank, "max_edge_rank")
        self.max_tree_rank = None if max_tree_rank is None else _validate_shape(max_tree_rank, "max_tree_rank")
        self.onesite_op = _as_square_operator(onesite_op, "onesite_op", dtype=dtype)
        self.twosite_op = _as_square_operator(twosite_op, "twosite_op", dtype=dtype)
        local_dim = self.onesite_op.shape[0]
        if self.twosite_op.shape != (local_dim**2, local_dim**2):
            raise ValueError("twosite_op shape must be (local_dim**2, local_dim**2).")
        self.dtype = np.result_type(self.onesite_op.dtype, self.twosite_op.dtype)
        self.onesite_op = np.asarray(self.onesite_op, dtype=self.dtype)
        self.twosite_op = np.asarray(self.twosite_op, dtype=self.dtype)
        internal_symmetry = _coerce_internal_symmetry(internal_symmetry)
        if internal_symmetry is not None:
            internal_symmetry.validate(self.twosite_op, self.onesite_op)
        self.internal_symmetry = internal_symmetry
        self.site_directions = {
            site: tuple(edge_index for edge_index, (source, target) in enumerate(lattice.edges) if site in (source, target))
            for site in lattice.sites
        }
        self._shapes_by_order = {}
        for shape in lattice.connected_cluster_shapes(self.order):
            self._shapes_by_order.setdefault(shape.nsites, []).append(shape)
        self._shapes_by_order = {
            size: tuple(shapes)
            for size, shapes in self._shapes_by_order.items()
        }

    @property
    def connected_cluster_shapes(self):
        """Return all connected finite graph clusters through ``order``."""
        return tuple(
            shape
            for size in range(1, self.order + 1)
            for shape in self._shapes_by_order.get(size, ())
        )

    def build(self, beta, *, materialize=True, return_report=False):
        """Build the graph PEPO at coefficient ``beta``."""
        work_dtype = np.result_type(self.dtype, np.asarray(beta).dtype)
        onesite_op = np.asarray(self.onesite_op, dtype=work_dtype)
        twosite_op = np.asarray(self.twosite_op, dtype=work_dtype)
        one_site_exp, start_factors, end_factors = _edge_factors(
            twosite_op,
            onesite_op,
            beta,
            edge_cutoff=self.edge_cutoff,
            max_edge_rank=self.max_edge_rank,
            symmetric=False,
        )
        if self.order < 2:
            start_factors = np.zeros(
                (0, one_site_exp.shape[0], one_site_exp.shape[1]),
                dtype=work_dtype,
            )
            end_factors = start_factors.copy()
        blocks = {
            site: {(0,) * len(self.site_directions[site]): one_site_exp}
            for site in self.lattice.sites
        }
        allocator = _SectorAllocator()
        edge_sectors = {}
        for edge_index, (source, target) in enumerate(self.lattice.edges):
            sectors = allocator.allocate(start_factors.shape[0])
            edge_sectors[edge_index] = sectors
            _add_graph_edge_blocks(
                blocks,
                self.site_directions,
                edge_index,
                source,
                target,
                start_factors,
                end_factors,
                sectors,
            )

        residuals = []
        targets = []
        ranks = []
        counts = {"edge": len(self.lattice.edges)}
        for cluster_order in range(3, self.order + 1):
            lower_active = GraphActivePEPOBlocks(
                sites=self.lattice.sites,
                edges=self.lattice.edges,
                bond_dim=allocator.next_sector,
                physical_dim=one_site_exp.shape[0],
                site_directions=self.site_directions,
                blocks={site: dict(site_blocks) for site, site_blocks in blocks.items()},
            )
            level_count = 0
            for shape in self._shapes_by_order.get(cluster_order, ()):
                exact = _expm(
                    -beta * _graph_hamiltonian(
                        shape.nsites,
                        shape.edges,
                        twosite_op,
                        onesite_op,
                    ),
                    work_dtype,
                )
                lower = _contract_graph_active_support(
                    lower_active,
                    shape.sites,
                    shape.edges,
                )
                residual = exact - lower
                residual_norm = float(np.linalg.norm(residual))
                level_count += 1
                if residual_norm == 0.0:
                    continue
                factorized = _graph_tree_factorize_operator(
                    residual,
                    shape.edges,
                    shape.nsites,
                    one_site_exp.shape[0],
                    self.max_tree_rank,
                )
                if factorized[0] is None:
                    continue
                (
                    local_tensors,
                    parent,
                    parent_edge,
                    _children,
                    edge_ranks,
                    cluster_rank,
                    error,
                    target,
                ) = factorized
                sectors = {
                    parent_edge[site]: allocator.allocate(rank)
                    for site, rank in edge_ranks.items()
                }
                _add_graph_tree_factor_blocks(
                    blocks,
                    self.site_directions,
                    shape.sites,
                    shape.edges,
                    local_tensors,
                    parent,
                    parent_edge,
                    sectors,
                    one_site_exp.shape[0],
                )
                residuals.append(float(error))
                targets.append(float(target))
                ranks.append(cluster_rank)
            counts[f"order_{cluster_order}"] = level_count

        active = GraphActivePEPOBlocks(
            sites=self.lattice.sites,
            edges=self.lattice.edges,
            bond_dim=allocator.next_sector,
            physical_dim=one_site_exp.shape[0],
            site_directions=self.site_directions,
            blocks=blocks,
            charge_symmetry=(
                None if self.internal_symmetry is None else self.internal_symmetry.name
            ),
            physical_sectors=(
                None
                if self.internal_symmetry is None
                else self.internal_symmetry.resolved_physical_sectors(one_site_exp.shape[0])
            ),
        )
        if start_factors.size:
            edge_residual = _edge_fit_residual(
                twosite_op,
                onesite_op,
                beta,
                one_site_exp,
                start_factors,
                end_factors,
            )
            edge_target = edge_residual + sum(
                (
                    np.kron(start_factors[index], end_factors[index])
                    for index in range(start_factors.shape[0])
                ),
                start=np.zeros_like(edge_residual),
            )
            residual_norms = {"edge": float(np.linalg.norm(edge_residual))}
            relative = {
                "edge": float(
                    np.linalg.norm(edge_residual)
                    / max(np.linalg.norm(edge_target), np.finfo(float).eps)
                )
            }
        else:
            residual_norms = {}
            relative = {}
        if residuals:
            residual_norms["graph_tree"] = max(residuals)
            relative["graph_tree"] = max(
                error / max(target, np.finfo(float).eps)
                for error, target in zip(residuals, targets)
            )
        report = ClusterExpansionReport(
            beta=beta,
            order=self.order,
            local_dim=one_site_exp.shape[0],
            edge_rank=start_factors.shape[0],
            tree_rank=max(ranks, default=0),
            loop_rank=0,
            cluster_counts=counts,
            residual_norms=residual_norms,
            relative_residual_norms=relative,
            active_block_count=active.active_block_count,
            active_nbytes=active.active_nbytes,
            dense_nbytes=active.dense_nbytes,
        )
        result = active.to_tensor_network() if materialize else active
        return (result, report) if return_report else result


def build_graph_cluster_expansion_pepo(
    lattice,
    beta,
    twosite_op,
    onesite_op,
    *,
    order=3,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    internal_symmetry=None,
    dtype=None,
    materialize=True,
    return_report=False,
):
    """Build a connected-cluster operator on an arbitrary finite graph.

    ``lattice`` is a :class:`ClusterLattice` or a ``(sites, edges)`` pair.
    The materialized result is a generic Quimb ``TensorNetwork`` with one
    virtual bond per graph edge. Use ``materialize=False`` to retain sparse
    active blocks, or call ``to_dense()`` for small validation graphs.
    """
    plan = GraphClusterExpansionPlan(
        lattice,
        twosite_op,
        onesite_op,
        order=order,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        internal_symmetry=internal_symmetry,
        dtype=dtype,
    )
    return plan.build(beta, materialize=materialize, return_report=return_report)

# Compatibility re-export; the implementation lives in ``pepo_product``.
from .pepo_product import (
    CompiledPEPOClusterProduct,
    PEPOClusterFactor,
    PEPOClusterProductExpansion,
)


class _SectorAllocator:
    """Allocate disjoint active-sector ranges for independent tree terms."""

    def __init__(self):
        self.next_sector = 1

    def allocate(self, rank):
        sectors = tuple(range(self.next_sector, self.next_sector + rank))
        self.next_sector += rank
        return sectors


def _initialize_blocks(lx, ly, one_site_exp, site_directions):
    return {
        (i, j): {(0,) * len(site_directions[(i, j)]): one_site_exp}
        for i in range(lx)
        for j in range(ly)
    }


def _add_block(blocks, site_directions, site, sector_by_direction, tensor):
    directions = site_directions[site]
    if not all(direction in directions for direction in sector_by_direction):
        return
    key = [0] * len(directions)
    for direction, sector in sector_by_direction.items():
        key[directions.index(direction)] = sector
    key = tuple(key)
    if key in blocks[site]:
        blocks[site][key] = blocks[site][key] + tensor
    else:
        blocks[site][key] = tensor


def _add_single_direction_blocks(
    blocks,
    site_directions,
    lx,
    ly,
    direction,
    sectors,
    factors,
    *,
    source,
    cyclic,
):
    """Add one endpoint role on every translated edge endpoint."""
    del lx, ly, cyclic  # geometry is already encoded by ``site_directions``
    factor_array = factors
    for site, directions in site_directions.items():
        try:
            direction_axis = directions.index(direction)
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(directions)
        for channel, sector in enumerate(sectors):
            key = key_template.copy()
            key[direction_axis] = sector
            key = tuple(key)
            tensor = factor_array[channel]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + tensor
            else:
                site_blocks[key] = tensor


def _add_positive_edge_channels(
    blocks,
    site_directions,
    start_factors,
    end_factors,
    sectors,
):
    """Add compact order-two endpoint channels for positive lattice edges."""
    for direction in _POSITIVE_DIRECTIONS:
        opposite = _OPPOSITE_DIRECTION[direction]
        for site, directions in site_directions.items():
            site_blocks = blocks[site]
            try:
                direction_axis = directions.index(direction)
            except ValueError:
                direction_axis = None
            if direction_axis is not None:
                key_template = [0] * len(directions)
                for channel, sector in enumerate(sectors):
                    key = key_template.copy()
                    key[direction_axis] = sector
                    key = tuple(key)
                    tensor = start_factors[channel]
                    if key in site_blocks:
                        site_blocks[key] = site_blocks[key] + tensor
                    else:
                        site_blocks[key] = tensor
            try:
                opposite_axis = directions.index(opposite)
            except ValueError:
                opposite_axis = None
            if opposite_axis is not None:
                key_template = [0] * len(directions)
                for channel, sector in enumerate(sectors):
                    key = key_template.copy()
                    key[opposite_axis] = sector
                    key = tuple(key)
                    tensor = end_factors[channel]
                    if key in site_blocks:
                        site_blocks[key] = site_blocks[key] + tensor
                    else:
                        site_blocks[key] = tensor


def _add_pair_blocks(
    blocks,
    site_directions,
    pair,
    tensor,
    sectors,
    *,
    transpose_opposite=False,
):
    """Add a pair-active center block, preserving direction-axis order."""
    if transpose_opposite and set(pair) in ({"u", "d"}, {"r", "l"}):
        tensor = tensor.transpose(1, 0, 2, 3)
    if len(sectors) == 2 and all(isinstance(axis, tuple) for axis in sectors):
        sector_axes = sectors
    else:
        sector_axes = (sectors, sectors)
    expected_shape = (len(sector_axes[0]), len(sector_axes[1]))
    if tensor.shape[:2] == expected_shape[::-1] and expected_shape[0] != expected_shape[1]:
        # C4 rotation can change the sorted direction order. Normalize the
        # tensor axes to the semantic sector order before inserting blocks.
        tensor = tensor.transpose(1, 0, 2, 3)
    elif tensor.shape[:2] != expected_shape:
        raise ValueError(
            "pair tensor active axes do not match sector dimensions: "
            f"tensor={tensor.shape[:2]}, sectors={expected_shape}."
        )
    for site, directions in site_directions.items():
        try:
            first_axis = directions.index(pair[0])
            second_axis = directions.index(pair[1])
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(directions)
        for first, second in np.ndindex(tensor.shape[:2]):
            key = key_template.copy()
            key[first_axis] = sector_axes[0][first]
            key[second_axis] = sector_axes[1][second]
            key = tuple(key)
            block = tensor[first, second]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + block
            else:
                site_blocks[key] = block


def _add_triple_blocks(blocks, site_directions, directions, tensor, sectors):
    """Add a three-active-leg center block for every translated star."""
    if len(sectors) == 3 and all(isinstance(axis, tuple) for axis in sectors):
        sector_axes = sectors
    else:
        sector_axes = (sectors, sectors, sectors)
    for site, present in site_directions.items():
        try:
            axes = tuple(present.index(direction) for direction in directions)
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(present)
        for first, second, third in np.ndindex(tensor.shape[:3]):
            key = key_template.copy()
            key[axes[0]] = sector_axes[0][first]
            key[axes[1]] = sector_axes[1][second]
            key[axes[2]] = sector_axes[2][third]
            key = tuple(key)
            block = tensor[first, second, third]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + block
            else:
                site_blocks[key] = block


def _add_pair_block_at_site(blocks, site_directions, site, pair, tensor, sectors):
    """Add one pair-active block without translating it over the lattice."""
    directions = site_directions[site]
    try:
        first_axis = directions.index(pair[0])
        second_axis = directions.index(pair[1])
    except ValueError:
        return
    sector_axes = tuple(sectors)
    expected_shape = (len(sector_axes[0]), len(sector_axes[1]))
    if tensor.shape[:2] != expected_shape:
        raise ValueError(
            "loop tensor active axes do not match sector dimensions: "
            f"tensor={tensor.shape[:2]}, sectors={expected_shape}."
        )
    site_blocks = blocks[site]
    key_template = [0] * len(directions)
    for first, second in np.ndindex(tensor.shape[:2]):
        key = key_template.copy()
        key[first_axis] = sector_axes[0][first]
        key[second_axis] = sector_axes[1][second]
        key = tuple(key)
        block = tensor[first, second]
        if key in site_blocks:
            site_blocks[key] = site_blocks[key] + block
        else:
            site_blocks[key] = block




def _plaquette_starts(lx, ly, cyclic):
    """Return lower-left sites whose four-edge plaquette is present."""
    starts = []
    for i in range(lx):
        for j in range(ly):
            first = (i, j)
            upper = _site_after(first, "u", lx, ly, cyclic)
            right = _site_after(first, "r", lx, ly, cyclic)
            if upper is None or right is None:
                continue
            diagonal = _site_after(upper, "r", lx, ly, cyclic)
            if diagonal is None or _site_after(right, "u", lx, ly, cyclic) != diagonal:
                continue
            starts.append(first)
    return tuple(starts)


def _path_start_sites(steps, lx, ly, cyclic):
    """Return starts whose directed path is present on the finite lattice."""
    starts = []
    for i in range(lx):
        for j in range(ly):
            site = (i, j)
            for direction in steps:
                site = _site_after(site, direction, lx, ly, cyclic)
                if site is None:
                    break
            else:
                starts.append((i, j))
    return tuple(starts)


def _valid_path_steps():
    """Generate non-self-closing four-site square-lattice tree paths."""
    paths = set()
    for first in _DIRECTIONS:
        for second in _DIRECTIONS:
            if second == _OPPOSITE_DIRECTION[first]:
                continue
            for third in _DIRECTIONS:
                if third == _OPPOSITE_DIRECTION[second]:
                    continue
                coordinates = [(0, 0)]
                for direction in (first, second, third):
                    previous = coordinates[-1]
                    vector = _DIRECTION_VECTORS[direction]
                    coordinates.append(
                        (previous[0] + vector[0], previous[1] + vector[1])
                    )
                if any(
                    abs(coordinates[left][0] - coordinates[right][0])
                    + abs(coordinates[left][1] - coordinates[right][1])
                    == 1
                    for left in range(4)
                    for right in range(left + 2, 4)
                ):
                    continue
                reverse = tuple(
                    _OPPOSITE_DIRECTION[direction]
                    for direction in reversed((first, second, third))
                )
                paths.add(min((first, second, third), reverse))
    return tuple(sorted(paths))


def _rotate_steps(steps):
    return tuple(_C4_ROTATION[direction] for direction in steps)


def _path_orbits(symmetry):
    remaining = set(_valid_path_steps())
    groups = []
    while remaining:
        representative = min(remaining)
        if symmetry != "C4":
            remaining.remove(representative)
            groups.append((representative, (representative,)))
            continue
        orbit = []
        steps = representative
        for _ in range(4):
            # Only group paths connected by an actual C4 rotation.  The
            # reversed path is physically equivalent, but its PEPO factors
            # require swapping endpoint tensors as well as rotating axes;
            # treating it as a plain axis rotation can cycle forever in
            # ``_rotate_direction_tensor`` and is not a valid factor map.
            if steps in remaining:
                orbit.append(steps)
                remaining.discard(steps)
            steps = _rotate_steps(steps)
        groups.append((representative, tuple(orbit)))
    return tuple(groups)


def _all_direction_pairs():
    return tuple(
        (first, second)
        for first_index, first in enumerate(_DIRECTIONS)
        for second in _DIRECTIONS[first_index + 1 :]
    )


def _rotate_pair(pair):
    rotated = tuple(_C4_ROTATION[direction] for direction in pair)
    return tuple(sorted(rotated, key=_DIRECTIONS.index))


def _pair_orbits():
    remaining = set(_all_direction_pairs())
    orbits = []
    while remaining:
        representative = min(remaining, key=lambda pair: tuple(_DIRECTIONS.index(x) for x in pair))
        orbit = []
        pair = representative
        while pair not in orbit:
            orbit.append(pair)
            remaining.discard(pair)
            pair = _rotate_pair(pair)
        orbits.append((representative, tuple(orbit)))
    return tuple(orbits)


def _rotate_pair_tensor(pair, target, tensor):
    """Rotate active axes while preserving the canonical direction order."""
    return _rotate_direction_tensor(pair, target, tensor)


def _rotate_direction_tensor(directions, target, tensor):
    """Rotate a tensor whose leading axes follow ``directions``."""
    current = directions
    rotated = tensor
    while current != target:
        raw = tuple(_C4_ROTATION[direction] for direction in current)
        order = tuple(sorted(range(len(raw)), key=lambda index: _DIRECTIONS.index(raw[index])))
        next_directions = tuple(raw[index] for index in order)
        rotated = ar.do(
            "transpose",
            rotated,
            order + tuple(range(len(directions), rotated.ndim)),
        )
        current = next_directions
    return rotated


def _direction_subsets(size):
    return tuple(
        tuple(
            direction
            for index, direction in enumerate(_DIRECTIONS)
            if mask & (1 << index)
        )
        for mask in range(1 << len(_DIRECTIONS))
        if mask.bit_count() == size
    )


def _subset_orbits(size, symmetry):
    subsets = set(_direction_subsets(size))
    if symmetry != "C4":
        return tuple((subset, (subset,)) for subset in subsets)
    orbits = []
    while subsets:
        representative = min(subsets, key=lambda subset: tuple(_DIRECTIONS.index(x) for x in subset))
        orbit = []
        current = representative
        while current not in orbit:
            orbit.append(current)
            subsets.discard(current)
            current = tuple(sorted((_C4_ROTATION[x] for x in current), key=_DIRECTIONS.index))
        orbits.append((representative, tuple(orbit)))
    return tuple(orbits)


def _solve_four_star(
    directions,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
    beta,
):
    """Solve a four-site T-shaped tree residual at its degree-three center."""
    rank = start_factors.shape[0]
    local_dim = onesite_op.shape[0]
    # Use the labeled order ``(endpoint-0, center, endpoint-1, endpoint-2)``
    # so the solved center block is aligned with its PEPO embedding.
    edges = (
        (1, 0, directions[0]),
        (1, 2, directions[1]),
        (1, 3, directions[2]),
    )
    _, residual = _lower_tree_residual(
        4,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
        pair_tensors,
    )
    endpoint_factors = [
        end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for direction in directions
    ]
    endpoint_matrices = tuple(
        factors.reshape(rank, local_dim**2).T for factors in endpoint_factors
    )
    endpoint_pseudoinverses = tuple(
        np.linalg.pinv(factors) for factors in endpoint_matrices
    )
    residual_tensor = _operator_tensor(residual, 4, local_dim)
    center_coefficients = np.einsum(
        "ijkl,ai,bk,cl->jabc",
        residual_tensor,
        endpoint_pseudoinverses[0],
        endpoint_pseudoinverses[1],
        endpoint_pseudoinverses[2],
    )
    solution = center_coefficients.transpose(1, 2, 3, 0).reshape(
        rank, rank, rank, local_dim, local_dim
    )
    fitted = np.einsum(
        "ia,jabc,kb,lc->ijkl",
        endpoint_matrices[0],
        center_coefficients,
        endpoint_matrices[1],
        endpoint_matrices[2],
    )
    return (
        solution,
        np.linalg.norm(residual_tensor - fitted),
        np.linalg.norm(residual_tensor),
    )


def _solve_four_path(
    steps,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
    beta,
    max_tree_rank,
):
    """Solve a four-site path using an endpoint projection and internal SVD."""
    rank = start_factors.shape[0]
    local_dim = onesite_op.shape[0]
    first_factors = (
        start_factors if steps[0] in _POSITIVE_DIRECTIONS else end_factors
    )
    last_factors = (
        end_factors if steps[2] in _POSITIVE_DIRECTIONS else start_factors
    )
    edges = tuple((index, index + 1, direction) for index, direction in enumerate(steps))
    _, residual = _lower_tree_residual(
        4,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
        pair_tensors,
    )

    # Project only the two endpoint physical spaces.  The remaining core is
    # a small matrix whose SVD supplies the two internal PEPO factors.
    first_matrix = first_factors.reshape(rank, local_dim**2).T
    last_matrix = last_factors.reshape(rank, local_dim**2).T
    first_pseudoinverse = np.linalg.pinv(first_matrix)
    last_pseudoinverse = np.linalg.pinv(last_matrix)
    residual_tensor = _operator_tensor(residual, 4, local_dim)
    projected = np.einsum(
        "ijkl,ai,dl->ajkd",
        residual_tensor,
        first_pseudoinverse,
        last_pseudoinverse,
    )
    # Arrange the right matrix as ``(last-site, operator-basis)``. This is
    # the coefficient ordering compatible with a Kronecker product of the
    # two local physical matrices after the SVD factorization.
    core = projected.transpose(0, 1, 3, 2).reshape(
        rank * local_dim * local_dim, local_dim * local_dim * rank
    )
    left, singular_values, right = np.linalg.svd(core, full_matrices=False)
    if not singular_values.size or singular_values[0] == 0.0:
        empty_left = np.zeros((rank, 0, local_dim, local_dim), dtype=onesite_op.dtype)
        empty_right = np.zeros((0, rank, local_dim, local_dim), dtype=onesite_op.dtype)
        return empty_left, empty_right, np.linalg.norm(residual.reshape(-1)), np.linalg.norm(residual.reshape(-1))
    threshold = singular_values[0] * np.finfo(singular_values.dtype).eps * max(1, core.shape[0])
    keep = np.flatnonzero(singular_values > threshold)
    if max_tree_rank is not None:
        keep = keep[:max_tree_rank]
    if not keep.size:
        keep = np.array([0])
    singular_values = singular_values[keep]
    root = np.sqrt(singular_values)
    left_factors = (left[:, keep] * root).reshape(
        rank, local_dim, local_dim, len(keep)
    ).transpose(0, 3, 1, 2)
    right_factors = (root[:, None] * right[keep, :]).reshape(
        len(keep), rank, local_dim, local_dim
    )
    reconstructed = np.einsum(
        "ia,acj,cdk,ld->ijkl",
        first_matrix,
        left_factors.reshape(rank, len(keep), local_dim**2),
        right_factors.reshape(len(keep), rank, local_dim**2),
        last_matrix,
    ).reshape(residual_tensor.shape)
    return (
        left_factors,
        right_factors,
        np.linalg.norm(residual_tensor - reconstructed),
        np.linalg.norm(residual_tensor),
    )


def _dense_loop_tensors(coefficients, local_dim):
    """Factor a four-site dense correction into an exact tensor ring.

    The local operator basis is the matrix-unit basis of size ``d**2``. A
    pair of adjacent basis labels is carried by each virtual leg, giving an
    exact rank ``d**4`` ring without a coefficient-dependent decomposition.
    This is deliberately the dense analogue of the fixed Pauli loop path:
    coefficient values remain ordinary NumPy values while the virtual
    topology is fixed and sparse at the PEPO level.
    """
    operator_rank = local_dim**2
    loop_coefficients = coefficients.transpose(0, 1, 3, 2).reshape(
        operator_rank,
        operator_rank,
        operator_rank**2,
    )
    matrix_units = np.eye(operator_rank, dtype=coefficients.dtype).reshape(
        operator_rank,
        local_dim,
        local_dim,
    )
    pair_labels = np.arange(operator_rank**2)
    first_units = matrix_units[pair_labels // operator_rank]
    second_units = matrix_units[pair_labels % operator_rank]
    diagonal = np.eye(operator_rank**2, dtype=coefficients.dtype)
    first_corner = (
        first_units.reshape(1, operator_rank**2, local_dim, local_dim)
        * diagonal.reshape(operator_rank**2, operator_rank**2, 1, 1)
    )
    second_corner = (
        loop_coefficients.reshape(operator_rank**2, operator_rank**2, 1, 1)
        * second_units.reshape(operator_rank**2, 1, local_dim, local_dim)
    )
    third_corner = (
        first_units.reshape(1, operator_rank**2, local_dim, local_dim)
        * diagonal.reshape(operator_rank**2, operator_rank**2, 1, 1)
    )
    fourth_corner = (
        second_units.reshape(operator_rank**2, 1, local_dim, local_dim)
        * np.ones((1, operator_rank**2, 1, 1), dtype=coefficients.dtype)
    )
    return first_corner, second_corner, third_corner, fourth_corner


def _cluster_shape_embeddings(shape, lx, ly, cyclic):
    """Return all valid finite-lattice translations of a cluster shape."""
    embeddings = []
    for start_i in range(lx):
        for start_j in range(ly):
            mapped = []
            valid = True
            for x, y in shape.sites:
                site = (start_i + x, start_j + y)
                if cyclic[0]:
                    site = (site[0] % lx, site[1])
                if cyclic[1]:
                    site = (site[0], site[1] % ly)
                if not (0 <= site[0] < lx and 0 <= site[1] < ly):
                    valid = False
                    break
                mapped.append(site)
            if not valid or len(set(mapped)) != shape.nsites:
                continue
            if any(
                _site_after(mapped[source], direction, lx, ly, cyclic)
                != mapped[target]
                for source, target, direction in shape.edges
            ):
                continue
            embeddings.append(tuple(mapped))
    return tuple(embeddings)


def _contract_tree_factor_coefficients(local_tensors, parent, children, nsites):
    """Contract coefficient tensors produced by ``_tree_factorize_operator``."""
    bond_labels = {
        child: nsites + child
        for child in range(nsites)
        if parent[child] is not None
    }
    operands = []
    for site in range(nsites):
        child_nodes, tensor = local_tensors[site]
        labels = [bond_labels[child] for child in child_nodes]
        labels.append(site)
        if parent[site] is not None:
            labels.append(bond_labels[site])
        operands.extend((tensor, labels))
    return np.einsum(*operands, list(range(nsites)), optimize=True)


def _tree_factorize_operator(operator, edges, nsites, local_dim, max_rank):
    """Factor an operator coefficient tensor over a spanning tree.

    Each edge is split with an SVD after all child subtrees have been
    collected. This is an exact tree tensor factorization when ``max_rank``
    is ``None`` and a controlled singular-value truncation otherwise.
    """
    operator_tensor = _operator_tensor(operator, nsites, local_dim)
    operator_rank = local_dim**2
    adjacency = [[] for _ in range(nsites)]
    for source, target, direction in edges:
        adjacency[source].append((target, direction))
        adjacency[target].append((source, _OPPOSITE_DIRECTION[direction]))

    parent = {0: None}
    parent_direction = {}
    traversal = [0]
    for site in traversal:
        for neighbour, direction in adjacency[site]:
            if neighbour in parent:
                continue
            parent[neighbour] = site
            parent_direction[neighbour] = direction
            traversal.append(neighbour)
    if len(parent) != nsites:
        raise ValueError("cluster graph must be connected for tree factorization.")

    children = {site: [] for site in range(nsites)}
    for site in traversal[1:]:
        children[parent[site]].append(site)

    current = operator_tensor
    axes = [("physical", site) for site in range(nsites)]
    local_tensors = {}
    ranks = {}
    for site in reversed(traversal[1:]):
        child_nodes = tuple(children[site])
        row_axes = [("bond", child) for child in child_nodes]
        row_axes.append(("physical", site))
        column_axes = [axis for axis in axes if axis not in row_axes]
        permutation = [axes.index(axis) for axis in row_axes + column_axes]
        matrix = current.transpose(permutation).reshape(
            int(np.prod([current.shape[index] for index in permutation[:len(row_axes)]])),
            -1,
        )
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if not singular_values.size or singular_values[0] == 0.0:
            return None, None, None, None, None, 0, 0.0, 0.0
        threshold = singular_values[0] * np.finfo(singular_values.dtype).eps * max(
            1,
            matrix.shape[0],
            matrix.shape[1],
        )
        keep = np.flatnonzero(singular_values > threshold)
        if max_rank is not None:
            keep = keep[:max_rank]
        if not keep.size:
            keep = np.array([0])
        rank = int(keep.size)
        local_tensors[site] = (
            child_nodes,
            left[:, keep].reshape(
                tuple(current.shape[axes.index(("bond", child))] for child in child_nodes)
                + (operator_rank, rank),
            ),
        )
        ranks[site] = rank
        current = (singular_values[keep, None] * right[keep, :]).reshape(
            (rank,) + tuple(current.shape[axes.index(axis)] for axis in column_axes)
        )
        axes = [("bond", site)] + column_axes

    root = 0
    root_axes = [("bond", child) for child in children[root]]
    root_axes.append(("physical", root))
    permutation = [axes.index(axis) for axis in root_axes]
    root_tensor = current.transpose(permutation).reshape(
        tuple(current.shape[index] for index in permutation)
    )
    local_tensors[root] = (tuple(children[root]), root_tensor)
    reconstructed = _contract_tree_factor_coefficients(
        local_tensors,
        parent,
        children,
        nsites,
    )
    return (
        local_tensors,
        parent,
        parent_direction,
        children,
        ranks,
        max(ranks.values(), default=0),
        float(np.linalg.norm(operator_tensor - reconstructed)),
        float(np.linalg.norm(operator_tensor)),
    )


def _tree_factorize_operator_backend(
    operator,
    edges,
    nsites,
    local_dim,
    max_rank=None,
):
    """Autodiff-safe exact spanning-tree factorization of a local operator.

    The topology and retained ranks are determined from static matrix shapes;
    no backend value is converted to NumPy and no singular-value threshold is
    used.  This keeps the factorization differentiable for Torch and JAX.
    Loop edges are intentionally accepted and ignored only by the
    factorization tree; they remain present in the operator supplied by the
    connected-cluster residual solver.
    """
    operator_tensor = _backend_operator_tensor(operator, nsites, local_dim)
    operator_rank = local_dim**2
    adjacency = [[] for _ in range(nsites)]
    for source, target, direction in edges:
        adjacency[source].append((target, direction))
        adjacency[target].append((source, _OPPOSITE_DIRECTION[direction]))

    parent = {0: None}
    parent_direction = {}
    traversal = [0]
    for site in traversal:
        for neighbour, direction in adjacency[site]:
            if neighbour in parent:
                continue
            parent[neighbour] = site
            parent_direction[neighbour] = direction
            traversal.append(neighbour)
    if len(parent) != nsites:
        raise ValueError("cluster graph must be connected for tree factorization.")

    children = {site: [] for site in range(nsites)}
    for site in traversal[1:]:
        children[parent[site]].append(site)

    current = operator_tensor
    axes = [("physical", site) for site in range(nsites)]
    local_tensors = {}
    ranks = {}
    for site in reversed(traversal[1:]):
        child_nodes = tuple(children[site])
        row_axes = [("bond", child) for child in child_nodes]
        row_axes.append(("physical", site))
        column_axes = [axis for axis in axes if axis not in row_axes]
        permutation = [axes.index(axis) for axis in row_axes + column_axes]
        transposed = ar.do("transpose", current, permutation)
        row_shape = tuple(
            int(current.shape[index]) for index in permutation[: len(row_axes)]
        )
        column_shape = tuple(
            int(current.shape[index]) for index in permutation[len(row_axes) :]
        )
        matrix = ar.do(
            "reshape",
            transposed,
            (int(np.prod(row_shape)), int(np.prod(column_shape))),
        )
        left, singular_values, right = _fixed_rank_svd(matrix)
        rank = min(int(matrix.shape[-2]), int(matrix.shape[-1]))
        if max_rank is not None:
            rank = min(rank, max_rank)
        if rank < 1:
            return None
        left = left[:, :rank]
        singular_values = singular_values[:rank]
        right = right[:rank, :]
        local_tensors[site] = (
            child_nodes,
            ar.do(
                "reshape",
                left,
                tuple(
                    int(current.shape[axes.index(("bond", child))])
                    for child in child_nodes
                )
                + (operator_rank, rank),
            ),
        )
        ranks[site] = rank
        weighted_right = ar.do(
            "multiply",
            ar.do("reshape", singular_values, (rank, 1)),
            right,
        )
        current = ar.do(
            "reshape",
            weighted_right,
            (rank,)
            + tuple(int(current.shape[axes.index(axis)]) for axis in column_axes),
        )
        axes = [("bond", site)] + column_axes

    root = 0
    root_axes = [("bond", child) for child in children[root]]
    root_axes.append(("physical", root))
    permutation = [axes.index(axis) for axis in root_axes]
    root_tensor = ar.do("transpose", current, permutation)
    root_shape = tuple(int(root_tensor.shape[index]) for index in range(root_tensor.ndim))
    local_tensors[root] = (tuple(children[root]), ar.do("reshape", root_tensor, root_shape))
    return local_tensors, parent, parent_direction, children, ranks


def _spanning_tree_edge_indices(edges, nsites):
    """Return a breadth-first spanning tree through ``edges``."""
    adjacency = [[] for _ in range(nsites)]
    for edge_index, (source, target, direction) in enumerate(edges):
        adjacency[source].append((target, edge_index, direction))
        adjacency[target].append((source, edge_index, _OPPOSITE_DIRECTION[direction]))

    visited = {0}
    queue = [0]
    tree_edge_indices = []
    while queue:
        source = queue.pop(0)
        for target, edge_index, _ in adjacency[source]:
            if target in visited:
                continue
            visited.add(target)
            queue.append(target)
            tree_edge_indices.append(edge_index)
    if len(visited) != nsites:
        raise ValueError("cluster graph must be connected for Quimb fitting.")
    return tuple(tree_edge_indices)


def _cluster_incident_edges(edges, nsites):
    """Return incident edge labels and directions for every cluster site."""
    incident = {site: [] for site in range(nsites)}
    for edge_index, (source, target, direction) in enumerate(edges):
        incident[source].append((edge_index, direction))
        incident[target].append((edge_index, _OPPOSITE_DIRECTION[direction]))
    return {
        site: tuple(sorted(site_edges))
        for site, site_edges in incident.items()
    }


def _quimb_loop_rank_schedule(
    local_dim,
    *,
    max_loop_rank,
    max_tree_rank,
    adaptive,
    loop_rank_start,
    loop_rank_step,
):
    """Return the loop ranks to try, from economical to expressive."""
    if not adaptive:
        return (max_loop_rank,)
    cap = max_loop_rank
    if cap is None:
        cap = max(local_dim**2, max_tree_rank or 1)
    start = 1 if loop_rank_start is None else min(loop_rank_start, cap)
    ranks = list(range(start, cap + 1, loop_rank_step))
    if not ranks or ranks[-1] != cap:
        ranks.append(cap)
    return tuple(ranks)


def _quimb_warm_start_arrays(
    fitted_arrays,
    fitted_edge_ranks,
    initial_arrays,
):
    """Pad a previous loop fit into a larger-rank initial ansatz."""
    warm_arrays = {}
    initial_local_arrays, initial_edge_ranks, _ = initial_arrays
    for site, (new_edge_ids, new_data) in initial_local_arrays.items():
        old_edge_ids, old_data = fitted_arrays[site]
        if old_edge_ids != new_edge_ids:
            raise ValueError("Quimb warm-start edge ordering changed unexpectedly.")
        if any(
            fitted_edge_ranks[edge_index] > initial_edge_ranks[edge_index]
            for edge_index in old_edge_ids
        ):
            raise ValueError("Quimb warm-start rank decreased unexpectedly.")
        slices = tuple(
            slice(0, fitted_edge_ranks[edge_index])
            for edge_index in old_edge_ids
        ) + (slice(None),)
        new_data[slices] = old_data
        warm_arrays[site] = (new_edge_ids, new_data)
    return warm_arrays, initial_edge_ranks, initial_arrays[2]


def _quimb_fit_initial_arrays(
    operator,
    edges,
    nsites,
    local_dim,
    *,
    method,
    max_tree_rank,
    max_loop_rank,
    fit_seed,
):
    """Construct a fixed-rank Quimb fitting ansatz for a cluster tensor."""
    operator_tensor = _operator_tensor(operator, nsites, local_dim)
    operator_rank = local_dim**2
    tree_edge_indices = _spanning_tree_edge_indices(edges, nsites)
    tree_edges = tuple(edges[index] for index in tree_edge_indices)
    tree_result = _tree_factorize_operator(
        operator,
        tree_edges,
        nsites,
        local_dim,
        max_tree_rank,
    )
    (
        tree_tensors,
        parent,
        _,
        children,
        tree_ranks,
        tree_rank,
        _,
        _,
    ) = tree_result
    if tree_tensors is None:
        return None

    tree_edge_ids = set(tree_edge_indices)
    incident = _cluster_incident_edges(edges, nsites)
    edge_ranks = {}
    for edge_index in tree_edge_indices:
        source, target, _ = edges[edge_index]
        if parent.get(target) == source:
            child = target
        elif parent.get(source) == target:
            child = source
        else:
            raise ValueError("spanning-tree edge is inconsistent with its parent map.")
        edge_ranks[edge_index] = tree_ranks[child]
    if method == "tree":
        edge_ranks = {
            edge_index: edge_ranks[edge_index]
            for edge_index in tree_edge_indices
        }
    else:
        loop_rank = max_loop_rank
        if loop_rank is None:
            loop_rank = max(max(tree_ranks.values(), default=1), operator_rank)
        edge_ranks.update(
            {
                edge_index: loop_rank
                for edge_index in range(len(edges))
                if edge_index not in edge_ranks
            }
        )

    # Convert the tree-SVD initialization to the edge ordering used by the
    # Quimb ansatz. For a loop, retain the tree factorization in the zero
    # slice of each additional edge and seed the other slices very lightly so
    # ALS can activate them.
    rng = np.random.default_rng(fit_seed)
    local_arrays = {}
    tree_edge_lookup = {
        frozenset((source, target)): edge_index
        for edge_index, (source, target, _) in zip(tree_edge_indices, tree_edges)
    }
    for site in range(nsites):
        child_nodes, tree_tensor = tree_tensors[site]
        source_axes = [
            tree_edge_lookup[frozenset((site, child))]
            for child in child_nodes
        ]
        source_axes.append("physical")
        if parent[site] is not None:
            source_axes.append(
                tree_edge_lookup[frozenset((site, parent[site]))]
            )
        source_axes = tuple(source_axes)
        target_edge_ids = tuple(edge_index for edge_index, _ in incident[site])
        tree_target_edge_ids = tuple(
            edge_index for edge_index in target_edge_ids if edge_index in tree_edge_ids
        )
        source_axis_positions = {axis: index for index, axis in enumerate(source_axes)}
        tree_data = np.asarray(tree_tensor).transpose(
            tuple(source_axis_positions[axis] for axis in tree_target_edge_ids)
            + (source_axis_positions["physical"],)
        )
        target_shape = tuple(edge_ranks[edge_index] for edge_index in target_edge_ids)
        target_shape += (operator_rank,)
        data = np.zeros(target_shape, dtype=operator_tensor.dtype)
        tree_slices = tuple(
            slice(None) if edge_index in tree_edge_ids else 0
            for edge_index in target_edge_ids
        )
        data[tree_slices + (slice(None),)] = tree_data
        if method != "tree" and any(edge_index not in tree_edge_ids for edge_index in target_edge_ids):
            scale = max(float(np.linalg.norm(operator_tensor)), 1.0) ** (1.0 / nsites)
            noise = rng.normal(size=target_shape) + 1j * rng.normal(size=target_shape)
            data = data + (1e-6 * scale / max(np.sqrt(data.size), 1.0)) * noise
        local_arrays[site] = (target_edge_ids, data)

    return local_arrays, edge_ranks, tree_rank


def _quimb_factorize_operator(
    operator,
    edges,
    nsites,
    local_dim,
    *,
    method,
    max_tree_rank,
    max_loop_rank,
    fit_steps,
    fit_tol,
    fit_solver_maxiter,
    fit_seed,
    warm_start=None,
):
    """Fit a cluster operator tensor with Quimb tree or ALS machinery."""
    if method not in {"tree", "als"}:
        raise ValueError("Quimb cluster fitting method must be 'tree' or 'als'.")
    if method == "tree" and len(edges) != nsites - 1:
        raise ValueError("Quimb tree fitting requires a loop-free cluster.")

    initial = _quimb_fit_initial_arrays(
        operator,
        edges,
        nsites,
        local_dim,
        method=method,
        max_tree_rank=max_tree_rank,
        max_loop_rank=max_loop_rank,
        fit_seed=fit_seed,
    )
    if initial is None:
        return None
    local_arrays, edge_ranks, tree_rank = initial
    if warm_start is not None:
        local_arrays, edge_ranks, tree_rank = _quimb_warm_start_arrays(
            warm_start[0],
            warm_start[1],
            initial,
        )
    incident = _cluster_incident_edges(edges, nsites)
    physical_inds = tuple(f"__pepsy_cluster_phys_{site}" for site in range(nsites))
    edge_inds = {
        edge_index: f"__pepsy_cluster_edge_{edge_index}"
        for edge_index in edge_ranks
    }
    target = qtn.Tensor(
        _operator_tensor(operator, nsites, local_dim),
        inds=physical_inds,
    ).as_network()
    tensors = []
    site_tags = {}
    for site, (edge_ids, data) in local_arrays.items():
        inds = tuple(edge_inds[edge_index] for edge_index in edge_ids)
        inds += (physical_inds[site],)
        tag = f"__pepsy_cluster_fit_site_{site}"
        site_tags[site] = tag
        tensors.append(qtn.Tensor(data, inds=inds, tags=tag))
    ansatz = qtn.TensorNetwork(tensors)
    fit_options = {
        "method": method,
        "tags": tuple(site_tags.values()),
        "steps": fit_steps,
        "tol": fit_tol,
        "progbar": False,
        "inplace": False,
        # ALS repeatedly reuses the same small cluster environment. Greedy
        # paths avoid re-running the heavier hyper-optimizer for each local
        # tensor while preserving the exact overlap objective.
        "contract_optimize": "greedy",
    }
    if method == "als":
        fit_options["solver_maxiter"] = fit_solver_maxiter
    fitted = ansatz.fit(target, **fit_options)
    operator_tensor = _operator_tensor(operator, nsites, local_dim)
    distance_method = "dense" if operator_tensor.size <= 1_000_000 else "overlap"
    relative_fit_error = float(
        tensor_network_distance(
            fitted,
            target,
            method=distance_method,
            normalized=True,
        )
    )
    target_norm = float(np.linalg.norm(operator_tensor))
    # Dense distance is exact for small local clusters, while the overlap
    # distance avoids materializing a d**(2 * nsites) local matrix for larger
    # clusters.
    fit_error = relative_fit_error * target_norm
    fitted_arrays = {}
    for site, tag in site_tags.items():
        (fitted_tensor,) = fitted.select_tensors(tag, which="all")
        expected_inds = tuple(
            edge_inds[edge_index] for edge_index, _ in incident[site]
        )
        expected_inds += (physical_inds[site],)
        fitted_arrays[site] = (
            tuple(edge_index for edge_index, _ in incident[site]),
            np.asarray(fitted_tensor.transpose(*expected_inds).data),
        )
    return (
        fitted_arrays,
        edge_ranks,
        max(edge_ranks.values(), default=tree_rank),
        fit_error,
        target_norm,
    )


def _rotate_direction(direction, turns):
    """Rotate one lattice direction by ``turns`` quarter turns."""
    for _ in range(turns % 4):
        direction = _C4_ROTATION[direction]
    return direction


def _cluster_shape_c4_orbit(shape):
    """Return C4-related shapes and site maps from one representative."""
    orbit = []
    seen = set()
    for turns in range(4):
        rotated_sites = tuple(
            _rotate_coordinate(site, turns)
            for site in shape.sites
        )
        normalized = _normalize_cluster_sites(rotated_sites)
        if normalized in seen:
            continue
        seen.add(normalized)
        min_x = min(x for x, _ in rotated_sites)
        min_y = min(y for _, y in rotated_sites)
        target_indices = {
            site: index for index, site in enumerate(normalized)
        }
        source_to_target = tuple(
            target_indices[(x - min_x, y - min_y)]
            for x, y in rotated_sites
        )
        orbit.append(
            (
                _make_cluster_shape(normalized),
                source_to_target,
                turns,
            )
        )
    return tuple(orbit)


def _shape_rotation_map(source_shape, target_shape):
    """Return the source-to-target site map for a C4-related shape."""
    source_sites = source_shape.sites
    target_sites = target_shape.sites
    target_indices = {site: index for index, site in enumerate(target_sites)}
    for turns in range(4):
        rotated_sites = tuple(
            _rotate_coordinate(site, turns)
            for site in source_sites
        )
        min_x = min(x for x, _ in rotated_sites)
        min_y = min(y for _, y in rotated_sites)
        normalized = tuple(
            sorted((x - min_x, y - min_y) for x, y in rotated_sites)
        )
        if normalized != target_sites:
            continue
        return (
            tuple(
                target_indices[(x - min_x, y - min_y)]
                for x, y in rotated_sites
            ),
            turns,
        )
    raise ValueError("cluster shapes are not related by a C4 rotation.")


def _rotate_coordinate(site, turns):
    """Rotate a coordinate around the origin by ``turns`` quarter turns."""
    x, y = site
    for _ in range(turns % 4):
        x, y = -y, x
    return x, y


def _transform_tree_factorization(
    local_tensors,
    parent,
    parent_direction,
    children,
    source_to_target,
    turns,
):
    """Transport a tree factorization to a C4-related site ordering."""
    target_parent = {}
    target_direction = {}
    target_children = {}
    target_tensors = {}
    for source, source_parent in parent.items():
        target = source_to_target[source]
        target_parent[target] = (
            None
            if source_parent is None
            else source_to_target[source_parent]
        )
        if source_parent is not None:
            target_direction[target] = _rotate_direction(
                parent_direction[source],
                turns,
            )
        source_children, tensor = local_tensors[source]
        target_child_nodes = tuple(source_to_target[child] for child in source_children)
        target_children[target] = target_child_nodes
        target_tensors[target] = (target_child_nodes, tensor)
    return target_tensors, target_parent, target_direction, target_children


def _add_tree_factor_blocks(
    blocks,
    site_directions,
    embeddings,
    local_tensors,
    parent,
    tree_directions,
    sectors,
    local_dim,
):
    """Insert a tree-factorized cluster at all valid translations."""
    matrix_units = np.eye(local_dim**2).reshape(local_dim**2, local_dim, local_dim)
    root = next(site for site, value in parent.items() if value is None)
    for embedding in embeddings:
        for site, (child_nodes, tensor) in local_tensors.items():
            physical_axis = len(child_nodes)
            local_blocks = np.tensordot(
                tensor,
                matrix_units,
                axes=([physical_axis], [0]),
            )
            lattice_site = embedding[site]
            directions = site_directions[lattice_site]
            for active_indices in np.ndindex(local_blocks.shape[:-2]):
                key = [0] * len(directions)
                for axis, child in enumerate(child_nodes):
                    direction = tree_directions[(site, child)]
                    key[directions.index(direction)] = sectors[child][
                        active_indices[axis]
                    ]
                if site != root:
                    direction = tree_directions[(site, parent[site])]
                    key[directions.index(direction)] = sectors[site][
                        active_indices[-1]
                    ]
                key = tuple(key)
                block = local_blocks[active_indices]
                if key in blocks[lattice_site]:
                    blocks[lattice_site][key] = blocks[lattice_site][key] + block
                else:
                    blocks[lattice_site][key] = block


def _add_tree_factor_blocks_backend(
    blocks,
    site_directions,
    embeddings,
    local_tensors,
    parent,
    tree_directions,
    sectors,
    local_dim,
):
    """Insert backend-valued tree factors without detaching their graph."""
    reference = next(
        tensor
        for _children, tensor in local_tensors.values()
    )
    matrix_units = _as_backend(
        np.eye(local_dim**2).reshape(local_dim**2, local_dim, local_dim),
        like=reference,
        dtype=reference.dtype,
    )
    root = next(site for site, value in parent.items() if value is None)
    pending = {site: {} for site in site_directions}
    for embedding in embeddings:
        for site, (child_nodes, tensor) in local_tensors.items():
            physical_axis = len(child_nodes)
            local_blocks = ar.do(
                "tensordot",
                tensor,
                matrix_units,
                axes=([physical_axis], [0]),
            )
            lattice_site = embedding[site]
            directions = site_directions[lattice_site]
            for active_indices in np.ndindex(local_blocks.shape[:-2]):
                key = [0] * len(directions)
                for axis, child in enumerate(child_nodes):
                    direction = tree_directions[(site, child)]
                    key[directions.index(direction)] = sectors[child][
                        active_indices[axis]
                    ]
                if site != root:
                    direction = tree_directions[(site, parent[site])]
                    key[directions.index(direction)] = sectors[site][
                        active_indices[-1]
                    ]
                key = tuple(key)
                block = local_blocks[active_indices]
                pending[lattice_site].setdefault(key, []).append(block)
    for site, site_blocks in pending.items():
        for key, contributions in site_blocks.items():
            if len(contributions) == 1:
                block = contributions[0]
            else:
                block = ar.do(
                    "sum",
                    ar.do("stack", tuple(contributions), axis=0),
                    axis=0,
                )
            if key in blocks[site]:
                blocks[site][key] = ar.do("add", blocks[site][key], block)
            else:
                blocks[site][key] = block


def _add_graph_factor_blocks(
    blocks,
    site_directions,
    embeddings,
    edges,
    local_tensors,
    sectors,
    local_dim,
):
    """Insert a tree- or loop-fitted cluster factorization into PEPO blocks."""
    matrix_units = np.eye(local_dim**2).reshape(local_dim**2, local_dim, local_dim)
    incident = _cluster_incident_edges(edges, len(local_tensors))
    for embedding in embeddings:
        for site, (edge_ids, tensor) in local_tensors.items():
            local_blocks = np.tensordot(
                tensor,
                matrix_units,
                axes=([-1], [0]),
            )
            lattice_site = embedding[site]
            directions = site_directions[lattice_site]
            edge_directions = dict(incident[site])
            for active_indices in np.ndindex(local_blocks.shape[:-2]):
                key = [0] * len(directions)
                for axis, edge_index in enumerate(edge_ids):
                    direction = edge_directions[edge_index]
                    try:
                        direction_axis = directions.index(direction)
                    except ValueError as exc:
                        raise ValueError(
                            "cluster edge does not map to an available PEPO leg."
                        ) from exc
                    key[direction_axis] = sectors[edge_index][
                        active_indices[axis]
                    ]
                key = tuple(key)
                block = local_blocks[active_indices]
                if key in blocks[lattice_site]:
                    blocks[lattice_site][key] = blocks[lattice_site][key] + block
                else:
                    blocks[lattice_site][key] = block


def _contract_active_support(active, sites, edges):
    """Contract an active PEPO on a finite support with zero boundary legs."""
    internal_directions = {site: set() for site in sites}
    edge_labels = {}
    for edge_label, (source, target, direction) in enumerate(edges):
        source_site = sites[source]
        target_site = sites[target]
        internal_directions[source_site].add(direction)
        internal_directions[target_site].add(_OPPOSITE_DIRECTION[direction])
        edge_labels[(source_site, direction)] = edge_label
        edge_labels[(target_site, _OPPOSITE_DIRECTION[direction])] = edge_label

    local_dim = active.physical_dim
    bond_dim = active.bond_dim
    next_physical_label = len(edges)
    operands = []
    row_labels = []
    column_labels = []
    for site_index, site in enumerate(sites):
        directions = active.site_directions[site]
        role_directions = tuple(
            direction
            for direction in directions
            if direction in internal_directions[site]
        )
        labels = [edge_labels[(site, direction)] for direction in role_directions]
        row_label = next_physical_label
        column_label = next_physical_label + 1
        next_physical_label += 2
        labels.extend((row_label, column_label))
        row_labels.append(row_label)
        column_labels.append(column_label)
        factor = np.zeros(
            (bond_dim,) * len(role_directions) + (local_dim, local_dim),
            dtype=next(iter(active.blocks[site].values())).dtype,
        )
        directions_to_axis = {direction: axis for axis, direction in enumerate(directions)}
        for key, block in active.blocks[site].items():
            if any(
                key[axis] != 0
                for direction, axis in directions_to_axis.items()
                if direction not in internal_directions[site]
            ):
                continue
            sector_indices = tuple(
                key[directions_to_axis[direction]]
                for direction in role_directions
            )
            factor[sector_indices] += block
        operands.extend((factor, labels))

    result = np.einsum(
        *operands,
        row_labels + column_labels,
        optimize=True,
    )
    return result.reshape(local_dim**len(sites), local_dim**len(sites))


def _contract_active_support_backend(active, sites, edges):
    """Backend-preserving active-support contraction with zero boundaries."""
    internal_directions = {site: set() for site in sites}
    edge_labels = {}
    for edge_label, (source, target, direction) in enumerate(edges):
        source_site = sites[source]
        target_site = sites[target]
        internal_directions[source_site].add(direction)
        internal_directions[target_site].add(_OPPOSITE_DIRECTION[direction])
        edge_labels[(source_site, direction)] = edge_label
        edge_labels[(target_site, _OPPOSITE_DIRECTION[direction])] = edge_label

    local_dim = active.physical_dim
    reference = next(iter(active.blocks[sites[0]].values()))

    def sum_values(values):
        values = list(values)
        if not values:
            return ar.do("zeros", (local_dim, local_dim), like=reference)
        while len(values) > 1:
            values = [
                ar.do("add", values[index], values[index + 1])
                for index in range(0, len(values) - 1, 2)
            ] + (values[-1:] if len(values) % 2 else [])
        return values[0]

    # Each factor is a sparse map from its incident virtual-sector tuple to a
    # physical matrix.  Keeping this representation sparse avoids creating a
    # ``bond_dim**4`` site tensor at a four-leg PBC vertex.
    factors = []
    for site in sites:
        directions = active.site_directions[site]
        role_directions = tuple(
            direction
            for direction in directions
            if direction in internal_directions[site]
        )
        role_edges = tuple(
            edge_labels[(site, direction)] for direction in role_directions
        )
        direction_to_axis = {
            direction: axis for axis, direction in enumerate(directions)
        }
        entries = {}
        for key, block in active.blocks[site].items():
            if any(
                key[axis] != 0
                for direction, axis in direction_to_axis.items()
                if direction not in internal_directions[site]
            ):
                continue
            sector_indices = tuple(
                key[direction_to_axis[direction]]
                for direction in role_directions
            )
            entries.setdefault(sector_indices, []).append(block)
        factors.append(
            {
                "edges": role_edges,
                "sites": (site,),
                "entries": {
                    sector_indices: sum_values(values)
                    for sector_indices, values in entries.items()
                },
            }
        )

    remaining_edges = set(range(len(edges)))
    while remaining_edges:
        # Eliminate the edge with the smallest resulting factor scope. This
        # is a tiny min-fill heuristic and is particularly useful for PBC
        # plaquettes and five-site loops.
        candidates = []
        for edge_label in remaining_edges:
            containing = [
                index
                for index, factor in enumerate(factors)
                if edge_label in factor["edges"]
            ]
            if not containing:
                raise ValueError("active support lost a cluster virtual edge.")
            if len(containing) == 1:
                scope_size = len(factors[containing[0]]["edges"])
            else:
                scope_size = len(
                    set(factors[containing[0]]["edges"])
                    | set(factors[containing[1]]["edges"])
                )
            candidates.append((scope_size, edge_label, containing))
        _scope_size, edge_label, containing = min(candidates)
        remaining_edges.remove(edge_label)

        if len(containing) == 1:
            factor = factors[containing[0]]
            position = factor["edges"].index(edge_label)
            new_edges = tuple(
                edge for index, edge in enumerate(factor["edges"])
                if index != position
            )
            reduced = {}
            for key, value in factor["entries"].items():
                new_key = key[:position] + key[position + 1 :]
                reduced.setdefault(new_key, []).append(value)
            factors[containing[0]] = {
                "edges": new_edges,
                "sites": factor["sites"],
                "entries": {
                    key: sum_values(values) for key, values in reduced.items()
                },
            }
            continue

        first_index, second_index = containing
        first = factors[first_index]
        second = factors[second_index]
        first_position = first["edges"].index(edge_label)
        second_position = second["edges"].index(edge_label)
        common_edges = tuple(
            edge
            for edge in first["edges"]
            if edge in second["edges"]
        )
        first_groups = {}
        for key, value in first["entries"].items():
            group_key = tuple(key[first["edges"].index(edge)] for edge in common_edges)
            first_groups.setdefault(group_key, []).append((key, value))
        second_groups = {}
        for key, value in second["entries"].items():
            group_key = tuple(key[second["edges"].index(edge)] for edge in common_edges)
            second_groups.setdefault(group_key, []).append((key, value))

        new_edges = tuple(
            edge for edge in first["edges"] if edge != edge_label
        ) + tuple(
            edge
            for edge in second["edges"]
            if edge != edge_label and edge not in first["edges"]
        )
        first_positions = {
            edge: first["edges"].index(edge)
            for edge in first["edges"]
            if edge != edge_label
        }
        second_positions = {
            edge: second["edges"].index(edge)
            for edge in second["edges"]
            if edge != edge_label
        }
        merged = {}
        for group_key, first_items in first_groups.items():
            second_items = second_groups.get(group_key, ())
            for first_key, first_value in first_items:
                for second_key, second_value in second_items:
                    if first_key[first_position] != second_key[second_position]:
                        continue
                    merged_key = tuple(
                        first_key[first_positions[edge]]
                        if edge in first_positions
                        else second_key[second_positions[edge]]
                        for edge in new_edges
                    )
                    value = ar.do(
                        "tensordot",
                        first_value,
                        second_value,
                        axes=0,
                    )
                    merged.setdefault(merged_key, []).append(value)
        merged_factor = {
            "edges": new_edges,
            "sites": first["sites"] + second["sites"],
            "entries": {
                key: sum_values(values) for key, values in merged.items()
            },
        }
        for index in sorted((first_index, second_index), reverse=True):
            factors.pop(index)
        factors.append(merged_factor)

    if len(factors) != 1:
        raise ValueError("active support contraction did not produce one factor.")
    final_factor = factors[0]
    value = final_factor["entries"].get(())
    if value is None:
        value = sum_values(tuple(final_factor["entries"].values()))
    site_order = final_factor["sites"]
    pair_permutation = tuple(
        axis
        for site in sites
        for axis in (
            2 * site_order.index(site),
            2 * site_order.index(site) + 1,
        )
    )
    if pair_permutation != tuple(range(2 * len(sites))):
        value = ar.do("transpose", value, pair_permutation)
    return _backend_operator_from_tensor(value, len(sites), local_dim)


def _add_generic_cluster_levels(
    blocks,
    allocator,
    plan,
    one_site_exp,
    twosite_op,
    onesite_op,
    beta,
):
    """Add generic connected-cluster corrections from P=5 through P."""
    residuals_by_order = {}
    targets_by_order = {}
    ranks_by_order = {}
    counts_by_order = {}
    loop_ranks_by_order = {}
    generic_shapes = generate_connected_cluster_shapes(
        plan.order,
        min_sites=5,
        quotient_rotations=plan.symmetry == "C4",
    )
    shapes_by_order = {}
    for shape in generic_shapes:
        shapes_by_order.setdefault(shape.nsites, []).append(shape)

    for cluster_order in range(5, plan.order + 1):
        level_residuals = []
        level_targets = []
        level_ranks = []
        level_loop_ranks = []
        cluster_count = 0
        # Each order subtracts the complete lower-order active PEPO. This is
        # important once P exceeds five: P=6 must include the P=5 correction
        # before solving its own residual, rather than jumping from P=4.
        lower_active = ActivePEPOBlocks(
            lx=plan.lx,
            ly=plan.ly,
            cyclic=plan.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=one_site_exp.shape[0],
            site_directions=plan.site_directions,
            blocks={
                site: dict(site_blocks)
                for site, site_blocks in blocks.items()
            },
        )
        for shape in shapes_by_order.get(cluster_order, ()):
            shape_variants = (
                _cluster_shape_c4_orbit(shape)
                if plan.symmetry == "C4"
                else ((shape, tuple(range(shape.nsites)), 0),)
            )
            variant_data = tuple(
                (
                    variant,
                    source_to_target,
                    turns,
                    _cluster_shape_embeddings(
                        variant,
                        plan.lx,
                        plan.ly,
                        plan.cyclic,
                    ),
                )
                for variant, source_to_target, turns in shape_variants
            )
            variant_data = tuple(data for data in variant_data if data[3])
            if not variant_data:
                continue

            # Pick an orientation that actually embeds in the finite lattice
            # as the solve representative. On rectangular lattices a rotated
            # representative may have no embedding, while its partner does.
            source_shape, _, _, source_embeddings = variant_data[0]
            source_hamiltonian = _tree_hamiltonian(
                source_shape.nsites,
                source_shape.edges,
                twosite_op,
                onesite_op,
            )
            source_exact = _expm(
                -beta * source_hamiltonian,
                np.asarray(one_site_exp).dtype,
            )
            source_lower = _contract_active_support(
                lower_active,
                source_embeddings[0],
                source_shape.edges,
            )
            source_residual = source_exact - source_lower
            source_norm = float(np.linalg.norm(source_residual))
            cluster_count += sum(len(data[3]) for data in variant_data)
            if source_norm == 0.0:
                continue
            factorization_cache = {}

            for variant, _, _, embeddings in variant_data:
                local_hamiltonian = _tree_hamiltonian(
                    variant.nsites,
                    variant.edges,
                    twosite_op,
                    onesite_op,
                )
                exact = _expm(
                    -beta * local_hamiltonian,
                    np.asarray(one_site_exp).dtype,
                )
                local_lower = _contract_active_support(
                    lower_active,
                    embeddings[0],
                    variant.edges,
                )
                residual = exact - local_lower
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm == 0.0:
                    continue

                if plan.fit_method is not None:
                    fit_method = (
                        "als"
                        if plan.fit_method == "als" or not variant.is_tree
                        else "tree"
                    )
                    fit_seed = plan.fit_seed
                    if fit_seed is not None:
                        fit_seed += cluster_order + sum(
                            (index + 1) * (x + 17 * y)
                            for index, (x, y) in enumerate(variant.sites)
                        )
                    rank_schedule = _quimb_loop_rank_schedule(
                        one_site_exp.shape[0],
                        max_loop_rank=plan.max_loop_rank,
                        max_tree_rank=plan.max_tree_rank,
                        adaptive=(
                            plan.adaptive_loop_rank
                            and fit_method == "als"
                            and not variant.is_tree
                        ),
                        loop_rank_start=plan.loop_rank_start,
                        loop_rank_step=plan.loop_rank_step,
                    )
                    fitted = None
                    warm_start = None
                    for loop_rank in rank_schedule:
                        candidate = _quimb_factorize_operator(
                            residual,
                            variant.edges,
                            variant.nsites,
                            one_site_exp.shape[0],
                            method=fit_method,
                            max_tree_rank=plan.max_tree_rank,
                            max_loop_rank=loop_rank,
                            fit_steps=plan.fit_steps,
                            fit_tol=plan.fit_tol,
                            fit_solver_maxiter=plan.fit_solver_maxiter,
                            fit_seed=fit_seed,
                            warm_start=warm_start
                            if plan.fit_warm_start
                            else None,
                        )
                        if candidate is None:
                            break
                        fitted = candidate
                        if plan.fit_warm_start:
                            warm_start = (candidate[0], candidate[1])
                        factorization_error, factorization_target = candidate[-2:]
                        if (
                            len(rank_schedule) == 1
                            or factorization_error
                            <= plan.fit_tol * max(factorization_target, np.finfo(float).eps)
                        ):
                            break
                    if fitted is None:
                        continue
                    (
                        local_tensors,
                        edge_ranks,
                        cluster_rank,
                        factorization_error,
                        factorization_target,
                    ) = fitted
                    sectors = {
                        edge_index: allocator.allocate(rank)
                        for edge_index, rank in edge_ranks.items()
                    }
                    _add_graph_factor_blocks(
                        blocks,
                        plan.site_directions,
                        embeddings,
                        variant.edges,
                        local_tensors,
                        sectors,
                        one_site_exp.shape[0],
                    )
                    level_residuals.append(float(factorization_error))
                    level_targets.append(float(factorization_target))
                    level_ranks.append(cluster_rank)
                    if not variant.is_tree:
                        level_loop_ranks.append(
                            max(
                                edge_ranks[edge_index]
                                for edge_index in range(len(variant.edges))
                                if edge_index
                                not in _spanning_tree_edge_indices(
                                    variant.edges,
                                    variant.nsites,
                                )
                            )
                        )
                    continue

                source_to_target, turns = _shape_rotation_map(
                    source_shape,
                    variant,
                )
                site_order = tuple(np.argsort(np.asarray(source_to_target)))
                transformed = _permute_operator_sites(
                    source_residual,
                    site_order,
                    one_site_exp.shape[0],
                )
                use_transformed = np.allclose(
                    residual,
                    transformed,
                    rtol=1e-10,
                    atol=1e-12,
                )
                cache_key = (
                    source_shape.sites,
                    plan.max_tree_rank,
                )
                if use_transformed and cache_key in factorization_cache:
                    (
                        cached_tensors,
                        cached_parent,
                        cached_parent_direction,
                        cached_children,
                        cached_ranks,
                        tree_rank,
                        factorization_error,
                        factorization_target,
                    ) = factorization_cache[cache_key]
                    (
                        local_tensors,
                        parent,
                        parent_direction,
                        children,
                    ) = _transform_tree_factorization(
                        cached_tensors,
                        cached_parent,
                        cached_parent_direction,
                        cached_children,
                        source_to_target,
                        turns,
                    )
                    ranks = {
                        source_to_target[source]: rank
                        for source, rank in cached_ranks.items()
                    }
                else:
                    (
                        local_tensors,
                        parent,
                        parent_direction,
                        children,
                        ranks,
                        tree_rank,
                        factorization_error,
                        factorization_target,
                    ) = _tree_factorize_operator(
                        residual,
                        variant.edges,
                        variant.nsites,
                        one_site_exp.shape[0],
                        plan.max_tree_rank,
                    )
                    if local_tensors is None:
                        continue
                    if use_transformed:
                        factorization_cache[cache_key] = (
                            local_tensors,
                            parent,
                            parent_direction,
                            children,
                            ranks,
                            tree_rank,
                            factorization_error,
                            factorization_target,
                        )
                if local_tensors is None:
                    continue
                level_residuals.append(float(factorization_error))
                level_targets.append(float(factorization_target))
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
                _add_tree_factor_blocks(
                    blocks,
                    plan.site_directions,
                    embeddings,
                    local_tensors,
                    parent,
                    tree_directions,
                    sectors,
                    one_site_exp.shape[0],
                )
                level_ranks.append(tree_rank)

        residuals_by_order[cluster_order] = level_residuals
        targets_by_order[cluster_order] = level_targets
        ranks_by_order[cluster_order] = level_ranks
        counts_by_order[cluster_order] = cluster_count
        loop_ranks_by_order[cluster_order] = level_loop_ranks
    return (
        residuals_by_order,
        targets_by_order,
        ranks_by_order,
        counts_by_order,
        loop_ranks_by_order,
    )


@dataclass
class ClusterExpansionPlan:
    """Reusable geometry and symmetry plan for dense cluster-expansion PEPOs.

    The lattice topology and cluster-orbit bookkeeping are cached in the plan;
    beta-dependent local exponentials and residual solves are performed by
    :meth:`build`. Set ``materialize=False`` to retain the sparse active-block
    representation instead of immediately creating dense Quimb tensors.
    Through order four this includes the specialized tree and plaquette
    implementations. Orders five through nine use the generic
    connected-subcluster residual and topology-aware spanning-tree
    factorization. With ``symmetry="C4"``, rotated generic shapes reuse a
    transported factorization when the lower-order residual has the same
    symmetry. Set ``fit_method="quimb"`` to use Quimb tree fitting for
    generic tree clusters and complex ALS fitting for generic loop clusters.
    This is a numerical, non-differentiable path intended for
    coefficient-dependent real-time or imaginary-time builds.
    """

    lx: int
    ly: int
    twosite_op: np.ndarray
    onesite_op: np.ndarray
    order: int = 3
    cyclic: bool | tuple[bool, bool] = False
    edge_cutoff: float = 0.0
    max_edge_rank: int | None = None
    max_tree_rank: int | None = None
    max_loop_rank: int | None = None
    symmetry: str | None = None
    internal_symmetry: ClusterInternalSymmetry | None = None
    dtype: object | None = None
    fit_method: str | None = None
    fit_steps: int = 100
    fit_tol: float = 1e-10
    fit_solver_maxiter: int = 8
    fit_seed: int | None = 0
    adaptive_loop_rank: bool = False
    loop_rank_start: int | None = None
    loop_rank_step: int = 2
    fit_warm_start: bool = True
    last_report: ClusterExpansionReport | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.lx = _validate_shape(self.lx, "lx")
        self.ly = _validate_shape(self.ly, "ly")
        self.cyclic = _validate_cyclic(self.cyclic, self.lx, self.ly)
        if not isinstance(self.order, Integral):
            raise TypeError("order must be an integer.")
        self.order = int(self.order)
        if self.order < 1:
            raise ValueError("order must be >= 1.")
        if self.order > 9:
            raise NotImplementedError(
                "dense cluster-expansion PEPOs currently support orders 1 through 9; "
                "higher-level clusters are not implemented yet."
            )
        if self.edge_cutoff < 0.0:
            raise ValueError("edge_cutoff must be >= 0.")
        if self.max_edge_rank is not None:
            self.max_edge_rank = _validate_shape(self.max_edge_rank, "max_edge_rank")
        if self.max_tree_rank is not None:
            self.max_tree_rank = _validate_shape(self.max_tree_rank, "max_tree_rank")
        if self.max_loop_rank is not None:
            self.max_loop_rank = _validate_shape(self.max_loop_rank, "max_loop_rank")
        if not isinstance(self.adaptive_loop_rank, (bool, np.bool_)):
            raise TypeError("adaptive_loop_rank must be a bool.")
        if self.loop_rank_start is not None:
            self.loop_rank_start = _validate_shape(
                self.loop_rank_start,
                "loop_rank_start",
            )
        self.loop_rank_step = _validate_shape(self.loop_rank_step, "loop_rank_step")
        if not isinstance(self.fit_warm_start, (bool, np.bool_)):
            raise TypeError("fit_warm_start must be a bool.")
        if self.symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")
        if self.fit_method not in (None, "quimb", "tree", "als"):
            raise ValueError(
                "fit_method must be None, 'quimb', 'tree', or 'als'."
            )
        self.fit_steps = _validate_shape(self.fit_steps, "fit_steps")
        self.fit_solver_maxiter = _validate_shape(
            self.fit_solver_maxiter,
            "fit_solver_maxiter",
        )
        if self.fit_tol <= 0.0:
            raise ValueError("fit_tol must be > 0.")
        if self.fit_seed is not None:
            if not isinstance(self.fit_seed, Integral) or self.fit_seed < 0:
                raise ValueError("fit_seed must be a non-negative integer or None.")
            self.fit_seed = int(self.fit_seed)

        self.onesite_op = _as_square_operator(self.onesite_op, "onesite_op", dtype=self.dtype)
        local_dim = self.onesite_op.shape[0]
        self.twosite_op = _as_square_operator(self.twosite_op, "twosite_op", dtype=self.dtype)
        if self.twosite_op.shape != (local_dim**2, local_dim**2):
            raise ValueError(
                "twosite_op must have shape "
                f"({local_dim**2}, {local_dim**2}) for local dimension {local_dim}."
            )
        self.internal_symmetry = _coerce_internal_symmetry(self.internal_symmetry)
        if self.internal_symmetry is not None:
            self.internal_symmetry.validate(self.twosite_op, self.onesite_op)
        if self.dtype is None:
            self.dtype = np.result_type(self.onesite_op.dtype, self.twosite_op.dtype)
        if self.symmetry == "C4":
            self.dtype = np.result_type(self.dtype, np.complex128)
        self.onesite_op = np.asarray(self.onesite_op, dtype=self.dtype)
        self.twosite_op = np.asarray(self.twosite_op, dtype=self.dtype)

        if self.symmetry == "C4":
            swapped = _swap_two_site_operator(self.twosite_op, local_dim)
            if not np.allclose(self.twosite_op, swapped, rtol=1e-10, atol=1e-12):
                raise ValueError("C4 cluster reduction requires a site-symmetric twosite_op.")

        self.site_directions = {
            (i, j): _site_directions(i, j, self.lx, self.ly, *self.cyclic)
            for i in range(self.lx)
            for j in range(self.ly)
        }
        self.pair_orbits = _pair_orbits() if self.symmetry == "C4" else tuple(
            (pair, (pair,)) for pair in _all_direction_pairs()
        )
        self.triple_orbits = _subset_orbits(3, self.symmetry)
        self.path_orbits = _path_orbits(self.symmetry)
        self.plaquette_starts = _plaquette_starts(
            self.lx,
            self.ly,
            self.cyclic,
        )

    @property
    def pair_representatives(self):
        """Return the connected three-site pair orientations to solve."""
        if self.order < 3:
            return ()
        return tuple(representative for representative, _ in self.pair_orbits)

    @property
    def tree_representatives(self):
        """Return the order-four star and path representatives."""
        if self.order < 4:
            return ()
        stars = tuple(representative for representative, _ in self.triple_orbits)
        paths = tuple(representative for representative, _ in self.path_orbits)
        return stars + paths

    @property
    def connected_cluster_shapes(self):
        """Return the generic geometry inventory through ``self.order``.

        This is an inspection and planning surface for the generic higher-order
        residual solver. The specialized order-four orbit metadata remains
        separate from the generic spanning-tree factorization.
        """
        return generate_connected_cluster_shapes(
            self.order,
            quotient_rotations=self.symmetry == "C4",
        )

    def build(self, beta, *, materialize=True, return_report=False):
        """Build at ``beta`` using the cached topology and symmetry plan."""
        work_dtype = np.result_type(self.dtype, np.asarray(beta).dtype)
        onesite_op = np.asarray(self.onesite_op, dtype=work_dtype)
        twosite_op = np.asarray(self.twosite_op, dtype=work_dtype)
        one_site_exp, start_factors, end_factors = _edge_factors(
            twosite_op,
            onesite_op,
            beta,
            edge_cutoff=self.edge_cutoff,
            max_edge_rank=self.max_edge_rank,
            symmetric=self.symmetry == "C4",
        )
        if self.order < 2:
            start_factors = np.zeros(
                (0, one_site_exp.shape[0], one_site_exp.shape[1]), dtype=work_dtype
            )
            end_factors = start_factors.copy()

        pair_tensors = {}
        pair_residuals = []
        pair_targets = []
        if self.order >= 3 and start_factors.shape[0]:
            for representative, orbit in self.pair_orbits:
                tensor, residual_norm, target_norm = _solve_three_site_pair(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    beta,
                )
                pair_tensors[representative] = tensor
                pair_residuals.append(residual_norm)
                pair_targets.append(target_norm)
                for pair in orbit[1:]:
                    pair_tensors[pair] = _rotate_pair_tensor(
                        representative,
                        pair,
                        tensor,
                    )

        blocks = _initialize_blocks(
            self.lx, self.ly, one_site_exp, self.site_directions
        )
        allocator = _SectorAllocator()
        edge_sectors = allocator.allocate(start_factors.shape[0])
        if edge_sectors:
            _add_positive_edge_channels(
                blocks,
                self.site_directions,
                start_factors,
                end_factors,
                edge_sectors,
            )
            for pair, tensor in pair_tensors.items():
                _add_pair_blocks(
                    blocks,
                    self.site_directions,
                    pair,
                    tensor,
                    edge_sectors,
                )

        star_residuals = []
        star_targets = []
        path_residuals = []
        path_targets = []
        path_ranks = []
        loop_residuals = []
        loop_targets = []
        loop_rank = 0
        solved_tree_groups = 0
        if self.order >= 4 and start_factors.shape[0]:
            star_tensors = {}
            for representative, orbit in self.triple_orbits:
                if not any(
                    all(direction in directions for direction in star)
                    for star in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                tensor, residual_norm, target_norm = _solve_four_star(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                    beta,
                )
                star_tensors[representative] = tensor
                star_residuals.append(residual_norm)
                star_targets.append(target_norm)
                solved_tree_groups += 1
                for directions in orbit[1:]:
                    star_tensors[directions] = _rotate_direction_tensor(
                        representative, directions, tensor
                    )

            for directions, tensor in star_tensors.items():
                sectors = allocator.allocate(start_factors.shape[0])
                _add_triple_blocks(
                    blocks, self.site_directions, directions, tensor, sectors
                )
                for direction in directions:
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        _OPPOSITE_DIRECTION[direction],
                        sectors,
                        end_factors
                        if direction in _POSITIVE_DIRECTIONS
                        else start_factors,
                        source=False,
                        cyclic=self.cyclic,
                    )

            for representative, orbit in self.path_orbits:
                if not any(
                    _path_start_sites(steps, self.lx, self.ly, self.cyclic)
                    for steps in orbit
                ):
                    continue
                left, right, residual_norm, target_norm = _solve_four_path(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                    beta,
                    self.max_tree_rank,
                )
                path_residuals.append(residual_norm)
                path_targets.append(target_norm)
                path_ranks.append(left.shape[1])
                solved_tree_groups += 1
                path_channel_rank = left.shape[1]
                if not left.shape[1]:
                    continue
                representative_left_dirs = tuple(
                    sorted(
                        (_OPPOSITE_DIRECTION[representative[0]], representative[1]),
                        key=_DIRECTIONS.index,
                    )
                )
                representative_left_role = (
                    _OPPOSITE_DIRECTION[representative[0]],
                    representative[1],
                )
                representative_right_dirs = tuple(
                    sorted(
                        (_OPPOSITE_DIRECTION[representative[1]], representative[2]),
                        key=_DIRECTIONS.index,
                    )
                )
                representative_right_role = (
                    _OPPOSITE_DIRECTION[representative[1]],
                    representative[2],
                )
                if representative_left_role != representative_left_dirs:
                    left = left.transpose(1, 0, 2, 3)
                if representative_right_role != representative_right_dirs:
                    right = right.transpose(1, 0, 2, 3)
                for steps in orbit:
                    if steps == representative:
                        rotated_left, rotated_right = left, right
                    else:
                        target_left_dirs = tuple(
                            sorted(
                                (_OPPOSITE_DIRECTION[steps[0]], steps[1]),
                                key=_DIRECTIONS.index,
                            )
                        )
                        target_right_dirs = tuple(
                            sorted(
                                (_OPPOSITE_DIRECTION[steps[1]], steps[2]),
                                key=_DIRECTIONS.index,
                            )
                        )
                        rotated_left = _rotate_direction_tensor(
                            representative_left_dirs, target_left_dirs, left
                        )
                        rotated_right = _rotate_direction_tensor(
                            representative_right_dirs, target_right_dirs, right
                        )
                    first_sectors = allocator.allocate(start_factors.shape[0])
                    middle_sectors = allocator.allocate(path_channel_rank)
                    last_sectors = allocator.allocate(end_factors.shape[0])
                    first_factor = (
                        start_factors
                        if steps[0] in _POSITIVE_DIRECTIONS
                        else end_factors
                    )
                    last_factor = (
                        end_factors
                        if steps[2] in _POSITIVE_DIRECTIONS
                        else start_factors
                    )
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        steps[0],
                        first_sectors,
                        first_factor,
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
                        last_factor,
                        source=False,
                        cyclic=self.cyclic,
                    )
                    back = _OPPOSITE_DIRECTION[steps[0]]
                    forward = steps[1]
                    left_role = (back, forward)
                    left_dirs = tuple(sorted(left_role, key=_DIRECTIONS.index))
                    left_tensor = rotated_left
                    if left_role != left_dirs:
                        left_tensor = left_tensor.transpose(1, 0, 2, 3)
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        left_role,
                        left_tensor,
                        (first_sectors, middle_sectors),
                    )
                    back = _OPPOSITE_DIRECTION[steps[1]]
                    forward = steps[2]
                    right_role = (back, forward)
                    right_dirs = tuple(sorted(right_role, key=_DIRECTIONS.index))
                    right_tensor = rotated_right
                    if right_role != right_dirs:
                        right_tensor = right_tensor.transpose(1, 0, 2, 3)
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        right_role,
                        right_tensor,
                        (middle_sectors, last_sectors),
                    )

            if self.plaquette_starts:
                loop_edges = _plaquette_edges()
                exact_loop, _ = _lower_loop_residual(
                    loop_edges,
                    twosite_op,
                    onesite_op,
                    beta,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                )
                loop_rank = one_site_exp.shape[0] ** 4
                for start in self.plaquette_starts:
                    upper = _site_after(
                        start,
                        "u",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    right = _site_after(
                        start,
                        "r",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    diagonal = _site_after(
                        upper,
                        "r",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    loop_sites = (start, upper, diagonal, right)
                    lower_loop = _cycle_active_operator(
                        blocks,
                        self.site_directions,
                        loop_sites,
                        one_site_exp.shape[0],
                    )
                    lower_loop = _permute_operator_sites(
                        lower_loop,
                        (0, 1, 3, 2),
                        one_site_exp.shape[0],
                    )
                    loop_residual = exact_loop - lower_loop
                    loop_tensor = _operator_tensor(
                        loop_residual,
                        4,
                        one_site_exp.shape[0],
                    )
                    loop_tensors = _dense_loop_tensors(
                        loop_tensor,
                        one_site_exp.shape[0],
                    )
                    loop_residuals.append(np.linalg.norm(loop_residual))
                    loop_targets.append(np.linalg.norm(loop_residual))
                    lower_bond = allocator.allocate(loop_rank)
                    right_bond = allocator.allocate(loop_rank)
                    upper_bond = allocator.allocate(loop_rank)
                    left_bond = allocator.allocate(loop_rank)
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

        generic_residuals = {}
        generic_targets = {}
        generic_ranks = {}
        generic_cluster_counts = {}
        generic_loop_ranks = {}
        if self.order >= 5:
            (
                generic_residuals,
                generic_targets,
                generic_ranks,
                generic_cluster_counts,
                generic_loop_ranks,
            ) = _add_generic_cluster_levels(
                blocks,
                allocator,
                self,
                one_site_exp,
                twosite_op,
                onesite_op,
                beta,
            )

        active = ActivePEPOBlocks(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=one_site_exp.shape[0],
            site_directions=self.site_directions,
            blocks=blocks,
            charge_symmetry=(
                None
                if self.internal_symmetry is None
                else self.internal_symmetry.name
            ),
            physical_sectors=(
                None
                if self.internal_symmetry is None
                else self.internal_symmetry.resolved_physical_sectors(
                    one_site_exp.shape[0]
                )
            ),
        )
        residual_norms = {}
        relative_residuals = {}
        if self.order >= 2:
            edge_residual = _edge_fit_residual(
                twosite_op,
                onesite_op,
                beta,
                one_site_exp,
                start_factors,
                end_factors,
            )
            edge_target = edge_residual + sum(
                (
                    np.kron(start_factors[index], end_factors[index])
                    for index in range(start_factors.shape[0])
                ),
                start=np.zeros_like(edge_residual),
            )
            residual_norms["edge"] = float(np.linalg.norm(edge_residual))
            relative_residuals["edge"] = float(
                np.linalg.norm(edge_residual) / max(np.linalg.norm(edge_target), np.finfo(float).eps)
            )
        if pair_residuals:
            residual_norms["three_site"] = float(max(pair_residuals))
            relative_residuals["three_site"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(pair_residuals, pair_targets)
                )
            )
        if star_residuals:
            residual_norms["four_site_star"] = float(max(star_residuals))
            relative_residuals["four_site_star"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(star_residuals, star_targets)
                )
            )
        if path_residuals:
            residual_norms["four_site_path"] = float(max(path_residuals))
            relative_residuals["four_site_path"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(path_residuals, path_targets)
                )
            )
        if loop_residuals:
            residual_norms["four_site_loop"] = float(max(loop_residuals))
            relative_residuals["four_site_loop"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(loop_residuals, loop_targets)
                )
            )
        for cluster_order, level_residuals in generic_residuals.items():
            if not level_residuals:
                continue
            generic_label = f"generic_order_{cluster_order}"
            level_targets = generic_targets[cluster_order]
            residual_norms[generic_label] = float(max(level_residuals))
            relative_residuals[generic_label] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(level_residuals, level_targets)
                )
            )
        generic_ranks_flat = tuple(
            rank
            for level_ranks in generic_ranks.values()
            for rank in level_ranks
        )
        generic_loop_ranks_flat = tuple(
            rank
            for level_ranks in generic_loop_ranks.values()
            for rank in level_ranks
        )
        counts = {
            "edge": sum(
                direction in _POSITIVE_DIRECTIONS
                for directions in self.site_directions.values()
                for direction in directions
            ),
            "three_site": (
                sum(
                    len(directions) * (len(directions) - 1) // 2
                    for directions in self.site_directions.values()
                )
                if self.order >= 3
                else 0
            ),
            "four_site_star": 0,
            "four_site_path": 0,
            "four_site_loop": len(self.plaquette_starts) if self.order >= 4 else 0,
            "four_site_tree_solved": (
                solved_tree_groups if self.order >= 4 else 0
            ),
            **{
                f"generic_order_{cluster_order}": count
                for cluster_order, count in generic_cluster_counts.items()
            },
            "generic_tree_solved": len(generic_ranks_flat),
            "generic_loop_solved": len(generic_loop_ranks_flat),
        }
        if self.order >= 4:
            for _, orbit in self.triple_orbits:
                counts["four_site_star"] += sum(
                    sum(all(direction in directions for direction in star) for directions in self.site_directions.values())
                    for star in orbit
                )
            for _, orbit in self.path_orbits:
                counts["four_site_path"] += sum(
                    len(_path_start_sites(steps, self.lx, self.ly, self.cyclic))
                    for steps in orbit
                )
        report = ClusterExpansionReport(
            beta=beta,
            order=self.order,
            local_dim=one_site_exp.shape[0],
            edge_rank=start_factors.shape[0],
            tree_rank=max((*path_ranks, *generic_ranks_flat), default=0),
            loop_rank=max((loop_rank, *generic_loop_ranks_flat), default=0),
            cluster_counts=counts,
            residual_norms=residual_norms,
            relative_residual_norms=relative_residuals,
            active_block_count=active.active_block_count,
            active_nbytes=active.active_nbytes,
            dense_nbytes=active.dense_nbytes,
            generic_loop_rank=max(generic_loop_ranks_flat, default=0),
        )
        self.last_report = report
        result = active.to_pepo() if materialize else active
        return (result, report) if return_report else result

    def build_composed(
        self,
        beta,
        *,
        composition="yoshida4",
        compress=False,
        **compress_opts,
    ):
        """Build a fractional-step composition of elementary PEPOs.

        ``composition="yoshida4"`` uses three order-three cluster-expansion
        PEPOs with coefficients ``(a, b, a)`` where

        ``a = 1 / (2 - 2**(1/3))`` and ``b = -2**(1/3) * a``.

        The order-three cluster expansion is accurate through second order,
        so this symmetric triple jump cancels its leading third-order error.
        The intermediate PEPOs are multiplied with Quimb's native
        :meth:`PEPO.apply`; no global dense operator is constructed. Virtual
        bonds grow multiplicatively unless ``compress=True`` is requested.

        Parameters
        ----------
        beta : scalar
            Target exponential convention ``exp(-beta * H)``.
        composition : {"yoshida4"}, optional
            Fractional-step composition policy.
        compress : bool, optional
            Whether to compress each intermediate Quimb PEPO product.
        compress_opts
            Options forwarded to Quimb when ``compress=True``.
        """
        if self.order != 3:
            raise ValueError(
                "the Yoshida fourth-order composition requires a plan with order=3."
            )
        if composition != "yoshida4":
            raise ValueError("composition must be 'yoshida4'.")

        layers = tuple(
            self.build(coefficient * beta, materialize=True)
            for coefficient in _yoshida4_coefficients()
        )
        return compose_pepo_layers(
            layers,
            compress=compress,
            **compress_opts,
        )


def build_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    twosite_op,
    onesite_op,
    *,
    order=3,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    max_loop_rank=None,
    dtype=None,
    symmetry=None,
    internal_symmetry=None,
    fit_method=None,
    fit_steps=100,
    fit_tol=1e-10,
    fit_solver_maxiter=8,
    fit_seed=0,
    adaptive_loop_rank=False,
    loop_rank_start=None,
    loop_rank_step=2,
    fit_warm_start=True,
    materialize=True,
    return_report=False,
):
    """Build a dense square-lattice PEPO cluster expansion.

    The PEPO represents an extensive approximation to
    ``exp(-beta * H)``.  ``onesite_op`` is the one-site Hamiltonian term and
    ``twosite_op`` is the canonical two-site term for a positive lattice
    direction (``u`` or ``r``).  The local two-site residual is factorized
    into virtual channels.  Order three additionally solves every connected
    three-site path and corner residual, filling the corresponding two-active
    virtual entries of the PEPO tensor. Order four adds four-site T-shaped
    and non-loop path tree residuals plus present plaquette-loop residuals.
    Orders five through nine add all finite-lattice connected shapes at the
    requested order using recursive lower-PEPO contraction and an
    SVD-factorized spanning tree. With ``symmetry="C4"``, rotated shapes
    reuse a transported factorization when their residuals are symmetry
    equivalent. Set ``fit_method="quimb"`` to fit generic tree residuals
    with Quimb's tree solver and generic loop residuals with complex ALS.
    Plaquettes use an exact fixed-rank tensor-ring factorization.

    Parameters
    ----------
    lx, ly : int
        Square-lattice dimensions.
    beta : scalar
        Real or complex imaginary-time step.
    twosite_op, onesite_op : array-like
        Dense square matrices of shapes ``(d**2, d**2)`` and ``(d, d)``.
    order : {1, 2, 3, 4, 5, 6, 7, 8, 9}, default=3
        Largest connected cluster size. Orders above nine require higher-level
        cluster construction.
    cyclic : bool or tuple[bool, bool], default=False
        Whether to close both lattice directions, or close x and y
        independently.
    edge_cutoff : float, default=0.0
        Relative singular-value cutoff for two-site residual channels.
    max_edge_rank : int | None, default=None
        Optional cap on the number of retained two-site channels.
    max_tree_rank : int | None, default=None
        Optional cap on internal SVD ranks of four-site paths and generic
        order-five through order-nine spanning-tree clusters.
    max_loop_rank : int | None, default=None
        Optional virtual rank for generic loop clusters when Quimb fitting is
        enabled. If omitted, the local operator rank is used as the loop
        ansatz rank.
    dtype : numpy dtype | None, default=None
        Optional dense dtype for all local tensors.
    symmetry : {None, "C4"}, default=None
        Reduce equivalent tree orientations using a symmetric virtual
        factorization. This is appropriate for square-lattice ITF and other
        site-symmetric C4 models.
    fit_method : {None, "quimb", "tree", "als"}, default=None
        Use Quimb fitting for generic order-five through order-nine clusters.
        ``"quimb"`` selects ``tree`` for tree shapes and ``als`` for loops.
        This numerical path is not coefficient-differentiable.
    fit_steps : int, default=100
        Maximum Quimb fitting sweeps per generic cluster.
    fit_tol : float, default=1e-10
        Quimb fitting stopping tolerance.
    fit_solver_maxiter : int, default=8
        Iterative local-solver iterations for Quimb ALS.
    fit_seed : int | None, default=0
        Seed for Quimb loop-ansatz initialization, or ``None`` for an
        unseeded initializer.
    adaptive_loop_rank : bool, default=False
        For generic loop clusters fitted with ALS, try increasing loop ranks
        from ``loop_rank_start`` through ``max_loop_rank`` until the local
        residual reaches ``fit_tol``. Tree clusters and fixed-rank fits are
        unaffected.
    loop_rank_start : int | None, default=None
        First loop rank for adaptive fitting. Defaults to one.
    loop_rank_step : int, default=2
        Increase between adaptive loop-rank trials. The configured maximum is
        always included as the final trial.
    fit_warm_start : bool, default=True
        Seed each larger adaptive ALS ansatz with the preceding fitted
        tensors. Disable this to independently initialize every rank.
    return_report : bool, default=False
        Return ``(pepo_or_active_blocks, ClusterExpansionReport)``.
    materialize : bool, default=True
        Return a dense Quimb PEPO when true, otherwise return active blocks.

    Returns
    -------
    quimb.tensor.PEPO
        A dense PEPO with physical ``b``/``k`` operator indices.

    Notes
    -----
    This implementation includes four-site plaquette loops, generic
    order-five through order-nine residual/factorization paths, and an active
    block/Symmray materialization boundary. Higher clusters and mixed-charge
    component splitting remain separate stages.
    """
    plan = ClusterExpansionPlan(
        lx,
        ly,
        twosite_op,
        onesite_op,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        max_loop_rank=max_loop_rank,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
        dtype=dtype,
        fit_method=fit_method,
        fit_steps=fit_steps,
        fit_tol=fit_tol,
        fit_solver_maxiter=fit_solver_maxiter,
        fit_seed=fit_seed,
        adaptive_loop_rank=adaptive_loop_rank,
        loop_rank_start=loop_rank_start,
        loop_rank_step=loop_rank_step,
        fit_warm_start=fit_warm_start,
    )
    return plan.build(
        beta,
        materialize=materialize,
        return_report=return_report,
    )


def build_model_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    model,
    *,
    order=3,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    max_loop_rank=None,
    dtype=None,
    symmetry=None,
    internal_symmetry=None,
    fit_method=None,
    fit_steps=100,
    fit_tol=1e-10,
    fit_solver_maxiter=8,
    fit_seed=0,
    adaptive_loop_rank=False,
    loop_rank_start=None,
    loop_rank_step=2,
    fit_warm_start=True,
    materialize=True,
    return_report=False,
):
    """Build a finite cluster PEPO from a dense model adapter.

    ``model`` may be a :class:`ClusterModelAdapter`, a mapping containing
    ``twosite_op`` and ``onesite_op`` (or their ``edge_op`` and ``onsite_op``
    aliases), or an object exposing those attributes.  If ``symmetry`` is
    omitted, the adapter's finite symmetry metadata is used.

    This convenience layer keeps model definitions separate from the dense
    cluster solver and is intentionally finite and non-fermionic.
    """
    adapter = adapt_cluster_model(
        model,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
    )
    if symmetry is None:
        symmetry = adapter.symmetry
    if internal_symmetry is None:
        internal_symmetry = adapter.internal_symmetry
    return build_cluster_expansion_pepo(
        lx,
        ly,
        beta,
        adapter.twosite_op,
        adapter.onesite_op,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        max_loop_rank=max_loop_rank,
        dtype=dtype,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
        fit_method=fit_method,
        fit_steps=fit_steps,
        fit_tol=fit_tol,
        fit_solver_maxiter=fit_solver_maxiter,
        fit_seed=fit_seed,
        adaptive_loop_rank=adaptive_loop_rank,
        loop_rank_start=loop_rank_start,
        loop_rank_step=loop_rank_step,
        fit_warm_start=fit_warm_start,
        materialize=materialize,
        return_report=return_report,
    )


def compose_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    twosite_op,
    onesite_op,
    *,
    order=3,
    composition="yoshida4",
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    dtype=None,
    symmetry=None,
    compress=False,
    **compress_opts,
):
    """Build a higher-order PEPO by composing cluster-expansion layers.

    The default Yoshida composition uses three order-three (second-order
    accurate) cluster-expansion PEPOs at fractional signed steps. It is a
    direct Quimb PEPO composition and is useful when a higher-order
    elementary cluster construction is less attractive than a few
    multiplicative layers. The resulting uncompressed virtual bond dimension
    grows with the number of layers; use ``compress=True`` with suitable
    Quimb compression options when that tradeoff is acceptable.

    ``order`` must be ``3`` because the fourth-order composition is derived
    for the order-three cluster expansion.
    """
    plan = ClusterExpansionPlan(
        lx,
        ly,
        twosite_op,
        onesite_op,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        symmetry=symmetry,
        dtype=dtype,
    )
    return plan.build_composed(
        beta,
        composition=composition,
        compress=compress,
        **compress_opts,
    )


def build_itf_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    *,
    J=1.0,
    field=1.0,
    order=3,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    max_loop_rank=None,
    dtype="float64",
    symmetry="C4",
    internal_symmetry=None,
    fit_method=None,
    fit_steps=100,
    fit_tol=1e-10,
    fit_solver_maxiter=8,
    fit_seed=0,
    adaptive_loop_rank=False,
    loop_rank_start=None,
    loop_rank_step=2,
    fit_warm_start=True,
    materialize=True,
    return_report=False,
):
    """Build a cluster-expansion PEPO for Pepsy's square-lattice ITF.

    The convention matches :meth:`pepsy.operators.ham_tn.build_itf`:
    ``H = J * sum Z_i Z_j + field * sum X_i``.
    """
    z = np.asarray(quimb.pauli("Z", dtype=dtype), dtype=dtype)
    x = np.asarray(quimb.pauli("X", dtype=dtype), dtype=dtype)
    return build_cluster_expansion_pepo(
        lx,
        ly,
        beta,
        J * np.kron(z, z),
        field * x,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        max_loop_rank=max_loop_rank,
        symmetry=symmetry,
        internal_symmetry=internal_symmetry,
        dtype=dtype,
        fit_method=fit_method,
        fit_steps=fit_steps,
        fit_tol=fit_tol,
        fit_solver_maxiter=fit_solver_maxiter,
        fit_seed=fit_seed,
        adaptive_loop_rank=adaptive_loop_rank,
        loop_rank_start=loop_rank_start,
        loop_rank_step=loop_rank_step,
        fit_warm_start=fit_warm_start,
        materialize=materialize,
        return_report=return_report,
    )

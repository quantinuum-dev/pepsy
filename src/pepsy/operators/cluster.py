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
component splitting remain separate extension points.
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

from .mpo_automaton import _as_backend, _backend_reference

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
        return np.asarray(quimb.expm(np.asarray(matrix)), dtype=np.asarray(matrix).dtype)
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


def _backend_dtype_itemsize(value):
    try:
        return np.dtype(value.dtype).itemsize
    except (TypeError, ValueError):
        bits = getattr(value.dtype, "bits", None)
        if bits is not None:
            return int(bits) // 8
        return np.asarray(value).dtype.itemsize


def _backend_nonzero(value):
    """Return whether a backend block contains any nonzero entries."""
    try:
        return bool(np.any(np.asarray(ar.to_numpy(value)) != 0))
    except (TypeError, ValueError, AttributeError):
        # A backend may not expose host conversion for a symbolic value. Such
        # a block is retained conservatively; dropping it would break the
        # autodiff graph and is less safe than carrying a zero block.
        return True


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


def _operator_from_tensor(operator_tensor, nsites, local_dim):
    """Undo :func:`_operator_tensor` for a local operator tensor."""
    pair_axes = tuple(range(2 * nsites))
    row_axes = pair_axes[::2]
    column_axes = pair_axes[1::2]
    return operator_tensor.reshape((local_dim, local_dim) * nsites).transpose(
        row_axes + column_axes
    ).reshape(local_dim**nsites, local_dim**nsites)


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


def _materialize_site_blocks(directions, blocks, bond_dim, dtype):
    physical_dim = blocks[(0,) * len(directions)].shape[0]
    reference = blocks[(0,) * len(directions)]
    shape = (bond_dim,) * len(directions) + (physical_dim, physical_dim)
    if ar.infer_backend(reference) not in ("builtins", "numpy"):
        data = ar.do("zeros", shape, like=reference)
        for key, block in blocks.items():
            mask = None
            for axis, sector in enumerate(key):
                selector = ar.do("eye", bond_dim, like=reference)[:, sector]
                selector_shape = [1] * len(directions)
                selector_shape[axis] = bond_dim
                selector = ar.do("reshape", selector, tuple(selector_shape))
                mask = selector if mask is None else ar.do("multiply", mask, selector)
            block = ar.do("transpose", block, (1, 0))
            block = ar.do(
                "reshape",
                block,
                (1,) * len(directions) + (physical_dim, physical_dim),
            )
            data = ar.do("add", data, ar.do("multiply", mask[..., None, None], block))
        return data

    data = np.zeros(shape, dtype=dtype)
    for key, block in blocks.items():
        # Quimb's ``to_dense`` convention transposes each local ``b``/``k``
        # block when flattening an operator. Store the inverse local
        # transpose here so the materialized PEPO has the requested matrix
        # orientation.
        data[key + (slice(None), slice(None))] = block.T
    return data


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


@dataclass
class ActivePEPOBlocks:
    """Sparse active virtual-sector blocks for a finite PEPO lattice.

    ``blocks[(i, j)]`` maps a tuple of virtual-sector integers to its physical
    operator block. Sector ``0`` is the trivial channel; positive sectors are
    compact active channels. The dense Quimb PEPO is created only by
    :meth:`to_pepo`, keeping the mostly-zero construction intermediate small.
    """

    lx: int
    ly: int
    cyclic: tuple[bool, bool]
    bond_dim: int
    physical_dim: int
    site_directions: dict
    blocks: dict
    charge_symmetry: str | None = None
    physical_sectors: dict | None = None
    virtual_sector_charges: dict | None = None

    @property
    def active_block_count(self):
        """Return the number of stored nonzero sector blocks."""
        return sum(len(site_blocks) for site_blocks in self.blocks.values())

    @property
    def dense_nbytes(self):
        """Estimate bytes required by dense PEPO site tensors."""
        reference = next(iter(next(iter(self.blocks.values())).values()))
        itemsize = _backend_dtype_itemsize(reference)
        return sum(
            self.bond_dim ** len(self.site_directions[site])
            * self.physical_dim**2
            * itemsize
            for site in self.blocks
        )

    @property
    def active_nbytes(self):
        """Return bytes occupied by the stored active blocks."""
        total = 0
        for site_blocks in self.blocks.values():
            for block in site_blocks.values():
                nbytes = getattr(block, "nbytes", None)
                total += int(nbytes if nbytes is not None else np.asarray(block).nbytes)
        return total

    def compact(self):
        """Remove zero blocks and globally orphaned virtual sectors.

        Sector ids are implementation labels, so compacting them is
        lossless. The relative order of surviving ids is preserved, which
        keeps repeated autodiff evaluations compatible with the same active
        topology while dropping channels that were never connected on the
        chosen finite lattice.
        """
        compact_blocks = {
            site: {
                key: block
                for key, block in site_blocks.items()
                if _backend_nonzero(block)
            }
            for site, site_blocks in self.blocks.items()
        }
        # A channel endpoint with no nonzero block on the opposite side of
        # its bond is an orphan.  Iterating to a fixed point also removes
        # higher-order blocks that became disconnected after their endpoint
        # channels were pruned.
        changed = True
        while changed:
            changed = False
            available = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                for axis, direction in enumerate(directions):
                    available[(site, direction)] = {
                        key[axis] for key in site_blocks
                    }
            retained = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                kept_site = {}
                for key, block in site_blocks.items():
                    keep = True
                    for axis, direction in enumerate(directions):
                        sector = key[axis]
                        if sector == 0:
                            continue
                        neighbor = _site_after(
                            site,
                            direction,
                            self.lx,
                            self.ly,
                            self.cyclic,
                        )
                        if neighbor is None:
                            keep = False
                            break
                        opposite = _OPPOSITE_DIRECTION[direction]
                        if sector not in available[(neighbor, opposite)]:
                            keep = False
                            break
                    if keep:
                        kept_site[key] = block
                    else:
                        changed = True
                retained[site] = kept_site
            compact_blocks = retained

        used = {0}
        for site_blocks in compact_blocks.values():
            for key in site_blocks:
                used.update(key)
        sector_map = {
            old: new for new, old in enumerate(sorted(used))
        }
        remapped_blocks = {
            site: {
                tuple(sector_map[sector] for sector in key): block
                for key, block in site_blocks.items()
            }
            for site, site_blocks in compact_blocks.items()
        }
        old_charges = self.virtual_sector_charges or {}
        remapped_charges = {
            sector_map[old]: old_charges.get(old, 0)
            for old in sorted(used)
        }
        return type(self)(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=len(used),
            physical_dim=self.physical_dim,
            site_directions=self.site_directions,
            blocks=remapped_blocks,
            charge_symmetry=self.charge_symmetry,
            physical_sectors=self.physical_sectors,
            virtual_sector_charges=remapped_charges,
        )

    remove_orphans = compact

    def to_symmray_pepo(
        self,
        *,
        symmetry=None,
        physical_sectors=None,
        virtual_charges=None,
        charge=0,
        fermionic=False,
        remove_orphans=True,
    ):
        """Materialize active blocks as a native Symmray-backed PEPO.

        ``virtual_charges`` maps the integer active-history ids to symmetry
        charges. Multiple history ids may share one charge; they become a
        proper Symmray degeneracy block rather than a dense virtual axis.
        Every nonzero local block is checked against the requested homogeneous
        operator ``charge``. This means a mixed-charge exponential (for
        example an unsplit ``exp(h X)`` under Z2) must be represented as
        separate charge components before conversion.

        The returned object is a Quimb PEPO whose site arrays are native
        Symmray arrays. Backend-valued blocks are sliced and assembled using
        Autoray operations, so Torch/JAX coefficient graphs are preserved.
        """
        from pepsy.tensors.symmetric import (  # pylint: disable=import-outside-toplevel
            _array_class_for_symmetry,
            default_physical_sectors,
        )

        if symmetry is None:
            symmetry = self.charge_symmetry or "U1"
        if (
            physical_sectors is None
            and self.charge_symmetry == symmetry
            and self.physical_sectors is None
        ):
            raise ValueError(
                "this active PEPO was validated in a dense basis without a "
                "native sector ordering; provide matching physical_sectors "
                "explicitly before Symmray conversion."
            )
        active = self
        provided_charges = virtual_charges
        if remove_orphans:
            if provided_charges is not None:
                active = type(self)(
                    lx=self.lx,
                    ly=self.ly,
                    cyclic=self.cyclic,
                    bond_dim=self.bond_dim,
                    physical_dim=self.physical_dim,
                    site_directions=self.site_directions,
                    blocks=self.blocks,
                    charge_symmetry=self.charge_symmetry,
                    physical_sectors=self.physical_sectors,
                    virtual_sector_charges=dict(provided_charges),
                )
            active = active.compact()
        if physical_sectors is None:
            physical_sectors = active.physical_sectors
        if physical_sectors is None:
            physical_sectors = default_physical_sectors(
                symmetry,
                active.physical_dim,
            )
        physical_sectors = dict(physical_sectors)
        if sum(int(size) for size in physical_sectors.values()) != active.physical_dim:
            raise ValueError(
                "physical_sectors must describe exactly the PEPO physical dimension."
            )
        if provided_charges is None:
            provided_charges = active.virtual_sector_charges
        if provided_charges is None:
            provided_charges = {
                sector: 0 for sector in range(active.bond_dim)
            }
        virtual_charges = dict(provided_charges)
        missing = set(range(active.bond_dim)) - set(virtual_charges)
        if missing:
            raise ValueError(
                "virtual_charges is missing active sector ids "
                f"{sorted(missing)}."
            )

        import symmray as sr  # pylint: disable=import-outside-toplevel
        array_cls = _array_class_for_symmetry(
            symmetry,
            fermionic=fermionic,
        )
        symmetry_obj = array_cls.get_class_symmetry(symmetry)
        physical_items = tuple(physical_sectors.items())
        physical_offsets = {}
        offset = 0
        for physical_charge, size in physical_items:
            size = int(size)
            physical_offsets[physical_charge] = (offset, offset + size)
            offset += size

        arrays = []
        native_arrays = {}
        for i in range(active.lx):
            row = []
            for j in range(active.ly):
                site = (i, j)
                directions = active.site_directions[site]
                virtual_duals = tuple(direction in ("d", "l") for direction in directions)
                charge_groups = {}
                for sector in range(active.bond_dim):
                    charge_groups.setdefault(virtual_charges[sector], []).append(sector)
                charge_sizes = {
                    axis_charge: len(sectors)
                    for axis_charge, sectors in charge_groups.items()
                }
                block_arrays = {}
                for key, block in active.blocks[site].items():
                    if not _backend_nonzero(block):
                        continue
                    virtual_charge_tuple = tuple(
                        virtual_charges[sector] for sector in key
                    )
                    virtual_offsets = tuple(
                        charge_groups[axis_charge].index(sector)
                        for axis_charge, sector in zip(
                            virtual_charge_tuple,
                            key,
                        )
                    )
                    for row_charge, (row_start, row_stop) in physical_offsets.items():
                        for column_charge, (column_start, column_stop) in physical_offsets.items():
                            source_block = block[row_start:row_stop, column_start:column_stop]
                            if not _backend_nonzero(source_block):
                                continue
                            # Quimb stores PEPO physical axes as (lower, upper),
                            # whereas active blocks use ordinary (row, column)
                            # matrix order.
                            physical_block = ar.do(
                                "transpose",
                                source_block,
                                (1, 0),
                            )
                            physical_row_charge = column_charge
                            physical_column_charge = row_charge
                            sector = (
                                *virtual_charge_tuple,
                                physical_row_charge,
                                physical_column_charge,
                            )
                            signed = tuple(
                                symmetry_obj.sign(
                                    sector_charge,
                                    dual,
                                )
                                for sector_charge, dual in zip(
                                    sector,
                                    virtual_duals + (False, True),
                                )
                            )
                            actual_charge = symmetry_obj.combine(*signed)
                            if actual_charge != charge:
                                raise ValueError(
                                    "Active PEPO block is not compatible with "
                                    f"{symmetry} charge {charge!r}: site={site}, "
                                    f"virtual={virtual_charge_tuple}, "
                                    f"physical=({physical_row_charge!r}, "
                                    f"{physical_column_charge!r}), "
                                    f"has charge {actual_charge!r}."
                                )
                            virtual_shape = tuple(
                                charge_sizes[axis_charge]
                                for axis_charge in virtual_charge_tuple
                            )
                            placed = ar.do(
                                "reshape",
                                physical_block,
                                (1,) * len(directions) + physical_block.shape,
                            )
                            for axis, (axis_size, axis_offset) in enumerate(
                                zip(virtual_shape, virtual_offsets)
                            ):
                                mask = np.zeros(axis_size, dtype=float)
                                mask[axis_offset] = 1.0
                                mask = _as_backend(mask, like=physical_block)
                                mask = ar.do(
                                    "reshape",
                                    mask,
                                    tuple(
                                        axis_size if index == axis else 1
                                        for index in range(len(directions) + 2)
                                    ),
                                )
                                placed = ar.do("multiply", placed, mask)
                            if sector in block_arrays:
                                block_arrays[sector] = ar.do(
                                    "add",
                                    block_arrays[sector],
                                    placed,
                                )
                            else:
                                block_arrays[sector] = placed

                duals = tuple(
                    sr.BlockIndex(
                        charge_sizes
                        if axis < len(directions)
                        else physical_sectors,
                        dual=dual,
                    )
                    for axis, dual in enumerate(virtual_duals + (False, True))
                )
                native = array_cls.from_blocks(
                    block_arrays,
                    duals=duals,
                    charge=charge,
                    symmetry=symmetry,
                )
                native_arrays[site] = native
                row.append(
                    np.zeros(
                        (active.bond_dim,) * len(directions)
                        + (active.physical_dim, active.physical_dim),
                        dtype=np.asarray(
                            ar.to_numpy(
                                next(iter(active.blocks[site].values()))
                            )
                        ).dtype,
                    )
                )
            arrays.append(row)

        pepo = qtn.PEPO(
            arrays,
            shape="urdlbk",
            cyclic=active.cyclic,
        )
        for site, native in native_arrays.items():
            pepo[site].modify(data=native)
        return pepo

    def to_pepo(self):
        """Materialize blocks as a dense Quimb ``PEPO``.

        This is an explicit interoperability boundary. The active-block
        representation is normally the smaller and clearer object to keep
        during autodiff or repeated coefficient evaluations.
        """
        arrays = []
        dtype = next(iter(next(iter(self.blocks.values())).values())).dtype
        for i in range(self.lx):
            row = []
            for j in range(self.ly):
                site = (i, j)
                row.append(
                    _materialize_site_blocks(
                        self.site_directions[site],
                        self.blocks[site],
                        self.bond_dim,
                        dtype,
                    )
                )
            arrays.append(row)
        return qtn.PEPO(arrays, shape="urdlbk", cyclic=self.cyclic)

    materialize = to_pepo


@dataclass
class GraphActivePEPOBlocks:
    """Sparse active blocks for an arbitrary finite graph PEPO.

    Each graph edge owns one shared virtual leg.  This is the general-geometry
    counterpart of :class:`ActivePEPOBlocks`; materialization returns a
    generic Quimb ``TensorNetwork`` because Quimb's ``PEPO`` wrapper is tied to
    four square-lattice legs.
    """

    sites: tuple[Hashable, ...]
    edges: tuple[tuple[Hashable, Hashable], ...]
    bond_dim: int
    physical_dim: int
    site_directions: dict
    blocks: dict
    charge_symmetry: str | None = None
    physical_sectors: dict | None = None
    virtual_sector_charges: dict | None = None

    @property
    def active_block_count(self):
        """Return the number of stored nonzero sector blocks."""
        return sum(len(site_blocks) for site_blocks in self.blocks.values())

    @property
    def active_nbytes(self):
        """Return bytes occupied by stored active blocks."""
        total = 0
        for site_blocks in self.blocks.values():
            for block in site_blocks.values():
                nbytes = getattr(block, "nbytes", None)
                total += int(nbytes if nbytes is not None else np.asarray(block).nbytes)
        return total

    @property
    def dense_nbytes(self):
        """Estimate bytes required by dense graph tensor-network tensors."""
        reference = next(iter(next(iter(self.blocks.values())).values()))
        itemsize = _backend_dtype_itemsize(reference)
        return sum(
            self.bond_dim ** len(self.site_directions[site])
            * self.physical_dim**2
            * itemsize
            for site in self.sites
        )

    def compact(self):
        """Remove zero and globally orphaned graph-edge sectors."""
        compact_blocks = {
            site: {
                key: block
                for key, block in site_blocks.items()
                if _backend_nonzero(block)
            }
            for site, site_blocks in self.blocks.items()
        }
        changed = True
        while changed:
            changed = False
            available = {
                (site, edge_index): {
                    key[self.site_directions[site].index(edge_index)]
                    for key in site_blocks
                    if edge_index in self.site_directions[site]
                }
                for site, site_blocks in compact_blocks.items()
                for edge_index in self.site_directions[site]
            }
            retained = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                kept = {}
                for key, block in site_blocks.items():
                    keep = True
                    for axis, edge_index in enumerate(directions):
                        sector = key[axis]
                        if sector == 0:
                            continue
                        source, target = self.edges[edge_index]
                        neighbor = target if site == source else source
                        if sector not in available[(neighbor, edge_index)]:
                            keep = False
                            break
                    if keep:
                        kept[key] = block
                    else:
                        changed = True
                retained[site] = kept
            compact_blocks = retained

        used = {0}
        for site_blocks in compact_blocks.values():
            for key in site_blocks:
                used.update(key)
        sector_map = {old: new for new, old in enumerate(sorted(used))}
        remapped = {
            site: {
                tuple(sector_map[sector] for sector in key): block
                for key, block in site_blocks.items()
            }
            for site, site_blocks in compact_blocks.items()
        }
        old_charges = self.virtual_sector_charges or {}
        charges = {
            sector_map[old]: old_charges.get(old, 0)
            for old in sorted(used)
        }
        return type(self)(
            sites=self.sites,
            edges=self.edges,
            bond_dim=len(used),
            physical_dim=self.physical_dim,
            site_directions=self.site_directions,
            blocks=remapped,
            charge_symmetry=self.charge_symmetry,
            physical_sectors=self.physical_sectors,
            virtual_sector_charges=charges,
        )

    remove_orphans = compact

    def to_tensor_network(self, *, remove_orphans=True):
        """Materialize the graph PEPO as a generic Quimb tensor network."""
        active = self.compact() if remove_orphans else self
        dtype = next(iter(next(iter(active.blocks.values())).values())).dtype
        edge_inds = {
            edge_index: ("graph-bond", edge_index)
            for edge_index in range(len(active.edges))
        }
        tensors = []
        for site in active.sites:
            directions = active.site_directions[site]
            data = _materialize_site_blocks(
                directions,
                active.blocks[site],
                active.bond_dim,
                dtype,
            )
            bra = ("graph-bra", site)
            ket = ("graph-ket", site)
            inds = tuple(edge_inds[edge_index] for edge_index in directions) + (
                bra,
                ket,
            )
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=inds,
                    tags={"GRAPH_PEPO", f"site={site!r}"},
                )
            )
        return qtn.TensorNetwork(tensors)

    def to_dense(self, *, remove_orphans=True):
        """Contract all graph bonds and return a dense operator matrix."""
        active = self.compact() if remove_orphans else self
        network = active.to_tensor_network(remove_orphans=False)
        output_inds = [("graph-bra", site) for site in active.sites]
        output_inds += [("graph-ket", site) for site in active.sites]
        tensor = network.contract(output_inds=output_inds)
        return np.asarray(tensor.data).reshape(
            active.physical_dim ** len(active.sites),
            active.physical_dim ** len(active.sites),
        )

    to_pepo = to_tensor_network
    materialize = to_tensor_network


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
    edge Pauli slots. The returned order-4 representation is normally kept
    as :class:`ActivePEPOBlocks`; dense Quimb materialization is intended for
    small validation lattices because a fixed Pauli channel basis is much
    larger than an SVD-compressed numerical basis.
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
    ):
        self.lx = _validate_shape(lx, "lx")
        self.ly = _validate_shape(ly, "ly")
        self.cyclic = _validate_cyclic(cyclic, self.lx, self.ly)
        if not isinstance(order, Integral):
            raise TypeError("order must be an integer.")
        self.order = int(order)
        if self.order < 1 or self.order > 4:
            raise ValueError("PauliPEPOBasis currently supports orders 1 through 4.")
        if symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")
        self.symmetry = symmetry
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
            "fused_pauli_slots": int(
                np.count_nonzero(self._onsite_term_map)
                + np.count_nonzero(self._edge_term_map)
            ),
            "cyclic": self.cyclic,
            "symmetry": self.symmetry,
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
        if self.order < 3:
            return self
        # Populate the process-wide physical basis cache before building the
        # per-basis cluster maps.
        _backend_pauli_basis(1)
        _backend_pauli_basis(2)
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
        onsite_components,
        edge_components,
        beta,
        one_exp,
        edge_residual,
    ):
        """Evaluate one connected residual using cached cluster embeddings."""
        reference = _backend_reference(
            (beta, one_exp, edge_residual, onsite_components, edge_components)
        )
        exact = _backend_expm(
            ar.do(
                "multiply",
                -beta,
                self._cluster_hamiltonian(
                    nsites,
                    edges,
                    onsite_components,
                    edge_components,
                    like=reference,
                ),
            )
        )
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
                    onsite_components,
                    edge_components,
                    beta,
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

    def _build_active(self, beta, values):
        onsite_components, edge_components = self._hamiltonian_components(
            values,
            beta,
        )
        onsite, edge = self._components_to_operators(
            onsite_components,
            edge_components,
        )
        reference = _backend_reference((beta, onsite, edge))
        beta = _as_backend(beta, like=reference)
        onsite = _as_backend(onsite, like=reference)
        edge = _as_backend(edge, like=reference)
        one_exp = _backend_expm(ar.do("multiply", -beta, onsite))
        edge_exact = _backend_expm(
            ar.do(
                "multiply",
                -beta,
                ar.do(
                    "add",
                    edge,
                    ar.do("add", _backend_embed_operator(onsite, (0,), 2, 2),
                          _backend_embed_operator(onsite, (1,), 2, 2)),
                ),
            )
        )
        edge_residual = ar.do(
            "subtract",
            edge_exact,
            _backend_operator_product([one_exp, one_exp]),
        )
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
                    onsite_components,
                    edge_components,
                    beta,
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
                    onsite_components,
                    edge_components,
                    beta,
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
                    onsite_components,
                    edge_components,
                    beta,
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
                onsite_components,
                edge_components,
                beta,
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


def _site_after(site, direction, lx, ly, cyclic):
    i, j = site
    di, dj = _DIRECTION_VECTORS[direction]
    ni, nj = i + di, j + dj
    if cyclic[0]:
        ni %= lx
    if cyclic[1]:
        nj %= ly
    if not (0 <= ni < lx and 0 <= nj < ly):
        return None
    return ni, nj


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

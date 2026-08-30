"""Geometry and spanning-tree plans for :class:`~.TreePeps`."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from numbers import Integral

from ...tensors.maps import OneDMap

__all__ = ["TreePepsPlan", "TreePepsGeometry"]

_MAX_TREE_PEPS_VIRTUAL_DEGREE = 3
_TOPOLOGY_ALIASES = {
    "tree": "tree",
    "branching": "tree",
    "non-mps": "tree",
    "non_mps": "tree",
    "path": "path",
    "chain": "path",
    "mps": "path",
}


def _normalize_topology(topology: str) -> str:
    """Normalize the explicit TreePeps virtual-topology contract."""

    normalized = str(topology).strip().lower().replace("_", "-")
    try:
        return _TOPOLOGY_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "topology must be 'tree' (branching) or 'path' (explicit MPS mode)"
        ) from exc


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a finite two- or three-dimensional shape."""

    if isinstance(shape, (str, bytes)):
        raise TypeError("shape must be a sequence of two or three integers")
    shape = tuple(shape)
    if len(shape) not in (2, 3):
        raise ValueError("TreePeps currently supports only 2D or 3D shapes")
    if any(not isinstance(size, Integral) or int(size) < 1 for size in shape):
        raise ValueError("each shape entry must be a positive integer")
    return tuple(int(size) for size in shape)


class TreePepsPlan:
    """A regular-lattice geometry together with a legal virtual-bond tree.

    Sites are addressed in two compatible ways: by a stable one-dimensional
    logical id ``q`` and by their lattice coordinate ``(x, y[, z])``.  The
    tree edges are stored using logical ids, while the coordinate maps remain
    available for PEPS-style tags and layout code. The default ``tree``
    topology is required to branch at least once; an MPS-like path is only
    accepted when ``topology="path"`` is explicit.
    """

    def __init__(
        self,
        shape: Sequence[int],
        *,
        one_d_to_coord: Sequence[tuple[int, ...]],
        tree_edges: Iterable[tuple[int, int]],
        lattice_edges: Iterable[tuple[int, int]],
        coord_to_one_d: dict[tuple[int, ...], int],
        root: int = 0,
        max_virtual_degree: int = 3,
        order: str = "snake",
        tree_order: str = "explicit",
        topology: str = "tree",
        boundary: str = "open",
    ) -> None:
        self.shape = _normalize_shape(shape)
        self.ndim = len(self.shape)
        self.size = _product(self.shape)
        self.order = str(order)
        self.tree_order = str(tree_order)
        self.topology = _normalize_topology(topology)
        self.boundary = str(boundary)
        if self.boundary != "open":
            raise NotImplementedError("TreePepsPlan currently supports open boundaries only")
        if not isinstance(max_virtual_degree, Integral) or max_virtual_degree < 1:
            raise ValueError("max_virtual_degree must be a positive integer")
        self.max_virtual_degree = int(max_virtual_degree)
        if self.max_virtual_degree > _MAX_TREE_PEPS_VIRTUAL_DEGREE:
            raise ValueError(
                "TreePeps virtual degree must be at most "
                f"{_MAX_TREE_PEPS_VIRTUAL_DEGREE}"
            )

        self.one_d_to_coord = tuple(tuple(coord) for coord in one_d_to_coord)
        if len(self.one_d_to_coord) != self.size:
            raise ValueError("one_d_to_coord does not match shape")
        if set(self.one_d_to_coord) != set(coord_to_one_d):
            raise ValueError("one_d_to_coord and coord_to_one_d disagree")
        self.coord_to_one_d = dict(coord_to_one_d)

        self.lattice_edges = _normalize_edges(lattice_edges, self.size)
        lattice_edge_set = set(self.lattice_edges)
        self.tree_edges = _normalize_edges(tree_edges, self.size)
        if len(self.tree_edges) != max(self.size - 1, 0):
            raise ValueError("a spanning tree on N sites must contain exactly N - 1 edges")
        if not set(self.tree_edges).issubset(lattice_edge_set):
            raise ValueError("tree_edges must be a subset of the lattice edges")

        self.root = self._resolve_site(root)
        adjacency = {q: set() for q in range(self.size)}
        for q0, q1 in self.tree_edges:
            if q0 == q1:
                raise ValueError("tree_edges cannot contain self-edges")
            adjacency[q0].add(q1)
            adjacency[q1].add(q0)
        if any(len(neighbors) > self.max_virtual_degree for neighbors in adjacency.values()):
            raise ValueError(f"tree degree exceeds max_virtual_degree={self.max_virtual_degree}")
        self._adjacency = {q: tuple(sorted(neighbors)) for q, neighbors in adjacency.items()}
        if self.topology == "tree" and self.max_degree < 3:
            raise ValueError(
                "topology='tree' requires at least one site with three virtual "
                "bonds (a rank-four tensor); use topology='path' explicitly "
                "for MPS-like or geometrically non-branching layouts"
            )
        if self.topology == "path" and self.max_degree > 2:
            raise ValueError("topology='path' permits at most two virtual bonds per site")
        self._parent, self._children = self._root_tree()

    @classmethod
    def from_shape(
        cls,
        shape: Sequence[int],
        *,
        order: str = "snake",
        tree_order: str | None = None,
        tree_edges: Iterable[tuple[int, int]] | None = None,
        root: int | tuple[int, ...] | str | None = None,
        max_virtual_degree: int = 3,
        topology: str = "tree",
        boundary: str = "open",
    ) -> "TreePepsPlan":
        """Build a plan from a 2D or 3D open regular lattice.

        ``order`` controls logical ids. ``tree_order`` independently selects
        the deterministic growth priority used to construct the retained
        virtual tree. ``topology='tree'`` is the default and requires a
        genuine branching site with three virtual bonds. ``topology='path'``
        is an explicit MPS-compatible mode for one-dimensional or otherwise
        non-branching geometries. Custom trees may be supplied with endpoints
        expressed as logical ids or coordinates.
        """

        shape = _normalize_shape(shape)
        topology = _normalize_topology(topology)
        if tree_order is None:
            tree_order = "snake"
        tree_mapper = OneDMap(*shape, mode=tree_order)
        tree_order = tree_mapper.mode
        mapper = OneDMap(*shape, mode=order)
        one_d_to_coord_map, coord_to_one_d = mapper.build()
        one_d_to_coord = tuple(tuple(one_d_to_coord_map[q]) for q in range(len(one_d_to_coord_map)))

        lattice_edges = []
        for coord in one_d_to_coord:
            for axis in range(len(shape)):
                neighbor = list(coord)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor in coord_to_one_d:
                    lattice_edges.append((coord_to_one_d[coord], coord_to_one_d[neighbor]))

        generated_tree = tree_edges is None
        if root is None:
            if tree_order == "inside-out":
                center = tuple((extent - 1) // 2 for extent in shape)
                root_q = coord_to_one_d[center]
            else:
                root_q = 0
        elif isinstance(root, str):
            if root.strip().lower().replace("_", "-") != "center":
                raise ValueError("root string must be 'center'")
            center = tuple((extent - 1) // 2 for extent in shape)
            root_q = coord_to_one_d[center]
        elif isinstance(root, Integral):
            root_q = int(root)
        else:
            root_q = _resolve_endpoint(root, coord_to_one_d, len(one_d_to_coord), len(shape))

        if tree_edges is None:
            tree_map, _ = tree_mapper.build()
            tree_order_q = tuple(
                coord_to_one_d[tuple(tree_map[q])] for q in range(len(tree_map))
            )
            snake_map, _ = OneDMap(*shape, mode="snake").build()
            fallback_order_q = tuple(
                coord_to_one_d[tuple(snake_map[q])] for q in range(len(snake_map))
            )
            ordered_edges = tuple(
                tuple(sorted((q0, q1)))
                for q0, q1 in zip(tree_order_q, tree_order_q[1:])
            )
            lattice_edge_set = {tuple(sorted(edge)) for edge in lattice_edges}
            if topology == "path":
                # A path is intentionally opt-in. Prefer the requested
                # traversal when it is lattice adjacent, otherwise use the
                # canonical snake path as a deterministic valid path.
                path_order = (
                    tree_order_q
                    if set(ordered_edges).issubset(lattice_edge_set)
                    else fallback_order_q
                )
                tree_edges = tuple(
                    tuple(sorted((q0, q1)))
                    for q0, q1 in zip(path_order, path_order[1:])
                )
            elif len(shape) == 2 and tree_order in {"row-major", "col-major"}:
                # In two dimensions these names describe the orientation of
                # the retained tree, not merely its growth priority. This is
                # the PEPS-like comb/rake shown in the layout documentation:
                # rows (columns) are long teeth joined by one column (row)
                # backbone. It remains a spanning tree with max degree 3.
                tree_edges = _comb_tree_edges(
                    shape,
                    coord_to_one_d,
                    orientation=("row" if tree_order == "row-major" else "column"),
                )
            else:
                tree_edges = _tree_edges_from_order(
                    tree_order_q,
                    lattice_edges,
                    root=root_q,
                    max_virtual_degree=max_virtual_degree,
                )
        else:
            tree_edges = tuple(
                (
                    _resolve_endpoint(edge[0], coord_to_one_d, len(one_d_to_coord), len(shape)),
                    _resolve_endpoint(edge[1], coord_to_one_d, len(one_d_to_coord), len(shape)),
                )
                for edge in tree_edges
            )

        return cls(
            shape,
            one_d_to_coord=one_d_to_coord,
            coord_to_one_d=coord_to_one_d,
            lattice_edges=lattice_edges,
            tree_edges=tree_edges,
            root=root_q,
            max_virtual_degree=max_virtual_degree,
            order=order,
            tree_order=(tree_order if generated_tree else "explicit"),
            topology=topology,
            boundary=boundary,
        )

    @property
    def coordinates(self) -> tuple[tuple[int, ...], ...]:
        """Coordinates indexed by logical site id."""

        return self.one_d_to_coord

    @property
    def adjacency(self) -> dict[int, tuple[int, ...]]:
        """The tree adjacency map, keyed by logical site id."""

        return dict(self._adjacency)

    @property
    def children(self) -> dict[int, tuple[int, ...]]:
        """Children in the tree rooted at :attr:`root`."""

        return dict(self._children)

    @property
    def parent(self) -> dict[int, int | None]:
        """Parent map in the tree rooted at :attr:`root`."""

        return dict(self._parent)

    @property
    def max_degree(self) -> int:
        """The largest number of virtual bonds incident on one site."""

        return max((len(neighbors) for neighbors in self._adjacency.values()), default=0)

    @property
    def max_tensor_rank(self) -> int:
        """The largest local tensor rank including its physical leg."""

        return 1 + self.max_degree

    @property
    def is_branching(self) -> bool:
        """Whether the retained virtual tree is genuinely non-MPS-like."""

        return self.max_degree >= 3

    @property
    def is_mps_topology(self) -> bool:
        """Whether the retained virtual graph is a path or a singleton."""

        return self.max_degree <= 2

    def tensor_rank(self, site: int | tuple[int, ...]) -> int:
        """Return ``1 +`` the virtual degree at ``site``."""

        return 1 + len(self.neighbors(site))

    def _resolve_site(self, site: int | tuple[int, ...]) -> int:
        return _resolve_endpoint(site, self.coord_to_one_d, self.size, self.ndim)

    def resolve_site(self, site: int | tuple[int, ...], *rest: int) -> int:
        """Resolve either ``q`` or a coordinate supplied as separate args."""

        if rest:
            site = (site, *rest)
        return self._resolve_site(site)

    def coordinate(self, site: int | tuple[int, ...]) -> tuple[int, ...]:
        """Return the coordinate for a logical id or coordinate selector."""

        return self.one_d_to_coord[self._resolve_site(site)]

    def logical_site(self, coordinate: tuple[int, ...]) -> int:
        """Return the logical id for a coordinate."""

        return self.coord_to_one_d[tuple(coordinate)]

    def neighbors(self, site: int | tuple[int, ...]) -> tuple[int, ...]:
        """Return the virtual-tree neighbors of ``site``."""

        return self._adjacency[self._resolve_site(site)]

    def path(
        self,
        site0: int | tuple[int, ...],
        site1: int | tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return the unique tree path between two sites, inclusive."""

        site0 = self._resolve_site(site0)
        site1 = self._resolve_site(site1)
        if site0 == site1:
            return (site0,)
        previous = {site0: None}
        queue = deque([site0])
        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency[current]:
                if neighbor in previous:
                    continue
                previous[neighbor] = current
                if neighbor == site1:
                    queue.clear()
                    break
                queue.append(neighbor)
        if site1 not in previous:
            raise ValueError("tree_edges do not form a connected spanning tree")
        result = []
        current = site1
        while current is not None:
            result.append(current)
            current = previous[current]
        return tuple(reversed(result))

    def is_connected(self, sites: Iterable[int | tuple[int, ...]]) -> bool:
        """Whether ``sites`` induce a connected subtree."""

        sites = {self._resolve_site(site) for site in sites}
        if not sites:
            return True
        seen = {next(iter(sites))}
        queue = deque(seen)
        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency[current]:
                if neighbor in sites and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen == sites

    def subtree_span(self, sites: Iterable[int | tuple[int, ...]]) -> frozenset[int]:
        """Return the minimal connected tree region spanning ``sites``."""

        sites = tuple(dict.fromkeys(self._resolve_site(site) for site in sites))
        if not sites:
            return frozenset()
        span = {sites[0]}
        for site in sites[1:]:
            span.update(self.path(sites[0], site))
        return frozenset(span)

    def _root_tree(self):
        parent: dict[int, int | None] = {self.root: None}
        queue = deque([self.root])
        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency[current]:
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if len(parent) != self.size:
            raise ValueError("tree_edges must form a connected spanning tree")
        children = {
            q: tuple(neighbor for neighbor in self._adjacency[q] if parent[q] != neighbor)
            for q in range(self.size)
        }
        return parent, children

    def __repr__(self) -> str:
        return (
            f"TreePepsPlan(shape={self.shape!r}, size={self.size}, "
            f"root={self.root}, max_degree={self.max_degree}, "
            f"tree_order={self.tree_order!r}, topology={self.topology!r})"
        )


TreePepsGeometry = TreePepsPlan


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _comb_tree_edges(
    shape: Sequence[int],
    coord_to_one_d: dict[tuple[int, ...], int],
    *,
    orientation: str,
) -> tuple[tuple[int, int], ...]:
    """Build an oriented 2D comb spanning tree.

    For ``orientation="row"``, each fixed-``y`` row is a horizontal tooth
    and the ``x=0`` column is the backbone. For ``orientation="column"``,
    each fixed-``x`` column is a vertical tooth and the ``y=0`` row is the
    backbone. The construction uses only nearest-neighbor lattice edges and
    has exactly one fewer edge than sites.
    """
    if len(shape) != 2:
        raise ValueError("oriented comb layouts are defined only for 2D shapes")
    if orientation not in {"row", "column"}:
        raise ValueError("comb orientation must be 'row' or 'column'")

    lx, ly = shape
    edges = []
    if orientation == "row":
        for y in range(ly):
            for x in range(lx - 1):
                edges.append(
                    tuple(
                        sorted(
                            (
                                coord_to_one_d[(x, y)],
                                coord_to_one_d[(x + 1, y)],
                            )
                        )
                    )
                )
        for y in range(ly - 1):
            edges.append(
                tuple(
                    sorted(
                        (
                            coord_to_one_d[(0, y)],
                            coord_to_one_d[(0, y + 1)],
                        )
                    )
                )
            )
    else:
        for x in range(lx):
            for y in range(ly - 1):
                edges.append(
                    tuple(
                        sorted(
                            (
                                coord_to_one_d[(x, y)],
                                coord_to_one_d[(x, y + 1)],
                            )
                        )
                    )
                )
        for x in range(lx - 1):
            edges.append(
                tuple(
                    sorted(
                        (
                            coord_to_one_d[(x, 0)],
                            coord_to_one_d[(x + 1, 0)],
                        )
                    )
                )
            )

    return tuple(sorted(set(edges)))


def _tree_edges_from_order(
    order: Sequence[int],
    lattice_edges: Iterable[tuple[int, int]],
    *,
    root: int,
    max_virtual_degree: int,
) -> tuple[tuple[int, int], ...]:
    """Grow a legal tree by attaching sites in a prescribed order.

    The order is deliberately not assumed to be a Hamiltonian path. This is
    what lets snake, row-major, Hilbert, and center-out traversals produce
    distinct branching trees while retaining the physical lattice as the only
    source of virtual bonds. Parent selection first grows from the currently
    most-connected available frontier site, then uses traversal priority for
    deterministic tie-breaking.
    """
    order = tuple(int(q) for q in order)
    if not order or len(set(order)) != len(order):
        raise ValueError("tree growth order must be a non-empty site permutation")
    if root not in order:
        raise ValueError("tree growth root must be present in the site order")
    if max_virtual_degree < 1:
        raise ValueError("max_virtual_degree must be positive")

    adjacency = {q: set() for q in order}
    for q0, q1 in lattice_edges:
        if q0 in adjacency and q1 in adjacency:
            adjacency[q0].add(q1)
            adjacency[q1].add(q0)

    rank = {q: index for index, q in enumerate(order)}
    visited = {root}
    degree = {q: 0 for q in order}
    edges = []
    while len(visited) < len(order):
        candidates = []
        for child in order:
            if child in visited or degree[child] >= max_virtual_degree:
                continue
            for parent in sorted(adjacency[child] & visited):
                if degree[parent] >= max_virtual_degree:
                    continue
                candidates.append(
                    (rank[child], -degree[parent], -rank[parent], child, parent)
                )
        if not candidates:
            break
        _, _, _, child, parent = min(candidates)
        edge = tuple(sorted((parent, child)))
        edges.append(edge)
        degree[parent] += 1
        degree[child] += 1
        visited.add(child)

    if len(visited) == len(order):
        return tuple(sorted(edges))

    raise ValueError(
        "could not grow a connected spanning tree within "
        f"max_virtual_degree={max_virtual_degree} from the requested order"
    )


def _resolve_endpoint(
    endpoint: int | Sequence[int],
    coord_to_one_d: dict[tuple[int, ...], int],
    size: int,
    ndim: int,
) -> int:
    if isinstance(endpoint, Integral):
        endpoint = int(endpoint)
        if not 0 <= endpoint < size:
            raise ValueError(f"logical site id {endpoint} is outside 0..{size - 1}")
        return endpoint
    if isinstance(endpoint, (str, bytes)):
        raise TypeError("site endpoints must be logical ids or coordinate tuples")
    coordinate = tuple(endpoint)
    if len(coordinate) != ndim or any(not isinstance(x, Integral) for x in coordinate):
        raise ValueError(f"coordinate endpoints must have {ndim} integer entries")
    coordinate = tuple(int(x) for x in coordinate)
    try:
        return coord_to_one_d[coordinate]
    except KeyError as exc:
        raise ValueError(f"coordinate {coordinate!r} is outside the lattice") from exc


def _normalize_edges(edges: Iterable[tuple[int, int]], size: int) -> tuple[tuple[int, int], ...]:
    normalized = []
    seen = set()
    for edge in edges:
        edge = tuple(edge)
        if len(edge) != 2:
            raise ValueError("each tree edge must contain exactly two endpoints")
        q0, q1 = (int(edge[0]), int(edge[1]))
        if not 0 <= q0 < size or not 0 <= q1 < size:
            raise ValueError("edge endpoint is outside the lattice")
        edge = tuple(sorted((q0, q1)))
        if edge in seen:
            raise ValueError(f"duplicate edge {edge!r}")
        seen.add(edge)
        normalized.append(edge)
    return tuple(sorted(normalized))

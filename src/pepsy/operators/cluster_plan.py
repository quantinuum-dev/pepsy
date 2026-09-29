"""Reusable, numerical-value-independent planning on finite interaction graphs."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import combinations, product
from math import prod
from numbers import Integral
from types import MappingProxyType

from ._cluster_collections import compile_collection_recursion
from ._cluster_symmetry import ClusterReusePlan, verify_spatial_symmetries


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@lru_cache(maxsize=128)
def _inventory(nsites, edges, cutoff, max_clusters):
    adjacency = [0] * nsites
    for a, b in edges:
        adjacency[a] |= 1 << b
        adjacency[b] |= 1 << a
    level = {1 << i for i in range(nsites)}
    masks = []
    for _ in range(min(cutoff, nsites)):
        if len(masks) + len(level) > max_clusters:
            raise ValueError(
                "cluster inventory exceeds max_clusters; increase the explicit budget."
            )
        masks.extend(level)
        following = set()
        if _ + 1 < min(cutoff, nsites):
            for mask in level:
                frontier = 0
                pending = mask
                while pending:
                    bit = pending & -pending
                    frontier |= adjacency[bit.bit_length() - 1]
                    pending ^= bit
                frontier &= ~mask
                while frontier:
                    bit = frontier & -frontier
                    following.add(mask | bit)
                    if len(masks) + len(following) > max_clusters:
                        raise ValueError(
                            "cluster inventory exceeds max_clusters; increase the explicit budget."
                        )
                    frontier ^= bit
        level = following
        if not level:
            break

    def sites(mask):
        indices = []
        while mask:
            bit = mask & -mask
            indices.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(indices)

    return tuple(
        sorted(
            (sites(mask) for mask in masks),
            key=lambda cluster: (len(cluster), cluster),
        )
    )


@lru_cache(maxsize=512)
def _local_blocks(nsites, edges):
    return _inventory(nsites, edges, nsites, 1 << nsites)


@lru_cache(maxsize=128)
def _collections(nsites, clusters, state_budget):
    return compile_collection_recursion(nsites, clusters, state_budget=state_budget)


@lru_cache(maxsize=512)
def _partition_count(nsites, edges, proper):
    blocks = _local_blocks(nsites, edges)
    return _collections(nsites, blocks, 1 << nsites)["collection_count"] + 1 - int(proper)


@lru_cache(maxsize=128)
def _partitions(nsites, edges, proper):
    blocks = tuple(
        (sum(1 << i for i in block), block)
        for block in _local_blocks(nsites, edges)
        if not proper or len(block) < nsites
    )

    def visit(mask):
        if not mask:
            yield ()
            return
        first = mask & -mask
        for bits, block in blocks:
            if bits & first and bits & mask == bits:
                for rest in visit(mask ^ bits):
                    yield (block, *rest)

    return tuple(visit((1 << nsites) - 1))


@dataclass(frozen=True)
class ClusterSymmetryPlan:
    """Verified local equivalences, including axes and finite multiplicities."""

    entries: Mapping
    search_fallbacks: int

    @property
    def multiplicities(self):
        return dict(Counter(source for source, _ in self.entries.values()))

    @property
    def info(self):
        representatives = len(self.multiplicities)
        return dict(
            targets=len(self.entries),
            representatives=representatives,
            reused=len(self.entries) - representatives,
            search_fallbacks=self.search_fallbacks,
        )


@dataclass(frozen=True, init=False)
class ClusterPlan:
    """A finite interaction graph and cached connected-cluster bookkeeping.

    ``cluster_size`` counts sites. Higher-body supports induce a clique in
    the connectivity graph; their occurrences remain separate in ``supports``.
    Coordinates/``shape`` and ``cyclic`` describe candidate spatial actions,
    never add interactions or infer missing periodic bonds. All counts are
    exact integers; exceeding a budget raises rather than truncating silently.

    Numerical arrays, bound parameters and autodiff graphs are never cached.
    MPO and Gaugy adapters use integer site indices in ``lattice.sites`` order.
    """

    lattice: object
    cluster_size: int
    supports: tuple
    shape: tuple | None
    cyclic: tuple
    max_clusters: int
    index_edges: tuple

    def __init__(
        self,
        lattice,
        *,
        cluster_size=2,
        supports=(),
        shape=None,
        cyclic=False,
        max_clusters=100_000,
    ):
        from .pepo_dense import ClusterLattice

        if not isinstance(lattice, ClusterLattice):
            raise TypeError("lattice must be a ClusterLattice.")
        cutoff = min(_positive(cluster_size, "cluster_size"), len(lattice.sites))
        budget = _positive(max_clusters, "max_clusters")
        indices = {site: i for i, site in enumerate(lattice.sites)}
        frozen_supports = []
        for support in supports:
            support = tuple(support)
            if not support or len(set(support)) != len(support):
                raise ValueError("each interaction support must contain distinct sites.")
            if any(site not in indices for site in support):
                raise ValueError("interaction support contains an unknown site.")
            frozen_supports.append(support)
        edges = tuple(sorted(tuple(sorted((indices[a], indices[b]))) for a, b in lattice.edges))
        if shape is None:
            sites = lattice.sites
            if sites == tuple(range(len(sites))):
                shape = (len(sites),)
            elif all(
                isinstance(s, tuple)
                and len(s) == 2
                and all(isinstance(x, Integral) and x >= 0 for x in s)
                for s in sites
            ):
                candidate = tuple(max(s[axis] for s in sites) + 1 for axis in range(2))
                if set(sites) == set(product(*(range(n) for n in candidate))):
                    shape = candidate
        if shape is not None:
            shape = tuple(_positive(n, "shape dimension") for n in shape)
            if len(shape) not in (1, 2) or prod(shape) != len(lattice.sites):
                raise ValueError("shape must describe all sites in one or two dimensions.")
        ndim = len(shape) if shape is not None else 1
        periodic = (cyclic,) * ndim if isinstance(cyclic, bool) else tuple(cyclic)
        if len(periodic) != ndim or any(not isinstance(b, bool) for b in periodic):
            raise ValueError("cyclic must be a bool or one bool per shape dimension.")
        if shape is None and any(periodic):
            raise ValueError("cyclic metadata requires coordinates or an explicit shape.")
        for name, value in dict(
            lattice=lattice,
            cluster_size=cutoff,
            supports=tuple(frozen_supports),
            shape=shape,
            cyclic=periodic,
            max_clusters=budget,
            index_edges=edges,
        ).items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_supports(cls, sites, supports, **kwargs):
        """Infer connectivity from every declared term, including zero bindings.

        Supply all sites explicitly to retain isolated sites. Repeated supports
        affect term multiplicity but produce only one undirected topology edge.
        """
        from .pepo_dense import ClusterLattice

        sites = tuple(range(sites)) if isinstance(sites, Integral) else tuple(sites)
        supports = tuple(tuple(s) for s in supports)
        indices = {site: i for i, site in enumerate(sites)}
        edges = {}
        for support in supports:
            for a, b in combinations(support, 2):
                if a not in indices or b not in indices:
                    raise ValueError("interaction support contains an unknown site.")
                edges.setdefault(frozenset((a, b)), (a, b))
        return cls(ClusterLattice(sites, tuple(edges.values())), supports=supports, **kwargs)

    @classmethod
    def from_terms(cls, terms, *, sites=None, **kwargs):
        """Plan located term objects (``.sites``) or mappings with ``sites``.

        Homogeneous direction templates must first be expanded by their basis.
        """
        supports = []
        for term in terms:
            support = (
                term.get("sites") if isinstance(term, Mapping) else getattr(term, "sites", None)
            )
            if support is None:
                raise TypeError("from_terms requires located terms with explicit sites.")
            supports.append(tuple(support))
        if sites is None:
            sites = tuple(dict.fromkeys(site for support in supports for site in support))
        return cls.from_supports(sites, supports, **kwargs)

    @property
    def sites(self):
        return self.lattice.sites

    @cached_property
    def index_clusters(self):
        """Lazily enumerate placements; named symmetry catalogs need not do so."""
        return _inventory(len(self.sites), self.index_edges, self.cluster_size, self.max_clusters)

    @cached_property
    def clusters(self):
        return tuple(tuple(self.sites[i] for i in cluster) for cluster in self.index_clusters)

    @property
    def counts(self):
        """Finite placements by site count, before any symmetry reduction."""
        return dict(sorted(Counter(map(len, self.index_clusters)).items()))

    @cached_property
    def graph_shapes(self):
        from .pepo_dense import GraphConnectedClusterShape

        shapes = []
        for sites in self.clusters:
            local = {site: i for i, site in enumerate(sites)}
            edges = tuple(
                (local[a], local[b], i)
                for i, (a, b) in enumerate(self.lattice.edges)
                if a in local and b in local
            )
            shapes.append(GraphConnectedClusterShape(sites, edges, len(edges) - len(sites) + 1))
        return tuple(shapes)

    def _local(self, cluster):
        cluster = tuple(cluster)
        if len(cluster) > self.cluster_size:
            raise ValueError("cluster exceeds the plan cluster_size.")
        if not cluster or len(set(cluster)) != len(cluster) or not set(cluster) <= set(self.sites):
            raise ValueError("cluster must contain distinct known sites.")
        local = {site: i for i, site in enumerate(cluster)}
        edges = tuple(
            sorted(
                tuple(sorted((local[a], local[b])))
                for a, b in self.lattice.edges
                if a in local and b in local
            )
        )
        blocks = _local_blocks(len(cluster), edges)
        if tuple(range(len(cluster))) not in blocks:
            raise ValueError("cluster must be connected.")
        return cluster, edges, blocks

    def subclusters(self, cluster, *, proper=True):
        """Connected subsets of a cluster, in the supplied site order."""
        cluster, _, blocks = self._local(cluster)
        return tuple(
            tuple(cluster[i] for i in block)
            for block in blocks
            if not proper or len(block) < len(cluster)
        )

    def partition_count(self, cluster, *, proper=True):
        """Count connected-block partitions; proper excludes the full block."""
        cluster, edges, _ = self._local(cluster)
        return _partition_count(len(cluster), edges, bool(proper))

    def partitions(self, cluster, *, proper=True, max_partitions=100_000):
        """Materialize exact connected-block partitions within a memory budget."""
        cluster, edges, _ = self._local(cluster)
        if self.partition_count(cluster, proper=proper) > _positive(
            max_partitions, "max_partitions"
        ):
            raise ValueError(
                "partition count exceeds max_partitions; use partition_count or raise the budget."
            )
        return tuple(
            tuple(tuple(cluster[i] for i in block) for block in partition)
            for partition in _partitions(len(cluster), edges, bool(proper))
        )

    def collection_count(self, *, include_background=True, state_budget=100_000):
        """Count disjoint retained polymers, with singleton background by default."""
        plan = _collections(
            len(self.sites), self.index_clusters, _positive(state_budget, "state_budget")
        )
        return plan["collection_count"] + int(include_background)

    def reuse(self, factors=None, *, clusters=None):
        """Find verified local representatives for ordered, labelled factors.

        Each factor contains ``(binding_key, ((site, operator_key), ...))``
        terms. Keys must be immutable and identify parameter *bindings*, not
        current values. ``None`` disables reuse. ``()`` explicitly requests
        geometry-only representatives, which are not numerical equivalences.
        Optional ``clusters`` restricts representatives to a supplied family of
        connected site-label tuples (for downstream window-based expansions).
        """
        plan = ClusterReusePlan()
        indexed = self._labels(factors, plan)
        if clusters is None:
            index_clusters = self.index_clusters
        else:
            positions = {site: i for i, site in enumerate(self.sites)}
            selected = (self._local(cluster)[0] for cluster in clusters)
            index_clusters = tuple(
                sorted(
                    {tuple(sorted(positions[site] for site in cluster)) for cluster in selected},
                    key=lambda c: (len(c), c),
                )
            )
        for cluster in index_clusters:
            positions = {site: i for i, site in enumerate(cluster)}
            edges = tuple(
                (positions[a], positions[b])
                for a, b in self.index_edges
                if a in positions and b in positions
            )
            local = (
                None
                if indexed is None
                else tuple(
                    tuple(
                        (
                            binding,
                            tuple((positions[site], operator) for site, operator in operators),
                        )
                        for binding, operators in factor
                        if all(site in positions for site, _ in operators)
                    )
                    for factor in indexed
                )
            )
            plan.add(cluster, len(cluster), edges, local)
        return ClusterSymmetryPlan(MappingProxyType(dict(plan.entries)), plan.search_fallbacks)

    def _labels(self, factors, plan):
        if factors is None:
            return None
        positions = {site: i for i, site in enumerate(self.sites)}
        return tuple(
            tuple(
                (
                    plan.label(binding),
                    tuple((positions[site], plan.label(operator)) for site, operator in operators),
                )
                for binding, operators in factor
            )
            for factor in factors
        )

    def spatial_symmetries(self, factors=None):
        """Return verified finite site permutations from coordinate candidates.

        Search 1D inversion, translations along periodic axes, and 2D integer
        linear actions with entries -1, 0, 1 (including square and axial
        triangular rotations/reflections). This is a candidate search, not a
        complete graph automorphism solver. Without labels, results certify
        only graph/support geometry. Use ``reuse`` for numerical equivalence.
        """
        nsites = len(self.sites)
        identity = tuple(range(nsites))
        candidates = {identity}
        if self.shape is not None:
            regular = tuple(product(*(range(n) for n in self.shape)))
            coordinates = self.sites if set(self.sites) == set(regular) else regular
            positions = {site: i for i, site in enumerate(coordinates)}
            ndim = len(self.shape)
            matrices = (
                ((-1,), (1,))
                if ndim == 1
                else (
                    m for m in product((-1, 0, 1), repeat=4) if abs(m[0] * m[3] - m[1] * m[2]) == 1
                )
            )
            for matrix in matrices:
                transformed = [
                    tuple(
                        sum(matrix[a * ndim + b] * site[b] for b in range(ndim))
                        for a in range(ndim)
                    )
                    for site in coordinates
                ]
                minima = tuple(min(site[a] for site in transformed) for a in range(ndim))
                transformed = [
                    tuple(
                        (site[a] % self.shape[a] if self.cyclic[a] else site[a] - minima[a])
                        for a in range(ndim)
                    )
                    for site in transformed
                ]
                if len(set(transformed)) == nsites and all(
                    site in positions for site in transformed
                ):
                    candidates.add(tuple(positions[site] for site in transformed))
            # Generators suffice to describe translations without O(N^2) storage.
            for axis, periodic in enumerate(self.cyclic):
                if periodic:
                    candidates.add(
                        tuple(
                            positions[
                                tuple(
                                    (x + 1) % self.shape[a] if a == axis else x
                                    for a, x in enumerate(site)
                                )
                            ]
                            for site in coordinates
                        )
                    )
        labels = self._labels(factors, ClusterReusePlan())
        site_indices = {site: i for i, site in enumerate(self.sites)}
        supports = Counter(
            tuple(sorted(site_indices[s] for s in support)) for support in self.supports
        )
        result = []
        for permutation in sorted(candidates):
            if (
                Counter(
                    tuple(sorted(permutation[s] for s in support))
                    for support in supports.elements()
                )
                != supports
            ):
                continue
            try:
                verify_spatial_symmetries(
                    (permutation,), nsites, self.index_edges, () if labels is None else labels
                )
            except ValueError:
                continue
            result.append(tuple(self.sites[i] for i in permutation))
        return tuple(result)

    @staticmethod
    def cache_info():
        """Bounded process-local structural caches (no numerical values)."""
        return {
            name: function.cache_info()._asdict()
            for name, function in (
                ("inventory", _inventory),
                ("local_blocks", _local_blocks),
                ("collections", _collections),
                ("partitions", _partitions),
            )
        }

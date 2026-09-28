"""Conservative local Hamiltonian equivalence under site permutations.

Only structural descriptions are retained in plans. Numerical targets belong
in an evaluation-local cache, so no backend values or autodiff graphs survive
an evaluation. Relabeling acts on physical sites, never on spin labels.
"""

from collections.abc import Mapping
from itertools import permutations, product
from math import factorial, prod

import autoray as ar
import numpy as np


def coefficient_key(value):
    """Identify immutable scalars/references; opaque values cannot be shared."""
    from .mpo_semantic import MPOParameter, _UNSET

    if type(value) in (bool, int, float, complex) or isinstance(value, np.number):
        array = np.asarray(value)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            return None
        return ("scalar", array.dtype.str, array.tobytes())
    if type(value) is MPOParameter:
        default = ("unset",) if value.default is _UNSET else coefficient_key(value.default)
        if default is not None:
            return ("parameter", value.name, default)
    return None


def permute_operator(matrix, axes, phys_dim):
    """Move source site axes into target order without leaving the backend."""
    nsites = len(axes)
    if axes == tuple(range(nsites)):
        return matrix
    tensor = ar.do("reshape", matrix, (phys_dim,) * (2 * nsites))
    tensor = ar.do("transpose", tensor, axes + tuple(nsites + i for i in axes))
    return ar.do("reshape", tensor, (phys_dim**nsites, phys_dim**nsites))


class ClusterReusePlan:
    """Intern exact labeled-graph matches with a bounded permutation search.

    A term is ``(coefficient_label, ((site, operator_label), ...))``. Factor
    order is significant, summand order is not. Edge multiplicity is retained.
    Labels are integer IDs supplied by :meth:`label`. Above the search budget,
    identity-only matching is safe and still permits translation reuse.
    """

    def __init__(self):
        self._labels = {}
        self._sources = {}
        self.entries = {}
        self.search_fallbacks = 0

    def label(self, key):
        return self._labels.setdefault(key, len(self._labels))

    def add(self, key, nsites, edges, factors):
        if key in self.entries:
            return self.entries[key]
        identity = tuple(range(nsites))
        if factors is None:
            entry = (key, identity)
        else:
            degrees = [0] * nsites
            for a, b in edges:
                degrees[a] += 1
                degrees[b] += 1
            groups = tuple(
                tuple(i for i, d in enumerate(degrees) if d == degree)
                for degree in sorted(set(degrees))
            )
            if prod(factorial(len(group)) for group in groups) > 4096:
                orders = (identity,)
                self.search_fallbacks += 1
            else:
                orders = (
                    sum(parts, ()) for parts in product(*(permutations(group) for group in groups))
                )
            best = None
            best_order = identity
            for order in orders:
                inverse = {site: i for i, site in enumerate(order)}
                graph = tuple(sorted(tuple(sorted((inverse[a], inverse[b]))) for a, b in edges))
                labeled = tuple(
                    tuple(
                        sorted(
                            (
                                coefficient,
                                tuple(sorted((inverse[site], label) for site, label in operators)),
                            )
                            for coefficient, operators in terms
                        )
                    )
                    for terms in factors
                )
                signature = (nsites, graph, labeled)
                if best is None or signature < best:
                    best, best_order = signature, order
            source, source_order = self._sources.setdefault(best, (key, best_order))
            inverse = {site: i for i, site in enumerate(best_order)}
            entry = (source, tuple(source_order[inverse[i]] for i in identity))
        self.entries[key] = entry
        return entry

    @property
    def info(self):
        count = len(self.entries)
        representatives = sum(key == source for key, (source, _) in self.entries.items())
        return {
            "targets": count,
            "representatives": representatives,
            "reused": count - representatives,
            "search_fallbacks": self.search_fallbacks,
        }


class SpatialProductData(tuple):
    """Ordered factor data with a numerical cache owned by one evaluation."""

    def __new__(cls, factors, plan):
        obj = super().__new__(cls, factors)
        obj.plan = plan
        obj.products = {}
        return obj


def normalize_spatial_symmetries(sites, symmetries):
    """Freeze supplied finite-lattice site permutations in positional order.

    Each declaration is a full site-to-site mapping or a sequence of target
    site labels in source-site order. Only lattice sites transform, not spins.
    """
    sites = tuple(sites)
    positions = {site: index for index, site in enumerate(sites)}
    normalized = []
    for symmetry in symmetries:
        if isinstance(symmetry, Mapping):
            if set(symmetry) != set(sites):
                raise ValueError("spatial_symmetries mappings must cover every lattice site.")
            targets = tuple(symmetry[site] for site in sites)
        else:
            targets = tuple(symmetry)
        if len(targets) != len(sites) or set(targets) != set(sites):
            raise ValueError("each spatial_symmetries entry must be a permutation of lattice sites.")
        normalized.append(tuple(positions[site] for site in targets))
    return tuple(dict.fromkeys(normalized))


def verify_spatial_symmetries(symmetries, nsites, edges, factors):
    """Verify declarations against graph multiplicity and ordered term labels.

    Reuse still proves each local equivalence. A declaration never authorizes
    merging independently parameterized terms or assuming spin rotations.
    """
    if not symmetries:
        return
    if factors is None:
        raise ValueError("cannot verify spatial_symmetries for opaque coefficient/operator bindings.")

    def signature(permutation):
        graph = tuple(sorted(tuple(sorted((permutation[a], permutation[b]))) for a, b in edges))
        terms = tuple(tuple(sorted((coefficient, tuple(sorted(
            (permutation[site], label) for site, label in operators
        ))) for coefficient, operators in factor)) for factor in factors)
        return graph, terms

    identity = signature(tuple(range(nsites)))
    for index, permutation in enumerate(symmetries):
        if signature(permutation) != identity:
            raise ValueError(
                f"spatial_symmetries[{index}] does not preserve lattice edges, "
                "Hamiltonian terms and coefficient bindings."
            )

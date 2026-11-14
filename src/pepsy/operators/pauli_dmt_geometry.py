"""Physical-region inventories for Pauli-MPO DMT."""

from itertools import combinations
from numbers import Integral


def square_lattice_dmt_regions(
    lx, ly, *, kind="plaquettes", mapper=None, map_mode="snake", site_order=None,
):
    """Return physical square-lattice regions as tuples of MPO site indices.

    ``kind`` is ``'plaquettes'``, ``'triples'`` (straight and L-shaped), or
    ``'l_shapes'``. An explicit mapper takes precedence over ``map_mode``.
    ``site_order`` optionally maps each MPO storage position to a logical
    site label, like ``MpsOptimizer.logical_order``. The mapper maps lattice
    coordinates to those logical labels. Without ``site_order``, mapper labels
    are MPO positions. Only open lattice edges are generated; custom or
    periodic regions can be supplied directly as MPO site indices.
    """
    from pepsy.tensors import OneDMap

    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in (lx, ly)):
        raise ValueError("lattice dimensions must be positive integers.")
    if kind not in {"plaquettes", "triples", "l_shapes"}:
        raise ValueError("kind must be 'plaquettes', 'triples', or 'l_shapes'.")
    if mapper is None:
        mapper = OneDMap(int(lx), int(ly), mode=map_mode)
    if not isinstance(mapper, OneDMap) or mapper.shape != (lx, ly):
        raise ValueError("mapper must be a OneDMap with the requested lattice shape.")
    _, indices = mapper.build()
    if site_order is not None:
        try:
            order = tuple(site_order)
        except TypeError as exc:
            raise ValueError("site_order must be a permutation of logical site indices.") from exc
        nsites = int(lx) * int(ly)
        if (len(order) != nsites
                or any(isinstance(i, bool) or not isinstance(i, Integral) for i in order)
                or set(order) != set(range(nsites))):
            raise ValueError("site_order must be a permutation of logical site indices.")
        positions = {int(logical): position for position, logical in enumerate(order)}
        indices = {coordinate: positions[logical] for coordinate, logical in indices.items()}
    regions = set()
    if kind == "plaquettes":
        for x in range(lx - 1):
            for y in range(ly - 1):
                regions.add(tuple(sorted(indices[p] for p in
                                         ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)))))
    else:
        for x in range(lx):
            for y in range(ly):
                neighbors = [(a, b) for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                             if 0 <= a < lx and 0 <= b < ly]
                for a, b in combinations(neighbors, 2):
                    if kind == "l_shapes" and (a[0] == b[0] or a[1] == b[1]):
                        continue
                    regions.add(tuple(sorted(indices[p] for p in ((x, y), a, b))))
    return tuple(sorted(regions))


def _normalize_regions(regions, nsites):
    if regions is None:
        return None
    try:
        regions = tuple(regions)
    except TypeError as exc:
        raise ValueError("regions must be a nonempty iterable of site sequences.") from exc
    if not regions:
        raise ValueError("regions must contain at least one region.")
    normalized = set()
    for region in regions:
        try:
            sites = tuple(region)
        except TypeError as exc:
            raise ValueError("each region must be a nonempty sequence of site indices.") from exc
        if not sites or any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < nsites
                            for s in sites) or len(set(sites)) != len(sites):
            raise ValueError("region sites must be unique integer MPO indices in range.")
        normalized.add(tuple(sorted(int(s) for s in sites)))
    return tuple(sorted(normalized))

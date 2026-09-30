"""Exact virtual-wire routing; physical cluster supports never change."""

from functools import lru_cache
from itertools import product
from math import prod

from .pepo_active import ActivePEPOBlocks, _DIRECTIONS, _OPPOSITE_DIRECTION, _site_after


@lru_cache(maxsize=128)
def square_routes(sites, edges, shape, cyclic):
    """Return deterministic shortest Manhattan wires in row-major coordinates."""
    if len(shape) != 2 or prod(shape) != len(sites) or any(n < 1 for n in shape):
        raise ValueError("square routing requires shape=(Lx, Ly) covering every site.")
    if any(periodic and n == 1 for n, periodic in zip(shape, cyclic)):
        raise ValueError("a periodic square axis must contain at least two sites.")
    coordinates = tuple(product(*(range(n) for n in shape)))
    positions = dict(zip(sites, coordinates))
    directions = {site: tuple(d for d in _DIRECTIONS
                             if _site_after(site, d, *shape, cyclic) is not None)
                  for site in coordinates}
    routes = []
    for a, b in edges:
        current, target = positions[a], positions[b]
        path = []
        for axis, (positive, negative) in enumerate((("u", "d"), ("r", "l"))):
            delta = target[axis] - current[axis]
            if cyclic[axis]:
                forward = delta % shape[axis]
                backward = (-delta) % shape[axis]
                delta = forward if forward <= backward else -backward
            direction = positive if delta >= 0 else negative
            for _ in range(abs(delta)):
                neighbor = _site_after(current, direction, *shape, cyclic)
                path.append(((current, direction), (neighbor, _OPPOSITE_DIRECTION[direction])))
                current = neighbor
        routes.append(tuple(path))
    return coordinates, directions, tuple(routes)


def route_graph_blocks(active, shape, cyclic=(False, False)):
    """Route independent graph legs, including simultaneous crossing wires.

    A wire passing through a physical site is an identity on its virtual
    index, tensor-multiplied with that site's original physical block. It
    must not exclude a different cluster occupying the same routing site.
    """
    if active.charge_symmetry is not None:
        raise NotImplementedError("square routing currently supports dense bosonic blocks.")
    shape, cyclic = tuple(shape), tuple(cyclic)
    coordinates, directions, routes = square_routes(active.sites, active.edges, shape, cyclic)
    sectors = []
    for index, (a, b) in enumerate(active.edges):
        values = {0}
        for site in (a, b):
            axis = active.site_directions[site].index(index)
            values.update(key[axis] for key in active.blocks[site])
        sectors.append(tuple(sorted(values)))
    wires = {(site, d): [] for site in coordinates for d in directions[site]}
    through = {site: set() for site in coordinates}
    for index, path in enumerate(routes):
        for leg, opposite in path:
            wires[leg].append(index)
            wires[opposite].append(index)
        for _, (site, _) in path[:-1]:
            through[site].add(index)
    # Each square leg encodes the tensor product of all graph wires using it.
    # Mixed-radix labels preserve the earlier lexicographic numbering without
    # allocating a dictionary for every Cartesian wire combination.
    positions = [{value: i for i, value in enumerate(values)} for values in sectors]
    sizes = {leg: prod(len(sectors[e]) for e in ids) for leg, ids in wires.items()}

    def encode(ids, assignment):
        label = 0
        for edge in ids:
            label = label*len(sectors[edge])+positions[edge][assignment[edge]]
        return label
    blocks = {}
    for original, site in zip(active.sites, coordinates):
        site_blocks = {}
        crossing = tuple(sorted(through[site]))
        for key, block in active.blocks[original].items():
            incident = dict(zip(active.site_directions[original], key))
            for values in product(*(sectors[e] for e in crossing)):
                assignment = {**incident, **dict(zip(crossing, values))}
                routed = tuple(encode(wires[site, d], assignment)
                               for d in directions[site])
                # Every original/through-wire value appears on at least one
                # leg, so this encoding is injective. Reuse the physical
                # block instead of copying it for each identity wire state.
                site_blocks[routed] = block
        blocks[site] = site_blocks
    return ActivePEPOBlocks(
        lx=shape[0], ly=shape[1], cyclic=cyclic,
        bond_dim=max(sizes.values(), default=1),
        physical_dim=active.physical_dim, site_directions=directions, blocks=blocks,
    )

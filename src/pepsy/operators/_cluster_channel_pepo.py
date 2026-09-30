"""Parameter-independent PEPO edge quotients and reusable reference templates.

A slice includes every other virtual leg and both physical indices. Selecting
independent slices and transferring constant linear combinations to the other
endpoint is valid on arbitrary networks, including loops. The generic mode
shares identical slices; the Pauli mode proves rational linear dependencies.
"""

from dataclasses import replace
from math import prod

import autoray as ar
import numpy as np

from ._cluster_channel_structure import _Linear, _map_matrix
from .mpo_automaton import _as_backend


def _nonzero(block):
    return any(_Linear.coerce(value).terms for value in block.flat)


def _groups(blocks, axis, size):
    signatures = [[] for _ in range(size)]
    for key, block in sorted(blocks.items()):
        entries = tuple(_Linear.coerce(value).terms for value in block.flat)
        if any(entries):
            signatures[key[axis]].append((key[:axis]+key[axis+1:], entries))
    groups, lookup = [[0]], {}
    for channel in range(1, size):
        signature = tuple(signatures[channel])
        if not signature:
            continue
        if signature not in lookup:
            lookup[signature] = len(groups)
            groups.append([])
        groups[lookup[signature]].append(channel)
    return tuple(tuple(group) for group in groups)


def _apply_groups(blocks, axis, groups, select):
    lookup = {old: new for new, group in enumerate(groups)
              for old in (group[:1] if select else group)}
    output = {}
    for key, block in blocks.items():
        if key[axis] not in lookup:
            continue
        key = key[:axis]+(lookup[key[axis]],)+key[axis+1:]
        output[key] = output.get(key, 0)+block
    return {key: block for key, block in output.items() if _nonzero(block)}


def _share(layout, blocks, limit, enabled, *, linear=False):
    """Exact edge-local sum/select quotient, with the singleton rail protected."""
    from .cluster_channels import _budget

    sizes = list(map(len, layout.sectors))
    for n in sizes:
        _budget((n, n), limit)
    first, second = [np.eye(n) for n in sizes], [np.eye(n) for n in sizes]
    # Normalize sparse global sector labels to edge-local channel positions.
    labels = [[None]*len(keys[0]) for keys in layout.keys]
    for sectors, ends in zip(layout.sectors, layout.ends):
        for site, axis in ends:
            labels[site][axis] = {s: i for i, s in enumerate(sectors)}
    blocks = [{tuple(labels[site][axis][s] for axis, s in enumerate(key)): value
               for key, value in entries.items()} for site, entries in enumerate(blocks)]
    if linear:
        from fractions import Fraction

        blocks = [{key: np.array([_Linear(tuple((atom, Fraction(c)) for atom, c in
                    _Linear.coerce(value).terms)) for value in block.flat], dtype=object).reshape(block.shape)
                   for key, block in entries.items()} for entries in blocks]
    rounds = 0
    if enabled:
        while True:
            previous = sum(sizes)
            rounds += 1
            for side in (1, 0):
                for edge in (range(len(sizes)-1, -1, -1) if side else range(len(sizes))):
                    site, axis = layout.ends[edge][side]
                    if linear:
                        from ._cluster_channel_linear import delinearize_slices, apply_transfer, transfer_matrix

                        result = delinearize_slices(blocks[site], axis, sizes[edge], limit)
                        if result is None:
                            continue
                        select, transfer, width = result
                        for endpoint, (s, a) in enumerate(layout.ends[edge]):
                            mapping = select if endpoint == side else transfer
                            blocks[s] = apply_transfer(blocks[s], a, mapping)
                            maps = first if endpoint == 0 else second
                            maps[edge] = maps[edge] @ transfer_matrix(mapping, width, limit)
                        sizes[edge] = width
                        continue
                    groups = _groups(blocks[site], axis, sizes[edge])
                    if len(groups) == sizes[edge]:
                        continue
                    for endpoint, (s, a) in enumerate(layout.ends[edge]):
                        select = endpoint == side
                        blocks[s] = _apply_groups(blocks[s], a, groups, select)
                        maps = first if endpoint == 0 else second
                        maps[edge] = maps[edge] @ _map_matrix(groups, sizes[edge], select, limit)
                    sizes[edge] = len(groups)
            if sum(sizes) == previous:
                break
    keys = tuple(tuple(sorted(site)) for site in blocks)
    reduced = replace(layout, keys=keys, sectors=tuple(tuple(range(n)) for n in sizes),
                      charges=tuple((0,)*n for n in sizes))
    return reduced, blocks, tuple(first), tuple(second), dict(
        bond_dimensions=tuple(sizes), removed_channels=sum(map(len, layout.sectors))-sum(sizes), sweeps=rounds,
        reduction='rational-delinearisation' if linear else 'identical-slices')


class _Expressions:
    """Static linear gather; no reference values or autodiff graphs are retained."""

    def __init__(self, layout, blocks, limit):
        from .cluster_channels import _budget, _readonly

        self.programs = []
        for keys, entries in zip(layout.keys, blocks):
            terms = [tuple(_Linear.coerce(v).terms) for key in keys for v in entries[key].flat]
            width = max(map(len, terms), default=1)
            _budget((2, max(1, width), len(terms)), limit)
            indices = np.zeros((max(1, width), len(terms)), dtype=np.int64)
            weights = np.zeros(indices.shape)
            for column, expression in enumerate(terms):
                for row, (atom, coefficient) in enumerate(expression):
                    indices[row, column] = atom
                    weights[row, column] = coefficient
            self.programs.append((_readonly(indices), _readonly(weights), (len(keys), layout.dimension, layout.dimension)))
        self.programs = tuple(self.programs)

    def __call__(self, residuals):
        like = residuals[0]
        flat = ar.do('concatenate', (ar.do('ones', (1,), like=like, dtype=like.dtype),
                                    *(v.reshape(-1) for v in residuals)))
        values = []
        for indices, weights, shape in self.programs:
            ids = _as_backend(np.array(indices, copy=True), like=flat)
            scale = ar.do('astype', _as_backend(np.array(weights, copy=True), like=flat, dtype=flat.dtype), flat.dtype)
            values.append(ar.do('sum', flat[ids]*scale, axis=0).reshape(shape))
        return tuple(values)


class SymbolicPEPO:
    """Reduce graph edges before routing, then share square-edge slices.

    Preparation enumerates local fixed factors and the remaining routed
    blocks once. It never calls the numerical expanded PEPO builder. The
    tensor-network replay contracts the original local cores with the frozen
    graph and square maps, without evaluating these sparse templates.
    """

    def __init__(self, wrapper, limit, structural_reuse, *, algebraic=False):
        from ._cluster_channel_assembly import _fixed_tree
        from .cluster_channels import _budget, _capture
        from .pepo_active import GraphActivePEPOBlocks
        from .pepo_dense import _SectorAllocator
        from .pepo_routing import route_graph_blocks, square_routes

        source = getattr(wrapper, '_graph', wrapper)
        plan, d = source.cluster_plan, source._local.phys_dim
        self.algebra = None
        if algebraic:
            from ._cluster_channel_pauli import PauliAlgebra

            self.algebra = PauliAlgebra(source, limit)
        self.keys = tuple(plan.index_clusters)
        entries = sum(d**(2*len(key)) for key in self.keys)
        _budget((entries,), limit, itemsize=128)
        residuals, atom = {}, 1
        for index, key in enumerate(self.keys):
            n = d**len(key)
            if self.algebra is None:
                residuals[key] = np.array([_Linear(((i, 1),)) for i in range(atom, atom+n*n)],
                                         dtype=object).reshape(n, n)
                atom += n*n
            else:
                space = self.algebra.spaces[index]
                residuals[key] = space.symbolic(atom, limit)
                atom += space.size
        directions = {s: tuple(e for e, pair in enumerate(plan.lattice.edges) if s in pair) for s in plan.sites}
        blocks = {s: {(0,)*len(directions[s]): residuals[(i,)]} for i, s in enumerate(plan.sites)}
        allocator = _SectorAllocator()
        graph_sectors = [{0} for _ in plan.lattice.edges]
        candidate_blocks, discarded = len(blocks), 0
        candidate_counts = dict.fromkeys(plan.sites, 1)
        for key in self.keys:
            if len(key) == 1:
                continue
            arrays, (tensors, parent, parent_edge, ranks, shape) = _fixed_tree(source, residuals[key], key)
            sectors = {parent_edge[s]: tuple(allocator.allocate(rank)) for s, rank in ranks.items()}
            for edge, values in sectors.items():
                graph_sectors[edge].update(values)
            for slot, (children, _) in tensors.items():
                site = shape.sites[slot]
                edges = tuple(parent_edge[c] for c in children)
                if parent[slot] is not None:
                    edges += (parent_edge[slot],)
                core = arrays[slot]
                count = prod(core.shape[:-2])
                candidate_blocks += count
                candidate_counts[site] += count
                _budget((candidate_blocks, d*d), limit, itemsize=128)
                for indices in np.ndindex(core.shape[:-2]):
                    block = core[indices]
                    if not _nonzero(block):
                        discarded += 1
                        continue
                    channel = [0]*len(directions[site])
                    for edge, index in zip(edges, indices):
                        channel[directions[site].index(edge)] = sectors[edge][index]
                    channel = tuple(channel)
                    blocks[site][channel] = blocks[site].get(channel, 0)+block
        active = GraphActivePEPOBlocks(sites=plan.sites, edges=plan.lattice.edges,
            bond_dim=allocator.next_sector, physical_dim=d, site_directions=directions, blocks=blocks)
        raw_graph, _ = _capture(active)
        raw_graph = replace(raw_graph, sectors=tuple(tuple(sorted(s)) for s in graph_sectors),
                            charges=tuple((0,)*len(s) for s in graph_sectors))
        graph, blocks, first, second, graph_report = _share(
            raw_graph, [blocks[s] for s in plan.sites], limit, structural_reuse, linear=algebraic)
        self.graph_maps = tuple(zip(raw_graph.sectors, first, second))
        self.graph_dimensions = tuple(map(len, graph.sectors))
        active = replace(active, blocks=dict(zip(plan.sites, blocks)), bond_dim=max(self.graph_dimensions, default=1))
        routed_blocks = sum(map(len, blocks))
        unreduced_blocks = candidate_blocks
        unreduced_dimensions = tuple(map(len, raw_graph.sectors))
        if hasattr(wrapper, '_graph'):
            coordinates, square_directions, routes = square_routes(plan.sites, plan.lattice.edges,
                                                    wrapper.cluster_plan.shape, wrapper.cluster_plan.cyclic)
            through = {s: set() for s in coordinates}
            wires = {(s, direction): [] for s in coordinates for direction in square_directions[s]}
            for edge, path in enumerate(routes):
                for a, b in path:
                    wires[a].append(edge)
                    wires[b].append(edge)
                for _, (site, _) in path[:-1]:
                    through[site].add(edge)
            routed_blocks = sum(len(blocks[i])*prod(self.graph_dimensions[e] for e in through[s])
                                for i, s in enumerate(coordinates))
            _budget((routed_blocks, d*d), limit, itemsize=128)
            unreduced_blocks = sum(candidate_counts[original]*prod(len(raw_graph.sectors[e]) for e in through[s])
                                   for original, s in zip(plan.sites, coordinates))
            active = route_graph_blocks(active, wrapper.cluster_plan.shape, wrapper.cluster_plan.cyclic)
        self.layout, _ = _capture(active)
        if self.layout.kind == 'square':
            unreduced_dimensions = tuple(prod(len(raw_graph.sectors[e]) for e in
                wires[self.layout.sites[s], square_directions[self.layout.sites[s]][axis]])
                for (s, axis), _ in self.layout.ends)
        blocks = [active.blocks[s] for s in self.layout.sites]
        self.raw_program = _Expressions(self.layout, blocks, limit)
        self.reduced_layout, reduced, self.first, self.second, square_report = _share(
            self.layout, blocks, limit, structural_reuse and self.layout.kind == 'square', linear=algebraic)
        self.reduced_program = _Expressions(self.reduced_layout, reduced, limit)
        self.container = replace(active, blocks={})
        self.structural_report = dict(method='exact-algebraic-pepo' if algebraic else 'exact-symbolic-pepo-slices',
            parameter_identity='independent-pauli-coordinates' if algebraic else 'independent-residual-entries', graph=graph_report,
            routed=square_report, removed_channels=graph_report['removed_channels']+square_report['removed_channels'])
        self.report = dict(preparation='algebraic' if algebraic else 'symbolic', expanded_reference_builds=0,
            symbolic_graph_candidate_blocks=candidate_blocks, symbolic_zero_blocks_removed=discarded,
            graph_bond_dimensions_before=tuple(map(len, raw_graph.sectors)),
            graph_bond_dimensions=self.graph_dimensions, symbolic_routed_blocks=routed_blocks,
            unreduced_bond_dimensions=unreduced_dimensions, unreduced_sparse_blocks=unreduced_blocks)
        if self.algebra is not None:
            self.report.update(self.algebra.report)

    def values(self, residuals, *, reduced=False):
        if tuple(residuals) != self.keys:
            raise ValueError('cluster residual structure changed; prepare a new channel plan.')
        values = tuple(residuals.values()) if self.algebra is None else self.algebra.coordinates(residuals)
        return (self.reduced_program if reduced else self.raw_program)(values)

    def evaluate(self, residuals):
        values = self.values(residuals)
        return replace(self.container, blocks={s: dict(zip(keys, rows))
            for s, keys, rows in zip(self.layout.sites, self.layout.keys, values)})

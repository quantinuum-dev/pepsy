"""Static contractions from local residual factors into compact site tensors.

Only preparation visits the expanded channel layout. Replay contracts one
local cluster factor at a time; crossing MPO paths and routed PEPO wires are
indices of the contraction, never expanded product tensors.
"""

from dataclasses import dataclass
from math import prod
from string import ascii_letters

import autoray as ar
import numpy as np

from pepsy.backends.convert import _array_namespace
from .mpo_automaton import _as_backend


class ChannelSource:
    """Resolve the existing public bindings to one fixed construction owner."""

    def __init__(self, source, *, preparation='reference', memory_budget=256*1024**2,
                 frontier_state_budget=None):
        from .mpo_product import MPOClusterProductExpansion
        from .pepo_product import PEPOClusterProductExpansion
        from .pepo_basis import PauliPEPOBasis
        from .square_pepo_product import SquarePEPOClusterProductExpansion

        self.owner = source
        if isinstance(source, PauliPEPOBasis):
            source = PEPOClusterProductExpansion.from_bases([source])
        self.reference = source
        self.mpo = isinstance(source, MPOClusterProductExpansion)
        self.product = isinstance(source, PEPOClusterProductExpansion)
        self.square = isinstance(source, SquarePEPOClusterProductExpansion)
        self.frontier = None
        self.pepo = None
        self.algebra = None
        if self.mpo:
            if preparation in ('symbolic', 'algebraic'):
                raise ValueError("symbolic/algebraic preparation currently supports graph/square PEPO builders only.")
            if source.factorization != 'fixed' or source.assembly not in ('direct', 'recursive'):
                raise ValueError('compact channel plans require fixed direct or recursive MPO construction.')
            if preparation in ('frontier', 'automaton'):
                from ._cluster_channel_frontier import FrontierMPO

                self.frontier = FrontierMPO(source, memory_budget, frontier_state_budget)
                if preparation == 'automaton':
                    from ._cluster_channel_pauli import PauliAlgebra

                    if source.physical_space.symmetry is not None:
                        raise ValueError('automaton preparation currently requires dense qubit Pauli terms; '
                                         'use frontier preparation for native charge sectors.')
                    self.algebra = PauliAlgebra(source, memory_budget,
                                               clusters=tuple(self.frontier.clusters.values()))
                    self.frontier.report.update(preparation='automaton', **self.algebra.report)
            elif source.assembly == 'recursive':
                from copy import copy

                # Prepare an equivalent exact channel topology, subject to the
                # source's collection budget. Do not change the caller's
                # recursive evaluator or fall back to a one-cluster target.
                self.reference = copy(source)
                self.reference.assembly = 'direct'
                self.reference.graph_assembly = 'exact'
                self.reference.max_collection_order = None
                self.reference._graph_collection_plan_cache = {}
                self.reference._coefficient_expansion = None
                self.reference.assembly_state_budget = 4096
                self.reference.assembly_batch_size = 'auto'
                self.reference.assembly_form = 'left'
        elif preparation in ('frontier', 'automaton'):
            raise ValueError('frontier/automaton preparation currently supports MPO builders only.')
        elif source.cache_info.get('factorization') != 'fixed':
            raise ValueError("compact channel plans require factorization='fixed'.")

    def resolve(self, arguments):
        args = dict(arguments)
        source = self.reference
        if self.product:
            source, bindings = source._projection_binding(args.get('parameters'), args.get('coefficients'),
                                                         allow_singletons=True)
            args.update(parameters=bindings, coefficients=None)
        elif self.square:
            source = source._projection_builder()
        return source, args

    def evaluate(self, arguments):
        if self.pepo is not None:
            return self.pepo.evaluate(self.residuals(arguments))
        if self.frontier is not None:
            residuals = self.residuals(arguments)
            if self.algebra is not None:
                residuals = {key: space.bind(value)[1](value)
                             for space, (key, value) in zip(self.algebra.spaces, residuals.items())}
            return self.frontier.evaluate(residuals)
        source, args = self.resolve(arguments)
        return source.exp(**args, **({'_sparse': True} if self.mpo else {'materialize': False}))

    def residuals(self, arguments):
        from .mpo_product import _cluster_to_backend

        source, args = self.resolve(arguments)
        if self.pepo is not None and self.pepo.algebra is not None:
            self.pepo.algebra.validate(getattr(source, '_graph', source))
        if self.mpo:
            if self.algebra is not None:
                self.algebra.validate(source)
            if args.get('coefficients') is not None:
                source, parameters = source._bind_coefficients(args.get('parameters'), args['coefficients'])
                args.update(parameters=parameters, coefficients=None)
            if source.physical_space.symmetry is not None and not source._native_static_validated:
                from ._cluster_native import validate_static_generators

                validate_static_generators(source)
                source._native_static_validated = True
            args['step'] = _cluster_to_backend(args['step'], source.to_backend)
        return source.residuals(**args)


def _fixed_mpo(source, residual, key):
    from .mpo_product import _operator_schmidt

    n = len(key) if source.graph is not None else key[1]-key[0]+1
    return tuple(_operator_schmidt(residual, n, source.phys_dim, 0.,
                                  factorization='fixed', physical_space=source.physical_space))


def _fixed_tree(source, residual, key):
    from .pepo_dense import _tree_factorize_operator_backend

    plan = source.cluster_plan
    shape = plan.graph_shapes[plan.index_clusters.index(key)]
    tensors, parent, parent_edge, _, ranks = _tree_factorize_operator_backend(
        residual, shape.edges, len(key), source._local.phys_dim, graph=True, factorization='fixed')
    arrays = []
    for site in range(len(key)):
        children, tensor = tensors[site]
        physical_axis = len(children)
        virtual_axes = tuple(i for i in range(tensor.ndim) if i != physical_axis)
        arrays.append(ar.do('reshape', ar.do('transpose', tensor, virtual_axes+(physical_axis,)),
                            tuple(tensor.shape[i] for i in virtual_axes)+(source._local.phys_dim,)*2))
    return tuple(arrays), (tensors, parent, parent_edge, ranks, shape)



def _bind_tree(source, key, like):
    """Freeze tree permutations and identity splits before backend compilation."""
    from .pepo_dense import _backend_tree_topology

    n, d = len(key), source._local.phys_dim
    shape = source.cluster_plan.graph_shapes[source.cluster_plan.index_clusters.index(key)]
    _, _, child_items, traversal = _backend_tree_topology(n, tuple(shape.edges), True)
    children = dict(child_items)
    axes, dimensions = [('physical', site) for site in range(n)], [d*d]*n
    steps, order = [], []
    for site in reversed(traversal[1:]):
        child_nodes = children[site]
        row_axes = [('bond', child) for child in child_nodes]+[('physical', site)]
        column_axes = [axis for axis in axes if axis not in row_axes]
        permutation = tuple(axes.index(axis) for axis in row_axes+column_axes)
        row_shape = tuple(dimensions[axes.index(axis)] for axis in row_axes)
        column_shape = tuple(dimensions[axes.index(axis)] for axis in column_axes)
        rows, columns = prod(row_shape), prod(column_shape)
        rank = min(rows, columns)
        eye = ar.do('astype', _as_backend(np.eye(rank), like=like, dtype=like.dtype), like.dtype)
        # Tree factors have (children, physical, parent); emitted site cores
        # have (children, parent, output, input).
        move_physical = (*range(len(child_nodes)), len(child_nodes)+1, len(child_nodes))
        core_shape = (*row_shape[:-1], rank, d, d)
        steps.append((permutation, (rows, columns), rows <= columns, eye,
                      (*row_shape, rank), move_physical, core_shape, (rank, *column_shape)))
        order.append(site)
        axes, dimensions = [('bond', site)]+column_axes, [rank, *column_shape]
    root = traversal[0]
    root_axes = [('bond', child) for child in children[root]]+[('physical', root)]
    root_permutation = tuple(axes.index(axis) for axis in root_axes)
    root_shape = tuple(dimensions[axes.index(axis)] for axis in root_axes[:-1])+(d, d)
    order.append(root)
    output_order = tuple(order.index(site) for site in range(n))
    physical_permutation = tuple(axis for site in range(n) for axis in (site, n+site))
    namespace = _array_namespace(like)

    def factor(operator):
        carry = namespace.transpose(operator.reshape((d,)*(2*n)), physical_permutation).reshape((d*d,)*n)
        arrays = []
        for permutation, matrix_shape, identity_left, eye, left_shape, move, core_shape, carry_shape in steps:
            matrix = namespace.transpose(carry, permutation).reshape(matrix_shape)
            left, right = (eye, matrix) if identity_left else (matrix, eye)
            arrays.append(namespace.transpose(left.reshape(left_shape), move).reshape(core_shape))
            carry = right.reshape(carry_shape)
        arrays.append(namespace.transpose(carry, root_permutation).reshape(root_shape))
        return tuple(arrays[i] for i in output_order)

    return factor


@dataclass(frozen=True)
class _Recipe:
    site: int
    slot: int
    constants: tuple
    steps: tuple
    output_shape: tuple
    peak: int

    def bind(self, like):
        namespace = _array_namespace(like)
        constants = tuple(ar.do('astype', _as_backend(np.array(c, copy=True), like=like,
                                                     dtype=like.dtype), like.dtype) for c in self.constants)
        steps = self.steps

        def contract(value):
            operands = [value, *constants]
            for positions, equation in steps:
                selected = [operands[i] for i in positions]
                result = namespace.einsum(equation, *selected)
                operands.append(result)
            return operands[-1]

        return contract


def _recipe(site, slot, shape, indices, constants, constant_indices, output, budget):
    """Plan bounded binary contractions once using public Cotengra APIs."""
    import cotengra as ctg
    from .cluster_channels import _budget, _readonly

    inputs = [tuple(indices), *map(tuple, constant_indices)]
    shapes = [shape, *(c.shape for c in constants)]
    sizes = {}
    for inds, dims in zip(inputs, shapes):
        _budget(dims, budget)
        for ind, dim in zip(inds, dims):
            if ind in sizes and sizes[ind] != dim:
                raise ValueError('inconsistent compact contraction dimensions')
            sizes[ind] = dim
    tree = ctg.array_contract_tree(inputs, output, shapes=shapes, optimize='greedy')
    path = tree.get_path()
    active = list(inputs)
    locations = list(range(len(inputs)))
    steps = []
    peak = max(map(prod, shapes))
    for positions in path:
        positions = tuple(map(int, positions))
        selected = [active[i] for i in positions]
        remaining = [inds for i, inds in enumerate(active) if i not in positions]
        needed = set(output).union(*(set(inds) for inds in remaining))
        labels = tuple(dict.fromkeys(ind for inds in selected for ind in inds))
        result = tuple(ind for ind in labels if ind in needed) if remaining else tuple(output)
        if len(labels) > len(ascii_letters):
            raise ValueError('compact contraction exceeds the backend einsum index limit')
        symbols = dict(zip(labels, ascii_letters))
        equation = ','.join(''.join(symbols[i] for i in inds) for inds in selected)
        equation += '->' + ''.join(symbols[i] for i in result)
        dims = tuple(sizes[i] for i in result)
        _budget(dims, budget)
        peak = max(peak, prod(dims))
        steps.append((tuple(locations[i] for i in positions), equation))
        for i in sorted(positions, reverse=True):
            active.pop(i)
            locations.pop(i)
        active.append(result)
        locations.append(len(inputs)+len(steps)-1)
    # All recipes include endpoint maps, including scalar physical sites.
    return _Recipe(site, slot, tuple(_readonly(c) for c in constants), tuple(steps),
                   tuple(sizes[i] for i in output), peak)


class ChannelAssembly:
    """A fixed schedule with no numerical tensors retained from preparation."""

    def __init__(self, binding, arguments, layout, bases, memory_budget, *, dual_bases=None):
        self.binding = binding
        self.layout = layout
        self.memory_budget = memory_budget
        source, _ = binding.resolve(arguments)
        self.source = source if binding.mpo else getattr(source, '_graph', source)
        # Shapes and factorization metadata only: use host zeros, not references.
        if binding.mpo:
            self.keys = tuple(self.source._intervals if self.source.graph is None else self.source._graph_clusters)
            sizes = tuple(key[1]-key[0]+1 if self.source.graph is None else len(key) for key in self.keys)
            dimension = self.source.phys_dim
        else:
            self.keys = tuple(self.source.cluster_plan.index_clusters)
            sizes = tuple(map(len, self.keys))
            dimension = self.source._local.phys_dim
        self.residual_shapes = tuple((dimension**n,)*2 for n in sizes)
        from .cluster_channels import _budget

        for shape in self.residual_shapes:
            _budget(shape, memory_budget)
        zeros = {key: np.zeros(shape, dtype=complex) for key, shape in zip(self.keys, self.residual_shapes)}
        self.groups = {key: [] for key in self.keys}
        self.weights = [[None]*len(keys[0]) for keys in layout.keys]
        for edge, ends in enumerate(layout.ends):
            for side, (site, axis) in enumerate(ends):
                dual = bases[edge].conj() if dual_bases is None else dual_bases[edge]
                self.weights[site][axis] = (layout.sectors[edge], bases[edge] if side == 0 else dual)
        if binding.mpo:
            for site in range(len(layout.sites)):
                for axis in range(2):
                    if self.weights[site][axis] is None:
                        self.weights[site][axis] = ((0,), np.ones((1, 1)))
            self._mpo(zeros)
        else:
            self._graph(zeros, source)
        self.groups = tuple(tuple(self.groups[key]) for key in self.keys)
        recipes = [recipe for group in self.groups for recipe in group]
        self.output_shapes = tuple(next(r.output_shape for r in recipes if r.site == site)
                                   for site in range(len(layout.sites)))
        self.report = dict(assembly='fused-local-contractions', replay_history_blocks=0,
                           projection_contractions=len(recipes),
                           peak_projection_entries=max((r.peak for r in recipes), default=0),
                           projection_constant_entries=sum(c.size for r in recipes for c in r.constants),
                           local_residual_entries=sum(prod(s) for s in self.residual_shapes))
        if binding.frontier is not None:
            self.report.update(binding.frontier.report)
        elif binding.pepo is not None:
            self.report.update(binding.pepo.report)
        else:
            self.report['preparation'] = 'reference'
        # Endpoint maps have been folded into the recipes.
        del self.weights

    def _add(self, key, slot, site, shape, indices, constants, constant_indices):
        output = tuple(('out', i) for i in range(len(self.layout.keys[site][0])))+('y', 'x')
        self.groups[key].append(_recipe(site, slot, shape, indices, constants, constant_indices,
                                       output, self.memory_budget))

    def _rows(self, site, axis, rows):
        sectors, matrix = self.weights[site][axis]
        lookup = {s: i for i, s in enumerate(sectors)}
        return matrix[[lookup[r] for r in rows]]

    def _mpo(self, zeros):
        if self.binding.frontier is not None:
            self._frontier_mpo(zeros)
            return
        source = self.source
        graph = source.graph is not None
        clusters = {key: tuple(key) if graph else tuple(range(key[0], key[1]+1)) for key in self.keys}
        cores = {key: _fixed_mpo(source, zeros[key], key) for key in self.keys if len(clusters[key]) > 1}
        singles = {sites[0]: key for key, sites in clusters.items() if len(sites) == 1}
        plan = source._graph_collection_plan() if graph else {'collections': ()}
        collections = plan['collections']

        def rank(key, cut):
            sites = clusters[key]
            count = sum(site < cut for site in sites)
            return 1 if count in (0, len(sites)) else cores[key][count-1].shape[1]

        # The source's direct layout places the neutral rail first on every cut.
        offsets = [dict() for _ in range(source.L+1)]
        for cut in range(1, source.L):
            offset = 1
            for label, collection in (enumerate(collections) if collections else
                                      ((key, (key,)) for key in cores)):
                if collections or min(clusters[label]) < cut <= max(clusters[label]):
                    offsets[cut][label] = offset
                    offset += prod(rank(key, cut) for key in collection)

        def emit(collection, site, left_rows, right_rows):
            occupied = [(key, clusters[key].index(site)) for key in collection if site in clusters[key]]
            if occupied:
                key, slot = occupied[0]
                shape = cores[key][slot].shape
                indices = [('left', key), ('right', key), 'y', 'x']
            else:
                key, slot = singles[site], -1
                shape, indices = zeros[key].shape, ['y', 'x']
            constants, constant_indices = [], []
            for axis, cut, rows in ((0, site, left_rows), (1, site+1, right_rows)):
                active = [k for k in collection if min(clusters[k]) < cut <= max(clusters[k])]
                # Dimensions one at a cluster endpoint still label a physical core axis.
                dims, inds = [], []
                for k in collection:
                    if k in active or (occupied and k == key):
                        dims.append(rank(k, cut))
                        inds.append((('left' if axis == 0 else 'right'), k) if occupied and k == key else ('wire', k))
                matrix = self._rows(site, axis, rows)
                constants.append(matrix.reshape((*dims, matrix.shape[-1])))
                constant_indices.append((*inds, ('out', axis)))
            self._add(key, slot, site, shape, indices, constants, constant_indices)

        for site in range(source.L):
            emit((), site, [0], [0])
        for label, collection in (enumerate(collections) if collections else ((key, (key,)) for key in cores)):
            sites = range(source.L) if collections else range(min(clusters[label]), max(clusters[label])+1)
            for site in sites:
                rows = []
                for cut in (site, site+1):
                    offset = offsets[cut].get(label, 0)
                    rows.append(range(offset, offset+prod(rank(key, cut) for key in collection)))
                emit(collection, site, *rows)

    def _frontier_mpo(self, zeros):
        plan = self.binding.frontier
        for site, transitions in enumerate(plan.transitions):
            for incoming, outgoing, collection in transitions:
                physical = next((key for key in collection if site in plan.clusters[key]), None)
                if physical is None:
                    key, slot = plan.singles[site], -1
                    shape, indices = zeros[key].shape, ['y', 'x']
                else:
                    key, slot = physical, plan.clusters[physical].index(site)
                    shape = plan.core_shapes[key][slot]
                    indices = [('left', key), ('right', key), 'y', 'x']
                constants, constant_indices = [], []
                for axis, cut, active in ((0, site, incoming), (1, site+1, outgoing)):
                    dims, inds = [], []
                    for k in collection:
                        if k in active or k == physical:
                            dims.append(plan.rank(k, cut))
                            inds.append((('left' if axis == 0 else 'right'), k)
                                        if k == physical else ('wire', k))
                    matrix = self._rows(site, axis, plan.rows(active, cut))
                    constants.append(matrix.reshape((*dims, matrix.shape[-1])))
                    constant_indices.append((*inds, ('out', axis)))
                self._add(key, slot, site, shape, indices, constants, constant_indices)

    def _graph(self, zeros, wrapper):
        from .pepo_dense import _SectorAllocator
        from .pepo_routing import square_routes

        source, layout = self.source, self.layout
        plan = source.cluster_plan
        allocator = _SectorAllocator()
        contributions = []
        graph_sectors = [{0} for _ in plan.lattice.edges]
        positions = {site: i for i, site in enumerate(plan.sites)}
        for key in self.keys:
            if len(key) == 1:
                contributions.append((key, -1, key[0], zeros[key].shape, (), ()))
                continue
            arrays, (tensors, parent, parent_edge, ranks, shape) = _fixed_tree(source, zeros[key], key)
            sectors = {parent_edge[site]: tuple(allocator.allocate(r)) for site, r in ranks.items()}
            for edge, values in sectors.items():
                graph_sectors[edge].update(values)
            for slot, (children, _) in tensors.items():
                edges = tuple(parent_edge[c] for c in children)
                if parent[slot] is not None:
                    edges += (parent_edge[slot],)
                contributions.append((key, slot, positions[shape.sites[slot]], arrays[slot].shape,
                                      edges, tuple(sectors[e] for e in edges)))
        graph_sectors = tuple(tuple(sorted(s)) for s in graph_sectors)
        template = self.binding.pepo
        if template is not None:
            graph_sectors = tuple(tuple(range(n)) for n in template.graph_dimensions)
        if layout.kind == 'square':
            coordinates, directions, routes = square_routes(plan.sites, plan.lattice.edges,
                                                            wrapper.cluster_plan.shape, wrapper.cluster_plan.cyclic)
            wires = {(site, d): [] for site in coordinates for d in directions[site]}
            through = {i: set() for i in range(len(coordinates))}
            coordinate_positions = {s: i for i, s in enumerate(coordinates)}
            for edge, path in enumerate(routes):
                for a, b in path:
                    wires[a].append(edge)
                    wires[b].append(edge)
                for _, (s, _) in path[:-1]:
                    through[coordinate_positions[s]].add(edge)
            axis_wires = [[wires[s, d] for d in directions[s]] for s in coordinates]
        else:
            axis_wires = [[[e] for e, pair in enumerate(plan.lattice.edges) if s in pair] for s in plan.sites]
            through = {i: set() for i in range(len(plan.sites))}
        for key, slot, site, shape, edges, values in contributions:
            selected = dict(zip(edges, values))
            constants, constant_indices = [], []
            for axis, wire_ids in enumerate(axis_wires[site]):
                sectors, matrix = self.weights[site][axis]
                dims = tuple(len(graph_sectors[e]) for e in wire_ids)
                # Square routing numbers the Cartesian product lexicographically.
                if layout.kind == 'square':
                    from .cluster_channels import _budget

                    _budget((prod(dims), matrix.shape[1]), self.memory_budget)
                    dense = np.zeros((prod(dims), matrix.shape[1]), dtype=matrix.dtype)
                    dense[list(sectors)] = matrix
                    matrix = dense.reshape((*dims, matrix.shape[1]))
                else:
                    matrix = matrix.reshape((*dims, matrix.shape[1]))
                indices = []
                current_axis = 0
                for edge in wire_ids:
                    if edge in selected:
                        if template is None:
                            lookup = {s: i for i, s in enumerate(graph_sectors[edge])}
                            matrix = np.take(matrix, [lookup[s] for s in selected[edge]], axis=current_axis)
                        else:
                            from .cluster_channels import _budget

                            sectors, first, second = template.graph_maps[edge]
                            lookup = {s: i for i, s in enumerate(sectors)}
                            endpoint = first if plan.lattice.edges[edge][0] == plan.sites[site] else second
                            endpoint = endpoint[[lookup[s] for s in selected[edge]]]
                            # Fold the graph quotient into this leg's constant
                            # map now; it adds no contraction to live replay.
                            dims = list(matrix.shape)
                            dims[current_axis] = len(selected[edge])
                            _budget(dims, self.memory_budget)
                            matrix = np.moveaxis(np.tensordot(endpoint, matrix, axes=(1, current_axis)),
                                                 0, current_axis)
                        indices.append(('wire', edge))
                        current_axis += 1
                    elif edge in through[site]:
                        indices.append(('wire', edge))
                        current_axis += 1
                    else:
                        matrix = np.take(matrix, 0, axis=current_axis)
                constants.append(matrix)
                constant_indices.append((*indices, ('out', axis)))
            # An isolated site still needs a scalar operand for the common planner.
            if not constants:
                constants, constant_indices = [np.array(1.)], [()]
            self._add(key, slot, site, shape, tuple(('wire', e) for e in edges)+('y', 'x'),
                      constants, constant_indices)

    def pack(self, arguments):
        residuals = self.binding.residuals(arguments)
        if tuple(residuals) != self.keys or tuple(tuple(v.shape) for v in residuals.values()) != self.residual_shapes:
            raise ValueError('cluster residual structure changed; prepare a new channel plan.')
        return tuple(residuals.values())

    def bind(self, like):
        if any(np.iscomplexobj(c) and np.any(c.imag) for group in self.groups for r in group for c in r.constants):
            if 'complex' not in str(like.dtype):
                like = like + 0j
        zero = ar.do('zeros', (), like=like)
        bound = tuple(tuple((r.site, r.slot, r.bind(like)) for r in group) for group in self.groups)
        source, keys, mpo = self.source, self.keys, self.binding.mpo
        empty = tuple(ar.do('zeros', shape, like=like) for shape in self.output_shapes)
        native = ()
        if mpo and source.physical_space.symmetry is not None:
            from ._cluster_native import bind_fixed_sector_operator

            native = tuple(bind_fixed_sector_operator(
                len(key) if source.graph is not None else key[1]-key[0]+1,
                source.physical_space, like) for key in keys)
        trees = tuple(_bind_tree(source, key, like) for key in keys) if not mpo else ()
        algebra = self.binding.algebra if self.binding.pepo is None else self.binding.pepo.algebra
        projections = () if algebra is None else tuple(space.bind(like)[1] for space in algebra.spaces)

        def assemble(residuals):
            if len(residuals) != len(keys):
                raise ValueError('expected one residual per compiled cluster')
            outputs = list(empty)
            for index, (key, value, recipes) in enumerate(zip(keys, residuals, bound)):
                value = value + zero
                if projections:
                    # Identity on the certified model family; arbitrary raw
                    # kernel inputs explicitly get the same static restriction.
                    value = projections[index](value)
                if not recipes:
                    continue
                if all(slot == -1 for _, slot, _ in recipes):
                    arrays = ()
                else:
                    if mpo:
                        arrays = native[index](value) if native else _fixed_mpo(source, value, key)
                    else:
                        arrays = trees[index](value)
                for site, slot, contract in recipes:
                    outputs[site] = outputs[site] + contract(value if slot == -1 else arrays[slot])
            return tuple(outputs)

        return assemble

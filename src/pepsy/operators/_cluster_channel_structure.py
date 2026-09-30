"""Exact MPO transition sharing before reference-based channel projection.

Residual matrix entries, or certified Pauli-span coefficients, are independent
formal atoms. Identities therefore hold for the whole declared family, including
currently equal/zero bindings. The resulting weighted automaton is not claimed
to be globally minimal.
"""

from dataclasses import dataclass, replace

import numpy as np

from ._mpo_sparse import SparseVirtualTensor


@dataclass(frozen=True)
class _Linear:
    # Atom zero is the constant one; positive atoms are residual entries.
    terms: tuple

    @staticmethod
    def coerce(value):
        if isinstance(value, _Linear):
            return value
        return _Linear(()) if value == 0 else _Linear(((0, value),))

    def __add__(self, other):
        terms = dict(self.terms)
        for atom, value in self.coerce(other).terms:
            terms[atom] = terms.get(atom, 0)+value
        return _Linear(tuple(sorted((atom, value) for atom, value in terms.items() if value != 0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = self.coerce(other)
        if not self.terms or not other.terms:
            return _Linear(())
        a, b = self.terms, other.terms
        if a[0][0] != 0 or len(a) != 1:
            a, b = b, a
        if a[0][0] != 0 or len(a) != 1:
            raise ValueError('symbolic fixed MPO core contains a nonlinear residual product')
        return _Linear(tuple((atom, a[0][1]*value) for atom, value in b))

    __rmul__ = __mul__

    def __bool__(self):
        return bool(self.terms)


def _symbolic_arrays(source, layout, limit, assemble=None, algebra=None):
    from .cluster_channels import _budget

    keys = source._intervals if source.graph is None else source._graph_clusters
    lengths = [key[1]-key[0]+1 if source.graph is None else len(key) for key in keys]
    # A coarse entry estimate for Python expression objects; this does not
    # promise a total allocator/RSS bound.
    entries = sum(source.phys_dim**(2*n) for n in lengths)
    entries += sum(map(len, layout.keys))*source.phys_dim**2
    _budget((entries,), limit, itemsize=128)
    residuals, atom = {}, 1
    for index, (key, n) in enumerate(zip(keys, lengths)):
        dimension = source.phys_dim**n
        if algebra is None:
            values = [_Linear(((i, 1),)) for i in range(atom, atom+dimension**2)]
            residuals[key] = np.array(values, dtype=object).reshape(dimension, dimension)
            atom += dimension**2
        else:
            space = algebra.spaces[index]
            residuals[key] = space.symbolic(atom, limit)
            atom += space.size
    if assemble is not None:
        arrays = assemble(residuals)
    elif source.graph is None:
        arrays, _, _ = source._assemble(residuals, sparse=True)
    else:
        plan = source._graph_collection_plan()
        if plan['collections']:
            arrays, _, _ = source._assemble_graph_collections(residuals, collections=plan['collections'], sparse=True)
        else:
            arrays, _, _ = source._assemble_graph(residuals, sparse=True)
    return list(arrays)


def _signature_groups(array, axis, charges):
    signatures = [[] for _ in charges]
    for (left, right), block in sorted(array.blocks.items()):
        source, other = (left, right) if axis == 0 else (right, left)
        entries = tuple(_Linear.coerce(value).terms for value in block.reshape(-1))
        if any(entries):
            signatures[source].append((other, entries))
    # Preserve the neutral rail and its normalization exactly.
    groups, lookup = [[0]], {}
    for state in range(1, len(charges)):
        if not signatures[state]:
            continue  # A formal zero continuation can never carry a derivative.
        signature = (charges[state], tuple(signatures[state]))
        if signature not in lookup:
            lookup[signature] = len(groups)
            groups.append([])
        groups[lookup[signature]].append(state)
    return tuple(tuple(group) for group in groups)


def _map_matrix(groups, size, select, limit):
    from .cluster_channels import _budget

    _budget((size, len(groups)), limit)
    matrix = np.zeros((size, len(groups)))
    for column, group in enumerate(groups):
        matrix[list(group[:1] if select else group), column] = 1
    return matrix


def share_mpo_channels(source, layout, references, limit, *, assemble=None, algebra=None):
    """Return an exact two-sided quotient and transformed QR snapshots.

    Identical rows are selected at one endpoint and summed at its neighbor;
    identical columns use the reverse rule. Keeping separate endpoint maps
    preserves path multiplicity without introducing square-root roundoff into
    the formal transition algebra. Numeric QR runs only after this quotient.
    With a certified Pauli algebra, rational slice elimination also proves constant linear state
    relations; this dense-only path never uses live coefficients as divisors.
    """
    from .cluster_channels import _budget

    symbolic = _symbolic_arrays(source, layout, limit, assemble, algebra)
    sizes = [len(sectors) for sectors in layout.sectors]
    for size in sizes:
        _budget((size, size), limit)
    first = [np.eye(n) for n in sizes]
    second = [np.eye(n) for n in sizes]
    charges = list(layout.charges)
    samples = [[SparseVirtualTensor(symbolic[i].shape, zip(layout.keys[i], values))
                for i, values in enumerate(snapshot)] for snapshot in references]
    merges, rounds = 0, 0
    while True:
        previous = sum(sizes)
        rounds += 1
        for right_to_left in (True, False):
            cuts = range(len(sizes)-1, -1, -1) if right_to_left else range(len(sizes))
            for cut in cuts:
                site, axis = (cut+1, 0) if right_to_left else (cut, 1)
                if algebra is not None:
                    from ._cluster_channel_linear import apply_transfer, delinearize_slices, transfer_matrix

                    # Operator-aware weighted-automaton reduction. These maps
                    # are proved over formal Pauli coefficients, not fitted at
                    # the reference parameters. The neutral rail stays separate.
                    result = delinearize_slices(symbolic[site].blocks, axis, sizes[cut], limit)
                    if result is None:
                        continue
                    select_map, transfer_map, width = result
                    left, right = (transfer_map, select_map) if right_to_left else (select_map, transfer_map)
                    a = tuple({old: row[new] for old, row in enumerate(left) if new in row}
                              for new in range(width))
                    b = tuple({old: row[new] for old, row in enumerate(right) if new in row}
                              for new in range(width))
                    # Keep exact rational arithmetic in the proof, but normal
                    # floating arrays in the numerical reference snapshots.
                    for endpoint, axis_, mapping in ((cut, 1, left), (cut+1, 0, right)):
                        array = symbolic[endpoint]
                        shape = list(array.shape)
                        shape[axis_] = width
                        symbolic[endpoint] = SparseVirtualTensor(
                            shape, apply_transfer(array.blocks, axis_, mapping))
                    for tensors in samples:
                        tensors[cut] = tensors[cut].apply_axis_groups(
                            tuple({i: float(v) for i, v in group.items()} for group in a), axis=1)
                        tensors[cut+1] = tensors[cut+1].apply_axis_groups(
                            tuple({i: float(v) for i, v in group.items()} for group in b), axis=0)
                    first[cut] = first[cut] @ transfer_matrix(left, width, limit)
                    second[cut] = second[cut] @ transfer_matrix(right, width, limit)
                    charges[cut] = (0,)*width  # automaton mode is dense-only
                    merges += sizes[cut]-width
                    sizes[cut] = width
                    continue
                groups = _signature_groups(symbolic[site], axis, charges[cut])
                if len(groups) == sizes[cut]:
                    continue
                select = tuple({g[0]: 1} for g in groups)
                add = tuple(dict.fromkeys(g, 1) for g in groups)
                a, b = (add, select) if right_to_left else (select, add)
                for tensors in (symbolic, *samples):
                    tensors[cut] = tensors[cut].apply_axis_groups(a, axis=1)
                    tensors[cut+1] = tensors[cut+1].apply_axis_groups(b, axis=0)
                first[cut] = first[cut] @ _map_matrix(groups, sizes[cut], not right_to_left, limit)
                second[cut] = second[cut] @ _map_matrix(groups, sizes[cut], right_to_left, limit)
                charges[cut] = tuple(charges[cut][g[0]] for g in groups)
                merges += sizes[cut]-len(groups)
                sizes[cut] = len(groups)
        if sum(sizes) == previous:
            break
    # Include every structurally generated block, even if a reference value is
    # zero. Formal zeros may remain in the map; they are harmless QR columns.
    keys = tuple(tuple(sorted(a.blocks)) for a in symbolic)
    reduced_layout = replace(layout, keys=keys,
                             sectors=tuple(tuple(range(n)) for n in sizes), charges=tuple(charges))
    reduced_samples = [tuple(np.stack([a.blocks.get(k, np.zeros((layout.dimension,)*2)) for k in site_keys])
                             for a, site_keys in zip(snapshot, keys)) for snapshot in samples]
    report = dict(method='exact-symbolic-transitions',
                  bond_dimensions=tuple(sizes), removed_channels=merges,
                  sweeps=rounds, parameter_identity='independent-residual-entries')
    if algebra is not None:
        report.update(method='exact-pauli-weighted-automaton',
                      reduction='rational-delinearisation',
                      parameter_identity='independent-pauli-residual-coefficients',
                      globally_minimal=False)
    return reduced_layout, reduced_samples, tuple(first), tuple(second), report

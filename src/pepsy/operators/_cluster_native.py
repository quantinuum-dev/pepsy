"""Charge-preserving residual factors and virtual levels for native MPOs."""

from itertools import product

import autoray as ar
import numpy as np

from ._mpo_sparse import _charge_groups, _combine_level_charge
from .mpo_automaton import _as_backend


class ChargedCores(tuple):
    """Local cores with structural charges on every cut, including boundaries."""

    def __new__(cls, cores, charges):
        obj = super().__new__(cls, cores)
        obj.charges = tuple(tuple(cut) for cut in charges)
        return obj


def charge_levels(charges):
    from .mpo_semantic import MPOLevel, MPOLevelToken

    return [[MPOLevel(('bond', cut, pos), (MPOLevelToken(
        1 if cut == 0 else 3 if cut == len(charges)-1 else 2,
        ('bond', cut, pos)),), charge)
        for pos, charge in enumerate(values)] for cut, values in enumerate(charges)]


def assembly_charges(states, cores, space, collections=()):
    symmetry = _symmetry(space)
    zero = symmetry.combine()
    result = []
    collection_cuts = [collection_charges(c, cores, len(states)-1, space) for c in collections]
    for cut, labels in enumerate(states):
        collection_values = [charges[cut] for charges in collection_cuts]
        values = []
        for label in labels:
            if label == ('rail',):
                values.append(zero)
                continue
            key, position = label
            if collections:
                values.append(collection_values[key][position])
            else:
                values.append(cores[key].charges[cut-min(key)][position])
        result.append(tuple(values))
    return tuple(result)


def collection_charges(collection, cores, length, space):
    symmetry = _symmetry(space)
    return tuple(tuple(symmetry.combine(*q) for q in product(*(
        cores[c].charges[cut-min(c)] for c in collection if min(c) < cut <= max(c))))
        for cut in range(length+1))


def padded_charges(cores, start, length, space):
    zero = (_symmetry(space).combine(),)
    return (zero,)*start + cores.charges + (zero,)*(length-start-len(cores))


def projector_sector_split(matrix, row_charges, col_charges, max_bond, cutoff, cutoff_mode):
    """Select a total sector rank, then differentiate retained factor pairs.

    Only singular spectra are read for the discrete rank decision. Matrix
    arithmetic and the composed VJP remain on the live backend and device.
    """
    from pepsy.backends.projector_split import projector_split
    from ._cluster_factorization import fixed_split
    from .mpo_semantic import _fixed_rank_svd, _host_singular_values, _select_tt_svd_rank

    rows_by_charge, _ = _charge_groups(row_charges)
    cols_by_charge, _ = _charge_groups(col_charges)
    blocks, spectrum = [], []
    for charge, rows in rows_by_charge.items():
        cols = cols_by_charge.get(charge)
        if not cols:
            continue
        block = ar.do('transpose', take_rows(ar.do('transpose', take_rows(matrix, rows)), cols))
        _, s, _ = _fixed_rank_svd(block)
        values = _host_singular_values(s)
        threshold = max(block.shape)*np.finfo(values.dtype).eps*values[0]
        index = len(blocks)
        blocks.append((charge, rows, cols, block))
        spectrum.extend((float(value), index) for value in values if value > threshold)
    spectrum.sort(key=lambda entry: -entry[0])
    values = np.asarray([value for value, _ in spectrum])
    rank = _select_tt_svd_rank(values, cutoff, cutoff_mode)
    if max_bond is not None:
        rank = min(rank, max_bond)
    counts = [0]*len(blocks)
    for _, index in spectrum[:rank]:
        counts[index] += 1
    if not spectrum:
        counts[0] = 1
    left, right, charges = [], [], []
    for (charge, rows, cols, block), count in zip(blocks, counts):
        if not count:
            continue
        if count >= block.shape[0]:
            # Keeping the whole row space needs no moving singular-vector
            # gauge. The identity is already an exact left isometry.
            u, v = fixed_split(block)
        else:
            u, _, v = projector_split(block, cutoff=0., cutoff_mode='rel', max_bond=count)
        left.append(place_rows(u, rows, len(row_charges)))
        right.append(ar.do('transpose', place_rows(ar.do('transpose', v), cols, len(col_charges))))
        charges.extend([charge]*int(u.shape[1]))
    return (ar.do('concatenate', left, axis=1), ar.do('concatenate', right, axis=0),
            tuple(charges), float(np.linalg.norm(values[rank:])))


def compress_native_accumulator(accumulator, max_bond, cutoff, cutoff_mode, form):
    """Sector TT rounding with a composed NumPy/Torch factor-pair derivative."""
    from .mpo_semantic import FirstDegreeMPO, MPOAdaptiveCompressionReport

    space = accumulator.physical_space
    symmetry = _symmetry(space)
    deltas = _physical_deltas(space, symmetry)
    d = space.phys_dim
    if ar.infer_backend(accumulator.arrays[0]) not in ('numpy', 'torch'):
        raise NotImplementedError('native intermediate rank selection requires NumPy or Torch; '
                                  'use recursive assembly with assembly_chi=None for JAX tracing.')
    if not accumulator.metadata.get('_native_charge_validated', False):
        accumulator.validate_charge_flow()

    def semantic(cores):
        return FirstDegreeMPO(cores, levels=charge_levels(cores.charges), degree=accumulator.degree,
                              physical_space=space, upper_ind_id=accumulator.upper_ind_id,
                              lower_ind_id=accumulator.lower_ind_id,
                              site_tag_id=accumulator.site_tag_id,
                              metadata={'history_valid': False, '_native_charge_validated': True})

    def reverse(cores):
        return ChargedCores((ar.do('transpose', core, (1, 0, 2, 3)) for core in reversed(cores)),
                            (tuple(symmetry.sign(q) for q in cut) for cut in reversed(cores.charges)))

    def sweep(cores, cap, tol):
        arrays, charges, discarded = [], [cores.charges[0]], []
        carry = None
        for site, core in enumerate(cores[:-1]):
            combined = core if carry is None else ar.do('tensordot', carry, core, axes=([1], [0]))
            rows = tuple(symmetry.combine(q, symmetry.sign(delta)) for q in charges[-1] for delta in deltas)
            matrix = ar.do('reshape', ar.do('transpose', combined, (0, 2, 3, 1)), (len(rows), -1))
            u, carry, right, weight = projector_sector_split(matrix, rows, cores.charges[site+1],
                                                            cap, tol, cutoff_mode)
            arrays.append(ar.do('transpose', ar.do('reshape', u, (len(charges[-1]), d, d, len(right))),
                                (0, 3, 1, 2)))
            charges.append(right)
            discarded.append(weight)
        arrays.append(cores[-1] if carry is None else
                      ar.do('tensordot', carry, cores[-1], axes=([1], [0])))
        return ChargedCores(arrays, (*charges, cores.charges[-1])), tuple(discarded)

    cores = ChargedCores(accumulator.arrays, [tuple(_combine_level_charge(level.charge, space.symmetry, symmetry)
                                                  for level in cut) for cut in accumulator.levels])
    if form == 'right':
        cores = reverse(cores)
    prepared, _ = sweep(reverse(cores), None, None)
    compressed, discarded = sweep(reverse(prepared), max_bond, cutoff)
    if form == 'right':
        compressed = reverse(compressed)
    result = semantic(compressed)
    report = MPOAdaptiveCompressionReport(
        method='native-sector-projector', form=form, max_bond=max_bond, cutoff=cutoff,
        cutoff_mode=cutoff_mode, initial_bond_dimensions=accumulator.bond_dimensions,
        final_bond_dimensions=result.bond_dimensions,
        discarded_weights=discarded,
        truncated=accumulator.bond_dimensions != result.bond_dimensions)
    return result, report


def validate_static_generators(basis):
    """Prove conservation for all parameter bindings before entering tracing."""
    from ._cluster_symmetry import coefficient_key

    terms_by_cluster = basis._graph_factor_terms if basis.graph is not None else basis._interval_factor_terms
    matrices_by_cluster = (basis._graph_static_matrices if basis.graph is not None
                           else basis._interval_static_matrices)
    symmetry = _symmetry(basis.physical_space)
    for cluster, factors in terms_by_cluster.items():
        nsites = len(cluster) if basis.graph is not None else cluster[1]-cluster[0]+1
        totals = tuple(symmetry.combine(*charges) for charges in
                       product(basis.physical_space.physical_charges, repeat=nsites))
        forbidden = np.array([[a != b for b in totals] for a in totals])
        for terms, matrices in zip(factors, matrices_by_cluster[cluster]):
            grouped = {}
            for index, (term, matrix) in enumerate(zip(terms, matrices)):
                if matrix is None:
                    raise ValueError('differentiable native clusters require static local operators; '
                                     'put live parameters in coefficients or factor scales.')
                key = coefficient_key(term.coefficient)
                if key is not None and key[0] == 'scalar':
                    key, matrix = ('constant',), term.coefficient*matrix
                elif key is None:
                    key = ('independent', index)
                grouped[key] = grouped.get(key, 0) + matrix
            if any(np.any(matrix[forbidden]) for matrix in grouped.values()):
                raise ValueError('native cluster generators must conserve charge for every '
                                 'independent coefficient binding.')


def fixed_sector_split(matrix, row_charges, col_charges):
    """An exact static split with a homogeneous charge for every channel."""
    from ._cluster_factorization import fixed_split

    rows_by_charge, _ = _charge_groups(row_charges)
    cols_by_charge, _ = _charge_groups(col_charges)
    left, right, charges = [], [], []
    for charge, rows in rows_by_charge.items():
        cols = cols_by_charge.get(charge)
        if not cols:
            continue
        block = take_rows(ar.do('transpose', take_rows(matrix, rows)), cols)
        block = ar.do('transpose', block)
        u, v = fixed_split(block)
        left.append(place_rows(u, rows, len(row_charges)))
        right.append(ar.do('transpose', place_rows(ar.do('transpose', v), cols, len(col_charges))))
        charges.extend([charge]*int(u.shape[1]))
    return ar.do('concatenate', left, axis=1), ar.do('concatenate', right, axis=0), tuple(charges)


def take_rows(matrix, positions):
    return matrix[_as_backend(np.asarray(positions, dtype=int), like=matrix)]


def place_rows(matrix, positions, size):
    """Scatter into a larger axis using a static gather and one zero row."""
    lookup = np.zeros(size, dtype=int)
    lookup[positions] = np.arange(len(positions)) + 1
    padded = ar.do('concatenate', (ar.do('zeros', (1, *matrix.shape[1:]), like=matrix), matrix), axis=0)
    return take_rows(padded, lookup)


def fixed_sector_operator(operator, nsites, space):
    symmetry = _symmetry(space)
    zero = symmetry.combine()
    d = space.phys_dim
    deltas = _physical_deltas(space, symmetry)
    suffix = [(zero,)]
    for _ in range(nsites):
        suffix.append(tuple(symmetry.combine(a, b) for a in deltas for b in suffix[-1]))
    axes = tuple(i for site in range(nsites) for i in (site, nsites+site))
    carry = ar.do('reshape', ar.do('transpose', ar.do('reshape', operator, (d,)*(2*nsites)), axes),
                  (1,)+(d*d,)*nsites)
    charges, cores = [(zero,)], []
    for site in range(nsites-1):
        rows = tuple(symmetry.combine(left, symmetry.sign(delta))
                     for left in charges[-1] for delta in deltas)
        matrix = ar.do('reshape', carry, (len(rows), -1))
        u, v, right = fixed_sector_split(matrix, rows, suffix[nsites-site-1])
        rank = len(right)
        cores.append(ar.do('transpose', ar.do('reshape', u, (len(charges[-1]), d, d, rank)),
                           (0, 3, 1, 2)))
        carry = ar.do('reshape', v, (rank,)+(d*d,)*(nsites-site-1))
        charges.append(right)
    cores.append(ar.do('reshape', carry, (len(charges[-1]), 1, d, d)))
    return ChargedCores(cores, (*charges, (zero,)))



def bind_fixed_sector_operator(nsites, space, like):
    """Bind structural charge gathers outside an array-only compiled kernel.

    This is the same identity-on-the-smaller-sector split as
    ``fixed_sector_operator``. It owns constants only, with no Symmray lookup,
    charge enumeration, numerical decomposition or rank choice during replay.
    """
    from pepsy.backends.convert import _array_namespace

    namespace = _array_namespace(like)
    backend = ar.infer_backend(like)
    concatenate = ar.get_lib_fn(backend, 'cat' if backend == 'torch' else 'concatenate')
    symmetry = _symmetry(space)
    zero, d = symmetry.combine(), space.phys_dim
    deltas = _physical_deltas(space, symmetry)
    suffix = [(zero,)]
    for _ in range(nsites):
        suffix.append(tuple(symmetry.combine(a, b) for a in deltas for b in suffix[-1]))
    charges, splits = (zero,), []

    def indices(values):
        return _as_backend(np.asarray(values, dtype=int), like=like)

    def placement(positions, size):
        lookup = np.zeros(size, dtype=int)
        lookup[positions] = np.arange(len(positions)) + 1
        return indices(lookup)

    for site in range(nsites-1):
        rows = tuple(symmetry.combine(left, symmetry.sign(delta)) for left in charges for delta in deltas)
        cols = suffix[nsites-site-1]
        row_groups, _ = _charge_groups(rows)
        col_groups, _ = _charge_groups(cols)
        groups, right_charges = [], []
        for charge, row_positions in row_groups.items():
            col_positions = col_groups.get(charge)
            if not col_positions:
                continue
            rank = min(len(row_positions), len(col_positions))
            eye = _as_backend(np.eye(rank), like=like, dtype=like.dtype)
            groups.append((indices(row_positions), indices(col_positions),
                           placement(row_positions, len(rows)), placement(col_positions, len(cols)),
                           len(row_positions) <= len(col_positions), eye,
                           ar.do('zeros', (1, rank), like=like)))
            right_charges.extend([charge]*rank)
        splits.append((len(charges), len(rows), len(cols), len(right_charges), tuple(groups)))
        charges = tuple(right_charges)
    axes = tuple(i for site in range(nsites) for i in (site, nsites+site))
    last_rank = len(charges)

    def factor(operator):
        carry = namespace.transpose(operator.reshape((d,)*(2*nsites)), axes).reshape((1,)+(d*d,)*nsites)
        cores = []
        for site, (left_rank, nrows, ncols, rank, groups) in enumerate(splits):
            matrix = carry.reshape(nrows, ncols)
            left, right = [], []
            for row_indices, col_indices, row_place, col_place, identity_left, eye, pad in groups:
                block = matrix[row_indices].T[col_indices].T
                u, v = (eye, block) if identity_left else (block, eye)
                left.append(concatenate((pad, u), 0)[row_place])
                right.append(concatenate((pad, v.T), 0)[col_place].T)
            u = concatenate(left, 1)
            v = concatenate(right, 0)
            cores.append(namespace.transpose(u.reshape(left_rank, d, d, rank), (0, 3, 1, 2)))
            carry = v.reshape((rank,)+(d*d,)*(nsites-site-1))
        cores.append(carry.reshape(last_rank, 1, d, d))
        return tuple(cores)

    return factor


def _symmetry(space):
    import symmray as sr

    return sr.get_symmetry(space.symmetry)


def _physical_deltas(space, symmetry):
    return tuple(symmetry.combine(out, symmetry.sign(inp))
                 for out in space.physical_charges for inp in space.physical_charges)


def sector_operator_schmidt(operator, nsites, space, cutoff, max_bond):
    """Split a neutral local residual without mixing distinct Abelian charges.

    The rank cutoff and cap apply to the combined spectrum across sectors at
    each cut. Zero cutoff retains null channels inside their own sectors.
    This numerical native construction is NumPy-only; no backend graph is
    copied to the host. Dense arrays here cover only the selected cluster.
    """
    from .mpo_product import _select_rank

    if not isinstance(operator, np.ndarray):
        raise NotImplementedError("native cluster MPO factorization requires NumPy residuals.")
    symmetry = _symmetry(space)
    zero = symmetry.combine()
    d = space.phys_dim
    if not np.any(operator):
        return ChargedCores([np.zeros((1, 1, d, d), dtype=operator.dtype)]*nsites,
                            [(zero,)]*(nsites+1))
    deltas = _physical_deltas(space, symmetry)
    suffix = [(zero,)]
    for _ in range(nsites):
        suffix.append(tuple(symmetry.combine(a, b) for a in deltas for b in suffix[-1]))
    axes = tuple(i for site in range(nsites) for i in (site, nsites+site))
    carry = operator.reshape((d,)*(2*nsites)).transpose(axes).reshape((1,)+(d*d,)*nsites)
    left_charges = (zero,)
    bond_charges = [left_charges]
    cores = []
    for site in range(nsites):
        matrix = carry.reshape(len(left_charges)*d*d, -1)
        row_charges = tuple(symmetry.combine(left, symmetry.sign(delta))
                            for left in left_charges for delta in deltas)
        col_charges = suffix[nsites-site-1]
        rows_by_charge, _ = _charge_groups(row_charges)
        cols_by_charge, _ = _charge_groups(col_charges)
        # Refuse a nonconserving target instead of projecting away forbidden
        # entries. Subsequent SVDs operate only within allowed sectors.
        for row, col in zip(*np.nonzero(matrix)):
            if row_charges[row] != col_charges[col]:
                raise ValueError(f"cluster residual violates {space.symmetry} charge conservation.")
        if site == nsites-1:
            cores.append(carry.reshape(len(left_charges), 1, d, d))
            break
        blocks = []
        spectrum = []
        for charge, rows in rows_by_charge.items():
            cols = cols_by_charge.get(charge)
            if not cols:
                continue
            u, s, vh = np.linalg.svd(matrix[np.ix_(rows, cols)], full_matrices=False)
            index = len(blocks)
            blocks.append((charge, rows, cols, u, s, vh))
            spectrum.extend((float(value), index, j) for j, value in enumerate(s))
        spectrum.sort(key=lambda item: -item[0])
        values = np.asarray([value for value, _, _ in spectrum])
        rank = _select_rank(values, cutoff)
        if max_bond is not None:
            rank = min(rank, max_bond)
        left = np.zeros((matrix.shape[0], rank), dtype=operator.dtype)
        right = np.zeros((rank, matrix.shape[1]), dtype=operator.dtype)
        right_charges = []
        for position, (_, index, j) in enumerate(spectrum[:rank]):
            charge, rows, cols, u, s, vh = blocks[index]
            left[rows, position] = u[:, j]
            right[position, cols] = s[j]*vh[j, :]
            right_charges.append(charge)
        cores.append(left.reshape(len(left_charges), d, d, rank).transpose(0, 3, 1, 2))
        carry = right.reshape((rank,)+(d*d,)*(nsites-site-1))
        left_charges = tuple(right_charges)
        bond_charges.append(left_charges)
    return ChargedCores(cores, (*bond_charges, (zero,)))

"""Bounded, backend-native gate fusion for dense exact replay.

Only small operators are assembled. The state is read once per fused block;
diagonal blocks use broadcasting, and dense blocks keep the contraction's
natural output ordering instead of transposing the whole state back.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from math import isqrt

import autoray as ar
import numpy as np

from ...backends import infer_backend_converter_from_sample


_DENSE_QUBITS = 4
_DIAGONAL_QUBITS = 12
_BACKENDS = frozenset({"numpy", "torch", "cupy"})


def supports_batch(state):
    """Capability gate without coercing native symmetry or other backends."""
    return (
        not state.isfermionic()
        and all(ar.infer_backend(t.data) in _BACKENDS for t in state.tensors)
        and all(state.ind_size(ix) == 2 for ix in state.outer_inds())
    )


def _diagonal(gate):
    """Recognize structural zeros, never discard a small off-diagonal entry."""
    if getattr(gate, "requires_grad", False):
        # A zero matrix element can still have a nonzero derivative. Keep
        # trainable matrices on the full, differentiable contraction path.
        return None
    size = isqrt(int(np.prod(gate.shape)))
    matrix = ar.do("reshape", gate, (size, size))
    if ar.infer_backend(matrix) == "torch":
        # NumPy export cannot represent Torch's lazy conjugate/negative bits.
        # Resolve only this tiny gate, never the full state.
        matrix = matrix.resolve_conj().resolve_neg()
    host = np.asarray(ar.to_numpy(matrix))
    off_diagonal = ~np.eye(size, dtype=bool)
    if np.any(host[off_diagonal] != 0):
        return None
    return ar.do("diagonal", matrix)


def _broadcast_diagonal(diagonal, inds, output_inds):
    """Align only a small operator, never transpose the full state."""
    order = tuple(ix for ix in output_inds if ix in inds)
    permutation = tuple(inds.index(ix) for ix in order)
    diagonal = ar.do("reshape", diagonal, (2,) * len(inds))
    if permutation != tuple(range(len(inds))):
        diagonal = ar.do("transpose", diagonal, permutation)
    return ar.do("reshape", diagonal, tuple(2 if ix in inds else 1 for ix in output_inds))


def _apply_dense(data, inds, gate, where):
    axes = tuple(inds.index(ix) for ix in where)
    k = len(where)
    matrix = ar.do("reshape", gate, (2,) * (2 * k))
    result = ar.do("tensordot", matrix, data, (tuple(range(k, 2 * k)), axes))
    # Indices, rather than data, record this permutation. Logical readout
    # already supplies explicit physical indices at the optimizer boundary.
    return result, (*where, *(ix for ix in inds if ix not in where))


@dataclass(frozen=True)
class ExactBatch:
    operator: object
    inds: tuple
    diagonal: bool
    locations: tuple

    def apply(self, tensor):
        """Replace the output array without writing through any shared input."""
        if self.diagonal:
            factor = _broadcast_diagonal(self.operator, self.inds, tensor.inds)
            tensor.modify(data=tensor.data * factor, left_inds=None)
        else:
            data, inds = _apply_dense(tensor.data, tensor.inds, self.operator, self.inds)
            tensor.modify(data=data, inds=inds, left_inds=None)


@dataclass(frozen=True)
class ExactStructuredBatch:
    """A long ZZ layer or same-pair parity block with a safe fallback."""

    kind: str
    operator: object
    inds: tuple
    diagonal: bool
    locations: tuple
    fallback: tuple

    def apply(self, tensor):
        from ._exact_structured import apply_grouped_phase, apply_parity

        n = len(tensor.inds)
        if n > 63:
            result = None
        elif self.kind == "phase":
            edges, same, different = self.operator
            groups = {}
            for a, b in edges:
                bit_a = n - 1 - tensor.inds.index(a)
                bit_b = n - 1 - tensor.inds.index(b)
                low, high = sorted((bit_a, bit_b))
                groups[high - low] = groups.get(high - low, 0) | (1 << low)
            offsets = np.asarray(tuple(groups), dtype=np.int32)
            masks = np.asarray(tuple(groups.values()), dtype=np.uint64)
            table = np.asarray(
                [same[0] ** (len(edges) - k) * different[0] ** k
                 for k in range(len(edges) + 1)],
                dtype=same.dtype,
            )
            result = apply_grouped_phase(tensor.data, offsets, masks, table)
        else:
            mask0, mask1 = (
                1 << (n - 1 - tensor.inds.index(ix)) for ix in self.inds
            )
            result = apply_parity(tensor.data, mask0, mask1, self.operator)

        if result is None:
            for batch in _iter_basic_batches(self.fallback):
                batch.apply(tensor)
        else:
            tensor.modify(data=result, left_inds=None)


def _fuse(entries, inds, diagonal):
    if diagonal:
        operator = _broadcast_diagonal(entries[0][2], entries[0][1], inds)
        for _gate, where, values, _location in entries[1:]:
            operator = operator * _broadcast_diagonal(values, where, inds)
    elif len(entries) == 1:
        operator = entries[0][0]
    else:
        # Identity and all gate composition work are at most 16 x 16.
        convert = infer_backend_converter_from_sample(entries[0][0])
        operator = convert(np.eye(2 ** len(inds))).reshape((2,) * (2 * len(inds)))
        input_inds = tuple(("input", ix) for ix in inds)
        operator_inds = (*inds, *input_inds)
        for gate, where, _values, _location in entries:
            operator, operator_inds = _apply_dense(operator, operator_inds, gate, where)
        permutation = tuple(operator_inds.index(ix) for ix in (*inds, *input_inds))
        operator = ar.do("transpose", operator, permutation)
    return ExactBatch(operator, inds, diagonal, tuple(entry[3] for entry in entries))


def _iter_basic_batches(gate_entries, matrices=None):
    """Existing bounded dense/diagonal fusion for formatted gate entries."""
    entries = []
    inds = ()
    diagonal = False
    diagonals = {}
    for index, (gate, where, location) in enumerate(gate_entries):
        if id(gate) not in diagonals:
            matrix = None if matrices is None else matrices[index]
            if matrix is None:
                diagonals[id(gate)] = _diagonal(gate)
            elif np.any(matrix[~np.eye(4, dtype=bool)] != 0):
                diagonals[id(gate)] = None
            else:
                diagonals[id(gate)] = ar.do(
                    "diagonal", ar.do("reshape", gate, (4, 4))
                )
        values = diagonals[id(gate)]
        is_diagonal = values is not None
        combined = tuple(dict.fromkeys((*inds, *where)))
        limit = _DIAGONAL_QUBITS if is_diagonal else _DENSE_QUBITS
        if entries and (diagonal != is_diagonal or len(combined) > limit):
            yield _fuse(entries, inds, diagonal)
            entries, inds = [], ()
            combined = where
        entries.append((gate, where, values, location))
        inds, diagonal = combined, is_diagonal
    if entries:
        yield _fuse(entries, inds, diagonal)


def _structured_matrix(gate):
    """Inspect only a fixed two-qubit matrix; preserve trainable gradients."""
    if getattr(gate, "requires_grad", False) or int(np.prod(gate.shape)) != 16:
        return None
    matrix = ar.do("reshape", gate, (4, 4))
    host = np.asarray(ar.to_numpy(matrix))
    return host if np.all(np.isfinite(host)) else None


def _zz_values(matrix):
    if matrix is None or np.any(matrix[~np.eye(4, dtype=bool)] != 0):
        return None
    diagonal = np.diag(matrix)
    if diagonal[0] != diagonal[3] or diagonal[1] != diagonal[2]:
        return None
    return diagonal[0], diagonal[1]


def _parity_matrix(matrix):
    if matrix is None:
        return False
    return not (
        np.any(matrix[np.ix_((0, 3), (1, 2))] != 0)
        or np.any(matrix[np.ix_((1, 2), (0, 3))] != 0)
    )


def _structured_batch(kind, entries, matrices, zz_values):
    fallback = tuple(entries)
    locations = tuple(entry[2] for entry in entries)
    if kind == "phase":
        edges = tuple(entry[1] for entry in entries)
        same = np.asarray([pair[0] for pair in zz_values])
        different = np.asarray([pair[1] for pair in zz_values])
        inds = tuple(dict.fromkeys(ix for edge in edges for ix in edge))
        operator = (edges, same, different)
        diagonal = True
    else:
        inds = entries[0][1]
        dtype = np.result_type(*(matrix.dtype for matrix in matrices))
        total = np.eye(4, dtype=dtype)
        swap = np.array([0, 2, 1, 3])
        for entry, matrix in zip(entries, matrices):
            if entry[1] != inds:
                matrix = matrix[np.ix_(swap, swap)]
            total = matrix @ total
        operator = np.asarray(
            (total[0, 0], total[0, 3], total[1, 1], total[1, 2],
             total[2, 1], total[2, 2], total[3, 0], total[3, 3]),
            dtype=dtype,
        )
        diagonal = False
    return ExactStructuredBatch(kind, operator, inds, diagonal, locations, fallback)


def iter_exact_batches(gates, locations, format_ind, *, backend=None, state_size=0):
    """Fuse ordered gates, selecting structural kernels when worthwhile.

    Gate values and classifications are rebuilt on replay so mutable payloads
    and autodiff graphs never go stale. A control event ends this segment.
    """
    entries = []
    for gate, location in zip(gates, locations):
        if len(location) not in (1, 2):
            raise ValueError("Each gate location must have one or two sites.")
        entries.append((gate, tuple(format_ind(site) for site in location), location))

    structured = backend == "cupy" or (backend == "numpy" and find_spec("numba"))
    if not structured or state_size < 1 << 16:
        yield from _iter_basic_batches(entries)
        return

    inspected = {}
    matrices = []
    for gate, where, _ in entries:
        if len(where) != 2:
            matrices.append(None)
        else:
            key = id(gate)
            if key not in inspected:
                inspected[key] = _structured_matrix(gate)
            matrices.append(inspected[key])
    zz = [_zz_values(matrix) for matrix in matrices]
    basic_start = 0
    i = 0
    while i < len(entries):
        j = i
        sites = set()
        while j < len(entries) and zz[j] is not None:
            sites.update(entries[j][1])
            j += 1
        equal_values = (
            j > i and all(pair == zz[i] for pair in zz[i:j])
        )
        unique_edges = {
            frozenset(entries[k][1]) for k in range(i, j)
        }
        if (len(sites) > _DIAGONAL_QUBITS and equal_values
                and len(unique_edges) == j - i):
            yield from _iter_basic_batches(entries[basic_start:i], matrices[basic_start:i])
            yield _structured_batch("phase", entries[i:j], matrices[i:j], zz[i:j])
            i = basic_start = j
            continue

        pair = set(entries[i][1])
        j = i
        while (
            len(pair) == 2 and j < len(entries)
            and set(entries[j][1]) == pair
            and _parity_matrix(matrices[j])
        ):
            j += 1
        if j - i >= 2 and (any(zz[i:j]) or j - i >= 3 or state_size >= 1 << 18):
            yield from _iter_basic_batches(entries[basic_start:i], matrices[basic_start:i])
            yield _structured_batch("parity", entries[i:j], matrices[i:j], zz[i:j])
            i = basic_start = j
            continue
        i += 1

    yield from _iter_basic_batches(entries[basic_start:], matrices[basic_start:])

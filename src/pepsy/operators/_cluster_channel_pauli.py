"""Static Pauli closure certificates and SVD/QR-free residual coordinates.

Use the real Weyl basis X**x Z**z. Y = i XZ, so all symbolic matrix entries
are integers; physical complex phases stay in live expansion coefficients.
"""

from dataclasses import dataclass

import autoray as ar
import numpy as np

from pepsy.backends.convert import _array_namespace
from ._cluster_channel_structure import _Linear
from .mpo_automaton import _as_backend


_PAULI = (np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1, -1]), np.array([[0, -1], [1, 0]]))


def _inventory(source):
    from .mpo_semantic import MPOProductTerm

    local = getattr(source, '_local', source)
    if local.phys_dim != 2:
        raise ValueError('Pauli algebra preparation requires qubit Pauli product terms.')
    words, signature = [], []
    for factor in source.factors:
        entries = []
        for term in factor.terms:
            if not isinstance(term, MPOProductTerm) or term.string_operators is not None:
                raise ValueError('Pauli algebra preparation requires Pauli product terms without string operators.')
            labels = []
            for operator in term.operators:
                if not isinstance(operator, np.ndarray) or operator.shape != (2, 2) or not np.all(np.isfinite(operator)):
                    raise ValueError('Pauli algebra preparation requires fixed finite NumPy Pauli matrices.')
                label = None
                for i, pauli in enumerate(_PAULI):
                    row, col = np.argwhere(pauli != 0)[0]
                    scale = operator[row, col]/pauli[row, col]
                    if np.array_equal(operator, scale*pauli):
                        label = i
                        break
                if label is None:
                    raise ValueError('Pauli algebra preparation requires each local operator to be proportional to I, X, Y or Z.')
                labels.append(label)
            words.append((tuple(term.sites), tuple(labels)))
            entries.append((tuple(term.sites), tuple(labels)))
        signature.append(tuple(entries))
    return tuple(words), tuple(signature)


def _hadamard(values, n, namespace, butterfly):
    dimension = 1 << n
    for bit in range(n):
        stride = 1 << bit
        blocks = values.reshape(dimension, dimension//(2*stride), 2, stride)
        # Fixed 2x2 contractions avoid backend keyword-translation wrappers
        # inside Torch full-graph capture as well as any factorization.
        blocks = namespace.transpose(blocks, (0, 1, 3, 2)) @ butterfly
        values = namespace.transpose(blocks, (0, 1, 3, 2)).reshape(dimension, dimension)
    return values


@dataclass(frozen=True)
class PauliSpace:
    sites: tuple
    words: tuple

    @property
    def dimension(self):
        return 1 << len(self.sites)

    @property
    def full(self):
        return len(self.words) == self.dimension**2

    @property
    def size(self):
        return self.dimension**2 if self.full else len(self.words)

    def symbolic(self, atom, limit):
        from .cluster_channels import _budget

        d = self.dimension
        if self.full:
            return np.array([_Linear(((i, 1),)) for i in range(atom, atom+d*d)], dtype=object).reshape(d, d)
        _budget((len(self.words), d), limit, itemsize=128)
        result = np.empty((d, d), dtype=object)
        result.fill(_Linear(()))
        for i, word in enumerate(self.words):
            x, z = word & (d-1), word >> len(self.sites)
            for column in range(d):
                sign = -1 if (z & column).bit_count() % 2 else 1
                result[column ^ x, column] += _Linear(((atom+i, sign),))
        return result

    def bind(self, like):
        """Bind forward coordinates and exact-on-algebra projection kernels."""
        if self.full:
            return lambda value: value.reshape(-1), lambda value: value
        d, n = self.dimension, len(self.sites)
        namespace = _array_namespace(like)
        # Arrange R[row, column] as R[column XOR x, column]. The permutation
        # is its own inverse. A Walsh transform then yields the X^x Z^z basis.
        ids = np.array([((column ^ x)*d+column) for x in range(d) for column in range(d)])
        rows = np.array([(word & (d-1))*d+(word >> n) for word in self.words])
        mask = np.zeros(d*d)
        mask[rows] = 1
        ids = _as_backend(ids, like=like)
        rows = _as_backend(rows, like=like)
        mask = ar.do('astype', _as_backend(mask.reshape(d, d), like=like, dtype=like.dtype), like.dtype)
        butterfly = ar.do('astype', _as_backend(np.array([[1., 1.], [1., -1.]]),
                                                like=like, dtype=like.dtype), like.dtype)

        def transform(value):
            return _hadamard(value.reshape(-1)[ids].reshape(d, d), n, namespace, butterfly)/d

        def coordinates(value):
            return transform(value).reshape(-1)[rows]

        def project(value):
            coefficients = transform(value)*mask
            return _hadamard(coefficients, n, namespace, butterfly).reshape(-1)[ids].reshape(d, d)

        return coordinates, project


class PauliAlgebra:
    """A closed span for every cluster, independent of all live coefficients."""

    def __init__(self, source, limit, *, clusters=None):
        from .cluster_channels import _budget

        words, self.signature = _inventory(source)
        spaces = []
        if clusters is None:
            clusters = source.cluster_plan.index_clusters
        for cluster in clusters:
            group, n = {0}, len(cluster)
            for sites, labels in words:
                if not set(sites).issubset(cluster):
                    continue
                x = z = 0
                for site, label in zip(sites, labels):
                    bit = 1 << (n-1-cluster.index(site))
                    if label & 1:
                        x |= bit
                    if label & 2:
                        z |= bit
                generator = x | (z << n)
                if generator not in group:
                    _budget((2*len(group),), limit, itemsize=64)
                    group.update(value ^ generator for value in tuple(group))
            spaces.append(PauliSpace(tuple(cluster), tuple(sorted(group))))
        self.spaces = tuple(spaces)
        self.report = dict(operator_basis='pauli-closure',
                           residual_algebra_dimensions=tuple(s.size for s in spaces),
                           unrestricted_residual_dimensions=tuple(s.dimension**2 for s in spaces))

    def validate(self, source):
        if _inventory(source)[1] != self.signature:
            raise ValueError('Pauli operator inventory changed; prepare a new channel plan.')

    def coordinates(self, residuals):
        return tuple(space.bind(value)[0](value) for space, value in zip(self.spaces, residuals.values()))

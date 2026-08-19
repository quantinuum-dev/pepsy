"""Dense square-lattice PEPO cluster-expansion builders.

This module contains the first dense Pepsy implementation of the connected
cluster construction from Vanhecke, Vanderstraeten, and Verstraete.  It is
intentionally separate from the snake-MPO Taylor path: the local residuals
are factorized into PEPO virtual channels, so the approximation is extensive
in the lattice size.

The current implementation supports tree and four-site plaquette-loop
clusters through order four for ordinary dense local operators.  It also has
a fixed-channel Pauli/autodiff path and an explicit Symmray conversion
boundary for homogeneous operator-charge sectors; higher orders and mixed
charge component splitting remain separate extension points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from numbers import Integral

import autoray as ar
import numpy as np
import quimb
import quimb.tensor as qtn

from .mpo_automaton import _as_backend, _backend_reference

__all__ = [
    "ActivePEPOBlocks",
    "ClusterExpansionReport",
    "ClusterExpansionPlan",
    "PauliPEPOTerm",
    "PauliPEPOBasis",
    "CompiledPEPOExp",
    "build_cluster_expansion_pepo",
    "build_itf_cluster_expansion_pepo",
]


_DIRECTIONS = ("u", "r", "d", "l")
_POSITIVE_DIRECTIONS = frozenset(("u", "r"))
_C4_ROTATION = {"u": "r", "r": "d", "d": "l", "l": "u"}
_OPPOSITE_DIRECTION = {"u": "d", "r": "l", "d": "u", "l": "r"}
_DIRECTION_VECTORS = {"u": (1, 0), "r": (0, 1), "d": (-1, 0), "l": (0, -1)}

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_BASIS_CACHE = {}


def _validate_shape(value, name):
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def _validate_cyclic(cyclic, lx, ly):
    if isinstance(cyclic, (bool, np.bool_)):
        cyclic_x = cyclic_y = bool(cyclic)
    else:
        try:
            cyclic_x, cyclic_y = cyclic
        except (TypeError, ValueError) as exc:
            raise TypeError("cyclic must be a bool or a two-item bool tuple.") from exc
        if not isinstance(cyclic_x, (bool, np.bool_)) or not isinstance(
            cyclic_y, (bool, np.bool_)
        ):
            raise TypeError("cyclic must be a bool or a two-item bool tuple.")
        cyclic_x, cyclic_y = bool(cyclic_x), bool(cyclic_y)

    if cyclic_x and lx == 1:
        raise ValueError("periodic x direction requires Lx >= 2.")
    if cyclic_y and ly == 1:
        raise ValueError("periodic y direction requires Ly >= 2.")
    return cyclic_x, cyclic_y


def _as_square_operator(operator, name, *, dtype=None):
    array = np.asarray(operator, dtype=dtype)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square rank-2 matrix.")
    return array


def _normalize_paulis(paulis, *, support):
    if isinstance(paulis, str):
        labels = tuple(paulis.upper())
    else:
        labels = tuple(str(label).upper() for label in paulis)
    expected = 1 if support == "onsite" else 2
    if len(labels) != expected:
        raise ValueError(
            f"{support} Pauli terms need {expected} label(s), got {len(labels)}."
        )
    if any(label not in _PAULI_LABELS for label in labels):
        raise ValueError(
            "Pauli labels must be drawn from 'I', 'X', 'Y', and 'Z'."
        )
    return labels


def _normalize_pauli_support(support):
    aliases = {
        "site": "onsite",
        "one_site": "onsite",
        "one-site": "onsite",
        "onsite": "onsite",
        "bond": "edge",
        "two_site": "edge",
        "two-site": "edge",
        "edge": "edge",
    }
    try:
        return aliases[str(support).lower()]
    except KeyError as exc:
        raise ValueError("PauliPEPOTerm support must be 'onsite' or 'edge'.") from exc


def _normalize_pauli_term(term):
    if isinstance(term, PauliPEPOTerm):
        return term
    if isinstance(term, Mapping):
        support = term.get("support", term.get("kind", term.get("type")))
        paulis = term.get("paulis", term.get("operators"))
        if support is None or paulis is None:
            raise ValueError(
                "each Pauli PEPO term needs support and paulis entries."
            )
        return PauliPEPOTerm(
            support,
            paulis,
            term.get("coefficient", 1.0),
        )
    if isinstance(term, (tuple, list)) and len(term) in (2, 3):
        return PauliPEPOTerm(
            term[0],
            term[1],
            term[2] if len(term) == 3 else 1.0,
        )
    raise TypeError(
        "Pauli PEPO terms must be PauliPEPOTerm values, mappings, or "
        "(support, paulis[, coefficient]) tuples."
    )


def _backend_kron_all(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = ar.do("kron", result, matrix)
    return result


def _backend_expm(matrix):
    """Evaluate a local exponential while retaining Torch/JAX graphs."""
    backend = ar.infer_backend(matrix)
    if backend in ("builtins", "numpy"):
        return np.asarray(quimb.expm(np.asarray(matrix)), dtype=np.asarray(matrix).dtype)
    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        return torch.matrix_exp(matrix)
    if backend == "jax":
        import jax.scipy.linalg as jsp  # pylint: disable=import-outside-toplevel

        return jsp.expm(matrix)
    raise TypeError(
        "PauliPEPOBasis does not have a matrix-exponential rule for backend "
        f"{backend!r}."
    )


def _complexify_backend(value):
    """Promote real backend values before contraction with complex Paulis."""
    return ar.do(
        "multiply",
        value,
        _as_backend(1.0 + 0.0j, like=value),
    )


def _backend_operator_product(factors):
    """Kronecker multiply local operators and return row/column axes grouped."""
    if len(factors) == 1:
        return factors[0]
    local_dim = int(factors[0].shape[0])
    result = _backend_kron_all(*factors)
    nsites = len(factors)
    # A Kronecker product of matrices already has grouped row axes followed
    # by grouped column axes when reshaped this way.
    return ar.do(
        "reshape",
        result,
        (local_dim**nsites, local_dim**nsites),
    )


def _plaquette_edges():
    """Return the canonical four-edge square plaquette topology."""
    return ((0, 1, "u"), (0, 2, "r"), (2, 3, "u"), (1, 3, "r"))


def _backend_embed_operator(operator, positions, nsites, local_dim):
    """Embed a small operator on ordered ``positions`` in a cluster."""
    positions = tuple(positions)
    if len(set(positions)) != len(positions):
        raise ValueError("operator positions must be distinct.")
    other_positions = tuple(site for site in range(nsites) if site not in positions)
    identity = ar.do("eye", local_dim, like=operator)
    factors = [operator, *([identity] * len(other_positions))]
    result = _backend_kron_all(*factors)
    result = ar.do("reshape", result, (local_dim,) * (2 * nsites))
    factor_order = positions + other_positions
    row_axes = tuple(factor_order.index(site) for site in range(nsites))
    axes = row_axes + tuple(nsites + axis for axis in row_axes)
    return ar.do("transpose", result, axes).reshape(
        local_dim**nsites,
        local_dim**nsites,
    )


def _backend_swap_two_site(operator, local_dim):
    reshaped = ar.do(
        "reshape",
        operator,
        (local_dim, local_dim, local_dim, local_dim),
    )
    reshaped = ar.do("transpose", reshaped, (1, 0, 3, 2))
    return ar.do("reshape", reshaped, (local_dim**2, local_dim**2))


def _backend_pauli_basis(nsites, *, like=None):
    """Return the physical Pauli basis, reusing its static NumPy layout."""
    try:
        matrices = _PAULI_BASIS_CACHE[nsites]
    except KeyError:
        matrices = np.stack(
            [
                _backend_operator_product(
                    [_PAULI_MATRICES[label] for label in labels]
                )
                for labels in product(_PAULI_LABELS, repeat=nsites)
            ],
            axis=0,
        )
        _PAULI_BASIS_CACHE[nsites] = matrices
    if like is None:
        return tuple(matrices)
    return tuple(_as_backend(matrix, like=like) for matrix in matrices)


def _backend_pauli_expand(operator, nsites):
    """Expand a local operator in the fixed physical Pauli basis."""
    basis = ar.do(
        "stack",
        _backend_pauli_basis(nsites, like=operator),
        axis=0,
    )
    coefficients = ar.do(
        "tensordot",
        ar.do("conj", basis),
        operator,
        axes=([1, 2], [0, 1]),
    )
    coefficients = ar.do("divide", coefficients, 2**nsites)
    return ar.do("reshape", coefficients, (4,) * nsites)


def _backend_sum_pauli(coefficients, axis):
    """Return ``sum_p coefficients[p] * P_p`` for one physical site."""
    del axis  # retained for the old helper signature
    basis = ar.do(
        "stack",
        _backend_pauli_basis(1, like=coefficients),
        axis=0,
    )
    return ar.do("tensordot", coefficients, basis, axes=([0], [0]))


def _backend_stack(values, *, axis=0):
    if len(values) == 1:
        return ar.do("expand_dims", values[0], axis=axis)
    return ar.do("stack", tuple(values), axis=axis)


def _backend_dtype_itemsize(value):
    try:
        return np.dtype(value.dtype).itemsize
    except (TypeError, ValueError):
        bits = getattr(value.dtype, "bits", None)
        if bits is not None:
            return int(bits) // 8
        return np.asarray(value).dtype.itemsize


def _backend_nonzero(value):
    """Return whether a backend block contains any nonzero entries."""
    try:
        return bool(np.any(np.asarray(ar.to_numpy(value)) != 0))
    except (TypeError, ValueError, AttributeError):
        # A backend may not expose host conversion for a symbolic value. Such
        # a block is retained conservatively; dropping it would break the
        # autodiff graph and is less safe than carrying a zero block.
        return True


def _expm(matrix, dtype):
    """Evaluate a dense exponential without making SciPy a hard dependency."""
    return np.asarray(quimb.expm(np.asarray(matrix, dtype=dtype)), dtype=dtype)


def _kron_all(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def _operator_tensor(operator, nsites, local_dim):
    """View a dense cluster operator as one flattened matrix per site."""
    axes = tuple(axis for site in range(nsites) for axis in (site, nsites + site))
    return operator.reshape((local_dim,) * (2 * nsites)).transpose(axes).reshape(
        (local_dim**2,) * nsites
    )


def _operator_from_tensor(operator_tensor, nsites, local_dim):
    """Undo :func:`_operator_tensor` for a local operator tensor."""
    pair_axes = tuple(range(2 * nsites))
    row_axes = pair_axes[::2]
    column_axes = pair_axes[1::2]
    return operator_tensor.reshape((local_dim, local_dim) * nsites).transpose(
        row_axes + column_axes
    ).reshape(local_dim**nsites, local_dim**nsites)


def _permute_operator_sites(operator, order, local_dim):
    """Return ``operator`` with its site order changed by ``order``."""
    tensor = _operator_tensor(operator, len(order), local_dim)
    tensor = tensor.transpose(order)
    return _operator_from_tensor(tensor, len(order), local_dim)


def _cycle_active_operator(blocks, site_directions, loop_sites, physical_dim):
    """Contract the existing active blocks around one four-site cycle.

    The dense plan shares tree channel ranges so some lower-order products can
    already close around a plaquette.  This helper measures that contribution
    before adding the explicit loop sector; it prevents double counting while
    retaining the existing tree-channel compatibility contract.
    """
    pairs = (("r", "u"), ("d", "r"), ("l", "d"), ("u", "l"))
    active = []
    for site, pair in zip(loop_sites, pairs):
        directions = site_directions[site]
        pair_indices = tuple(directions.index(direction) for direction in pair)
        entries = {}
        for key, block in blocks[site].items():
            if any(
                key[index]
                for index, direction in enumerate(directions)
                if direction not in pair
            ):
                continue
            pair_key = tuple(key[index] for index in pair_indices)
            entries[pair_key] = (
                entries[pair_key] + block
                if pair_key in entries
                else block
            )
        active.append(entries)

    result = np.zeros(
        (physical_dim**4, physical_dim**4),
        dtype=np.result_type(
            *(block for entries in active for block in entries.values())
        ),
    )
    for (right, upper), first in active[0].items():
        for (down, diagonal), second in active[1].items():
            if down != upper:
                continue
            for (upper_again, right_again), third in active[2].items():
                if upper_again != diagonal:
                    continue
                fourth = active[3].get((right_again, right))
                if fourth is None:
                    continue
                result += _kron_all(first, second, third, fourth)
    return result


def _swap_two_site_operator(operator, local_dim):
    return operator.reshape(local_dim, local_dim, local_dim, local_dim).transpose(
        1, 0, 3, 2
    ).reshape(local_dim**2, local_dim**2)


def _oriented_two_site_operator(operator, local_dim, direction):
    if direction in _POSITIVE_DIRECTIONS:
        return operator
    return _swap_two_site_operator(operator, local_dim)


def _embed_one_site_operator(operator, position, nsites, local_dim):
    identity = np.eye(local_dim, dtype=operator.dtype)
    factors = [identity] * nsites
    factors[position] = operator
    return _kron_all(*factors)


def _embed_two_site_operator(operator, positions, nsites, local_dim):
    """Embed a rank-two-site matrix, preserving the supplied site ordering."""
    pos0, pos1 = positions
    dimension = local_dim**nsites
    result = np.zeros((dimension, dimension), dtype=operator.dtype)
    for row in np.ndindex(*(local_dim for _ in range(nsites))):
        row_flat = np.ravel_multi_index(row, (local_dim,) * nsites)
        for col in np.ndindex(*(local_dim for _ in range(nsites))):
            if any(row[pos] != col[pos] for pos in range(nsites) if pos not in positions):
                continue
            col_flat = np.ravel_multi_index(col, (local_dim,) * nsites)
            row_pair = row[pos0] * local_dim + row[pos1]
            col_pair = col[pos0] * local_dim + col[pos1]
            result[row_flat, col_flat] = operator[row_pair, col_pair]
    return result


def _tree_hamiltonian(nsites, edges, twosite_op, onesite_op):
    """Build the dense Hamiltonian on a small embedded tree cluster."""
    local_dim = onesite_op.shape[0]
    result = sum(
        (
            _embed_one_site_operator(onesite_op, position, nsites, local_dim)
            for position in range(nsites)
        ),
        start=np.zeros((local_dim**nsites, local_dim**nsites), dtype=onesite_op.dtype),
    )
    for source, target, direction in edges:
        result += _embed_two_site_operator(
            _oriented_two_site_operator(twosite_op, local_dim, direction),
            (source, target),
            nsites,
            local_dim,
        )
    return result


def _edge_contribution(nsites, edges, one_site_exp, start_factors, end_factors):
    """Evaluate all two-site PEPO channels on a small tree."""
    if not start_factors.shape[0]:
        return np.zeros_like(
            _kron_all(*(one_site_exp for _ in range(nsites)))
        )

    result = np.zeros(
        (one_site_exp.shape[0] ** nsites, one_site_exp.shape[1] ** nsites),
        dtype=np.result_type(one_site_exp, start_factors, end_factors),
    )
    for source, target, direction in edges:
        source_factors = start_factors if direction in _POSITIVE_DIRECTIONS else end_factors
        target_factors = end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for index in range(start_factors.shape[0]):
            factors = [one_site_exp] * nsites
            factors[source] = source_factors[index]
            factors[target] = target_factors[index]
            result += _kron_all(*factors)
    return result


def _three_cluster_contribution(
    nsites,
    center,
    branches,
    pair_tensor,
    start_factors,
    end_factors,
    one_site_exp,
):
    """Evaluate one connected three-site correction inside a larger tree."""
    if not pair_tensor.size:
        return np.zeros_like(_kron_all(*(one_site_exp for _ in range(nsites))))

    result = np.zeros(
        (one_site_exp.shape[0] ** nsites, one_site_exp.shape[1] ** nsites),
        dtype=np.result_type(one_site_exp, pair_tensor),
    )
    directions = tuple(direction for _, direction in branches)
    order = np.argsort([_DIRECTIONS.index(direction) for direction in directions])
    ordered_branches = tuple(branches[index] for index in order)
    for first in range(pair_tensor.shape[0]):
        for second in range(pair_tensor.shape[1]):
            factors = [one_site_exp] * nsites
            factors[center] = pair_tensor[first, second]
            for channel, (endpoint, direction) in enumerate(ordered_branches):
                endpoint_factors = (
                    end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
                )
                factors[endpoint] = endpoint_factors[
                    (first, second)[channel]
                ]
            result += _kron_all(*factors)
    return result


def _three_subtrees(edges):
    """Yield the two-edge connected subtrees of a four-site tree."""
    for first_index, first in enumerate(edges):
        for second in edges[first_index + 1 :]:
            first_vertices = {first[0], first[1]}
            second_vertices = {second[0], second[1]}
            common = first_vertices & second_vertices
            if len(common) != 1:
                continue
            center = next(iter(common))
            branches = []
            for source, target, direction in (first, second):
                if source == center:
                    branches.append((target, direction))
                else:
                    branches.append((source, _OPPOSITE_DIRECTION[direction]))
            yield center, tuple(branches)


def _pair_pair_contribution(
    edges,
    pair_tensors,
    start_factors,
    end_factors,
    one_site_exp,
):
    """Evaluate the lower-order product of two adjacent pair-center blocks."""
    adjacency = {index: [] for index in range(4)}
    for source, target, direction in edges:
        adjacency[source].append((target, direction))
        adjacency[target].append((source, _OPPOSITE_DIRECTION[direction]))
    endpoints = [vertex for vertex, links in adjacency.items() if len(links) == 1]
    if len(endpoints) != 2 or any(len(links) > 2 for links in adjacency.values()):
        return np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))

    path = [endpoints[0]]
    steps = []
    previous = None
    while len(path) < 4:
        current = path[-1]
        next_vertex, direction = next(
            link for link in adjacency[current] if link[0] != previous
        )
        steps.append(direction)
        path.append(next_vertex)
        previous = current

    first_factors = start_factors if steps[0] in _POSITIVE_DIRECTIONS else end_factors
    last_factors = end_factors if steps[2] in _POSITIVE_DIRECTIONS else start_factors
    rank = start_factors.shape[0]
    result = np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))

    def pair_entry(first_direction, second_direction, first, second):
        pair = tuple(
            sorted((first_direction, second_direction), key=_DIRECTIONS.index)
        )
        tensor = pair_tensors[pair]
        indices = {
            first_direction: first,
            second_direction: second,
        }
        return tensor[indices[pair[0]], indices[pair[1]]]

    del path  # only the ordered edge directions are needed below
    first_back = _OPPOSITE_DIRECTION[steps[0]]
    first_forward = steps[1]
    second_back = _OPPOSITE_DIRECTION[steps[1]]
    second_forward = steps[2]
    for first in range(rank):
        for middle in range(rank):
            for last in range(rank):
                result += _kron_all(
                    first_factors[first],
                    pair_entry(
                        first_back,
                        first_forward,
                        first,
                        middle,
                    ),
                    pair_entry(
                        second_back,
                        second_forward,
                        middle,
                        last,
                    ),
                    last_factors[last],
                )
    return result


def _disconnected_edge_product(edges, start_factors, end_factors, one_site_exp):
    """Evaluate products of disjoint two-site residual channels."""
    result = np.zeros_like(_kron_all(*(one_site_exp for _ in range(4))))
    for first_index, first_edge in enumerate(edges):
        first_vertices = {first_edge[0], first_edge[1]}
        for second_edge in edges[first_index + 1 :]:
            if first_vertices & {second_edge[0], second_edge[1]}:
                continue
            first_source, first_target, first_direction = first_edge
            second_source, second_target, second_direction = second_edge
            first_source_factors = (
                start_factors
                if first_direction in _POSITIVE_DIRECTIONS
                else end_factors
            )
            first_target_factors = (
                end_factors
                if first_direction in _POSITIVE_DIRECTIONS
                else start_factors
            )
            second_source_factors = (
                start_factors
                if second_direction in _POSITIVE_DIRECTIONS
                else end_factors
            )
            second_target_factors = (
                end_factors
                if second_direction in _POSITIVE_DIRECTIONS
                else start_factors
            )
            for first_channel in range(start_factors.shape[0]):
                for second_channel in range(start_factors.shape[0]):
                    factors = [one_site_exp] * 4
                    factors[first_source] = first_source_factors[first_channel]
                    factors[first_target] = first_target_factors[first_channel]
                    factors[second_source] = second_source_factors[second_channel]
                    factors[second_target] = second_target_factors[second_channel]
                    result += _kron_all(*factors)
    return result


def _lower_tree_residual(
    nsites,
    edges,
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors=None,
):
    """Return a tree residual after subtracting all lower connected terms."""
    exact = _expm(
        -beta * _tree_hamiltonian(nsites, edges, twosite_op, onesite_op),
        onesite_op.dtype,
    )
    residual = exact - _kron_all(*(one_site_exp for _ in range(nsites)))
    residual -= _edge_contribution(
        nsites, edges, one_site_exp, start_factors, end_factors
    )
    if pair_tensors:
        for center, branches in _three_subtrees(edges):
            directions = tuple(direction for _, direction in branches)
            pair = tuple(sorted(directions, key=_DIRECTIONS.index))
            residual -= _three_cluster_contribution(
                nsites,
                center,
                branches,
                pair_tensors[pair],
                start_factors,
                end_factors,
                one_site_exp,
            )
        if nsites == 4 and len(edges) == 3:
            residual -= _disconnected_edge_product(
                edges,
                start_factors,
                end_factors,
                one_site_exp,
            )
            residual -= _pair_pair_contribution(
                edges,
                pair_tensors,
                start_factors,
                end_factors,
                one_site_exp,
            )
    return exact, residual


def _lower_loop_residual(
    edges,
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
):
    """Return a four-site plaquette residual after all lower clusters.

    A plaquette has four connected three-site subtrees and two disconnected
    opposite-edge products.  The latter is the only lower-order subtraction
    that differs from the four-site tree residual used by
    :func:`_lower_tree_residual`; no four-site path is embedded on a plaquette
    because the path endpoints are already nearest neighbours.
    """
    exact = _expm(
        -beta * _tree_hamiltonian(4, edges, twosite_op, onesite_op),
        onesite_op.dtype,
    )
    residual = exact - _kron_all(*(one_site_exp for _ in range(4)))
    residual -= _edge_contribution(
        4, edges, one_site_exp, start_factors, end_factors
    )
    for center, branches in _three_subtrees(edges):
        directions = tuple(direction for _, direction in branches)
        pair = tuple(sorted(directions, key=_DIRECTIONS.index))
        residual -= _three_cluster_contribution(
            4,
            center,
            branches,
            pair_tensors[pair],
            start_factors,
            end_factors,
            one_site_exp,
        )
    residual -= _disconnected_edge_product(
        edges,
        start_factors,
        end_factors,
        one_site_exp,
    )
    return exact, residual


def _edge_factors(
    twosite_op,
    onesite_op,
    beta,
    *,
    edge_cutoff,
    max_edge_rank,
    symmetric=False,
):
    """Return the two endpoint factors of the connected two-site residual."""
    local_dim = onesite_op.shape[0]
    identity = np.eye(local_dim, dtype=onesite_op.dtype)
    one_site_exp = _expm(-beta * onesite_op, onesite_op.dtype)
    two_site_hamiltonian = (
        twosite_op
        + np.kron(onesite_op, identity)
        + np.kron(identity, onesite_op)
    )
    residual = _expm(-beta * two_site_hamiltonian, onesite_op.dtype) - np.kron(
        one_site_exp, one_site_exp
    )

    residual = residual.reshape(local_dim, local_dim, local_dim, local_dim).transpose(
        0, 2, 1, 3
    ).reshape(local_dim**2, local_dim**2)
    if symmetric:
        if not np.allclose(residual, residual.T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "C4 cluster reduction requires a site-exchange-symmetric "
                "two-site residual."
            )
        if not np.allclose(residual, residual.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "C4 cluster reduction currently requires a real/Hermitian "
                "reshuffled two-site residual."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        ordering = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[ordering]
        eigenvectors = eigenvectors[:, ordering]
        singular_values = np.abs(eigenvalues)
    else:
        left, singular_values, right = np.linalg.svd(residual, full_matrices=False)
    if not singular_values.size or singular_values[0] == 0:
        empty = np.zeros((0, local_dim, local_dim), dtype=onesite_op.dtype)
        return one_site_exp, empty, empty

    edge_cutoff = float(edge_cutoff)
    threshold = singular_values[0] * (
        edge_cutoff
        if edge_cutoff > 0.0
        else np.finfo(singular_values.dtype).eps * local_dim**2
    )
    keep = np.flatnonzero(singular_values > threshold)
    if max_edge_rank is not None:
        keep = keep[:max_edge_rank]
    singular_values = singular_values[keep]
    if not singular_values.size:
        empty = np.zeros((0, local_dim, local_dim), dtype=onesite_op.dtype)
        return one_site_exp, empty, empty

    if symmetric:
        root = np.sqrt(eigenvalues[keep].astype(np.result_type(onesite_op.dtype, complex)))
        factors = (eigenvectors[:, keep] * root).T.reshape(-1, local_dim, local_dim)
        return one_site_exp, factors, factors.copy()

    root = np.sqrt(singular_values)
    start = (left[:, keep] * root).T.reshape(-1, local_dim, local_dim)
    end = (root[:, None] * right[keep, :]).reshape(-1, local_dim, local_dim)
    return one_site_exp, start, end


def _edge_fit_residual(
    twosite_op,
    onesite_op,
    beta,
    one_site_exp,
    start_factors,
    end_factors,
):
    """Return the unrepresented two-site residual after channel truncation."""
    local_dim = onesite_op.shape[0]
    identity = np.eye(local_dim, dtype=onesite_op.dtype)
    hamiltonian = (
        twosite_op
        + np.kron(onesite_op, identity)
        + np.kron(identity, onesite_op)
    )
    exact = _expm(-beta * hamiltonian, onesite_op.dtype)
    residual = exact - np.kron(one_site_exp, one_site_exp)
    for index in range(start_factors.shape[0]):
        residual -= np.kron(start_factors[index], end_factors[index])
    return residual


def _solve_three_site_pair(
    pair,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    beta,
):
    """Solve the residual for one center with two active virtual directions."""
    local_dim = onesite_op.shape[0]
    rank = start_factors.shape[0]

    edges = ((0, 1, pair[0]), (0, 2, pair[1]))
    _, residual = _lower_tree_residual(
        3,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
    )

    endpoint_factors = [
        end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for direction in pair
    ]
    # Project each endpoint directly onto its operator-Schmidt space.  The
    # old implementation built a ``d**6 x (r**2 * d**2)`` Kronecker design
    # and solved it as one dense system.  That is mathematically equivalent,
    # but it makes a small four-site cluster pay for a much larger LAPACK
    # solve.  These two pseudoinverses are only ``d**2 x r`` and preserve the
    # least-squares behavior when an endpoint channel basis is truncated.
    endpoint_matrices = tuple(
        factors.reshape(rank, local_dim**2).T for factors in endpoint_factors
    )
    endpoint_pseudoinverses = tuple(
        np.linalg.pinv(factors) for factors in endpoint_matrices
    )
    residual_tensor = _operator_tensor(residual, 3, local_dim)
    center_coefficients = np.einsum(
        "ijk,aj,bk->iab",
        residual_tensor,
        endpoint_pseudoinverses[0],
        endpoint_pseudoinverses[1],
    )
    solution = center_coefficients.transpose(1, 2, 0).reshape(
        rank, rank, local_dim, local_dim
    )
    fitted = np.einsum(
        "ia,jab,kb->ijk",
        endpoint_matrices[0],
        center_coefficients,
        endpoint_matrices[1],
    )
    return solution, np.linalg.norm(residual_tensor - fitted), np.linalg.norm(residual_tensor)


def _site_directions(i, j, lx, ly, cyclic_x, cyclic_y):
    return tuple(
        direction
        for direction, present in (
            ("u", cyclic_x or i < lx - 1),
            ("r", cyclic_y or j < ly - 1),
            ("d", cyclic_x or i > 0),
            ("l", cyclic_y or j > 0),
        )
        if present
    )


def _materialize_site_blocks(directions, blocks, bond_dim, dtype):
    physical_dim = blocks[(0,) * len(directions)].shape[0]
    reference = blocks[(0,) * len(directions)]
    shape = (bond_dim,) * len(directions) + (physical_dim, physical_dim)
    if ar.infer_backend(reference) not in ("builtins", "numpy"):
        data = ar.do("zeros", shape, like=reference)
        for key, block in blocks.items():
            mask = None
            for axis, sector in enumerate(key):
                selector = ar.do("eye", bond_dim, like=reference)[:, sector]
                selector_shape = [1] * len(directions)
                selector_shape[axis] = bond_dim
                selector = ar.do("reshape", selector, tuple(selector_shape))
                mask = selector if mask is None else ar.do("multiply", mask, selector)
            block = ar.do("transpose", block, (1, 0))
            block = ar.do(
                "reshape",
                block,
                (1,) * len(directions) + (physical_dim, physical_dim),
            )
            data = ar.do("add", data, ar.do("multiply", mask[..., None, None], block))
        return data

    data = np.zeros(shape, dtype=dtype)
    for key, block in blocks.items():
        # Quimb's ``to_dense`` convention transposes each local ``b``/``k``
        # block when flattening an operator. Store the inverse local
        # transpose here so the materialized PEPO has the requested matrix
        # orientation.
        data[key + (slice(None), slice(None))] = block.T
    return data


@dataclass(frozen=True)
class ClusterExpansionReport:
    """Numerical and storage diagnostics from one cluster-expansion build.

    ``residual_norms`` report the largest local residual left after the
    corresponding factorization or least-squares solve. They are local
    operator norms in the dense Frobenius metric, not a global PEPO error
    bound. ``cluster_counts`` count embedded cluster instances on the chosen
    finite lattice; the separate ``*_solved`` entries expose C4 reuse.
    """

    beta: object
    order: int
    local_dim: int
    edge_rank: int
    tree_rank: int
    loop_rank: int
    cluster_counts: dict[str, int]
    residual_norms: dict[str, float]
    relative_residual_norms: dict[str, float]
    active_block_count: int
    active_nbytes: int
    dense_nbytes: int

    @property
    def max_residual_norm(self):
        """Return the largest reported local residual."""
        return max(self.residual_norms.values(), default=0.0)

    @property
    def max_relative_residual(self):
        """Return the largest residual relative to its uncompressed target."""
        return max(self.relative_residual_norms.values(), default=0.0)


@dataclass
class ActivePEPOBlocks:
    """Sparse active virtual-sector blocks for a finite PEPO lattice.

    ``blocks[(i, j)]`` maps a tuple of virtual-sector integers to its physical
    operator block. Sector ``0`` is the trivial channel; positive sectors are
    compact active channels. The dense Quimb PEPO is created only by
    :meth:`to_pepo`, keeping the mostly-zero construction intermediate small.
    """

    lx: int
    ly: int
    cyclic: tuple[bool, bool]
    bond_dim: int
    physical_dim: int
    site_directions: dict
    blocks: dict
    charge_symmetry: str | None = None
    physical_sectors: dict | None = None
    virtual_sector_charges: dict | None = None

    @property
    def active_block_count(self):
        """Return the number of stored nonzero sector blocks."""
        return sum(len(site_blocks) for site_blocks in self.blocks.values())

    @property
    def dense_nbytes(self):
        """Estimate bytes required by dense PEPO site tensors."""
        reference = next(iter(next(iter(self.blocks.values())).values()))
        itemsize = _backend_dtype_itemsize(reference)
        return sum(
            self.bond_dim ** len(self.site_directions[site])
            * self.physical_dim**2
            * itemsize
            for site in self.blocks
        )

    @property
    def active_nbytes(self):
        """Return bytes occupied by the stored active blocks."""
        total = 0
        for site_blocks in self.blocks.values():
            for block in site_blocks.values():
                nbytes = getattr(block, "nbytes", None)
                total += int(nbytes if nbytes is not None else np.asarray(block).nbytes)
        return total

    def compact(self):
        """Remove zero blocks and globally orphaned virtual sectors.

        Sector ids are implementation labels, so compacting them is
        lossless. The relative order of surviving ids is preserved, which
        keeps repeated autodiff evaluations compatible with the same active
        topology while dropping channels that were never connected on the
        chosen finite lattice.
        """
        compact_blocks = {
            site: {
                key: block
                for key, block in site_blocks.items()
                if _backend_nonzero(block)
            }
            for site, site_blocks in self.blocks.items()
        }
        # A channel endpoint with no nonzero block on the opposite side of
        # its bond is an orphan.  Iterating to a fixed point also removes
        # higher-order blocks that became disconnected after their endpoint
        # channels were pruned.
        changed = True
        while changed:
            changed = False
            available = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                for axis, direction in enumerate(directions):
                    available[(site, direction)] = {
                        key[axis] for key in site_blocks
                    }
            retained = {}
            for site, site_blocks in compact_blocks.items():
                directions = self.site_directions[site]
                kept_site = {}
                for key, block in site_blocks.items():
                    keep = True
                    for axis, direction in enumerate(directions):
                        sector = key[axis]
                        if sector == 0:
                            continue
                        neighbor = _site_after(
                            site,
                            direction,
                            self.lx,
                            self.ly,
                            self.cyclic,
                        )
                        if neighbor is None:
                            keep = False
                            break
                        opposite = _OPPOSITE_DIRECTION[direction]
                        if sector not in available[(neighbor, opposite)]:
                            keep = False
                            break
                    if keep:
                        kept_site[key] = block
                    else:
                        changed = True
                retained[site] = kept_site
            compact_blocks = retained

        used = {0}
        for site_blocks in compact_blocks.values():
            for key in site_blocks:
                used.update(key)
        sector_map = {
            old: new for new, old in enumerate(sorted(used))
        }
        remapped_blocks = {
            site: {
                tuple(sector_map[sector] for sector in key): block
                for key, block in site_blocks.items()
            }
            for site, site_blocks in compact_blocks.items()
        }
        old_charges = self.virtual_sector_charges or {}
        remapped_charges = {
            sector_map[old]: old_charges.get(old, 0)
            for old in sorted(used)
        }
        return type(self)(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=len(used),
            physical_dim=self.physical_dim,
            site_directions=self.site_directions,
            blocks=remapped_blocks,
            charge_symmetry=self.charge_symmetry,
            physical_sectors=self.physical_sectors,
            virtual_sector_charges=remapped_charges,
        )

    remove_orphans = compact

    def to_symmray_pepo(
        self,
        *,
        symmetry="U1",
        physical_sectors=None,
        virtual_charges=None,
        charge=0,
        fermionic=False,
        remove_orphans=True,
    ):
        """Materialize active blocks as a native Symmray-backed PEPO.

        ``virtual_charges`` maps the integer active-history ids to symmetry
        charges. Multiple history ids may share one charge; they become a
        proper Symmray degeneracy block rather than a dense virtual axis.
        Every nonzero local block is checked against the requested homogeneous
        operator ``charge``. This means a mixed-charge exponential (for
        example an unsplit ``exp(h X)`` under Z2) must be represented as
        separate charge components before conversion.

        The returned object is a Quimb PEPO whose site arrays are native
        Symmray arrays. Backend-valued blocks are sliced and assembled using
        Autoray operations, so Torch/JAX coefficient graphs are preserved.
        """
        from pepsy.tensors.symmetric import (  # pylint: disable=import-outside-toplevel
            _array_class_for_symmetry,
            default_physical_sectors,
        )

        active = self
        provided_charges = virtual_charges
        if remove_orphans:
            if provided_charges is not None:
                active = type(self)(
                    lx=self.lx,
                    ly=self.ly,
                    cyclic=self.cyclic,
                    bond_dim=self.bond_dim,
                    physical_dim=self.physical_dim,
                    site_directions=self.site_directions,
                    blocks=self.blocks,
                    charge_symmetry=self.charge_symmetry,
                    physical_sectors=self.physical_sectors,
                    virtual_sector_charges=dict(provided_charges),
                )
            active = active.compact()
        if physical_sectors is None:
            physical_sectors = active.physical_sectors
        if physical_sectors is None:
            physical_sectors = default_physical_sectors(
                symmetry,
                active.physical_dim,
            )
        physical_sectors = dict(physical_sectors)
        if sum(int(size) for size in physical_sectors.values()) != active.physical_dim:
            raise ValueError(
                "physical_sectors must describe exactly the PEPO physical dimension."
            )
        if provided_charges is None:
            provided_charges = active.virtual_sector_charges
        if provided_charges is None:
            provided_charges = {
                sector: 0 for sector in range(active.bond_dim)
            }
        virtual_charges = dict(provided_charges)
        missing = set(range(active.bond_dim)) - set(virtual_charges)
        if missing:
            raise ValueError(
                "virtual_charges is missing active sector ids "
                f"{sorted(missing)}."
            )

        import symmray as sr  # pylint: disable=import-outside-toplevel
        array_cls = _array_class_for_symmetry(
            symmetry,
            fermionic=fermionic,
        )
        symmetry_obj = array_cls.get_class_symmetry(symmetry)
        physical_items = tuple(physical_sectors.items())
        physical_offsets = {}
        offset = 0
        for physical_charge, size in physical_items:
            size = int(size)
            physical_offsets[physical_charge] = (offset, offset + size)
            offset += size

        arrays = []
        native_arrays = {}
        for i in range(active.lx):
            row = []
            for j in range(active.ly):
                site = (i, j)
                directions = active.site_directions[site]
                virtual_duals = tuple(direction in ("d", "l") for direction in directions)
                charge_groups = {}
                for sector in range(active.bond_dim):
                    charge_groups.setdefault(virtual_charges[sector], []).append(sector)
                charge_sizes = {
                    axis_charge: len(sectors)
                    for axis_charge, sectors in charge_groups.items()
                }
                block_arrays = {}
                for key, block in active.blocks[site].items():
                    if not _backend_nonzero(block):
                        continue
                    virtual_charge_tuple = tuple(
                        virtual_charges[sector] for sector in key
                    )
                    virtual_offsets = tuple(
                        charge_groups[axis_charge].index(sector)
                        for axis_charge, sector in zip(
                            virtual_charge_tuple,
                            key,
                        )
                    )
                    for row_charge, (row_start, row_stop) in physical_offsets.items():
                        for column_charge, (column_start, column_stop) in physical_offsets.items():
                            source_block = block[row_start:row_stop, column_start:column_stop]
                            if not _backend_nonzero(source_block):
                                continue
                            # Quimb stores PEPO physical axes as (lower, upper),
                            # whereas active blocks use ordinary (row, column)
                            # matrix order.
                            physical_block = ar.do(
                                "transpose",
                                source_block,
                                (1, 0),
                            )
                            physical_row_charge = column_charge
                            physical_column_charge = row_charge
                            sector = (
                                *virtual_charge_tuple,
                                physical_row_charge,
                                physical_column_charge,
                            )
                            signed = tuple(
                                symmetry_obj.sign(
                                    sector_charge,
                                    dual,
                                )
                                for sector_charge, dual in zip(
                                    sector,
                                    virtual_duals + (False, True),
                                )
                            )
                            actual_charge = symmetry_obj.combine(*signed)
                            if actual_charge != charge:
                                raise ValueError(
                                    "Active PEPO block is not compatible with "
                                    f"{symmetry} charge {charge!r}: site={site}, "
                                    f"virtual={virtual_charge_tuple}, "
                                    f"physical=({physical_row_charge!r}, "
                                    f"{physical_column_charge!r}), "
                                    f"has charge {actual_charge!r}."
                                )
                            virtual_shape = tuple(
                                charge_sizes[axis_charge]
                                for axis_charge in virtual_charge_tuple
                            )
                            placed = ar.do(
                                "reshape",
                                physical_block,
                                (1,) * len(directions) + physical_block.shape,
                            )
                            for axis, (axis_size, axis_offset) in enumerate(
                                zip(virtual_shape, virtual_offsets)
                            ):
                                mask = np.zeros(axis_size, dtype=float)
                                mask[axis_offset] = 1.0
                                mask = _as_backend(mask, like=physical_block)
                                mask = ar.do(
                                    "reshape",
                                    mask,
                                    tuple(
                                        axis_size if index == axis else 1
                                        for index in range(len(directions) + 2)
                                    ),
                                )
                                placed = ar.do("multiply", placed, mask)
                            if sector in block_arrays:
                                block_arrays[sector] = ar.do(
                                    "add",
                                    block_arrays[sector],
                                    placed,
                                )
                            else:
                                block_arrays[sector] = placed

                duals = tuple(
                    sr.BlockIndex(
                        charge_sizes
                        if axis < len(directions)
                        else physical_sectors,
                        dual=dual,
                    )
                    for axis, dual in enumerate(virtual_duals + (False, True))
                )
                native = array_cls.from_blocks(
                    block_arrays,
                    duals=duals,
                    charge=charge,
                    symmetry=symmetry,
                )
                native_arrays[site] = native
                row.append(
                    np.zeros(
                        (active.bond_dim,) * len(directions)
                        + (active.physical_dim, active.physical_dim),
                        dtype=np.asarray(
                            ar.to_numpy(
                                next(iter(active.blocks[site].values()))
                            )
                        ).dtype,
                    )
                )
            arrays.append(row)

        pepo = qtn.PEPO(
            arrays,
            shape="urdlbk",
            cyclic=active.cyclic,
        )
        for site, native in native_arrays.items():
            pepo[site].modify(data=native)
        return pepo

    def to_pepo(self):
        """Materialize blocks as a dense Quimb ``PEPO``.

        This is an explicit interoperability boundary. The active-block
        representation is normally the smaller and clearer object to keep
        during autodiff or repeated coefficient evaluations.
        """
        arrays = []
        dtype = next(iter(next(iter(self.blocks.values())).values())).dtype
        for i in range(self.lx):
            row = []
            for j in range(self.ly):
                site = (i, j)
                row.append(
                    _materialize_site_blocks(
                        self.site_directions[site],
                        self.blocks[site],
                        self.bond_dim,
                        dtype,
                    )
                )
            arrays.append(row)
        return qtn.PEPO(arrays, shape="urdlbk", cyclic=self.cyclic)

    materialize = to_pepo


@dataclass(frozen=True)
class PauliPEPOTerm:
    """One translation-invariant Pauli slot in a square-lattice PEPO basis.

    ``support="onsite"`` contributes the same one-site Pauli operator to
    every lattice site. ``support="edge"`` contributes the same ordered
    two-site Pauli operator to every positive (``u`` and ``r``) lattice edge.
    The scalar ``coefficient`` may be a Python number, a Torch/JAX scalar, or
    a callable accepting the parameter container passed to
    :meth:`PauliPEPOBasis.exp`.
    """

    support: str
    paulis: object
    coefficient: object = 1.0

    def __post_init__(self):
        support = _normalize_pauli_support(self.support)
        labels = _normalize_paulis(self.paulis, support=support)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "paulis", labels)

    @classmethod
    def from_pauli(cls, support, paulis, *, coefficient=1.0):
        """Construct a term from ``"X"`` or ``"ZZ"`` labels."""
        return cls(support, paulis, coefficient)


class CompiledPEPOExp:
    """Cached callable for repeated fixed-topology PEPO exponentials.

    The callable owns no coefficient or autodiff values. It only points to a
    :class:`PauliPEPOBasis` whose lattice, Pauli channels, C4 orbits, and
    active-sector layout were compiled once. Each :meth:`exp` call returns
    fresh :class:`ActivePEPOBlocks` unless ``materialize=True`` is requested.
    """

    def __init__(self, basis):
        if not isinstance(basis, PauliPEPOBasis):
            raise TypeError("basis must be a PauliPEPOBasis.")
        self.basis = basis
        # Compile only value-independent cluster embeddings here. Matrix
        # exponentials and coefficient contractions still happen per call.
        basis._prepare_exp_plan()

    def exp(
        self,
        step,
        parameters=None,
        *,
        coefficients=None,
        materialize=False,
    ):
        """Evaluate ``exp(step * H)`` with fresh backend values.

        ``step=-1j * tau`` is real-time evolution. ``parameters`` resolves
        callable/parameterized slots; ``coefficients`` is a one-dimensional
        batch in the basis term order, and the two inputs are mutually
        exclusive.
        """
        return self.basis.exp(
            step,
            parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    evaluate = exp
    __call__ = exp


class PauliPEPOBasis:
    """Compiled fixed-channel Pauli basis for square-lattice evolution.

    This is the PEPO analogue of :class:`~pepsy.operators.mpo.MPOBasis`.
    The lattice, cluster shapes, Pauli channels, and active-sector topology
    are compiled once. Each evaluation only assembles local cluster
    exponentials and fills those fixed channels with the current coefficient
    and time-step values, so backend scalar graphs are not cached or copied.

    The supported Hamiltonian family is

    ``H(theta) = sum_i h(theta)_i + sum_<ij> v(theta)_{ij}``,

    where ``h`` and ``v`` are linear combinations of the supplied onsite and
    edge Pauli slots. The returned order-4 representation is normally kept
    as :class:`ActivePEPOBlocks`; dense Quimb materialization is intended for
    small validation lattices because a fixed Pauli channel basis is much
    larger than an SVD-compressed numerical basis.
    """

    def __init__(
        self,
        lx,
        ly,
        terms,
        *,
        order=4,
        cyclic=False,
        symmetry=None,
    ):
        self.lx = _validate_shape(lx, "lx")
        self.ly = _validate_shape(ly, "ly")
        self.cyclic = _validate_cyclic(cyclic, self.lx, self.ly)
        if not isinstance(order, Integral):
            raise TypeError("order must be an integer.")
        self.order = int(order)
        if self.order < 1 or self.order > 4:
            raise ValueError("PauliPEPOBasis currently supports orders 1 through 4.")
        if symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")
        self.symmetry = symmetry
        self._terms = tuple(_normalize_pauli_term(term) for term in terms)
        if not self._terms:
            raise ValueError("terms must contain at least one Pauli slot.")
        # Static one-hot maps let each evaluation fuse all coefficient slots
        # into onsite and edge Pauli components in two backend contractions.
        # They contain topology only, so they are safe to retain across
        # Torch/JAX autodiff calls.
        self._onsite_term_map = np.zeros((len(self._terms), 4), dtype=float)
        self._edge_term_map = np.zeros((len(self._terms), 16), dtype=float)
        for term_index, term in enumerate(self._terms):
            labels = tuple(_PAULI_LABELS.index(label) for label in term.paulis)
            if term.support == "onsite":
                self._onsite_term_map[term_index, labels[0]] = 1.0
            else:
                self._edge_term_map[term_index, labels[0] * 4 + labels[1]] = 1.0
        self._cluster_embedding_cache = {}
        self.site_directions = {
            (i, j): _site_directions(i, j, self.lx, self.ly, *self.cyclic)
            for i in range(self.lx)
            for j in range(self.ly)
        }
        self.plaquette_starts = _plaquette_starts(self.lx, self.ly, self.cyclic)
        self.pair_orbits = _pair_orbits() if symmetry == "C4" else tuple(
            (pair, (pair,)) for pair in _all_direction_pairs()
        )
        self.triple_orbits = _subset_orbits(3, symmetry)
        self.path_orbits = _path_orbits(symmetry)
        self._build_count = 0
        self._compiled_exp = None

    @classmethod
    def compile(cls, lx, ly, terms, **kwargs):
        """Compile fixed lattice and Pauli topology for repeated evaluations.

        This constructor does not evaluate an exponential and does not cache
        backend coefficient values. Use :meth:`exp` for a one-off call or
        :meth:`compile_exp` when the exponential policy should be reused.
        """
        return cls(lx, ly, terms, **kwargs)

    @property
    def terms(self):
        """Read-only translation-invariant Pauli slots."""
        return self._terms

    @property
    def num_terms(self):
        """Number of coefficient slots."""
        return len(self._terms)

    @property
    def cache_info(self):
        """Return topology-only compilation diagnostics."""
        return {
            "compiled": True,
            "builds": self._build_count,
            "terms": self.num_terms,
            "order": self.order,
            "pair_orbits": len(self.pair_orbits),
            "tree_orbits": len(self.triple_orbits) + len(self.path_orbits),
            "plaquettes": len(self.plaquette_starts),
            "cluster_embedding_plans": len(self._cluster_embedding_cache),
            "fused_pauli_slots": int(
                np.count_nonzero(self._onsite_term_map)
                + np.count_nonzero(self._edge_term_map)
            ),
            "cyclic": self.cyclic,
            "symmetry": self.symmetry,
            "compiled_exp": self._compiled_exp is not None,
        }

    def compile_exp(self):
        """Return the cached fixed-topology :class:`CompiledPEPOExp`.

        Only geometry and channel structure are cached. Coefficients and the
        exponential step are supplied afresh to every call so Torch/JAX
        autodiff graphs cannot become stale.
        """
        if self._compiled_exp is None:
            self._compiled_exp = CompiledPEPOExp(self)
        return self._compiled_exp

    def _prepare_exp_plan(self):
        """Precompute all small cluster embedding maps for this basis."""
        if self.order < 3:
            return self
        # Populate the process-wide physical basis cache before building the
        # per-basis cluster maps.
        _backend_pauli_basis(1)
        _backend_pauli_basis(2)
        representatives = []
        for representative, _orbit in self.pair_orbits:
            representatives.append(
                tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
            )
        if self.order >= 4:
            for representative, _orbit in self.triple_orbits:
                representatives.append(
                    tuple(
                        (0, index + 1, direction)
                        for index, direction in enumerate(representative)
                    )
                )
            for representative, _orbit in self.path_orbits:
                representatives.append(
                    tuple(
                        (index, index + 1, direction)
                        for index, direction in enumerate(representative)
                    )
                )
        for edges in representatives:
            nsites = max(
                max(source, target) for source, target, _direction in edges
            ) + 1
            self._cluster_embedding_plan(nsites, edges)
            if nsites == 4:
                for center, branches in _three_subtrees(edges):
                    three_edges = tuple(
                        (0, index + 1, direction)
                        for index, (_endpoint, direction) in enumerate(branches)
                    )
                    self._cluster_embedding_plan(3, three_edges)
        if self.order >= 4 and self.plaquette_starts:
            loop_edges = _plaquette_edges()
            self._cluster_embedding_plan(4, loop_edges)
            for _center, branches in _three_subtrees(loop_edges):
                three_edges = tuple(
                    (0, index + 1, direction)
                    for index, (_endpoint, direction) in enumerate(branches)
                )
                self._cluster_embedding_plan(3, three_edges)
        return self

    def _coefficient_values(self, parameters, coefficients):
        if coefficients is not None and parameters is not None:
            raise ValueError("parameters and coefficients are mutually exclusive.")
        if coefficients is None:
            values = []
            for term in self._terms:
                value = term.coefficient
                if hasattr(value, "resolve"):
                    value = value.resolve(parameters)
                elif callable(value):
                    if parameters is None:
                        raise KeyError(
                            "callable Pauli coefficients require parameters."
                        )
                    value = value(parameters)
                values.append(value)
        else:
            shape = getattr(coefficients, "shape", None)
            if shape is not None:
                shape = tuple(shape)
                if not shape:
                    if self.num_terms != 1:
                        raise ValueError(
                            "a scalar coefficient batch is valid only for one term."
                        )
                    values = [coefficients]
                elif len(shape) == 1:
                    if int(shape[0]) != self.num_terms:
                        raise ValueError(
                            f"coefficients must have length {self.num_terms}, "
                            f"got {shape[0]}."
                        )
                    values = [coefficients[index] for index in range(self.num_terms)]
                else:
                    raise ValueError("coefficients must be one-dimensional.")
            else:
                try:
                    values = list(coefficients)
                except TypeError as exc:
                    raise TypeError("coefficients must be one-dimensional.") from exc
                if len(values) != self.num_terms:
                    raise ValueError(
                        f"coefficients must have length {self.num_terms}, "
                        f"got {len(values)}."
                    )
        for index, value in enumerate(values):
            ndim = getattr(value, "ndim", None)
            if ndim is None:
                ndim = np.ndim(value)
            if ndim != 0:
                raise TypeError(f"coefficient[{index}] must be scalar.")
        reference = _backend_reference(values)
        return tuple(_as_backend(value, like=reference) for value in values)

    def coefficients(self, parameters=None):
        """Evaluate the coefficient slots as one backend-native vector."""
        values = self._coefficient_values(parameters, None)
        return _backend_stack(values)

    def _hamiltonian_components(self, values, beta):
        """Fuse coefficient slots into onsite and edge Pauli components."""
        reference = _backend_reference((*values, beta))
        coefficient_batch = _backend_stack(values)
        onsite_map = _as_backend(self._onsite_term_map, like=reference)
        edge_map = _as_backend(self._edge_term_map, like=reference)
        return (
            ar.do(
                "tensordot",
                coefficient_batch,
                onsite_map,
                axes=([0], [0]),
            ),
            ar.do(
                "tensordot",
                coefficient_batch,
                edge_map,
                axes=([0], [0]),
            ),
        )

    @staticmethod
    def _components_to_operators(onsite_components, edge_components):
        """Convert Pauli components into local Hamiltonian matrices."""
        onsite_components = _complexify_backend(onsite_components)
        edge_components = _complexify_backend(edge_components)
        onsite_basis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=onsite_components),
            axis=0,
        )
        edge_basis = ar.do(
            "stack",
            _backend_pauli_basis(2, like=edge_components),
            axis=0,
        )
        return (
            ar.do("tensordot", onsite_components, onsite_basis, axes=([0], [0])),
            ar.do("tensordot", edge_components, edge_basis, axes=([0], [0])),
        )

    def _hamiltonian(self, values, beta):
        """Return local Hamiltonian matrices after fused slot assembly."""
        components = self._hamiltonian_components(values, beta)
        return self._components_to_operators(*components)

    @staticmethod
    def _oriented_edge(operator, direction):
        return (
            operator
            if direction in _POSITIVE_DIRECTIONS
            else _backend_swap_two_site(operator, 2)
        )

    def _cluster_embedding_plan(self, nsites, edges):
        """Cache linear maps from local Pauli components to a cluster matrix."""
        key = (nsites, tuple(edges))
        try:
            return self._cluster_embedding_cache[key]
        except KeyError:
            pass
        dimension = 2**nsites
        onsite_basis = np.stack(
            [
                sum(
                    (
                        np.asarray(
                            _backend_embed_operator(
                                matrix,
                                (site,),
                                nsites,
                                2,
                            )
                        )
                        for site in range(nsites)
                    ),
                    start=np.zeros((dimension, dimension), dtype=complex),
                )
                for matrix in _PAULI_BASIS_CACHE[1]
            ],
            axis=0,
        )
        edge_basis = np.zeros((16, dimension, dimension), dtype=complex)
        for component, matrix in enumerate(_PAULI_BASIS_CACHE[2]):
            for source, target, direction in edges:
                oriented = (
                    matrix
                    if direction in _POSITIVE_DIRECTIONS
                    else _swap_two_site_operator(matrix, 2)
                )
                edge_basis[component] += np.asarray(
                    _backend_embed_operator(
                        oriented,
                        (source, target),
                        nsites,
                        2,
                    )
                )
        plan = {"onsite": onsite_basis, "edge": edge_basis}
        self._cluster_embedding_cache[key] = plan
        return plan

    def _cluster_hamiltonian(
        self,
        nsites,
        edges,
        onsite_components,
        edge_components,
        *,
        like,
    ):
        """Assemble a cluster Hamiltonian from cached component embeddings."""
        onsite_components = _complexify_backend(onsite_components)
        edge_components = _complexify_backend(edge_components)
        plan = self._cluster_embedding_plan(nsites, edges)
        onsite_basis = _as_backend(plan["onsite"], like=like)
        edge_basis = _as_backend(plan["edge"], like=like)
        onsite_part = ar.do(
            "tensordot",
            onsite_components,
            onsite_basis,
            axes=([0], [0]),
        )
        edge_part = ar.do(
            "tensordot",
            edge_components,
            edge_basis,
            axes=([0], [0]),
        )
        return ar.do("add", onsite_part, edge_part)

    @staticmethod
    def _embed_with_background(operator, positions, nsites, background):
        """Embed a connected correction and dress untouched sites by ``E1``."""
        result = _backend_embed_operator(operator, positions, nsites, 2)
        positions = set(positions)
        for site in range(nsites):
            if site in positions:
                continue
            result = ar.do(
                "matmul",
                result,
                _backend_embed_operator(background, (site,), nsites, 2),
            )
        return result

    def _connected_residual(
        self,
        nsites,
        edges,
        onsite_components,
        edge_components,
        beta,
        one_exp,
        edge_residual,
    ):
        """Evaluate one connected residual using cached cluster embeddings."""
        reference = _backend_reference(
            (beta, one_exp, edge_residual, onsite_components, edge_components)
        )
        exact = _backend_expm(
            ar.do(
                "multiply",
                -beta,
                self._cluster_hamiltonian(
                    nsites,
                    edges,
                    onsite_components,
                    edge_components,
                    like=reference,
                ),
            )
        )
        residual = ar.do(
            "subtract",
            exact,
            _backend_operator_product([one_exp] * nsites),
        )
        for source, target, direction in edges:
            lower = self._oriented_edge(edge_residual, direction)
            residual = ar.do(
                "subtract",
                residual,
                self._embed_with_background(
                    lower,
                    (source, target),
                    nsites,
                    one_exp,
                ),
            )
        if nsites == 4:
            for first_index, first in enumerate(edges):
                first_sites = {first[0], first[1]}
                for second in edges[first_index + 1 :]:
                    if first_sites & {second[0], second[1]}:
                        continue
                    first_lower = _backend_embed_operator(
                        self._oriented_edge(edge_residual, first[2]),
                        (first[0], first[1]),
                        nsites,
                        2,
                    )
                    second_lower = _backend_embed_operator(
                        self._oriented_edge(edge_residual, second[2]),
                        (second[0], second[1]),
                        nsites,
                        2,
                    )
                    residual = ar.do(
                        "subtract",
                        residual,
                        ar.do("matmul", first_lower, second_lower),
                    )
            for center, branches in _three_subtrees(edges):
                endpoints = tuple(endpoint for endpoint, _direction in branches)
                three_edges = tuple(
                    (0, index + 1, direction)
                    for index, (_endpoint, direction) in enumerate(branches)
                )
                lower = self._connected_residual(
                    3,
                    three_edges,
                    onsite_components,
                    edge_components,
                    beta,
                    one_exp,
                    edge_residual,
                )
                residual = ar.do(
                    "subtract",
                    residual,
                    self._embed_with_background(
                        lower,
                        (center, *endpoints),
                        nsites,
                        one_exp,
                    ),
                )
        return residual

    @staticmethod
    def _center_tensor(coefficients):
        """Convert ``(physical, active...)`` Pauli coefficients to blocks."""
        basis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=coefficients),
            axis=0,
        )
        # One contraction replaces one Python/backend operation per active
        # sector. The result already has ``active_shape + (2, 2)`` layout.
        return ar.do("tensordot", coefficients, basis, axes=([0], [0]))

    @staticmethod
    def _path_tensors(coefficients):
        """Build a fixed-rank two-block factorization of a four-site path."""
        paulis = ar.do(
            "stack",
            _backend_pauli_basis(1, like=coefficients),
            axis=0,
        )
        coefficient_view = ar.do(
            "reshape",
            coefficients,
            (4, 4, 4, 4, 1, 1),
        )
        p1_view = ar.do("reshape", paulis, (1, 4, 1, 1, 2, 2))
        left = ar.do("multiply", coefficient_view, p1_view)
        left = ar.do("reshape", left, (4, 64, 2, 2))

        # The right factor is a fixed selector tensor: its last virtual index
        # must equal the final physical Pauli label. Constructing it by
        # broadcasting avoids the old 256-entry Python loop on every call.
        selector = _as_backend(np.eye(4), like=coefficients)
        selector = ar.do("reshape", selector, (1, 1, 4, 4, 1, 1))
        p2_view = ar.do("reshape", paulis, (1, 4, 1, 1, 2, 2))
        right = ar.do("multiply", p2_view, selector)
        right = ar.do(
            "multiply",
            right,
            _as_backend(np.ones((4, 1, 1, 1, 1, 1)), like=coefficients),
        )
        right = ar.do("reshape", right, (64, 4, 2, 2))
        return (
            left,
            right,
        )

    @staticmethod
    def _loop_tensors(coefficients):
        """Build fixed-rank corner tensors for a four-site plaquette loop.

        The physical coefficient tensor is ordered around the cycle as
        ``(lower-left, upper-left, upper-right, lower-right)``. Every loop
        bond carries a fixed 16-state pair history. The corner tensors pass
        that history around the cycle, so this is an exact tensor-ring
        factorization with no coefficient-dependent SVD.
        """
        loop_coefficients = ar.do("transpose", coefficients, (0, 1, 3, 2))
        loop_coefficients = ar.do("reshape", loop_coefficients, (4, 4, 16))
        pair_labels = np.arange(16)
        # Keep these labels as static NumPy indices.  The resulting Pauli
        # banks are constants, while ``loop_coefficients`` remains a backend
        # array and therefore stays on the autodiff graph.
        pauli_values = np.stack(_backend_pauli_basis(1), axis=0)
        first_paulis = _as_backend(
            pauli_values[pair_labels // 4],
            like=coefficients,
        )
        second_paulis = _as_backend(
            pauli_values[pair_labels % 4],
            like=coefficients,
        )
        diagonal = _as_backend(np.eye(16), like=coefficients)
        ones = _as_backend(np.ones((1, 16, 1, 1)), like=coefficients)
        first_corner = ar.do(
            "multiply",
            ar.do("reshape", first_paulis, (1, 16, 2, 2)),
            ar.do("reshape", diagonal, (16, 16, 1, 1)),
        )
        second_corner = ar.do(
            "multiply",
            ar.do("reshape", loop_coefficients, (16, 16, 1, 1)),
            ar.do("reshape", second_paulis, (16, 1, 2, 2)),
        )
        third_corner = ar.do(
            "multiply",
            ar.do("reshape", first_paulis, (1, 16, 2, 2)),
            ar.do("reshape", diagonal, (16, 16, 1, 1)),
        )
        fourth_corner = ar.do(
            "multiply",
            ar.do("reshape", second_paulis, (16, 1, 2, 2)),
            ones,
        )
        return first_corner, second_corner, third_corner, fourth_corner

    def _build_active(self, beta, values):
        onsite_components, edge_components = self._hamiltonian_components(
            values,
            beta,
        )
        onsite, edge = self._components_to_operators(
            onsite_components,
            edge_components,
        )
        reference = _backend_reference((beta, onsite, edge))
        beta = _as_backend(beta, like=reference)
        onsite = _as_backend(onsite, like=reference)
        edge = _as_backend(edge, like=reference)
        one_exp = _backend_expm(ar.do("multiply", -beta, onsite))
        edge_exact = _backend_expm(
            ar.do(
                "multiply",
                -beta,
                ar.do(
                    "add",
                    edge,
                    ar.do("add", _backend_embed_operator(onsite, (0,), 2, 2),
                          _backend_embed_operator(onsite, (1,), 2, 2)),
                ),
            )
        )
        edge_residual = ar.do(
            "subtract",
            edge_exact,
            _backend_operator_product([one_exp, one_exp]),
        )
        paulis = _backend_pauli_basis(1, like=reference)
        blocks = _initialize_blocks(self.lx, self.ly, one_exp, self.site_directions)
        allocator = _SectorAllocator()

        if self.order >= 2:
            edge_coefficients = _backend_pauli_expand(edge_residual, 2)
            channels = tuple(product(range(4), repeat=2))
            source = _backend_stack([paulis[first] for first, _ in channels])
            target = _backend_stack(
                [
                    ar.do("multiply", edge_coefficients[first, second], paulis[second])
                    for first, second in channels
                ]
            )
            sectors = allocator.allocate(len(channels))
            _add_positive_edge_channels(
                blocks,
                self.site_directions,
                source,
                target,
                sectors,
            )

        if self.order >= 3:
            for representative, orbit in self.pair_orbits:
                if not any(
                    all(direction in directions for direction in pair)
                    for pair in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                edges = tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    3,
                    edges,
                    onsite_components,
                    edge_components,
                    beta,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 3)
                sectors = (
                    allocator.allocate(4),
                    allocator.allocate(4),
                )
                center = self._center_tensor(coefficients)
                for pair in orbit:
                    pair_center = (
                        center
                        if pair == representative
                        else _rotate_direction_tensor(representative, pair, center)
                    )
                    for axis, direction in enumerate(pair):
                        _add_single_direction_blocks(
                            blocks,
                            self.site_directions,
                            self.lx,
                            self.ly,
                            _OPPOSITE_DIRECTION[direction],
                            sectors[axis],
                            paulis,
                            source=False,
                            cyclic=self.cyclic,
                        )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        pair,
                        pair_center,
                        sectors,
                    )

        if self.order >= 4:
            for representative, orbit in self.triple_orbits:
                if not any(
                    all(direction in directions for direction in star)
                    for star in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                edges = tuple(
                    (0, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    4,
                    edges,
                    onsite_components,
                    edge_components,
                    beta,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 4)
                sectors = tuple(allocator.allocate(4) for _ in range(3))
                center = self._center_tensor(coefficients)
                for star in orbit:
                    star_center = (
                        center
                        if star == representative
                        else _rotate_direction_tensor(representative, star, center)
                    )
                    for axis, direction in enumerate(star):
                        _add_single_direction_blocks(
                            blocks,
                            self.site_directions,
                            self.lx,
                            self.ly,
                            _OPPOSITE_DIRECTION[direction],
                            sectors[axis],
                            paulis,
                            source=False,
                            cyclic=self.cyclic,
                        )
                    _add_triple_blocks(
                        blocks,
                        self.site_directions,
                        star,
                        star_center,
                        sectors[0],
                    )

            for representative, orbit in self.path_orbits:
                if not any(
                    _path_start_sites(steps, self.lx, self.ly, self.cyclic)
                    for steps in orbit
                ):
                    continue
                edges = tuple(
                    (index, index + 1, direction)
                    for index, direction in enumerate(representative)
                )
                residual = self._connected_residual(
                    4,
                    edges,
                    onsite_components,
                    edge_components,
                    beta,
                    one_exp,
                    edge_residual,
                )
                coefficients = _backend_pauli_expand(residual, 4)
                left, right = self._path_tensors(coefficients)
                for steps in orbit:
                    first_sectors = allocator.allocate(4)
                    middle_sectors = allocator.allocate(64)
                    last_sectors = allocator.allocate(4)
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        steps[0],
                        first_sectors,
                        paulis,
                        source=True,
                        cyclic=self.cyclic,
                    )
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        _OPPOSITE_DIRECTION[steps[2]],
                        last_sectors,
                        paulis,
                        source=False,
                        cyclic=self.cyclic,
                    )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        (_OPPOSITE_DIRECTION[steps[0]], steps[1]),
                        left,
                        (first_sectors, middle_sectors),
                    )
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        (_OPPOSITE_DIRECTION[steps[1]], steps[2]),
                        right,
                        (middle_sectors, last_sectors),
                    )

        if self.order >= 4 and self.plaquette_starts:
            loop_edges = _plaquette_edges()
            loop_residual = self._connected_residual(
                4,
                loop_edges,
                onsite_components,
                edge_components,
                beta,
                one_exp,
                edge_residual,
            )
            loop_coefficients = _backend_pauli_expand(loop_residual, 4)
            loop_tensors = self._loop_tensors(loop_coefficients)
            for start in self.plaquette_starts:
                upper = _site_after(start, "u", self.lx, self.ly, self.cyclic)
                right = _site_after(start, "r", self.lx, self.ly, self.cyclic)
                diagonal = _site_after(upper, "r", self.lx, self.ly, self.cyclic)
                loop_sites = (start, upper, diagonal, right)
                lower_bond = allocator.allocate(16)
                right_bond = allocator.allocate(16)
                upper_bond = allocator.allocate(16)
                left_bond = allocator.allocate(16)
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[0],
                    ("r", "u"),
                    loop_tensors[0],
                    (left_bond, lower_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[1],
                    ("d", "r"),
                    loop_tensors[1],
                    (lower_bond, right_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[2],
                    ("l", "d"),
                    loop_tensors[2],
                    (right_bond, upper_bond),
                )
                _add_pair_block_at_site(
                    blocks,
                    self.site_directions,
                    loop_sites[3],
                    ("u", "l"),
                    loop_tensors[3],
                    (upper_bond, left_bond),
                )

        self._build_count += 1
        return ActivePEPOBlocks(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=2,
            site_directions=self.site_directions,
            blocks=blocks,
        )

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        tau=None,
        beta=None,
        materialize=False,
    ):
        """Evaluate ``exp(step * H(coefficients))`` as fixed-channel blocks.

        Parameters
        ----------
        step : scalar, optional
            Actual scalar in the exponential. Real time is
            ``step=-1j * tau``; imaginary time can use ``step=-beta``.
        parameters : mapping or sequence, optional
            Values used by parameterized/callable Pauli slots.
        coefficients : one-dimensional array-like, optional
            Values in ``basis.terms`` order. Mutually exclusive with
            ``parameters``.
        tau, beta : scalar, optional
            Compatibility shorthands. ``tau`` maps to ``-1j * tau`` and
            ``beta`` maps to ``-beta``.
        materialize : bool, optional
            If false (default), return sparse :class:`ActivePEPOBlocks`; if
            true, return a dense Quimb ``PEPO``.
        """
        if step is not None and (tau is not None or beta is not None):
            raise ValueError("step cannot be combined with tau or beta.")
        if tau is not None and beta is not None:
            raise ValueError("tau and beta are mutually exclusive.")
        if step is None:
            if tau is not None:
                step = -1j * tau
            elif beta is not None:
                step = -beta
            else:
                raise TypeError("exp requires step, tau, or beta.")
        values = self._coefficient_values(parameters, coefficients)
        active = self._build_active(-step, values)
        return active.to_pepo() if materialize else active

    def evaluate(
        self,
        tau=None,
        parameters=None,
        *,
        coefficients=None,
        step=None,
        beta=None,
        materialize=False,
    ):
        """Compatibility wrapper for the former ``evaluate(tau=...)`` API.

        New code should use :meth:`exp` so the scalar convention is explicit.
        """
        if step is not None or beta is not None:
            return self.exp(
                step,
                parameters,
                coefficients=coefficients,
                beta=beta,
                materialize=materialize,
            )
        return self.exp(
            tau=tau,
            parameters=parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    def time_evolution(
        self,
        tau,
        parameters=None,
        *,
        coefficients=None,
        materialize=False,
    ):
        """Compatibility alias for ``exp(step=-1j * tau)``."""
        return self.exp(
            -1j * tau,
            parameters,
            coefficients=coefficients,
            materialize=materialize,
        )

    def build(self, parameters=None, *, coefficients=None, tau=None, beta=None, materialize=False):
        """Compatibility alias for :meth:`evaluate`."""
        return self.evaluate(
            tau=tau,
            parameters=parameters,
            coefficients=coefficients,
            beta=beta,
            materialize=materialize,
        )


class _SectorAllocator:
    """Allocate disjoint active-sector ranges for independent tree terms."""

    def __init__(self):
        self.next_sector = 1

    def allocate(self, rank):
        sectors = tuple(range(self.next_sector, self.next_sector + rank))
        self.next_sector += rank
        return sectors


def _initialize_blocks(lx, ly, one_site_exp, site_directions):
    return {
        (i, j): {(0,) * len(site_directions[(i, j)]): one_site_exp}
        for i in range(lx)
        for j in range(ly)
    }


def _add_block(blocks, site_directions, site, sector_by_direction, tensor):
    directions = site_directions[site]
    if not all(direction in directions for direction in sector_by_direction):
        return
    key = [0] * len(directions)
    for direction, sector in sector_by_direction.items():
        key[directions.index(direction)] = sector
    key = tuple(key)
    if key in blocks[site]:
        blocks[site][key] = blocks[site][key] + tensor
    else:
        blocks[site][key] = tensor


def _add_single_direction_blocks(
    blocks,
    site_directions,
    lx,
    ly,
    direction,
    sectors,
    factors,
    *,
    source,
    cyclic,
):
    """Add one endpoint role on every translated edge endpoint."""
    del lx, ly, cyclic  # geometry is already encoded by ``site_directions``
    factor_array = factors
    for site, directions in site_directions.items():
        try:
            direction_axis = directions.index(direction)
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(directions)
        for channel, sector in enumerate(sectors):
            key = key_template.copy()
            key[direction_axis] = sector
            key = tuple(key)
            tensor = factor_array[channel]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + tensor
            else:
                site_blocks[key] = tensor


def _add_positive_edge_channels(
    blocks,
    site_directions,
    start_factors,
    end_factors,
    sectors,
):
    """Add compact order-two endpoint channels for positive lattice edges."""
    for direction in _POSITIVE_DIRECTIONS:
        opposite = _OPPOSITE_DIRECTION[direction]
        for site, directions in site_directions.items():
            site_blocks = blocks[site]
            try:
                direction_axis = directions.index(direction)
            except ValueError:
                direction_axis = None
            if direction_axis is not None:
                key_template = [0] * len(directions)
                for channel, sector in enumerate(sectors):
                    key = key_template.copy()
                    key[direction_axis] = sector
                    key = tuple(key)
                    tensor = start_factors[channel]
                    if key in site_blocks:
                        site_blocks[key] = site_blocks[key] + tensor
                    else:
                        site_blocks[key] = tensor
            try:
                opposite_axis = directions.index(opposite)
            except ValueError:
                opposite_axis = None
            if opposite_axis is not None:
                key_template = [0] * len(directions)
                for channel, sector in enumerate(sectors):
                    key = key_template.copy()
                    key[opposite_axis] = sector
                    key = tuple(key)
                    tensor = end_factors[channel]
                    if key in site_blocks:
                        site_blocks[key] = site_blocks[key] + tensor
                    else:
                        site_blocks[key] = tensor


def _add_pair_blocks(
    blocks,
    site_directions,
    pair,
    tensor,
    sectors,
    *,
    transpose_opposite=False,
):
    """Add a pair-active center block, preserving direction-axis order."""
    if transpose_opposite and set(pair) in ({"u", "d"}, {"r", "l"}):
        tensor = tensor.transpose(1, 0, 2, 3)
    if len(sectors) == 2 and all(isinstance(axis, tuple) for axis in sectors):
        sector_axes = sectors
    else:
        sector_axes = (sectors, sectors)
    expected_shape = (len(sector_axes[0]), len(sector_axes[1]))
    if tensor.shape[:2] == expected_shape[::-1] and expected_shape[0] != expected_shape[1]:
        # C4 rotation can change the sorted direction order. Normalize the
        # tensor axes to the semantic sector order before inserting blocks.
        tensor = tensor.transpose(1, 0, 2, 3)
    elif tensor.shape[:2] != expected_shape:
        raise ValueError(
            "pair tensor active axes do not match sector dimensions: "
            f"tensor={tensor.shape[:2]}, sectors={expected_shape}."
        )
    for site, directions in site_directions.items():
        try:
            first_axis = directions.index(pair[0])
            second_axis = directions.index(pair[1])
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(directions)
        for first, second in np.ndindex(tensor.shape[:2]):
            key = key_template.copy()
            key[first_axis] = sector_axes[0][first]
            key[second_axis] = sector_axes[1][second]
            key = tuple(key)
            block = tensor[first, second]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + block
            else:
                site_blocks[key] = block


def _add_triple_blocks(blocks, site_directions, directions, tensor, sectors):
    """Add a three-active-leg center block for every translated star."""
    if len(sectors) == 3 and all(isinstance(axis, tuple) for axis in sectors):
        sector_axes = sectors
    else:
        sector_axes = (sectors, sectors, sectors)
    for site, present in site_directions.items():
        try:
            axes = tuple(present.index(direction) for direction in directions)
        except ValueError:
            continue
        site_blocks = blocks[site]
        key_template = [0] * len(present)
        for first, second, third in np.ndindex(tensor.shape[:3]):
            key = key_template.copy()
            key[axes[0]] = sector_axes[0][first]
            key[axes[1]] = sector_axes[1][second]
            key[axes[2]] = sector_axes[2][third]
            key = tuple(key)
            block = tensor[first, second, third]
            if key in site_blocks:
                site_blocks[key] = site_blocks[key] + block
            else:
                site_blocks[key] = block


def _add_pair_block_at_site(blocks, site_directions, site, pair, tensor, sectors):
    """Add one pair-active block without translating it over the lattice."""
    directions = site_directions[site]
    try:
        first_axis = directions.index(pair[0])
        second_axis = directions.index(pair[1])
    except ValueError:
        return
    sector_axes = tuple(sectors)
    expected_shape = (len(sector_axes[0]), len(sector_axes[1]))
    if tensor.shape[:2] != expected_shape:
        raise ValueError(
            "loop tensor active axes do not match sector dimensions: "
            f"tensor={tensor.shape[:2]}, sectors={expected_shape}."
        )
    site_blocks = blocks[site]
    key_template = [0] * len(directions)
    for first, second in np.ndindex(tensor.shape[:2]):
        key = key_template.copy()
        key[first_axis] = sector_axes[0][first]
        key[second_axis] = sector_axes[1][second]
        key = tuple(key)
        block = tensor[first, second]
        if key in site_blocks:
            site_blocks[key] = site_blocks[key] + block
        else:
            site_blocks[key] = block


def _site_after(site, direction, lx, ly, cyclic):
    i, j = site
    di, dj = _DIRECTION_VECTORS[direction]
    ni, nj = i + di, j + dj
    if cyclic[0]:
        ni %= lx
    if cyclic[1]:
        nj %= ly
    if not (0 <= ni < lx and 0 <= nj < ly):
        return None
    return ni, nj


def _plaquette_starts(lx, ly, cyclic):
    """Return lower-left sites whose four-edge plaquette is present."""
    starts = []
    for i in range(lx):
        for j in range(ly):
            first = (i, j)
            upper = _site_after(first, "u", lx, ly, cyclic)
            right = _site_after(first, "r", lx, ly, cyclic)
            if upper is None or right is None:
                continue
            diagonal = _site_after(upper, "r", lx, ly, cyclic)
            if diagonal is None or _site_after(right, "u", lx, ly, cyclic) != diagonal:
                continue
            starts.append(first)
    return tuple(starts)


def _path_start_sites(steps, lx, ly, cyclic):
    """Return starts whose directed path is present on the finite lattice."""
    starts = []
    for i in range(lx):
        for j in range(ly):
            site = (i, j)
            for direction in steps:
                site = _site_after(site, direction, lx, ly, cyclic)
                if site is None:
                    break
            else:
                starts.append((i, j))
    return tuple(starts)


def _valid_path_steps():
    """Generate non-self-closing four-site square-lattice tree paths."""
    paths = set()
    for first in _DIRECTIONS:
        for second in _DIRECTIONS:
            if second == _OPPOSITE_DIRECTION[first]:
                continue
            for third in _DIRECTIONS:
                if third == _OPPOSITE_DIRECTION[second]:
                    continue
                coordinates = [(0, 0)]
                for direction in (first, second, third):
                    previous = coordinates[-1]
                    vector = _DIRECTION_VECTORS[direction]
                    coordinates.append(
                        (previous[0] + vector[0], previous[1] + vector[1])
                    )
                if any(
                    abs(coordinates[left][0] - coordinates[right][0])
                    + abs(coordinates[left][1] - coordinates[right][1])
                    == 1
                    for left in range(4)
                    for right in range(left + 2, 4)
                ):
                    continue
                reverse = tuple(
                    _OPPOSITE_DIRECTION[direction]
                    for direction in reversed((first, second, third))
                )
                paths.add(min((first, second, third), reverse))
    return tuple(sorted(paths))


def _rotate_steps(steps):
    return tuple(_C4_ROTATION[direction] for direction in steps)


def _path_orbits(symmetry):
    remaining = set(_valid_path_steps())
    groups = []
    while remaining:
        representative = min(remaining)
        if symmetry != "C4":
            remaining.remove(representative)
            groups.append((representative, (representative,)))
            continue
        orbit = []
        steps = representative
        for _ in range(4):
            # Only group paths connected by an actual C4 rotation.  The
            # reversed path is physically equivalent, but its PEPO factors
            # require swapping endpoint tensors as well as rotating axes;
            # treating it as a plain axis rotation can cycle forever in
            # ``_rotate_direction_tensor`` and is not a valid factor map.
            if steps in remaining:
                orbit.append(steps)
                remaining.discard(steps)
            steps = _rotate_steps(steps)
        groups.append((representative, tuple(orbit)))
    return tuple(groups)


def _all_direction_pairs():
    return tuple(
        (first, second)
        for first_index, first in enumerate(_DIRECTIONS)
        for second in _DIRECTIONS[first_index + 1 :]
    )


def _rotate_pair(pair):
    rotated = tuple(_C4_ROTATION[direction] for direction in pair)
    return tuple(sorted(rotated, key=_DIRECTIONS.index))


def _pair_orbits():
    remaining = set(_all_direction_pairs())
    orbits = []
    while remaining:
        representative = min(remaining, key=lambda pair: tuple(_DIRECTIONS.index(x) for x in pair))
        orbit = []
        pair = representative
        while pair not in orbit:
            orbit.append(pair)
            remaining.discard(pair)
            pair = _rotate_pair(pair)
        orbits.append((representative, tuple(orbit)))
    return tuple(orbits)


def _rotate_pair_tensor(pair, target, tensor):
    """Rotate active axes while preserving the canonical direction order."""
    return _rotate_direction_tensor(pair, target, tensor)


def _rotate_direction_tensor(directions, target, tensor):
    """Rotate a tensor whose leading axes follow ``directions``."""
    current = directions
    rotated = tensor
    while current != target:
        raw = tuple(_C4_ROTATION[direction] for direction in current)
        order = tuple(sorted(range(len(raw)), key=lambda index: _DIRECTIONS.index(raw[index])))
        next_directions = tuple(raw[index] for index in order)
        rotated = ar.do(
            "transpose",
            rotated,
            order + tuple(range(len(directions), rotated.ndim)),
        )
        current = next_directions
    return rotated


def _direction_subsets(size):
    return tuple(
        tuple(
            direction
            for index, direction in enumerate(_DIRECTIONS)
            if mask & (1 << index)
        )
        for mask in range(1 << len(_DIRECTIONS))
        if mask.bit_count() == size
    )


def _subset_orbits(size, symmetry):
    subsets = set(_direction_subsets(size))
    if symmetry != "C4":
        return tuple((subset, (subset,)) for subset in subsets)
    orbits = []
    while subsets:
        representative = min(subsets, key=lambda subset: tuple(_DIRECTIONS.index(x) for x in subset))
        orbit = []
        current = representative
        while current not in orbit:
            orbit.append(current)
            subsets.discard(current)
            current = tuple(sorted((_C4_ROTATION[x] for x in current), key=_DIRECTIONS.index))
        orbits.append((representative, tuple(orbit)))
    return tuple(orbits)


def _solve_four_star(
    directions,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
    beta,
):
    """Solve a four-site T-shaped tree residual at its degree-three center."""
    rank = start_factors.shape[0]
    local_dim = onesite_op.shape[0]
    # Use the labeled order ``(endpoint-0, center, endpoint-1, endpoint-2)``
    # so the solved center block is aligned with its PEPO embedding.
    edges = (
        (1, 0, directions[0]),
        (1, 2, directions[1]),
        (1, 3, directions[2]),
    )
    _, residual = _lower_tree_residual(
        4,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
        pair_tensors,
    )
    endpoint_factors = [
        end_factors if direction in _POSITIVE_DIRECTIONS else start_factors
        for direction in directions
    ]
    endpoint_matrices = tuple(
        factors.reshape(rank, local_dim**2).T for factors in endpoint_factors
    )
    endpoint_pseudoinverses = tuple(
        np.linalg.pinv(factors) for factors in endpoint_matrices
    )
    residual_tensor = _operator_tensor(residual, 4, local_dim)
    center_coefficients = np.einsum(
        "ijkl,ai,bk,cl->jabc",
        residual_tensor,
        endpoint_pseudoinverses[0],
        endpoint_pseudoinverses[1],
        endpoint_pseudoinverses[2],
    )
    solution = center_coefficients.transpose(1, 2, 3, 0).reshape(
        rank, rank, rank, local_dim, local_dim
    )
    fitted = np.einsum(
        "ia,jabc,kb,lc->ijkl",
        endpoint_matrices[0],
        center_coefficients,
        endpoint_matrices[1],
        endpoint_matrices[2],
    )
    return (
        solution,
        np.linalg.norm(residual_tensor - fitted),
        np.linalg.norm(residual_tensor),
    )


def _solve_four_path(
    steps,
    twosite_op,
    onesite_op,
    one_site_exp,
    start_factors,
    end_factors,
    pair_tensors,
    beta,
    max_tree_rank,
):
    """Solve a four-site path using an endpoint projection and internal SVD."""
    rank = start_factors.shape[0]
    local_dim = onesite_op.shape[0]
    first_factors = (
        start_factors if steps[0] in _POSITIVE_DIRECTIONS else end_factors
    )
    last_factors = (
        end_factors if steps[2] in _POSITIVE_DIRECTIONS else start_factors
    )
    edges = tuple((index, index + 1, direction) for index, direction in enumerate(steps))
    _, residual = _lower_tree_residual(
        4,
        edges,
        twosite_op,
        onesite_op,
        beta,
        one_site_exp,
        start_factors,
        end_factors,
        pair_tensors,
    )

    # Project only the two endpoint physical spaces.  The remaining core is
    # a small matrix whose SVD supplies the two internal PEPO factors.
    first_matrix = first_factors.reshape(rank, local_dim**2).T
    last_matrix = last_factors.reshape(rank, local_dim**2).T
    first_pseudoinverse = np.linalg.pinv(first_matrix)
    last_pseudoinverse = np.linalg.pinv(last_matrix)
    residual_tensor = _operator_tensor(residual, 4, local_dim)
    projected = np.einsum(
        "ijkl,ai,dl->ajkd",
        residual_tensor,
        first_pseudoinverse,
        last_pseudoinverse,
    )
    # Arrange the right matrix as ``(last-site, operator-basis)``. This is
    # the coefficient ordering compatible with a Kronecker product of the
    # two local physical matrices after the SVD factorization.
    core = projected.transpose(0, 1, 3, 2).reshape(
        rank * local_dim * local_dim, local_dim * local_dim * rank
    )
    left, singular_values, right = np.linalg.svd(core, full_matrices=False)
    if not singular_values.size or singular_values[0] == 0.0:
        empty_left = np.zeros((rank, 0, local_dim, local_dim), dtype=onesite_op.dtype)
        empty_right = np.zeros((0, rank, local_dim, local_dim), dtype=onesite_op.dtype)
        return empty_left, empty_right, np.linalg.norm(residual.reshape(-1)), np.linalg.norm(residual.reshape(-1))
    threshold = singular_values[0] * np.finfo(singular_values.dtype).eps * max(1, core.shape[0])
    keep = np.flatnonzero(singular_values > threshold)
    if max_tree_rank is not None:
        keep = keep[:max_tree_rank]
    if not keep.size:
        keep = np.array([0])
    singular_values = singular_values[keep]
    root = np.sqrt(singular_values)
    left_factors = (left[:, keep] * root).reshape(
        rank, local_dim, local_dim, len(keep)
    ).transpose(0, 3, 1, 2)
    right_factors = (root[:, None] * right[keep, :]).reshape(
        len(keep), rank, local_dim, local_dim
    )
    reconstructed = np.einsum(
        "ia,acj,cdk,ld->ijkl",
        first_matrix,
        left_factors.reshape(rank, len(keep), local_dim**2),
        right_factors.reshape(len(keep), rank, local_dim**2),
        last_matrix,
    ).reshape(residual_tensor.shape)
    return (
        left_factors,
        right_factors,
        np.linalg.norm(residual_tensor - reconstructed),
        np.linalg.norm(residual_tensor),
    )


def _dense_loop_tensors(coefficients, local_dim):
    """Factor a four-site dense correction into an exact tensor ring.

    The local operator basis is the matrix-unit basis of size ``d**2``. A
    pair of adjacent basis labels is carried by each virtual leg, giving an
    exact rank ``d**4`` ring without a coefficient-dependent decomposition.
    This is deliberately the dense analogue of the fixed Pauli loop path:
    coefficient values remain ordinary NumPy values while the virtual
    topology is fixed and sparse at the PEPO level.
    """
    operator_rank = local_dim**2
    loop_coefficients = coefficients.transpose(0, 1, 3, 2).reshape(
        operator_rank,
        operator_rank,
        operator_rank**2,
    )
    matrix_units = np.eye(operator_rank, dtype=coefficients.dtype).reshape(
        operator_rank,
        local_dim,
        local_dim,
    )
    pair_labels = np.arange(operator_rank**2)
    first_units = matrix_units[pair_labels // operator_rank]
    second_units = matrix_units[pair_labels % operator_rank]
    diagonal = np.eye(operator_rank**2, dtype=coefficients.dtype)
    first_corner = (
        first_units.reshape(1, operator_rank**2, local_dim, local_dim)
        * diagonal.reshape(operator_rank**2, operator_rank**2, 1, 1)
    )
    second_corner = (
        loop_coefficients.reshape(operator_rank**2, operator_rank**2, 1, 1)
        * second_units.reshape(operator_rank**2, 1, local_dim, local_dim)
    )
    third_corner = (
        first_units.reshape(1, operator_rank**2, local_dim, local_dim)
        * diagonal.reshape(operator_rank**2, operator_rank**2, 1, 1)
    )
    fourth_corner = (
        second_units.reshape(operator_rank**2, 1, local_dim, local_dim)
        * np.ones((1, operator_rank**2, 1, 1), dtype=coefficients.dtype)
    )
    return first_corner, second_corner, third_corner, fourth_corner


@dataclass
class ClusterExpansionPlan:
    """Reusable geometry and symmetry plan for dense cluster-expansion PEPOs.

    The lattice topology and cluster-orbit bookkeeping are cached in the plan;
    beta-dependent local exponentials and residual solves are performed by
    :meth:`build`. Set ``materialize=False`` to retain the sparse active-block
    representation instead of immediately creating dense Quimb tensors.
    Through order four this includes tree clusters and every geometrically
    present four-site plaquette loop.
    """

    lx: int
    ly: int
    twosite_op: np.ndarray
    onesite_op: np.ndarray
    order: int = 3
    cyclic: bool | tuple[bool, bool] = False
    edge_cutoff: float = 0.0
    max_edge_rank: int | None = None
    max_tree_rank: int | None = None
    symmetry: str | None = None
    dtype: object | None = None
    last_report: ClusterExpansionReport | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.lx = _validate_shape(self.lx, "lx")
        self.ly = _validate_shape(self.ly, "ly")
        self.cyclic = _validate_cyclic(self.cyclic, self.lx, self.ly)
        if not isinstance(self.order, Integral):
            raise TypeError("order must be an integer.")
        self.order = int(self.order)
        if self.order < 1:
            raise ValueError("order must be >= 1.")
        if self.order > 4:
            raise NotImplementedError(
                "dense cluster-expansion PEPOs currently support orders 1 through 4; "
                "higher-level clusters are not implemented yet."
            )
        if self.edge_cutoff < 0.0:
            raise ValueError("edge_cutoff must be >= 0.")
        if self.max_edge_rank is not None:
            self.max_edge_rank = _validate_shape(self.max_edge_rank, "max_edge_rank")
        if self.max_tree_rank is not None:
            self.max_tree_rank = _validate_shape(self.max_tree_rank, "max_tree_rank")
        if self.symmetry not in (None, "C4"):
            raise ValueError("symmetry must be None or 'C4'.")

        self.onesite_op = _as_square_operator(self.onesite_op, "onesite_op", dtype=self.dtype)
        local_dim = self.onesite_op.shape[0]
        self.twosite_op = _as_square_operator(self.twosite_op, "twosite_op", dtype=self.dtype)
        if self.twosite_op.shape != (local_dim**2, local_dim**2):
            raise ValueError(
                "twosite_op must have shape "
                f"({local_dim**2}, {local_dim**2}) for local dimension {local_dim}."
            )
        if self.dtype is None:
            self.dtype = np.result_type(self.onesite_op.dtype, self.twosite_op.dtype)
        if self.symmetry == "C4":
            self.dtype = np.result_type(self.dtype, np.complex128)
        self.onesite_op = np.asarray(self.onesite_op, dtype=self.dtype)
        self.twosite_op = np.asarray(self.twosite_op, dtype=self.dtype)

        if self.symmetry == "C4":
            swapped = _swap_two_site_operator(self.twosite_op, local_dim)
            if not np.allclose(self.twosite_op, swapped, rtol=1e-10, atol=1e-12):
                raise ValueError("C4 cluster reduction requires a site-symmetric twosite_op.")

        self.site_directions = {
            (i, j): _site_directions(i, j, self.lx, self.ly, *self.cyclic)
            for i in range(self.lx)
            for j in range(self.ly)
        }
        self.pair_orbits = _pair_orbits() if self.symmetry == "C4" else tuple(
            (pair, (pair,)) for pair in _all_direction_pairs()
        )
        self.triple_orbits = _subset_orbits(3, self.symmetry)
        self.path_orbits = _path_orbits(self.symmetry)
        self.plaquette_starts = _plaquette_starts(
            self.lx,
            self.ly,
            self.cyclic,
        )

    @property
    def pair_representatives(self):
        """Return the connected three-site pair orientations to solve."""
        if self.order < 3:
            return ()
        return tuple(representative for representative, _ in self.pair_orbits)

    @property
    def tree_representatives(self):
        """Return the order-four star and path representatives."""
        if self.order < 4:
            return ()
        stars = tuple(representative for representative, _ in self.triple_orbits)
        paths = tuple(representative for representative, _ in self.path_orbits)
        return stars + paths

    def build(self, beta, *, materialize=True, return_report=False):
        """Build at ``beta`` using the cached topology and symmetry plan."""
        work_dtype = np.result_type(self.dtype, np.asarray(beta).dtype)
        onesite_op = np.asarray(self.onesite_op, dtype=work_dtype)
        twosite_op = np.asarray(self.twosite_op, dtype=work_dtype)
        one_site_exp, start_factors, end_factors = _edge_factors(
            twosite_op,
            onesite_op,
            beta,
            edge_cutoff=self.edge_cutoff,
            max_edge_rank=self.max_edge_rank,
            symmetric=self.symmetry == "C4",
        )
        if self.order < 2:
            start_factors = np.zeros(
                (0, one_site_exp.shape[0], one_site_exp.shape[1]), dtype=work_dtype
            )
            end_factors = start_factors.copy()

        pair_tensors = {}
        pair_residuals = []
        pair_targets = []
        if self.order >= 3 and start_factors.shape[0]:
            for representative, orbit in self.pair_orbits:
                tensor, residual_norm, target_norm = _solve_three_site_pair(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    beta,
                )
                pair_tensors[representative] = tensor
                pair_residuals.append(residual_norm)
                pair_targets.append(target_norm)
                for pair in orbit[1:]:
                    pair_tensors[pair] = _rotate_pair_tensor(
                        representative,
                        pair,
                        tensor,
                    )

        blocks = _initialize_blocks(
            self.lx, self.ly, one_site_exp, self.site_directions
        )
        allocator = _SectorAllocator()
        edge_sectors = allocator.allocate(start_factors.shape[0])
        if edge_sectors:
            _add_positive_edge_channels(
                blocks,
                self.site_directions,
                start_factors,
                end_factors,
                edge_sectors,
            )
            for pair, tensor in pair_tensors.items():
                _add_pair_blocks(
                    blocks,
                    self.site_directions,
                    pair,
                    tensor,
                    edge_sectors,
                )

        star_residuals = []
        star_targets = []
        path_residuals = []
        path_targets = []
        path_ranks = []
        loop_residuals = []
        loop_targets = []
        loop_rank = 0
        solved_tree_groups = 0
        if self.order >= 4 and start_factors.shape[0]:
            star_tensors = {}
            for representative, orbit in self.triple_orbits:
                if not any(
                    all(direction in directions for direction in star)
                    for star in orbit
                    for directions in self.site_directions.values()
                ):
                    continue
                tensor, residual_norm, target_norm = _solve_four_star(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                    beta,
                )
                star_tensors[representative] = tensor
                star_residuals.append(residual_norm)
                star_targets.append(target_norm)
                solved_tree_groups += 1
                for directions in orbit[1:]:
                    star_tensors[directions] = _rotate_direction_tensor(
                        representative, directions, tensor
                    )

            for directions, tensor in star_tensors.items():
                sectors = allocator.allocate(start_factors.shape[0])
                _add_triple_blocks(
                    blocks, self.site_directions, directions, tensor, sectors
                )
                for direction in directions:
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        _OPPOSITE_DIRECTION[direction],
                        sectors,
                        end_factors
                        if direction in _POSITIVE_DIRECTIONS
                        else start_factors,
                        source=False,
                        cyclic=self.cyclic,
                    )

            for representative, orbit in self.path_orbits:
                if not any(
                    _path_start_sites(steps, self.lx, self.ly, self.cyclic)
                    for steps in orbit
                ):
                    continue
                left, right, residual_norm, target_norm = _solve_four_path(
                    representative,
                    twosite_op,
                    onesite_op,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                    beta,
                    self.max_tree_rank,
                )
                path_residuals.append(residual_norm)
                path_targets.append(target_norm)
                path_ranks.append(left.shape[1])
                solved_tree_groups += 1
                path_channel_rank = left.shape[1]
                if not left.shape[1]:
                    continue
                representative_left_dirs = tuple(
                    sorted(
                        (_OPPOSITE_DIRECTION[representative[0]], representative[1]),
                        key=_DIRECTIONS.index,
                    )
                )
                representative_left_role = (
                    _OPPOSITE_DIRECTION[representative[0]],
                    representative[1],
                )
                representative_right_dirs = tuple(
                    sorted(
                        (_OPPOSITE_DIRECTION[representative[1]], representative[2]),
                        key=_DIRECTIONS.index,
                    )
                )
                representative_right_role = (
                    _OPPOSITE_DIRECTION[representative[1]],
                    representative[2],
                )
                if representative_left_role != representative_left_dirs:
                    left = left.transpose(1, 0, 2, 3)
                if representative_right_role != representative_right_dirs:
                    right = right.transpose(1, 0, 2, 3)
                for steps in orbit:
                    if steps == representative:
                        rotated_left, rotated_right = left, right
                    else:
                        target_left_dirs = tuple(
                            sorted(
                                (_OPPOSITE_DIRECTION[steps[0]], steps[1]),
                                key=_DIRECTIONS.index,
                            )
                        )
                        target_right_dirs = tuple(
                            sorted(
                                (_OPPOSITE_DIRECTION[steps[1]], steps[2]),
                                key=_DIRECTIONS.index,
                            )
                        )
                        rotated_left = _rotate_direction_tensor(
                            representative_left_dirs, target_left_dirs, left
                        )
                        rotated_right = _rotate_direction_tensor(
                            representative_right_dirs, target_right_dirs, right
                        )
                    first_sectors = allocator.allocate(start_factors.shape[0])
                    middle_sectors = allocator.allocate(path_channel_rank)
                    last_sectors = allocator.allocate(end_factors.shape[0])
                    first_factor = (
                        start_factors
                        if steps[0] in _POSITIVE_DIRECTIONS
                        else end_factors
                    )
                    last_factor = (
                        end_factors
                        if steps[2] in _POSITIVE_DIRECTIONS
                        else start_factors
                    )
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        steps[0],
                        first_sectors,
                        first_factor,
                        source=True,
                        cyclic=self.cyclic,
                    )
                    _add_single_direction_blocks(
                        blocks,
                        self.site_directions,
                        self.lx,
                        self.ly,
                        _OPPOSITE_DIRECTION[steps[2]],
                        last_sectors,
                        last_factor,
                        source=False,
                        cyclic=self.cyclic,
                    )
                    back = _OPPOSITE_DIRECTION[steps[0]]
                    forward = steps[1]
                    left_role = (back, forward)
                    left_dirs = tuple(sorted(left_role, key=_DIRECTIONS.index))
                    left_tensor = rotated_left
                    if left_role != left_dirs:
                        left_tensor = left_tensor.transpose(1, 0, 2, 3)
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        left_role,
                        left_tensor,
                        (first_sectors, middle_sectors),
                    )
                    back = _OPPOSITE_DIRECTION[steps[1]]
                    forward = steps[2]
                    right_role = (back, forward)
                    right_dirs = tuple(sorted(right_role, key=_DIRECTIONS.index))
                    right_tensor = rotated_right
                    if right_role != right_dirs:
                        right_tensor = right_tensor.transpose(1, 0, 2, 3)
                    _add_pair_blocks(
                        blocks,
                        self.site_directions,
                        right_role,
                        right_tensor,
                        (middle_sectors, last_sectors),
                    )

            if self.plaquette_starts:
                loop_edges = _plaquette_edges()
                exact_loop, _ = _lower_loop_residual(
                    loop_edges,
                    twosite_op,
                    onesite_op,
                    beta,
                    one_site_exp,
                    start_factors,
                    end_factors,
                    pair_tensors,
                )
                loop_rank = one_site_exp.shape[0] ** 4
                for start in self.plaquette_starts:
                    upper = _site_after(
                        start,
                        "u",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    right = _site_after(
                        start,
                        "r",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    diagonal = _site_after(
                        upper,
                        "r",
                        self.lx,
                        self.ly,
                        self.cyclic,
                    )
                    loop_sites = (start, upper, diagonal, right)
                    lower_loop = _cycle_active_operator(
                        blocks,
                        self.site_directions,
                        loop_sites,
                        one_site_exp.shape[0],
                    )
                    lower_loop = _permute_operator_sites(
                        lower_loop,
                        (0, 1, 3, 2),
                        one_site_exp.shape[0],
                    )
                    loop_residual = exact_loop - lower_loop
                    loop_tensor = _operator_tensor(
                        loop_residual,
                        4,
                        one_site_exp.shape[0],
                    )
                    loop_tensors = _dense_loop_tensors(
                        loop_tensor,
                        one_site_exp.shape[0],
                    )
                    loop_residuals.append(np.linalg.norm(loop_residual))
                    loop_targets.append(np.linalg.norm(loop_residual))
                    lower_bond = allocator.allocate(loop_rank)
                    right_bond = allocator.allocate(loop_rank)
                    upper_bond = allocator.allocate(loop_rank)
                    left_bond = allocator.allocate(loop_rank)
                    _add_pair_block_at_site(
                        blocks,
                        self.site_directions,
                        loop_sites[0],
                        ("r", "u"),
                        loop_tensors[0],
                        (left_bond, lower_bond),
                    )
                    _add_pair_block_at_site(
                        blocks,
                        self.site_directions,
                        loop_sites[1],
                        ("d", "r"),
                        loop_tensors[1],
                        (lower_bond, right_bond),
                    )
                    _add_pair_block_at_site(
                        blocks,
                        self.site_directions,
                        loop_sites[2],
                        ("l", "d"),
                        loop_tensors[2],
                        (right_bond, upper_bond),
                    )
                    _add_pair_block_at_site(
                        blocks,
                        self.site_directions,
                        loop_sites[3],
                        ("u", "l"),
                        loop_tensors[3],
                        (upper_bond, left_bond),
                    )

        active = ActivePEPOBlocks(
            lx=self.lx,
            ly=self.ly,
            cyclic=self.cyclic,
            bond_dim=allocator.next_sector,
            physical_dim=one_site_exp.shape[0],
            site_directions=self.site_directions,
            blocks=blocks,
        )
        residual_norms = {}
        relative_residuals = {}
        if self.order >= 2:
            edge_residual = _edge_fit_residual(
                twosite_op,
                onesite_op,
                beta,
                one_site_exp,
                start_factors,
                end_factors,
            )
            edge_target = edge_residual + sum(
                (
                    np.kron(start_factors[index], end_factors[index])
                    for index in range(start_factors.shape[0])
                ),
                start=np.zeros_like(edge_residual),
            )
            residual_norms["edge"] = float(np.linalg.norm(edge_residual))
            relative_residuals["edge"] = float(
                np.linalg.norm(edge_residual) / max(np.linalg.norm(edge_target), np.finfo(float).eps)
            )
        if pair_residuals:
            residual_norms["three_site"] = float(max(pair_residuals))
            relative_residuals["three_site"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(pair_residuals, pair_targets)
                )
            )
        if star_residuals:
            residual_norms["four_site_star"] = float(max(star_residuals))
            relative_residuals["four_site_star"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(star_residuals, star_targets)
                )
            )
        if path_residuals:
            residual_norms["four_site_path"] = float(max(path_residuals))
            relative_residuals["four_site_path"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(path_residuals, path_targets)
                )
            )
        if loop_residuals:
            residual_norms["four_site_loop"] = float(max(loop_residuals))
            relative_residuals["four_site_loop"] = float(
                max(
                    residual / max(target, np.finfo(float).eps)
                    for residual, target in zip(loop_residuals, loop_targets)
                )
            )
        counts = {
            "edge": sum(
                direction in _POSITIVE_DIRECTIONS
                for directions in self.site_directions.values()
                for direction in directions
            ),
            "three_site": (
                sum(
                    len(directions) * (len(directions) - 1) // 2
                    for directions in self.site_directions.values()
                )
                if self.order >= 3
                else 0
            ),
            "four_site_star": 0,
            "four_site_path": 0,
            "four_site_loop": len(self.plaquette_starts) if self.order >= 4 else 0,
            "four_site_tree_solved": (
                solved_tree_groups if self.order >= 4 else 0
            ),
        }
        if self.order >= 4:
            for _, orbit in self.triple_orbits:
                counts["four_site_star"] += sum(
                    sum(all(direction in directions for direction in star) for directions in self.site_directions.values())
                    for star in orbit
                )
            for _, orbit in self.path_orbits:
                counts["four_site_path"] += sum(
                    len(_path_start_sites(steps, self.lx, self.ly, self.cyclic))
                    for steps in orbit
                )
        report = ClusterExpansionReport(
            beta=beta,
            order=self.order,
            local_dim=one_site_exp.shape[0],
            edge_rank=start_factors.shape[0],
            tree_rank=max(path_ranks, default=0),
            loop_rank=loop_rank,
            cluster_counts=counts,
            residual_norms=residual_norms,
            relative_residual_norms=relative_residuals,
            active_block_count=active.active_block_count,
            active_nbytes=active.active_nbytes,
            dense_nbytes=active.dense_nbytes,
        )
        self.last_report = report
        result = active.to_pepo() if materialize else active
        return (result, report) if return_report else result


def build_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    twosite_op,
    onesite_op,
    *,
    order=3,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    dtype=None,
    symmetry=None,
    materialize=True,
    return_report=False,
):
    """Build a dense square-lattice PEPO cluster expansion.

    The PEPO represents an extensive approximation to
    ``exp(-beta * H)``.  ``onesite_op`` is the one-site Hamiltonian term and
    ``twosite_op`` is the canonical two-site term for a positive lattice
    direction (``u`` or ``r``).  The local two-site residual is factorized
    into virtual channels.  Order three additionally solves every connected
    three-site path and corner residual, filling the corresponding two-active
    virtual entries of the PEPO tensor. Order four adds four-site T-shaped
    and non-loop path tree residuals plus present plaquette-loop residuals.
    Plaquettes use an exact fixed-rank tensor-ring factorization.

    Parameters
    ----------
    lx, ly : int
        Square-lattice dimensions.
    beta : scalar
        Real or complex imaginary-time step.
    twosite_op, onesite_op : array-like
        Dense square matrices of shapes ``(d**2, d**2)`` and ``(d, d)``.
    order : {1, 2, 3, 4}, default=3
        Largest connected cluster size. Orders above four require higher-level
        cluster construction.
    cyclic : bool or tuple[bool, bool], default=False
        Whether to close both lattice directions, or close x and y
        independently.
    edge_cutoff : float, default=0.0
        Relative singular-value cutoff for two-site residual channels.
    max_edge_rank : int | None, default=None
        Optional cap on the number of retained two-site channels.
    max_tree_rank : int | None, default=None
        Optional cap on the internal SVD rank of four-site path clusters.
    dtype : numpy dtype | None, default=None
        Optional dense dtype for all local tensors.
    symmetry : {None, "C4"}, default=None
        Reduce equivalent tree orientations using a symmetric virtual
        factorization. This is appropriate for square-lattice ITF and other
        site-symmetric C4 models.
    return_report : bool, default=False
        Return ``(pepo_or_active_blocks, ClusterExpansionReport)``.
    materialize : bool, default=True
        Return a dense Quimb PEPO when true, otherwise return active blocks.

    Returns
    -------
    quimb.tensor.PEPO
        A dense PEPO with physical ``b``/``k`` operator indices.

    Notes
    -----
    This implementation includes four-site plaquette loops and an active
    block/Symmray materialization boundary. Higher clusters and mixed-charge
    component splitting remain separate stages.
    """
    plan = ClusterExpansionPlan(
        lx,
        ly,
        twosite_op,
        onesite_op,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        symmetry=symmetry,
        dtype=dtype,
    )
    return plan.build(
        beta,
        materialize=materialize,
        return_report=return_report,
    )


def build_itf_cluster_expansion_pepo(
    lx,
    ly,
    beta,
    *,
    J=1.0,
    field=1.0,
    order=3,
    cyclic=False,
    edge_cutoff=0.0,
    max_edge_rank=None,
    max_tree_rank=None,
    dtype="float64",
    symmetry="C4",
    materialize=True,
    return_report=False,
):
    """Build a cluster-expansion PEPO for Pepsy's square-lattice ITF.

    The convention matches :meth:`pepsy.operators.ham_tn.build_itf`:
    ``H = J * sum Z_i Z_j + field * sum X_i``.
    """
    z = np.asarray(quimb.pauli("Z", dtype=dtype), dtype=dtype)
    x = np.asarray(quimb.pauli("X", dtype=dtype), dtype=dtype)
    return build_cluster_expansion_pepo(
        lx,
        ly,
        beta,
        J * np.kron(z, z),
        field * x,
        order=order,
        cyclic=cyclic,
        edge_cutoff=edge_cutoff,
        max_edge_rank=max_edge_rank,
        max_tree_rank=max_tree_rank,
        symmetry=symmetry,
        dtype=dtype,
        materialize=materialize,
        return_report=return_report,
    )

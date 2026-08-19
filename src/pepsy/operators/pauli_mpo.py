"""Sparse Pauli-basis operators with an MPO execution boundary.

``PauliMPO`` keeps an operator in a canonical sparse expansion over
``I``, ``X``, ``Y`` and ``Z`` strings.  It is deliberately a front end to
Pepsy's existing :class:`~pepsy.operators.mpo.FirstDegreeMPO`: exact Pauli
algebra and Pauli-basis traces stay native, while general tensor-network
operations such as MPS application, Quimb contraction, MPO compression, and
higher-order exponentials use the established MPO implementation.

Short strings passed to :meth:`PauliMPO.from_terms` are translated across the
chain.  Thus ``(J, "ZZ")`` is the nearest-neighbor sum and ``(h, "X")`` is
the onsite sum.  A full-length word, or an explicit ``sites`` entry, denotes
one product term.  Set ``boundary="periodic"`` to include wrapped
translations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import itertools
import math
from numbers import Integral

import autoray as ar
import numpy as np

from .mpo import FirstDegreeMPO, MPOProductTerm

__all__ = [
    "PauliBondCompressionReport",
    "PauliCompressionReport",
    "PauliMPO",
    "decompose_pauli",
]


@dataclass(frozen=True)
class PauliBondCompressionReport:
    """Diagnostics for one native Pauli-MPO bond compression."""

    bond: int
    original_bond: int
    final_bond: int
    discarded_rank: int
    discarded_weight: float
    largest_singular_value: float
    singular_values: tuple


@dataclass(frozen=True)
class PauliCompressionReport:
    """Diagnostics for native Pauli-basis tensor-train compression."""

    original_bond_dimensions: tuple
    final_bond_dimensions: tuple
    discarded_singular_weight: float
    discarded_ranks: tuple
    cutoff: float
    cutoff_mode: str
    max_bond: int | None
    form: object = "right"
    renorm: int | bool | None = False
    method: str = "svd"
    bond_reports: tuple = ()

    @property
    def exact(self):
        """Whether the SVD stage discarded no singular weight."""

        return math.isclose(self.discarded_singular_weight, 0.0, abs_tol=1.0e-14)

    @property
    def per_bond(self):
        """Alias for the detailed per-bond diagnostics."""

        return self.bond_reports


_PAULIS = frozenset("IXYZ")
_PAULI_PRODUCT = {
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),
    ("Y", "I"): ("Y", 1),
    ("Z", "I"): ("Z", 1),
    ("X", "X"): ("I", 1),
    ("Y", "Y"): ("I", 1),
    ("Z", "Z"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}
_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def _check_scalar(value, *, name="coefficient"):
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        ndim = np.ndim(value)
    if ndim != 0:
        raise TypeError(f"{name} must be scalar, got ndim={ndim}.")


def _add(left, right):
    return ar.do("add", left, right)


def _multiply(left, right):
    return ar.do("multiply", left, right)


def _conjugate(value):
    return ar.do("conj", value)


def _copy_array(value):
    """Copy an array without forcing it through NumPy."""

    if hasattr(value, "clone"):
        return value.clone()
    try:
        return value.copy()
    except AttributeError:  # pragma: no cover - backend fallback
        return ar.do("array", value)


def _reshape(value, shape):
    return ar.do("reshape", value, shape)


def _transpose(value, axes):
    return ar.do("transpose", value, axes)


def _einsum(equation, *operands):
    # ``optimize`` is not accepted by several backend einsum implementations
    # (notably Torch), so keep this wrapper deliberately backend-neutral.
    return ar.do("einsum", equation, *operands)


def _concatenate(values, axis):
    return ar.do("concatenate", tuple(values), axis)


def _zeros(shape, *, like):
    return ar.do("zeros", tuple(shape), like=like)


def _array_like(value, like):
    return ar.do("array", np.asarray(value), like=like)


def _is_complex_array(value):
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        dtype_name = str(dtype).lower()
        if "complex" in dtype_name:
            return True
        if "float" in dtype_name or "int" in dtype_name:
            return False
    return np.issubdtype(_host_array(value).dtype, np.complexfloating)


def _complexify(value):
    """Promote a backend array to a complex dtype while preserving its graph."""

    if _is_complex_array(value):
        return value
    return ar.do("astype", value, dtype=complex)


def _host_array(value):
    try:
        return np.asarray(ar.to_numpy(value))
    except Exception:  # pragma: no cover - backend-specific fallback
        return np.asarray(value)


def _host_shape(value):
    return tuple(int(size) for size in getattr(value, "shape", np.shape(value)))


def _backend_name(value):
    try:
        return ar.infer_backend(value)
    except Exception:  # pragma: no cover - defensive backend fallback
        return None


def _host_zero(value):
    """Return whether a readily inspectable scalar is exactly zero."""

    if _backend_name(value) not in {"builtins", "numpy"}:
        return False
    try:
        return bool(np.asarray(value) == 0)
    except (TypeError, ValueError):  # pragma: no cover - backend guard
        return False


def _normalize_word(word):
    if not isinstance(word, str):
        try:
            word = "".join(word)
        except TypeError as exc:
            raise TypeError("Pauli word must be a string or character sequence.") from exc
    word = word.replace(" ", "").upper()
    if not word:
        raise ValueError("Pauli words must not be empty.")
    invalid = sorted(set(word) - _PAULIS)
    if invalid:
        raise ValueError(
            "Pauli words may contain only I, X, Y, and Z; "
            f"invalid label(s): {', '.join(invalid)}."
        )
    return word


def _normalize_boundary(boundary):
    boundary = str(boundary).strip().lower().replace("-", "_")
    if boundary in {"pbc", "cyclic"}:
        boundary = "periodic"
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'.")
    return boundary


def _normalize_sites(sites, nsites):
    if isinstance(sites, Integral):
        sites = (int(sites),)
    else:
        try:
            sites = tuple(int(site) for site in sites)
        except TypeError as exc:
            raise TypeError("sites must be an integer or an iterable of integers.") from exc
    if not sites:
        raise ValueError("sites must not be empty.")
    if len(set(sites)) != len(sites):
        raise ValueError("sites must be distinct.")
    if any(site < 0 or site >= nsites for site in sites):
        raise ValueError(f"sites must lie in the range 0..{nsites - 1}.")
    return tuple(sorted(sites))


def _normalize_where(where, nsites):
    """Normalize gate support while preserving the supplied qubit order."""

    if isinstance(where, Integral):
        where = (int(where),)
    else:
        try:
            where = tuple(int(site) for site in where)
        except TypeError as exc:
            raise TypeError(
                "where must be an integer or an iterable of integers."
            ) from exc
    if not where:
        raise ValueError("where must contain at least one site.")
    if len(set(where)) != len(where):
        raise ValueError("where must contain distinct sites.")
    if any(site < 0 or site >= nsites for site in where):
        raise ValueError(f"where must lie in the range 0..{nsites - 1}.")
    return where


def _as_dense_matrix(operator):
    """Convert a dense/operator-like input to a host-side matrix."""

    try:
        operator = ar.to_numpy(operator)
    except Exception as exc:  # pragma: no cover - backend-specific fallback
        raise TypeError("dense operators must be array-like matrices.") from exc
    matrix = np.asarray(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "dense operators must be square rank-2 matrices, "
            f"got shape {matrix.shape}."
        )
    dimension = int(matrix.shape[0])
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError(
            "dense qubit operators must have dimension 2**k with k >= 1, "
            f"got {dimension}."
        )
    nqubits = dimension.bit_length() - 1
    if dimension != 2**nqubits:  # pragma: no cover - guarded by power-of-two check
        raise ValueError(f"invalid qubit operator dimension {dimension}.")
    return matrix, nqubits


def _kron_pauli(word):
    result = _PAULI_MATRICES[word[0]]
    for label in word[1:]:
        result = np.kron(result, _PAULI_MATRICES[label])
    return result


def _validate_tolerance(value, *, name):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}.")
    return value


def decompose_pauli(
    operator,
    *,
    atol=None,
    rtol=0.0,
    max_qubits=None,
):
    """Decompose a dense qubit operator into Pauli strings.

    Parameters
    ----------
    operator : array-like
        A square ``2**k`` by ``2**k`` matrix.  The decomposition is performed
        on the host, so a differentiable dense backend array is treated as a
        numerical gate at this boundary.
    atol, rtol : float or None
        Coefficients with magnitude no greater than
        ``atol + rtol * max(abs(operator))`` are omitted.  By default ``atol``
        is a dtype-scaled roundoff threshold and ``rtol=0``.
    max_qubits : int or None
        Optional guard against accidentally enumerating ``4**k`` strings.

    Returns
    -------
    tuple
        Canonical ``(coefficient, word)`` pairs, ordered lexicographically by
        the Pauli word.  The expansion is exact up to the requested tolerance.
    """

    matrix, nqubits = _as_dense_matrix(operator)
    if max_qubits is not None:
        if not isinstance(max_qubits, Integral) or int(max_qubits) < 1:
            raise ValueError("max_qubits must be a positive integer or None.")
        if nqubits > int(max_qubits):
            raise ValueError(
                f"decomposing a {nqubits}-qubit operator enumerates {4**nqubits} "
                f"Pauli strings, exceeding max_qubits={int(max_qubits)}."
            )
    atol = _validate_tolerance(atol, name="atol")
    rtol = _validate_tolerance(rtol, name="rtol")
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if atol is None:
        real_dtype = np.asarray(matrix.real).dtype
        if not np.issubdtype(real_dtype, np.inexact):
            real_dtype = np.dtype("float64")
        atol = nqubits * 2 * np.finfo(real_dtype).eps * scale
    threshold = atol + rtol * scale
    normalization = 1.0 / (2**nqubits)
    terms = []
    for labels in itertools.product("IXYZ", repeat=nqubits):
        word = "".join(labels)
        coefficient = normalization * np.trace(_kron_pauli(word) @ matrix)
        if abs(coefficient) > threshold:
            terms.append((complex(coefficient), word))
    return tuple(terms)


def _validate_pauli_cores(cores, *, copy=True):
    cores = tuple(cores)
    if not cores:
        raise ValueError("cores must contain at least one Pauli-MPO core.")
    for site, core in enumerate(cores):
        shape = _host_shape(core)
        if len(shape) != 3 or shape[2] != 4:
            raise ValueError(
                "each Pauli-MPO core must have shape (left, right, 4); "
                f"core {site} has shape {shape}."
            )
        if any(int(size) < 1 for size in shape[:2]):
            raise ValueError(f"core {site} has an empty virtual bond.")
        if site and _host_shape(cores[site - 1])[1] != shape[0]:
            raise ValueError(
                f"Pauli-MPO bond mismatch between sites {site - 1} and {site}."
            )
    if _host_shape(cores[0])[0] != 1 or _host_shape(cores[-1])[1] != 1:
        raise ValueError("the first left and last right Pauli-MPO bonds must be one.")
    return tuple(_copy_array(core) if copy else core for core in cores)


def _core_bond_dimensions(cores):
    return tuple(_host_shape(core)[1] for core in cores[:-1])


def _sparse_pauli_cores(terms, nsites):
    """Build an exact Pauli-label MPO from a sparse word expansion."""

    terms = tuple(terms)
    if not terms:
        return tuple(np.zeros((1, 1, 4), dtype=complex) for _ in range(nsites))
    raw_coefficients = [coefficient for coefficient, _ in terms]
    backend_coefficients = [
        coefficient
        for coefficient in raw_coefficients
        if _backend_name(coefficient) not in {"builtins", "numpy"}
    ]
    if backend_coefficients:
        like = backend_coefficients[0]
        backend_raw_coefficients = tuple(
            coefficient
            if _backend_name(coefficient) not in {"builtins", "numpy"}
            else _array_like(coefficient, like)
            for coefficient in raw_coefficients
        )
        coefficients = ar.do("stack", backend_raw_coefficients, like=like)
        dtype = None
    else:
        coefficients = [complex(ar.to_numpy(coefficient)) for coefficient in raw_coefficients]
        like = None
        dtype = np.result_type(np.asarray(coefficients).dtype, complex)
    nterms = len(terms)
    if nsites == 1:
        if like is None:
            core = np.zeros((1, 1, 4), dtype=dtype)
            for coefficient, (_, word) in zip(coefficients, terms):
                core[0, 0, "IXYZ".index(word[0])] += coefficient
        else:
            mask = np.zeros((nterms, 4), dtype=float)
            for term_index, (_, word) in enumerate(terms):
                mask[term_index, "IXYZ".index(word[0])] = 1.0
            core = _reshape(
                _einsum("np,n->p", _array_like(mask, like), coefficients),
                (1, 1, 4),
            )
        return (core,)

    cores = []
    if like is None:
        first = np.zeros((1, nterms, 4), dtype=dtype)
        for term_index, (coefficient, (_, word)) in enumerate(zip(coefficients, terms)):
            first[0, term_index, "IXYZ".index(word[0])] = coefficient
    else:
        mask = np.zeros((nterms, 4), dtype=float)
        for term_index, (_, word) in enumerate(terms):
            mask[term_index, "IXYZ".index(word[0])] = 1.0
        first = _reshape(
            _array_like(mask, like) * _reshape(coefficients, (1, nterms, 1)),
            (1, nterms, 4),
        )
    cores.append(first)
    for site in range(1, nsites - 1):
        middle_mask = np.zeros((nterms, nterms, 4), dtype=float)
        for term_index, (_, word) in enumerate(terms):
            middle_mask[term_index, term_index, "IXYZ".index(word[site])] = 1.0
        middle = (
            _array_like(middle_mask, like)
            if like is not None
            else middle_mask.astype(dtype)
        )
        cores.append(middle)
    last_mask = np.zeros((nterms, 4), dtype=float)
    for term_index, (_, word) in enumerate(terms):
        last_mask[term_index, "IXYZ".index(word[-1])] = 1.0
    last = _reshape(
        _array_like(last_mask, like) if like is not None else last_mask.astype(dtype),
        (nterms, 1, 4),
    )
    cores.append(last)
    return tuple(cores)


def _left_canonicalize_cores(cores):
    cores = [_copy_array(core) for core in cores]
    for site in range(len(cores) - 1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left * physical, right))
        q, r = ar.do("linalg.qr", matrix, mode="reduced")
        rank = _host_shape(q)[1]
        cores[site] = _transpose(_reshape(q, (left, physical, rank)), (0, 2, 1))
        cores[site + 1] = _einsum(
            "ab,bcp->acp",
            r,
            cores[site + 1],
        )
    return tuple(cores)


def _canonicalize_core_chain(cores, center):
    """Canonicalize Pauli-label cores around ``center`` using QR factors."""

    cores = list(_left_canonicalize_cores(cores))
    for site in range(len(cores) - 1, center, -1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left, physical * right))
        q, r = ar.do("linalg.qr", _transpose(matrix, (1, 0)), mode="reduced")
        rank = _host_shape(q)[1]
        cores[site] = _transpose(
            _reshape(_transpose(q, (1, 0)), (rank, physical, right)),
            (0, 2, 1),
        )
        cores[site - 1] = _einsum(
            "abp,bc->acp",
            cores[site - 1],
            _transpose(r, (1, 0)),
        )
    return tuple(cores)


def _right_canonicalize_cores(cores):
    """Right-canonicalize Pauli-label cores with QR/LQ sweeps."""

    cores = [_copy_array(core) for core in cores]
    for site in range(len(cores) - 1, 0, -1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left, physical * right))
        q, r = ar.do("linalg.qr", _transpose(matrix, (1, 0)), mode="reduced")
        rank = _host_shape(q)[1]
        cores[site] = _transpose(
            _reshape(_transpose(q, (1, 0)), (rank, physical, right)),
            (0, 2, 1),
        )
        cores[site - 1] = _einsum(
            "abp,bc->acp",
            cores[site - 1],
            _transpose(r, (1, 0)),
        )
    return tuple(cores)


def _truncate_rank(singular_values, *, cutoff, cutoff_mode, max_bond):
    if singular_values.size == 0:
        return 1
    if cutoff_mode == "rel":
        rank = int(np.count_nonzero(singular_values > cutoff * singular_values[0]))
    elif cutoff_mode == "abs":
        rank = int(np.count_nonzero(singular_values > cutoff))
    elif cutoff_mode in {"sum2", "rsum2"}:
        discarded_budget = cutoff
        if cutoff_mode == "rsum2":
            discarded_budget *= float(np.vdot(singular_values, singular_values).real)
        rank = len(singular_values)
        for candidate in range(len(singular_values) + 1):
            tail = singular_values[candidate:]
            if float(np.vdot(tail, tail).real) <= discarded_budget:
                rank = candidate
                break
    elif cutoff_mode in {"sum1", "rsum1"}:
        discarded_budget = cutoff
        if cutoff_mode == "rsum1":
            discarded_budget *= float(np.sum(singular_values))
        rank = len(singular_values)
        for candidate in range(len(singular_values) + 1):
            if float(np.sum(singular_values[candidate:])) <= discarded_budget:
                rank = candidate
                break
    else:  # pragma: no cover - validated by the public method
        raise ValueError(f"unsupported cutoff_mode={cutoff_mode!r}.")
    rank = max(1, rank)
    if max_bond is not None:
        rank = min(rank, max_bond)
    return rank


def _renormalize_singular_values(singular_values, kept, renorm, cutoff_mode):
    if not renorm:
        return singular_values[:kept]
    if renorm is True:
        renorm = 1 if cutoff_mode in {"sum1", "rsum1"} else 2
    if renorm not in {1, 2}:
        raise ValueError("renorm must be False, True, 1, or 2.")
    host_singular_values = _host_array(singular_values).real
    if renorm == 1:
        denominator = float(np.sum(host_singular_values[:kept]))
        target = float(np.sum(host_singular_values))
    else:
        denominator = float(np.vdot(host_singular_values[:kept], host_singular_values[:kept]).real)
        target = float(np.vdot(host_singular_values, host_singular_values).real)
    if denominator == 0.0:
        return singular_values[:kept]
    factor = math.sqrt(target / denominator) if renorm == 2 else target / denominator
    return singular_values[:kept] * factor


def _coefficient_cutoff(cutoff, cutoff_mode, nsites):
    """Convert Quimb operator-norm cutoffs to unnormalized Pauli coefficients."""

    # Each local Pauli has Hilbert-Schmidt norm sqrt(2), whereas the native
    # coefficient cores use the unnormalized labels I/X/Y/Z. Every operator
    # Schmidt spectrum is therefore larger by 2**(L / 2) at the Quimb MPO
    # boundary. Relative policies are invariant; absolute and sum policies
    # need this fixed conversion to select the same ranks as Quimb.
    physical_scale = 2.0 ** (0.5 * nsites)
    if cutoff_mode in {"abs", "sum1"}:
        return cutoff / physical_scale
    if cutoff_mode == "sum2":
        return cutoff / (physical_scale**2)
    return cutoff


def _native_direct_sum(left_cores, right_cores, sign=1):
    """Add two Pauli core trains with block-diagonal virtual bonds."""

    if len(left_cores) != len(right_cores):
        raise ValueError("Pauli core trains must have the same length.")
    if len(left_cores) == 1:
        return (_add(left_cores[0], _multiply(sign, right_cores[0])),)
    result = []
    left = left_cores[0]
    right = _multiply(sign, right_cores[0])
    result.append(_concatenate((left, right), 1))
    for left, right in zip(left_cores[1:-1], right_cores[1:-1]):
        ll, lr, physical = _host_shape(left)
        rl, rr, _ = _host_shape(right)
        top = _concatenate(
            (left, _zeros((ll, rr, physical), like=left)),
            1,
        )
        bottom = _concatenate(
            (_zeros((rl, lr, physical), like=left), _multiply(sign, right)),
            1,
        )
        result.append(_concatenate((top, bottom), 0))
    result.append(
        _concatenate(
            (left_cores[-1], _multiply(sign, right_cores[-1])),
            0,
        )
    )
    return tuple(result)


def _native_product(left_cores, right_cores):
    """Multiply two Pauli core trains using the local IXYZ product tensor."""

    if len(left_cores) != len(right_cores):
        raise ValueError("Pauli core trains must have the same length.")
    product_tensor = np.zeros((4, 4, 4), dtype=complex)
    for (left, right), (out, phase) in _PAULI_PRODUCT.items():
        product_tensor["IXYZ".index(left), "IXYZ".index(right), "IXYZ".index(out)] = phase
    result = []
    for left, right in zip(left_cores, right_cores):
        left = _complexify(left)
        right = _complexify(right)
        ll, lr, _ = _host_shape(left)
        rl, rr, _ = _host_shape(right)
        tensor = _array_like(product_tensor, left)
        product = _einsum("iap,jbq,pqr->ijabr", left, right, tensor)
        result.append(_reshape(product, (ll * rl, lr * rr, 4)))
    return tuple(result)


def _native_trace(cores, *, normalized=False):
    factor = 1.0 if normalized else 2.0
    state = None
    for core in cores:
        local = _multiply(core[:, :, 0], factor)
        if state is None:
            state = local[0, :]
        else:
            state = _einsum("a,ab->b", state, local)
    return state[0]


def _native_inner(left_cores, right_cores, *, normalized=False):
    if len(left_cores) != len(right_cores):
        raise ValueError("Pauli core trains must have the same length.")
    use_complex = any(
        _is_complex_array(core) for core in (*left_cores, *right_cores)
    )
    if use_complex:
        left_cores = tuple(_complexify(core) for core in left_cores)
        right_cores = tuple(_complexify(core) for core in right_cores)
    state = _array_like([[1.0]], left_cores[0])
    for left, right in zip(left_cores, right_cores):
        state = _einsum(
            "ij,iap,jbp->ab",
            state,
            _conjugate(left),
            right,
        )
    factor = 1.0 if normalized else float(2**len(left_cores))
    return _multiply(state[0, 0], factor)


def _native_partial_trace(cores, traced, *, normalized=False):
    traced = set(traced)
    factor = 1.0 if normalized else 2.0
    kept_cores = []
    pending = None
    for site, core in enumerate(cores):
        if site in traced:
            transfer = _multiply(core[:, :, 0], factor)
            if pending is None:
                pending = transfer
            else:
                pending = _einsum("ab,bc->ac", pending, transfer)
            continue
        if pending is None:
            kept = core
        else:
            kept = _einsum("ab,bcp->acp", pending, core)
            pending = None
        kept_cores.append(kept)
    if not kept_cores:
        if pending is None:
            return _array_like(1.0, cores[0])
        return pending[0, 0]
    if pending is not None:
        kept_cores[-1] = _einsum("abp,bc->acp", kept_cores[-1], pending)
    return tuple(kept_cores)


def _randomized_svd(matrix, *, max_rank=None, oversample=8, n_iter=2, seed=None):
    """Backend-native randomized SVD with a host-generated fixed probe."""

    shape = _host_shape(matrix)
    nrows, ncols = shape
    min_dim = min(nrows, ncols)
    target = min_dim if max_rank is None else min(min_dim, int(max_rank))
    if target >= min_dim:
        return ar.do("linalg.svd", matrix, full_matrices=False)
    if not isinstance(oversample, Integral) or int(oversample) < 0:
        raise ValueError("oversample must be a nonnegative integer.")
    if not isinstance(n_iter, Integral) or int(n_iter) < 0:
        raise ValueError("n_iter must be a nonnegative integer.")
    width = min(min_dim, target + int(oversample))
    rng = np.random.default_rng(seed)
    probe = _array_like(rng.standard_normal((ncols, width)), matrix)
    y = matrix @ probe
    for _ in range(int(n_iter)):
        q, _ = ar.do("linalg.qr", y, mode="reduced")
        y = matrix @ (_conjugate(_transpose(matrix, (1, 0))) @ q)
    q, _ = ar.do("linalg.qr", y, mode="reduced")
    reduced = _conjugate(_transpose(q, (1, 0))) @ matrix
    u_small, singular_values, vh = ar.do(
        "linalg.svd",
        reduced,
        full_matrices=False,
    )
    u = q @ u_small
    return u, singular_values, vh


def _svd(matrix, *, method="svd", max_rank=None, oversample=8, n_iter=2, seed=None):
    """Dispatch full, randomized, or iterative SVD without backend conversion."""

    method = str(method).strip().lower().replace("-", "_")
    if method in {"svd", "auto"}:
        return ar.do("linalg.svd", matrix, full_matrices=False)
    if method in {"rsvd", "svd:rand", "randomized", "isvd"}:
        return _randomized_svd(
            matrix,
            max_rank=max_rank,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed,
        )
    if method == "svds":
        backend = _backend_name(matrix)
        k = min(_host_shape(matrix)) - 1
        if max_rank is not None:
            k = min(k, int(max_rank))
        min_dim = min(_host_shape(matrix))
        if backend in {"builtins", "numpy"} and 1 <= k < min_dim - 1:
            try:
                from scipy.sparse.linalg import svds

                u, singular_values, vh = svds(
                    _host_array(matrix),
                    k=k,
                    which="LM",
                )
                order = np.argsort(singular_values)[::-1]
                return u[:, order], singular_values[order], vh[order]
            except ImportError:  # pragma: no cover - optional SciPy fallback
                pass
        return _randomized_svd(
            matrix,
            max_rank=max_rank,
            oversample=oversample,
            n_iter=max(2, int(n_iter)),
            seed=seed,
        )
    raise ValueError(
        "native Pauli compression method must be 'svd', 'rsvd', 'svds', "
        "'isvd', or 'auto'."
    )


def _compress_core_chain(
    cores,
    *,
    cutoff,
    cutoff_mode,
    max_bond,
    renorm=False,
    method="svd",
    oversample=8,
    n_iter=2,
    seed=None,
):
    """Left-orthogonalize then SVD-round a Pauli-label tensor train."""

    cores = list(_left_canonicalize_cores(cores))
    original_bonds = _core_bond_dimensions(cores)
    discarded_weight_squared = 0.0
    discarded_ranks = []
    bond_reports = []
    for site in range(len(cores) - 1, 0, -1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left, physical * right))
        u, singular_values, vh = _svd(
            matrix,
            method=method,
            max_rank=max_bond,
            oversample=oversample,
            n_iter=n_iter,
            seed=None if seed is None else int(seed) + site,
        )
        host_singular_values = _host_array(singular_values).real
        rank = _truncate_rank(
            host_singular_values,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
        )
        discarded = host_singular_values[rank:]
        discarded_weight_squared += float(np.vdot(discarded, discarded).real)
        discarded_ranks.append(int(len(host_singular_values) - rank))
        bond_reports.append(
            PauliBondCompressionReport(
                bond=site - 1,
                original_bond=left,
                final_bond=rank,
                discarded_rank=max(left - rank, 0),
                discarded_weight=float(np.linalg.norm(discarded)),
                largest_singular_value=(
                    float(host_singular_values[0])
                    if len(host_singular_values)
                    else 0.0
                ),
                singular_values=tuple(float(value) for value in host_singular_values),
            )
        )
        cores[site] = _transpose(
            _reshape(vh[:rank], (rank, physical, right)),
            (0, 2, 1),
        )
        kept_singular_values = _renormalize_singular_values(
            singular_values,
            rank,
            renorm,
            cutoff_mode,
        )
        transfer = u[:, :rank] * kept_singular_values
        cores[site - 1] = _einsum(
            "abp,bc->acp",
            cores[site - 1],
            transfer,
        )
    return (
        tuple(cores),
        original_bonds,
        tuple(reversed(discarded_ranks)),
        math.sqrt(discarded_weight_squared),
        tuple(reversed(bond_reports)),
    )


def _compress_core_chain_left(
    cores,
    *,
    cutoff,
    cutoff_mode,
    max_bond,
    renorm=False,
    method="svd",
    oversample=8,
    n_iter=2,
    seed=None,
):
    """SVD-round a Pauli-label tensor train into left-canonical form."""

    cores = list(_right_canonicalize_cores(cores))
    original_bonds = _core_bond_dimensions(cores)
    discarded_weight_squared = 0.0
    discarded_ranks = []
    bond_reports = []
    for site in range(len(cores) - 1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left * physical, right))
        u, singular_values, vh = _svd(
            matrix,
            method=method,
            max_rank=max_bond,
            oversample=oversample,
            n_iter=n_iter,
            seed=None if seed is None else int(seed) + site,
        )
        host_singular_values = _host_array(singular_values).real
        rank = _truncate_rank(
            host_singular_values,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
        )
        discarded = host_singular_values[rank:]
        discarded_weight_squared += float(np.vdot(discarded, discarded).real)
        discarded_ranks.append(int(len(host_singular_values) - rank))
        bond_reports.append(
            PauliBondCompressionReport(
                bond=site,
                original_bond=right,
                final_bond=rank,
                discarded_rank=max(right - rank, 0),
                discarded_weight=float(np.linalg.norm(discarded)),
                largest_singular_value=(
                    float(host_singular_values[0])
                    if len(host_singular_values)
                    else 0.0
                ),
                singular_values=tuple(float(value) for value in host_singular_values),
            )
        )
        cores[site] = _transpose(
            _reshape(u[:, :rank], (left, physical, rank)),
            (0, 2, 1),
        )
        kept_singular_values = _renormalize_singular_values(
            singular_values,
            rank,
            renorm,
            cutoff_mode,
        )
        transfer = kept_singular_values[:, None] * vh[:rank]
        cores[site + 1] = _einsum(
            "ab,bcp->acp",
            transfer,
            cores[site + 1],
        )
    return (
        tuple(cores),
        original_bonds,
        tuple(discarded_ranks),
        math.sqrt(discarded_weight_squared),
        tuple(bond_reports),
    )


def _compress_core_chain_flat(
    cores,
    *,
    cutoff,
    cutoff_mode,
    max_bond,
    renorm=False,
    method="svd",
    oversample=8,
    n_iter=2,
    seed=None,
):
    """Compress the disjoint left and right halves without canonicalizing."""

    cores = [_copy_array(core) for core in cores]
    original_bonds = _core_bond_dimensions(cores)
    discarded_weight_squared = 0.0
    bond_reports = {}
    middle = len(cores) // 2

    for site in range(len(cores) - 1, middle, -1):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left, physical * right))
        u, singular_values, vh = _svd(
            matrix,
            method=method,
            max_rank=max_bond,
            oversample=oversample,
            n_iter=n_iter,
            seed=None if seed is None else int(seed) + site,
        )
        host_singular_values = _host_array(singular_values).real
        rank = _truncate_rank(
            host_singular_values,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
        )
        discarded = host_singular_values[rank:]
        discarded_weight_squared += float(np.vdot(discarded, discarded).real)
        kept = _renormalize_singular_values(
            singular_values,
            rank,
            renorm,
            cutoff_mode,
        )
        root = ar.do("sqrt", kept)
        transfer = u[:, :rank] * root
        cores[site - 1] = _einsum("abp,bc->acp", cores[site - 1], transfer)
        cores[site] = _transpose(
            _reshape(root[:, None] * vh[:rank], (rank, physical, right)),
            (0, 2, 1),
        )
        bond_reports[site - 1] = PauliBondCompressionReport(
            bond=site - 1,
            original_bond=left,
            final_bond=rank,
            discarded_rank=max(left - rank, 0),
            discarded_weight=float(np.linalg.norm(discarded)),
            largest_singular_value=(
                float(host_singular_values[0]) if len(host_singular_values) else 0.0
            ),
            singular_values=tuple(float(value) for value in host_singular_values),
        )

    for site in range(0, middle):
        left, right, physical = _host_shape(cores[site])
        matrix = _reshape(_transpose(cores[site], (0, 2, 1)), (left * physical, right))
        u, singular_values, vh = _svd(
            matrix,
            method=method,
            max_rank=max_bond,
            oversample=oversample,
            n_iter=n_iter,
            seed=None if seed is None else int(seed) + site,
        )
        host_singular_values = _host_array(singular_values).real
        rank = _truncate_rank(
            host_singular_values,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
        )
        discarded = host_singular_values[rank:]
        discarded_weight_squared += float(np.vdot(discarded, discarded).real)
        kept = _renormalize_singular_values(
            singular_values,
            rank,
            renorm,
            cutoff_mode,
        )
        root = ar.do("sqrt", kept)
        cores[site] = _transpose(
            _reshape(u[:, :rank] * root, (left, physical, rank)),
            (0, 2, 1),
        )
        transfer = root[:, None] * vh[:rank]
        cores[site + 1] = _einsum("ab,bcp->acp", transfer, cores[site + 1])
        bond_reports[site] = PauliBondCompressionReport(
            bond=site,
            original_bond=right,
            final_bond=rank,
            discarded_rank=max(right - rank, 0),
            discarded_weight=float(np.linalg.norm(discarded)),
            largest_singular_value=(
                float(host_singular_values[0]) if len(host_singular_values) else 0.0
            ),
            singular_values=tuple(float(value) for value in host_singular_values),
        )

    return (
        tuple(cores),
        original_bonds,
        tuple(
            bond_reports[bond].discarded_rank
            for bond in sorted(bond_reports)
        ),
        math.sqrt(discarded_weight_squared),
        tuple(bond_reports[bond] for bond in sorted(bond_reports)),
    )


def _parse_term(term):
    """Return ``(coefficient, word, sites_or_none)`` from public input."""

    if isinstance(term, Mapping):
        coefficient = term.get("coefficient", term.get("weight", 1.0))
        word = term.get("paulis", term.get("word", term.get("string")))
        sites = term.get("sites", term.get("where", term.get("locations")))
        if word is None:
            raise ValueError("a Pauli term mapping needs 'paulis' or 'word'.")
        return coefficient, word, sites

    if not isinstance(term, (tuple, list)):
        raise TypeError(
            "Pauli terms must be (coefficient, word), "
            "(coefficient, sites, word), or mappings."
        )
    if len(term) == 2:
        coefficient, word = term
        return coefficient, word, None
    if len(term) != 3:
        raise ValueError("Pauli terms must contain two or three entries.")

    first, second, third = term
    if isinstance(second, str):
        # Compatibility with FirstDegreeMPO's (sites, paulis, coefficient)
        # spelling, while the new PauliMPO spelling remains coefficient first.
        return third, second, first
    return first, third, second


def _embed_word(word, sites, nsites):
    word = _normalize_word(word)
    sites = _normalize_sites(sites, nsites)
    if len(word) != len(sites):
        raise ValueError(
            f"word length {len(word)} does not match sites length {len(sites)}."
        )
    result = ["I"] * nsites
    for site, label in zip(sites, word):
        result[site] = label
    return "".join(result)


def _canonical_terms(terms, nsites):
    combined = {}
    for coefficient, word in terms:
        _check_scalar(coefficient)
        word = _normalize_word(word)
        if len(word) != nsites:
            raise ValueError(
                f"internal Pauli words must have length {nsites}, got {len(word)}."
            )
        if word in combined:
            combined[word] = _add(combined[word], coefficient)
        else:
            combined[word] = coefficient
    return tuple(
        (coefficient, word)
        for word, coefficient in sorted(combined.items())
        if not _host_zero(coefficient)
    )


class PauliMPO:
    """A sparse operator in the qubit Pauli basis with MPO operations.

    Parameters
    ----------
    nsites : int
        Number of qubits in the operator.
    terms : iterable
        Canonical full-length ``(coefficient, word)`` pairs.  Prefer
        :meth:`from_terms` for translated local strings.
    boundary : {"open", "periodic"}
        Boundary convention retained for constructors and metadata.

    Notes
    -----
    Numerical MPO compression generally leaves the sparse Pauli-word basis:
    the default ``compress`` object is a Quimb MPO.  This is unavoidable for
    a generic low-rank MPO, whose Pauli expansion can be exponentially dense.
    ``compress(basis="native")`` instead keeps Pauli physical legs in
    coefficient cores. Exact Pauli algebra remains available before either
    conversion boundary.

    :meth:`compress_pauli` provides the native alternative: it compresses a
    tensor train whose physical index is the four-element ``I/X/Y/Z`` basis.
    Its cores remain Pauli-basis tensors, while its virtual bonds can become
    dense numerical factors.  This is the same operator compression problem
    as MPO SVD rounding, but the physical legs never leave the Pauli basis.
    """

    def __init__(self, nsites, terms=(), *, boundary="open"):
        if not isinstance(nsites, Integral) or int(nsites) < 1:
            raise ValueError("nsites must be a positive integer.")
        self.nsites = int(nsites)
        self.boundary = _normalize_boundary(boundary)
        self._terms = _canonical_terms(terms, self.nsites)
        self._cores = None

    @classmethod
    def from_pauli_cores(cls, cores, *, boundary="open", copy=True):
        """Construct from native Pauli-label cores.

        Each core has shape ``(left_bond, right_bond, 4)`` with physical
        labels ordered ``I, X, Y, Z``.  The boundary virtual bonds must be
        singleton.  Cores are useful when a native SVD-compressed result must
        be passed between algorithms without expanding it into Pauli words.
        """

        boundary = _normalize_boundary(boundary)
        cores = _validate_pauli_cores(cores, copy=copy)
        result = cls.__new__(cls)
        result.nsites = len(cores)
        result.boundary = boundary
        result._terms = None
        result._cores = tuple(cores)
        return result

    @classmethod
    def from_terms(cls, nsites, terms, *, boundary="open", translate=True):
        """Construct from Pauli words and translated local stencils.

        ``(coefficient, "ZIZ")`` adds one full-chain term.  A shorter word
        is translated over all valid contiguous positions, e.g. ``(J, "ZZ")``
        gives ``J * sum_i Z_i Z_{i+1}``.  Explicit support can be supplied as
        ``(coefficient, (0, 3), "ZX")`` or with a mapping containing
        ``coefficient``, ``sites``, and ``paulis``/``word``.
        """

        if not isinstance(nsites, Integral) or int(nsites) < 1:
            raise ValueError("nsites must be a positive integer.")
        nsites = int(nsites)
        boundary = _normalize_boundary(boundary)
        if not isinstance(translate, bool):
            raise TypeError("translate must be boolean.")

        expanded = []
        for term in terms:
            coefficient, raw_word, sites = _parse_term(term)
            _check_scalar(coefficient)
            word = _normalize_word(raw_word)
            if sites is not None:
                expanded.append((coefficient, _embed_word(word, sites, nsites)))
                continue
            if len(word) == nsites:
                expanded.append((coefficient, word))
                continue
            if not translate:
                raise ValueError(
                    f"word {word!r} has length {len(word)}, expected {nsites}; "
                    "set translate=True or provide explicit sites."
                )
            if len(word) > nsites:
                raise ValueError(
                    f"local word length {len(word)} cannot exceed nsites={nsites}."
                )
            starts = (
                range(nsites)
                if boundary == "periodic"
                else range(nsites - len(word) + 1)
            )
            for start in starts:
                full = ["I"] * nsites
                for offset, label in enumerate(word):
                    full[(start + offset) % nsites] = label
                expanded.append((coefficient, "".join(full)))
        return cls(nsites, expanded, boundary=boundary)

    @classmethod
    def from_dense(
        cls,
        operator,
        *,
        boundary="open",
        atol=None,
        rtol=0.0,
        max_qubits=None,
    ):
        """Construct a PauliMPO by decomposing a dense qubit operator.

        This is intended for small local gates and exact dense checks.  The
        resulting object is still a sparse Pauli expansion, so its size can be
        as large as ``4**k`` for a generic ``k``-qubit matrix.
        """

        matrix, nqubits = _as_dense_matrix(operator)
        terms = decompose_pauli(
            matrix,
            atol=atol,
            rtol=rtol,
            max_qubits=max_qubits,
        )
        return cls.from_terms(
            nqubits,
            terms,
            boundary=boundary,
            translate=False,
        )

    from_matrix = from_dense

    @classmethod
    def identity(cls, nsites, *, coefficient=1.0, boundary="open"):
        """Construct ``coefficient * I``."""

        return cls.from_terms(
            nsites,
            [(coefficient, "I" * int(nsites))],
            boundary=boundary,
            translate=False,
        )

    @classmethod
    def zero(cls, nsites, *, boundary="open"):
        """Construct the zero operator."""

        return cls(nsites, (), boundary=boundary)

    @property
    def terms(self):
        """Canonical ``(coefficient, full_pauli_word)`` pairs."""

        if self._terms is None:
            self._terms = self._materialize_core_terms()
        return self._terms

    @property
    def pauli_terms(self):
        """Alias for :attr:`terms`."""

        return self.terms

    @property
    def num_terms(self):
        return len(self.terms)

    @property
    def support(self):
        """Sites on which at least one stored term is non-identity."""

        return tuple(
            site
            for site in range(self.nsites)
            if any(word[site] != "I" for _, word in self.terms)
        )

    def copy(self):
        if self._cores is not None:
            return type(self).from_pauli_cores(
                self._cores,
                boundary=self.boundary,
            )
        return type(self)(self.nsites, self.terms, boundary=self.boundary)

    def canonicalize(
        self,
        *,
        atol=0.0,
        rtol=0.0,
        native=False,
        center=None,
        inplace=False,
    ):
        """Return a canonically combined Pauli-basis representation.

        Equal full Pauli words are combined exactly, terms are ordered
        lexicographically, and optional tolerances remove numerically small
        coefficients.  This is the native sparse Pauli canonicalization; it
        is distinct from QR/SVD gauge canonicalization of the compiled MPO.
        Set both tolerances to zero (the default) to preserve backend scalar
        coefficients without a numerical host-side pruning decision.  Pass
        ``native=True`` to request QR canonicalization of Pauli-label cores;
        ``center`` and ``inplace`` then have the same meaning as in
        :meth:`canonicalize_native`.
        """

        atol = _validate_tolerance(atol, name="atol")
        rtol = _validate_tolerance(rtol, name="rtol")
        if atol is None:
            atol = 0.0
        if self._cores is not None or native:
            return self.canonicalize_native(center=center, inplace=inplace)
        host_coefficients = [
            coefficient
            for coefficient, _ in self.terms
            if _backend_name(coefficient) in {"builtins", "numpy"}
        ]
        scale = max(
            (float(abs(np.asarray(coefficient))) for coefficient in host_coefficients),
            default=0.0,
        )
        threshold = atol + rtol * scale
        if threshold == 0.0:
            return self.copy()
        terms = [
            (coefficient, word)
            for coefficient, word in self.terms
            if _backend_name(coefficient) not in {"builtins", "numpy"}
            or abs(np.asarray(coefficient)) > threshold
        ]
        result = type(self)(self.nsites, terms, boundary=self.boundary)
        if inplace:
            self._terms = result.terms
            return self
        return result

    simplify = canonicalize

    def __repr__(self):
        if self._cores is not None and self._terms is None:
            return (
                f"{type(self).__name__}(nsites={self.nsites}, "
                f"pauli_bonds={_core_bond_dimensions(self._cores)!r}, "
                f"boundary={self.boundary!r})"
            )
        return (
            f"{type(self).__name__}(nsites={self.nsites}, "
            f"num_terms={self.num_terms}, boundary={self.boundary!r})"
        )

    def _materialize_core_terms(self, *, max_terms=1_000_000):
        """Materialize native cores only when the Pauli-word expansion is small."""

        states = {(0, ""): 1.0 + 0.0j}
        for core in self._cores:
            next_states = {}
            for (left, word), coefficient in states.items():
                for right in range(core.shape[1]):
                    for physical, label in enumerate("IXYZ"):
                        local = core[left, right, physical]
                        if _host_zero(local):
                            continue
                        value = coefficient * local
                        key = (right, word + label)
                        next_states[key] = (
                            next_states[key] + value if key in next_states else value
                        )
            states = next_states
            if len(states) > max_terms:
                raise ValueError(
                    "the native Pauli-MPO expansion exceeds max_terms="
                    f"{max_terms}; use to_pauli_cores() or to_mpo() without "
                    "materializing every Pauli word."
                )
        terms = [
            (coefficient, word)
            for (right, word), coefficient in states.items()
            if right == 0 and not _host_zero(coefficient)
        ]
        return _canonical_terms(terms, self.nsites)

    def to_pauli_cores(self, *, copy=True):
        """Return cores with physical basis order ``I, X, Y, Z``.

        Sparse terms are converted to an exact term-path tensor train.  A
        compressed PauliMPO returns its native numerical cores directly.
        """

        cores = self._cores
        if cores is None:
            cores = _sparse_pauli_cores(self.terms, self.nsites)
        if copy:
            return tuple(_copy_array(core) for core in cores)
        return tuple(cores)

    @property
    def pauli_bond_dimensions(self):
        """Virtual dimensions of the native Pauli-label tensor train."""

        return _core_bond_dimensions(self.to_pauli_cores(copy=False))

    def canonicalize_native(self, *, center=None, inplace=False):
        """QR-canonicalize the native Pauli-label tensor train.

        The physical index remains exactly the four-element Pauli basis.  The
        returned object is core-backed, so its Pauli-word expansion is not
        materialized unless a word-oriented method such as ``terms`` is read.
        """

        if center is None:
            center = self.nsites - 1
        if not isinstance(center, Integral) or not 0 <= int(center) < self.nsites:
            raise ValueError(f"center must lie in the range 0..{self.nsites - 1}.")
        cores = _canonicalize_core_chain(
            self.to_pauli_cores(copy=True),
            int(center),
        )
        if inplace:
            self._cores = cores
            self._terms = None
            return self
        return type(self).from_pauli_cores(cores, boundary=self.boundary)

    canonize_native = canonicalize_native

    def compress_pauli(
        self,
        *,
        max_bond=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        form=None,
        method="svd",
        oversample=8,
        n_iter=2,
        seed=None,
        renorm=False,
        inplace=False,
        return_report=False,
    ):
        """Compress while retaining the native Pauli physical basis.

        This performs tensor-train SVD rounding on coefficient cores with
        physical index ``I/X/Y/Z``.  It never constructs a dense ``2**L`` by
        ``2**L`` operator and never changes the local physical basis.  The
        approximation error is the discarded coefficient-tensor singular
        weight reported in :class:`PauliCompressionReport`.

        ``form`` follows Quimb's 1D convention for canonical forms: ``None``
        and ``"right"``
        produce a right-canonical result with center at site zero,
        ``"left"`` centers it at the last site, and an integer centers the
        result at that site. ``method`` can be ``"svd"``, ``"rsvd"`` or
        ``"svd:rand"`` for randomized SVD, and ``"svds"``/``"isvd"`` for
        iterative-style truncation. ``oversample``, ``n_iter``, and ``seed``
        control the randomized methods.
        """

        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        cutoff = _validate_tolerance(cutoff, name="cutoff")
        if cutoff is None:
            cutoff = 0.0
        cutoff_mode = str(cutoff_mode).strip().lower().replace("-", "_")
        if cutoff_mode not in {"rel", "abs", "sum1", "rsum1", "sum2", "rsum2"}:
            raise ValueError(
                "cutoff_mode must be one of 'rel', 'abs', 'sum1', 'rsum1', "
                "'sum2', or 'rsum2'."
            )
        method = str(method).strip().lower().replace("-", "_")
        if method == "auto":
            method = "svd"
        if method not in {"svd", "rsvd", "svd:rand", "randomized", "svds", "isvd"}:
            raise ValueError(
                "native Pauli compression method must be 'svd', 'rsvd', "
                "'svds', 'isvd', or 'auto'."
            )
        if not isinstance(oversample, Integral) or int(oversample) < 0:
            raise ValueError("oversample must be a nonnegative integer.")
        if not isinstance(n_iter, Integral) or int(n_iter) < 0:
            raise ValueError("n_iter must be a nonnegative integer.")
        if not isinstance(renorm, (bool, Integral)):
            raise TypeError("renorm must be False, True, 1, or 2.")
        if renorm not in {False, True, 1, 2}:
            raise ValueError("renorm must be False, True, 1, or 2.")
        if form is None:
            form = "right"
        if isinstance(form, Integral):
            if not 0 <= int(form) < self.nsites:
                raise ValueError(f"form must lie in the range 0..{self.nsites - 1}.")
            form = int(form)
        else:
            form = str(form).strip().lower()
            if form not in {"left", "right", "flat"}:
                raise ValueError(
                    "form must be None, 'left', 'right', 'flat', or an integer."
                )
        cores = self.to_pauli_cores(copy=True)
        original_bonds = _core_bond_dimensions(cores)
        coefficient_cutoff = _coefficient_cutoff(cutoff, cutoff_mode, self.nsites)
        physical_scale = 2.0 ** (0.5 * self.nsites)
        if form == "left":
            compressed, _, discarded_ranks, discarded_weight, bond_reports = (
                _compress_core_chain_left(
                    cores,
                    cutoff=coefficient_cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=max_bond,
                    renorm=renorm,
                    method=method,
                    oversample=int(oversample),
                    n_iter=int(n_iter),
                    seed=seed,
                )
            )
        elif form == "flat":
            compressed, _, discarded_ranks, discarded_weight, bond_reports = (
                _compress_core_chain_flat(
                    cores,
                    cutoff=coefficient_cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=max_bond,
                    renorm=renorm,
                    method=method,
                    oversample=int(oversample),
                    n_iter=int(n_iter),
                    seed=seed,
                )
            )
        else:
            # Quimb's default/right form is a right-to-left sweep. For an
            # integer center, round all bonds then move the orthogonality
            # center to the requested site, matching the public form contract.
            compressed, _, discarded_ranks, discarded_weight, bond_reports = _compress_core_chain(
                cores,
                cutoff=coefficient_cutoff,
                cutoff_mode=cutoff_mode,
                max_bond=max_bond,
                renorm=renorm,
                method=method,
                oversample=int(oversample),
                n_iter=int(n_iter),
                seed=seed,
            )
            if isinstance(form, Integral) and form:
                compressed = _canonicalize_core_chain(compressed, form)
        discarded_weight *= physical_scale
        bond_reports = tuple(
            replace(
                bond_report,
                discarded_weight=bond_report.discarded_weight * physical_scale,
                largest_singular_value=(
                    bond_report.largest_singular_value * physical_scale
                ),
                singular_values=tuple(
                    value * physical_scale for value in bond_report.singular_values
                ),
            )
            for bond_report in bond_reports
        )
        result = type(self).from_pauli_cores(compressed, boundary=self.boundary)
        report = PauliCompressionReport(
            original_bond_dimensions=original_bonds,
            final_bond_dimensions=_core_bond_dimensions(compressed),
            discarded_singular_weight=discarded_weight,
            discarded_ranks=discarded_ranks,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            form=form,
            renorm=renorm,
            method=method,
            bond_reports=bond_reports,
        )
        if inplace:
            self._cores = result.to_pauli_cores(copy=False)
            self._terms = None
            result = self
        if return_report:
            return result, report
        return result

    compress_native = compress_pauli

    def _combine(self, other, sign=1):
        if not isinstance(other, PauliMPO):
            return NotImplemented
        if self.nsites != other.nsites:
            raise ValueError("PauliMPO operators must have the same nsites.")
        if self._cores is not None or other._cores is not None:
            cores = _native_direct_sum(
                self.to_pauli_cores(copy=False),
                other.to_pauli_cores(copy=False),
                sign=sign,
            )
            return type(self).from_pauli_cores(cores, boundary=self.boundary)
        terms = list(self.terms)
        terms.extend(
            (_multiply(sign, coefficient), word) for coefficient, word in other.terms
        )
        return type(self)(self.nsites, terms, boundary=self.boundary)

    def __add__(self, other):
        return self._combine(other)

    def __sub__(self, other):
        return self._combine(other, sign=-1)

    def __neg__(self):
        return self * -1

    def scale(self, coefficient):
        _check_scalar(coefficient, name="coefficient")
        if self._cores is not None:
            cores = list(self.to_pauli_cores(copy=False))
            cores[0] = _multiply(coefficient, cores[0])
            return type(self).from_pauli_cores(cores, boundary=self.boundary)
        return type(self)(
            self.nsites,
            [(_multiply(coefficient, value), word) for value, word in self.terms],
            boundary=self.boundary,
        )

    def __mul__(self, coefficient):
        if isinstance(coefficient, PauliMPO):
            return NotImplemented
        return self.scale(coefficient)

    def __rmul__(self, coefficient):
        return self.scale(coefficient)

    def product(self, other):
        """Return the exact Pauli product ``self @ other``."""

        if not isinstance(other, PauliMPO):
            return NotImplemented
        if self.nsites != other.nsites:
            raise ValueError("PauliMPO operators must have the same nsites.")
        if self._cores is not None or other._cores is not None:
            cores = _native_product(
                self.to_pauli_cores(copy=False),
                other.to_pauli_cores(copy=False),
            )
            return type(self).from_pauli_cores(cores, boundary=self.boundary)
        result = {}
        for left_coefficient, left_word in self.terms:
            for right_coefficient, right_word in other.terms:
                phase = 1
                word = []
                for left, right in zip(left_word, right_word):
                    label, local_phase = _PAULI_PRODUCT[(left, right)]
                    word.append(label)
                    phase *= local_phase
                value = _multiply(
                    _multiply(left_coefficient, right_coefficient),
                    phase,
                )
                word = "".join(word)
                result[word] = _add(result[word], value) if word in result else value
        return type(self)(
            self.nsites,
            [(value, word) for word, value in result.items()],
            boundary=self.boundary,
        )

    def __matmul__(self, other):
        return self.product(other)

    def commutator(self, other):
        return self @ other - other @ self

    def dagger(self):
        """Return the Hermitian adjoint in the Pauli basis."""

        if self._cores is not None:
            return type(self).from_pauli_cores(
                tuple(_conjugate(core) for core in self.to_pauli_cores(copy=False)),
                boundary=self.boundary,
            )
        return type(self)(
            self.nsites,
            [(_conjugate(value), word) for value, word in self.terms],
            boundary=self.boundary,
        )

    adjoint = dagger

    def conjugate(self):
        """Return elementwise complex conjugation in the Pauli basis."""

        if self._cores is not None:
            cores = []
            signs = _array_like([1.0, 1.0, -1.0, 1.0], self._cores[0])
            for core in self.to_pauli_cores(copy=False):
                cores.append(_multiply(_conjugate(core), signs))
            return type(self).from_pauli_cores(cores, boundary=self.boundary)
        terms = []
        for value, word in self.terms:
            y_count = word.count("Y")
            sign = -1 if y_count % 2 else 1
            terms.append((_multiply(_conjugate(value), sign), word))
        return type(self)(self.nsites, terms, boundary=self.boundary)

    def transpose(self):
        """Return the matrix transpose in the Pauli basis."""

        if self._cores is not None:
            signs = _array_like([1.0, 1.0, -1.0, 1.0], self._cores[0])
            return type(self).from_pauli_cores(
                tuple(_multiply(core, signs) for core in self.to_pauli_cores(copy=False)),
                boundary=self.boundary,
            )
        terms = []
        for value, word in self.terms:
            sign = -1 if word.count("Y") % 2 else 1
            terms.append((_multiply(value, sign), word))
        return type(self)(self.nsites, terms, boundary=self.boundary)

    def trace(self, *, normalized=False):
        """Return the exact (unnormalized by default) operator trace."""

        if self._cores is not None:
            return _native_trace(
                self.to_pauli_cores(copy=False),
                normalized=normalized,
            )
        result = None
        identity = "I" * self.nsites
        factor = 1.0 if normalized else float(2**self.nsites)
        for coefficient, word in self.terms:
            if word == identity:
                value = _multiply(coefficient, factor)
                result = value if result is None else _add(result, value)
        return 0.0 if result is None else result

    def inner(self, other, *, normalized=False):
        """Return the Hilbert--Schmidt inner product ``Tr(self† other)``."""

        if not isinstance(other, PauliMPO):
            return NotImplemented
        if self.nsites != other.nsites:
            raise ValueError("PauliMPO operators must have the same nsites.")
        if self._cores is not None or other._cores is not None:
            return _native_inner(
                self.to_pauli_cores(copy=False),
                other.to_pauli_cores(copy=False),
                normalized=normalized,
            )
        right = {word: coefficient for coefficient, word in other.terms}
        result = None
        factor = 1.0 if normalized else float(2**self.nsites)
        for coefficient, word in self.terms:
            if word not in right:
                continue
            value = _multiply(
                _multiply(_conjugate(coefficient), right[word]),
                factor,
            )
            result = value if result is None else _add(result, value)
        return 0.0 if result is None else result

    def norm(self, *, normalized=False):
        value = self.inner(self, normalized=normalized)
        return ar.do("sqrt", value)

    def partial_trace(self, sites, *, keep=False, normalized=False):
        """Trace out sites and return the reduced Pauli operator.

        By default ``sites`` are traced out.  With ``keep=True``, ``sites``
        instead names the sites retained in the reduced operator.  The
        default is the unnormalized partial trace, so every identity on a
        traced qubit contributes a factor of two.  Tracing all sites returns
        a scalar.
        """

        if not isinstance(keep, bool):
            raise TypeError("keep must be boolean.")
        sites = _normalize_sites(sites, self.nsites)
        if keep:
            kept = sites
            traced = tuple(site for site in range(self.nsites) if site not in kept)
        else:
            traced = sites
            kept = tuple(site for site in range(self.nsites) if site not in traced)
        factor = 1.0 if normalized else float(2 ** len(traced))
        if self._cores is not None:
            reduced = _native_partial_trace(
                self.to_pauli_cores(copy=False),
                traced,
                normalized=normalized,
            )
            if not kept:
                return reduced
            return type(self).from_pauli_cores(
                reduced,
                boundary=self.boundary,
            )
        if not kept:
            result = None
            for coefficient, word in self.terms:
                if any(word[site] != "I" for site in traced):
                    continue
                value = _multiply(coefficient, factor)
                result = value if result is None else _add(result, value)
            return 0.0 if result is None else result

        terms = []
        for coefficient, word in self.terms:
            if any(word[site] != "I" for site in traced):
                continue
            reduced = "".join(word[site] for site in kept)
            terms.append((_multiply(coefficient, factor), reduced))
        return type(self)(len(kept), terms, boundary=self.boundary)

    def apply_gate(
        self,
        gate,
        where,
        *,
        mode="conjugate",
        atol=None,
        rtol=0.0,
        max_qubits=None,
        inplace=False,
    ):
        """Apply a dense local gate in the native Pauli basis.

        ``mode="conjugate"`` maps ``P`` to ``G P G†`` and is the natural
        Schrödinger-picture transformation of an operator.  The
        ``"heisenberg"`` mode maps ``P`` to ``G† P G``.  ``"left"`` and
        ``"right"`` implement ``G P`` and ``P G`` respectively.  The dense
        gate is Pauli-decomposed locally, so no full ``2**nsites`` matrix is
        formed.  ``where`` preserves the tensor-product order of the gate and
        may contain non-contiguous sites.

        Parameters
        ----------
        gate : array-like or PauliMPO
            A ``2**k`` by ``2**k`` local operator acting on ``len(where)``
            qubits.
        where : int or iterable of int
            Sites on which the gate acts.
        mode : {"conjugate", "heisenberg", "left", "right"}
            Local operator transformation to perform.
        inplace : bool
            If true, replace this object's canonical terms and return ``self``.
            The default returns a new PauliMPO.
        """

        where = _normalize_where(where, self.nsites)
        if isinstance(gate, PauliMPO):
            if gate.nsites != len(where):
                raise ValueError(
                    f"gate has {gate.nsites} qubits but where has {len(where)} sites."
                )
            gate = gate.to_dense()
        gate, nqubits = _as_dense_matrix(gate)
        if nqubits != len(where):
            raise ValueError(
                f"gate acts on {nqubits} qubit(s) but where={where!r} has "
                f"{len(where)} site(s)."
            )
        mode = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "conjugate": "conjugate",
            "similarity": "conjugate",
            "heisenberg": "heisenberg",
            "adjoint": "heisenberg",
            "left": "left",
            "pre": "left",
            "right": "right",
            "post": "right",
        }
        try:
            mode = aliases[mode]
        except KeyError as exc:
            raise ValueError(
                "mode must be one of 'conjugate', 'heisenberg', 'left', or 'right'."
            ) from exc
        if mode == "conjugate":
            transformed_gate = gate
            post = gate.conj().T
        elif mode == "heisenberg":
            transformed_gate = gate.conj().T
            post = gate
        elif mode == "left":
            transformed_gate = gate
            post = None
        else:
            transformed_gate = None
            post = gate

        if self._cores is not None:
            local_terms = decompose_pauli(
                gate,
                atol=atol,
                rtol=rtol,
                max_qubits=max_qubits,
            )
            embedded_terms = []
            for local_coefficient, local_word in local_terms:
                full_word = ["I"] * self.nsites
                for site, label in zip(where, local_word):
                    full_word[site] = label
                embedded_terms.append((local_coefficient, "".join(full_word)))
            global_gate = type(self)(self.nsites, embedded_terms, boundary=self.boundary)
            if self._cores:
                like = self._cores[0]
                global_gate = type(self).from_pauli_cores(
                    tuple(_array_like(core, like) for core in global_gate.to_pauli_cores()),
                    boundary=self.boundary,
                )
            if mode == "conjugate":
                result = global_gate @ self @ global_gate.dagger()
            elif mode == "heisenberg":
                result = global_gate.dagger() @ self @ global_gate
            elif mode == "left":
                result = global_gate @ self
            else:
                result = self @ global_gate
            if inplace:
                self._cores = result.to_pauli_cores(copy=False)
                self._terms = None
                return self
            return result

        accumulated = {}
        for coefficient, word in self.terms:
            local_word = "".join(word[site] for site in where)
            local_operator = _kron_pauli(local_word)
            if mode in {"conjugate", "heisenberg"}:
                local_operator = transformed_gate @ local_operator @ post
            elif mode == "left":
                local_operator = transformed_gate @ local_operator
            else:
                local_operator = local_operator @ post
            local_terms = decompose_pauli(
                local_operator,
                atol=atol,
                rtol=rtol,
                max_qubits=max_qubits,
            )
            for local_coefficient, replacement in local_terms:
                new_word = list(word)
                for site, label in zip(where, replacement):
                    new_word[site] = label
                new_word = "".join(new_word)
                value = _multiply(coefficient, local_coefficient)
                if new_word in accumulated:
                    accumulated[new_word] = _add(accumulated[new_word], value)
                else:
                    accumulated[new_word] = value
        result = type(self)(
            self.nsites,
            [(value, word) for word, value in accumulated.items()],
            boundary=self.boundary,
        )
        if inplace:
            self._terms = result.terms
            self._cores = None
            return self
        return result

    def apply_channel(
        self,
        kraus,
        where,
        *,
        picture="heisenberg",
        weights=None,
        atol=None,
        rtol=0.0,
        max_qubits=None,
        inplace=False,
    ):
        """Apply a local Kraus channel in the Pauli basis.

        In the default Heisenberg picture, ``P`` is mapped to
        ``sum_a K_a† P K_a``.  Use ``picture="schrodinger"`` for
        ``sum_a K_a P K_a†``.  ``weights`` optionally multiplies each Kraus
        contribution and is useful for explicitly represented probabilistic
        branches.
        """

        try:
            kraus_ndim = int(kraus.ndim)
        except (AttributeError, TypeError):
            try:
                kraus_ndim = int(np.ndim(kraus))
            except TypeError:
                kraus_ndim = None
        if kraus_ndim == 2:
            kraus = (kraus,)
        else:
            try:
                kraus = tuple(kraus)
            except TypeError as exc:
                raise TypeError("kraus must be a matrix or an iterable of matrices.") from exc
        if not kraus:
            raise ValueError("kraus must contain at least one matrix.")
        if weights is None:
            weights = (1.0,) * len(kraus)
        else:
            try:
                weights = tuple(weights)
            except TypeError as exc:
                raise TypeError("weights must be an iterable of scalars.") from exc
            if len(weights) != len(kraus):
                raise ValueError("weights and kraus must have the same length.")
        picture = str(picture).strip().lower().replace("-", "_")
        if picture not in {"heisenberg", "schrodinger"}:
            raise ValueError("picture must be 'heisenberg' or 'schrodinger'.")
        result = type(self).zero(self.nsites, boundary=self.boundary)
        mode = "heisenberg" if picture == "heisenberg" else "conjugate"
        for matrix, weight in zip(kraus, weights):
            _check_scalar(weight, name="weight")
            result = result + self.apply_gate(
                matrix,
                where,
                mode=mode,
                atol=atol,
                rtol=rtol,
                max_qubits=max_qubits,
            ).scale(weight)
        if inplace:
            self._terms = result.terms
            self._cores = None
            return self
        return result

    def _semantic_mpo(self):
        if self._cores is not None:
            arrays = tuple(
                _einsum(
                    "abp,pij->abij",
                    _complexify(core),
                    _array_like(
                        np.stack(tuple(_PAULI_MATRICES[label] for label in "IXYZ")),
                        _complexify(core),
                    ),
                )
                for core in self._cores
            )
            return FirstDegreeMPO(arrays)
        product_terms = []
        identity_coefficient = None
        for coefficient, word in self.terms:
            support = tuple(site for site, label in enumerate(word) if label != "I")
            if not support:
                identity_coefficient = (
                    coefficient
                    if identity_coefficient is None
                    else _add(identity_coefficient, coefficient)
                )
                continue
            product_terms.append(
                MPOProductTerm.from_pauli(
                    support,
                    "".join(word[site] for site in support),
                    coefficient=coefficient,
                )
            )

        if product_terms:
            semantic = FirstDegreeMPO.from_local_terms(
                self.nsites,
                product_terms,
                phys_dim=2,
                # Keep the exact Pauli paths independent at this boundary.
                # The shared-channel automaton has a known ambiguity when
                # translated copies of the same one-site Pauli occur beside
                # longer terms; numerical MPO compression follows below.
                share_channels=False,
            )
        else:
            semantic = FirstDegreeMPO.identity(self.nsites, 2)
            return semantic.scale(
                0.0 if identity_coefficient is None else identity_coefficient
            )
        if identity_coefficient is not None:
            semantic = semantic.add(
                FirstDegreeMPO.identity(self.nsites, 2).scale(identity_coefficient)
            )
        return semantic

    def to_mpo(self):
        """Compile to a Quimb ``MatrixProductOperator``."""

        mpo = self._semantic_mpo().to_mpo()
        mpo.pepsy_pauli_mpo = self.copy()
        return mpo

    def to_semantic_mpo(self):
        """Compile to Pepsy's semantic :class:`FirstDegreeMPO`."""

        return self._semantic_mpo()

    def to_dense(self):
        """Materialize a dense matrix; intended for small-system checks."""

        return self.to_mpo().to_dense()

    def expectation(self, mps, *, contraction_opt=None):
        """Evaluate the normalized ``<mps|self|mps>`` contraction."""

        return self._semantic_mpo().expectation(
            mps,
            contraction_opt=contraction_opt,
        )

    def apply(self, mps, *, method="direct", inplace=False, **compress_opts):
        """Apply the operator to an MPS through Quimb's MPO-MPS path."""

        return self._semantic_mpo().apply_to_mps(
            mps,
            method=method,
            inplace=inplace,
            **compress_opts,
        )

    def compress_numerical(self, **kwargs):
        """Compress the compiled MPO using Quimb's numerical sweep."""

        return self._semantic_mpo().compress_numerical(**kwargs)

    def compress(self, *, basis="mpo", **kwargs):
        """Compress in the requested MPO or native Pauli representation.

        ``basis="mpo"`` preserves the historical Quimb-compatible numerical
        compression boundary.  ``basis="pauli"`` (or ``"native"``) keeps
        physical indices in the ``I/X/Y/Z`` basis and calls
        :meth:`compress_pauli`.

        The default result is a Quimb MPO because generic numerical
        compression does not preserve a sparse Pauli expansion.  The native
        branch returns a core-backed PauliMPO instead.
        """

        basis = str(basis).strip().lower()
        if basis in {"pauli", "native"}:
            return self.compress_pauli(**kwargs)
        if basis != "mpo":
            raise ValueError("basis must be 'mpo', 'pauli', or 'native'.")
        return self.compress_numerical(**kwargs)

    def compress_fixed_rank(self, max_bond, *, return_report=False):
        """Use Pepsy's backend-differentiable fixed-rank MPO compression."""

        return self._semantic_mpo().compress_fixed_rank(
            max_bond,
            return_report=return_report,
        )

    def compress_to_bond(self, chi, **kwargs):
        """Compress to a final MPO bond cap through Pepsy's MPO API."""

        return self._semantic_mpo().compress_to_bond(chi, **kwargs)

    def exp(self, step=None, *, dt=None, **kwargs):
        """Build an exponential; the result leaves the sparse Pauli basis."""

        return self._semantic_mpo().exp(step, dt=dt, **kwargs)

    def time_evolution(self, dt, **kwargs):
        """Build ``exp(-1j * dt * self)`` through the existing MPO engine."""

        return self._semantic_mpo().time_evolution(dt, **kwargs)

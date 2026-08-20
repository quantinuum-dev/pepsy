"""Semantic finite-chain MPOs for higher-order operator construction.

This module is the first layer above ordinary Quimb MPO tensors needed by the
higher-order exponential construction of Van Damme et al. It deliberately
keeps the virtual-level history separate from the tensor data.  Ordinary
Quimb MPOs remain the compiled interchange format, while :class:`FirstDegreeMPO`
retains enough structure for exact algebra and history compression.

The implementation is finite-chain and exact at this stage.  The extensive
Taylor construction is assembled from local MPO blocks and virtual channels;
it never forms a global operator matrix.  Numerical bond truncation and native
Symmray compilation remain separate follow-up layers.

Design contract
---------------
``FirstDegreeMPO`` is the semantic construction object.  Its virtual-bond
histories are part of the data model because the paper's Algorithms 1--4 act
on those histories, not just on the numerical MPO entries.  ``to_mpo()`` is
the compatibility boundary: it produces an ordinary Quimb MPO for existing
contraction and MPS-application code, while retaining a copy of the semantic
object on the compiled MPO.

The exact paths only use local tensor operations and exact equality checks.
``mode="algorithm4"`` is deliberately separate because Algorithm 4 changes the
analytical history representation even though it does not use an SVD cutoff.
This module currently targets ordinary NumPy/Autoray-compatible tensors and
finite open chains; fermionic/Symmray compilation is a future backend layer.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from math import factorial
from numbers import Integral
import warnings

import autoray as ar
import numpy as np

from .mpo_automaton import (
    MPOAutomaton,
    _as_backend,
    _backend_reference,
    _backend_name,
    _multiply_scalar,
)

__all__ = [
    "MPOParameter",
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOCompressionReport",
    "MPONumericalCompressionReport",
    "MPODifferentiableCompressionReport",
    "FirstDegreeMPO",
    "CompiledMPOExp",
    "CompiledMPOEvolution",
    "MPOBasis",
]


# Dense virtual transfer maps are much cheaper than repeated backend scatter
# updates for the small-to-medium history bonds normally produced by Taylor
# orders two through five.  Keep a conservative escape hatch for very large
# bonds, where a dense ``new_bond x old_bond`` map would use more memory than
# the original gather/scatter implementation.
_MAX_HISTORY_TRANSFER_ELEMENTS = 4_000_000

# Keep the fused coefficient bank bounded.  Very long-range or high-rank
# bases can have many terms relative to their small sparse slot count; those
# cases retain the exact grouped scatter fallback rather than allocating a
# mostly-zero ``num_terms x left x right x phys x phys`` bank.
_MAX_FUSED_SLOT_BANK_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class MPOLevelToken:
    """One symbolic level in a virtual-state history.

    ``level`` follows the paper's first-degree convention: ``1`` and ``3``
    denote the two identity rails and ``2`` denotes an active operator
    channel.  ``payload`` distinguishes independent operator channels that
    happen to have the same level number.  The level number is intentionally
    small and symbolic; backend charge information belongs on ``MPOLevel``.
    """

    level: int
    payload: Hashable = None

    def __post_init__(self):
        if self.level not in (1, 2, 3):
            raise ValueError("MPO level tokens must have level 1, 2, or 3.")
        if self.payload is not None and not isinstance(self.payload, Hashable):
            raise TypeError("MPO level token payload must be hashable or None.")


@dataclass(frozen=True)
class MPOLevel:
    """A virtual state together with its symbolic history."""

    label: Hashable
    history: tuple[MPOLevelToken, ...]
    charge: object = None

    def __post_init__(self):
        if not isinstance(self.label, Hashable):
            raise TypeError("MPO level labels must be hashable.")
        object.__setattr__(self, "history", tuple(self.history))
        if not self.history:
            raise ValueError("MPO level history must contain at least one token.")
        if not all(isinstance(token, MPOLevelToken) for token in self.history):
            raise TypeError("MPO level history must contain MPOLevelToken values.")


@dataclass(frozen=True)
class MPOProductTerm:
    """A factorized local product term used to build a first-degree MPO.

    ``sites`` and ``operators`` describe only the non-identity factors.  A
    string operator can be supplied for fermion-compatible automaton routes,
    but this higher-order implementation does not enable native fermionic
    compilation yet.  ``charge`` is carried as metadata for a future
    block-sparse backend and is not interpreted by this class.
    """

    sites: tuple[int, ...]
    operators: tuple[object, ...]
    coefficient: object = 1.0
    string_operators: tuple[object, ...] | None = None
    charge: object = None

    @classmethod
    def from_pauli(
        cls,
        sites,
        paulis,
        *,
        coefficient=1.0,
        string_paulis=None,
        charge=None,
    ):
        """Construct a product term from labels such as ``"ZXY"``.

        ``sites`` lists the non-identity support positions. Gaps receive
        identities unless ``string_paulis`` supplies one label per gap.
        """
        return cls(
            sites=sites,
            operators=paulis,
            coefficient=coefficient,
            string_operators=string_paulis,
            charge=charge,
        )

    def __post_init__(self):
        sites = tuple(int(site) for site in self.sites)
        operators = _normalize_operator_sequence(self.operators, name="operators")
        if not sites or len(sites) != len(operators):
            raise ValueError("sites and operators must be non-empty and aligned.")
        if any(left >= right for left, right in zip(sites, sites[1:])):
            raise ValueError("term sites must be strictly increasing.")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "operators", operators)
        if self.string_operators is not None:
            object.__setattr__(
                self,
                "string_operators",
                _normalize_operator_sequence(
                    self.string_operators,
                    name="string_operators",
                ),
            )


@dataclass(frozen=True)
class MPOCompressionReport:
    """Diagnostics returned by exact or analytical history compression.

    ``exact`` distinguishes scalar gauge eliminations from Algorithm 4's
    order-controlled analytical approximation.  ``merges`` contains stable,
    human-readable provenance records rather than backend tensor objects, so
    reports can be logged or serialized by callers.  The report describes the
    semantic history stage; it does not describe a later Quimb SVD/truncation.
    """

    method: str
    exact: bool
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    merged_channels: int
    merges: tuple[Mapping[str, object], ...] = ()
    skipped_candidates: int = 0


@dataclass(frozen=True)
class MPONumericalCompressionReport:
    """Report the numerical compression boundary delegated to Quimb.

    The semantic history representation cannot survive a numerical bond
    truncation, so this report deliberately describes the compiled Quimb MPO
    only.  When requested, the operator error is measured as a tensor-network
    Frobenius norm of the difference between the pre- and post-compression
    MPOs.  This avoids forming a global dense matrix, but remains an optional
    contraction because it can be expensive for a large MPO.
    """

    method: str
    form: object
    max_bond: int | None
    cutoff: float
    cutoff_mode: str
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    truncated: bool
    truncation_error: object = None
    operator_frobenius_error: object = None
    operator_frobenius_relative_error: object = None
    error_estimator: str | None = None


@dataclass(frozen=True)
class MPODifferentiableCompressionReport:
    """Report a fixed-rank, autodiff-friendly MPO compression.

    Unlike cutoff-based compression, fixed-rank TT-SVD never changes its
    selected rank as a function of singular values.  The resulting map is
    therefore differentiable through the backend SVD away from the usual
    repeated-singular-value singularities.
    """

    method: str
    max_bond: int
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    truncated: bool
    differentiable: bool = True


def _check_scalar(value, *, name):
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        ndim = np.ndim(value)
    if ndim != 0:
        raise TypeError(f"{name} must be scalar, got ndim={ndim}.")


def _resolve_exp_step(step, dt):
    """Resolve the canonical ``step`` and legacy ``dt`` spellings.

    Public exponential methods use ``step`` in their documentation.  The
    optional ``dt`` keyword is accepted only as a compatibility spelling so
    existing callers can migrate without changing numerical semantics.
    """
    if step is not None and dt is not None:
        raise TypeError("pass either step or dt, not both.")
    if step is None:
        if dt is None:
            raise TypeError("exp requires a scalar step.")
        step = dt
    _check_scalar(step, name="step")
    return step


def _fixed_rank_svd(matrix):
    """Run the configured thin SVD, enabling the safe Torch VJP when needed."""
    if _backend_name(matrix) != "torch":
        return ar.do("linalg.svd", matrix, full_matrices=False)

    # Native Torch's singular-vector VJP is undefined at repeated or zero
    # singular values, which are common in MPO blocks. Reuse Pepsy's one
    # public Torch policy for this path and scope the registration so an MPO
    # compression does not silently change the caller's global backend mode.
    from pepsy.backends import (  # pylint: disable=import-outside-toplevel
        TorchLinalgConfig,
        get_torch_linalg_config,
    )

    current = get_torch_linalg_config()
    if current is not None and current.stabilized:
        return ar.do("linalg.svd", matrix)
    mode = "complex" if getattr(matrix.dtype, "is_complex", False) else "real"
    config = TorchLinalgConfig(stabilized=True, mode=mode)
    with config.activated():
        return ar.do("linalg.svd", matrix)


_UNSET = object()


@dataclass(frozen=True)
class MPOParameter:
    """A named or positional scalar coefficient resolved by :class:`MPOBasis`.

    The object is intentionally only a reference to a coefficient.  The
    current backend scalar is supplied to :meth:`MPOBasis.build`, which means
    Torch/JAX values are never copied into a topology cache and remain in the
    autodiff graph.

    Parameters
    ----------
    name : hashable
        Mapping key, or integer positional index for sequence-like parameter
        containers.
    default : scalar, optional
        Value used when ``build`` receives no parameter container.  Omitting
        it makes the parameter required.
    """

    name: Hashable
    default: object = _UNSET

    def __post_init__(self):
        if not isinstance(self.name, Hashable):
            raise TypeError("MPO parameter names must be hashable.")

    def resolve(self, parameters):
        """Resolve this reference against a mapping or positional container."""
        if parameters is None:
            if self.default is _UNSET:
                raise KeyError(
                    f"missing MPO parameter {self.name!r}; pass it to build()."
                )
            value = self.default
        elif isinstance(parameters, Mapping):
            try:
                value = parameters[self.name]
            except KeyError as exc:
                raise KeyError(
                    f"missing MPO parameter {self.name!r}; pass it to build()."
                ) from exc
        else:
            if not isinstance(self.name, Integral):
                raise TypeError(
                    "non-integer MPO parameter names require a mapping; got "
                    f"{type(parameters).__name__}."
                )
            try:
                value = parameters[int(self.name)]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(
                    f"missing positional MPO parameter {self.name!r}."
                ) from exc
        _check_scalar(value, name=f"parameter {self.name!r}")
        return value


def _as_4d(data, *, site, length):
    """Normalize a Quimb MPO tensor to ``(left, right, up, down)``."""
    shape = tuple(getattr(data, "shape", ()))
    if len(shape) == 4:
        out = data
    elif len(shape) == 3:
        if length == 1:
            raise ValueError("a one-site MPO tensor must have rank 2 or 4.")
        if site == 0:
            out = ar.do("reshape", data, (1, shape[0], shape[1], shape[2]))
        else:
            out = ar.do("reshape", data, (shape[0], 1, shape[1], shape[2]))
    elif len(shape) == 2 and length == 1:
        out = ar.do("reshape", data, (1, 1, shape[0], shape[1]))
    else:
        raise ValueError(
            "MPO tensors must have rank 4, rank 3 at an open boundary, "
            "or rank 2 for a one-site MPO."
        )
    if out.shape[-1] != out.shape[-2]:
        raise ValueError("MPO physical output and input dimensions must match.")
    return out


def _pauli_matrix(label):
    """Return a dense one-qubit Pauli matrix for a single-character label."""
    label = str(label).upper()
    matrices = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    try:
        return matrices[label]
    except KeyError as exc:
        raise ValueError(
            f"Pauli labels must be one of 'I', 'X', 'Y', or 'Z', got {label!r}."
        ) from exc


def _normalize_operator_sequence(operators, *, name):
    """Normalize a matrix sequence or a compact Pauli label string."""
    if isinstance(operators, str):
        operators = tuple(operators)
    else:
        operators = tuple(operators)
    normalized = []
    for operator in operators:
        if isinstance(operator, str):
            if len(operator) != 1:
                raise ValueError(
                    f"{name} string entries must be single Pauli labels, "
                    f"got {operator!r}."
                )
            normalized.append(_pauli_matrix(operator))
        else:
            normalized.append(operator)
    return tuple(normalized)


def _zeros(shape, *, like):
    try:
        return ar.do("zeros", tuple(shape), like=like)
    except Exception:  # pragma: no cover - backend compatibility fallback
        return np.zeros(tuple(shape), dtype=np.asarray(like).dtype)


def _stack(blocks, *, axis):
    if len(blocks) == 1:
        return ar.do("expand_dims", blocks[0], axis=axis)
    return ar.do("stack", tuple(blocks), axis=axis)


def _concat(blocks, *, axis):
    return ar.do("concatenate", tuple(blocks), axis=axis)


def _drop_axis(array, axis, position):
    """Remove one virtual channel for the sequential reference primitive."""
    size = int(array.shape[axis])
    if size <= 1:
        raise ValueError("cannot remove the last virtual channel.")
    pieces = []
    if position:
        index = [slice(None)] * len(array.shape)
        index[axis] = slice(0, position)
        pieces.append(array[tuple(index)])
    if position + 1 < size:
        index = [slice(None)] * len(array.shape)
        index[axis] = slice(position + 1, None)
        pieces.append(array[tuple(index)])
    if len(pieces) == 1:
        return pieces[0]
    return _concat(pieces, axis=axis)


def _backend_index_2d(array, rows, columns):
    """Gather a virtual submatrix on the active Autoray backend."""
    rows = _as_backend(np.asarray(rows, dtype=int), like=array)
    columns = _as_backend(np.asarray(columns, dtype=int), like=array)
    return array[rows[:, None], columns[None, :]]


def _backend_index_pairs(array, rows, columns):
    """Gather paired virtual entries on the active Autoray backend."""
    rows = _as_backend(np.asarray(rows, dtype=int), like=array)
    columns = _as_backend(np.asarray(columns, dtype=int), like=array)
    return array[rows, columns]


def _scatter_add_2d(array, rows, columns, values):
    """Scatter-add paired virtual blocks without leaving the backend graph."""
    backend = ar.infer_backend(array)
    value_dtype = getattr(values, "dtype", None)
    if value_dtype is not None and getattr(array, "dtype", None) != value_dtype:
        array = ar.do("astype", array, value_dtype)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)

    if backend == "numpy":
        delta = np.zeros_like(array)
        np.add.at(delta, (rows_np, columns_np), values)
        return array + delta
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        delta = ar.do("zeros_like", array)
        delta = delta.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=True,
        )
        return array + delta
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        delta = ar.do("zeros_like", array)
        delta = delta.at[(rows_backend, columns_backend)].add(values)
        return array + delta
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        delta = cp.zeros_like(array)
        cp.add.at(delta, (rows_np, columns_np), values)
        return array + delta

    # Keep an Autoray-compatible fallback for backends without a native
    # scatter-add rule. The common NumPy/Torch/JAX/CuPy paths above are the
    # performance-critical implementations.
    delta = ar.do("zeros_like", array)
    for row, column, value in zip(rows_np, columns_np, values):
        current = delta[row, column]
        delta = _setitem(delta, (row, column), current + value)
    return array + delta


def _scatter_set_2d(array, rows, columns, values):
    """Scatter unique paired virtual blocks without allocating a delta tensor."""
    backend = ar.infer_backend(array)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)
    if backend == "numpy":
        array[rows_np, columns_np] = values
        return array
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=False,
        )
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.at[(rows_backend, columns_backend)].set(values)
    if backend == "cupy":
        array[rows_np, columns_np] = values
        return array

    updated = array
    for row, column, value in zip(rows_np, columns_np, values):
        updated = _setitem(updated, (row, column), value)
    return updated


def _scatter_add_into_2d(array, rows, columns, values):
    """Scatter-add into a fresh array without creating a second full tensor."""
    backend = ar.infer_backend(array)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)
    if backend == "numpy":
        np.add.at(array, (rows_np, columns_np), values)
        return array
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=True,
        )
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.at[(rows_backend, columns_backend)].add(values)
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        cp.add.at(array, (rows_np, columns_np), values)
        return array

    updated = array
    for row, column, value in zip(rows_np, columns_np, values):
        updated = _setitem(updated, (row, column), updated[row, column] + value)
    return updated


def _history_transfer_matrix(groups, old_size):
    """Compile ordered virtual-channel groups into a dense transfer map.

    ``groups`` is the symbolic representation used by the history
    compression planners: each output channel maps to a weighted collection
    of original input channels.  The returned matrix is structural NumPy
    data, so it is safe to cache and later move to Torch/JAX/CuPy alongside
    the numerical tensor being transformed.
    """
    new_size = len(groups)
    if new_size * old_size > _MAX_HISTORY_TRANSFER_ELEMENTS:
        return None
    transfer = np.zeros((new_size, old_size), dtype=float)
    for target, group in enumerate(groups):
        for source, weight in group.items():
            transfer[target, int(source)] += float(weight)
    return transfer


def _apply_history_transfer(array, transfer, *, axis):
    """Apply a cached virtual transfer map with one backend contraction."""
    transfer = _as_backend(transfer, like=array)
    if getattr(transfer, "dtype", None) != getattr(array, "dtype", None):
        transfer = ar.do("astype", transfer, array.dtype)
    mapped = ar.do(
        "tensordot",
        transfer,
        array,
        axes=([1], [axis]),
    )
    if axis == 1:
        # tensordot puts the new right-channel axis first.
        mapped = ar.do("transpose", mapped, (1, 0, 2, 3))
    return mapped


def _complex_dtype(dtype):
    """Return whether a backend dtype is complex without importing it."""
    if bool(getattr(dtype, "is_complex", False)):
        return True
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.complexfloating))
    except (TypeError, ValueError):
        return False


def _align_tensordot_dtypes(left, right):
    """Align contraction operands where Torch requires identical dtypes."""
    left_dtype = getattr(left, "dtype", None)
    right_dtype = getattr(right, "dtype", None)
    if left_dtype == right_dtype:
        return left, right
    if _complex_dtype(left_dtype) and not _complex_dtype(right_dtype):
        right = ar.do("astype", right, left_dtype)
    else:
        left = ar.do("astype", left, right_dtype)
    return left, right


def _array_equal(left, right):
    """Check exact equality without introducing a numerical cutoff."""
    # NumPy is the cheap path for ordinary arrays.  The Autoray fallback keeps
    # backend tensors supported, but may still transfer a small local block to
    # the host through ``np.asarray``.  Future native backends should register
    # a structural equality/fingerprint here to avoid that synchronization.
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        try:
            equal = ar.do("equal", left, right)
            result = ar.do("all", equal)
            return bool(result.item() if hasattr(result, "item") else result)
        except Exception:  # pragma: no cover - defensive backend guard
            return False


def _setitem(array, index, value):
    """Set one tensor entry on mutable and functional array backends."""
    backend = ar.infer_backend(array)
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is None and backend == "numpy":
        value_dtype = np.asarray(value).dtype
    if value_dtype is not None and getattr(array, "dtype", None) != value_dtype:
        # Real Hamiltonian tensors must be promoted before Algorithm 3 inserts
        # complex real-time terms. Otherwise NumPy/Torch assignment or JAX
        # scatter updates silently discard the imaginary component.
        array = ar.do("astype", array, value_dtype)
    if backend == "jax":
        return array.at[index].set(value)
    if backend == "torch":
        # Do not mutate a view participating in an autodiff graph.  A clone
        # retains the graph while turning the indexed update into a tracked
        # CopySlices operation, avoiding Torch's version-counter error during
        # the later virtual-channel contractions.
        updated = array.clone()
        if isinstance(index, tuple):
            index = tuple(
                _as_backend(item, like=array)
                if isinstance(item, np.ndarray)
                else item
                for item in index
            )
        elif isinstance(index, np.ndarray):
            index = _as_backend(index, like=array)
        updated[index] = value
        return updated
    array[index] = value
    return array


def _level_number(token):
    return token.level if isinstance(token, MPOLevelToken) else int(token)


def _move_level_front(history, level):
    selected = tuple(token for token in history if _level_number(token) == level)
    remaining = tuple(token for token in history if _level_number(token) != level)
    return selected + remaining


def _history_signature(history):
    """Return the level-only signature used by the paper algorithms."""
    return tuple(_level_number(token) for token in history)


def _sort_history_front(history, level):
    """Move all tokens with ``level`` to the front, preserving the rest."""
    return _move_level_front(history, level)


def _term_from_input(term):
    if isinstance(term, MPOProductTerm):
        return term
    if isinstance(term, Mapping):
        sites = term.get("sites", term.get("locations"))
        operators = term.get("operators", term.get("paulis"))
        if sites is None or operators is None:
            raise ValueError("each product term needs sites and operators.")
        coefficient = term.get("coefficient", _UNSET)
        if coefficient is _UNSET:
            if "parameter" in term:
                coefficient = MPOParameter(
                    term["parameter"],
                    term.get("default", _UNSET),
                )
            else:
                coefficient = 1.0
        return MPOProductTerm(
            sites=sites,
            operators=operators,
            coefficient=coefficient,
            string_operators=term.get(
                "string_operators",
                term.get("string_paulis"),
            ),
            charge=term.get("charge"),
        )
    if isinstance(term, (tuple, list)) and len(term) in (2, 3):
        return MPOProductTerm(
            sites=term[0],
            operators=term[1],
            coefficient=term[2] if len(term) == 3 else 1.0,
        )
    raise TypeError(
        "terms must contain MPOProductTerm values, mappings, or "
        "(sites, operators[, coefficient]) pairs."
    )


class FirstDegreeMPO:
    """A finite-chain MPO with explicit virtual-level histories.

    The object is intentionally usable for intermediate products as well as
    first-degree Hamiltonians.  ``degree`` records the algebraic degree of
    the current expression; it is ``1`` for a Hamiltonian built from local
    terms and increases under products.

    The public algebraic methods return new semantic objects.  The one
    exception is ``compress_exact(inplace=True)``, which is explicit because
    it mutates virtual-bond tensors and histories.  ``arrays`` is a read-only
    tuple view, but the backend arrays inside it are not copied; callers that
    need ownership should call :meth:`copy` before mutating backend data.

    This is a semantic MPO for the paper's construction, not a drop-in
    replacement for every arbitrary Quimb MPO.  Higher-order history methods
    require the first-degree level-1/2/3 rail structure created by
    :meth:`from_automaton` or :meth:`from_local_terms`.

    Parameters
    ----------
    arrays : sequence[array_like]
        MPO tensors in ``lrud`` order.  Open-boundary rank-3 tensors and a
        rank-2 one-site tensor are accepted.
    levels : sequence[sequence[MPOLevel]], optional
        Labels for the ``L + 1`` virtual bonds, including the two singleton
        boundary bonds.  When omitted, neutral level metadata is generated.
    degree : int, default=1
        Algebraic degree represented by the expression.
    """

    def __init__(
        self,
        arrays,
        *,
        levels=None,
        degree=1,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        metadata=None,
    ):
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError("arrays must contain at least one MPO tensor.")
        if not isinstance(degree, Integral) or int(degree) < 0:
            raise ValueError("degree must be a non-negative integer.")
        self.L = len(arrays)
        self._arrays = tuple(
            _as_4d(array, site=site, length=self.L)
            for site, array in enumerate(arrays)
        )
        self.degree = int(degree)
        self.upper_ind_id = upper_ind_id
        self.lower_ind_id = lower_ind_id
        self.site_tag_id = site_tag_id
        self.metadata = dict(metadata or {})
        # Keep this attribute present on every instance so callers do not
        # need to probe for it after an optional compression stage. It is
        # populated by ``compress_exact`` or ``extensive_exponential``.
        self.compression_report = self.metadata.get("compression_report")
        # Populated by ``from_automaton`` when the source graph is available.
        # Keeping this private means direct array construction remains a
        # supported fallback, while local-term constructors can avoid
        # materializing unreachable Cartesian history states.
        self._structural_transitions = None
        # Raw history topology depends only on the structural MPO and the
        # Taylor order. Keep it separate from value-dependent exact merges so
        # parameterized builds can reuse the expensive graph walk safely.
        self._history_topology_cache = {}
        # The following caches contain symbolic indices and histories only.
        # Numerical tensors are deliberately never retained here: a bound
        # Torch/JAX MPO must build a new autodiff graph for every call.
        self._history_symbolic_cache = None
        self._history_extension_plan_cache = {}
        self._history_compression_plan_cache = {}
        self._history_approximation_plan_cache = {}
        self._history_tensor_plan_cache = {}
        self._base_level_position_cache = None
        self._levels = self._normalize_levels(levels)
        self._validate()

    def _normalize_levels(self, levels):
        if levels is None:
            out = [[MPOLevel(
                ("left",),
                (MPOLevelToken(1),),
            )]]
            for bond, array in enumerate(self._arrays[:-1], start=1):
                out.append([
                    MPOLevel(
                        ("bond", bond, pos),
                        (MPOLevelToken(2, ("bond", bond, pos)),),
                    )
                    for pos in range(array.shape[1])
                ])
            out.append([
                MPOLevel(("right",), (MPOLevelToken(3),))
            ])
            return out

        if len(levels) != self.L + 1:
            raise ValueError(f"levels must have length L + 1 = {self.L + 1}.")
        normalized = []
        for bond, (bond_levels, array) in enumerate(zip(levels, [None, *self._arrays])):
            values = []
            for pos, level in enumerate(bond_levels):
                if isinstance(level, MPOLevel):
                    values.append(level)
                else:
                    values.append(
                        MPOLevel(
                            level,
                            (MPOLevelToken(2, level),),
                        )
                    )
            normalized.append(values)
        return normalized

    def _validate(self):
        if len(self._levels) != self.L + 1:
            raise ValueError("there must be one level list per virtual bond.")
        for site, array in enumerate(self._arrays):
            if len(self._levels[site]) != array.shape[0]:
                raise ValueError(
                    f"bond {site} has {len(self._levels[site])} levels but "
                    f"tensor {site} has left dimension {array.shape[0]}.")
            if len(self._levels[site + 1]) != array.shape[1]:
                raise ValueError(
                    f"bond {site + 1} has {len(self._levels[site + 1])} levels but "
                    f"tensor {site} has right dimension {array.shape[1]}.")
            if array.shape[-1] != array.shape[-2]:
                raise ValueError("all local MPO physical dimensions must be square.")
        phys_dim = self._arrays[0].shape[-1]
        if any(array.shape[-1] != phys_dim for array in self._arrays):
            raise ValueError("all MPO sites must have the same physical dimension.")
        if any(len(levels) != 1 for levels in (self._levels[0], self._levels[-1])):
            raise ValueError("open-boundary first-degree MPOs need singleton boundary bonds.")

    @property
    def arrays(self):
        """Read-only tuple of normalized ``(left, right, up, down)`` tensors.

        The tuple itself is immutable, but the backend tensors are returned
        by reference to preserve Autoray dtype/device/backend behavior.
        """
        return self._arrays

    @property
    def levels(self):
        """Read-only level metadata grouped by virtual bond."""
        return tuple(tuple(levels) for levels in self._levels)

    @property
    def bond_dimensions(self):
        return tuple(len(levels) for levels in self._levels[1:-1])

    @property
    def phys_dim(self):
        return int(self._arrays[0].shape[-1])

    @property
    def is_first_degree(self):
        return self.degree == 1

    def copy(self):
        """Return a semantic copy sharing backend tensor storage.

        This is intentionally a structural copy, not a deep array copy.  It
        is sufficient for the non-mutating algebraic API and avoids an
        unnecessary device transfer; use backend-specific copying before
        editing tensor values in place.
        """
        out = type(self)(
            self._arrays,
            levels=self.levels,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=self.metadata,
        )
        out._structural_transitions = self._structural_transitions
        out._history_topology_cache = self._history_topology_cache
        out._history_symbolic_cache = self._history_symbolic_cache
        out._history_extension_plan_cache = self._history_extension_plan_cache
        out._history_compression_plan_cache = self._history_compression_plan_cache
        out._history_approximation_plan_cache = (
            self._history_approximation_plan_cache
        )
        out._history_tensor_plan_cache = self._history_tensor_plan_cache
        out._base_level_position_cache = self._base_level_position_cache
        return out

    def _bind_arrays(self, arrays):
        """Make a lightweight view with new local tensors and shared plans.

        ``MPOBasis`` uses this private primitive for its value-only compiled
        evaluator.  Reconstructing a semantic MPO through ``__init__`` would
        repeat validation and level normalization on every optimizer step;
        the structural metadata and history plans are immutable for this
        purpose, while the backend arrays must remain fresh so their current
        autodiff graph is never cached.
        """
        arrays = tuple(arrays)
        if len(arrays) != self.L:
            raise ValueError(f"arrays must have length {self.L}.")
        out = object.__new__(type(self))
        out.L = self.L
        out._arrays = tuple(
            _as_4d(array, site=site, length=self.L)
            for site, array in enumerate(arrays)
        )
        out.degree = self.degree
        out.upper_ind_id = self.upper_ind_id
        out.lower_ind_id = self.lower_ind_id
        out.site_tag_id = self.site_tag_id
        out.metadata = {}
        out.compression_report = None
        out._structural_transitions = self._structural_transitions
        out._history_topology_cache = self._history_topology_cache
        out._history_symbolic_cache = self._history_symbolic_cache
        out._history_extension_plan_cache = self._history_extension_plan_cache
        out._history_compression_plan_cache = self._history_compression_plan_cache
        out._history_approximation_plan_cache = (
            self._history_approximation_plan_cache
        )
        out._history_tensor_plan_cache = self._history_tensor_plan_cache
        out._base_level_position_cache = self._base_level_position_cache
        out._levels = self._levels
        return out

    @staticmethod
    def _history_for_state(state, *, start_state, done_state):
        if state == start_state:
            return (MPOLevelToken(1),)
        if state == done_state:
            return (MPOLevelToken(3),)
        return (MPOLevelToken(2, state),)

    @classmethod
    def from_automaton(cls, automaton, *, degree=1, **kwargs):
        """Compile an :class:`MPOAutomaton` into a semantic MPO.

        The automaton remains the source of numerical local blocks, while the
        returned object adds the symbolic level histories needed by the
        higher-order algorithms.  No dense operator is constructed.
        """
        if not isinstance(automaton, MPOAutomaton):
            raise TypeError("automaton must be an MPOAutomaton.")
        arrays = automaton.to_arrays()
        levels = [[
            MPOLevel(
                automaton.start_state,
                cls._history_for_state(
                    automaton.start_state,
                    start_state=automaton.start_state,
                    done_state=automaton.done_state,
                ),
            )
        ]]
        for cut_channels in automaton.channels:
            levels.append([
                MPOLevel(
                    channel.state,
                    cls._history_for_state(
                        channel.state,
                        start_state=automaton.start_state,
                        done_state=automaton.done_state,
                    ),
                    charge=channel.charge,
                )
                for channel in cut_channels
            ])
        levels.append([MPOLevel(
            automaton.done_state,
            cls._history_for_state(
                automaton.done_state,
                start_state=automaton.start_state,
                done_state=automaton.done_state,
            ),
        )])
        result = cls(arrays, levels=levels, degree=degree, **kwargs)
        if result.L > 1:
            result._structural_transitions = (
                result._structural_transitions_from_automaton(automaton)
            )
            result._history_symbolic_cache = None
        return result

    @classmethod
    def from_local_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        degree=1,
        share_channels=True,
        **kwargs,
    ):
        """Build a first-degree MPO from factorized local product terms.

        This is the preferred public constructor for Hamiltonian-like sums.
        The input terms are compiled through :class:`MPOAutomaton`, which
        keeps the identity rails and active channels explicit.
        """
        terms = tuple(_term_from_input(term) for term in terms)
        if not terms:
            raise ValueError("terms must contain at least one product term.")
        if phys_dim is None:
            first_operator = terms[0].operators[0]
            phys_dim = int(first_operator.shape[0])
        automaton = MPOAutomaton.from_product_terms(
            L,
            terms,
            phys_dim=phys_dim,
            share_channels=share_channels,
        )
        return cls.from_automaton(automaton, degree=degree, **kwargs)

    @classmethod
    def from_pauli_terms(
        cls,
        L,
        terms,
        *,
        degree=1,
        share_channels=True,
        **kwargs,
    ):
        """Build a first-degree MPO from compact Pauli product terms.

        Each term may be an :class:`MPOProductTerm`, a mapping with
        ``sites`` (or ``locations``), ``paulis`` (or ``operators``), and an
        optional ``coefficient``, or a ``(sites, paulis)``/
        ``(sites, paulis, coefficient)`` tuple. Pauli strings such as
        ``"ZXY"`` and label sequences such as ``("Z", "X", "Y")`` are
        accepted. Sites are zero-based and list the non-identity support.
        """
        return cls.from_local_terms(
            L,
            terms,
            phys_dim=2,
            degree=degree,
            share_channels=share_channels,
            **kwargs,
        )

    @classmethod
    def identity(cls, L, phys_dim, *, like=None, **kwargs):
        """Construct an exact identity MPO with degree zero.

        ``like`` optionally supplies the backend and dtype for the local
        identity blocks; it is useful when the identity is used as the
        neutral element of a backend-native algebraic operation.
        """
        return cls.from_automaton(
            MPOAutomaton.identity(L, phys_dim, like=like),
            degree=0,
            **kwargs,
        )

    def to_mpo(self):
        """Compile to a Quimb ``MatrixProductOperator`` without compression.

        This method is the deliberate interop boundary.  It preserves local
        tensor backend/dtype information, performs no SVD or bond truncation,
        and attaches a semantic copy as ``pepsy_first_degree`` so callers can
        move between Quimb execution and Pepsy history inspection.
        """
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        # Quimb is the stable tensor-network interchange boundary: callers
        # can immediately use the returned object with existing contraction,
        # compression, and MPS-application APIs. The semantic object remains
        # attached for code that needs the level histories later. Keeping this
        # adapter one-way avoids duplicating Quimb's MPO implementation here.
        if self.L == 1:
            compiled_arrays = (self._arrays[0][0, 0],)
        else:
            compiled_arrays = (
                self._arrays[0][0],
                *self._arrays[1:-1],
                self._arrays[-1][:, 0],
            )
        mpo = qtn.MatrixProductOperator(
            compiled_arrays,
            shape="lrud",
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
        )
        mpo.pepsy_first_degree = self.copy()
        return mpo

    def apply_to_mps(
        self, mps, *, method="direct", inplace=False, **compress_opts,
    ):
        """Apply this MPO to a Quimb MPS using tensor-network compression.

        The semantic object is compiled at the Quimb boundary and delegated
        to ``MatrixProductState.gate_with_mpo``.  No dense state or operator
        is formed by this method.
        """
        if not hasattr(mps, "gate_with_mpo"):
            raise TypeError("mps must provide Quimb's gate_with_mpo method.")
        return mps.gate_with_mpo(
            self.to_mpo(),
            method=method,
            inplace=inplace,
            **compress_opts,
        )

    def compress_numerical(
        self,
        *,
        form=None,
        max_bond=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        create_bond=False,
        estimate_error=False,
        return_report=False,
        **compress_opts,
    ):
        """Numerically compress the compiled MPO through Quimb.

        This is intentionally separate from :meth:`compress_exact` and the
        paper's analytical Algorithm 4.  The returned object is an ordinary
        Quimb ``MatrixProductOperator`` because a numerical truncation can no
        longer be represented faithfully by the original semantic histories.
        The report records the requested policy and the bond dimensions before
        and after the Quimb sweep.  Set ``estimate_error=True`` to additionally
        contract the Frobenius norm of ``MPO_before - MPO_after`` without
        densifying either operator.
        """
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        _check_scalar(cutoff, name="cutoff")
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        if not isinstance(cutoff_mode, str):
            raise TypeError("cutoff_mode must be a string.")
        if not isinstance(estimate_error, bool):
            raise TypeError("estimate_error must be a boolean.")
        if "max_bond" in compress_opts or "cutoff" in compress_opts:
            raise TypeError(
                "max_bond and cutoff must be supplied as explicit compression "
                "arguments, not duplicated in compress_opts."
            )

        reference = self.to_mpo() if estimate_error else None
        mpo = self.to_mpo()
        initial_bond_dimensions = tuple(int(size) for size in mpo.bond_sizes())
        mpo.compress(
            form=form,
            create_bond=create_bond,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **compress_opts,
        )
        final_bond_dimensions = tuple(int(size) for size in mpo.bond_sizes())
        operator_frobenius_error = None
        operator_frobenius_relative_error = None
        error_estimator = None
        if estimate_error:
            # ``TensorNetwork.norm`` contracts the doubled network and hence
            # computes the operator Frobenius norm here. Keeping this behind
            # an explicit flag leaves the normal compression path unchanged.
            difference = reference - mpo
            operator_frobenius_error = difference.norm()
            reference_norm = reference.norm()
            try:
                has_reference_norm = bool(reference_norm != 0)
            except (TypeError, ValueError):  # pragma: no cover - tracer guard
                has_reference_norm = True
            if has_reference_norm:
                operator_frobenius_relative_error = (
                    operator_frobenius_error / reference_norm
                )
            error_estimator = "tensor-network-frobenius"
        report = MPONumericalCompressionReport(
            method="quimb",
            form=form,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            truncated=final_bond_dimensions != initial_bond_dimensions,
            truncation_error=operator_frobenius_error,
            operator_frobenius_error=operator_frobenius_error,
            operator_frobenius_relative_error=operator_frobenius_relative_error,
            error_estimator=error_estimator,
        )
        # Numerical truncation invalidates the semantic history attachment.
        mpo.pepsy_first_degree = None
        mpo.pepsy_numerical_compression_report = report
        if return_report:
            return mpo, report
        return mpo

    def compress_fixed_rank(self, max_bond, *, return_report=False):
        """Compress with a fixed-rank, backend-differentiable TT-SVD sweep.

        This is the autodiff-oriented numerical path. It selects at most
        ``max_bond`` singular vectors at every cut, with the rank determined
        only by the requested cap and matrix dimensions. Unlike a cutoff
        sweep, it never branches on singular values, so gradients can flow
        through the backend SVD away from repeated-singular-value points.

        The returned object is a ``FirstDegreeMPO`` with neutral virtual
        history metadata: a numerical rank reduction changes the original
        paper-history representation. Use :meth:`to_mpo` for contraction or
        use :meth:`compress_exact` before this method if analytical history
        compression is also desired.
        """
        if not isinstance(max_bond, Integral) or int(max_bond) < 1:
            raise ValueError("max_bond must be a positive integer.")
        max_bond = int(max_bond)
        initial_bond_dimensions = tuple(self.bond_dimensions)

        if self.L == 1:
            output = self.copy()
            output.metadata.update({
                "operation": "fixed_rank_compression",
                "history_valid": False,
                "max_bond": max_bond,
            })
            report = MPODifferentiableCompressionReport(
                method="fixed-rank-tt-svd",
                max_bond=max_bond,
                initial_bond_dimensions=initial_bond_dimensions,
                final_bond_dimensions=initial_bond_dimensions,
                truncated=False,
            )
            output.metadata["compression_report"] = report
            output.compression_report = report
            return (output, report) if return_report else output

        arrays = []
        carry = None
        for site, array in enumerate(self._arrays[:-1]):
            if site == 0:
                combined = array
            else:
                combined = ar.do("tensordot", carry, array, axes=([1], [0]))

            left_dim, right_dim, phys_up, phys_down = combined.shape
            matrix_data = ar.do(
                "transpose",
                combined,
                (0, 2, 3, 1),
            )
            matrix = ar.do(
                "reshape",
                matrix_data,
                (int(left_dim) * int(phys_up) * int(phys_down), int(right_dim)),
            )
            u, singular_values, vh = _fixed_rank_svd(matrix)
            rank = min(max_bond, int(singular_values.shape[0]))
            u = u[:, :rank]
            singular_values = singular_values[:rank]
            vh = vh[:rank, :]

            local = ar.do(
                "reshape",
                u,
                (int(left_dim), int(phys_up), int(phys_down), rank),
            )
            local = ar.do("transpose", local, (0, 3, 1, 2))
            arrays.append(local)
            carry = ar.do(
                "multiply",
                ar.do("reshape", singular_values, (rank, 1)),
                vh,
            )

        # Absorb the final tensor without another factorization. Its right
        # boundary is retained, preserving the normalized MPO layout.
        arrays.append(
            ar.do("tensordot", carry, self._arrays[-1], axes=([1], [0]))
        )
        final_bond_dimensions = tuple(
            int(array.shape[1]) for array in arrays[:-1]
        )
        output = type(self)(
            arrays,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "fixed_rank_compression",
                "history_valid": False,
                "max_bond": max_bond,
            },
        )
        report = MPODifferentiableCompressionReport(
            method="fixed-rank-tt-svd",
            max_bond=max_bond,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            truncated=final_bond_dimensions != initial_bond_dimensions,
        )
        output.metadata["compression_report"] = report
        output.compression_report = report
        return (output, report) if return_report else output

    def compress_to_bond(
        self,
        chi,
        *,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        **compress_opts,
    ):
        """Compress to a requested MPO bond dimension.

        ``chi`` is the final MPO bond cap. It is deliberately separate from
        ``extensive_exponential(max_bond=...)``, whose guard applies only to
        the temporary paper-history representation. The default compression
        is Quimb's cutoff-based sweep for ordinary numerical execution and a
        fixed-rank TT-SVD sweep when ``differentiable=True``.

        The fixed-rank path returns a :class:`FirstDegreeMPO` with invalidated
        analytical histories, while the Quimb path returns an ordinary
        ``MatrixProductOperator``. Numerical compression cannot preserve the
        pre-compression history table.
        """
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")
        chi = int(chi)
        if compression is None:
            compression = "fixed_rank" if differentiable else "quimb"
        if compression not in {"quimb", "fixed_rank"}:
            raise ValueError(
                "compression must be 'quimb' or 'fixed_rank'."
            )
        if differentiable and compression != "fixed_rank":
            raise ValueError(
                "differentiable=True requires compression='fixed_rank'."
            )
        if compression == "fixed_rank":
            if compress_opts:
                unexpected = ", ".join(sorted(compress_opts))
                raise TypeError(
                    "fixed-rank compression does not accept Quimb options: "
                    f"{unexpected}."
                )
            return self.compress_fixed_rank(chi, return_report=return_report)
        return self.compress_numerical(
            max_bond=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            return_report=return_report,
            **compress_opts,
        )

    def _exp_with_compression(
        self,
        step,
        *,
        metadata_dt,
        metadata_operation,
        order=1,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        **kwargs,
    ):
        """Build ``exp(step * self)`` and optionally compress the result."""
        if chi is None:
            if compression is not None:
                raise ValueError(
                    "compression requires chi; omit it for an uncompressed MPO."
                )
            if return_report:
                raise ValueError("return_report requires chi compression.")
            result = self.extensive_exponential(
                step,
                order=order,
                **kwargs,
            )
            result.metadata.update({
                "operation": metadata_operation,
                "dt": metadata_dt,
                "exponent": step,
            })
            return result

        output = self.extensive_exponential(
            step,
            order=order,
            **kwargs,
        )
        compressed = output.compress_to_bond(
            chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )
        if return_report:
            result, report = compressed
        else:
            result, report = compressed, None
        exponential_metadata = {
            "operation": metadata_operation,
            "dt": metadata_dt,
            "exponent": step,
            "order": order,
            "chi": int(chi),
            "compression": (
                "fixed_rank" if differentiable and compression is None
                else compression or "quimb"
            ),
            "differentiable": bool(differentiable),
        }
        for key in (
            "mode",
            "history_storage",
            "history_cache_hit",
            "tensor_plan_cache_hit",
            "compression_plan_cache_hit",
            "extension_plan_cache_hit",
            "approximation_plan_cache_hit",
        ):
            if key in output.metadata:
                exponential_metadata[key] = output.metadata[key]
        if output.compression_report is not None:
            exponential_metadata["analytical_compression_report"] = (
                output.compression_report
            )
        if isinstance(result, FirstDegreeMPO):
            result.metadata.update(exponential_metadata)
        else:
            setattr(result, "pepsy_exp_metadata", exponential_metadata)
            if metadata_operation == "time_evolution":
                # Keep the attribute used by the original real-time API.
                result.pepsy_evolution_metadata = exponential_metadata
        return (result, report) if return_report else result

    def exp(
        self,
        step=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
    ):
        """Build ``exp(step * self)`` with optional final compression.

        Parameters
        ----------
        step : scalar
            Actual scalar in the exponential. Use ``-1j * tau`` for
            real-time evolution and ``-beta`` for imaginary time.
        dt : scalar, optional
            Compatibility spelling for ``step``. Supplying both raises.
        order, mode, max_bond, on_exceed, cache_history, history_storage
            Controls for the analytical higher-order history construction.
        chi : int, optional
            Separate final numerical MPO bond cap. It is not the temporary
            history-bond guard ``max_bond``.
        differentiable : bool, optional
            With ``chi``, select fixed-rank autodiff compression instead of a
            value-dependent numerical cutoff.
        """
        step = _resolve_exp_step(step, dt)
        return self._exp_with_compression(
            step,
            metadata_dt=step,
            metadata_operation="exp",
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )

    def time_evolution(
        self,
        dt,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
    ):
        """Build real-time ``exp(-1j * dt * self)``.

        This is a convenience wrapper around :meth:`exp`; use :meth:`exp`
        directly when the exponential step should be explicit.
        """
        _check_scalar(dt, name="dt")
        return self._exp_with_compression(
            -1j * dt,
            metadata_dt=dt,
            metadata_operation="time_evolution",
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )

    def exp_arrays(
        self,
        step=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Return ``exp(step * self)`` tensors without a Quimb wrapper.

        This is the low-level numerical interface for compiled optimization
        loops. It returns the normalized ``(left, right, up, down)`` tensor
        tuple directly. The backend values remain connected to ``step`` and
        the Hamiltonian coefficients for autodiff.
        """
        step = _resolve_exp_step(step, dt)
        return self.exp(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        ).arrays

    def time_evolution_arrays(
        self,
        dt,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Return real-time evolution tensors without a semantic wrapper."""
        return self.time_evolution(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        ).arrays

    def expectation(self, mps, *, contraction_opt=None):
        """Evaluate ``<mps|self|mps>`` through Pepsy's MPS contraction API."""
        from pepsy.tensors import expec_mpo  # pylint: disable=import-outside-toplevel

        # Quimb's contraction backend requires every operand to use one
        # array backend.  A parameterized observable has a Torch/JAX MPO while
        # a convenient product-state constructor often returns NumPy tensors;
        # align that fixed state data to the observable backend without
        # touching the observable's differentiable tensors.
        reference = next(
            (
                array
                for array in self._arrays
                if _backend_name(array) not in {"builtins", "numpy"}
            ),
            None,
        )
        if reference is not None and hasattr(mps, "copy"):
            mps = mps.copy()
            for tensor in getattr(mps, "tensors", ()):
                data = _as_backend(
                    tensor.data,
                    like=reference,
                    dtype=getattr(reference, "dtype", None),
                )
                if data is not tensor.data:
                    tensor.modify(data=data)

        return expec_mpo(
            self.to_mpo(),
            mps,
            contraction_opt=contraction_opt,
        )

    def scale(self, coefficient):
        """Return ``coefficient * self`` by scaling one boundary tensor."""
        _check_scalar(coefficient, name="coefficient")
        arrays = list(self._arrays)
        arrays[0] = _multiply_scalar(coefficient, arrays[0])
        out = type(self)(
            arrays,
            levels=self.levels,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={**self.metadata, "scale": coefficient},
        )
        out._structural_transitions = self._structural_transitions
        return out

    def add(self, other):
        """Return the exact direct sum ``self + other``."""
        self._check_compatible(other)
        if self.L == 1:
            arrays = (self._arrays[0] + other._arrays[0],)
            levels = [[self._levels[0][0]], [self._levels[1][0]]]
        else:
            arrays = []
            levels = [[self._levels[0][0]]]
            first = _concat((self._arrays[0], other._arrays[0]), axis=1)
            arrays.append(first)
            for site in range(1, self.L - 1):
                left, right = self._arrays[site], other._arrays[site]
                top = _concat(
                    (
                        left,
                        _zeros((left.shape[0], right.shape[1], *left.shape[2:]), like=left),
                    ),
                    axis=1,
                )
                bottom = _concat(
                    (
                        _zeros((right.shape[0], left.shape[1], *right.shape[2:]), like=right),
                        right,
                    ),
                    axis=1,
                )
                arrays.append(_concat((top, bottom), axis=0))
            arrays.append(_concat((self._arrays[-1], other._arrays[-1]), axis=0))
            for bond in range(1, self.L):
                levels.append([
                    *self._levels[bond],
                    *other._levels[bond],
                ])
            levels.append([self._levels[-1][0]])

        return type(self)(
            arrays,
            levels=levels,
            degree=max(self.degree, other.degree),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": "add"},
        )

    def product(self, other, *, kind="ordinary"):
        """Return an exact virtual-space product of two MPO expressions.

        ``kind`` is provenance metadata only; it does not change the
        multiplication.  The tensor product is exact and keeps all paths.
        The paper-specific extensive-path filtering is intentionally applied
        later by the Taylor builder so this foundational algebra remains
        predictable.

        The explicit local loops are a deliberate tensor-network choice: they
        multiply physical blocks site by site and retain histories, rather
        than converting either MPO to a global matrix.  A future optimized
        implementation can replace the loops with a backend-aware batched
        kernel without changing this semantic contract.
        """
        self._check_compatible(other)
        arrays = []
        levels = []
        for site, (left, right) in enumerate(zip(self._arrays, other._arrays)):
            # Pairing virtual states explicitly is more verbose than calling
            # Quimb's generic MPO product, but it preserves the symbolic
            # history needed by the paper's later exact rewiring steps.
            Dl1, Dr1, d, _ = left.shape
            Dl2, Dr2, _, _ = right.shape
            rows = []
            for left_pos in range(Dl1):
                for right_pos in range(Dl2):
                    blocks = []
                    for left_next in range(Dr1):
                        for right_next in range(Dr2):
                            blocks.append(
                                ar.do(
                                    "matmul",
                                    left[left_pos, left_next],
                                    right[right_pos, right_next],
                                )
                            )
                    rows.append(_stack(blocks, axis=0).reshape(Dr1 * Dr2, d, d))
            arrays.append(_stack(rows, axis=0).reshape(Dl1 * Dl2, Dr1 * Dr2, d, d))
            levels.append([
                MPOLevel(
                    ("product", a.label, b.label),
                    a.history + b.history,
                    charge=(a.charge, b.charge),
                )
                for a in self._levels[site]
                for b in other._levels[site]
            ])
        levels.append([
            MPOLevel(
                ("product", a.label, b.label),
                a.history + b.history,
                charge=(a.charge, b.charge),
            )
            for a in self._levels[-1]
            for b in other._levels[-1]
        ])
        return type(self)(
            arrays,
            levels=levels,
            degree=self.degree + other.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": kind},
        )

    def non_disjoint_product(self, other):
        """Return the exact raw product used as the algebraic input.

        The result retains the level histories and all connected and
        disconnected paths.  The later Taylor construction will apply the
        paper's level rewiring and exact-compression rules; keeping this
        operation explicit prevents accidental use of generic Quimb MPO
        multiplication in that implementation.  The name records that no
        support analysis or connected-term filtering has happened yet.
        """
        return self.product(other, kind="non_disjoint")

    def disjoint_product(self, other):
        """Return an explicitly labelled exact product of two expressions.

        ``disjoint`` is currently a provenance label, not an assertion or an
        automatic support decomposition.  Future extensive builders can use
        this hook for overlap-aware channel pruning once that analysis is
        implemented.
        """
        return self.product(other, kind="disjoint")

    def commutator(self, other):
        """Return the exact commutator ``self @ other - other @ self``."""
        return self.non_disjoint_product(other).add(
            other.non_disjoint_product(self).scale(-1)
        )

    def power(self, exponent):
        """Return an exact non-negative integer power."""
        if not isinstance(exponent, Integral) or int(exponent) < 0:
            raise ValueError("exponent must be a non-negative integer.")
        exponent = int(exponent)
        if exponent == 0:
            return type(self).identity(
                self.L,
                self.phys_dim,
                like=self._arrays[0],
                upper_ind_id=self.upper_ind_id,
                lower_ind_id=self.lower_ind_id,
                site_tag_id=self.site_tag_id,
            )
        result = self.copy()
        for _ in range(1, exponent):
            result = result.non_disjoint_product(self)
        return result

    def power_raw(self, exponent):
        """Return an exact MPO power without forming a global dense operator.

        The virtual indices are paired at every site and the physical blocks
        are multiplied locally.  The resulting histories therefore retain the
        factor order required by the higher-order MPO construction.

        This compatibility spelling currently delegates to :meth:`power`.
        It remains explicit because callers working from the paper often need
        to distinguish a raw power from a later history-rewired power.
        """
        return self.power(exponent)

    def _history_schemas(self):
        """Return the base-level schema used at every raw-history cut."""
        return tuple(
            self._levels[1]
            if bond == 0
            else self._levels[-2]
            if bond == self.L
            else self._levels[bond]
            for bond in range(self.L + 1)
        )

    def _history_symbolic_data(self):
        """Build structural history lookup tables once per MPO topology.

        The history algorithms used to resolve a token by scanning a whole
        level list and to test reachability by walking all earlier sites for
        every candidate.  Both operations are structural.  This table turns
        them into dictionary lookups and one precomputed forward reachability
        walk.  It is safe to retain across parameter rebinding because it
        contains no numerical tensor or backend scalar.
        """
        if self._history_symbolic_cache is not None:
            return self._history_symbolic_cache

        schemas = self._history_schemas()
        token_levels = tuple(
            {
                level.history: level
                for level in schema
                if len(level.history) == 1
            }
            for schema in schemas
        )
        reachable_labels = None
        if self._structural_transitions is not None:
            start_labels = frozenset(
                level.label
                for level in schemas[0]
                if _level_number(level.history[0]) == 1
            )
            reachable = [start_labels]
            for edges in self._structural_transitions:
                reachable.append(frozenset(
                    right
                    for left, right in edges
                    if left in reachable[-1]
                ))
            reachable_labels = tuple(reachable)

        self._history_symbolic_cache = {
            "schemas": schemas,
            "token_levels": token_levels,
            "reachable_labels": reachable_labels,
            "inserted_states": {},
        }
        return self._history_symbolic_cache

    def _history_inserted_states(self, bond, history, token):
        """Return reachable factor states after inserting one token.

        Equal insertions (for example inserting a level-1 token next to an
        existing level-1 token) are grouped with a multiplicity.  This is the
        same sum as the paper's insertion loop, but avoids evaluating the
        same local product repeatedly.
        """
        symbolic = self._history_symbolic_data()
        key = (bond, tuple(history), token)
        cached = symbolic["inserted_states"].get(key)
        if cached is not None:
            return cached

        grouped = {}
        for position in range(len(history) + 1):
            extended = (
                history[:position]
                + (token,)
                + history[position:]
            )
            state = self._history_tokens_reachable(
                symbolic["schemas"],
                bond,
                extended,
                {},
            )
            if state is None:
                continue
            if extended in grouped:
                old_state, multiplicity = grouped[extended]
                grouped[extended] = (old_state, multiplicity + 1)
            else:
                grouped[extended] = (state, 1)

        result = tuple(grouped.values())
        symbolic["inserted_states"][key] = result
        return result

    @staticmethod
    def _history_level_positions(levels):
        """Return the current virtual positions indexed by history."""
        return {
            level.history: position
            for position, level in enumerate(levels)
        }

    def _base_level_positions(self):
        """Cache label/history positions for the original first-degree MPO."""
        if self._base_level_position_cache is None:
            self._base_level_position_cache = tuple(
                {
                    "label": {
                        level.label: position
                        for position, level in enumerate(levels)
                    },
                    "history": {
                        level.history: position
                        for position, level in enumerate(levels)
                    },
                    "number": {
                        number: position
                        for position, level in enumerate(levels)
                        for number in [_level_number(level.history[0])]
                        if number in (1, 3)
                    },
                }
                for levels in self._levels
            )
        return self._base_level_position_cache

    def _structural_transitions_from_automaton(self, automaton):
        """Map automaton edges onto the raw-history boundary schemas."""
        transitions = []
        schemas = self._history_schemas()
        for site in range(self.L):
            left_schema = schemas[site]
            right_schema = schemas[site + 1]
            left_labels = {level.label: level for level in left_schema}
            right_labels = {level.label: level for level in right_schema}
            edges = set()
            for transition, _implicit in automaton._materialized_transitions(site):
                if (
                    transition.left_state in left_labels
                    and transition.right_state in right_labels
                ):
                    edges.add((transition.left_state, transition.right_state))

            # The finite Hamiltonian's last tensor is right-boundary
            # contracted onto level 3, while the transformed evolution MPO
            # uses an all-one right boundary. The batched local-product
            # executor supplies this synthetic identity edge, so include it
            # in the structural graph as well.
            if site == self.L - 1:
                start_level = next(
                    level
                    for level in left_schema
                    if _level_number(level.history[0]) == 1
                )
                if start_level.label in right_labels:
                    edges.add((start_level.label, start_level.label))
            transitions.append(frozenset(edges))
        return tuple(transitions)

    def _reachable_history_states(
        self,
        schemas,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
    ):
        """Generate only raw histories reachable from the left boundary.

        The previous reference implementation materialized every Cartesian
        product at every cut, including channels that cannot be reached from
        the finite-chain all-one left boundary.  This forward graph walk keeps
        the same factor ordering and level metadata while avoiding those dead
        states.  Direct array construction, which has no source automaton,
        falls back to the complete local schema for compatibility.
        """
        start_levels = tuple(
            level
            for level in schemas[0]
            if _level_number(level.history[0]) == 1
        )
        if len(start_levels) != 1:
            raise ValueError(
                "history construction requires one level-1 starting channel."
            )
        warned = False

        def check_bond_limit(count, site):
            nonlocal warned
            if max_bond is None or count <= max_bond or on_exceed == "ignore":
                return
            message = (
                "extensive_exponential history bond dimension "
                f"{count} exceeds max_bond={max_bond} after site {site}."
            )
            if on_exceed == "raise":
                raise MemoryError(message)
            if not warned:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                warned = True

        state_lists = [
            (tuple(start_levels[0] for _ in range(exponent)),)
        ]

        for site in range(self.L):
            right_schema = schemas[site + 1]
            edges = (
                None
                if self._structural_transitions is None
                else self._structural_transitions[site]
            )
            next_states = []
            seen = set()
            for left_state in state_lists[-1]:
                options = []
                for left_level in left_state:
                    if edges is None:
                        level_options = right_schema
                    else:
                        level_options = tuple(
                            level
                            for level in right_schema
                            if (left_level.label, level.label) in edges
                        )
                    if not level_options:
                        break
                    options.append(level_options)
                else:
                    for right_state in product(*options):
                        key = tuple(level.label for level in right_state)
                        if key not in seen:
                            seen.add(key)
                            next_states.append(right_state)
                            check_bond_limit(len(next_states), site)
            if not next_states:
                raise ValueError(
                    f"raw-history construction found no reachable states after "
                    f"site {site}."
                )
            state_lists.append(tuple(next_states))
        return state_lists

    def _history_topology(
        self,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
    ):
        """Return cached raw history states and whether the cache was used.

        The state graph is structural: it depends on the base MPO channels,
        not on ``dt`` or on the numerical values in the local operator
        blocks.  Generate it without a limit on the first request, then
        apply the caller's per-request safety policy to the cached result.
        This keeps a failed small ``max_bond`` request from poisoning the
        reusable topology cache.
        """
        exponent = int(exponent)
        state_lists = (
            self._history_topology_cache.get(exponent)
            if cache_history
            else None
        )
        cache_hit = state_lists is not None
        guarded_generation = False
        if state_lists is None:
            guarded_generation = max_bond is not None and on_exceed != "ignore"
            state_lists = self._reachable_history_states(
                self._history_schemas(),
                exponent,
                max_bond=max_bond if guarded_generation else None,
                on_exceed=on_exceed if guarded_generation else "ignore",
            )
        if cache_history and not cache_hit:
            self._history_topology_cache[exponent] = state_lists

        # A first-time guarded generation already enforced the policy while
        # walking the graph. Rechecking it would duplicate warnings; cached
        # topologies still need the caller's per-request policy applied here.
        if guarded_generation:
            return state_lists, cache_hit

        warned = False
        if max_bond is not None and on_exceed != "ignore":
            for site, states in enumerate(state_lists[1:]):
                count = len(states)
                if count <= max_bond:
                    continue
                message = (
                    "extensive_exponential history bond dimension "
                    f"{count} exceeds max_bond={max_bond} after site {site}."
                )
                if on_exceed == "raise":
                    raise MemoryError(message)
                if not warned:
                    warnings.warn(message, RuntimeWarning, stacklevel=3)
                    warned = True
        return state_lists, cache_hit

    def _history_levels_for_states(self, state_lists, exponent):
        """Convert raw state tuples into the public level metadata."""
        return [
            [
                MPOLevel(
                    ("raw-history", exponent, bond, pos),
                    tuple(
                        token
                        for factor in state
                        for token in factor.history
                    ),
                    charge=tuple(factor.charge for factor in state),
                )
                for pos, state in enumerate(states)
            ]
            for bond, states in enumerate(state_lists)
        ]

    def _history_state_step(
        self,
        schemas,
        left_states,
        site,
        *,
        max_bond,
        on_exceed,
        warned,
    ):
        """Advance one raw-history cut without retaining earlier cuts."""
        right_schema = schemas[site + 1]
        edges = (
            None
            if self._structural_transitions is None
            else self._structural_transitions[site]
        )
        next_states = []
        seen = set()
        for left_state in left_states:
            options = []
            for left_level in left_state:
                if edges is None:
                    level_options = right_schema
                else:
                    level_options = tuple(
                        level
                        for level in right_schema
                        if (left_level.label, level.label) in edges
                    )
                if not level_options:
                    break
                options.append(level_options)
            else:
                for right_state in product(*options):
                    key = tuple(level.label for level in right_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_states.append(right_state)
                    if max_bond is not None and len(next_states) > max_bond:
                        message = (
                            "extensive_exponential history bond dimension "
                            f"{len(next_states)} exceeds max_bond={max_bond} "
                            f"after site {site}."
                        )
                        if on_exceed == "raise":
                            raise MemoryError(message)
                        if on_exceed == "warn" and not warned[0]:
                            warnings.warn(message, RuntimeWarning, stacklevel=3)
                            warned[0] = True
        if not next_states:
            raise ValueError(
                f"raw-history construction found no reachable states after "
                f"site {site}."
            )
        return tuple(next_states)

    def _history_transition_allowed(self, site, left_state, right_state):
        """Return whether a compound history has a structural local edge."""
        if self._structural_transitions is None:
            return True
        edges = self._structural_transitions[site]
        return all(
            (left_level.label, right_level.label) in edges
            for left_level, right_level in zip(left_state, right_state)
        )

    def _history_local_position_arrays(self, site, left_states, right_states):
        """Resolve all base virtual positions for one local history batch.

        The old history builder resolved a base block independently for every
        pair of compound histories.  The lookup is structural, so resolve it
        once per factor and gather all physical blocks in one backend call.
        ``-1`` denotes the edge padding handled by
        :meth:`_history_local_product_batch_values`.
        """
        base_positions = self._base_level_positions()

        def resolve(bond, level):
            positions = base_positions[bond]
            position = positions["label"].get(level.label)
            if position is None:
                position = positions["history"].get(level.history)
            return -1 if position is None else position

        order = len(left_states[0])
        left_positions = tuple(
            np.asarray(
                [resolve(site, state[factor]) for state in left_states],
                dtype=int,
            )
            for factor in range(order)
        )
        right_positions = tuple(
            np.asarray(
                [resolve(site + 1, state[factor]) for state in right_states],
                dtype=int,
            )
            for factor in range(order)
        )
        left_numbers = tuple(
            np.asarray(
                [
                    _level_number(state[factor].history[0])
                    for state in left_states
                ],
                dtype=int,
            )
            for factor in range(order)
        )
        right_numbers = tuple(
            np.asarray(
                [
                    _level_number(state[factor].history[0])
                    for state in right_states
                ],
                dtype=int,
            )
            for factor in range(order)
        )
        return (
            left_positions,
            right_positions,
            left_numbers,
            right_numbers,
        )

    def _history_allowed_pairs(self, site, left_states, right_states, *, sparse):
        """Return virtual pairs with a nonzero structural local product."""
        if not sparse or self._structural_transitions is None:
            left = np.repeat(
                np.arange(len(left_states), dtype=int),
                len(right_states),
            )
            right = np.tile(
                np.arange(len(right_states), dtype=int),
                len(left_states),
            )
            return left, right

        left = []
        right = []
        for left_pos, left_state in enumerate(left_states):
            for right_pos, right_state in enumerate(right_states):
                if self._history_transition_allowed(
                    site, left_state, right_state,
                ):
                    left.append(left_pos)
                    right.append(right_pos)
        return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

    def _history_tensor_execution_plan(
        self,
        exponent,
        state_lists,
        *,
        sparse,
        cache_history,
    ):
        """Compile local gather metadata for one raw-history order.

        History topology caching used to leave one structural cost on every
        numerical evaluation: each site recomputed allowed pairs, base-level
        positions, and identity-rail metadata before doing the actual local
        products.  This plan contains only NumPy integer/boolean arrays and
        can therefore be shared safely by parameterized Torch and JAX calls.
        """
        key = (int(exponent), bool(sparse))
        if cache_history:
            cached = self._history_tensor_plan_cache.get(key)
            if cached is not None:
                return cached, True

        sites = []
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            left_indices, right_indices = self._history_allowed_pairs(
                site,
                left_states,
                right_states,
                sparse=sparse,
            )
            sites.append({
                "left_indices": left_indices,
                "right_indices": right_indices,
                "positions": self._history_local_position_arrays(
                    site,
                    left_states,
                    right_states,
                ),
                "total_blocks": len(left_states) * len(right_states),
            })

        plan = tuple(sites)
        if cache_history:
            self._history_tensor_plan_cache[key] = plan
        return plan, False

    def _history_local_product_batch_values(
        self,
        site,
        positions,
        left_indices,
        right_indices,
    ):
        """Evaluate many history products with batched physical matmuls."""
        array = self._arrays[site]
        left_positions, right_positions, left_numbers, right_numbers = positions
        reference = array[0, 0]
        product_block = None
        for factor in range(len(left_positions)):
            local_left = left_positions[factor][left_indices]
            local_right = right_positions[factor][right_indices]
            left_valid = local_left >= 0
            right_valid = local_right >= 0
            safe_left = np.where(left_valid, local_left, 0)
            safe_right = np.where(right_valid, local_right, 0)
            local = _backend_index_pairs(array, safe_left, safe_right)
            valid = _as_backend(left_valid & right_valid, like=local)
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )

            # The finite Hamiltonian's final tensor is contracted onto the
            # level-3 rail.  History products use the all-one boundary, so the
            # missing (1 -> 1) edge is the identity rather than zero.
            synthetic = (
                left_numbers[factor][left_indices] == 1
            ) & (
                right_numbers[factor][right_indices] == 1
            ) & ~right_valid if site == self.L - 1 else np.zeros(
                len(left_indices),
                dtype=bool,
            )
            if np.any(synthetic):
                synthetic = _as_backend(synthetic, like=local)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    ar.do("eye", self.phys_dim, like=reference),
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _history_tensor_batch(
        self,
        site,
        left_states,
        right_states,
        *,
        sparse,
        chunk_size=65536,
        execution_plan=None,
    ):
        """Build one history tensor without dispatching per virtual pair."""
        if execution_plan is None:
            left_indices, right_indices = self._history_allowed_pairs(
                site,
                left_states,
                right_states,
                sparse=sparse,
            )
            positions = self._history_local_position_arrays(
                site,
                left_states,
                right_states,
            )
        else:
            left_indices = execution_plan["left_indices"]
            right_indices = execution_plan["right_indices"]
            positions = execution_plan["positions"]
        reference = self._arrays[site]
        result = _zeros(
            (
                len(left_states),
                len(right_states),
                *reference.shape[-2:],
            ),
            like=reference,
        )
        if not len(left_indices):
            return result, 0, len(left_states) * len(right_states)

        for start in range(0, len(left_indices), chunk_size):
            stop = start + chunk_size
            values = self._history_local_product_batch_values(
                site,
                positions,
                left_indices[start:stop],
                right_indices[start:stop],
            )
            result = _scatter_set_2d(
                result,
                left_indices[start:stop],
                right_indices[start:stop],
                values,
            )
        return result, len(left_indices), len(left_states) * len(right_states)

    def _batched_history_power_data(
        self,
        exponent,
        *,
        state_lists,
        storage_mode,
        cache_hit,
        execution_plan=None,
        tensor_plan_cache_hit=False,
    ):
        """Build raw history tensors using batched local block products."""
        levels = self._history_levels_for_states(state_lists, exponent)
        arrays = []
        total_blocks = 0
        stored_blocks = 0
        sparse = storage_mode == "sparse"
        for site in range(self.L):
            array, stored, total = self._history_tensor_batch(
                site,
                state_lists[site],
                state_lists[site + 1],
                sparse=sparse,
                execution_plan=(
                    None
                    if execution_plan is None
                    else execution_plan[site]
                ),
            )
            arrays.append(array)
            stored_blocks += stored
            total_blocks += total
        storage_info = {
            "mode": storage_mode,
            "stored_blocks": stored_blocks,
            "total_blocks": total_blocks,
            "tensor_plan_cache_hit": bool(tensor_plan_cache_hit),
        }
        return arrays, levels, cache_hit, storage_info

    def _stream_history_power_data(
        self,
        exponent,
        *,
        schemas,
        max_bond,
        on_exceed,
        sparse,
    ):
        """Build non-cached history tensors with batched local products.

        The final MPO tensors still have to be materialized for Algorithms
        1--4, but this path does not retain the topology cache. Its sparse
        variant also avoids structurally impossible local block products.
        The resulting virtual arrays are still dense because Algorithms 1--4
        operate on those arrays.
        """
        state_lists = self._reachable_history_states(
            schemas,
            exponent,
            max_bond=max_bond,
            on_exceed=on_exceed,
        )
        return self._batched_history_power_data(
            exponent,
            state_lists=state_lists,
            storage_mode="sparse" if sparse else "streaming",
            cache_hit=False,
        )

    @property
    def history_cache_info(self):
        """Describe cached raw history topologies without exposing arrays."""
        return {
            "orders": tuple(sorted(self._history_topology_cache)),
            "bond_dimensions": {
                order: tuple(len(states) for states in state_lists[1:-1])
                for order, state_lists in self._history_topology_cache.items()
            },
            "compression_plan_orders": tuple(
                sorted(self._history_compression_plan_cache),
            ),
            "approximation_plan_orders": tuple(
                sorted(self._history_approximation_plan_cache),
            ),
            "tensor_plan_orders": tuple(
                sorted({order for order, _sparse in self._history_tensor_plan_cache}),
            ),
            "extension_plan_orders": tuple(
                sorted(self._history_extension_plan_cache),
            ),
            "extension_plan_batches": {
                order: len(plan["batches"])
                for order, plan in self._history_extension_plan_cache.items()
            },
        }

    def clear_history_cache(self):
        """Release cached history topologies and symbolic execution plans.

        The first-degree MPO itself remains usable.  This only clears the
        reusable structural data used by higher-order exponential calls; it
        never changes the current local tensors or their autodiff graph.
        Shared :class:`MPOBasis` templates observe the same clear operation.
        """
        self._history_topology_cache.clear()
        self._history_extension_plan_cache.clear()
        self._history_compression_plan_cache.clear()
        self._history_approximation_plan_cache.clear()
        self._history_tensor_plan_cache.clear()
        self._history_symbolic_cache = None
        return self

    def _history_power_data(
        self,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="dense",
    ):
        """Build the full virtual-history representation of ``H**exponent``.

        The ordinary :meth:`power` method intentionally keeps singleton open
        boundaries.  Algorithms 1--4 in the paper are most naturally applied
        before those boundary vectors are contracted, so this private helper
        also includes the virtual histories at both boundary cuts.  Boundary
        histories that are unreachable from a finite-chain boundary have zero
        tensor entries and are removed when the final boundary vectors are
        imposed.
        """
        if not isinstance(exponent, Integral) or int(exponent) < 1:
            raise ValueError("exponent must be a positive integer.")
        exponent = int(exponent)
        if self.metadata.get("history_valid", True) is False:
            raise ValueError(
                "history construction is unavailable after numerical "
                "fixed-rank compression."
            )
        if not self.is_first_degree:
            raise ValueError("history powers require a first-degree MPO.")
        self._first_degree_structure()

        if history_storage not in {"auto", "dense", "sparse", "streaming"}:
            raise ValueError(
                "history_storage must be one of 'auto', 'dense', 'sparse', "
                "or 'streaming'."
            )
        if history_storage == "streaming" and cache_history:
            raise ValueError(
                "history_storage='streaming' requires cache_history=False; "
                "use history_storage='auto' for cached construction."
            )
        if history_storage == "auto":
            # A local-term automaton supplies exact structural transitions.
            # Preserve only those nonzero history blocks by default; MPSKit's
            # corresponding path is sparse as well.  Directly constructed
            # MPOs have no safe structural filter and retain the dense path.
            if self._structural_transitions is not None and cache_history:
                history_storage = "sparse"
            else:
                history_storage = "streaming" if not cache_history else "dense"
        schemas = self._history_schemas()
        if history_storage in {"sparse", "streaming"}:
            if cache_history:
                state_lists, cache_hit = self._history_topology(
                    exponent,
                    max_bond=max_bond,
                    on_exceed=on_exceed,
                    cache_history=True,
                )
                tensor_plan, tensor_plan_cache_hit = (
                    self._history_tensor_execution_plan(
                        exponent,
                        state_lists,
                        sparse=history_storage == "sparse",
                        cache_history=True,
                    )
                )
                return self._batched_history_power_data(
                    exponent,
                    state_lists=state_lists,
                    storage_mode=history_storage,
                    cache_hit=cache_hit,
                    execution_plan=tensor_plan,
                    tensor_plan_cache_hit=tensor_plan_cache_hit,
                )
            return self._stream_history_power_data(
                exponent,
                schemas=schemas,
                max_bond=max_bond,
                on_exceed=on_exceed,
                sparse=history_storage == "sparse",
            )

        state_lists, cache_hit = self._history_topology(
            exponent,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
        )
        tensor_plan, tensor_plan_cache_hit = (
            self._history_tensor_execution_plan(
                exponent,
                state_lists,
                sparse=history_storage == "sparse",
                cache_history=cache_history,
            )
        )
        return self._batched_history_power_data(
            exponent,
            state_lists=state_lists,
            storage_mode=history_storage,
            cache_hit=cache_hit,
            execution_plan=tensor_plan,
            tensor_plan_cache_hit=tensor_plan_cache_hit,
        )

    def _history_levels_from_tokens(self, schemas, bond, history):
        """Resolve flattened history tokens to base levels at one cut."""
        token_levels = self._history_symbolic_data()["token_levels"][bond]
        levels = []
        for token in history:
            level = token_levels.get((token,))
            if level is None:
                return None
            levels.append(level)
        return tuple(levels)

    def _history_tokens_reachable(
        self,
        schemas,
        bond,
        history,
        cache,
    ):
        """Check one raw-history state without building its order table.

        Raw product histories are reachable exactly when each factor follows
        a valid base-MPO transition from the finite all-one left boundary.
        This factor-wise check is equivalent to the forward product walk in
        ``_reachable_history_states`` but only resolves the candidate state
        requested by Algorithm 3.
        """
        key = (bond, tuple(history))
        if key in cache:
            return cache[key]
        factor_levels = self._history_levels_from_tokens(
            schemas,
            bond,
            history,
        )
        if factor_levels is None:
            cache[key] = None
            return None
        reachable_labels = self._history_symbolic_data()["reachable_labels"]
        if reachable_labels is not None and any(
            factor_level.label not in reachable_labels[bond]
            for factor_level in factor_levels
        ):
            cache[key] = None
            return None
        cache[key] = factor_levels
        return factor_levels

    @staticmethod
    def _remove_history_column(arrays, levels, bond, source, target, coefficient):
        """Apply one sequential Algorithm-1/4 column elimination.

        The production executor uses fused transfer maps. This small
        primitive remains available for deterministic reference tests and for
        debugging a compiled elimination schedule.
        """
        left = arrays[bond - 1]
        left = _setitem(
            left,
            (slice(None), target),
            left[:, target]
            + _multiply_scalar(coefficient, left[:, source]),
        )
        arrays[bond - 1] = _drop_axis(left, axis=1, position=source)
        if bond < len(arrays):
            arrays[bond] = _drop_axis(arrays[bond], axis=0, position=source)
        levels[bond].pop(source)

    @staticmethod
    def _history_axis_groups(size, operations):
        """Compile ordered channel merges into original-index contributions."""
        groups = [{position: 1.0} for position in range(size)]
        for source, target, merge in operations:
            if merge:
                for original, weight in groups[source].items():
                    groups[target][original] = (
                        groups[target].get(original, 0.0) + weight
                    )
            groups.pop(source)
        return groups

    @staticmethod
    def _apply_history_axis_groups(array, groups, *, axis):
        """Apply all channel deletions on one virtual axis in one scatter."""
        transfer = _history_transfer_matrix(groups, int(array.shape[axis]))
        if transfer is not None:
            return _apply_history_transfer(array, transfer, axis=axis)

        sources = []
        targets = []
        weights = []
        for target, group in enumerate(groups):
            for source, weight in group.items():
                sources.append(source)
                targets.append(target)
                weights.append(weight)

        output_shape = list(array.shape)
        output_shape[axis] = len(groups)
        result = _zeros(output_shape, like=array)
        if not sources:
            return result

        source_indices = np.asarray(sources, dtype=int)
        target_indices = np.asarray(targets, dtype=int)
        if axis == 1:
            source_backend = _as_backend(source_indices, like=array)
            values = array[:, source_backend]
            rows = np.repeat(
                np.arange(array.shape[0], dtype=int),
                len(source_indices),
            )
            columns = np.tile(target_indices, array.shape[0])
        else:
            source_backend = _as_backend(source_indices, like=array)
            values = array[source_backend, :]
            rows = np.repeat(target_indices, array.shape[1])
            columns = np.tile(
                np.arange(array.shape[1], dtype=int),
                len(source_indices),
            )
        coefficient = _as_backend(np.asarray(weights), like=values)
        weight_shape = (
            (1, len(coefficient), 1, 1)
            if axis == 1
            else (len(coefficient), 1, 1, 1)
        )
        values = ar.do(
            "multiply",
            values,
            ar.do("reshape", coefficient, weight_shape),
        )
        return _scatter_add_2d(
            result,
            rows,
            columns,
            values.reshape((-1, *array.shape[-2:])),
        )

    @staticmethod
    def _history_axis_polynomial_groups(size, operations):
        """Compile channel merges whose weights are polynomials in ``dt``."""
        groups = [
            {position: [(0, 1.0)]}
            for position in range(size)
        ]
        for source, target, merge, power, coefficient in operations:
            if merge and coefficient != 0.0:
                for original, terms in groups[source].items():
                    shifted = [
                        (old_power + power, old_coefficient * coefficient)
                        for old_power, old_coefficient in terms
                    ]
                    groups[target].setdefault(original, []).extend(shifted)
            groups.pop(source)
        return groups

    @staticmethod
    def _apply_history_polynomial_groups(array, groups, dt, *, axis):
        """Apply a parameterized channel schedule in one backend scatter."""
        old_size = int(array.shape[axis])
        power_maps = {}
        for target, group in enumerate(groups):
            for source, terms in group.items():
                for power, coefficient in terms:
                    power_maps.setdefault(int(power), []).append(
                        (target, source, coefficient),
                    )
        if (
            len(groups) * old_size <= _MAX_HISTORY_TRANSFER_ELEMENTS
            and power_maps
        ):
            transfer = None
            for power, entries in power_maps.items():
                structural = np.zeros((len(groups), old_size), dtype=float)
                for target, source, coefficient in entries:
                    structural[target, source] += float(coefficient)
                structural = _as_backend(structural, like=array)
                weighted = _multiply_scalar(dt ** power, structural)
                transfer = weighted if transfer is None else ar.do(
                    "add",
                    transfer,
                    weighted,
                )
            return _apply_history_transfer(array, transfer, axis=axis)

        sources = []
        targets = []
        powers = []
        coefficients = []
        for target, group in enumerate(groups):
            for source, terms in group.items():
                for power, coefficient in terms:
                    sources.append(source)
                    targets.append(target)
                    powers.append(power)
                    coefficients.append(coefficient)

        output_shape = list(array.shape)
        output_shape[axis] = len(groups)
        result = _zeros(output_shape, like=array)
        if not sources:
            return result

        source_indices = np.asarray(sources, dtype=int)
        target_indices = np.asarray(targets, dtype=int)
        if axis == 1:
            source_backend = _as_backend(source_indices, like=array)
            values = array[:, source_backend]
            rows = np.repeat(
                np.arange(array.shape[0], dtype=int),
                len(source_indices),
            )
            columns = np.tile(target_indices, array.shape[0])
        else:
            source_backend = _as_backend(source_indices, like=array)
            values = array[source_backend, :]
            rows = np.repeat(target_indices, array.shape[1])
            columns = np.tile(
                np.arange(array.shape[1], dtype=int),
                len(source_indices),
            )

        powers = tuple(int(power) for power in powers)
        power_values = {
            power: dt ** power
            for power in set(powers)
        }
        weights = tuple(
            _multiply_scalar(power_values[power], coefficient)
            for power, coefficient in zip(powers, coefficients)
        )
        if any(
            _backend_name(value) not in {"builtins", "numpy"}
            for value in weights
        ):
            weights = ar.do("stack", weights)
        else:
            weights = np.asarray(weights)
            weights = _as_backend(weights, like=values)
        weight_shape = (
            (1, len(weights), 1, 1)
            if axis == 1
            else (len(weights), 1, 1, 1)
        )
        values = ar.do(
            "multiply",
            values,
            ar.do("reshape", weights, weight_shape),
        )
        return _scatter_add_into_2d(
            result,
            rows,
            columns,
            values.reshape((-1, *array.shape[-2:])),
        )

    def _algorithm_two_bond(self, arrays, levels, bond, actions):
        """Apply one bond's Algorithm-2 schedule as two fused transforms."""
        current = list(levels[bond])
        row_operations = []
        column_operations = []
        merges = []
        for source_history, canonical, mode, source_label in actions:
            positions = self._history_level_positions(current)
            source = positions.get(source_history)
            target = positions.get(canonical)
            if source is None or target is None or target == source:
                continue
            if mode == "row":
                if bond >= len(arrays):
                    continue
                row_operations.append((source, target, True))
                column_operations.append((source, target, False))
            else:
                column_operations.append((source, target, True))
                row_operations.append((source, target, False))
            current.pop(source)
            merges.append({
                "bond": bond - 1,
                "source": source_label,
                "target": canonical,
                "mode": mode,
                "history": source_history,
            })

        if not merges:
            return merges
        if column_operations:
            arrays[bond - 1] = self._apply_history_axis_groups(
                arrays[bond - 1],
                self._history_axis_groups(
                    arrays[bond - 1].shape[1],
                    column_operations,
                ),
                axis=1,
            )
        if row_operations and bond < len(arrays):
            arrays[bond] = self._apply_history_axis_groups(
                arrays[bond],
                self._history_axis_groups(
                    arrays[bond].shape[0],
                    row_operations,
                ),
                axis=0,
            )
        levels[bond] = current
        return merges

    def _history_compression_plan(self, levels, order, *, cache_history):
        """Compile Algorithms 1--2 as a topology-only elimination plan.

        The plan is generated by simulating only virtual histories.  Applying
        it later performs the same ordered eliminations on the current
        numerical arrays, with ``dt`` supplied at execution time.  This is
        the key separation needed by parameter optimization: the expensive
        history decisions are cached, while all values remain differentiable
        and are recomputed for each call.
        """
        if cache_history:
            cached = self._history_compression_plan_cache.get(order)
            if cached is not None:
                return cached, True

        working = [list(bond_levels) for bond_levels in levels]
        algorithm_one = []
        target_history = tuple(MPOLevelToken(1) for _ in range(order))
        for bond in range(1, self.L + 1):
            for number_of_threes in range(1, order + 1):
                for level in tuple(working[bond]):
                    history = level.history
                    if not (
                        all(
                            _level_number(token) in (1, 3)
                            for token in history
                        )
                        and sum(
                            _level_number(token) == 3 for token in history
                        ) == number_of_threes
                    ):
                        continue
                    positions = self._history_level_positions(working[bond])
                    source = positions.get(history)
                    target = positions.get(target_history)
                    if source is None or target is None or source == target:
                        raise ValueError(
                            "history power lost its all-one Algorithm-1 target."
                        )
                    algorithm_one.append((
                        bond,
                        history,
                        target_history,
                        number_of_threes,
                    ))
                    working[bond].pop(source)

        algorithm_two = []
        for bond in range(1, self.L + 1):
            changed = True
            while changed:
                changed = False
                for level in tuple(working[bond]):
                    history = level.history
                    number_of_ones = sum(
                        _level_number(token) == 1 for token in history
                    )
                    number_of_threes = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if number_of_threes <= number_of_ones:
                        canonical = _sort_history_front(history, 1)
                        mode = "row"
                    else:
                        canonical = _sort_history_front(history, 3)
                        mode = "column"
                    if canonical == history:
                        continue
                    positions = self._history_level_positions(working[bond])
                    source = positions.get(history)
                    target = positions.get(canonical)
                    if source is None or target is None or source == target:
                        continue
                    algorithm_two.append((
                        bond,
                        history,
                        canonical,
                        mode,
                        level.label,
                    ))
                    working[bond].pop(source)
                    changed = True
                    break

        plan = {
            "algorithm_one": tuple(algorithm_one),
            "algorithm_two": tuple(algorithm_two),
        }
        if cache_history:
            self._history_compression_plan_cache[order] = plan
        return plan, False

    def _algorithm_one(self, arrays, levels, order, dt, *, plan=None):
        """Apply the paper's extensive prefactor transformation.

        This pass removes all-identity/all-level-3 histories into the all-one
        target with the paper's factorial coefficient.  It is intentionally
        separate from Algorithm 2 so a report can distinguish coefficient
        rewiring from exact equal-history compression.
        """
        if plan is None:
            plan, _ = self._history_compression_plan(
                levels, order, cache_history=False,
            )
        coefficient_denominator = factorial(order)
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for action in plan["algorithm_one"]:
            actions_by_bond[action[0]].append(action[1:])

        # Compile one polynomial virtual transfer per bond.  This is
        # algebraically identical to the ordered column eliminations, but it
        # replaces one full tensor copy per history channel by one backend
        # contraction per side of the cut.
        for bond, actions in enumerate(actions_by_bond):
            if not actions:
                continue
            current = list(levels[bond])
            polynomial_operations = []
            removal_operations = []
            for (
                source_history,
                target_history,
                number_of_threes,
            ) in actions:
                positions = self._history_level_positions(current)
                source = positions.get(source_history)
                target = positions.get(target_history)
                if source is None:
                    continue
                if target is None or target == source:
                    raise ValueError(
                        "history power lost its all-one Algorithm-1 target."
                    )
                polynomial_operations.append((
                    source,
                    target,
                    True,
                    number_of_threes,
                    factorial(order - number_of_threes)
                    / coefficient_denominator,
                ))
                removal_operations.append((source, target, False))
                current.pop(source)

            if not polynomial_operations:
                continue
            arrays[bond - 1] = self._apply_history_polynomial_groups(
                arrays[bond - 1],
                self._history_axis_polynomial_groups(
                    arrays[bond - 1].shape[1],
                    polynomial_operations,
                ),
                dt,
                axis=1,
            )
            if bond < len(arrays):
                arrays[bond] = self._apply_history_axis_groups(
                    arrays[bond],
                    self._history_axis_groups(
                        arrays[bond].shape[0],
                        removal_operations,
                    ),
                    axis=0,
                )
            levels[bond] = current

    def _algorithm_two(self, arrays, levels, *, plan=None):
        """Apply the paper's exact history-only compression transformations.

        The implementation chooses the row or column orientation from the
        number of level-1 and level-3 tokens.  This is a structural rule from
        the paper, not a numerical rank heuristic; no backend conversion or
        tolerance is involved.
        """
        if plan is None:
            plan, _ = self._history_compression_plan(
                levels, len(levels[0][0].history), cache_history=False,
            )
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for bond, source_history, canonical, mode, source_label in plan[
            "algorithm_two"
        ]:
            actions_by_bond[bond].append(
                (source_history, canonical, mode, source_label),
            )

        merges = []
        for bond, actions in enumerate(actions_by_bond):
            if actions:
                merges.extend(
                    self._algorithm_two_bond(arrays, levels, bond, actions),
                )
        return merges

    def _base_level_position(self, bond, level):
        """Resolve a symbolic base level to its original virtual position."""
        positions = self._base_level_positions()[bond]
        position = positions["label"].get(level.label)
        if position is not None:
            return position
        return positions["history"].get(level.history)

    def _history_extension_plan(self, levels, order, *, cache_history):
        """Compile Algorithm 3 as batched local insertion transitions.

        The naive pseudocode has a pair of history loops inside every site.
        The selected contribution is separable at the virtual level: for a
        fixed insertion position, one only needs the valid left state list,
        the valid right state list, and the local base-block positions for
        each factor. Store those short lists and let execution perform one
        batched physical-matrix product per site. The plan avoids Python
        objects for every scalar transition, but its flattened site plans
        still materialize every selected left/right pair; their memory scales
        with the selected pair count. A future blockwise executor can remove
        that remaining materialization without changing the symbolic plan.
        """
        if cache_history:
            cached = self._history_extension_plan_cache.get(order)
            if cached is not None:
                return cached, True

        symbolic = self._history_symbolic_data()
        schemas = symbolic["schemas"]
        snapshot = [tuple(bond_levels) for bond_levels in levels]
        one = MPOLevelToken(1)
        three = MPOLevelToken(3)
        batches = []
        batches_by_site = [[] for _ in range(self.L)]
        selected_terms = 0

        for site in range(self.L):
            left_levels = snapshot[site]
            right_levels = snapshot[site + 1]
            left_candidates = []
            for left_pos, left_level in enumerate(left_levels):
                history = left_level.history
                numbers = _history_signature(history)
                if all(number in (1, 3) for number in numbers) and 3 in numbers:
                    continue
                left_candidates.append((left_pos, history, numbers))
            right_candidates = [
                (right_pos, right_level.history, _history_signature(right_level.history))
                for right_pos, right_level in enumerate(right_levels)
                if all(number > 1 for number in _history_signature(right_level.history))
            ]
            if not left_candidates or not right_candidates:
                continue

            left_insertions = []
            for insert_position in range(order + 1):
                entries = []
                for left_pos, history, numbers in left_candidates:
                    extended = (
                        history[:insert_position]
                        + (one,)
                        + history[insert_position:]
                    )
                    state = self._history_tokens_reachable(
                        schemas,
                        site,
                        extended,
                        {},
                    )
                    if state is not None:
                        entries.append((left_pos, numbers, state))
                if entries:
                    left_insertions.append(tuple(entries))
                else:
                    left_insertions.append(())

            right_insertions = []
            for insert_position in range(order + 1):
                entries = []
                for right_pos, history, numbers in right_candidates:
                    extended = (
                        history[:insert_position]
                        + (three,)
                        + history[insert_position:]
                    )
                    state = self._history_tokens_reachable(
                        schemas,
                        site + 1,
                        extended,
                        {},
                    )
                    if state is not None:
                        entries.append((right_pos, numbers, state))
                if entries:
                    right_insertions.append(tuple(entries))
                else:
                    right_insertions.append(())

            for left_entries in left_insertions:
                if not left_entries:
                    continue
                for right_entries in right_insertions:
                    if not right_entries:
                        continue
                    left_targets = np.asarray(
                        [entry[0] for entry in left_entries],
                        dtype=int,
                    )
                    right_targets = np.asarray(
                        [entry[0] for entry in right_entries],
                        dtype=int,
                    )
                    left_weights = np.asarray(
                        [
                            1.0 / ((order + 1) * (entry[1].count(1) + 1))
                            for entry in left_entries
                        ],
                        dtype=float,
                    )
                    right_weights = np.asarray(
                        [1.0 / (entry[1].count(3) + 1) for entry in right_entries],
                        dtype=float,
                    )
                    left_states = tuple(entry[2] for entry in left_entries)
                    right_states = tuple(entry[2] for entry in right_entries)
                    left_positions = tuple(
                        np.asarray(
                            [
                                self._base_level_position(site, state[factor])
                                if self._base_level_position(site, state[factor]) is not None
                                else -1
                                for state in left_states
                            ],
                            dtype=int,
                        )
                        for factor in range(order + 1)
                    )
                    right_positions = tuple(
                        np.asarray(
                            [
                                self._base_level_position(site + 1, state[factor])
                                if self._base_level_position(site + 1, state[factor]) is not None
                                else -1
                                for state in right_states
                            ],
                            dtype=int,
                        )
                        for factor in range(order + 1)
                    )
                    left_identity = tuple(
                        np.asarray(
                            [
                                _level_number(state[factor].history[0]) == 1
                                for state in left_states
                            ],
                            dtype=bool,
                        )
                        for factor in range(order + 1)
                    )
                    right_identity = tuple(
                        np.asarray(
                            [
                                _level_number(state[factor].history[0]) == 1
                                for state in right_states
                            ],
                            dtype=bool,
                        )
                        for factor in range(order + 1)
                    )
                    weight_matrix = left_weights[:, None] * right_weights[None, :]
                    batch = {
                        "site": site,
                        "left_targets": left_targets,
                        "right_targets": right_targets,
                        "left_positions": left_positions,
                        "right_positions": right_positions,
                        "left_identity": left_identity,
                        "right_identity": right_identity,
                        "weights": weight_matrix,
                    }
                    batches.append(batch)
                    batches_by_site[site].append(batch)
                    selected_terms += int(left_targets.size * right_targets.size)

        site_plans = []
        for site, site_batches in enumerate(batches_by_site):
            if not site_batches:
                site_plans.append(None)
                continue

            left_targets = []
            right_targets = []
            weights = []
            left_positions = [[] for _ in range(order + 1)]
            right_positions = [[] for _ in range(order + 1)]
            left_identity = [[] for _ in range(order + 1)]
            right_identity = [[] for _ in range(order + 1)]
            for batch in site_batches:
                left_size = batch["left_targets"].size
                right_size = batch["right_targets"].size
                left_targets.append(
                    np.repeat(batch["left_targets"], right_size),
                )
                right_targets.append(
                    np.tile(batch["right_targets"], left_size),
                )
                weights.append(batch["weights"].reshape(-1))
                for factor in range(order + 1):
                    left_positions[factor].append(
                        np.repeat(
                            batch["left_positions"][factor],
                            right_size,
                        ),
                    )
                    right_positions[factor].append(
                        np.tile(
                            batch["right_positions"][factor],
                            left_size,
                        ),
                    )
                    left_identity[factor].append(
                        np.repeat(
                            batch["left_identity"][factor],
                            right_size,
                        ),
                    )
                    right_identity[factor].append(
                        np.tile(
                            batch["right_identity"][factor],
                            left_size,
                        ),
                    )

            site_plans.append({
                "site": site,
                "left_targets": np.concatenate(left_targets),
                "right_targets": np.concatenate(right_targets),
                "left_positions": tuple(
                    np.concatenate(values) for values in left_positions
                ),
                "right_positions": tuple(
                    np.concatenate(values) for values in right_positions
                ),
                "left_identity": tuple(
                    np.concatenate(values) for values in left_identity
                ),
                "right_identity": tuple(
                    np.concatenate(values) for values in right_identity
                ),
                "weights": np.concatenate(weights),
            })

        plan = {
            "batches": tuple(batches),
            "site_plans": tuple(site_plans),
            "selected_terms": selected_terms,
        }
        if cache_history:
            self._history_extension_plan_cache[order] = plan
        return plan, False

    def _history_local_product_batch(self, batch, order):
        """Evaluate one Algorithm-3 batch as a backend-native matrix product."""
        site = batch["site"]
        reference = self._arrays[site][0, 0]
        product_block = None
        for factor in range(order + 1):
            left_positions = batch["left_positions"][factor]
            right_positions = batch["right_positions"][factor]
            left_valid = left_positions >= 0
            right_valid = right_positions >= 0
            safe_left = np.where(left_valid, left_positions, 0)
            safe_right = np.where(right_valid, right_positions, 0)
            local = _backend_index_2d(
                self._arrays[site],
                safe_left,
                safe_right,
            )
            valid = _as_backend(
                left_valid[:, None] & right_valid[None, :],
                like=local,
            )
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )
            synthetic = _as_backend(
                batch["left_identity"][factor][:, None]
                & (~right_valid[None, :])
                & batch["right_identity"][factor][None, :],
                like=local,
            )
            if bool(np.any(batch["right_identity"][factor] & ~right_valid)):
                identity = ar.do("eye", self.phys_dim, like=reference)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    identity,
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _history_local_product_site_batch(self, site_plan, order):
        """Evaluate every selected Algorithm-3 product at one site together."""
        site = site_plan["site"]
        reference = self._arrays[site][0, 0]
        product_block = None
        for factor in range(order + 1):
            left_positions = site_plan["left_positions"][factor]
            right_positions = site_plan["right_positions"][factor]
            left_valid = left_positions >= 0
            right_valid = right_positions >= 0
            safe_left = np.where(left_valid, left_positions, 0)
            safe_right = np.where(right_valid, right_positions, 0)
            local = _backend_index_pairs(
                self._arrays[site],
                safe_left,
                safe_right,
            )
            valid = _as_backend(
                left_valid & right_valid,
                like=local,
            )
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )
            synthetic = _as_backend(
                site_plan["left_identity"][factor]
                & (~right_valid)
                & site_plan["right_identity"][factor],
                like=local,
            )
            if bool(
                np.any(
                    site_plan["right_identity"][factor] & ~right_valid,
                )
            ):
                identity = ar.do("eye", self.phys_dim, like=reference)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    identity,
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _algorithm_three_extension(
        self,
        arrays,
        levels,
        order,
        dt,
        *,
        cache_history=True,
    ):
        """Add Algorithm 3's selected ``N + 1`` local history transitions.

        The topology-only insertion plan is cached; this numerical pass only
        evaluates selected local products and scatters them into the existing
        order-``N`` tensor.  Thus parameter rebinding does not reuse a stale
        backend value or autodiff graph.
        """
        plan, cache_hit = self._history_extension_plan(
            levels,
            order,
            cache_history=cache_history,
        )
        for site_plan in plan["site_plans"]:
            if site_plan is None:
                continue
            site = site_plan["site"]
            extension = self._history_local_product_site_batch(
                site_plan,
                order,
            )
            weights = _as_backend(site_plan["weights"], like=extension)
            extension = ar.do(
                "multiply",
                extension,
                weights[..., None, None],
            )
            arrays[site] = _scatter_add_2d(
                arrays[site],
                site_plan["left_targets"],
                site_plan["right_targets"],
                _multiply_scalar(dt, extension),
            )
        return plan["selected_terms"], cache_hit

    def _history_approximation_plan(self, levels, order, *, cache_history):
        """Compile Algorithm 4's ordered structural merge actions."""
        if cache_history:
            cached = self._history_approximation_plan_cache.get(order)
            if cached is not None:
                return cached, True

        working = [list(bond_levels) for bond_levels in levels]
        actions = []
        for bond in range(1, self.L + 1):
            while True:
                positions = self._history_level_positions(working[bond])
                source_history = None
                canonical = None
                number_of_threes = None
                for level in tuple(working[bond]):
                    history = level.history
                    if any(_level_number(token) == 1 for token in history):
                        continue
                    count = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if count == 0:
                        continue
                    candidate = tuple(
                        MPOLevelToken(
                            1 if _level_number(token) == 3 else token.level,
                            token.payload,
                        )
                        for token in history
                    )
                    source = positions.get(history)
                    target = positions.get(candidate)
                    if source is None or target is None or source == target:
                        continue
                    source_history = history
                    canonical = candidate
                    number_of_threes = count
                    break
                if source_history is None:
                    break
                actions.append((
                    bond,
                    positions[source_history],
                    positions[canonical],
                    number_of_threes,
                ))
                working[bond].pop(positions[source_history])

        plan = {"actions": tuple(actions)}
        if cache_history:
            self._history_approximation_plan_cache[order] = plan
        return plan, False

    def _algorithm_four(
        self,
        arrays,
        levels,
        order,
        dt,
        *,
        cache_history=True,
    ):
        """Apply the paper's order-controlled approximate compression.

        The merge schedule is structural and compiled once per Taylor order.
        Coefficients are evaluated during every numerical pass so ``dt`` can
        remain a backend value connected to an autodiff graph.
        """
        plan, cache_hit = self._history_approximation_plan(
            levels,
            order,
            cache_history=cache_history,
        )
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for bond, source, target, number_of_threes in plan["actions"]:
            actions_by_bond[bond].append(
                (source, target, number_of_threes),
            )

        removed = 0
        for bond, actions in enumerate(actions_by_bond):
            current = list(levels[bond])
            operations = []
            for source, target, number_of_threes in actions:
                # The approximation plan stores positions in the working
                # level list at the point where each action was compiled.
                # They therefore remain valid as long as we replay the
                # actions in their original order, just like the sequential
                # implementation did.
                if (
                    source >= len(current)
                    or target >= len(current)
                    or source == target
                ):
                    continue
                if number_of_threes <= order:
                    coefficient = (
                        factorial(order - number_of_threes)
                        / factorial(order)
                    )
                    power = number_of_threes
                else:
                    coefficient = 0.0
                    power = 0
                operations.append((source, target, True, power, coefficient))
                current.pop(source)

            if not operations:
                continue
            arrays[bond - 1] = self._apply_history_polynomial_groups(
                arrays[bond - 1],
                self._history_axis_polynomial_groups(
                    arrays[bond - 1].shape[1],
                    operations,
                ),
                dt,
                axis=1,
            )
            if bond < len(arrays):
                # A column elimination also removes the matching row from
                # the tensor on the right of the cut.  Keep this as a fused
                # gather so Algorithm 4 does not fall back to one full-tensor
                # copy per merge.
                arrays[bond] = self._apply_history_axis_groups(
                    arrays[bond],
                    self._history_axis_groups(
                        arrays[bond].shape[0],
                        [
                            (source, target, False)
                            for source, target, _merge, _power, _coefficient
                            in operations
                        ],
                    ),
                    axis=0,
                )
            levels[bond] = current
            removed += len(operations)
        return removed, cache_hit

    def _contract_history_boundaries(self, arrays, levels, order):
        """Impose the finite-chain all-one boundary vectors.

        Boundary contraction is delayed until after Algorithms 1--4 because
        those algorithms need the histories at both open cuts.  This ordering
        is essential to the finite-chain implementation and avoids treating
        unreachable edge states as physical channels.
        """
        boundary_history = tuple(MPOLevelToken(1) for _ in range(order))
        left_target = self._history_level_positions(levels[0]).get(
            boundary_history,
        )
        right_target = self._history_level_positions(levels[-1]).get(
            boundary_history,
        )
        if left_target is None or right_target is None:
            raise ValueError("history construction lost a finite boundary state.")

        first = arrays[0]
        arrays[0] = _stack([first[left_target]], axis=0)
        levels[0] = [MPOLevel(("boundary", "left", order), boundary_history)]

        last = arrays[-1]
        arrays[-1] = _stack([last[:, right_target]], axis=1)
        levels[-1] = [MPOLevel(("boundary", "right", order), boundary_history)]

    def _extensive_history_exponential(
        self,
        dt,
        *,
        order,
        extend=False,
        approximate=False,
        mode="base",
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Construct an arbitrary-order MPO using Algorithms 1--4.

        This is the single multi-site execution path.  The order of passes is
        part of the paper implementation contract: optional Algorithm 3 first
        adds selected next-order terms, Algorithm 1 rewires extensive
        prefactors, Algorithm 2 performs exact compression, and optional
        Algorithm 4 applies the analytical approximation.
        """
        arrays, levels, history_cache_hit, storage_info = self._history_power_data(
            order,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        )
        initial_bond_dimensions = tuple(
            len(bond_levels) for bond_levels in levels[1:-1]
        )
        compression_plan, compression_plan_cache_hit = (
            self._history_compression_plan(
                levels,
                order,
                cache_history=cache_history,
            )
        )
        extension_terms = 0
        extension_plan_cache_hit = False
        if extend:
            extension_terms, extension_plan_cache_hit = (
                self._algorithm_three_extension(
                    arrays,
                    levels,
                    order,
                    dt,
                    cache_history=cache_history,
                )
            )
        self._algorithm_one(
            arrays,
            levels,
            order,
            dt,
            plan=compression_plan,
        )
        exact_merges = self._algorithm_two(
            arrays,
            levels,
            plan=compression_plan,
        )
        approximate_merges = 0
        approximation_plan_cache_hit = False
        if approximate:
            approximate_merges, approximation_plan_cache_hit = (
                self._algorithm_four(
                    arrays,
                    levels,
                    order,
                    dt,
                    cache_history=cache_history,
                )
            )
        self._contract_history_boundaries(arrays, levels, order)
        final_bond_dimensions = tuple(
            len(bond_levels) for bond_levels in levels[1:-1]
        )
        metadata = {
            "operation": "extensive_exponential",
            "dt": dt,
            "order": order,
            "mode": mode,
            "max_bond": max_bond,
            "on_exceed": on_exceed,
            "cache_history": bool(cache_history),
            "history_storage": storage_info["mode"],
            "history_storage_requested": history_storage,
            "history_storage_blocks": storage_info,
            "tensor_plan_cache_hit": storage_info.get(
                "tensor_plan_cache_hit",
                False,
            ),
            "initial_bond_dimensions": initial_bond_dimensions,
            "history_generation": (
                "reachable" if self._structural_transitions is not None else "cartesian"
            ),
            "history_cache_hit": history_cache_hit,
            "compression_plan_cache_hit": compression_plan_cache_hit,
            "extension_plan_cache_hit": extension_plan_cache_hit,
            "approximation_plan_cache_hit": approximation_plan_cache_hit,
            "algorithms": (1, 2) + ((3,) if extend else ()) + ((4,) if approximate else ()),
            "exact_history_merges": len(exact_merges),
            "approximate_history_merges": approximate_merges,
            "extension_terms": extension_terms,
            "approximate": bool(approximate),
        }
        output = type(self)(
            arrays,
            levels=levels,
            degree=order,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=metadata,
        )
        report = MPOCompressionReport(
            method="paper-history",
            exact=not approximate,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            merged_channels=len(exact_merges) + approximate_merges,
            merges=tuple(exact_merges),
        )
        output.compression_report = report
        output.metadata["compression_report"] = report
        if not cache_history:
            # One-off builds should not leave the raw topology, insertion
            # states, or execution plans attached to the source object.
            self._history_topology_cache.pop(order, None)
            self._history_symbolic_cache = None
            self._history_extension_plan_cache.pop(order, None)
            self._history_compression_plan_cache.pop(order, None)
            self._history_approximation_plan_cache.pop(order, None)
            self._history_tensor_plan_cache.pop((order, True), None)
            self._history_tensor_plan_cache.pop((order, False), None)
        return output

    def _first_degree_structure(self):
        """Validate and return active channel counts on each virtual bond."""
        if self.L == 1:
            return (0,)
        if _level_number(self._levels[0][0].history[0]) != 1:
            raise ValueError("the left MPO boundary must be level 1.")
        if _level_number(self._levels[-1][0].history[0]) != 3:
            raise ValueError("the Hamiltonian MPO right boundary must be level 3.")
        active_counts = []
        for bond, levels in enumerate(self._levels[1:-1], start=1):
            numbers = [_level_number(level.history[0]) for level in levels]
            if numbers.count(1) != 1 or numbers.count(3) != 1:
                raise ValueError(
                    "extensive_exponential requires one level-1 and one level-3 "
                    f"rail on internal bond {bond}, got {numbers!r}."
                )
            if any(number not in (1, 2, 3) for number in numbers):
                raise ValueError("first-degree virtual levels must be 1, 2, or 3.")
            active_counts.append(numbers.count(2))
        return (0, *active_counts, 0)

    def extensive_exponential(
        self,
        dt,
        *,
        order=1,
        extend=False,
        approximate=False,
        mode=None,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Build the paper's size-extensive higher-order MPO.

        The construction is local in the MPO tensors.  It contracts only
        physical operator blocks at each site and assembles the new virtual
        channels with ``stack``; it never forms ``H`` or ``exp(dt * H)`` as a
        global dense matrix.

        Parameters
        ----------
        dt : scalar
            Time-step or imaginary-time parameter ``tau``.
        order : int, default=1
            Taylor order. Multi-site chains use the generic history engine for
            every positive order. One-site chains use a direct local Taylor
            polynomial at arbitrary order.
        extend : bool, default=False
            Include Algorithm 3's selected order ``N + 1`` terms without
            increasing the analytical history bond dimension.
        approximate : bool, default=False
            Apply Algorithm 4's order-controlled analytical compression after
            exact history compression. This is not a numerical cutoff.
        mode : {None, "base", "algorithm4", "optimal", "approximate"}, optional
            Named construction policy. ``"base"`` selects Algorithms 1--2,
            ``"algorithm4"`` selects Algorithms 1, 2, and 4,
            ``"optimal"`` selects the paper's exact extended construction
            (Algorithms 1--3), and ``"approximate"`` selects the extended
            construction followed by Algorithm 4. When omitted, the legacy
            ``extend`` and ``approximate`` flags are used unchanged. The
            compatibility spellings ``"paper_algorithm4"``,
            ``"paper_optimal"``, and ``"paper_approximate"`` are accepted but
            normalized to the canonical names in metadata.
        max_bond : int, optional
            Maximum temporary history bond dimension allowed during
            construction. This guard applies before exact history compression
            and therefore protects the allocation stage. ``None`` disables
            the guard.
        on_exceed : {"raise", "warn", "ignore"}, default="raise"
            Action when ``max_bond`` is exceeded. ``"raise"`` stops before
            completing the oversized history table, ``"warn"`` continues,
            and ``"ignore"`` disables the action while retaining the value in
            metadata.
        cache_history : bool, default=True
            Retain the raw reachable history topology for later calls. Set
            this to ``False`` for large-order or one-off constructions so the
            topology is released after the current MPO is assembled. With the
            default ``history_storage="auto"``, this also selects the
            streaming local-history builder.
        history_storage : {"auto", "dense", "sparse", "streaming"}, default="auto"
            Storage policy for temporary raw-history tensors. ``"dense"``
            retains all structural local pairs. ``"sparse"`` skips
            structurally impossible local transition products and batches the
            remaining physical block products. ``"streaming"`` keeps the
            topology ephemeral when ``cache_history=False``. For an
            automaton-built MPO, ``"auto"`` selects the structural sparse path
            for cached builds and the compatibility streaming path otherwise;
            direct MPO construction retains the dense/streaming policy. The
            final MPO tensors are still dense virtual arrays for Algorithms
            1--4.

        Notes
        -----
        The first implementation targets ordinary NumPy/Autoray-compatible
        local MPO blocks. Native fermionic/Symmray compilation is deliberately
        not enabled by this method yet. The one-site path is exact through its
        requested local Taylor order; Algorithm 3/4 have no virtual history to
        extend or merge there.
        """
        _check_scalar(dt, name="dt")
        if not isinstance(order, Integral) or int(order) < 1:
            raise ValueError("order must be a positive integer.")
        order = int(order)
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        if on_exceed not in {"raise", "warn", "ignore"}:
            raise ValueError(
                "on_exceed must be one of 'raise', 'warn', or 'ignore'."
            )
        if not isinstance(cache_history, bool):
            raise TypeError("cache_history must be a boolean.")
        if history_storage not in {"auto", "dense", "sparse", "streaming"}:
            raise ValueError(
                "history_storage must be one of 'auto', 'dense', 'sparse', "
                "or 'streaming'."
            )
        if history_storage == "streaming" and cache_history:
            raise ValueError(
                "history_storage='streaming' requires cache_history=False; "
                "use history_storage='auto' for cached construction."
            )
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("mode must be a string or None.")
            mode_aliases = {
                "base": (False, False, "base"),
                "algorithm4": (False, True, "algorithm4"),
                "paper_algorithm4": (False, True, "algorithm4"),
                "optimal": (True, False, "optimal"),
                "paper_optimal": (True, False, "optimal"),
                "approximate": (True, True, "approximate"),
                "paper_approximate": (True, True, "approximate"),
            }
            try:
                mode_extend, mode_approximate, canonical_mode = mode_aliases[mode]
            except KeyError as exc:
                allowed = ", ".join(sorted(mode_aliases))
                raise ValueError(
                    f"unknown mode {mode!r}; expected one of {allowed}."
                ) from exc
            if extend or approximate:
                raise ValueError(
                    "mode cannot be combined with extend or approximate flags."
                )
            extend = mode_extend
            approximate = mode_approximate
        else:
            canonical_mode = (
                "approximate" if approximate and extend
                else "optimal" if extend
                else "algorithm4" if approximate
                else "base"
            )
        if self.L > 1:
            return self._extensive_history_exponential(
                dt,
                order=order,
                extend=extend,
                approximate=approximate,
                mode=canonical_mode,
                max_bond=max_bond,
                on_exceed=on_exceed,
                cache_history=cache_history,
                history_storage=history_storage,
            )
        # A one-site operator has no non-trivial virtual history. Evaluate its
        # requested Taylor polynomial directly, with Algorithm 3's extension
        # represented by one additional local Taylor term.
        effective_order = order + int(extend)
        reference = self._arrays[0][0, 0]
        identity = ar.do("eye", self.phys_dim, like=reference)
        h = reference
        data = identity
        power = identity
        for power_order in range(1, effective_order + 1):
            power = ar.do("matmul", power, h)
            data = data + _multiply_scalar(
                dt ** power_order / factorial(power_order),
                power,
            )
        return type(self)(
            (data,),
            levels=[[
                MPOLevel(
                    ("extensive", effective_order, 0, ("11", None, None)),
                    tuple(
                        self._levels[0][0].history[0]
                        for _ in range(effective_order)
                    ),
                )
            ]] * 2,
            degree=effective_order,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "extensive_exponential",
                "dt": dt,
                "order": effective_order,
                "requested_order": order,
                "mode": canonical_mode,
                "max_bond": max_bond,
                "on_exceed": on_exceed,
                "cache_history": bool(cache_history),
                "history_storage": "one-site-local",
                "history_storage_requested": history_storage,
                "algorithms": ("one-site-taylor",),
                "approximate": False,
                "approximation_requested": bool(approximate),
                "extension_requested": bool(extend),
            },
        )

    def compress_exact(self, *, inplace=False):
        """Apply exact history/column compression without a numerical cutoff.

        The candidate histories follow the paper: move level-1 tokens to the
        front for column equivalence and level-3 tokens to the front for row
        equivalence.  A candidate is accepted only when the corresponding
        operator-valued rows or columns are exactly equal, so this method is
        conservative and cannot introduce a truncation error.
        """
        target = self if inplace else self.copy()
        initial = target.bond_dimensions
        merges = []
        skipped = 0

        for bond in range(1, target.L):
            changed = True
            while changed:
                changed = False
                levels = target._levels[bond]
                for source_pos, source in enumerate(levels):
                    history = source.history
                    n1 = sum(_level_number(token) == 1 for token in history)
                    n3 = sum(_level_number(token) == 3 for token in history)
                    if n3 <= n1:
                        canonical = _move_level_front(history, 1)
                        mode = "column"
                    else:
                        canonical = _move_level_front(history, 3)
                        mode = "row"
                    if canonical == history:
                        continue
                    target_pos = next(
                        (
                            pos
                            for pos, candidate in enumerate(levels)
                            if pos != source_pos and candidate.history == canonical
                        ),
                        None,
                    )
                    if target_pos is None:
                        skipped += 1
                        continue
                    # A history match alone is not enough: the corresponding
                    # operator-valued row or column must also be identical.
                    # This conservative check is why this stage is exact and
                    # has no numerical cutoff or hidden approximation.
                    if target._try_merge(bond, source_pos, target_pos, mode):
                        merges.append({
                            "bond": bond - 1,
                            "source": source.label,
                            "target": levels[target_pos].label,
                            "mode": mode,
                            "history": history,
                        })
                        changed = True
                        break
                    skipped += 1

        report = MPOCompressionReport(
            method="exact-history",
            exact=True,
            initial_bond_dimensions=tuple(initial),
            final_bond_dimensions=target.bond_dimensions,
            merged_channels=len(merges),
            merges=tuple(merges),
            skipped_candidates=skipped,
        )
        target.metadata["compression_report"] = report
        target.compression_report = report
        return target

    def _try_merge(self, bond, source, target, mode):
        """Try one exact scalar gauge elimination on a virtual bond."""
        if source == target:
            return False
        left = self._arrays[bond - 1]
        right = self._arrays[bond]
        if mode == "column":
            source_block = left[:, source]
            target_block = left[:, target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[source] = left_blocks[source] - left_blocks[target]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[target] = right_blocks[target] + right_blocks[source]
        else:
            source_block = right[source]
            target_block = right[target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[target] = left_blocks[target] + left_blocks[source]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[source] = right_blocks[source] - right_blocks[target]

        left_blocks = [block for pos, block in enumerate(left_blocks) if pos != source]
        right_blocks = [block for pos, block in enumerate(right_blocks) if pos != source]
        self._arrays = (
            *self._arrays[: bond - 1],
            _stack(left_blocks, axis=1),
            _stack(right_blocks, axis=0),
            *self._arrays[bond + 1 :],
        )
        self._levels[bond].pop(source)
        self._base_level_position_cache = None
        self._validate()
        return True

    def _check_compatible(self, other):
        if not isinstance(other, FirstDegreeMPO):
            raise TypeError("other must be a FirstDegreeMPO.")
        if self.L != other.L:
            raise ValueError(f"MPO lengths differ: {self.L} and {other.L}.")
        if self.phys_dim != other.phys_dim:
            raise ValueError(
                f"MPO physical dimensions differ: {self.phys_dim} and {other.phys_dim}."
            )

    def __add__(self, other):
        return self.add(other)

    def __matmul__(self, other):
        return self.non_disjoint_product(other)

    def __mul__(self, coefficient):
        return self.scale(coefficient)

    def __rmul__(self, coefficient):
        return self.scale(coefficient)


class CompiledMPOExp:
    """Value-only higher-order exponential evaluator for an :class:`MPOBasis`.

    The object owns the structural pieces of a parameterized exponential step:
    the unit-coefficient MPO tensors, coefficient-slot indices, static local
    operator banks, and the history execution plans. Calls assemble fresh
    backend tensors and then execute the same Algorithms 1--4 as
    :meth:`FirstDegreeMPO.extensive_exponential`, so coefficients and the
    exponential step remain in the current Torch/JAX autodiff graph.

    ``CompiledMPOExp`` is intentionally a numerical boundary.  Its
    primary methods return raw ``(left, right, up, down)`` tensor tuples;
    :meth:`evaluate` is available when a semantic :class:`FirstDegreeMPO`
    wrapper is needed outside a compiled optimizer loop.
    """

    _MODE_ALIASES = {
        "base": (False, False, "base"),
        "algorithm4": (False, True, "algorithm4"),
        "paper_algorithm4": (False, True, "algorithm4"),
        "optimal": (True, False, "optimal"),
        "paper_optimal": (True, False, "optimal"),
        "approximate": (True, True, "approximate"),
        "paper_approximate": (True, True, "approximate"),
    }

    def __init__(
        self,
        basis,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        history_storage="auto",
    ):
        if not isinstance(basis, MPOBasis):
            raise TypeError("basis must be an MPOBasis.")
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("mode must be a string or None.")
            try:
                mode_extend, mode_approximate, canonical_mode = (
                    self._MODE_ALIASES[mode]
                )
            except KeyError as exc:
                allowed = ", ".join(sorted(self._MODE_ALIASES))
                raise ValueError(
                    f"unknown mode {mode!r}; expected one of {allowed}."
                ) from exc
            if extend or approximate:
                raise ValueError(
                    "mode cannot be combined with extend or approximate flags."
                )
            extend = mode_extend
            approximate = mode_approximate
        else:
            canonical_mode = (
                "approximate" if approximate and extend
                else "optimal" if extend
                else "algorithm4" if approximate
                else "base"
            )

        if history_storage == "streaming":
            raise ValueError(
                "compiled evolution requires cached history; use "
                "history_storage='auto', 'sparse', or 'dense'."
            )

        self.basis = basis
        self.order = order
        self.mode = canonical_mode
        self.extend = bool(extend)
        self.approximate = bool(approximate)
        self.max_bond = max_bond
        self.on_exceed = on_exceed
        self.history_storage = history_storage

        # This validates the complete option set and fills every symbolic
        # history/tensor plan once.  The unit-coefficient numerical result is
        # discarded; only its structural caches are retained.
        basis._template.extensive_exponential(  # pylint: disable=protected-access
            0.0,
            order=order,
            mode=canonical_mode,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=True,
            history_storage=history_storage,
        )

        self._base_arrays = tuple(basis._template.arrays)  # pylint: disable=protected-access
        self._site_records = self._compile_slot_records(basis)
        self._fused_site_banks = self._compile_fused_site_banks(basis)

    @staticmethod
    def _compile_slot_records(basis):
        """Resolve coefficient slots to dense virtual tensor positions."""
        automaton = basis._automaton  # pylint: disable=protected-access
        records_by_site = [[] for _ in range(basis.L)]

        def position(site, state, *, left):
            if basis.L == 1:
                return 0
            if left and site == 0:
                return 0
            if not left and site == basis.L - 1:
                return 0
            cut = site - 1 if left else site
            return {
                channel.state: index
                for index, channel in enumerate(automaton.channels[cut])
            }[state]

        for site, transition_index, contributions in basis._slot_groups:
            transition = automaton.transitions[site][transition_index]
            row = position(site, transition.left_state, left=True)
            column = position(site, transition.right_state, left=False)
            term_indices = np.asarray(
                [term_index for term_index, _operator in contributions],
                dtype=int,
            )
            operators = tuple(operator for _term_index, operator in contributions)
            if all(
                _backend_name(operator) in {"builtins", "numpy"}
                for operator in operators
            ):
                operator_bank = np.stack(
                    [np.asarray(operator) for operator in operators],
                    axis=0,
                )
            else:
                operator_bank = None
            records_by_site[site].append(
                (
                    term_indices,
                    row,
                    column,
                    operators,
                    transition.operator,
                    operator_bank,
                ),
            )

        compiled = []
        for records in records_by_site:
            if not records:
                compiled.append(None)
                continue
            compiled.append(tuple(records))
        return tuple(compiled)

    def _compile_fused_site_banks(self, basis):
        """Build optional dense affine coefficient banks per MPO site.

        A unit-coefficient template contains the structural rails and one
        copy of every slot edge.  Subtracting those unit edge blocks from a
        static bias leaves an affine representation

        ``local_tensor = bias + sum(term_coefficient * operator_bank[term])``.

        The bank turns coefficient assembly into one backend contraction per
        site.  It is used only for ordinary host-backed static operators and
        only below a memory bound; backend-native operator arrays and large
        sparse layouts use the grouped autodiff-safe fallback.
        """
        fused = []
        for base, records in zip(self._base_arrays, self._site_records):
            if records is None or _backend_name(base) not in {"builtins", "numpy"}:
                fused.append(None)
                continue
            if any(
                _backend_name(operator) not in {"builtins", "numpy"}
                for record in records
                for operator in (*record[3], record[4])
            ):
                fused.append(None)
                continue
            try:
                bank_elements = basis.num_terms * int(np.prod(base.shape))
                if bank_elements > _MAX_FUSED_SLOT_BANK_ELEMENTS:
                    fused.append(None)
                    continue
                operators = [
                    np.asarray(operator)
                    for record in records
                    for operator in record[3]
                ]
                unit_operators = [
                    np.asarray(record[4])
                    for record in records
                ]
                dtype = np.result_type(
                    np.asarray(base).dtype,
                    *(operator.dtype for operator in operators),
                    *(operator.dtype for operator in unit_operators),
                    np.float32,
                )
                bias = np.asarray(base, dtype=dtype).copy()
                bank = np.zeros(
                    (basis.num_terms, *tuple(int(size) for size in base.shape)),
                    dtype=dtype,
                )
                for record in records:
                    term_indices, row, column, operators, unit_operator, _bank = (
                        record
                    )
                    bias[row, column] -= np.asarray(unit_operator, dtype=dtype)
                    for term_index, operator in zip(term_indices, operators):
                        bank[int(term_index), row, column] += np.asarray(
                            operator,
                            dtype=dtype,
                        )
            except (TypeError, ValueError):
                fused.append(None)
                continue
            fused.append((bias, bank))
        return tuple(fused)

    @property
    def cache_info(self):
        """Return the shared structural cache diagnostics."""
        info = dict(self.basis.cache_info)
        info.update({
            "fused_slot_sites": sum(
                bank is not None for bank in self._fused_site_banks
            ),
            "fused_slot_bank_elements": sum(
                int(np.prod(bank[1].shape))
                for bank in self._fused_site_banks
                if bank is not None
            ),
        })
        return info

    def _assemble_arrays(self, dt, parameters, coefficients):
        coefficient_values = self.basis._coefficient_values(  # pylint: disable=protected-access
            parameters,
            coefficients,
        )
        coefficient_batch = _stack(coefficient_values, axis=0)
        reference = _backend_reference((dt, *coefficient_values, *self._base_arrays))
        step = _as_backend(dt, like=reference)
        arrays = []

        for base, records, fused in zip(
            self._base_arrays,
            self._site_records,
            self._fused_site_banks,
        ):
            if fused is not None:
                bias, bank = fused
                array = _as_backend(bias, like=reference)
                if _complex_dtype(getattr(step, "dtype", None)) and not _complex_dtype(
                    getattr(array, "dtype", None)
                ):
                    array = ar.do(
                        "multiply",
                        array,
                        _as_backend(1.0 + 0.0j, like=step),
                    )
                bank = _as_backend(bank, like=coefficient_batch)
                coefficients_backend, bank = _align_tensordot_dtypes(
                    coefficient_batch,
                    bank,
                )
                weighted = ar.do(
                    "tensordot",
                    coefficients_backend,
                    bank,
                    axes=([0], [0]),
                )
                array, weighted = _align_tensordot_dtypes(array, weighted)
                arrays.append(ar.do("add", array, weighted))
                continue
            array = _as_backend(base, like=reference)
            if _complex_dtype(getattr(step, "dtype", None)) and not _complex_dtype(
                getattr(array, "dtype", None)
            ):
                # Higher-order prefactors can be complex even when H and its
                # coefficients are real. Promote the fresh local view before
                # any history transfer so no imaginary component is lost.
                array = ar.do(
                    "multiply",
                    array,
                    _as_backend(1.0 + 0.0j, like=step),
                )
            if records is not None:
                corrections = []
                rows = []
                columns = []
                for (
                    term_indices,
                    row,
                    column,
                    operators,
                    unit_operator,
                    operator_bank,
                ) in records:
                    if operator_bank is not None:
                        indices = _as_backend(
                            term_indices,
                            like=coefficient_batch,
                        )
                        selected = coefficient_batch[indices]
                        operator_values = _as_backend(
                            operator_bank,
                            like=selected,
                        )
                        selected, operator_values = _align_tensordot_dtypes(
                            selected,
                            operator_values,
                        )
                        weighted = ar.do(
                            "tensordot",
                            selected,
                            operator_values,
                            axes=([0], [0]),
                        )
                    else:
                        weighted = None
                        for term_index, operator in zip(
                            term_indices,
                            operators,
                        ):
                            contribution = _multiply_scalar(
                                coefficient_values[term_index],
                                operator,
                            )
                            if weighted is None:
                                weighted = contribution
                                continue
                            reference = _backend_reference(
                                (weighted, contribution),
                            )
                            weighted = ar.do(
                                "add",
                                _as_backend(weighted, like=reference),
                                _as_backend(contribution, like=reference),
                            )
                    unit_operator = _as_backend(unit_operator, like=weighted)
                    weighted, unit_operator = _align_tensordot_dtypes(
                        weighted,
                        unit_operator,
                    )
                    corrections.append(
                        ar.do("subtract", weighted, unit_operator),
                    )
                    rows.append(row)
                    columns.append(column)
                values = _stack(corrections, axis=0)
                array, values = _align_tensordot_dtypes(array, values)
                delta = _zeros(array.shape, like=array)
                delta = _scatter_add_into_2d(
                    delta,
                    rows,
                    columns,
                    values,
                )
                array = ar.do("add", array, delta)
            arrays.append(array)
        return tuple(arrays)

    def evaluate_arrays(self, dt, parameters=None, *, coefficients=None):
        """Compatibility wrapper for :meth:`exp_arrays`."""
        return self.exp_arrays(dt, parameters, coefficients=coefficients)

    def exp_arrays(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
    ):
        """Evaluate ``exp(step * H)`` as fresh backend-native tensor tuples.

        ``dt`` is accepted as a compatibility keyword for ``step``. The
        returned tuple is suitable for a backend contraction kernel and does
        not create a semantic MPO wrapper.
        """
        step = _resolve_exp_step(step, dt)
        arrays = self._assemble_arrays(step, parameters, coefficients)
        bound = self.basis._template._bind_arrays(arrays)  # pylint: disable=protected-access
        return bound.extensive_exponential(
            step,
            order=self.order,
            mode=self.mode,
            max_bond=self.max_bond,
            on_exceed=self.on_exceed,
            cache_history=True,
            history_storage=self.history_storage,
        ).arrays

    def evaluate(self, dt, parameters=None, *, coefficients=None):
        """Compatibility wrapper for :meth:`exp`."""
        return self.exp(dt, parameters, coefficients=coefficients)

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
    ):
        """Evaluate ``exp(step * H)`` as a semantic :class:`FirstDegreeMPO`.

        Use this form when downstream code needs MPO metadata or methods such
        as ``to_mpo()``. Use :meth:`exp_arrays` when it only needs raw tensors.
        """
        step = _resolve_exp_step(step, dt)
        arrays = self._assemble_arrays(step, parameters, coefficients)
        bound = self.basis._template._bind_arrays(arrays)  # pylint: disable=protected-access
        result = bound.extensive_exponential(
            step,
            order=self.order,
            mode=self.mode,
            max_bond=self.max_bond,
            on_exceed=self.on_exceed,
            cache_history=True,
            history_storage=self.history_storage,
        )
        result.metadata["compiled_exp"] = True
        # Retain the historical metadata key for callers that inspect it.
        result.metadata["compiled_evolution"] = True
        return result

    def time_evolution_arrays(self, dt, parameters=None, *, coefficients=None):
        """Evaluate ``exp(-1j * dt * H)`` as backend-native tensors."""
        return self.evaluate_arrays(
            -1j * dt,
            parameters,
            coefficients=coefficients,
        )

    def time_evolution(self, dt, parameters=None, *, coefficients=None):
        """Evaluate real-time evolution and return a semantic MPO."""
        return self.evaluate(
            -1j * dt,
            parameters,
            coefficients=coefficients,
        )

    __call__ = exp_arrays


# Compatibility name for callers of the original evolution-oriented API.
# Keep it as an exact alias so existing isinstance checks remain valid while
# ``CompiledMPOExp`` remains the canonical class name.
CompiledMPOEvolution = CompiledMPOExp


class MPOBasis:
    """Reusable coefficient basis for parameterized first-degree MPOs.

    ``MPOBasis`` separates the topology of a Hamiltonian from the scalar
    coefficients that are changed by an optimization loop.  It builds one
    shared automaton and records coefficient contributions for term-specific
    path transition slots.  :meth:`build` then only copies the small
    transition table and assembles those slots; it does not rebuild the term
    graph or infer channel topology.

    Coefficients can be ordinary scalars, :class:`MPOParameter` references,
    or callables receiving the parameter container.  The latter two forms are
    useful when the values are Torch or JAX scalars.  The output is a normal
    :class:`FirstDegreeMPO`, so it can immediately be passed to
    :meth:`FirstDegreeMPO.extensive_exponential`.

    The cache is deliberately topology-only.  Caching a completed MPO by a
    backend tensor's object identity would return stale values after an
    optimizer updates that tensor and could also retain an obsolete autodiff
    graph.  Every call to :meth:`build` therefore creates fresh local blocks
    while reusing the compiled channel layout.

    Parameters
    ----------
    L : int
        Number of sites.
    terms : iterable
        ``MPOProductTerm`` values or the same term forms accepted by
        :meth:`FirstDegreeMPO.from_local_terms`.
    phys_dim : int, optional
        Local physical dimension.  It is inferred from the first operator
        when omitted.
    """

    def __init__(
        self,
        L,
        terms,
        *,
        phys_dim=None,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
    ):
        if not isinstance(L, Integral):
            raise TypeError("L must be an integer.")
        L = int(L)
        if L < 1:
            raise ValueError("L must be >= 1.")
        terms = tuple(_term_from_input(term) for term in terms)
        if not terms:
            raise ValueError("terms must contain at least one product term.")
        if phys_dim is None:
            shape = tuple(getattr(terms[0].operators[0], "shape", ()))
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("cannot infer phys_dim from the first operator.")
            phys_dim = int(shape[0])

        # Compile the topology with unit coefficients once.  The shared
        # builder returns independent path coefficient slots, so common
        # prefixes and suffixes remain compressed without coupling terms'
        # autodiff values.
        unit_terms = tuple(replace(term, coefficient=1.0) for term in terms)
        automaton, slots = MPOAutomaton.from_product_terms(
            L,
            unit_terms,
            share_channels=True,
            return_slots=True,
            phys_dim=int(phys_dim),
        )

        self.L = L
        self.phys_dim = int(phys_dim)
        self._terms = terms
        self._slots = tuple(slots)
        slot_groups = {}
        for term_index, (site, transition_index) in enumerate(self._slots):
            slot_groups.setdefault((site, transition_index), []).append(
                (
                    term_index,
                    self._local_operator(self._terms[term_index], site),
                )
            )
        self._slot_groups = tuple(
            (site, transition_index, tuple(contributions))
            for (site, transition_index), contributions in slot_groups.items()
        )
        self._vectorized_slot_groups = tuple(
            (
                site,
                transition_index,
                np.asarray([term_index for term_index, _ in contributions], dtype=int),
                np.stack(
                    [np.asarray(operator) for _, operator in contributions],
                    axis=0,
                ),
            )
            for site, transition_index, contributions in self._slot_groups
            if all(
                _backend_name(operator) in {"builtins", "numpy"}
                for _, operator in contributions
            )
        )
        self._vectorized_slot_keys = frozenset(
            (site, transition_index)
            for site, transition_index, _indices, _operators
            in self._vectorized_slot_groups
        )
        self._automaton = automaton
        self._template = FirstDegreeMPO.from_automaton(
            automaton,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
        )
        self._build_count = 0
        self._compiled_evolution_cache = {}

    @classmethod
    def from_local_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        **kwargs,
    ):
        """Create a reusable basis from factorized local product terms."""
        return cls(L, terms, phys_dim=phys_dim, **kwargs)

    @classmethod
    def from_pauli_terms(
        cls,
        L,
        terms,
        **kwargs,
    ):
        """Create a reusable qubit basis from compact Pauli term labels."""
        return cls(L, terms, phys_dim=2, **kwargs)

    @property
    def terms(self):
        """Read-only term specifications, including coefficient references."""
        return self._terms

    @property
    def num_terms(self):
        """Number of coefficient slots in the basis."""
        return len(self._terms)

    @property
    def template(self):
        """The unit-coefficient first-degree MPO defining this basis."""
        return self._template.copy()

    @property
    def bond_dimensions(self):
        """Internal bond dimensions of the cached common automaton."""
        return self._template.bond_dimensions

    @property
    def cache_info(self):
        """Small immutable cache diagnostic for optimization bookkeeping."""
        return {
            "compiled": True,
            "compiled_terms": self.num_terms,
            "builds": self._build_count,
            "compiled_exp_variants": len(self._compiled_evolution_cache),
            # Compatibility diagnostic name.
            "compiled_evolution_variants": len(self._compiled_evolution_cache),
            "topology_bond_dimensions": self.bond_dimensions,
            "vectorized_slot_groups": len(self._vectorized_slot_groups),
            "history": self._template.history_cache_info,
        }

    def clear_history_cache(self):
        """Release cached higher-order plans while retaining the basis graph."""
        self._template.clear_history_cache()
        self._compiled_evolution_cache.clear()
        return self

    def compile_evolution(
        self,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        history_storage="auto",
    ):
        """Compatibility wrapper for :meth:`compile_exp`.

        New code should call ``compile_exp``. This historical name remains
        available for existing programs and returns the same cached
        :class:`CompiledMPOExp` object.
        """
        key = (
            order,
            mode,
            extend,
            approximate,
            max_bond,
            on_exceed,
            history_storage,
        )
        try:
            compiled = self._compiled_evolution_cache.get(key)
        except TypeError:
            compiled = None
        if compiled is None:
            compiled = CompiledMPOExp(
                self,
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
            )
            try:
                self._compiled_evolution_cache[key] = compiled
            except TypeError:
                pass
        return compiled

    def compile_exp(self, **kwargs):
        """Compile a reusable value-only exponential evaluator.

        The returned :class:`CompiledMPOExp` stores only reusable structure:
        history topology, virtual transfer plans, coefficient-slot indices,
        and static operator banks. Each ``exp`` or ``exp_arrays`` call creates
        fresh backend values, keeping Torch/JAX autodiff graphs current.

        Parameters are the same as :meth:`MPOBasis.exp`, except that the
        ``step`` and coefficient values are supplied when the compiled object
        is called. Use ``compiled.exp`` for a semantic MPO and
        ``compiled.exp_arrays`` for raw backend tensor tuples.
        """
        return self.compile_evolution(**kwargs)

    def _resolve_coefficient(self, coefficient, parameters):
        if isinstance(coefficient, MPOParameter):
            return coefficient.resolve(parameters)
        if callable(coefficient):
            if parameters is None:
                raise KeyError(
                    "callable MPO coefficients require a parameter container."
                )
            coefficient = coefficient(parameters)
        _check_scalar(coefficient, name="MPO coefficient")
        return coefficient

    @staticmethod
    def _local_operator(term, site):
        """Return the local factor carried by ``term`` at ``site``."""
        for position, support_site in enumerate(term.sites):
            if support_site == site:
                return term.operators[position]
        string_position = 0
        for left, right in zip(term.sites, term.sites[1:]):
            for gap_site in range(left + 1, right):
                if gap_site == site:
                    if term.string_operators is None:
                        return ar.do("eye", term.operators[0].shape[0], like=term.operators[0])
                    return term.string_operators[string_position]
                string_position += 1
        raise ValueError(
            f"coefficient slot site {site} is outside term support {term.sites}."
        )

    def coefficients(self, parameters=None):
        """Evaluate all term coefficients as one backend-native batch."""
        values = tuple(
            self._resolve_coefficient(term.coefficient, parameters)
            for term in self._terms
        )
        reference = _backend_reference(values)
        values = tuple(_as_backend(value, like=reference) for value in values)
        return ar.do("stack", values, axis=0)

    def _coefficient_values(self, parameters, coefficients):
        if coefficients is None:
            batch = self.coefficients(parameters)
            return tuple(batch[index] for index in range(self.num_terms))
        if parameters is not None:
            raise ValueError(
                "parameters and coefficients are mutually exclusive."
            )
        shape = getattr(coefficients, "shape", None)
        if shape is not None:
            shape = tuple(shape)
            if not shape:
                if self.num_terms != 1:
                    raise ValueError(
                        "a scalar coefficient batch is valid only for one term."
                    )
                values = (coefficients,)
            elif len(shape) == 1:
                if int(shape[0]) != self.num_terms:
                    raise ValueError(
                        f"coefficients must have length {self.num_terms}, "
                        f"got {shape[0]}."
                    )
                values = tuple(coefficients[index] for index in range(self.num_terms))
            else:
                raise ValueError("coefficients must be a one-dimensional batch.")
        else:
            try:
                values = tuple(coefficients)
            except TypeError as exc:
                raise TypeError(
                    "coefficients must be a one-dimensional batch."
                ) from exc
            if len(values) != self.num_terms:
                raise ValueError(
                    f"coefficients must have length {self.num_terms}, "
                    f"got {len(values)}."
                )
        for index, value in enumerate(values):
            _check_scalar(value, name=f"coefficients[{index}]")
        reference = _backend_reference(values)
        return tuple(_as_backend(value, like=reference) for value in values)

    def build(self, parameters=None, *, coefficients=None):
        """Bind current coefficients and return ``H(parameters)``.

        Parameters
        ----------
        parameters : mapping or sequence, optional
            Values used by :class:`MPOParameter` references and coefficient
            callables.  A backend scalar is inserted directly into the local
            transition, preserving its autodiff graph.
        coefficients : one-dimensional array-like, optional
            Already-evaluated coefficient batch. This is useful when an
            optimizer evaluates all coefficients in one backend operation.
            It is mutually exclusive with ``parameters``.
        """
        transitions = [list(site_transitions) for site_transitions in self._automaton.transitions]
        coefficient_values = self._coefficient_values(parameters, coefficients)
        coefficient_batch = _stack(coefficient_values, axis=0)
        vectorized_keys = self._vectorized_slot_keys
        for site, transition_index, term_indices, operators in (
            self._vectorized_slot_groups
        ):
            index = _as_backend(term_indices, like=coefficient_batch)
            selected = coefficient_batch[index]
            operator_batch = _as_backend(operators, like=selected)
            selected, operator_batch = _align_tensordot_dtypes(
                selected,
                operator_batch,
            )
            weighted = ar.do(
                "tensordot",
                selected,
                operator_batch,
                axes=([0], [0]),
            )
            transition = transitions[site][transition_index]
            transitions[site][transition_index] = type(transition)(
                transition.left_state,
                transition.right_state,
                weighted,
            )
        for site, transition_index, contributions in self._slot_groups:
            if (site, transition_index) in vectorized_keys:
                continue
            weighted = None
            for term_index, operator in contributions:
                contribution = _multiply_scalar(
                    coefficient_values[term_index],
                    operator,
                )
                if weighted is None:
                    weighted = contribution
                    continue
                reference = _backend_reference((weighted, contribution))
                weighted = ar.do(
                    "add",
                    _as_backend(weighted, like=reference),
                    _as_backend(contribution, like=reference),
                )
            transition = transitions[site][transition_index]
            transitions[site][transition_index] = type(transition)(
                transition.left_state,
                transition.right_state,
                weighted,
            )

        automaton = MPOAutomaton(
            self.L,
            channels=self._automaton.channels,
            transitions=transitions,
            start_state=self._automaton.start_state,
            done_state=self._automaton.done_state,
            phys_dim=self.phys_dim,
        )
        result = FirstDegreeMPO.from_automaton(
            automaton,
            upper_ind_id=self._template.upper_ind_id,
            lower_ind_id=self._template.lower_ind_id,
            site_tag_id=self._template.site_tag_id,
            degree=1,
        )
        result._history_topology_cache = self._template._history_topology_cache
        result._history_symbolic_cache = self._template._history_symbolic_cache
        result._history_extension_plan_cache = (
            self._template._history_extension_plan_cache
        )
        result._history_compression_plan_cache = (
            self._template._history_compression_plan_cache
        )
        result._history_approximation_plan_cache = (
            self._template._history_approximation_plan_cache
        )
        result._history_tensor_plan_cache = self._template._history_tensor_plan_cache
        result.metadata.update({
            "operation": "mpo_basis_build",
            "basis_terms": self.num_terms,
            "basis_build": self._build_count + 1,
            "coefficient_batch": True,
        })
        self._build_count += 1
        return result

    __call__ = build

    def extensive_exponential(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Build ``exp(dt * H(parameters))`` with the higher-order MPO path."""
        return self.build(
            parameters,
            coefficients=coefficients,
        ).extensive_exponential(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        )

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
    ):
        """Build ``exp(step * H(parameters))`` with optional compression.

        ``step`` is the actual scalar multiplying the Hamiltonian. For
        real-time evolution, pass ``step=-1j * tau``; ``dt=...`` remains a
        compatibility keyword. ``chi`` is the final MPO bond cap, while
        ``max_bond`` only guards the temporary higher-order history.
        """
        step = _resolve_exp_step(step, dt)
        return self.build(parameters, coefficients=coefficients).exp(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )

    def time_evolution(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
    ):
        """Build the real-time MPO ``exp(-1j * dt * H(parameters))``."""
        return self.build(parameters, coefficients=coefficients).time_evolution(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )

    def exp_arrays(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Evaluate ``exp(step * H)`` as backend-native tensor tuples.

        This method is intended for JAX/Torch compiled numerical kernels:
        the reusable basis remains a Python-side structural object, while
        the returned tensors are the only values captured by the optimizer's
        autodiff graph.
        """
        step = _resolve_exp_step(step, dt)
        if cache_history:
            return self.compile_exp(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
            ).exp_arrays(
                step,
                parameters,
                coefficients=coefficients,
            )
        return self.build(
            parameters,
            coefficients=coefficients,
        ).exp_arrays(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        )

    def time_evolution_arrays(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Evaluate real-time evolution as backend-native tensor tuples."""
        if cache_history:
            return self.compile_evolution(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
            ).time_evolution_arrays(
                dt,
                parameters,
                coefficients=coefficients,
            )
        return self.build(
            parameters,
            coefficients=coefficients,
        ).time_evolution_arrays(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
        )

    def exp_batch(
        self,
        step=None,
        coefficients=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
    ):
        """Evaluate ``exp(step * H)`` for a batch of coefficient vectors.

        The structural plan is compiled once and every row is evaluated with
        fresh backend tensors, preserving gradients with respect to both the
        coefficient batch and ``step``. JAX and current Torch releases use
        native ``vmap``; NumPy and unsupported backend operations use a small
        autodiff-safe assembly loop.
        """
        step = _resolve_exp_step(step, dt)
        if coefficients is None:
            raise TypeError("exp_batch requires a coefficient batch.")
        shape = getattr(coefficients, "shape", None)
        if shape is None or len(tuple(shape)) != 2:
            raise ValueError(
                "coefficients must have shape (batch, number_of_terms)."
            )
        if int(shape[1]) != self.num_terms:
            raise ValueError(
                f"coefficients must have {self.num_terms} columns, got {shape[1]}."
            )
        options = {
            "order": order,
            "mode": mode,
            "extend": extend,
            "approximate": approximate,
            "max_bond": max_bond,
            "on_exceed": on_exceed,
            "cache_history": cache_history,
            "history_storage": history_storage,
        }
        if cache_history:
            compiled = self.compile_exp(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
            )
            if _backend_name(coefficients) == "jax":
                import jax  # pylint: disable=import-outside-toplevel

                return jax.vmap(
                    lambda row: compiled.exp_arrays(
                        step,
                        coefficients=row,
                    )
                )(coefficients)
            if _backend_name(coefficients) == "torch":
                try:
                    import torch  # pylint: disable=import-outside-toplevel

                    vmap = getattr(torch, "vmap", None)
                    if vmap is None:
                        from torch.func import vmap  # pylint: disable=import-outside-toplevel
                    return vmap(
                        lambda row: compiled.exp_arrays(
                            step,
                            coefficients=row,
                        )
                    )(coefficients)
                except (ImportError, RuntimeError, TypeError):
                    pass
            rows = [
                compiled.exp_arrays(
                    step,
                    coefficients=coefficients[index],
                )
                for index in range(int(shape[0]))
            ]
            if not rows:
                return tuple()
            return tuple(
                ar.do("stack", tuple(row[site] for row in rows), axis=0)
                for site in range(self.L)
            )
        backend = _backend_name(coefficients)
        if backend == "jax":
            import jax  # pylint: disable=import-outside-toplevel

            return jax.vmap(
                lambda row: self.exp_arrays(
                    step,
                    coefficients=row,
                    **options,
                )
            )(coefficients)
        if backend == "torch":
            try:
                import torch  # pylint: disable=import-outside-toplevel

                vmap = getattr(torch, "vmap", None)
                if vmap is None:
                    from torch.func import vmap  # pylint: disable=import-outside-toplevel
                return vmap(
                    lambda row: self.exp_arrays(
                        step,
                        coefficients=row,
                        **options,
                    )
                )(coefficients)
            except (ImportError, RuntimeError, TypeError):
                # Some older Torch releases cannot batch one of the backend
                # scatter primitives. The ordinary loop remains fully
                # differentiable and is a safe compatibility fallback.
                pass
        rows = [
            self.exp_arrays(
                step,
                coefficients=coefficients[index],
                **options,
            )
            for index in range(int(shape[0]))
        ]
        if not rows:
            return tuple()
        return tuple(
            ar.do("stack", tuple(row[site] for row in rows), axis=0)
            for site in range(self.L)
        )

    def evolution_mpo(
        self,
        parameters=None,
        *,
        dt,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
    ):
        """Build ``exp(-1j * dt * H(parameters))`` with optional compression.

        ``chi`` is the final MPO bond cap. It is separate from the
        ``max_bond`` keyword accepted by the higher-order builder, which only
        protects the temporary history representation. With ``chi=None`` the
        method returns the existing semantic :class:`FirstDegreeMPO`. When
        ``chi`` is set, the default numerical path returns a Quimb MPO; set
        ``differentiable=True`` for a fixed-rank autodiff-compatible semantic
        MPO instead.
        """
        return self.time_evolution(
            dt,
            parameters,
            coefficients=coefficients,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
        )

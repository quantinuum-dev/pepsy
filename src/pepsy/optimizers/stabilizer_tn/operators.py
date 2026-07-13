"""MPO / gate builders for STN coefficient-state (``|nu>``) updates.

A physical single-qubit (or Pauli-string) rotation ``exp(-i theta/2 * A)`` on
``|psi>`` maps to ``exp(-i theta/2 * M)`` on ``|nu>`` with ``M = C^dagger A C``
a signed Pauli string (see :meth:`STNState.nu_frame_pauli`).  The operator
``exp(-i theta/2 * M) = cos(theta/2) I - i sin(theta/2) M`` has an exact
bond-dimension-2 MPO, built here.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np
import quimb.tensor as qtn

_I = np.eye(2, dtype=complex)
_PAULI = {
    "I": _I,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(axis: str) -> np.ndarray:
    """Return the 2x2 Pauli matrix for ``'I'/'X'/'Y'/'Z'``."""
    return _PAULI[str(axis).upper()]


def _kron_pauli(labels: Sequence[str]) -> np.ndarray:
    """Return the Kronecker product of the single-qubit Paulis in ``labels``."""
    mat = _PAULI[str(labels[0]).upper()]
    for ch in labels[1:]:
        mat = np.kron(mat, _PAULI[str(ch).upper()])
    return mat


def pauli_decomposition(
    gate: np.ndarray, k: int, *, tol: float | None = None
) -> List[Tuple[Tuple[str, ...], complex]]:
    """Decompose a ``2**k x 2**k`` matrix into a Pauli-string basis.

    Returns a list of ``(labels, coeff)`` where ``labels`` is a length-``k``
    tuple over ``{'I','X','Y','Z'}`` and ``coeff = Tr(P_labels @ gate) / 2**k``,
    keeping only terms with ``abs(coeff) > tol``. If ``tol`` is ``None``, use a
    matrix-scale-relative threshold derived from the input dtype. Works for any
    (unitary or non-unitary) matrix; ``G = sum_labels coeff * P_labels`` up to
    the requested numerical tolerance.

    Enumerates ``4**k`` Paulis, so intended for small ``k`` (few-qubit gates).
    """
    gate = np.asarray(gate)
    dim = 2 ** k
    if gate.shape != (dim, dim):
        raise ValueError(f"gate must be {dim}x{dim} for k={k}, got {gate.shape}.")
    if tol is None:
        if np.issubdtype(gate.dtype, np.complexfloating):
            real_dtype = np.empty((), dtype=gate.dtype).real.dtype
        elif np.issubdtype(gate.dtype, np.floating):
            real_dtype = gate.dtype
        else:
            real_dtype = np.dtype("float64")
        matrix_scale = float(np.max(np.abs(gate), initial=0.0))
        tol = dim * np.finfo(real_dtype).eps * matrix_scale
    else:
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError(f"tol must be finite and nonnegative, got {tol!r}.")
    scale = 1.0 / dim
    terms: List[Tuple[Tuple[str, ...], complex]] = []
    for labels in itertools.product("IXYZ", repeat=k):
        coeff = scale * np.trace(_kron_pauli(labels) @ gate)
        if abs(coeff) > tol:
            terms.append((labels, complex(coeff)))
    return terms


def single_qubit_combo_matrix(
    c: complex, coef: complex, axis: str, dtype: str = "complex128"
) -> np.ndarray:
    """Return ``c I + coef P`` for a single-qubit Pauli ``axis``."""
    return (c * _I + coef * _PAULI[str(axis).upper()]).astype(dtype)


def single_qubit_rotation_matrix(
    theta: float, axis: str, sign: float = 1.0, dtype: str = "complex128"
) -> np.ndarray:
    """Return ``cos(theta/2) I - i sin(theta/2) * sign * P`` for a 1-qubit axis."""
    c = np.cos(theta / 2)
    coef = -1j * sign * np.sin(theta / 2)
    return single_qubit_combo_matrix(c, coef, axis, dtype)


def pauli_combo_mpo(
    c: complex,
    coef: complex,
    paulis: Sequence[str],
    *,
    dtype: str = "complex128",
) -> qtn.MatrixProductOperator:
    """Exact bond-dim-2 MPO for ``c I^{x L} + coef (P_0 x ... x P_{L-1})``.

    Parameters
    ----------
    c, coef : complex
        Coefficients of the identity and Pauli-string branches.
    paulis : sequence of {'I','X','Y','Z'}
        One Pauli per MPS site, in site order.  Length must be >= 2 (handle the
        single-site case with :func:`single_qubit_combo_matrix`).
    """
    axes = [str(p).upper() for p in paulis]
    L = len(axes)
    if L < 2:
        raise ValueError(
            "pauli_combo_mpo requires >= 2 sites; use "
            "single_qubit_combo_matrix for a single-site operator."
        )
    # Window to the support span: outside [lo, hi] both branches (c*I and coef*P)
    # are identity, so the operator factors as I_outside (x) window-operator and
    # the MPO bond there is 1.  This keeps ``mpo.apply`` cheap for local operators.
    support = [i for i, ch in enumerate(axes) if ch != "I"]
    if not support:  # pure-identity operator (c + coef) * I (degenerate)
        lo = hi = 0
    else:
        lo, hi = support[0], support[-1]

    arrays = []
    for i, ch in enumerate(axes):
        pmat = _PAULI[ch]
        if not support:
            op = (c + coef) * _I
            w = _boundary(op, i, L)
        elif i < lo or i > hi:  # bond-1 identity outside the support span
            w = _boundary(_I, i, L)
        elif lo == hi:  # single-site window carries the whole combo
            w = _boundary(c * _I + coef * pmat, i, L)
        elif i == lo:  # open the two channels: left bond 1, right bond 2
            if i == 0:
                w = np.zeros((2, 2, 2), dtype=complex)  # (right, up, down)
                w[0] = _I
                w[1] = pmat
            else:
                w = np.zeros((1, 2, 2, 2), dtype=complex)  # (left, right, up, down)
                w[0, 0] = _I
                w[0, 1] = pmat
        elif i == hi:  # close the channels: left bond 2, right bond 1
            if i == L - 1:
                w = np.zeros((2, 2, 2), dtype=complex)  # (left, up, down)
                w[0] = c * _I
                w[1] = coef * pmat
            else:
                w = np.zeros((2, 1, 2, 2), dtype=complex)  # (left, right, up, down)
                w[0, 0] = c * _I
                w[1, 0] = coef * pmat
        else:  # middle of the window: bond 2
            w = np.zeros((2, 2, 2, 2), dtype=complex)  # (left, right, up, down)
            w[0, 0] = _I
            w[1, 1] = pmat
        arrays.append(w.astype(dtype))
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def _boundary(op: np.ndarray, i: int, L: int) -> np.ndarray:
    """Return a bond-1 MPO site tensor carrying single-site operator ``op``."""
    if i == 0:
        w = np.zeros((1, 2, 2), dtype=complex)  # (right, up, down)
        w[0] = op
    elif i == L - 1:
        w = np.zeros((1, 2, 2), dtype=complex)  # (left, up, down)
        w[0] = op
    else:
        w = np.zeros((1, 1, 2, 2), dtype=complex)  # (left, right, up, down)
        w[0, 0] = op
    return w


def pauli_rotation_mpo(
    theta: float,
    paulis: Sequence[str],
    *,
    sign: float = 1.0,
    dtype: str = "complex128",
) -> qtn.MatrixProductOperator:
    """Exact bond-dim-2 MPO for ``exp(-i theta/2 * sign * (P_0 x ... x P_{L-1}))``."""
    c = np.cos(theta / 2)
    coef = -1j * sign * np.sin(theta / 2)
    return pauli_combo_mpo(c, coef, paulis, dtype=dtype)


def pauli_combo_submpo(
    c: complex,
    coef: complex,
    terms: dict,
    L: int,
    *,
    dtype: str = "complex128",
):
    """Windowed sub-MPO for ``c I + coef (prod_i P_i)`` placed on its true sites.

    ``terms`` maps ``site -> 'X'/'Y'/'Z'`` (the non-identity support, size >= 2).
    The returned MPO spans only the contiguous window ``[min, max]`` of the
    support and is built on those actual sites (so quimb's
    ``gate_with_submpo_`` aligns and compresses only that region).

    Returns
    -------
    (MatrixProductOperator, tuple[int])
        The sub-MPO and its support ``where`` (contiguous sites ``lo..hi``).
    """
    support = sorted(terms)
    lo, hi = support[0], support[-1]
    axes = [terms.get(i, "I") for i in range(lo, hi + 1)]
    w = len(axes)  # >= 2 by contract (single-support handled by the caller)
    arrays = []
    for i, ch in enumerate(axes):
        pmat = _PAULI[ch]
        if i == 0:
            t = np.zeros((2, 2, 2), dtype=complex)  # (right, up, down)
            t[0] = _I
            t[1] = pmat
        elif i == w - 1:
            t = np.zeros((2, 2, 2), dtype=complex)  # (left, up, down)
            t[0] = c * _I
            t[1] = coef * pmat
        else:
            t = np.zeros((2, 2, 2, 2), dtype=complex)  # (left, right, up, down)
            t[0, 0] = _I
            t[1, 1] = pmat
        arrays.append(t.astype(dtype))
    where = tuple(range(lo, hi + 1))
    mpo = qtn.MatrixProductOperator(arrays, sites=where, L=L, shape="lrud")
    return mpo, where

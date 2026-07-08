"""MPO / gate builders for STN coefficient-state (``|nu>``) updates.

A physical single-qubit (or Pauli-string) rotation ``exp(-i theta/2 * A)`` on
``|psi>`` maps to ``exp(-i theta/2 * M)`` on ``|nu>`` with ``M = C^dagger A C``
a signed Pauli string (see :meth:`STNState.nu_frame_pauli`).  The operator
``exp(-i theta/2 * M) = cos(theta/2) I - i sin(theta/2) M`` has an exact
bond-dimension-2 MPO, built here.
"""

from __future__ import annotations

from typing import Sequence

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
    arrays = []
    for i, ch in enumerate(axes):
        pmat = _PAULI[ch]
        if i == 0:
            w = np.zeros((2, 2, 2), dtype=complex)  # (right, up, down)
            w[0] = _I
            w[1] = pmat
        elif i == L - 1:
            w = np.zeros((2, 2, 2), dtype=complex)  # (left, up, down)
            w[0] = c * _I
            w[1] = coef * pmat
        else:
            w = np.zeros((2, 2, 2, 2), dtype=complex)  # (left, right, up, down)
            w[0, 0] = _I
            w[1, 1] = pmat
        arrays.append(w.astype(dtype))
    return qtn.MatrixProductOperator(arrays, shape="lrud")


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

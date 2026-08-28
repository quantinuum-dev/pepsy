"""Pauli helpers for the STN simulator (stim-backed).

These build :class:`stim.PauliString` objects for physical rotation axes and
convert a Hermitian Pauli string into ``({site: 'X'/'Y'/'Z'}, sign)`` form for
the coefficient-state (``|nu>``) rotation MPO.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

_AXIS_CODE = {"i": 0, "x": 1, "y": 2, "z": 3}
_CODE_AXIS = {1: "X", 2: "Y", 3: "Z"}


def _resolve_measurement_disentangle(
    absorb_basis,
    disentangle,
    *,
    default: bool,
) -> bool:
    """Normalize the measurement basis-update compatibility options."""
    if absorb_basis is None:
        resolved = bool(default)
    elif not isinstance(absorb_basis, (bool, np.bool_)):
        raise TypeError("absorb_basis must be a boolean.")
    else:
        resolved = bool(absorb_basis)

    if disentangle is None:
        return resolved
    if not isinstance(disentangle, (bool, np.bool_)):
        raise TypeError("disentangle must be a boolean.")
    if absorb_basis is not None and bool(disentangle) != resolved:
        raise ValueError(
            "absorb_basis and disentangle specify different measurement modes."
        )
    return bool(disentangle)


def single_pauli(axis: str, q: int, n: int):
    """Return a length-``n`` :class:`stim.PauliString` with ``axis`` on qubit ``q``."""
    import stim

    ps = stim.PauliString(n)
    ps[int(q)] = _AXIS_CODE[str(axis).lower()]
    return ps


def pauli_string(paulis: Iterable[str], where: Iterable[int], n: int):
    """Return a length-``n`` :class:`stim.PauliString` from per-qubit axes."""
    import stim

    axes = tuple(str(ch).lower() for ch in paulis)
    sites = tuple(int(q) for q in where)
    if len(axes) != len(sites):
        raise ValueError(
            f"Pauli axes {axes!r} and where {sites!r} have different lengths."
        )
    if len(set(sites)) != len(sites):
        raise ValueError(f"Pauli support sites must be distinct, got {sites!r}.")
    for axis in axes:
        if axis not in _AXIS_CODE:
            raise ValueError(
                f"Invalid Pauli axis {axis!r}; expected only I, X, Y, or Z."
            )
    for q in sites:
        if not 0 <= q < int(n):
            raise ValueError(f"Qubit index {q} out of range for {n}-qubit state.")

    ps = stim.PauliString(n)
    for axis, q in zip(axes, sites):
        ps[q] = _AXIS_CODE[axis]
    return ps


def hermitian_pauli_terms(ps) -> Tuple[Dict[int, str], float]:
    """Return ``({site: 'X'/'Y'/'Z'}, real_sign)`` for a Hermitian Pauli string.

    ``ps`` must have a real sign (``+1`` or ``-1``); conjugating a Hermitian
    single-/multi-qubit Pauli by a Clifford always yields such a string.
    """
    sign = ps.sign
    if abs(getattr(sign, "imag", 0.0)) > 1e-9:
        raise ValueError(
            f"Expected a Hermitian Pauli string (real sign), got sign={sign!r}."
        )
    real_sign = float(getattr(sign, "real", sign))
    terms = {i: _CODE_AXIS[ps[i]] for i in range(len(ps)) if ps[i] != 0}
    return terms, real_sign

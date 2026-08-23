"""Shared validation for explicit dense gates in stabilizer optimizers."""

from __future__ import annotations

import autoray as ar
import numpy as np


def _as_gate_matrix(gate, n_qubits: int) -> np.ndarray:
    """Convert a matrix or Pepsy rank-``2 * n_qubits`` gate tensor to a matrix."""
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("A dense gate must act on at least one qubit.")
    array = np.asarray(ar.to_numpy(gate), dtype=complex)
    if array.ndim == 2:
        return array
    expected_shape = (2,) * (2 * n_qubits)
    if array.shape == expected_shape:
        dimension = 2**n_qubits
        return array.reshape(dimension, dimension)
    raise ValueError(
        "Dense gates must be square matrices or rank-2k tensors with one "
        f"binary index per leg; got shape {array.shape} for {n_qubits} qubits."
    )


def _is_unitary(gate: np.ndarray, tol: float = 1e-9) -> bool:
    """Return whether ``gate`` is unitary within the STN tolerance."""
    gate = np.asarray(gate, dtype=complex)
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        return False
    return np.allclose(
        gate.conj().T @ gate,
        np.eye(gate.shape[0], dtype=complex),
        rtol=tol,
        atol=tol,
    )


def _tableau_from_exact_unitary(gate: np.ndarray):
    """Return a Stim tableau only when it exactly represents ``gate``.

    ``stim.Tableau.from_unitary_matrix`` recognizes stabilizer actions but does
    not reliably reject arbitrary unitary matrices. In particular, a small
    non-Clifford rotation can be returned as the identity tableau. Comparing
    Stim's reconstructed matrix projectively prevents that silent misrouting.
    """
    if not _is_unitary(gate):
        return None

    import stim

    try:
        tableau = stim.Tableau.from_unitary_matrix(gate, endian="big")
        reconstructed = np.asarray(
            tableau.to_unitary_matrix(endian="big"), dtype=complex
        )
    except (ValueError, RuntimeError):
        return None

    pivot = np.unravel_index(np.argmax(np.abs(gate)), gate.shape)
    reconstructed_pivot = reconstructed[pivot]
    if abs(reconstructed_pivot) <= 1e-14:
        return None
    phase = gate[pivot] / reconstructed_pivot
    if np.allclose(gate, phase * reconstructed, rtol=1e-8, atol=1e-8):
        return tableau
    return None

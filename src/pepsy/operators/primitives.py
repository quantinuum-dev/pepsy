"""Standalone standard gate primitives.

This module contains small matrix/gate constructors separated from the
large routing and tensor-network application implementation.
"""

from __future__ import annotations

import quimb as qu
import quimb.tensor as qtn

__all__ = [
    "rx", "ry", "rz", "pauli", "x", "y", "z", "s", "sdg", "t", "tdg",
    "hadamard", "h", "cnot", "cx", "cy", "cz", "swap", "iswap",
    "phase", "u1", "u2", "cphase", "cu1", "crx", "cry", "crz",
    "cu2", "cu3", "rzz", "rxx", "ryy", "su4", "u3", "fsim", "fsimg",
]

def rx(theta):
    """Return a one-qubit RX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rx_gate_param_gen([theta])


def ry(theta):
    """Return a one-qubit RY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.ry_gate_param_gen([theta])


def rz(theta):
    """Return a one-qubit RZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rz_gate_param_gen([theta])


def pauli(which, dtype=None):
    """Return a one-qubit Pauli matrix by label, e.g. ``'X'`` or ``'Z'``."""
    label = str(which).upper()
    if dtype is None:
        return qu.pauli(label)
    return qu.pauli(label, dtype=dtype)


def x():
    """Return the one-qubit Pauli-X gate."""
    return pauli("X")


def y():
    """Return the one-qubit Pauli-Y gate."""
    return pauli("Y")


def z():
    """Return the one-qubit Pauli-Z gate."""
    return pauli("Z")


def s():
    """Return the one-qubit S gate."""
    return qu.S_gate()


def sdg():
    """Return the one-qubit S-dagger gate."""
    return s().H


def t():
    """Return the one-qubit T gate."""
    return qu.T_gate()


def tdg():
    """Return the one-qubit T-dagger gate."""
    return t().H


def hadamard():
    """Return the one-qubit Hadamard gate."""
    return qu.hadamard()


def h():
    """Alias for :func:`hadamard`."""
    return hadamard()


def cnot():
    """Return the two-qubit controlled-X (CNOT) gate."""
    return qu.CNOT()


def cx():
    """Alias for :func:`cnot`."""
    return cnot()


def cy():
    """Return the two-qubit controlled-Y gate."""
    return qu.cY()


def cz():
    """Return the two-qubit controlled-Z gate."""
    return qu.cZ()


def swap():
    """Return the two-qubit SWAP gate."""
    return qu.swap()


def iswap():
    """Return the two-qubit iSWAP gate."""
    return qu.iswap()


def phase(theta):
    """Alias for :func:`u1`."""
    return u1(theta)


def u1(theta):
    """Return a one-qubit U1 gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.u1_gate_param_gen([theta])


def u2(params):
    """Return a one-qubit U2 gate from 2 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters.
    """
    if len(params) != 2:
        raise ValueError("u2 expects exactly 2 parameters.")
    return qtn.circuit.u2_gate_param_gen(params)


def cphase(theta):
    """Return a two-qubit controlled-phase gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.cu1_param_gen([theta])


def cu1(theta):
    """Alias for :func:`cphase`."""
    return cphase(theta)


def crx(theta):
    """Return a two-qubit controlled-RX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.crx_param_gen([theta])


def cry(theta):
    """Return a two-qubit controlled-RY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.cry_param_gen([theta])


def crz(theta):
    """Return a two-qubit controlled-RZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.crz_param_gen([theta])


def cu2(params):
    """Return a two-qubit controlled-U2 gate from 2 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters.
    """
    if len(params) != 2:
        raise ValueError("cu2 expects exactly 2 parameters.")
    return qtn.circuit.cu2_param_gen(params)


def cu3(params):
    """Return a two-qubit controlled-U3 gate from 3 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 3 parameters.
    """
    if len(params) != 3:
        raise ValueError("cu3 expects exactly 3 parameters.")
    return qtn.circuit.cu3_param_gen(params)


def rzz(theta):
    """Return a two-qubit RZZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rzz_param_gen([theta])


def rxx(theta):
    """Return a two-qubit RXX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rxx_param_gen([theta])


def ryy(theta):
    """Return a two-qubit RYY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.ryy_param_gen([theta])


def su4(params):
    """Return a two-qubit SU(4) gate from 15 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 15 parameters.
    """
    if len(params) != 15:
        raise ValueError("su4 expects exactly 15 parameters.")
    return qtn.circuit.su4_gate_param_gen(params)


def u3(params):
    """Return a one-qubit U3 gate from 3 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 3 parameters.
    """
    if len(params) != 3:
        raise ValueError("u3 expects exactly 3 parameters.")
    return qtn.circuit.u3_gate_param_gen(params)


def fsim(params):
    """Return a two-qubit fSim gate from 2 parameters.

    The fSim gate is defined as::

        [[1,           0,           0, 0          ],
         [0,  cos(theta), -i*sin(theta), 0          ],
         [0, -i*sin(theta),  cos(theta), 0          ],
         [0,           0,           0, exp(-i*phi)]]

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters ``(theta, phi)``.
    """
    if len(params) != 2:
        raise ValueError("fsim expects exactly 2 parameters (theta, phi).")
    return qtn.circuit.fsim_param_gen(params)


def fsimg(params):
    """Return a two-qubit generalized fSim gate from 5 parameters.

    The most general number-conserving two-qubit gate parametrized by
    ``(theta, zeta, chi, gamma, phi)``.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 5 parameters ``(theta, zeta, chi, gamma, phi)``.
    """
    if len(params) != 5:
        raise ValueError("fsimg expects exactly 5 parameters (theta, zeta, chi, gamma, phi).")
    return qtn.circuit.fsimg_param_gen(params)




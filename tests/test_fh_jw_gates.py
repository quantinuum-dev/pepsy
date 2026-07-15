"""Tests for the bosonic Jordan-Wigner Fermi-Hubbard ``U1U1`` gate streams.

These validate :func:`fermi_hubbard_u1u1_jw_hopping_gate_stream`,
:func:`fermi_hubbard_u1u1_jw_interaction_gate_stream`, and
:func:`fermi_hubbard_u1u1_jw_gate_stream` at the operator level: the gates are
bosonic (no fermionic swap phases), unitary, exponentiate the intended term, and
reproduce the spectrum of an independent Jordan-Wigner exact-diagonalization.
"""

import numpy as np
import pytest

from pepsy.tensors import (
    SymGateStream,
    fermi_hubbard_u1u1_jw_gate_stream,
    fermi_hubbard_u1u1_jw_hopping_gate_stream,
    fermi_hubbard_u1u1_jw_interaction_gate_stream,
)


def _dense(arr):
    """Dense NumPy view of a Symmray gate/term in its pepsy per-site basis."""
    return np.asarray(arr.to_dense())


def _dense_jw_two_site(*, t=1.0, U=0.0, mu=0.0):
    """Independent two-site spinful FH Hamiltonian via explicit Jordan-Wigner.

    Qubit modes are ordered ``(up0, dn0, up1, dn1)`` with parity strings on the
    earlier modes. Returned in the qubit basis, so only its (basis-independent)
    spectrum is compared against the pepsy per-site operator.
    """
    n = 4
    eye = np.eye(2)
    zed = np.diag([1.0, -1.0])
    low = np.array([[0.0, 1.0], [0.0, 0.0]])

    def annihilate(m):
        mats = [zed] * m + [low] + [eye] * (n - m - 1)
        out = mats[0]
        for mat in mats[1:]:
            out = np.kron(out, mat)
        return out

    def mode(site, spin):
        return 2 * site + spin

    ham = np.zeros((16, 16))
    for spin in (0, 1):
        hop = annihilate(mode(0, spin)).conj().T @ annihilate(mode(1, spin))
        ham += -t * (hop + hop.conj().T)
    for site in (0, 1):
        num_up = annihilate(mode(site, 0)).conj().T @ annihilate(mode(site, 0))
        num_dn = annihilate(mode(site, 1)).conj().T @ annihilate(mode(site, 1))
        ham += U * (num_up @ num_dn) - mu * (num_up + num_dn)
    return ham


def test_jw_gate_streams_are_bosonic_u1u1():
    pytest.importorskip("symmray")
    hop = fermi_hubbard_u1u1_jw_hopping_gate_stream([(0, 1)], 0.1, t=1.0)
    inter = fermi_hubbard_u1u1_jw_interaction_gate_stream([0], 0.1, U=8.0)
    assert isinstance(hop, SymGateStream)
    assert isinstance(inter, SymGateStream)
    for gate, _where in tuple(hop) + tuple(inter):
        name = type(gate).__name__
        assert "Fermionic" not in name, name
        assert name.startswith("U1U1"), name


def test_jw_hopping_gate_is_unitary_and_expm_of_term():
    pytest.importorskip("symmray")
    sla = pytest.importorskip("scipy.linalg")
    from pepsy.tensors import symmetric as sym

    dt = 0.1
    term = sym._fh_u1u1_jw_hopping_term(t=1.0)
    ham = _dense(term).reshape(16, 16)
    assert np.allclose(ham, ham.conj().T, atol=1e-12)

    gate = _dense(
        fermi_hubbard_u1u1_jw_hopping_gate_stream([(0, 1)], dt, t=1.0)[0][0]
    ).reshape(16, 16)
    assert np.allclose(gate.conj().T @ gate, np.eye(16), atol=1e-10)
    assert np.allclose(gate, sla.expm(-1j * dt * ham), atol=1e-10)


def test_jw_hopping_imaginary_gate_is_expm_of_term():
    pytest.importorskip("symmray")
    sla = pytest.importorskip("scipy.linalg")
    from pepsy.tensors import symmetric as sym

    dt = 0.05
    ham = _dense(sym._fh_u1u1_jw_hopping_term(t=1.0)).reshape(16, 16)
    gate = _dense(
        fermi_hubbard_u1u1_jw_hopping_gate_stream([(0, 1)], dt, t=1.0, imaginary=True)[0][0]
    ).reshape(16, 16)
    assert np.allclose(gate, sla.expm(-dt * ham), atol=1e-10)


@pytest.mark.parametrize("t,U,mu", [(1.0, 0.0, 0.0), (1.3, 0.0, 0.0)])
def test_jw_hopping_spectrum_matches_jw_ed(t, U, mu):
    pytest.importorskip("symmray")
    from pepsy.tensors import symmetric as sym

    ham = _dense(sym._fh_u1u1_jw_hopping_term(t=t)).reshape(16, 16)
    ham = (ham + ham.conj().T) / 2
    spectrum = np.sort(np.linalg.eigvalsh(ham))
    reference = np.sort(np.linalg.eigvalsh(_dense_jw_two_site(t=t, U=U, mu=mu)))
    assert np.allclose(spectrum, reference, atol=1e-10)


def test_jw_hopping_peierls_phase_matches_expm():
    pytest.importorskip("symmray")
    sla = pytest.importorskip("scipy.linalg")
    from pepsy.tensors import symmetric as sym

    dt, angle = 0.1, 0.7
    ham = _dense(sym._fh_u1u1_jw_hopping_term(t=1.0, peierls_angle=angle)).reshape(16, 16)
    assert np.allclose(ham, ham.conj().T, atol=1e-12)  # still hermitian
    gate = _dense(
        fermi_hubbard_u1u1_jw_hopping_gate_stream(
            [(0, 1)], dt, t=1.0, peierls_angle=angle
        )[0][0]
    ).reshape(16, 16)
    assert np.allclose(gate, sla.expm(-1j * dt * ham), atol=1e-10)


def test_jw_interaction_gate_is_double_occupancy_phase():
    pytest.importorskip("symmray")
    dt, U = 0.1, 8.0
    gate = _dense(
        fermi_hubbard_u1u1_jw_interaction_gate_stream([0], dt, U=U)[0][0]
    ).reshape(4, 4)
    # local basis order (empty, up, down, double); only double occupancy phases.
    ref = np.diag(np.exp(-1j * dt * U * np.array([0.0, 0.0, 0.0, 1.0])))
    assert np.allclose(gate, ref, atol=1e-12)

    mu = 0.5
    gate_mu = _dense(
        fermi_hubbard_u1u1_jw_interaction_gate_stream([0], dt, U=U, mu=mu)[0][0]
    ).reshape(4, 4)
    onsite = (
        U * np.array([0.0, 0.0, 0.0, 1.0])
        - mu * np.array([0.0, 1.0, 0.0, 1.0])
        - mu * np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert np.allclose(gate_mu, np.diag(np.exp(-1j * dt * onsite)), atol=1e-12)


def test_jw_gate_stream_structure_and_orientation():
    pytest.importorskip("symmray")
    dt = 0.05
    edges = [(0, 1), (1, 2)]

    order1 = fermi_hubbard_u1u1_jw_gate_stream(edges, dt, order=1)
    order2 = fermi_hubbard_u1u1_jw_gate_stream(edges, dt, order=2)
    # order 1: onsite (3 sites) + hopping (2 bonds)
    assert len(order1) == 3 + 2
    # order 2 Strang: onsite half + hopping + onsite half
    assert len(order2) == 3 + 2 + 3
    assert order2.order == 2

    # a reversed edge is normalized to ascending (lo, hi) orientation.
    reversed_stream = fermi_hubbard_u1u1_jw_hopping_gate_stream([(2, 1)], dt)
    assert reversed_stream[0][1] == (1, 2)


def test_jw_hopping_rejects_non_adjacent_and_bad_order():
    pytest.importorskip("symmray")
    dt = 0.05
    with pytest.raises(ValueError):
        fermi_hubbard_u1u1_jw_hopping_gate_stream([(0, 2)], dt)
    with pytest.raises(ValueError):
        fermi_hubbard_u1u1_jw_gate_stream([(0, 1)], dt, order=3)
    with pytest.raises(ValueError):
        fermi_hubbard_u1u1_jw_interaction_gate_stream([], dt)

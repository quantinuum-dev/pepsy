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


def _dense_operator(arr):
    """Dense NumPy operator from a Symmray gate or MPO (double densify)."""
    out = arr.to_dense()
    return np.asarray(out.to_dense() if hasattr(out, "to_dense") else out)


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


def test_jw_trotter_gates_matches_low_level_wiring():
    """SymHamiltonian.jw_trotter_gates wires terms/params into the low-level stream."""
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian

    dt = 0.1
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", [(0, 1), (1, 2)], t=1.3, U=5.0, mu=0.4
    )
    method = ham.jw_trotter_gates(dt, order=2)
    manual = fermi_hubbard_u1u1_jw_gate_stream(
        [(0, 1), (1, 2)], dt, sites=range(3), t=1.3, U=5.0, mu=0.4, order=2
    )
    assert isinstance(method, SymGateStream)
    assert len(method) == len(manual)
    for (gate_m, where_m), (gate_l, where_l) in zip(method, manual):
        assert where_m == where_l
        assert np.allclose(_dense(gate_m), _dense(gate_l), atol=1e-12)


def test_jw_trotter_gates_spectrum_consistent_with_to_mpo():
    """jw_trotter_gates and to_mpo share one Jordan-Wigner conversion (same spectrum)."""
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian

    dt = 0.1
    # Pure hopping (U=0, mu=0): the whole L=2 MPO is exactly the hopping term.
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", [(0, 1)], t=1.0, U=0.0, mu=0.0
    )
    mpo_ham = _dense_operator(ham.to_mpo(L=2, compress=False)).reshape(16, 16)
    mpo_ham = (mpo_ham + mpo_ham.conj().T) / 2

    hop = [
        gate
        for gate, where in ham.jw_trotter_gates(dt, order=1)
        if isinstance(where, tuple) and len(where) == 2
    ][0]
    gate = _dense(hop).reshape(16, 16)

    gate_spectrum = np.sort(-np.angle(np.linalg.eigvals(gate)) / dt)
    mpo_spectrum = np.sort(np.linalg.eigvalsh(mpo_ham))
    assert np.allclose(gate_spectrum, mpo_spectrum, atol=1e-8)


def test_jw_trotter_gates_guards():
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian

    dt = 0.1
    # A bond that maps to non-adjacent chain sites is not a two-site JW gate.
    with pytest.raises(ValueError):
        SymHamiltonian.from_edges(
            "fermi_hubbard_u1u1", "U1U1", [(0, 1), (0, 2)], t=1.0
        ).jw_trotter_gates(dt)
    # The density-density V term is not yet supported by the gate path.
    with pytest.raises(NotImplementedError):
        SymHamiltonian.from_edges(
            "fermi_hubbard_u1u1", "U1U1", [(0, 1)], t=1.0, V=0.5
        ).jw_trotter_gates(dt)
    # Only the spinful U1U1 Fermi-Hubbard model has a Jordan-Wigner gate path.
    with pytest.raises(NotImplementedError):
        SymHamiltonian.from_edges("heisenberg", "U1", [(0, 1)]).jw_trotter_gates(dt)
    # Only Lie/Strang Trotter orders are defined.
    with pytest.raises(ValueError):
        SymHamiltonian.from_edges(
            "fermi_hubbard_u1u1", "U1U1", [(0, 1)], t=1.0
        ).jw_trotter_gates(dt, order=3)


def test_jw_energy_of_neel_product_state_is_zero():
    pytest.importorskip("symmray")
    from pepsy.tensors import (
        SymHamiltonian,
        SymMPS,
        site_charge_from_occupations,
    )

    L = 4
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(i, i + 1) for i in range(L - 1)],
        t=1.0,
        U=4.0,
        mu=0.0,
    )
    psi = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        L,
        bond_dim=1,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)]),
        seed=3,
        dtype="complex128",
        fermionic=False,
    )
    # A Neel product state has no double occupancy and no hopping expectation.
    assert abs(ham.jw_energy(psi)) < 1e-10


def test_jw_energy_rejects_non_symmps():
    pytest.importorskip("symmray")
    import quimb.tensor as qtn

    from pepsy.tensors import SymHamiltonian

    ham = SymHamiltonian.from_edges("fermi_hubbard_u1u1", "U1U1", [(0, 1)], t=1.0)
    with pytest.raises(TypeError):
        ham.jw_energy(qtn.MPS_rand_state(2, 2))


def test_jw_imaginary_time_evolution_reaches_ground_state():
    """End-to-end: jw_trotter_gates imaginary time + jw_energy reach the DMRG GS."""
    pytest.importorskip("symmray")
    import pepsy
    from pepsy.tensors import (
        SymHamiltonian,
        SymMPS,
        site_charge_from_occupations,
    )

    L = 4
    edges = [(i, i + 1) for i in range(L - 1)]
    sc = site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)])
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", edges, t=1.0, U=4.0, mu=0.0
    )
    mpo = ham.to_mpo(L=L, compress=False)

    ref = SymMPS.for_model(
        "fermi_hubbard_u1u1", L, bond_dim=1, site_charge=sc, seed=1, dtype="complex128"
    )
    opt = pepsy.SymDMRG2(
        mpo,
        ref,
        bond_dims=[1, 2, 4, 8, 16],
        cutoffs=[1e-10],
        mixer="density_matrix",
        compute_initial_energy=False,
    )
    opt.solve(max_sweeps=20, sweep_sequence="RL", tol=1e-11)
    e_gs = float(opt.energy)

    psi = SymMPS.for_model(
        "fermi_hubbard_u1u1", L, bond_dim=1, site_charge=sc, seed=2,
        dtype="complex128", fermionic=False,
    )
    step = ham.jw_trotter_gates(0.05, imaginary=True, order=2)
    for _ in range(200):
        psi.apply_gates(step, method="direct", max_bond=16, cutoff=1e-10, normalize=True)

    e_final = float(ham.jw_energy(psi))
    assert e_final < -1.0  # well below the product-state energy (0)
    assert abs(e_final - e_gs) < 5e-3


def test_jw_bond_layout_classifies_adjacent_and_long_range():
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian

    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", [(0, 1), (1, 2), (0, 2), (2, 3)], t=1.0
    )
    layout = ham.jw_bond_layout()
    assert layout["adjacent"] == [(0, 1), (1, 2), (2, 3)]
    assert layout["long_range"] == [(0, 2)]
    assert layout["sites"] == [0, 1, 2, 3]


def test_jw_bond_layout_with_2d_snake_mapper():
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian, OneDMap

    edges = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (1, 1))]
    ham = SymHamiltonian.from_edges("fermi_hubbard_u1u1", "U1U1", edges, t=1.0, U=4.0)
    layout = ham.jw_bond_layout(mapper=OneDMap(2, 2, mode="snake"))
    # A snake ordering of a 2x2 lattice makes at least one bond long-range.
    assert len(layout["long_range"]) >= 1
    assert len(layout["adjacent"]) + len(layout["long_range"]) == 4
    assert layout["sites"] == [0, 1, 2, 3]

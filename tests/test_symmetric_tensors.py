"""Tests for Symmray-backed symmetric MPS/PEPS helpers."""

import numpy as np
import pytest

import pepsy
from pepsy.operators import gate, gate_simple
from pepsy.tensors import (
    SymGateStream,
    SymHamiltonian,
    SymMPS,
    SymPEPS,
    default_physical_sectors,
    sector_index_map,
    site_charge_alternating,
    site_charge_from_map,
    site_charge_from_occupations,
    site_charge_uniform,
    symm_operator_from_dense,
)


sr = pytest.importorskip("symmray")


def test_sector_and_charge_helpers_make_total_charge_explicit():
    """Physical sectors and local charges should be easy to inspect."""
    assert default_physical_sectors("U1", 4) == {0: 1, 1: 2, 2: 1}
    assert default_physical_sectors(model="fermi_hubbard") == {0: 1, 1: 2, 2: 1}
    assert sector_index_map({0: 1, 1: 2, 2: 1}) == {0: 0, 1: 1, 2: 1, 3: 2}

    occupations = [1, 0, 1, 0]
    state = SymMPS.random(
        4,
        symmetry="U1",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(occupations),
        seed=31,
        dtype="complex128",
    )

    assert state.phys_sectors == {0: 1, 1: 1}
    assert state.site_charges() == {0: 1, 1: 0, 2: 1, 3: 0}
    assert state.overall_charge() == 2
    assert state.overall_parity() == 0
    assert site_charge_uniform(1)("anything") == 1
    assert site_charge_alternating(even=0, odd=1)((1, 2)) == 1
    assert site_charge_from_map({(0, 0): 1}, default=0)((1, 1)) == 0


def test_symmps_heisenberg_builds_energy_and_imaginary_step():
    """SymMPS should build U(1) Heisenberg terms and evolve in place."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )

    ham = state.build_hamiltonian()

    assert isinstance(ham, SymHamiltonian)
    assert state.symmetry == "U1"
    assert not state.fermionic
    assert len(ham.terms) == 3
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())
    assert state.tn.L == 4

    energy_before = state.energy(ham)
    state.ground_state(dt=0.01, steps=1, hamiltonian=ham, max_bond=4)
    energy_after = state.energy(ham)

    assert np.isfinite(np.real(energy_before))
    assert np.isfinite(np.real(energy_after))
    assert state.tn.max_bond() <= 4
    assert state.norm() == pytest.approx(1.0)


def test_symmps_measures_dense_generic_observables():
    """SymMPS.measure should convert dense local operators to Symmray arrays."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=32,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])
    zz_op = np.diag([1.0, -1.0, -1.0, 1.0])
    z_sym = symm_operator_from_dense(
        z_op,
        state.phys_sectors,
        symmetry=state.symmetry,
        charge=0,
    )

    measured_dense = state.measure(z_op, where=1, contraction_opt="auto-hq")
    measured_sym = state.measure(z_sym, where=1, contraction_opt="auto-hq")
    measured_zz = state.measure(zz_op, where=(1, 2), contraction_opt="auto-hq")

    assert measured_dense == pytest.approx(measured_sym)
    assert np.isfinite(np.real(measured_zz))


def test_symmps_fermi_hubbard_defaults_to_fermionic_u1():
    """Fermi-Hubbard convenience defaults should use U(1) fermionic tensors."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=2,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(t=1.0, U=4.0, mu=0.5)
    gates = state.trotter_gates(0.01, hamiltonian=ham, imaginary=True)

    assert state.symmetry == "U1"
    assert state.fermionic
    assert len(ham.terms) == 2
    assert all(term.shape == (4, 4, 4, 4) for term in ham.terms.values())
    assert isinstance(gates, SymGateStream)
    assert len(gates) == 2

    evolved = state.time_evolve(
        0.01,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        max_bond=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.max_bond() <= 4
    assert evolved.norm() == pytest.approx(1.0)


def test_symmps_gate_stream_runs_mps_optimizer_mpo_heisenberg():
    """Symmray U(1) gates should run through MpsOptimizer(mode='mpo')."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=7,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.01)

    opt = pepsy.MpsOptimizer(state.tn.copy(), gates, chi=4, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-10, fidelity_samples=0)

    assert out.L == 4
    assert out.max_bond() <= 4
    assert np.isfinite(np.real((out.H & out).contract(all, optimize="auto-hq")))


def test_symmps_mps_optimizer_handles_spinful_fermi_hubbard_dims():
    """MpsOptimizer MPO mode should accept 4-state Fermi-Hubbard gates."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=8,
        dtype="complex128",
    )
    ham = state.build_hamiltonian(t=1.0, U=2.0, mu=0.1)

    evolved = state.time_evolve_mps_optimizer(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        chi=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 3
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))
    raw = evolved.tn.copy()
    raw.exponent = 0
    assert (raw.H & raw).contract(all, optimize="auto-hq") == pytest.approx(1.0)


def test_symmps_mps_optimizer_coerces_dense_hamiltonian_terms():
    """Dense custom Hamiltonian terms should not mix NumPy gates into Symmray MPS."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=12,
        dtype="complex128",
    )
    zz_term = np.diag([1.0, -1.0, -1.0, 1.0]).reshape(2, 2, 2, 2)
    dense_hamiltonian = {(i, i + 1): zz_term for i in range(3)}

    ham = state.require_hamiltonian(hamiltonian=dense_hamiltonian)
    assert all(type(term).__module__.split(".")[0] == "symmray" for term in ham.terms.values())

    evolved = state.time_evolve_mps_optimizer(
        0.01,
        steps=1,
        hamiltonian=dense_hamiltonian,
        chi=4,
        mode="mpo",
        run_kwargs={"progbar": False, "fidelity_samples": 5},
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 4
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))


def test_sympeps_tfim_builds_z2_terms_and_step():
    """SymPEPS should build Z2 TFIM terms on a square grid."""
    state = SymPEPS.for_model(
        "itf",
        2,
        2,
        bond_dim=2,
        seed=3,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(jx=-1.0, hz=-0.5)

    assert state.symmetry == "Z2"
    assert not state.fermionic
    assert state.Lx == 2
    assert state.Ly == 2
    assert len(ham.terms) == 4
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())

    state.time_evolve(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=False,
        max_bond=4,
        normalize=False,
    )

    assert state.tn.max_bond() <= 4
    assert np.isfinite(np.real(state.norm()))


def test_sympeps_measures_dense_generic_observables_and_parity():
    """SymPEPS.measure should use quimb PEPS boundary contraction."""
    charges = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(charges),
        seed=33,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])

    bdy = {}
    measured = state.measure(
        z_op,
        where=(0, 0),
        contraction_opt="auto-hq",
        chi=4,
        bdy=bdy,
        progress=False,
    )
    exact = pepsy.measure_obs(
        state.tn,
        state.operator_from_dense(z_op),
        where=(0, 0),
        ind_id=state.site_ind_id,
        contraction_opt="auto-hq",
    )

    assert state.site_charges() == charges
    assert state.overall_parity() == 1
    assert "plaquette_envs" in bdy
    assert "plaquette_map" in bdy
    assert measured == pytest.approx(exact)


def test_sympeps_measure_requires_chi_without_boundary_holder():
    """Quimb PEPS boundary measurement should make chi selection explicit."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=34, dtype="complex128")
    z_op = np.diag([1.0, -1.0])

    with pytest.raises(ValueError, match="Provide chi"):
        state.measure(z_op, where=(0, 0), progress=False)


def test_sympeps_measure_delegates_to_quimb_boundary_modes():
    """Quimb MPS/projector boundary modes and CTMRG should accept Symmray data."""
    state = SymPEPS.for_model("itf", 3, 3, bond_dim=2, seed=44, dtype="complex128")
    z_op = np.diag([1.0, -1.0])
    z_sym = state.operator_from_dense(z_op)

    direct_quimb = state.tn.compute_local_expectation(
        {(1, 1): z_sym},
        max_bond=8,
        normalized=True,
        mode="mps",
        contract_optimize="auto-hq",
    )
    wrapped_mps = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="mps",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_projector = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="projector",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_ctmrg_alias = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="ctmrg",
        contraction_opt="auto-hq",
        progress=False,
    )

    norm = state.tn.make_norm()
    exact_norm = norm.contract(all, optimize="auto-hq")
    ctmrg_norm = norm.contract_ctmrg(
        max_bond=8,
        mode="projector",
        final_contract=True,
        final_contract_opts={"optimize": "auto-hq"},
        progbar=False,
    )

    assert wrapped_mps == pytest.approx(direct_quimb)
    assert wrapped_projector == pytest.approx(direct_quimb)
    assert wrapped_ctmrg_alias == pytest.approx(direct_quimb)
    assert ctmrg_norm == pytest.approx(exact_norm)


def test_sympeps_gate_stream_runs_pepsy_gate_and_gate_simple():
    """SymPEPS gate streams should work with PEPSY gate wrappers."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=9,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.005)

    out_gate = state.copy().apply_gates(
        gates,
        method="gate",
        max_bond=4,
        cutoff=1e-10,
    )
    gauges = {}
    out_simple = state.copy().apply_gates(
        gates,
        method="simple",
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
    )

    assert out_gate.tn.max_bond() <= 4
    assert out_simple.tn.max_bond() <= 4
    assert len(gauges) > 0
    assert np.isfinite(np.real(out_gate.norm()))
    assert np.isfinite(np.real(out_simple.norm()))


def test_sympeps_raw_pepsy_gate_functions_accept_symmray_streams():
    """The plain gate functions should accept a SymGateStream directly."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=10, dtype="complex128")
    gates = state.build_hamiltonian().gate_stream(0.005)
    gauges = {}

    out_gate = gate(state.tn.copy(), gates, max_bond=4, cutoff=1e-10, inplace=False)
    out_simple = gate_simple(
        state.tn.copy(),
        gates,
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
        inplace=False,
    )

    assert out_gate.max_bond() <= 4
    assert out_simple.max_bond() <= 4
    assert len(gauges) > 0


def test_symmetric_classes_are_top_level_lazy_exports():
    """Top-level pepsy exports should resolve to the tensor namespace classes."""
    assert pepsy.SymHamiltonian is SymHamiltonian
    assert pepsy.SymGateStream is SymGateStream
    assert pepsy.SymMPS is SymMPS
    assert pepsy.SymPEPS is SymPEPS
    assert pepsy.default_physical_sectors is default_physical_sectors
    assert pepsy.symm_operator_from_dense is symm_operator_from_dense

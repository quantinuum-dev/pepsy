"""Tests for the SymDMRG2 public driver scaffold."""

import quimb.tensor as qtn
import pytest

import pepsy
from pepsy.tensors import SymHamiltonian, SymMPS, site_charge_from_occupations


def test_symdmrg2_delegates_dense_mpo_to_quimb_dmrg2():
    """Dense/quimb MPOs should run through quimb's DMRG2 implementation."""
    mpo = qtn.MPO_ham_ising(2, j=1.0, bx=0.5)

    opt = pepsy.SymDMRG2(
        mpo,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        backend="quimb",
        tol=1e-6,
    )
    out = opt.solve()

    assert out is opt
    assert opt.backend == "quimb"
    assert opt.driver.__class__.__name__ == "DMRG2"
    assert opt.state.L == 2
    assert opt.state.max_bond() <= 4
    assert len(opt.energies) == 1
    assert opt.energy == pytest.approx(-0.5590169943749471)
    assert opt.summary()["energy"] == opt.energy


def test_symdmrg2_solves_two_site_symmray_fh_u1u1_dense_reference():
    """The first Symmray path should solve the whole L=2 theta sector."""
    pytest.importorskip("symmray")
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=11,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=8.0,
        mu=0.2,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    opt = pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)
    ref_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert opt.backend == "symmray"
    assert opt.uses_symmray
    assert opt.total_charge == (1, 1)
    assert complex(opt.initial_energy) == pytest.approx(complex(ref_energy))
    assert opt.summary()["total_charge"] == (1, 1)

    assert opt.environment_energy() == pytest.approx(complex(ref_energy))
    local_energy, theta = opt.dense_local_eigensolve(0)
    assert theta.inds == opt.two_site_theta(0).inds

    out = opt.solve(sweeps=2)
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.converged
    assert opt.state.max_bond() <= 4
    assert opt.energy == pytest.approx(local_energy)
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.energy <= ref_energy.real


def test_symdmrg2_solves_longer_chain_with_effective_norm():
    """L>2 uses H and N environments for a safe dense reference sweep."""
    pytest.importorskip("symmray")
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        3,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0)]),
        seed=13,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1), (1, 2)],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    mpo = ham.to_mpo(L=3, compress=False)

    opt = pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)
    ref_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    left, right = opt.build_environments()
    assert len(left) == len(right) == 4
    assert opt.environment_energy() == pytest.approx(complex(ref_energy))
    nleft, nright = opt.build_norm_environments()
    assert len(nleft) == len(nright) == 4
    assert opt.norm_environment_value() == pytest.approx(opt._current_norm())

    theta = opt.two_site_theta(0)
    htheta = opt.two_site_matvec(0, theta)
    ntheta = opt.two_site_norm_matvec(0, theta)
    assert htheta.inds == theta.inds
    assert ntheta.inds == theta.inds
    assert set(htheta.data.blocks) == set(theta.data.blocks)
    assert set(ntheta.data.blocks) == set(theta.data.blocks)

    local_energy, local_theta = opt.dense_generalized_local_eigensolve(0)
    assert local_theta.inds == theta.inds

    out = opt.solve()
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.state.max_bond() <= 4
    assert len(opt.energies) == 1
    assert opt.energy <= local_energy
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.environment_energy() == pytest.approx(complex(opt.energy))


def test_symdmrg2_rejects_quimb_backend_for_symmray_arrays():
    """Explicit quimb delegation should not silently accept Symmray arrays."""
    pytest.importorskip("symmray")
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=7,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=0.0,
        mu=0.0,
    )

    with pytest.raises(ValueError, match="backend='quimb'"):
        pepsy.SymDMRG2(ham.to_mpo(L=2, compress=False), state, backend="quimb")

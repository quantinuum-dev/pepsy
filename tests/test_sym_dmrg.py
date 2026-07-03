"""Tests for the SymDMRG2 public driver scaffold."""

import quimb.tensor as qtn
import pytest

import pepsy
from pepsy.tensors import SymHamiltonian, SymMPS, site_charge_from_occupations


def _fh_u1u1_chain(L, occupations, *, bond_dim=3, seed=13, U=2.0, mu=0.1):
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        L,
        bond_dim=bond_dim,
        site_charge=site_charge_from_occupations(occupations),
        seed=seed,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(site, site + 1) for site in range(L - 1)],
        t=1.0,
        U=U,
        mu=mu,
    )
    return state, ham.to_mpo(L=L, compress=False)


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
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

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
    assert opt.energy <= local_energy + 1e-12
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.environment_energy() == pytest.approx(complex(opt.energy))


def test_symdmrg2_linear_operator_matches_dense_effective_hamiltonian():
    """The Lanczos LinearOperator should be the same H_eff as dense columns."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)
    opt._canonize_for_sweep("right")
    opt.build_environments()
    theta = opt.two_site_theta(0)
    space = opt.two_site_theta_space(0, theta)
    operator = opt.two_site_effective_hamiltonian(0, theta)
    matrix = opt._dense_operator_matrix(
        0,
        theta,
        space.metadata,
        opt.two_site_matvec,
    )

    vector = space.vector
    assert operator.shape == matrix.shape
    assert operator @ vector == pytest.approx(matrix @ vector)

    vectors = vector.reshape(-1, 1)
    vectors = vectors.repeat(2, axis=1)
    assert operator @ vectors == pytest.approx(matrix @ vectors)


def test_symdmrg2_lanczos_matches_dense_after_canonicalization():
    """Canonical-center Lanczos should reproduce the dense local reference."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-12,
        local_eig_ncv=8,
    )
    opt._canonize_for_sweep("right")
    opt.build_environments()
    opt.build_norm_environments()
    theta = opt.two_site_theta(0)

    assert opt.effective_norm_identity_error(0, theta, samples=3) < 1e-12
    hermitian, herm_error = opt.check_two_site_hermiticity(0, theta, samples=3)
    assert hermitian, herm_error

    dense_energy, dense_theta = opt.dense_local_eigensolve(0)
    lanczos_energy, lanczos_theta = opt.lanczos_local_eigensolve(0, theta=theta)

    assert lanczos_energy == pytest.approx(dense_energy)
    assert lanczos_theta.inds == dense_theta.inds
    assert set(lanczos_theta.data.blocks) == set(dense_theta.data.blocks)


def test_symdmrg2_rejects_cyclic_symmray_mps_chain():
    """The Symmray DMRG path is deliberately restricted to OBC MPS chains."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])
    state.mps.cyclic = True

    with pytest.raises(ValueError, match="assumes an OBC MPS chain"):
        pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)


def test_symdmrg2_rejects_cyclic_symmray_mpo_chain():
    """Periodic lattice physics should be encoded as long-range OBC MPO terms."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])
    mpo.cyclic = True

    with pytest.raises(ValueError, match="assumes an OBC MPO chain"):
        pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)


def test_symdmrg2_forces_lanczos_sweep_on_four_site_chain():
    """Forced Lanczos sweeps should still match the independent MPO energy."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
        seed=17,
        U=1.0,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=3,
        cutoff=1e-10,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-10,
        local_eig_ncv=8,
    )
    out = opt.solve()
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.state.max_bond() <= 3
    assert len(opt.energies) == 1
    assert complex(post_energy) == pytest.approx(complex(opt.energy))


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

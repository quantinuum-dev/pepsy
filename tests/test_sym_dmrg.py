"""Tests for the SymDMRG2 public driver scaffold."""

import numpy as np
import importlib.util
from pathlib import Path
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


def _dense_jw_fermi_hubbard(L, edges, *, t=1.0, U=4.0, mu=0.3):
    n_modes = 2 * L
    eye = np.eye(2)
    zed = np.array([[1.0, 0.0], [0.0, -1.0]])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])

    def annihilate(mode):
        mats = [zed] * mode + [lower] + [eye] * (n_modes - mode - 1)
        out = mats[0]
        for mat in mats[1:]:
            out = np.kron(out, mat)
        return out

    def mode(site, spin):
        return 2 * site + spin

    ham = np.zeros((2**n_modes, 2**n_modes))
    for i, j in edges:
        for spin in (0, 1):
            hop = annihilate(mode(i, spin)).conj().T @ annihilate(mode(j, spin))
            ham += -t * (hop + hop.conj().T)
    for site in range(L):
        num_up = annihilate(mode(site, 0)).conj().T @ annihilate(mode(site, 0))
        num_dn = annihilate(mode(site, 1)).conj().T @ annihilate(mode(site, 1))
        ham += U * (num_up @ num_dn) - mu * (num_up + num_dn)
    return ham


def _u1u1_sector_indices(L, total_charge):
    n_modes = 2 * L
    num_up, num_down = tuple(total_charge)
    indices = []
    for basis in range(2**n_modes):
        up_count = 0
        down_count = 0
        for site in range(L):
            up_count += (basis >> (n_modes - 1 - 2 * site)) & 1
            down_count += (basis >> (n_modes - 2 - 2 * site)) & 1
        if (up_count, down_count) == (num_up, num_down):
            indices.append(basis)
    return np.asarray(indices, dtype=int)


def _fixed_u1u1_sector_ground_energy(L, edges, total_charge, *, t=1.0, U=4.0, mu=0.3):
    ham = _dense_jw_fermi_hubbard(L, edges, t=t, U=U, mu=mu)
    sector = _u1u1_sector_indices(L, total_charge)
    sector_ham = ham[np.ix_(sector, sector)]
    sector_ham = (sector_ham + sector_ham.conj().T) / 2
    return float(np.linalg.eigvalsh(sector_ham)[0].real)


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


def test_symdmrg2_initial_energy_can_be_disabled_or_lazy(monkeypatch):
    """Startup energy measurement should be optional or deferred."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(2, [(1, 0), (0, 1)])

    def raise_if_called(*_args, **_kwargs):
        raise AssertionError("initial energy should not be computed")

    monkeypatch.setattr(pepsy.SymDMRG2, "_compute_initial_energy", raise_if_called)
    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        compute_initial_energy=False,
    )

    assert opt.initial_energy is None
    assert opt.energy is None
    assert opt.summary()["initial_energy_mode"] == "off"
    assert opt.summary()["initial_energy_computed"] is False
    assert opt.summary()["initial_energy"] is None
    assert opt.summary()["energy"] is None

    calls = []

    def fake_initial_energy(self):
        calls.append(self)
        return -1.25

    state, mpo = _fh_u1u1_chain(2, [(1, 0), (0, 1)])
    monkeypatch.setattr(pepsy.SymDMRG2, "_compute_initial_energy", fake_initial_energy)
    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        compute_initial_energy="lazy",
    )

    assert calls == []
    assert opt.summary()["initial_energy_mode"] == "lazy"
    assert opt.summary()["initial_energy_computed"] is False
    assert opt.summary()["initial_energy"] is None
    assert opt.summary()["energy"] is None
    assert calls == []
    assert opt.initial_energy == pytest.approx(-1.25)
    assert opt.energy == pytest.approx(-1.25)
    assert len(calls) == 1
    assert opt.summary()["initial_energy_computed"] is True


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

    out = opt.solve(max_sweeps=2, sweep_sequence="RL")
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.state.max_bond() <= 4
    assert len(opt.energies) == 2
    assert opt.energy <= local_energy + 1e-12
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.environment_energy() == pytest.approx(complex(opt.energy))


def test_symdmrg2_directional_environment_builds_only_static_side(monkeypatch):
    """Sweep setup should not prebuild environments that are updated locally."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(4, [(1, 0), (0, 1), (1, 0), (0, 1)])
    opt = pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, sweeps=1)

    calls = {"left": 0, "right": 0}
    left_step = opt._left_env_step
    right_step = opt._right_env_step

    def counted_left_step(site, env, bra):
        calls["left"] += 1
        return left_step(site, env, bra)

    def counted_right_step(site, env, bra):
        calls["right"] += 1
        return right_step(site, env, bra)

    monkeypatch.setattr(opt, "_left_env_step", counted_left_step)
    monkeypatch.setattr(opt, "_right_env_step", counted_right_step)

    opt.build_sweep_environments("right")
    assert calls == {"left": 0, "right": opt.state.L}
    assert opt.left_envs[0] is not None
    assert all(env is None for env in opt.left_envs[1:])
    assert all(env is not None for env in opt.right_envs)

    calls["left"] = 0
    calls["right"] = 0
    opt.build_sweep_environments("left")
    assert calls == {"left": opt.state.L, "right": 0}
    assert all(env is not None for env in opt.left_envs)
    assert all(env is None for env in opt.right_envs[:-1])
    assert opt.right_envs[-1] is not None


def test_symdmrg2_accepts_quimb_style_solve_controls():
    """SymDMRG2 should accept DMRG2-style p0/bond_dims/cutoffs controls."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        p0=state,
        bond_dims=[3, 4],
        cutoffs=[1e-9, 1e-10],
        local_solver="dense",
    )
    out = opt.solve(max_sweeps=2, sweep_sequence="RL", tol=0.0)

    assert out is opt
    assert opt.summary()["bond_dims"] == (3, 4)
    assert opt.summary()["cutoffs"] == (1e-09, 1e-10)
    assert len(opt.energies) == 2
    assert len(opt.local_energies) == 2
    assert len(opt.total_energies) == 2
    assert {diag["chi"] for diag in opt.svd_diagnostics} == {3, 4}
    assert opt.state.max_bond() <= 4


def test_symdmrg2_sweep_uses_quimb_progress_bar(monkeypatch):
    """verbosity>0 should wrap the site sweep with quimb's progbar helper."""
    pytest.importorskip("symmray")
    import quimb.utils as qutils

    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])
    opt = pepsy.SymDMRG2(mpo, state, chi=4, cutoff=1e-10, local_solver="dense")
    calls = []

    class DummyProgbar:
        def __init__(self, iterable):
            self.iterable = iterable
            self.closed = False

        def __iter__(self):
            return iter(self.iterable)

        def close(self):
            self.closed = True

    def fake_progbar(iterable, **kwargs):
        bar = DummyProgbar(iterable)
        calls.append((kwargs, bar))
        return bar

    monkeypatch.setattr(qutils, "progbar", fake_progbar)

    energy = opt.sweep("R", verbosity=1, max_bond=4, cutoff=1e-10)

    assert isinstance(energy, float)
    assert len(opt.energies) == 0
    assert len(opt.local_energies) == 1
    assert calls[0][0]["total"] == opt.state.L - 1
    assert calls[0][0]["ncols"] == 80
    assert calls[0][1].closed


def test_symdmrg2_profile_records_phase_timings():
    """Opt-in profiling should record useful timing phases."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        profile=True,
    )
    opt.solve(max_sweeps=1)
    phases = {event["phase"] for event in opt.profile_diagnostics}
    summary = opt.profile_summary()

    assert {
        "build_dense_environments",
        "build_norm_environments",
        "build_block_environments",
        "local_solve",
        "local_eigensolver",
        "matvec",
        "norm_check",
        "svd_split",
        "sweep",
        "solve",
    } <= phases
    assert all(event["elapsed"] >= 0.0 for event in opt.profile_diagnostics)
    assert summary["enabled"]
    assert summary["num_events"] == len(opt.profile_diagnostics)
    assert summary["num_matvecs"] >= 1
    assert summary["phase_counts"]["matvec"] >= 1
    assert opt.summary()["num_profile_diagnostics"] == len(opt.profile_diagnostics)
    assert opt.summary()["last_profile_diagnostic"] == opt.last_profile_diagnostic


def test_symdmrg2_benchmark_harness_returns_json_ready_result():
    """The benchmark helper should produce structured profiling data."""
    pytest.importorskip("symmray")
    bench_path = Path(__file__).resolve().parents[1] / "benchmarks" / "symdmrg2_fh_u1u1.py"
    spec = importlib.util.spec_from_file_location("symdmrg2_fh_u1u1_benchmark", bench_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_benchmark(
        length=2,
        chi=3,
        initial_bond_dim=2,
        sweeps=1,
        local_solver="dense",
        dense_threshold=100,
        include_events=True,
    )

    assert result["case"]["length"] == 2
    assert result["case"]["norm_check"] == "strict"
    assert result["case"]["norm_check_interval"] == 1
    assert result["case"]["residual_check"] == "sampled"
    assert result["case"]["residual_check_interval"] == 1
    assert result["case"]["residual_check_tol"] is None
    assert result["case"]["matvec_diagnostics"] == "sampled"
    assert result["case"]["matvec_diagnostics_interval"] == 1
    assert result["case"]["compute_initial_energy"] is False
    assert result["result"]["num_sweeps"] == 1
    assert isinstance(result["result"]["energy"], float)
    assert result["result"]["num_residual_diagnostics"] == 1
    assert result["result"]["num_matvec_diagnostics"] == result["profile"]["num_matvecs"]
    assert result["profile"]["enabled"]
    assert result["profile"]["num_events"] == len(result["profile_events"])
    assert result["profile"]["num_matvec_diagnostics"] == len(
        result["matvec_diagnostics"]
    )
    assert result["profile"]["phase_counts"]["sweep"] == 1
    assert result["profile"]["phase_counts"]["residual_check"] == 1
    assert result["profile"]["num_residual_checks"] == 1
    assert result["profile"]["num_matvecs"] >= 1
    assert result["compression"]["num_splits"] == result["result"]["num_svd_diagnostics"]
    assert result["compression"]["max_bond_dim"] <= 3


def test_symdmrg2_norm_check_off_skips_effective_norm_probe(monkeypatch):
    """Production runs can skip the expensive N_eff ~= I probe explicitly."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        norm_check="off",
    )

    def raise_if_called(*_args, **_kwargs):
        raise AssertionError("effective_norm_identity_error should be skipped")

    monkeypatch.setattr(opt, "effective_norm_identity_error", raise_if_called)
    opt.solve(max_sweeps=1)

    assert len(opt.norm_identity_diagnostics) == 2
    assert all(diag["skipped"] for diag in opt.norm_identity_diagnostics)
    assert {diag["mode"] for diag in opt.norm_identity_diagnostics} == {"off"}
    assert all(diag["samples"] == 0 for diag in opt.norm_identity_diagnostics)
    assert all(diag["error"] is None for diag in opt.norm_identity_diagnostics)
    assert all(diag["norm_error"] is None for diag in opt.local_solve_diagnostics)
    assert opt.summary()["norm_check"] == "off"


def test_symdmrg2_skipped_canonicalization_forces_norm_probe(monkeypatch):
    """H-only solves must validate N_eff if requested canonicalization is unavailable."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        norm_check="off",
    )

    calls = []

    def skipped_canonize(direction):
        assert direction == "right"
        return False

    def identity_norm_probe(site, theta):
        calls.append(int(site))
        return 0.0

    monkeypatch.setattr(opt, "_canonize_for_sweep", skipped_canonize)
    monkeypatch.setattr(opt, "effective_norm_identity_error", identity_norm_probe)

    opt.sweep("R", max_bond=4, cutoff=1e-10)

    assert calls == [0, 1]
    assert len(opt.norm_identity_diagnostics) == 2
    assert all(not diag["skipped"] for diag in opt.norm_identity_diagnostics)
    assert all(diag["forced"] for diag in opt.norm_identity_diagnostics)
    assert {diag["reason"] for diag in opt.norm_identity_diagnostics} == {
        "right_canonize_unavailable"
    }
    assert all(diag["norm_error"] == 0.0 for diag in opt.local_solve_diagnostics)


def test_symdmrg2_skipped_canonicalization_bad_norm_raises(monkeypatch):
    """Forced N_eff checks should fail fast when canonicalization is unsafe."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        norm_check="off",
        norm_check_tol=1e-6,
    )

    monkeypatch.setattr(opt, "_canonize_for_sweep", lambda direction: False)
    monkeypatch.setattr(opt, "effective_norm_identity_error", lambda site, theta: 1e-3)

    with pytest.raises(ValueError, match="Effective norm is not identity-like"):
        opt.sweep("R", max_bond=4, cutoff=1e-10)

    assert len(opt.norm_identity_diagnostics) == 1
    assert opt.norm_identity_diagnostics[0]["forced"] is True
    assert opt.norm_identity_diagnostics[0]["reason"] == "right_canonize_unavailable"
    assert opt.norm_identity_diagnostics[0]["passed"] is False


def test_symdmrg2_sampled_norm_check_checks_boundaries_and_skips_interval():
    """Sampled checks should keep boundary probes and skip selected interiors."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        norm_check="sampled",
        norm_check_interval=2,
    )
    opt.solve(max_sweeps=1, sweep_sequence="R")

    by_site = {diag["site"]: diag for diag in opt.norm_identity_diagnostics}

    assert set(by_site) == {0, 1, 2}
    assert not by_site[0]["skipped"]
    assert by_site[1]["skipped"]
    assert not by_site[2]["skipped"]
    assert by_site[0]["error"] is not None
    assert by_site[1]["error"] is None
    assert by_site[2]["error"] is not None
    assert {diag["mode"] for diag in by_site.values()} == {"sampled"}
    assert opt.summary()["norm_check_interval"] == 2


def test_symdmrg2_sampled_residual_check_records_boundaries_and_interval():
    """Sampled residuals should catch local eigensolver correctness cheaply."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        residual_check="sampled",
        residual_check_interval=2,
        residual_check_tol=1e-8,
    )
    opt.solve(max_sweeps=1, sweep_sequence="R")

    by_site = {diag["site"]: diag for diag in opt.residual_diagnostics}
    by_local_site = {diag["site"]: diag for diag in opt.local_solve_diagnostics}

    assert set(by_site) == {0, 1, 2}
    assert not by_site[0]["skipped"]
    assert by_site[1]["skipped"]
    assert not by_site[2]["skipped"]
    assert by_site[0]["residual_norm"] < 1e-8
    assert by_site[1]["residual_norm"] is None
    assert by_site[2]["residual_norm"] < 1e-8
    assert by_site[0]["passed"] is True
    assert by_site[1]["passed"] is None
    assert by_site[2]["passed"] is True
    assert {diag["mode"] for diag in by_site.values()} == {"sampled"}
    assert opt.summary()["residual_check_interval"] == 2
    assert opt.summary()["num_residual_diagnostics"] == len(opt.residual_diagnostics)
    assert opt.summary()["last_residual_diagnostic"] == opt.last_residual_diagnostic
    assert by_local_site[0]["residual_norm"] == pytest.approx(
        by_site[0]["residual_norm"]
    )
    assert by_local_site[1]["residual_check_skipped"] is True
    assert by_local_site[2]["residual_check_passed"] is True


def test_symdmrg2_convergence_uses_residual_and_truncation_gates():
    """Optional convergence gates should use per-sweep diagnostic maxima."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=3,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        local_solver="dense",
        convergence_residual_tol=1e-8,
        convergence_truncation_tol=1.0,
        energy_tol_per_site=True,
    )
    opt.solve(max_sweeps=2, sweep_sequence="RR", tol=1e9)

    first, last = opt.convergence_diagnostics
    summary = opt.summary()

    assert opt.converged is True
    assert opt.summary()["residual_check"] == "strict"
    assert summary["energy_tol_per_site"] is True
    assert summary["num_convergence_diagnostics"] == 2
    assert summary["last_convergence_diagnostic"] == opt.last_convergence_diagnostic
    assert first["energy_converged"] is False
    assert last["energy_converged"] is True
    assert last["residual_converged"] is True
    assert last["truncation_converged"] is True
    assert last["converged"] is True
    assert last["num_local_solves"] == 3
    assert last["num_svd_splits"] == 3
    assert last["num_residual_checks"] == 3
    assert last["num_skipped_residual_checks"] == 0
    assert last["max_residual_norm"] < 1e-8
    assert last["energy_scale"] == pytest.approx(4.0)


def test_symdmrg2_convergence_truncation_gate_can_block_energy_convergence():
    """A strict truncation gate should keep energy-only convergence honest."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=3,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=1,
        cutoff=1e-10,
        local_solver="dense",
        convergence_truncation_tol=0.0,
    )
    opt.solve(max_sweeps=1, sweep_sequence="R", tol=1e9)

    offsets = opt._sweep_convergence_offsets()
    energy = opt.sweep("R")
    opt.energies.append(energy)
    for diagnostic in opt.svd_diagnostics[offsets["svd"]:]:
        diagnostic["truncation_error"] = 1e-3
    opt.converged = opt._check_convergence(1e9, offsets)
    last = opt.last_convergence_diagnostic

    assert opt.converged is False
    assert len(opt.energies) == 2
    assert last["energy_converged"] is True
    assert last["truncation_converged"] is False
    assert last["max_truncation_error"] == pytest.approx(1e-3)


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


def test_symdmrg2_block_native_matvec_matches_dense_reference_all_sites():
    """The Symmray matvec should equal the dense validator in theta space."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
        seed=19,
        U=1.25,
        mu=0.05,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        matvec_backend="symmray",
    )
    opt._canonize_for_sweep("right")
    opt.build_environments()
    opt.build_block_environments()

    rng = np.random.default_rng(4)
    for site in range(opt.state.L - 1):
        theta = opt.two_site_theta(site)
        space = opt.two_site_theta_space(site, theta)
        vector = rng.standard_normal(space.dim) + 1.0j * rng.standard_normal(space.dim)
        trial = space.unflatten(vector)

        dense = opt.two_site_matvec_dense_reference(site, trial)
        native = opt.two_site_matvec_symmray(site, trial)
        problem = opt._last_matvec_projected_problem
        summary = problem.summary()
        expected_order = (
            "left_first"
            if summary["left_projector_dim"] > 2 * summary["right_projector_dim"]
            else "right_first"
        )

        assert native.inds == dense.inds == theta.inds
        assert summary["matvec_contraction_order"] == expected_order
        for prefix, contraction in (
            ("left_contract", problem.left_contraction),
            ("right_contract", problem.right_contraction),
        ):
            if contraction.shared:
                sizes = [contraction.left.ind_size(ind) for ind in contraction.shared]
                assert summary[f"{prefix}_contracted_ind_size"] == max(sizes)
        assert set(native.data.blocks) == set(dense.data.blocks) == set(theta.data.blocks)
        for sector in theta.data.blocks:
            assert native.data.blocks[sector] == pytest.approx(dense.data.blocks[sector])


def test_symdmrg2_block_native_matvec_reuses_projected_problem_cache(monkeypatch):
    """Repeated matvecs for one window should reuse cached local projectors."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
        seed=29,
        U=1.25,
        mu=0.05,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        matvec_backend="symmray",
    )
    opt._canonize_for_sweep("right")
    opt.build_environments()
    opt.build_block_environments()

    calls = []
    original = opt._active_mpo_tensor_for_matvec

    def wrapped(site, input_map):
        calls.append(int(site))
        return original(site, input_map)

    monkeypatch.setattr(opt, "_active_mpo_tensor_for_matvec", wrapped)

    site = 1
    theta = opt.two_site_theta(site)
    space = opt.two_site_theta_space(site, theta)
    rng = np.random.default_rng(41)
    trial_a = space.unflatten(
        rng.standard_normal(space.dim) + 1.0j * rng.standard_normal(space.dim)
    )
    trial_b = space.unflatten(
        rng.standard_normal(space.dim) + 1.0j * rng.standard_normal(space.dim)
    )

    dense_a = opt.two_site_matvec_dense_reference(site, trial_a)
    native_a = opt.two_site_matvec_symmray(site, trial_a)
    dense_b = opt.two_site_matvec_dense_reference(site, trial_b)
    native_b = opt.two_site_matvec_symmray(site, trial_b)

    assert calls == [site, site + 1]
    assert opt.projected_problem_cache_misses == 1
    assert opt.projected_problem_cache_hits == 1
    assert opt.summary()["projected_problem_cache_hits"] == 1
    assert opt.profile_summary()["projected_problem_cache_misses"] == 1
    for sector in theta.data.blocks:
        assert native_a.data.blocks[sector] == pytest.approx(dense_a.data.blocks[sector])
        assert native_b.data.blocks[sector] == pytest.approx(dense_b.data.blocks[sector])


def test_symdmrg2_matvec_diagnostics_record_cache_and_projector_stats():
    """Sampled matvec records should explain cached block-native matvec cost."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        4,
        [(1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
        seed=31,
        U=1.25,
        mu=0.05,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        matvec_backend="symmray",
        matvec_diagnostics="strict",
        profile=True,
    )
    opt._canonize_for_sweep("right")
    opt.build_environments()
    opt.build_block_environments()

    site = 1
    theta = opt.two_site_theta(site)
    space = opt.two_site_theta_space(site, theta)
    rng = np.random.default_rng(43)
    trial_a = space.unflatten(
        rng.standard_normal(space.dim) + 1.0j * rng.standard_normal(space.dim)
    )
    trial_b = space.unflatten(
        rng.standard_normal(space.dim) + 1.0j * rng.standard_normal(space.dim)
    )

    opt.two_site_matvec(site, trial_a)
    opt.two_site_matvec(site, trial_b)

    first, second = opt.matvec_diagnostic_records

    assert len(opt.matvec_diagnostic_records) == 2
    assert first["projected_problem_cache_hit"] is False
    assert second["projected_problem_cache_hit"] is True
    assert first["theta_dim"] == space.dim
    assert first["theta_num_blocks"] == len(theta.data.blocks)
    assert first["matvec_num_contractions"] == 2
    assert first["matvec_contraction_order"] in {"right_first", "left_first"}
    assert first["projected_block_terms"] > 0
    assert first["left_projector_num_blocks"] > 0
    assert first["right_projector_num_blocks"] > 0
    assert first["right_contract_shared_inds"] >= 1
    assert first["left_contract_shared_inds"] >= 1
    assert first["matvec_input_reindex_elapsed"] >= 0.0
    assert first["matvec_right_contract_elapsed"] >= 0.0
    assert first["matvec_left_contract_elapsed"] >= 0.0
    assert first["matvec_output_blocks_elapsed"] >= 0.0
    assert first["elapsed"] >= 0.0
    assert opt.summary()["num_matvec_diagnostics"] == 2
    assert opt.summary()["last_matvec_diagnostic"] == opt.last_matvec_diagnostic
    assert opt.profile_summary()["num_matvec_diagnostics"] == 2
    assert opt.profile_summary()["matvec_timing_totals"][
        "matvec_right_contract_elapsed"
    ] >= 0.0


def test_symdmrg2_matvec_skips_block_stats_without_profile_or_diagnostics(monkeypatch):
    """Lanczos hot-loop metadata should be free when profile/diagnostics are off."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        matvec_backend="dense_reference",
        matvec_diagnostics="off",
        profile=False,
    )
    theta = opt.two_site_theta(0)

    def raise_if_called(*_args, **_kwargs):
        raise AssertionError("_tensor_block_stats should not run")

    monkeypatch.setattr(opt, "_tensor_block_stats", raise_if_called)

    out = opt.two_site_matvec(0, theta)

    assert out.inds == theta.inds
    assert opt.profile_diagnostics == []
    assert opt.matvec_diagnostic_records == []


def test_symdmrg2_dense_reference_matvec_backend_remains_selectable():
    """The dense-aligned matvec stays available as a debug fallback."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(3, [(1, 0), (0, 1), (1, 0)])

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        matvec_backend="dense_reference",
    )
    theta = opt.two_site_theta(0)
    via_dispatch = opt.two_site_matvec(0, theta)
    direct = opt.two_site_matvec_dense_reference(0, theta)

    assert opt.summary()["matvec_backend"] == "dense_reference"
    assert opt.summary()["resolved_matvec_backend"] == "dense_reference"
    assert set(via_dispatch.data.blocks) == set(direct.data.blocks)
    for sector in theta.data.blocks:
        assert via_dispatch.data.blocks[sector] == pytest.approx(direct.data.blocks[sector])


def test_symdmrg2_rejects_fermionic_state_without_bosonic_symmray_mpo(monkeypatch):
    """Bosonizing a fermionic state requires a matching bosonic/JW MPO."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(2, [(1, 0), (0, 1)])

    monkeypatch.setattr(
        pepsy.MpsEnergyOptimizer,
        "_mpo_uses_bosonic_symmray",
        classmethod(lambda cls, mpo: False),
    )

    with pytest.raises(ValueError, match="bosonic/Jordan-Wigner Symmray MPO"):
        pepsy.SymDMRG2(
            mpo,
            state,
            chi=4,
            cutoff=1e-10,
            sweeps=1,
        )


def test_symdmrg2_rejects_fermionic_state_with_fermionic_symmray_mpo(monkeypatch):
    """A mixed/fermionic Symmray MPO should not pass the bosonization guard."""
    pytest.importorskip("symmray")
    import pepsy.optimizers.sym_dmrg as sym_dmrg_mod

    state, mpo = _fh_u1u1_chain(2, [(1, 0), (0, 1)])
    mpo_data_ids = {id(tensor.data) for tensor in mpo}
    original = sym_dmrg_mod._is_fermionic_symmray_array

    def mark_mpo_data_fermionic(data):
        if id(data) in mpo_data_ids:
            return True
        return original(data)

    monkeypatch.setattr(
        sym_dmrg_mod,
        "_is_fermionic_symmray_array",
        mark_mpo_data_fermionic,
    )

    with pytest.raises(ValueError, match="fermionic Symmray MPO"):
        pepsy.SymDMRG2(
            mpo,
            state,
            chi=4,
            cutoff=1e-10,
            sweeps=1,
        )


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
    current_theta = opt.two_site_theta(0)
    theta = opt.two_site_variational_theta(0, current_theta)

    assert opt.effective_norm_identity_error(0, theta, samples=3) < 1e-12
    hermitian, herm_error = opt.check_two_site_hermiticity(0, theta, samples=3)
    assert hermitian, herm_error

    dense_energy, dense_theta = opt.dense_local_eigensolve(0, theta=theta)
    lanczos_energy, lanczos_theta = opt.lanczos_local_eigensolve(0, theta=theta)

    assert sum(block.size for block in theta.data.blocks.values()) >= sum(
        block.size for block in current_theta.data.blocks.values()
    )
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
    out = opt.solve(max_sweeps=2, sweep_sequence="RL")
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.state.max_bond() <= 3
    assert len(opt.energies) == 2
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert len(opt.svd_diagnostics) == 2 * (4 - 1)
    assert opt.summary()["num_svd_diagnostics"] == len(opt.svd_diagnostics)
    assert opt.summary()["last_svd_diagnostic"] == opt.last_svd_diagnostic
    assert opt.summary()["compression_summary"] == opt.compression_summary()
    assert len(opt.norm_identity_diagnostics) == len(opt.svd_diagnostics)
    assert len(opt.local_solve_diagnostics) == len(opt.svd_diagnostics)
    assert opt.summary()["num_norm_identity_diagnostics"] == len(
        opt.norm_identity_diagnostics
    )
    assert opt.summary()["last_norm_identity_diagnostic"] == (
        opt.norm_identity_diagnostics[-1]
    )
    assert opt.summary()["num_local_solve_diagnostics"] == len(
        opt.local_solve_diagnostics
    )
    assert opt.summary()["last_local_solve_diagnostic"] == (
        opt.local_solve_diagnostics[-1]
    )
    assert {diag["direction"] for diag in opt.norm_identity_diagnostics} == {
        "right",
        "left",
    }
    assert all(diag["passed"] for diag in opt.norm_identity_diagnostics)
    assert max(diag["error"] for diag in opt.norm_identity_diagnostics) < 1e-12
    assert all(
        diag["requested_solver"] == "lanczos" for diag in opt.local_solve_diagnostics
    )
    assert all(
        diag["matvec_backend"] == "symmray" for diag in opt.local_solve_diagnostics
    )
    compression = opt.compression_summary()
    assert compression["num_splits"] == len(opt.svd_diagnostics)
    assert compression["max_bond_dim"] <= 3
    assert compression["num_truncation_errors"] == len(opt.svd_diagnostics)
    assert compression["num_missing_truncation_errors"] == 0
    assert compression["max_truncation_error"] is not None
    for diagnostic in opt.svd_diagnostics:
        assert "truncation_error" in diagnostic
        assert diagnostic["left"]["bond_dim"] <= 3
        assert diagnostic["right"]["bond_dim"] <= 3
        assert diagnostic["left"]["num_sectors"] >= 1
        assert diagnostic["right"]["num_sectors"] >= 1


def test_symdmrg2_lanczos_reaches_fixed_sector_ed_with_full_initial_support():
    """With enough initial bond sectors, Lanczos reaches the OBC FH ED energy."""
    pytest.importorskip("symmray")
    edges = [(site, site + 1) for site in range(3)]
    occupations = [(1, 0), (0, 1), (1, 0), (0, 1)]
    state, mpo = _fh_u1u1_chain(
        4,
        occupations,
        bond_dim=12,
        seed=31,
        U=1.0,
        mu=0.1,
    )
    ed_energy = _fixed_u1u1_sector_ground_energy(
        4,
        edges,
        (2, 2),
        t=1.0,
        U=1.0,
        mu=0.1,
    )
    initial_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy.real

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=16,
        cutoff=1e-12,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-11,
        local_eig_ncv=16,
        norm_check_samples=3,
    )
    opt.solve(tol=0.0, max_sweeps=2, sweep_sequence="RL")
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert opt.energy == pytest.approx(ed_energy, abs=1e-10)
    assert opt.energy <= initial_energy
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.state.max_bond() <= 16
    assert len(opt.norm_identity_diagnostics) == 2 * (4 - 1)
    assert len(opt.local_solve_diagnostics) == 2 * (4 - 1)
    assert all(diag["passed"] for diag in opt.norm_identity_diagnostics)
    assert max(diag["error"] for diag in opt.norm_identity_diagnostics) < 1e-12
    assert all(diag["solver"] == "lanczos" for diag in opt.local_solve_diagnostics)
    assert all(diag["theta_dim"] > 2 for diag in opt.local_solve_diagnostics)


def test_symdmrg2_nucleates_sectors_from_product_state_without_enrichment():
    """The local two-site space should include absent but charge-valid sectors."""
    pytest.importorskip("symmray")
    edges = [(site, site + 1) for site in range(3)]
    occupations = [(1, 0), (0, 1), (1, 0), (0, 1)]
    state, mpo = _fh_u1u1_chain(
        4,
        occupations,
        bond_dim=1,
        seed=31,
        U=1.0,
        mu=0.1,
    )
    ed_energy = _fixed_u1u1_sector_ground_energy(
        4,
        edges,
        (2, 2),
        t=1.0,
        U=1.0,
        mu=0.1,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=16,
        cutoff=1e-12,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-11,
        local_eig_ncv=16,
        norm_check_samples=3,
        sector_enrichment="none",
        mixer="none",
    )
    current_theta = opt.two_site_theta(1)
    variational_theta = opt.two_site_variational_theta(1, current_theta)

    opt.solve(tol=0.0, max_sweeps=2, sweep_sequence="RL")

    assert sum(block.size for block in current_theta.data.blocks.values()) == 1
    assert sum(block.size for block in variational_theta.data.blocks.values()) > 1
    assert opt.energy == pytest.approx(ed_energy, abs=1e-10)
    assert opt.state.max_bond() > 1
    assert len(opt.sector_enrichment_diagnostics) == 0
    assert len(opt.variational_sector_diagnostics) >= 1
    assert len(opt.mixer_diagnostics) == 0
    basis = opt.variational_sector_diagnostics[0]
    assert basis["mode"] == "variational_basis"
    assert basis["map_source"] == "prefix_closure"
    assert basis["noise"] == 0.0
    assert basis["added_blocks"] > 0
    assert basis["modified_tensors"] > 0
    assert [
        basis["bonds"][opt.state.bond(site, site + 1)]["num_sectors"]
        for site in range(3)
    ] == [4, 9, 4]
    assert opt.summary()["last_variational_sector_diagnostic"] == (
        opt.last_variational_sector_diagnostic
    )
    assert any(
        diagnostic["left"]["num_sectors"] > 1
        for diagnostic in opt.svd_diagnostics
    )


def test_symdmrg2_sector_enrichment_reaches_ed_from_narrow_initial_support():
    """Template sector enrichment lets a narrow initial MPS reach FH ED."""
    pytest.importorskip("symmray")
    edges = [(site, site + 1) for site in range(3)]
    occupations = [(1, 0), (0, 1), (1, 0), (0, 1)]
    state, mpo = _fh_u1u1_chain(
        4,
        occupations,
        bond_dim=2,
        seed=31,
        U=1.0,
        mu=0.1,
    )
    ed_energy = _fixed_u1u1_sector_ground_energy(
        4,
        edges,
        (2, 2),
        t=1.0,
        U=1.0,
        mu=0.1,
    )
    initial_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy.real

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=16,
        cutoff=1e-12,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-11,
        local_eig_ncv=16,
        norm_check_samples=3,
        sector_enrichment="template",
        sector_enrichment_bond_dim=12,
        sector_noise=1e-8,
        sector_enrichment_seed=123,
    )
    opt.solve(tol=0.0, max_sweeps=2, sweep_sequence="RL")
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    enrichment = opt.sector_enrichment_diagnostics[0]

    assert opt.energy == pytest.approx(ed_energy, abs=1e-10)
    assert opt.energy <= initial_energy
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert opt.state.max_bond() <= 16
    assert len(opt.sector_enrichment_diagnostics) == 1
    assert enrichment["mode"] == "template"
    assert enrichment["bond_dim"] == 12
    assert enrichment["noise"] == pytest.approx(1e-8)
    assert enrichment["added_blocks"] > 0
    assert opt.summary()["last_sector_enrichment_diagnostic"] == enrichment
    assert len(opt.norm_identity_diagnostics) == 2 * (4 - 1)
    assert all(diag["passed"] for diag in opt.norm_identity_diagnostics)
    assert all(diag["solver"] == "lanczos" for diag in opt.local_solve_diagnostics)


def test_symdmrg2_subspace_mixer_reaches_ed_from_narrow_initial_support():
    """The opt-in mixer should use H-aware directions without random noise."""
    pytest.importorskip("symmray")
    edges = [(site, site + 1) for site in range(3)]
    occupations = [(1, 0), (0, 1), (1, 0), (0, 1)]
    state, mpo = _fh_u1u1_chain(
        4,
        occupations,
        bond_dim=2,
        seed=31,
        U=1.0,
        mu=0.1,
    )
    ed_energy = _fixed_u1u1_sector_ground_energy(
        4,
        edges,
        (2, 2),
        t=1.0,
        U=1.0,
        mu=0.1,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=16,
        cutoff=1e-12,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-11,
        local_eig_ncv=16,
        norm_check_samples=3,
        mixer="density_matrix",
        mixer_bond_dim=12,
        mixer_amplitude=1e-8,
        mixer_decay=0.5,
        mixer_disable_after=1,
    )
    opt.solve(tol=0.0, max_sweeps=2, sweep_sequence="RL")

    expansion = opt.mixer_diagnostics[0]
    local_mixer = [
        diagnostic
        for diagnostic in opt.mixer_diagnostics
        if diagnostic["kind"] == "local_subspace"
    ]
    summary = opt.summary()

    assert opt.energy == pytest.approx(ed_energy, abs=1e-10)
    assert len(opt.sector_enrichment_diagnostics) == 0
    assert summary["mixer"] == "subspace_expansion"
    assert summary["mixer_bond_dim"] == 12
    assert summary["num_mixer_diagnostics"] == len(opt.mixer_diagnostics)
    assert summary["last_mixer_diagnostic"] == opt.last_mixer_diagnostic
    assert summary["active_mixer_amplitude"] == 0.0
    assert expansion["kind"] == "sector_expansion"
    assert expansion["mode"] == "subspace_expansion"
    assert expansion["noise"] == 0.0
    assert expansion["amplitude"] == pytest.approx(1e-8)
    assert expansion["added_blocks"] > 0
    assert len(local_mixer) == 3
    assert all(diagnostic["applied"] for diagnostic in local_mixer)
    assert all(diagnostic["injected_norm"] > 0.0 for diagnostic in local_mixer)
    assert {diagnostic["sweep"] for diagnostic in local_mixer} == {0}


def test_symdmrg2_adaptive_sector_enrichment_runs_each_sweep():
    """Adaptive template enrichment should re-run before every sweep."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        3,
        [(1, 0), (0, 1), (1, 0)],
        bond_dim=2,
        seed=37,
        U=1.0,
        mu=0.1,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=6,
        cutoff=1e-10,
        sweeps=2,
        sector_enrichment="adaptive",
        sector_enrichment_bond_dim=5,
        sector_noise=0.0,
        sector_enrichment_seed=456,
    )
    opt.solve(tol=0.0)

    assert opt.summary()["sector_enrichment"] == "adaptive_template"
    assert len(opt.energies) == 2
    assert len(opt.sector_enrichment_diagnostics) == 2
    assert [diag["sweep"] for diag in opt.sector_enrichment_diagnostics] == [0, 1]
    assert {
        diag["mode"] for diag in opt.sector_enrichment_diagnostics
    } == {"adaptive_template"}


def test_symdmrg2_adaptive_enrichment_forces_canonize_on_alternating_sweep(monkeypatch):
    """A pre-sweep sector expansion invalidates the alternating-sweep center."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        3,
        [(1, 0), (0, 1), (1, 0)],
        bond_dim=2,
        seed=38,
        U=1.0,
        mu=0.1,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=6,
        cutoff=1e-10,
        sweeps=2,
        sector_enrichment="adaptive",
        sector_enrichment_bond_dim=5,
        sector_noise=0.0,
    )
    canonize_flags = []
    enrichment_sweeps = []

    def fake_enrich_sectors(*, bond_dim, noise, mode, sweep):
        enrichment_sweeps.append(sweep)
        return {"bond_dim": bond_dim, "noise": noise, "mode": mode, "sweep": sweep}

    def fake_sweep(direction, canonize=True, **kwargs):
        canonize_flags.append((direction, canonize))
        return -1.0 - len(canonize_flags)

    monkeypatch.setattr(opt, "enrich_sectors", fake_enrich_sectors)
    monkeypatch.setattr(opt, "sweep", fake_sweep)
    opt.solve(max_sweeps=2, sweep_sequence="RL", tol=0.0)

    assert enrichment_sweeps == [0, 1]
    assert canonize_flags == [("R", True), ("L", True)]


def test_symdmrg2_lanczos_stress_obc_six_site_chain_tracks_svd_sectors():
    """A longer forced-Lanczos OBC sweep should keep auditable SVD sectors."""
    pytest.importorskip("symmray")
    state, mpo = _fh_u1u1_chain(
        6,
        [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)],
        bond_dim=2,
        seed=23,
        U=1.5,
        mu=0.05,
    )

    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=4,
        cutoff=1e-10,
        sweeps=1,
        local_solver="lanczos",
        dense_threshold=0,
        local_eig_tol=1e-10,
        local_eig_ncv=8,
    )
    out = opt.solve(max_sweeps=2, sweep_sequence="RL")
    post_energy = pepsy.MpsEnergyOptimizer(
        opt.state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert out is opt
    assert opt.state.max_bond() <= 4
    assert len(opt.energies) == 2
    assert complex(post_energy) == pytest.approx(complex(opt.energy))
    assert len(opt.svd_diagnostics) == 2 * (6 - 1)
    assert opt.summary()["last_svd_diagnostic"] == opt.svd_diagnostics[-1]
    for diagnostic in opt.svd_diagnostics:
        assert diagnostic["left"]["bond_dim"] <= 4
        assert diagnostic["right"]["bond_dim"] <= 4
        assert diagnostic["left"]["num_sectors"] >= 1
        assert diagnostic["right"]["num_sectors"] >= 1


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

"""Independent references and genuinely lossy protected dynamics checks."""

import numpy as np
import pytest
from scipy.linalg import expm

from examples import dmt_ising_dynamics as bench


def test_bond_fields_and_statevector_reference_match_dense_hamiltonian():
    n, hx, hz = 4, .9045, .809
    terms = bench.bond_hamiltonians(n, hx, hz)
    embedded = sum(np.kron(np.kron(np.eye(2**j), h), np.eye(2 ** (n - j - 2)))
                   for j, h in enumerate(terms))
    np.testing.assert_allclose(bench.sparse_hamiltonian(n, hx, hz).toarray(), embedded)
    psi = bench.product_vector(bench.initial_vectors(n))
    np.testing.assert_allclose(bench.measure_vector(psi, n, hx, hz)["energy"],
                               np.vdot(psi, embedded @ psi), atol=1e-13)
    dt = .02
    errors = []
    for step in (dt, dt / 2):
        actual = psi.copy()
        for gate, sites in bench.sweep_gates(terms, step):
            actual = bench.apply_vector(actual, gate, sites, n)
        errors.append(np.linalg.norm(actual - expm(-1j * step * embedded) @ psi))
    assert 7.5 < errors[0] / errors[1] < 8.5


def test_figure_state_matches_published_initial_spin_and_energy():
    import quimb.tensor as qtn
    vectors = bench.initial_vectors(64, "near-y-figure")
    measured = bench.measure_mps(qtn.MPS_product_state(vectors), .9045, .809)
    expected = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * .099009900990099
    np.testing.assert_allclose(measured["xyz"][:8, 2] / 2, expected, atol=2e-14)
    np.testing.assert_allclose(measured["energy"] / 4, .3038917753161455, atol=2e-13)


def test_figure8_gate_reference_matches_digitized_published_curve():
    import json
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    root = Path(__file__).resolve().parents[1]
    paper = json.loads((root / "examples/data/white2018_figures.json").read_text())
    expected = paper["fig8"]["16"]["mps-chi256"]["values"][:8]
    psi = bench.product_vector(bench.initial_vectors(16, "near-y-figure"))
    gates = bench.sweep_gates(bench.bond_hamiltonians(16, .9045, .809), .5,
                              merge_turnaround=False)
    actual = []
    with threadpool_limits(limits=1):
        for _ in range(8):
            measured = bench.measure_vector(psi, 16, .9045, .809)
            actual.append(np.real(measured["energy_fourier"] * np.exp(-1j * np.pi / 4)) / 4)
            for gate, where in gates:
                psi = bench.apply_vector(psi, gate, where, 16)
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=0.)


@pytest.mark.parametrize("method", ["mps-svd", "mps-direct", "mpo-dmt", "pauli-dmt", "pauli-dmt-real"])
def test_untruncated_complex_density_matches_independent_trotter(method):
    n, hx, hz = 4, .9045, .809
    vectors = bench.initial_vectors(n)
    gates = bench.sweep_gates(bench.bond_hamiltonians(n, hx, hz), .11)
    psi = bench.product_vector(vectors)
    for _ in range(2):
        for gate, sites in gates:
            psi = bench.apply_vector(psi, gate, sites, n)
    expected = bench.measure_vector(psi, n, hx, hz)
    rows, timing = bench.evolve(method, vectors, gates, 16, hx, hz, 2, {0, 1, 2}, {1, 2})
    np.testing.assert_allclose(rows[-1]["triple_rdm"], expected["triple_rdm"], atol=3e-12)
    assert timing["final_max_bond"] <= 16
    assert timing["audit"]["max_triple_defect"] < 3e-12


@pytest.mark.parametrize("method", ["mps-direct", "mpo-dmt", "pauli-dmt"])
def test_torch_complex128_reference_and_precision(method):
    pytest.importorskip("torch")
    n, hx, hz = 4, .9045, .809
    vectors = bench.initial_vectors(n)
    gates = bench.sweep_gates(bench.bond_hamiltonians(n, hx, hz), .11)
    expected, _ = bench.evolve(method, vectors, gates, 16, hx, hz, 2, {0, 2}, {1, 2})
    actual, info = bench.evolve(method, vectors, gates, 16, hx, hz, 2, {0, 2}, {1, 2}, backend="torch")
    assert info["dtype"] == "complex128"
    assert info["gate_dtype"] == "torch.complex128"
    np.testing.assert_allclose(actual[-1]["triple_rdm"], expected[-1]["triple_rdm"], atol=3e-12)


def test_mps_transfer_readout_matches_dense_with_nontrivial_gauge_and_scale():
    import quimb.tensor as qtn
    mps = qtn.MPS_rand_state(7, bond_dim=5, dtype="complex128", seed=17)
    mps.exponent = .3
    expected = bench.measure_vector(np.asarray(mps.to_dense()).reshape(-1), 7, .9045, .809)
    actual = bench.measure_mps(mps, .9045, .809)
    for key in ("triple_rdm", "xyz", "energy", "trace"):
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-13)


def test_global_negativity_can_precede_local_negativity_without_breaking_protection():
    from examples.dmt_positivity_diagnostic import diagnose
    result = diagnose(n=6, chi=16, steps=5)
    first = result["first_global_loss"]
    assert first["pre_gate_global_min_eigenvalue"] > -1e-10
    assert first["post_compression_global_min_eigenvalue"] < -1e-8
    assert first["post_compression_min_triple_eigenvalue"] > -1e-10
    later = result["first_local_loss"]
    assert later["pre_gate_union_min_eigenvalue"] < -1e-8
    assert later["gated_target_min_triple_eigenvalue"] < -1e-9
    assert later["max_triple_preservation_defect"] < 1e-12


def test_torch_benchmark_starts_without_prior_backend_registration(tmp_path):
    pytest.importorskip("torch")
    import json
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "examples/dmt_ising_dynamics.py"),
                    "--backend", "torch", "--length", "4", "--models", "tilted",
                    "--chis", "16", "--methods", "mps-direct", "--dt", ".11",
                    "--time", ".22", "--sample-every", "1", "--output", str(tmp_path)],
                   cwd=root, capture_output=True, text=True, check=True, timeout=60)
    row = json.loads((tmp_path / "summary.json").read_text())[-1]
    assert row["dtype"] == "complex128"
    assert row["gate_dtype"] == "torch.complex128"
    assert row["triple_rdm_rmse_vs_trotter"] < 2e-12


@pytest.mark.parametrize("method", ["mpo-dmt", "pauli-dmt", "pauli-dmt-real"])
def test_lossy_dmt_preserves_each_audited_target_and_is_observational(method):
    n, hx, hz = 6, .9045, .809
    vectors = bench.initial_vectors(n)
    gates = bench.sweep_gates(bench.bond_hamiltonians(n, hx, hz), .19)
    checked, timing = bench.evolve(method, vectors, gates, 8, hx, hz, 4, {0, 4}, {1, 2, 3, 4})
    unchecked, _ = bench.evolve(method, vectors, gates, 8, hx, hz, 4, {0, 4}, set())
    assert timing["lossy_splits"] > 0
    assert timing["audit"]["checked_gates"] == 12
    assert timing["audit"]["max_triple_defect"] < 2e-10
    assert timing["audit"]["max_energy_jump"] < 2e-10
    assert abs(checked[-1]["trace"] - 1) < 2e-10
    np.testing.assert_allclose(checked[-1]["triple_rdm"], unchecked[-1]["triple_rdm"], atol=1e-12)


def test_numpy_norm_tracking_never_requests_stop_gradient(monkeypatch):
    import autoray as ar
    import quimb.tensor as qtn
    from pepsy.optimizers import MpsOptimizer

    original = ar.do

    def checked(name, *args, **kwargs):
        if name == "stop_gradient" and ar.infer_backend(args[0]) in {"numpy", "builtins"}:
            raise AssertionError("NumPy has no gradient tape to detach")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ar, "do", checked)
    gate = expm(-.2j * np.kron(bench.X, bench.X))
    opt = MpsOptimizer(qtn.MPS_computational_state("00", dtype="complex128"),
                       [(gate, (0, 1))], chi=1, mode="svd")
    opt.run(cutoff=0., stabilize_unitary=True)
    np.testing.assert_allclose(opt.p.norm(), 1., atol=1e-13)
    assert opt.norm_diagnostics()["cumulative_infidelity"] > 0
    filtered = MpsOptimizer(qtn.MPS_computational_state("00", dtype="complex128"),
                            [(expm(.2 * np.kron(bench.X, bench.X)), (0, 1))], chi=1,
                            mode="direct")
    filtered.run(cutoff=0., non_unitary=True)
    assert filtered.get_nonunitary_diagnostics()["error"] is None

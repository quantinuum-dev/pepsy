"""Independent Hamiltonian/reference and audit checks for the 2D benchmark."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples import dmt_2d_dynamics as bench
from examples import dmt_ising_dynamics as ref


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("mode", ["snake", "snake-row-major", "row-major"])
def test_edge_hamiltonians_sum_to_independent_full_2d_model(mode):
    model = bench.problem(3, 2, .1, map_mode=mode)
    assert len(model["edges"]) == 7
    assert any(abs(a - b) > 1 for a, b in model["edges"])
    eye = np.eye(64)
    actual = sum(np.column_stack([ref.apply_vector(v, h, sites, 6) for v in eye.T])
                 for h, sites in zip(model["terms"], model["edges"]))
    np.testing.assert_allclose(actual, model["hamiltonian"], atol=2e-15)
    np.testing.assert_allclose(model["initial"].to_dense(), bench.density(model["psi"], .6), atol=1e-15)


def test_symmetric_2d_schedule_has_second_order_error_at_fixed_time():
    errors = []
    for dt, steps in [(.1, 4), (.05, 8)]:
        model = bench.problem(3, 2, dt)
        exact, trotter = bench.exact_references(model, dt, steps, steps)[steps]
        errors.append(np.linalg.norm(exact - trotter))
        assert abs(np.trace(trotter) - 1) < 2e-14
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_mixed_density_gate_oracle_and_reference_are_independent_and_positive():
    from scipy.linalg import expm

    model = bench.problem(2, 2, .1, initial_kind="product-mixed")
    source = model["initial_density"]
    gate, where = model["gates"][0]
    unitary = np.column_stack([ref.apply_vector(v, gate, where, 4) for v in np.eye(16).T])
    np.testing.assert_allclose(bench.apply_density(source, gate, where, 4),
                               unitary @ source @ unitary.conj().T, atol=2e-15)
    exact, trotter = bench.exact_references(model, .1, 2, 1)[2]
    evolution = expm(-.2j * model["hamiltonian"])
    np.testing.assert_allclose(exact, evolution @ source @ evolution.conj().T, atol=2e-15)
    for state in (exact, trotter):
        assert abs(np.trace(state) - 1) < 1e-14
        np.testing.assert_allclose(np.linalg.eigvalsh(state), np.linalg.eigvalsh(source), atol=2e-15)


@pytest.mark.parametrize("method", ["dmt", "fit"])
def test_untruncated_2d_replay_and_each_gate_audit_match_trotter(method):
    model = bench.problem(3, 2, .1)
    references = bench.exact_references(model, .1, 2, 1)
    row = bench.trajectory(model, references, method=method, chi=64, steps=2, dt=.1)
    assert row["status"] == "completed"
    assert row["audited_gates"] == 26
    # This catches auditing FIT against self.p, which is not its live per-gate
    # source. An exact fit would then appear to violate protected marginals.
    assert row["max_per_gate_preservation_error"] < 3e-12
    assert max(s["hs_relative_trotter"] for s in row["samples"]) < 3e-12
    assert row["max_bond_observed"] <= 64
    assert row["storage_backends"] == ["numpy"]
    assert row["storage_dtypes"] == ["complex128"]


def test_lossy_dmt_preserves_targets_without_claiming_exact_dynamics():
    model = bench.problem(3, 2, .1)
    references = bench.exact_references(model, .1, 2, 1)
    row = bench.trajectory(model, references, method="dmt", chi=32, steps=2, dt=.1)
    assert row["status"] == "completed"
    assert row["max_bond_observed"] <= 32
    assert row["max_per_gate_preservation_error"] < 3e-12
    assert row["samples"][-1]["hs_relative_trotter"] > 1e-3
    assert row["samples"][-1]["max_plaquette_error"] > 1e-4


def test_budget_rejection_is_retained_as_a_failed_row():
    model = bench.problem(3, 2, .1)
    references = bench.exact_references(model, .1, 1, 1)
    row = bench.trajectory(model, references, method="dmt", chi=20, steps=1, dt=.1)
    assert row["status"] == "failed" and row["completed_steps"] == 0
    assert "protected budget" in row["error"]


def test_raw_trace_and_nonhermiticity_are_not_hidden_by_normalization():
    model = bench.problem(2, 2, .1)
    exact = bench.density(model["psi"], .6)
    corrupted = 1.1 * exact
    corrupted[0, 1] += .01j
    metrics = bench.diagnostics(corrupted, exact, exact, model)
    assert metrics["trace_error"] == pytest.approx(.1)
    assert metrics["relative_hermiticity_defect"] > 1e-3
    assert metrics["hs_relative_exact"] > .09
    np.testing.assert_array_equal(corrupted[0, 1], 1.1 * exact[0, 1] + .01j)
    scaled = bench.diagnostics(1.1 * exact, exact, exact, model)
    assert scaled["trace_error"] == pytest.approx(.1)
    assert scaled["energy_error_per_site_trace_normalized"] < 1e-14


@pytest.mark.parametrize("method", ["dmt", "fit"])
def test_torch_complex128_2d_replay_keeps_backend_and_exact_reference(method):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = bench.problem(2, 2, .1)
    references = bench.exact_references(model, .1, 1, 1)
    row = bench.trajectory(model, references, method=method, chi=16, steps=1, dt=.1, backend="torch")
    assert row["status"] == "completed", row.get("error")
    assert row["storage_backends"] == ["torch"]
    assert row["storage_dtypes"] == ["complex128"]
    assert row["samples"][-1]["hs_relative_trotter"] < 3e-12
    assert row["max_per_gate_preservation_error"] < 3e-12

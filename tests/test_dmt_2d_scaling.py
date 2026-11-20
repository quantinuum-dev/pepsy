"""Sparse references and exact target construction for larger 2D MPOs."""

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.linalg import expm
from scipy.sparse import issparse
from threadpoolctl import threadpool_limits

from examples import dmt_2d_dynamics as dense
from examples import dmt_2d_scaling as bench


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def test_sparse_hamiltonian_matches_dense_model():
    small = dense.problem(3, 2, .1)
    large = bench.problem(3, 2, .1)
    np.testing.assert_array_equal(large["hamiltonian"].toarray(), small["hamiltonian"])
    assert large["initial_density"] is None


def test_4x4_build_does_not_construct_a_global_density(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("constructed a full density matrix")
    monkeypatch.setattr(dense, "density", forbidden)
    model = bench.problem(4, 4, .1)
    assert model["initial_density"] is None
    assert issparse(model["hamiltonian"])
    assert model["hamiltonian"].shape == (65536, 65536)
    assert model["hamiltonian"].nnz <= 17 * 65536
    assert len(model["edges"]) == 24 and len(model["regions"]) == 9


@pytest.mark.parametrize("where", [(0, 1), (1, 0), (0, 3), (3, 1)])
def test_analytic_zz_gate_matches_bit_order_oracle(where):
    operator = bench.zz_operator(4, where, .13)
    gate = expm(-.13j * np.kron(bench.ref.Z, bench.ref.Z))
    expected = np.column_stack([bench.ref.apply_vector(v, gate, where, 4) for v in np.eye(16).T])
    np.testing.assert_allclose(operator.to_dense(), expected, atol=1e-15)
    assert not operator.cyclic and operator.max_bond() == 2


def test_exact_field_and_diagonal_layer_match_explicit_dense_gates():
    model = bench.problem(2, 2, .1)
    psi = model["psi"].copy()
    for site in range(4):
        psi = bench.ref.apply_vector(psi, model["field_gate"], (site,), 4)
    for gate, where in model["gates"]:
        psi = bench.ref.apply_vector(psi, gate, where, 4)
    for site in range(4):
        psi = bench.ref.apply_vector(psi, model["field_gate"], (site,), 4)
    np.testing.assert_allclose(bench.vector_step(model["psi"], model), psi, atol=2e-15)
    # A nonsymmetric complex unitary also checks upper/lower-index conventions.
    unitary = expm(-.2j * (bench.ref.X + .7 * bench.ref.Y))
    actual = model["initial"].copy()
    target = actual.to_dense()
    bonds = actual.bond_sizes()
    bench.field_layer(actual, unitary)
    for site in range(4):
        target = dense.apply_density(target, unitary, (site,), 4)
    np.testing.assert_allclose(actual.to_dense(), target, atol=2e-15)
    assert actual.bond_sizes() == bonds


def test_sparse_local_references_and_cache_match_full_exponential(tmp_path):
    model = bench.problem(3, 2, .1)
    path = tmp_path / "references.npz"
    samples, preparation = bench.references(model, .1, 2, 1, cache=path)
    assert not preparation["cache_hit"]
    psi = expm(-.2j * model["hamiltonian"].toarray()) @ model["psi"]
    expected = bench.local_reference(psi, model)
    for name in expected:
        np.testing.assert_allclose(samples[2]["exact"][name], expected[name], atol=2e-14)
    loaded, preparation = bench.references(model, .1, 2, 1, cache=path)
    assert preparation["cache_hit"]
    np.testing.assert_array_equal(loaded[2]["exact"]["plaquettes"], samples[2]["exact"]["plaquettes"])
    with pytest.raises(ValueError, match="configuration"):
        bench.references(model, .05, 2, 1, cache=path)


def test_global_norm_and_hermiticity_match_dense_nonhermitian_operator():
    rng = np.random.default_rng(52)
    matrix = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    mpo = qtn.MatrixProductOperator.from_dense(matrix, dims=[2] * 4, cutoff=0.)
    result = bench.global_operator_norms(bench.ref.mpo_cores(mpo))
    assert result["hs_norm"] == pytest.approx(np.linalg.norm(matrix), rel=1e-13)
    assert result["global_relative_hermiticity"] == pytest.approx(
        np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix), rel=1e-13)
    assert result["global_min_eigenvalue"] is None


def test_field_zz_schedule_is_second_order():
    errors = []
    for dt, steps in [(.1, 4), (.05, 8)]:
        model = bench.problem(3, 2, dt)
        samples, _ = bench.references(model, dt, steps, steps)
        sample = samples[steps]
        errors.append(max(np.linalg.norm(a - b) for a, b in zip(
            sample["exact"]["plaquettes"], sample["circuit"]["plaquettes"])))
    assert 3.8 < errors[0] / errors[1] < 4.2


@pytest.mark.parametrize("method,update", [("dmt", "routed"), ("dmt", "full"),
                                           ("dmt", "analytic"), ("fit", "routed")])
def test_exact_cap_replay_all_update_routes_match_same_circuit(method, update):
    model = bench.problem(3, 2, .1)
    samples, _ = bench.references(model, .1, 2, 1)
    row = bench.trajectory(model, samples, chi=64, steps=2, dt=.1, method=method, update=update)
    assert row["status"] == "completed", row.get("error")
    assert row["audited_two_site_gates"] == 14
    assert row["max_per_gate_preservation_error"] < 3e-12
    assert row["samples"][-1]["max_plaquette_circuit_error"] < 3e-12
    assert row["max_bond_observed"] <= 64
    assert row["storage_dtypes"] == ["complex128"]


def test_lossy_analytic_sweeps_preserve_each_gated_target():
    model = bench.problem(3, 2, .1)
    samples, _ = bench.references(model, .1, 3, 1)
    row = bench.trajectory(model, samples, chi=32, steps=3, dt=.1, update="analytic")
    assert row["status"] == "completed", row.get("error")
    assert row["max_per_gate_preservation_error"] < 3e-12
    assert row["samples"][-1]["max_plaquette_circuit_error"] > 1e-6
    assert row["max_bond_observed"] <= 32

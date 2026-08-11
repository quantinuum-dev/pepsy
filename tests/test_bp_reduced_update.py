"""Focused tests for SU- and D2BP-closed reduced PEPS updates."""

from dataclasses import replace
import importlib

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy as py
from pepsy.bp import (
    apply_reduced_loop_cluster_gate,
    compress_reduced_loop_cluster,
    exact_reduced_update_problem,
    prepare_reduced_bond_pair,
    solve_reduced_als,
    su_cluster_reduced_update_problem,
    two_norm_bp,
)
from pepsy.operators import gate_loop_cluster


def _random_peps():
    return qtn.PEPS.rand(2, 2, bond_dim=2, seed=7, dtype="complex128")


def test_reduced_loop_cluster_accepts_directed_d2bp_messages():
    peps = _random_peps()
    gate = np.asarray(py.rzz(0.07))
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
    )
    problem = su_cluster_reduced_update_problem(pair, gate, radius=0)

    assert pair.boundary_messages
    assert problem.metric.shape == (np.prod(pair.theta_shape),) * 2
    assert np.allclose(
        problem.linear_term,
        problem.metric @ problem.target.reshape(-1),
    )


def test_reduced_loop_cluster_gate_can_start_from_d2bp_messages():
    peps = _random_peps()
    gate = np.asarray(py.rzz(0.07))
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = apply_reduced_loop_cluster_gate(
        peps,
        gauges=None,
        gate=gate,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        max_loop_size=0,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
        inplace=False,
    )

    assert result.solution.left.shape[-1] == 1
    assert result.solution.right.shape[0] == 1
    assert result.problem.max_loop_size == 0
    assert len(result.gauges) == len(tuple(result.core.inner_inds()))


def test_reduced_loop_cluster_message_only_call_accepts_gate_as_second_arg():
    peps = _random_peps()
    gate = np.asarray(py.rzz(0.07))
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = apply_reduced_loop_cluster_gate(
        peps,
        gate,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        max_loop_size=0,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
        inplace=False,
    )

    assert result.problem.max_loop_size == 0
    assert result.gauges


def test_gate_loop_cluster_accepts_d2bp_messages_without_auto_gauging():
    peps = _random_peps()
    gate = np.asarray(py.rzz(0.07))
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)
    gauges = {}

    out, results = gate_loop_cluster(
        peps,
        ((gate, ((0, 0), (1, 0))),),
        gauges=gauges,
        boundary_messages=bp.messages,
        max_bond=1,
        max_loop_size=0,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
        inplace=False,
        return_results=True,
    )

    assert out is not peps
    assert len(results) == 1
    assert len(gauges) == len(tuple(out.inner_inds()))
    assert results[0].pair.boundary_messages


def test_cluster_compression_truncates_one_bond_without_refreshing_gauges():
    peps = _random_peps()
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = compress_reduced_loop_cluster(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        max_distance=0,
        als_opts={"max_iterations": 2},
        inplace=False,
    )

    assert not result.regauged
    assert result.gauges == {}
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1
    assert result.problem.boundary_inds
    assert np.allclose(
        result.problem.target,
        result.pair.theta_array(),
    )


def test_cluster_compression_accepts_su_gauge_boundary_closures():
    peps = _random_peps()
    core, gauges, _ = py.gauge_all_simple(
        peps,
        max_iterations=2,
        tol=0.0,
    )

    result = compress_reduced_loop_cluster(
        core,
        where=((0, 0), (1, 0)),
        gauges=gauges,
        max_bond=1,
        max_distance=0,
        als_opts={"max_iterations": 2},
        inplace=False,
    )

    assert result.pair.boundary_messages is None
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1
    assert result.problem.boundary_inds


def test_cluster_compression_can_refresh_su_gauges_after_truncation():
    peps = _random_peps()
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = compress_reduced_loop_cluster(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        max_distance=1,
        regauge=True,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
        inplace=False,
    )

    assert result.regauged
    assert len(result.gauges) == len(tuple(result.core.inner_inds()))
    assert not result.problem.boundary_inds
    rebuilt = result.core.copy()
    rebuilt.gauge_simple_insert(result.gauges)
    assert np.allclose(rebuilt.to_dense(), result.physical_tn.to_dense())


def test_cluster_compression_can_add_open_leg_loop_clusters():
    peps = _random_peps()
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = compress_reduced_loop_cluster(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        max_distance=0,
        max_loop_size=4,
        als_opts={"max_iterations": 2},
        inplace=False,
    )

    assert result.problem.max_loop_size == 4
    assert result.problem.terms
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1


def test_real_d4_to_d2_full_cluster_matches_exact_reduced_als():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=4,
        phys_dim=2,
        dtype="float64",
        seed=19,
    )
    where = ((1, 1), (1, 2))
    reference_pair = prepare_reduced_bond_pair(peps, where=where)
    identity = np.eye(
        reference_pair.theta_shape[1] * reference_pair.theta_shape[2]
    )
    exact = exact_reduced_update_problem(reference_pair, identity)
    exact_solution = solve_reduced_als(
        exact,
        max_bond=2,
        max_iterations=8,
        rcond=1e-11,
    )
    expected = reference_pair.reconstruct_tn(
        exact_solution.left,
        exact_solution.right,
    )

    result = compress_reduced_loop_cluster(
        peps,
        where=where,
        max_bond=2,
        max_distance=reference_pair.full_cluster_radius(),
        als_opts={"max_iterations": 8, "rcond": 1e-11},
        inplace=False,
    )

    assert peps.ind_size(reference_pair.bond_ind) == 4
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 2
    assert not result.problem.boundary_inds
    assert np.allclose(result.problem.metric, exact.metric, atol=1e-8, rtol=1e-8)
    assert np.allclose(
        result.physical_tn.to_dense(),
        expected.to_dense(),
        atol=1e-7,
        rtol=1e-7,
    )


def test_real_4x4_d4_to_d2_distance_three_is_exact_su_environment():
    """The requested 4x4 D=4 audit reaches the exact finite environment."""
    peps = qtn.PEPS.rand(
        4,
        4,
        bond_dim=4,
        phys_dim=2,
        dtype="float64",
        seed=20260810,
    )
    core, gauges, _ = py.gauge_all_simple(
        peps,
        max_iterations=4,
        tol=0.0,
        progbar=False,
    )
    where = ((1, 1), (1, 2))
    pair = prepare_reduced_bond_pair(core, gauges, where=where)
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    exact = exact_reduced_update_problem(pair, identity)

    result = compress_reduced_loop_cluster(
        core,
        gauges=gauges,
        where=where,
        max_bond=2,
        max_distance=3,
        als_opts={"max_iterations": 2, "rcond": 1e-11},
        inplace=False,
    )

    assert pair.theta_shape == (8, 2, 2, 8)
    assert not result.problem.boundary_inds
    metric_error = np.linalg.norm(result.problem.metric - exact.metric)
    assert metric_error / np.linalg.norm(exact.metric) < 1e-12
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 2


def test_reduced_boundary_messages_are_hermitian_psd_projected():
    peps = _random_peps()
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)
    messages = bp.snapshot()
    key = next(iter(messages))
    messages[key] = np.array([[1.0, 2.0], [-0.5, -1.0]])

    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=messages,
    )

    for message in pair.boundary_messages.values():
        np.testing.assert_allclose(message, message.conj().T)
        assert np.linalg.eigvalsh(message).min() >= -1e-12


def test_reduced_als_qr_regularization_handles_nearly_singular_metric():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(peps, where=((0, 0), (1, 0)))
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    reference = exact_reduced_update_problem(pair, identity)
    dimension = reference.metric.shape[0]
    metric = np.diag(np.geomspace(1e-14, 1.0, dimension)).astype(complex)
    problem = replace(
        reference,
        metric=metric,
        linear_term=metric @ reference.target.reshape(-1),
    )

    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=4,
        rcond=1e-12,
        regularization=1e-10,
        solver="qr",
    )

    assert np.all(np.isfinite(solution.left))
    assert np.all(np.isfinite(solution.right))
    assert np.all(np.isfinite(solution.costs))
    assert np.isfinite(problem.cost(solution.theta()))


def test_reduced_problem_api_keeps_open_environment_and_explicit_dense_views():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(peps, where=((0, 0), (1, 0)))
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])

    lazy = exact_reduced_update_problem(pair, identity)
    eager = exact_reduced_update_problem(
        pair,
        identity,
        materialize_metric=True,
    )

    assert isinstance(lazy.environment, qtn.Tensor)
    assert lazy.environment.shape == (
        pair.theta_shape[0],
        pair.theta_shape[3],
        pair.theta_shape[0],
        pair.theta_shape[3],
    )
    np.testing.assert_allclose(lazy.dense_metric(), eager.metric)
    np.testing.assert_allclose(lazy.dense_linear_term(), eager.linear_term)
    assert lazy.target_norm == pytest.approx(eager.target_norm)


def test_reduced_als_can_use_quimbs_public_tensor_network_als():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(peps, where=((0, 0), (1, 0)))
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    problem = exact_reduced_update_problem(pair, identity, optimize="greedy")

    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=3,
        rcond=1e-12,
        solver="quimb",
        quimb_opts={"solver_maxiter": 8, "contract_optimize": "greedy"},
    )

    assert np.all(np.isfinite(solution.left))
    assert np.all(np.isfinite(solution.right))
    assert np.isfinite(problem.cost(solution.theta()))
    assert solution.costs[-1] <= solution.costs[0] + 1e-12


def test_quimb_reduced_als_uses_retained_open_environment(monkeypatch):
    """Native ALS must not reconstruct the full physical metric first."""
    reduced_update = importlib.import_module("pepsy.bp.reduced_update")
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(peps, where=((0, 0), (1, 0)))
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    problem = exact_reduced_update_problem(pair, identity, optimize="greedy")

    assert isinstance(problem.environment, qtn.Tensor)

    def fail_dense_metric(*args, **kwargs):
        raise AssertionError("native Quimb ALS should not materialize N_red")

    monkeypatch.setattr(reduced_update, "_metric_from_environment", fail_dense_metric)
    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=2,
        solver="quimb",
        quimb_opts={"solver_maxiter": 8, "contract_optimize": "greedy"},
    )

    assert np.all(np.isfinite(solution.left))
    assert np.all(np.isfinite(solution.right))


def test_reduced_als_qr_requires_a_psd_metric():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(peps, where=((0, 0), (1, 0)))
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    reference = exact_reduced_update_problem(pair, identity)
    metric = np.eye(reference.metric.shape[0], dtype=complex)
    metric[0, 0] = -1.0
    problem = replace(
        reference,
        metric=metric,
        linear_term=metric @ reference.target.reshape(-1),
    )

    with pytest.raises(np.linalg.LinAlgError, match="positive-semidefinite"):
        solve_reduced_als(problem, max_bond=1, max_iterations=1, solver="qr")

    fallback = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=1,
        regularization=1e-8,
        solver="auto",
    )
    assert np.all(np.isfinite(fallback.left))
    assert np.all(np.isfinite(fallback.right))


def test_pepo_cluster_compression_restores_operator_legs_and_su_gauges():
    pepo = qtn.PEPO.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=23,
    )
    bp = two_norm_bp(pepo, max_iterations=2, tol=0.0, diis=False)
    where = ((0, 0), (1, 0))

    result = compress_reduced_loop_cluster(
        pepo,
        where=where,
        boundary_messages=bp.messages,
        max_bond=1,
        max_distance=0,
        als_opts={"max_iterations": 2},
    )

    assert isinstance(result.physical_tn, qtn.PEPO)
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1
    for site in where:
        tensor = result.physical_tn.tensor_map[
            result.pair.left_tid if site == where[0] else result.pair.right_tid
        ]
        assert pepo.lower_ind(site) in tensor.inds
        assert pepo.upper_ind(site) in tensor.inds
    assert result.pair.physical_left_ind not in result.physical_tn.tensor_map[
        result.pair.left_tid
    ].inds
    assert np.linalg.eigvalsh(result.problem.metric).min() >= -1e-10

    regauged = compress_reduced_loop_cluster(
        pepo,
        where=where,
        boundary_messages=bp.messages,
        max_bond=1,
        max_distance=0,
        regauge=True,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
    )
    rebuilt = regauged.core.copy()
    rebuilt.gauge_simple_insert(regauged.gauges)
    np.testing.assert_allclose(
        rebuilt.to_dense(),
        regauged.physical_tn.to_dense(),
        rtol=1e-10,
        atol=1e-10,
    )

    su_core, su_gauges, _ = py.gauge_all_simple(
        pepo,
        max_iterations=2,
        tol=0.0,
    )
    su_result = compress_reduced_loop_cluster(
        su_core,
        gauges=su_gauges,
        where=where,
        max_bond=1,
        max_distance=0,
        als_opts={"max_iterations": 2},
    )
    assert isinstance(su_result.physical_tn, qtn.PEPO)
    assert su_result.pair.boundary_messages is None

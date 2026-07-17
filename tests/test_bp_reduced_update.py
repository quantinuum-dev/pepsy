"""Exact-oracle tests for the SU-gauged reduced PEPS update."""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from pepsy import build_optimizer  # noqa: E402
from pepsy.bp import (  # noqa: E402
    apply_reduced_loop_cluster_gate,
    exact_reduced_update_problem,
    gauge_all_simple,
    loop_cluster_reduced_update_problem,
    prepare_reduced_bond_pair,
    solve_reduced_als,
    su_cluster_reduced_update_problem,
)
from pepsy.operators import gate_loop_cluster  # noqa: E402


def _su_gauged_peps(rows=2, cols=2, seed=23):
    peps = qtn.PEPS.rand(
        rows,
        cols,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=seed,
    )
    core, gauges, _ = gauge_all_simple(
        peps,
        max_iterations=30,
        tol=1e-12,
    )
    return core, gauges


def _cnot():
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def _pair():
    core, gauges = _su_gauged_peps()
    return prepare_reduced_bond_pair(core, gauges, where=((0, 0), (0, 1)))


def _physical_tn(core, gauges):
    out = core.copy()
    out.gauge_simple_insert(gauges)
    return out


def _relative_error(actual, expected):
    return abs(actual - expected) / max(1.0, abs(expected))


def test_reduced_pair_reconstructs_the_su_gauged_state_exactly():
    pair = _pair()
    rebuilt = pair.reconstruct_tn()

    assert np.allclose(
        rebuilt.to_dense(),
        pair.tn.to_dense(),
        atol=1e-10,
        rtol=1e-10,
    )


def test_reduced_pair_reconstructs_a_truncated_active_bond():
    pair = _pair()
    problem = exact_reduced_update_problem(pair, _cnot())
    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=12,
        rcond=1e-11,
    )
    rebuilt = pair.reconstruct_tn(solution.left, solution.right)

    assert rebuilt.ind_size(pair.bond_ind) == 1
    assert np.all(np.isfinite(rebuilt.to_dense()))


def test_exact_reduced_problem_is_hermitian_psd_and_has_the_full_norm():
    pair = _pair()
    problem = exact_reduced_update_problem(pair, _cnot())
    theta = pair.theta_array().reshape(-1)
    state = np.asarray(pair.tn.to_dense())

    assert np.allclose(problem.metric, problem.metric.conj().T, atol=1e-10)
    assert np.linalg.eigvalsh(problem.metric).min() >= -1e-10
    assert np.allclose(
        np.vdot(theta, problem.metric @ theta),
        np.vdot(state, state),
        atol=1e-10,
    )
    assert np.allclose(
        problem.linear_term,
        problem.metric @ problem.target.reshape(-1),
        atol=1e-10,
    )


def test_exact_reduced_als_decreases_the_true_gate_projection_error():
    pair = _pair()
    problem = exact_reduced_update_problem(pair, _cnot())
    solution = solve_reduced_als(problem, max_iterations=12, rcond=1e-11)
    candidate = pair.reconstruct_tn(solution.left, solution.right)
    target = pair.gate_target_tn(_cnot())
    direct_error = np.vdot(
        np.asarray(candidate.to_dense() - target.to_dense()),
        np.asarray(candidate.to_dense() - target.to_dense()),
    ).real

    assert all(
        later <= earlier + 1e-9 * max(1.0, earlier)
        for earlier, later in zip(solution.costs, solution.costs[1:])
    )
    assert solution.costs[-1] < solution.costs[0]
    assert np.allclose(direct_error, solution.costs[-1], atol=1e-9)


def test_su_cluster_full_radius_recovers_the_exact_open_leg_metric():
    pair = _pair()
    exact = exact_reduced_update_problem(pair, _cnot())
    cluster = su_cluster_reduced_update_problem(
        pair,
        _cnot(),
        radius=pair.full_cluster_radius(),
    )

    assert cluster.radius == cluster.full_radius
    assert not cluster.boundary_inds
    assert set(cluster.cluster_tids) == {
        tid
        for tid in pair.tn.tensor_map
        if tid not in {pair.left_tid, pair.right_tid}
    }
    assert np.allclose(cluster.metric, exact.metric, atol=1e-10, rtol=1e-10)
    assert np.allclose(
        cluster.linear_term,
        exact.linear_term,
        atol=1e-10,
        rtol=1e-10,
    )


def test_radius_zero_su_cluster_is_psd_and_has_a_consistent_als_objective():
    pair = _pair()
    cluster = su_cluster_reduced_update_problem(pair, _cnot(), radius=0)
    solution = solve_reduced_als(cluster, max_iterations=12, rcond=1e-11)

    assert not cluster.cluster_tids
    assert cluster.boundary_inds
    assert np.allclose(cluster.metric, cluster.metric.conj().T, atol=1e-10)
    assert np.linalg.eigvalsh(cluster.metric).min() >= -1e-10
    assert np.allclose(
        cluster.linear_term,
        cluster.metric @ cluster.target.reshape(-1),
        atol=1e-10,
    )
    assert all(
        later <= earlier + 1e-9 * max(1.0, earlier)
        for earlier, later in zip(solution.costs, solution.costs[1:])
    )


def test_zero_loop_open_leg_cluster_matches_the_su_boundary_cluster():
    pair = _pair()
    su_cluster = su_cluster_reduced_update_problem(pair, _cnot(), radius=0)
    loop_cluster = loop_cluster_reduced_update_problem(
        pair,
        _cnot(),
        max_loop_size=0,
        base_radius=0,
        psd_project=False,
    )

    assert not loop_cluster.loop_regions
    assert len(loop_cluster.terms) == 1
    assert loop_cluster.terms[0].count == 1
    assert loop_cluster.terms[0].cluster_tids == su_cluster.cluster_tids
    assert loop_cluster.terms[0].boundary_inds == su_cluster.boundary_inds
    assert np.allclose(
        loop_cluster.metric,
        su_cluster.metric,
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        loop_cluster.linear_term,
        su_cluster.linear_term,
        atol=1e-10,
        rtol=1e-10,
    )


def test_full_open_leg_loop_cluster_recovers_the_exact_metric():
    pair = _pair()
    exact = exact_reduced_update_problem(pair, _cnot())
    loop_cluster = loop_cluster_reduced_update_problem(
        pair,
        _cnot(),
        max_loop_size=len(pair.tn.tensor_map),
        psd_project=False,
    )

    assert len(loop_cluster.terms) == 1
    assert loop_cluster.terms[0].region_tids == frozenset(pair.tn.tensor_map)
    assert not loop_cluster.terms[0].boundary_inds
    assert np.allclose(
        loop_cluster.raw_metric,
        exact.metric,
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        loop_cluster.linear_term,
        exact.linear_term,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("bond_dim", [2, 3])
def test_four_by_four_full_gloop_matches_exact_nred_and_bred(bond_dim):
    """System-covering gloops should reproduce exact dense 4x4 N_red/b_red."""
    peps = qtn.PEPS.rand(
        4,
        4,
        bond_dim=bond_dim,
        phys_dim=2,
        dtype="complex128",
        seed=80 + bond_dim,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        {},
        where=((1, 1), (1, 2)),
    )
    optimize = build_optimizer(
        progbar=False,
        max_time=1.0,
        max_repeats=64,
    )
    gate = _cnot()
    exact = exact_reduced_update_problem(pair, gate, optimize=optimize)
    gloop = loop_cluster_reduced_update_problem(
        pair,
        gate,
        max_loop_size=4,
        include_full_system=True,
        psd_project=False,
        optimize=optimize,
    )

    metric_scale = max(1.0, np.linalg.norm(exact.metric))
    linear_scale = max(1.0, np.linalg.norm(exact.linear_term))
    assert gloop.loop_regions
    assert len(gloop.terms) == 1
    assert gloop.terms[0].region_tids == frozenset(pair.tn.tensor_map)
    assert not gloop.terms[0].boundary_inds
    assert np.linalg.norm(gloop.raw_metric - exact.metric) / metric_scale < 1e-8
    assert (
        np.linalg.norm(gloop.linear_term - exact.linear_term) / linear_scale
        < 1e-8
    )

    rng = np.random.default_rng(200 + bond_dim)
    left = rng.normal(
        size=(pair.theta_shape[0], pair.theta_shape[1], pair.bond_dimension)
    ) + 1j * rng.normal(
        size=(pair.theta_shape[0], pair.theta_shape[1], pair.bond_dimension)
    )
    right = rng.normal(
        size=(pair.bond_dimension, pair.theta_shape[2], pair.theta_shape[3])
    ) + 1j * rng.normal(
        size=(pair.bond_dimension, pair.theta_shape[2], pair.theta_shape[3])
    )
    theta = np.einsum("aps,sqb->apqb", left, right, optimize=True).reshape(-1)
    candidate = pair.reconstruct_tn(left, right)
    target = pair.gate_target_tn(gate)
    candidate_dense = np.asarray(candidate.to_dense(optimize=optimize)).reshape(-1)
    target_dense = np.asarray(target.to_dense(optimize=optimize)).reshape(-1)

    exact_norm = np.vdot(candidate_dense, candidate_dense)
    exact_overlap = np.vdot(candidate_dense, target_dense)
    metric_norm = np.vdot(theta, gloop.metric @ theta)
    metric_overlap = np.vdot(theta, gloop.linear_term)

    assert _relative_error(metric_norm, exact_norm) < 1e-8
    assert _relative_error(metric_overlap, exact_overlap) < 1e-8
    assert _relative_error(gloop.target_norm, np.vdot(target_dense, target_dense)) < 1e-8


def test_su_cluster_radius_grows_through_a_nontrivial_three_by_three_peps():
    core, gauges = _su_gauged_peps(3, 3, seed=9)
    pair = prepare_reduced_bond_pair(core, gauges, where=((1, 1), (1, 2)))
    exact = exact_reduced_update_problem(pair, _cnot())
    radius_one = su_cluster_reduced_update_problem(pair, _cnot(), radius=1)
    full = su_cluster_reduced_update_problem(
        pair,
        _cnot(),
        radius=pair.full_cluster_radius(),
    )

    assert pair.full_cluster_radius() == 2
    assert radius_one.cluster_tids
    assert radius_one.boundary_inds
    assert not full.boundary_inds
    assert np.allclose(full.metric, exact.metric, atol=1e-8, rtol=1e-10)


def test_open_leg_loop_cluster_adds_plaquette_regions_additively():
    core, gauges = _su_gauged_peps(3, 3, seed=9)
    pair = prepare_reduced_bond_pair(core, gauges, where=((1, 1), (1, 2)))
    loop_cluster = loop_cluster_reduced_update_problem(
        pair,
        _cnot(),
        max_loop_size=4,
    )
    active_tids = {pair.left_tid, pair.right_tid}
    solution = solve_reduced_als(loop_cluster, max_iterations=12, rcond=1e-11)

    assert loop_cluster.loop_regions
    assert len(loop_cluster.terms) > 1
    assert all(
        active_tids.issubset(term.region_tids)
        for term in loop_cluster.terms
    )
    assert any(term.count < 0 for term in loop_cluster.terms)
    assert loop_cluster.psd_projected
    assert np.allclose(
        loop_cluster.metric,
        loop_cluster.metric.conj().T,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(loop_cluster.metric).min() >= -1e-10
    assert np.allclose(
        loop_cluster.linear_term,
        loop_cluster.metric @ loop_cluster.target.reshape(-1),
        atol=1e-10,
    )
    assert all(
        later <= earlier + 1e-9 * max(1.0, earlier)
        for earlier, later in zip(solution.costs, solution.costs[1:])
    )


def test_apply_reduced_loop_cluster_gate_regauges_the_updated_peps():
    core, gauges = _su_gauged_peps()
    result = apply_reduced_loop_cluster_gate(
        core,
        gauges,
        _cnot(),
        where=((0, 0), (0, 1)),
        max_loop_size=0,
        regauge_opts={"max_iterations": 6, "tol": 0.0},
        als_opts={"max_iterations": 12, "rcond": 1e-11},
    )
    rebuilt = _physical_tn(result.core, result.gauges)

    assert result.core is not core
    assert result.gauges is not gauges
    assert result.reused_gauge_count >= 1
    assert np.allclose(
        rebuilt.to_dense(),
        result.physical_tn.to_dense(),
        atol=1e-9,
        rtol=1e-9,
    )
    assert np.allclose(
        result.problem.linear_term,
        result.problem.metric @ result.problem.target.reshape(-1),
        atol=1e-10,
    )


def test_apply_reduced_loop_cluster_gate_matches_exact_full_system_update():
    core, gauges = _su_gauged_peps()
    result = apply_reduced_loop_cluster_gate(
        core,
        gauges,
        _cnot(),
        where=((0, 0), (0, 1)),
        max_loop_size=0,
        include_full_system=True,
        psd_project=False,
        regauge_opts={"max_iterations": 4, "tol": 0.0},
        als_opts={"max_iterations": 12, "rcond": 1e-11},
    )
    exact = exact_reduced_update_problem(result.pair, _cnot())
    exact_solution = solve_reduced_als(
        exact,
        max_iterations=12,
        rcond=1e-11,
    )
    expected = result.pair.reconstruct_tn(
        exact_solution.left,
        exact_solution.right,
    )

    assert np.allclose(result.problem.raw_metric, exact.metric, atol=1e-10)
    assert np.allclose(
        result.physical_tn.to_dense(),
        expected.to_dense(),
        atol=1e-9,
        rtol=1e-9,
    )


def test_apply_reduced_loop_cluster_gate_truncates_and_can_update_inplace():
    core, gauges = _su_gauged_peps()
    original_physical = _physical_tn(core, gauges)
    result = apply_reduced_loop_cluster_gate(
        core,
        gauges,
        _cnot(),
        where=((0, 0), (0, 1)),
        max_bond=1,
        max_loop_size=0,
        regauge_opts={"max_iterations": 4, "tol": 0.0},
        als_opts={"max_iterations": 12, "rcond": 1e-11},
        inplace=True,
    )
    rebuilt = _physical_tn(core, gauges)

    assert result.core is core
    assert result.gauges is gauges
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1
    assert core.ind_size(result.pair.bond_ind) == 1
    assert np.allclose(
        rebuilt.to_dense(),
        result.physical_tn.to_dense(),
        atol=1e-9,
        rtol=1e-9,
    )
    assert not np.allclose(rebuilt.to_dense(), original_physical.to_dense())


def test_gate_loop_cluster_wrapper_applies_a_nearest_neighbor_stream():
    core, gauges = _su_gauged_peps()
    out, results = gate_loop_cluster(
        core,
        ((_cnot(), ((0, 0), (0, 1))),),
        gauges=gauges,
        max_loop_size=0,
        regauge_opts={"max_iterations": 4, "tol": 0.0},
        als_opts={"max_iterations": 12, "rcond": 1e-11},
        inplace=False,
        return_results=True,
    )
    rebuilt = _physical_tn(out, gauges)

    assert out is not core
    assert len(results) == 1
    assert np.allclose(
        rebuilt.to_dense(),
        results[0].physical_tn.to_dense(),
        atol=1e-9,
        rtol=1e-9,
    )

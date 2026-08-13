"""Focused tests for SU- and D2BP-closed reduced PEPS updates."""

from dataclasses import replace
import importlib

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy as py
from pepsy.bp import (
    CompressionBudgetError,
    ContractionPlanCache,
    apply_reduced_loop_cluster_gate,
    compress_reduced_loop_cluster,
    exact_reduced_update_problem,
    loop_cluster_reduced_update_problem,
    prepare_reduced_bond_pair,
    ReducedLoopClusterCache,
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


def test_reduced_prepare_runs_fresh_bp_when_no_boundary_data_is_supplied():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    assert pair.bp_info["source"] == "fresh_bp"
    assert pair.boundary_messages


def test_reduced_exact_environment_is_psd_and_exposes_cost_preflight():
    peps = _random_peps()
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    gate = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    problem = exact_reduced_update_problem(
        pair,
        gate,
        materialize_metric=True,
        cost_check=True,
    )
    assert problem.psd_projected
    assert problem.contraction_cost["flops_log10"] >= 0.0
    assert np.min(np.linalg.eigvalsh(problem.metric)) >= -1e-10
    with pytest.raises(CompressionBudgetError):
        exact_reduced_update_problem(
            pair,
            gate,
            max_flops_log10=-1.0,
        )


def test_reduced_update_preserves_torch_backend_for_all_als_paths():
    torch = pytest.importorskip("torch")
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=67,
    )
    for tensor in peps.tensor_map.values():
        tensor.modify(data=torch.as_tensor(tensor.data))

    bp = two_norm_bp(peps, max_iterations=1, tol=0.0, diis=False)
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
    )
    gate = torch.eye(4, dtype=torch.float64)
    for solver in ("quimb", "qr", "normal"):
        problem = exact_reduced_update_problem(
            pair,
            gate,
            materialize_metric=solver != "quimb",
        )
        solution = solve_reduced_als(
            problem,
            max_bond=1,
            max_iterations=2,
            solver=solver,
        )
        assert isinstance(problem.target, torch.Tensor)
        assert isinstance(solution.left, torch.Tensor)
        assert isinstance(solution.right, torch.Tensor)


def test_reduced_update_exposes_native_quimb_autodiff_solver():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, dtype="float64", seed=73)
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    problem = exact_reduced_update_problem(
        pair,
        np.eye(pair.theta_shape[1] * pair.theta_shape[2]),
    )
    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=4,
        solver="autodiff",
        autodiff_opts={
            "autodiff_backend": "autograd",
            "optimizer": "L-BFGS-B",
            "progbar": False,
        },
    )
    assert solution.left.shape[-1] == 1
    assert solution.right.shape[0] == 1
    assert np.isfinite(solution.costs[-1])


def test_reduced_update_autodiff_keeps_torch_tensors_native():
    torch = pytest.importorskip("torch")
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=74,
    )
    for tensor in peps.tensor_map.values():
        tensor.modify(data=torch.as_tensor(tensor.data))
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    problem = exact_reduced_update_problem(
        pair,
        torch.eye(4, dtype=torch.float64),
    )
    solution = solve_reduced_als(
        problem,
        max_bond=1,
        max_iterations=2,
        solver="autodiff",
        autodiff_opts={"autodiff_backend": "torch", "optimizer": "Adam"},
    )
    assert isinstance(solution.left, torch.Tensor)
    assert isinstance(solution.right, torch.Tensor)


def test_loop_cluster_normalizes_rescaled_directed_message_pairs():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=71,
    )
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)
    scaled_messages = {
        key: 100.0 * message for key, message in bp.messages.items()
    }
    where = ((0, 0), (1, 0))
    pair = prepare_reduced_bond_pair(
        peps,
        where=where,
        boundary_messages=bp.messages,
    )
    scaled_pair = prepare_reduced_bond_pair(
        peps,
        where=where,
        boundary_messages=scaled_messages,
    )
    gate = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    problem = loop_cluster_reduced_update_problem(
        pair,
        gate,
        max_loop_size=4,
        psd_project=False,
    )
    scaled_problem = loop_cluster_reduced_update_problem(
        scaled_pair,
        gate,
        max_loop_size=4,
        psd_project=False,
    )

    assert np.allclose(problem.environment.data, scaled_problem.environment.data)


def test_reduced_loop_cluster_cache_reuses_larger_cutoff_geometry():
    peps = qtn.PEPS.rand(
        4,
        4,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260812,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        where=((1, 1), (1, 2)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    cache = ReducedLoopClusterCache()

    loops_10 = cache.precompute(pair, max_loop_size=10)
    loops_4 = cache.loops_for(pair, max_loop_size=4)

    assert cache.generated_max_size == 10
    assert cache.raw_loop_count == len(loops_10)
    assert loops_4 == tuple(loop for loop in loops_10 if len(loop) <= 4)

    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    first = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_loop_size=4,
        loop_cache=cache,
        psd_project=False,
    )
    second = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_loop_size=8,
        loop_cache=cache,
        psd_project=False,
    )
    assert cache.generated_max_size == 10
    anchor = frozenset(
        {pair.left_tid, pair.right_tid, *pair._cluster_tids(0)}
    )
    expected_first = tuple(
        sorted(
            {
                frozenset(anchor | loop)
                for loop in loops_4
                if frozenset(anchor | loop) != anchor
            },
            key=lambda region: (len(region), tuple(sorted(map(repr, region)))),
        )
    )
    assert first.loop_regions == expected_first
    assert len(second.loop_regions) > len(first.loop_regions)


def test_loop_cluster_total_cutoff_and_tree_reduction_are_explicit():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260815,
    )
    where = ((1, 1), (1, 2))
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)
    pair = prepare_reduced_bond_pair(
        peps,
        where=where,
        boundary_messages=bp.messages,
    )
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    problem = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=4,
        tree_reduction=True,
        psd_project=False,
    )

    assert problem.max_cluster_size == 4
    assert problem.tree_reduction
    assert problem.combination == "additive"
    assert all(len(term.region_tids) <= 4 for term in problem.terms)

    anchor = frozenset({pair.left_tid, pair.right_tid, *pair._cluster_tids(0)})
    for term in problem.terms:
        remaining = set(term.region_tids) - set(anchor)
        degrees = {tid: 0 for tid in remaining}
        for index, tids in pair.tn.ind_map.items():
            if index == pair.bond_ind or len(tids) != 2:
                continue
            left, right = tuple(tids)
            if left in term.region_tids and right in term.region_tids:
                if left in remaining:
                    degrees[left] += 1
                if right in remaining:
                    degrees[right] += 1
        assert all(degree >= 2 for degree in degrees.values())


def test_loop_cluster_full_total_cutoff_matches_exact_open_environment():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260816,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        bp_opts={"max_iterations": 2, "tol": 0.0, "diis": False},
    )
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    exact = exact_reduced_update_problem(
        pair,
        identity,
        psd_project=False,
    )
    loop = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=len(pair.tn.tensor_map),
        include_full_system=True,
        tree_reduction=False,
        psd_project=False,
    )

    np.testing.assert_allclose(loop.environment.data, exact.environment.data)


def test_large_full_total_cutoff_shortcuts_loop_enumeration():
    peps = qtn.PEPS.rand(
        4,
        5,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260824,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        where=((1, 2), (2, 2)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    exact = exact_reduced_update_problem(pair, identity, psd_project=False)
    loop = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=len(pair.tn.tensor_map),
        include_full_system=True,
        tree_reduction=False,
        psd_project=False,
    )

    assert loop.loop_regions == ()
    assert len(loop.terms) == 1
    np.testing.assert_allclose(loop.environment.data, exact.environment.data)


def test_loop_cluster_cutoff_sweep_reaches_exact_reference():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260817,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        where=((1, 1), (1, 2)),
        bp_opts={"max_iterations": 2, "tol": 0.0, "diis": False},
    )
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    exact = exact_reduced_update_problem(pair, identity, psd_project=False)
    cache = ReducedLoopClusterCache()
    errors = []
    term_counts = []
    for cutoff in (2, 4, len(pair.tn.tensor_map)):
        problem = loop_cluster_reduced_update_problem(
            pair,
            identity,
            max_cluster_size=cutoff,
            include_full_system=cutoff == len(pair.tn.tensor_map),
            tree_reduction=False,
            loop_cache=cache,
            psd_project=False,
        )
        errors.append(
            np.linalg.norm(problem.environment.data - exact.environment.data)
        )
        term_counts.append(len(problem.terms))

    assert all(np.isfinite(error) for error in errors)
    assert term_counts[-1] >= term_counts[0]
    assert errors[-1] < 1e-10
    assert cache.generated_max_size >= 0
    assert cache.generated_max_size < len(pair.tn.tensor_map)


def test_loop_cluster_plan_cache_reuses_topology_identical_contractions():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260818,
    )
    pair = prepare_reduced_bond_pair(
        peps,
        where=((1, 1), (1, 2)),
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    identity = np.eye(pair.theta_shape[1] * pair.theta_shape[2])
    cache = ContractionPlanCache()
    loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=4,
        plan_cache=cache,
        psd_project=False,
    )
    first = cache.snapshot()
    loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=4,
        plan_cache=cache,
        psd_project=False,
    )
    second = cache.snapshot()

    assert first["misses"] > 0
    assert second["hits"] > first["hits"]
    assert second["plans"] == first["plans"]


def test_loop_cluster_plan_cache_reuses_fresh_reduced_pairs():
    """Gate streams can reuse plans when each update rebuilds its pair."""
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260819,
    )
    where = ((1, 1), (1, 2))
    pair = prepare_reduced_bond_pair(
        peps,
        where=where,
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    pair_again = prepare_reduced_bond_pair(
        peps,
        where=where,
        boundary_messages=pair.boundary_messages,
    )
    assert pair.reduced_left_ind == pair_again.reduced_left_ind
    assert pair.reduced_right_ind == pair_again.reduced_right_ind

    cache = ContractionPlanCache()
    identity = np.eye(4)
    loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_cluster_size=4,
        plan_cache=cache,
        psd_project=False,
    )
    first = cache.snapshot()
    loop_cluster_reduced_update_problem(
        pair_again,
        identity,
        max_cluster_size=4,
        plan_cache=cache,
        psd_project=False,
    )
    second = cache.snapshot()

    assert first["misses"] > 0
    assert second["hits"] > first["hits"]
    assert second["plans"] == first["plans"]


def test_reduced_loop_cluster_bp_convergence_policy_can_raise():
    peps = _random_peps()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        warned_pair = prepare_reduced_bond_pair(
            peps,
            where=((0, 0), (1, 0)),
            bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
            bp_convergence="warn",
        )
    assert warned_pair.bp_info["bp_convergence"] == "warn"

    with pytest.raises(RuntimeError, match="did not converge"):
        prepare_reduced_bond_pair(
            peps,
            where=((0, 0), (1, 0)),
            bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
            bp_convergence="raise",
        )

    with pytest.raises(ValueError, match="bp_convergence"):
        prepare_reduced_bond_pair(
            peps,
            where=((0, 0), (1, 0)),
            run_bp=False,
            bp_convergence="invalid",
        )


def test_reduced_loop_cluster_cache_ignores_pepo_singleton_index_names():
    pepo = qtn.PEPO.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260813,
    )
    first = prepare_reduced_bond_pair(
        pepo,
        where=((0, 0), (0, 1)),
        run_bp=False,
    )
    second = prepare_reduced_bond_pair(
        pepo,
        where=((0, 0), (0, 1)),
        run_bp=False,
    )
    cache = ReducedLoopClusterCache()

    first_loops = cache.loops_for(first, max_loop_size=4)
    second_loops = cache.loops_for(second, max_loop_size=4)

    assert second_loops == first_loops


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


def test_reduced_loop_cluster_gate_reuses_explicit_loop_cache():
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=20260814,
    )
    where = ((1, 0), (1, 1))
    bp = two_norm_bp(peps, max_iterations=1, tol=0.0, diis=False)
    pair = prepare_reduced_bond_pair(
        peps,
        where=where,
        boundary_messages=bp.messages,
    )
    cache = ReducedLoopClusterCache()
    cache.precompute(pair, max_loop_size=6)

    result = apply_reduced_loop_cluster_gate(
        peps,
        np.eye(4),
        where=where,
        boundary_messages=bp.messages,
        max_bond=1,
        max_loop_size=4,
        loop_cache=cache,
        als_opts={"max_iterations": 1},
        regauge_opts={"max_iterations": 1},
        inplace=False,
    )

    assert result.problem.max_loop_size == 4
    assert cache.generated_max_size == 6


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
    pair = prepare_reduced_bond_pair(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
    )
    cache = ReducedLoopClusterCache()
    cache.precompute(pair, max_loop_size=4)

    out, results = gate_loop_cluster(
        peps,
        ((gate, ((0, 0), (1, 0))),),
        gauges=gauges,
        boundary_messages=bp.messages,
        max_bond=1,
        max_loop_size=3,
        loop_cache=cache,
        als_opts={"max_iterations": 2},
        regauge_opts={"max_iterations": 1},
        inplace=False,
        return_results=True,
    )

    assert out is not peps
    assert len(results) == 1
    assert len(gauges) == len(tuple(out.inner_inds()))
    assert results[0].pair.boundary_messages
    assert cache.generated_max_size == 4


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

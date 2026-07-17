"""Exact-oracle tests for the SU-gauged reduced PEPS update."""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from pepsy.bp import (  # noqa: E402
    exact_reduced_update_problem,
    gauge_all_simple,
    prepare_reduced_bond_pair,
    solve_reduced_als,
)


def _su_gauged_peps():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=23,
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


def test_reduced_pair_reconstructs_the_su_gauged_state_exactly():
    pair = _pair()
    rebuilt = pair.reconstruct_tn()

    assert np.allclose(
        rebuilt.to_dense(),
        pair.tn.to_dense(),
        atol=1e-10,
        rtol=1e-10,
    )


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

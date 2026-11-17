"""Multiple ALS initializations select in one fixed metric and retain native arrays."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn
from cotengra.contract import default_implementation

from pepsy.bp import LoopClusterUpdateTracker, PreparedLoopClusterUpdate, ReducedEnvironmentUpdateProblem
from pepsy.bp import prepare_reduced_bond_pair, solve_reduced_als
from pepsy.bp._als_restarts import random_factors
from pepsy.bp.reduced_update import _open_environment_tensor, _svd_initial_factors
from pepsy.optimizers import PepsOptimizer
from pepsy.operators import gate_loop_cluster


@pytest.fixture(autouse=True)
def contraction_implementation():
    # Match production replay and avoid old Autoray's backend-mixed cache
    # dispatch when NumPy and Torch exercise identical contraction layouts.
    from pepsy.backends.config import (
        _patch_backend_string_namespace_key, _patch_unhashable_device_namespace_key,
    )

    _patch_unhashable_device_namespace_key()
    _patch_backend_string_namespace_key()
    with default_implementation('cotengra'):
        yield


def problem(backend='numpy', dtype='complex128', device='cpu'):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=7, dtype=dtype)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        state.apply_to_arrays(lambda x: torch.as_tensor(x, device=device))
    elif backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            cp.zeros(1)
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(str(exc))
        state.apply_to_arrays(cp.asarray)
    pair = prepare_reduced_bond_pair(state, where=((0, 0), (1, 0)), run_bp=False)
    target = pair.theta_array()
    a, _, _, b = pair.theta_shape
    rng = np.random.default_rng(0)
    u = np.linalg.qr(rng.normal(size=(a*b, a*b))+1j*rng.normal(size=(a*b, a*b)))[0]
    metric = (u*np.geomspace(1e-5, 1, a*b)) @ u.conj().T
    metric = ar.do('array', metric.reshape(a, b, a, b), like=target,
                   dtype=ar.to_backend_dtype(ar.get_dtype_name(target), like=target))
    return ReducedEnvironmentUpdateProblem(pair, _open_environment_tensor(pair, metric), target)


def test_random_trial_can_win_and_returns_its_factors_and_diagnostics():
    target = problem()
    options = dict(max_bond=1, max_iterations=1, tol=0., monitor_convergence=True)
    baseline = solve_reduced_als(target, **options)
    expected = solve_reduced_als(target, initial_factors=random_factors(
        _svd_initial_factors(target.target, 1), 4), **options)
    actual = solve_reduced_als(target, random_restarts=1, restart_seed=4, **options)
    report = actual.solver_report['restarts']
    assert expected.costs[-1] < baseline.costs[-1]
    np.testing.assert_allclose(actual.theta(), expected.theta(), rtol=1e-12, atol=1e-12)
    assert report['selected_trial'] == 1
    assert [t['initialization'] for t in report['trials']] == ['svd', 'random']
    assert report['total_iterations'] == 2 and actual.iterations == 1
    for row, solution in zip(report['trials'], (baseline, expected)):
        assert row['cost'] == pytest.approx(target.cost(solution.theta()), rel=1e-12)
        assert row['normalized_residual'] == pytest.approx(row['cost']/target.target_norm)


@pytest.mark.parametrize('backend,solver,device', [
    ('numpy', 'quimb', 'cpu'), ('numpy', 'qr', 'cpu'),
    ('torch', 'quimb', 'cpu'), ('torch', 'torch', 'cpu'),
    ('torch', 'torch', 'cuda'),
    ('cupy', 'quimb', 'cuda'), ('cupy', 'qr', 'cuda'),
])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_restarts_reproduce_without_mutating_inputs_or_global_rng(backend, solver, device, dtype):
    target = problem(backend, dtype, device)
    before = ar.to_numpy(target.target).copy()
    numpy_state = np.random.get_state()
    torch_state = None
    if backend == 'torch':
        import torch
        torch_state = torch.random.get_rng_state().clone()
        if device == 'cuda':
            cuda_state = torch.cuda.get_rng_state().clone()
    options = dict(solver=solver, max_bond=1, max_iterations=3, tol=0.,
                   monitor_convergence=True, measure_fidelity=True)
    baseline = solve_reduced_als(target, **options)
    first = solve_reduced_als(target, random_restarts=1, restart_seed=4, **options)
    second = solve_reduced_als(target, random_restarts=1, restart_seed=4, **options)
    tol = 2e-5 if dtype == 'complex64' else 2e-11
    np.testing.assert_allclose(ar.to_numpy(first.theta()), ar.to_numpy(second.theta()), rtol=tol, atol=tol)
    assert target.cost(first.theta()) <= target.cost(baseline.theta()) + tol*target.target_norm
    np.testing.assert_array_equal(ar.to_numpy(target.target), before)
    assert ar.get_dtype_name(first.left) == dtype
    assert first.fidelity_status == 'valid' and first.local_fidelity is not None
    for actual, expected in zip(np.random.get_state(), numpy_state):
        np.testing.assert_equal(actual, expected)
    if torch_state is not None:
        assert first.left.device == target.target.device
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        if device == 'cuda':
            assert torch.equal(torch.cuda.get_rng_state(), cuda_state)


@pytest.mark.parametrize('kwargs', [
    {'random_restarts': -1}, {'random_restarts': True}, {'random_restarts': 1.5},
    {'restart_seed': -1}, {'restart_seed': True},
])
def test_restart_options_validate_before_fitting(kwargs):
    with pytest.raises(ValueError, match='nonnegative integer'):
        solve_reduced_als(problem(), **kwargs)


def test_autodiff_rejects_dense_random_restart_option():
    with pytest.raises(NotImplementedError, match='requires dense ALS'):
        solve_reduced_als(problem(), solver='autodiff', random_restarts=1)


def test_random_restarts_support_compiled_torch_kernels():
    torch = pytest.importorskip('torch')
    target = problem('torch')
    options = dict(solver='torch', max_bond=1, max_iterations=3, tol=0.,
                   monitor_convergence=True, random_restarts=1, restart_seed=4)
    eager = solve_reduced_als(target, **options)
    compiled = solve_reduced_als(target, torch_backend='aot_eager', **options)
    torch.testing.assert_close(compiled.theta(), eager.theta(), rtol=1e-9, atol=1e-10)
    assert compiled.solver_report['restarts']['selected_trial'] == eager.solver_report['restarts']['selected_trial']


def test_supplied_factors_remain_the_first_trial_without_an_extra_svd(monkeypatch):
    import pepsy.bp.reduced_update as module

    target = problem()
    initial = random_factors(_svd_initial_factors(target.target, 1), 17)
    expected = target.cost(np.einsum('apk,kqb->apqb', *initial))

    def forbidden(*args, **kwargs):
        pytest.fail('supplied warm start was replaced by SVD')

    monkeypatch.setattr(module, '_svd_initial_factors', forbidden)
    result = solve_reduced_als(target, initial_factors=initial, random_restarts=1,
        max_bond=2, max_iterations=2, tol=0., monitor_convergence=True)
    assert result.left.shape[-1] == 1
    assert result.solver_report['restarts']['trials'][0]['initialization'] == 'supplied'
    first = solve_reduced_als(target, initial_factors=initial, max_bond=2, max_iterations=2, tol=0.)
    assert first.costs[0] == pytest.approx(expected)


@pytest.mark.parametrize('controls', [dict(solver='lbfgs'), dict(tensor_mode='full')])
def test_peps_restart_option_rejects_non_als_modes(controls):
    from pepsy.optimizers.peps._full_update import options

    with pytest.raises(ValueError, match='requires reduced ALS'):
        options(dict(random_restarts=1, **controls))


@pytest.mark.parametrize('operator', [False, True])
def test_peps_optimizer_full_update_reports_both_trials_and_preserves_fidelity(operator):
    torch = pytest.importorskip('torch')
    state = (qtn.PEPO.rand(2, 2, bond_dim=2, phys_dim=2, dtype='complex128', seed=7)
             if operator else qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=7))
    state.apply_to_arrays(torch.as_tensor)
    gate = torch.diag(torch.exp(.2j*torch.tensor([1., -1., -1., 1.], dtype=torch.float64)))
    where = ((0, 0), (1, 0))
    gates = [(gate, where, 'upper')] if operator else [(gate, where)]
    optimizer = PepsOptimizer(state, gates, chi=1, mode='full-update',
        fit_mode='direct', boundary_chi=32, boundary_convergence=False,
        normalize_initial=False, contraction_opt='greedy',
        full_update_kwargs=dict(random_restarts=1, restart_seed=4, max_iterations=3))
    out = optimizer.run(normalize_target=False, normalize_final=False,
                        measure_infidelity=False, measure_final_infidelity=False)
    report = optimizer.get_step_records()[0]['optimizer_result']
    trials = report['als_restarts']['trials']
    assert len(trials) == 2 and trials[0]['initialization'] == 'svd'
    assert trials[1]['seed'] == 4
    if operator:
        from pepsy.operators import gate as apply_gate
        target = apply_gate(state, gate, where, which='upper', contract=False)
    else:
        target = state.gate(gate, where, contract=False)
    a, b = out.to_dense().reshape(-1), target.to_dense().reshape(-1)
    fidelity = float(abs(torch.vdot(a, b))**2/(torch.vdot(a, a).real*torch.vdot(b, b).real))
    assert report['fidelity'] == pytest.approx(fidelity, abs=2e-10)


def test_prepared_and_reference_loop_cluster_restarts_match():
    torch = pytest.importorskip('torch')
    state = qtn.PEPO.rand(2, 3, bond_dim=2, phys_dim=2, dtype='complex128', seed=177)
    state.apply_to_arrays(torch.as_tensor)
    where = ((0, 1), (1, 1))
    gate = torch.diag(torch.exp(.17j*torch.tensor([1., -1., -1., 1.], dtype=torch.float64)))
    options = dict(random_restarts=1, restart_seed=4, max_iterations=3, tol=0.)
    plan = PreparedLoopClusterUpdate(state, [(where, 'upper')], max_bond=1,
                                     optimize='greedy', **options)
    gauges, tracker = {}, LoopClusterUpdateTracker()
    out, results = plan.apply(state.copy(), [gate], gauges=gauges, return_results=True, tracker=tracker)
    ref_gauges = {}
    reference, refs = gate_loop_cluster(state.copy(), gate, where=where, which='upper',
        gauges=ref_gauges, max_bond=1, cluster_mode='loopunion', base_radius=4,
        max_loop_size=0, optimize='greedy', psd_project=False, strip_exponent=True,
        return_results=True,
        als_opts=dict(solver='torch', monitor_convergence=True, **options))
    for network, weights in ((out, gauges), (reference, ref_gauges)):
        network.gauge_simple_insert(weights)
    torch.testing.assert_close(out.to_dense(), reference.to_dense(), rtol=2e-9, atol=2e-9)
    for result in (results[0], refs[0]):
        info = result.solution.solver_report['restarts']
        assert len(info['trials']) == 2
        assert info['selected_trial'] == min(range(2), key=lambda i: info['trials'][i]['cost'])
    assert tracker.records[0]['als_restarts'] == results[0].solution.solver_report['restarts']
    results[0].solution.solver_report['restarts']['selected_trial'] = 99
    assert tracker.records[0]['als_restarts']['selected_trial'] != 99


def test_prepared_batches_keep_per_pair_restart_selection():
    from test_bp_cluster_batches import setup, matrices, weights, pair_amplitude

    state, schedule = setup()
    outputs = []
    for size in (1, 2, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=3,
            tol=0., optimize='greedy', batch_size=size, random_restarts=1, restart_seed=4)
        gauges = weights(state)
        out, results = plan.apply(state.copy(), matrices(.17, state), gauges=gauges,
                                  return_results=True)
        assert plan.info['batched_pair_updates'] == (0 if size == 1 else 2)
        outputs.append((out, gauges, results))
        for result in results:
            info = result.solution.solver_report['restarts']
            assert len(info['trials']) == 2 and info['total_iterations'] == 6
            assert info['selected_trial'] == min(range(2), key=lambda i: info['trials'][i]['cost'])
    import torch
    for out, gauges, _ in outputs[1:]:
        for sites, _ in schedule:
            torch.testing.assert_close(pair_amplitude(out, gauges, sites),
                                       pair_amplitude(*outputs[0][:2], sites), rtol=2e-8, atol=2e-8)

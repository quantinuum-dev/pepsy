"""Public strip ALS defaults keep healthy fits free of cap searches."""

from unittest.mock import patch

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer, SweepOptimizer
from test_peps_strip_batches import inputs


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('style,axis', [('row-column', 'y'), ('row', 'x')])
def test_default_sweep_matches_explicit_strip_recipe(backend, style, axis, monkeypatch):
    state, gate = inputs(backend, (2, 3))
    sites = ([(0, 0), (0, 1), (0, 2)] if axis == 'x'
             else [(0, 0), (1, 0)])
    gates = [(gate, (a, b)) for a, b in zip(sites, sites[1:])]
    original = ar.to_numpy(state.to_dense()).copy()
    calls = []
    optimize_axis = SweepOptimizer.optimize_axis

    def check(self, actual_axis, **kwargs):
        assert self.chi == (8, 16)
        assert self.fit_init_strategy == 'reuse-direct'
        assert self.boundary_engine == 'dmrg'
        assert '__PEPS_GATE__' in self.state_target.tags
        assert kwargs['solver'] == 'als'
        assert kwargs['solver_options']['linear_solver'] == 'auto'
        assert kwargs['solver_options']['n_round_trips'] == 2
        assert kwargs['env_n_iter'] == 10
        assert actual_axis == axis
        assert kwargs['n_round_trips'] == 2
        result = optimize_axis(self, actual_axis, **kwargs)
        assert all(1 <= report['n_round_trips'] <= 2 for report in result)
        calls.append(True)
        return result

    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', check)
    driver = PepsOptimizer(state, gates, chi=2, mode='sweep',
                           update_style=style, contraction_opt='greedy')
    result = driver.run(infidelity_tol=0.)
    explicit = PepsOptimizer(
        state, gates, chi=2, mode='sweep', update_style=style,
        gate_order='row' if axis == 'x' else 'column', optimizer='als',
        optimizer_options={'linear_solver': 'auto', 'n_round_trips': 2},
        sweep_optimize_kwargs={'n_round_trips': 2},
        boundary_chi=(8, 16), boundary_convergence=False,
        fit_mode='dmrg', fit_init_strategy='reuse-direct',
        boundary_kwargs={'n_iter': 10}, contraction_opt='greedy',
    ).run(k_2q_batch='strip', warmstart_method='simple-update', infidelity_tol=0.)
    np.testing.assert_allclose(ar.to_numpy(result.to_dense()), ar.to_numpy(explicit.to_dense()), atol=1e-10)
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), original)
    assert calls == [True, True]
    assert driver.boundary_convergence is None
    record, = driver.get_step_records()
    assert record['warmstart_method'] == 'simple-update'
    assert record['target_representation'] == 'layered'


@pytest.mark.parametrize('constructor_mode,run_mode', [('global', 'sweep'), ('sweep', 'global')])
def test_run_mode_override_resolves_sweep_defaults(constructor_mode, run_mode, monkeypatch):
    state, gate = inputs(shape=(2, 3))
    driver = PepsOptimizer(state, [(gate, ((0, 0), (1, 0)))], chi=2,
                           mode=constructor_mode, contraction_opt='greedy')
    # Inspect the actual target and warm start while avoiding a global AD run.
    observed = []
    def optimize(warmstart, target, **kwargs):
        observed.append('__PEPS_GATE__' in target.tags)
        return warmstart, None, None
    monkeypatch.setattr(driver, '_optimize_state', optimize)
    driver.run(mode=run_mode, infidelity_tol=0.)
    assert observed == [run_mode == 'sweep']
    assert driver.last_gate_order['policy'] == ('column' if run_mode == 'sweep' else 'input')
    assert driver.boundary_kwargs['fit_init_strategy'] == ('reuse-direct' if run_mode == 'sweep' else 'guess-direct')


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('mode', ['global', 'full-update'])
def test_default_boundary_guess_normalizes_without_src(backend, mode, monkeypatch):
    state, _ = inputs(backend)
    state *= 3
    original = ar.to_numpy(state.to_dense()).copy()
    compress = qtn.tensor_network_1d_compress
    methods = []

    def capture(*args, **kwargs):
        methods.append(kwargs['method'])
        return compress(*args, **kwargs)

    monkeypatch.setattr(qtn, 'tensor_network_1d_compress', capture)
    driver = PepsOptimizer(state, [], chi=2, mode=mode, boundary_chi=32,
                           boundary_convergence=False, contraction_opt='greedy')
    result = driver.run()
    vector = ar.to_numpy(result.to_dense())
    assert methods and set(methods) == {'direct'}
    np.testing.assert_allclose(vector, original / np.linalg.norm(original), atol=1e-10)
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), original)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_sweep_starts_at_gated_strip_and_refreshes_environments_on_returns(axis, monkeypatch):
    state, gate = inputs(shape=(3, 3))
    where = ((1, 0), (1, 1)) if axis == 'x' else ((0, 1), (1, 1))
    target = state.copy().gate_(gate, where, contract='split', cutoff=0.).to_dense().ravel()
    visits = []
    solve = SweepOptimizer._optimize_axis_slice_with_current_env

    def loss(state):
        vector = state.to_dense().ravel()
        return 1 - abs(np.vdot(target, vector))**2 / (np.vdot(target, target).real * np.vdot(vector, vector).real)

    def check(self, index, **kwargs):
        before = loss(self.state)
        result = solve(self, index, **kwargs)
        after = loss(self.state)
        assert result['loss_initial'] == pytest.approx(before, abs=1e-9)
        assert result['loss_final'] == pytest.approx(after, abs=1e-9)
        assert after <= before + 1e-9
        visits.append(index)
        return result

    monkeypatch.setattr(SweepOptimizer, '_optimize_axis_slice_with_current_env', check)
    driver = PepsOptimizer(
        state, [(gate, where)], chi=2, contraction_opt='greedy',
        boundary_chi=(32, 64), normalize_chi=32, evaluation_chi=(32, 64),
        fit_mode='direct',
    )
    driver.run(infidelity_tol=0.)
    assert visits == [1, 2, 1, 0, 1, 2, 1, 0, 1, 2]
    history = driver.get_step_records()[0]['optimizer_result']['sweep_history']
    assert [r['index'] for r in history] == visits
    assert {r['axis'] for r in history} == {axis}
    assert [r['sweep'] for r in history] == ['forward'] * 2 + (['backward'] * 2 + ['forward'] * 2) * 2
    assert all(r['contraction_cache']['hits'] > 0 for r in history)
    assert all(all(c['hits'] > 0 for c in r['environment_reuse']) for r in history)


@pytest.mark.parametrize('measure', [False, True])
@pytest.mark.parametrize('accept_if_improved', [False, True])
@pytest.mark.parametrize('final_metric', [False, True])
def test_invalid_strip_losses_retain_warm_start(measure, accept_if_improved, final_metric, monkeypatch):
    state, gate = inputs(shape=(2, 3))
    gates = [(gate, ((0, 1), (1, 1)))]
    kwargs = dict(chi=2, fit_mode='direct', contraction_opt='greedy',
                  boundary_convergence=False,
                  sweep_optimize_kwargs={'n_round_trips': 0, 'compute_final_loss': final_metric})
    warm = PepsOptimizer(state, gates, **kwargs).run(
        optimize=False, measure_infidelity=measure)
    solve = SweepOptimizer.optimize_axis

    def invalid(self, *args, **options):
        with patch.object(self, '_scaled_overlap_fidelity', return_value=1.25):
            return solve(self, *args, **options)

    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', invalid)
    # An explicitly requested final diagnostic can also be nonfinite. A failed
    # solve must still return through the driver's warm-start recovery path.
    monkeypatch.setattr(SweepOptimizer, '_debug_loss', lambda *args, **kwargs: float('nan'))
    driver = PepsOptimizer(state, gates, **kwargs)
    result = driver.run(measure_infidelity=measure, infidelity_tol=0.,
                        accept_if_improved=accept_if_improved)
    record, = driver.get_step_records()
    summary = record['optimizer_result']
    assert record['reason'] == 'optimizer_failed'
    assert record['optimizer_infidelity'] is None
    assert not summary['success']
    assert summary['termination_reason'] == 'no_valid_local_updates'
    assert summary['applied_local_updates'] == 0
    assert summary['invalid_local_updates'] == summary['n_runs'] == 2
    assert len(summary['sweep_history']) == 2
    assert all(r['invalid_loss'] for r in summary['sweep_history'])
    np.testing.assert_allclose(result.to_dense(), warm.to_dense(), atol=1e-12)


def test_strip_run_honors_controls_and_reports_progress(monkeypatch, capsys):
    state, gate = inputs(shape=(2, 3))
    calls = []
    solve = SweepOptimizer.optimize_axis
    normalize = SweepOptimizer._normalize_state

    def check(self, axis, **kwargs):
        assert kwargs['renormalize'] is True
        assert kwargs['normalize_boundaries'] is True
        assert kwargs['collect_boundary_norms'] is True
        assert callable(kwargs['run_callback'])
        calls.append(kwargs['start_index'])
        return solve(self, axis, **kwargs)

    normalizations = []
    def track_normalize(self, *args, **kwargs):
        normalizations.append(True)
        return normalize(self, *args, **kwargs)

    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', check)
    monkeypatch.setattr(SweepOptimizer, '_normalize_state', track_normalize)
    driver = PepsOptimizer(
        state, [(gate, ((0, 1), (1, 1)))], chi=2,
        fit_mode='direct', contraction_opt='greedy',
        sweep_optimize_kwargs={'n_cycles': 2, 'n_round_trips': 0,
                               'renormalize': True, 'normalize_boundaries': True,
                               'collect_boundary_norms': True, 'compute_initial_loss': True,
                               'compute_final_loss': True},
    )
    driver.run(measure_infidelity=False, sweep_progress=True)
    summary = driver.get_step_records()[0]['optimizer_result']
    assert calls == [1, 0]
    assert len(normalizations) == 3  # Each cycle, then the retained best state.
    assert summary['n_runs'] == 5
    assert summary['final_loss_measured']
    assert np.isfinite(summary['loss_before'])
    assert np.isfinite(summary['loss_after'])
    assert all(r['bdy_norm'] is not None and r['bdy_overlap_norm'] is not None
               for r in summary['sweep_history'])
    stderr = capsys.readouterr().err
    assert 'PEPS sweep' in stderr and '5/5' in stderr and 'up_' in stderr


def test_strip_run_rejects_conflicting_axes():
    state, gate = inputs(shape=(2, 3))
    driver = PepsOptimizer(
        state, [(gate, ((0, 1), (1, 1)))], chi=2,
        fit_mode='direct', contraction_opt='greedy',
        sweep_optimize_kwargs={'axes': ('x',)},
    )
    with pytest.raises(ValueError, match='axes must match'):
        driver.run(measure_infidelity=False)


@pytest.mark.parametrize('style', ['row', 'column'])
def test_strip_order_style_allows_both_gate_orientations(style):
    state, gate = inputs(shape=(2, 3))
    driver = PepsOptimizer(
        state, [(gate, ((0, 1), (1, 1))), (gate, ((0, 0), (0, 1)))], chi=2,
        update_style=style, fit_mode='direct', contraction_opt='greedy',
        sweep_optimize_kwargs={'n_round_trips': 0},
    )
    driver.run(measure_infidelity=False)
    histories = [r['optimizer_result']['sweep_history'] for r in driver.get_step_records()]
    assert {r['axis'] for history in histories for r in history} == {'x', 'y'}

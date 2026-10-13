"""Bounded retries must preserve states and leave healthy sweeps unprobed."""

import json

import autoray as ar
import numpy as np
import pytest

from pepsy.optimizers import BoundaryChiLimitError, PepsOptimizer, SweepOptimizer
from pepsy.optimizers._boundary_control import (
    BoundaryAccuracyError, BoundaryGrowth, growth_options, metric_failure,
)
from pepsy.optimizers.peps import optimizer as peps_mod
from test_peps_strip_batches import inputs


def driver(backend='numpy', **kwargs):
    state, gate = inputs(backend, (2, 3))
    kwargs.setdefault('normalize_initial', False)
    return PepsOptimizer(
        state, [(gate, ((0, 1), (1, 1)))], chi=2, contraction_opt='greedy',
        boundary_chi=(8, 16),
        sweep_kwargs={'local_contraction_opt': 'greedy'}, **kwargs,
    )


@pytest.mark.parametrize('components,expected', [
    (('norm',), (8, 12)), (('overlap',), (4, 24)),
    (('norm', 'overlap'), (8, 24)),
])
def test_independent_caps_double_and_report_without_clamping(components, expected):
    growth = BoundaryGrowth((4, 12), max_chi=(8, 24))
    failure = BoundaryAccuracyError('test_signal', components=components, stage='local_als')
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        assert growth.increase(failure, step=7, axis='y', index=2) == expected
    assert growth.events[0]['old_chi'] == (4, 12)
    assert growth.events[0]['new_chi'] == expected
    assert growth.events[0]['index'] == 2
    assert growth.events[0]['step'] == 7
    assert growth.retain((2, 2)) == expected
    with pytest.raises(BoundaryChiLimitError) as caught:
        growth.increase(failure, step=8)
    assert caught.value.report['status'] == 'limit_reached'
    assert growth.chi == expected
    json.dumps(growth.events, allow_nan=False)


def test_a_partial_doubling_is_never_silently_clamped():
    growth = BoundaryGrowth((8, 16), max_chi=(12, 32))
    with pytest.raises(BoundaryChiLimitError) as caught:
        growth.increase(BoundaryAccuracyError('bad_norm', components=('norm',)))
    assert growth.chi == (8, 16)
    assert caught.value.report['required_chi'] == (16, 16)


@pytest.mark.parametrize('result,components', [
    ({'norm': -1., 'norm_target': 1., 'overlap': .5, 'infidelity': .75}, ('norm',)),
    ({'norm': 1., 'norm_target': 1. + .1j, 'overlap': .5, 'infidelity': .75}, ('norm',)),
    ({'norm': 1., 'norm_target': 1., 'overlap': complex(float('nan')), 'infidelity': .75}, ('overlap',)),
    ({'norm': 1., 'norm_target': 1., 'overlap': 1.1, 'infidelity': -.21}, ('norm', 'overlap')),
])
def test_component_attribution_does_not_guess_from_fidelity_alone(result, components):
    failure = metric_failure(result, tolerance=1e-3, roundoff=1e-12)
    assert failure.components == components


@pytest.mark.parametrize('options', [
    {'policy': 'on_failure', 'growth': 1.5},
    {'policy': 'on_failure', 'start_chi': 4},
    {'policy': 'on_failure', 'max_chi': 0},
])
def test_policy_rejects_search_or_non_doubling_options(options):
    with pytest.raises(ValueError):
        growth_options(options)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_healthy_default_adds_no_probes_and_matches_measured_norm_fixed_caps(monkeypatch, backend):
    original = peps_mod.boundary_infidelity
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs['chi'])
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', capture)
    automatic = driver(backend)
    a = automatic.run(infidelity_tol=0.)
    auto_calls = list(calls)
    calls.clear()
    fixed = driver(backend, boundary_convergence=False)
    b = fixed.run(infidelity_tol=0., infidelity_kwargs={'norm_target': None})
    assert calls == auto_calls == [(8, 16), (8, 16)]
    np.testing.assert_allclose(ar.to_numpy(a.to_dense()), ar.to_numpy(b.to_dense()), atol=1e-10)
    assert automatic.get_boundary_growth()['events'] == []
    assert automatic._active_boundary_growth is None


@pytest.mark.parametrize('component,expected', [('norm', (16, 16)), ('overlap', (8, 32))])
def test_metric_failure_retries_only_failed_batch_and_retains_caps(monkeypatch, component, expected):
    original = peps_mod.boundary_infidelity
    calls = []
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        calls.append(kwargs['chi'])
        if not failed:
            failed = True
            return {'infidelity': .1, 'norm': -1. if component == 'norm' else 1.,
                    'norm_target': 1., 'overlap': complex(float('nan')) if component == 'overlap' else .5}
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', fail_once)
    opt = driver(boundary_convergence={'policy': 'on_failure', 'max_chi': (32, 64)})
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        output = opt.run(infidelity_tol=0.)
    assert calls == [(8, 16), expected, expected]
    report = opt.get_boundary_growth()
    assert report['chi'] == expected
    assert report['events'][0]['stage'] == 'evaluation'
    assert report['events'][0]['index'] == 1
    record, = opt.get_step_records()
    assert record['two_site_batch'] == 1
    assert record['boundary_growth']['events'] == report['events']
    assert record['normalize_chi'] == expected[0]
    assert record['effective_evaluation_chi'] == expected
    reference = driver(boundary_convergence=False)
    reference.boundary_chi = expected
    desired = reference.run(infidelity_tol=0., infidelity_kwargs={'norm_target': None})
    np.testing.assert_allclose(output.to_dense(), desired.to_dense(), atol=1e-10)
    calls.clear()
    opt.set_gates(opt.gates)
    opt.run(infidelity_tol=0.)
    assert calls and set(calls) == {expected}
    assert opt.get_boundary_growth()['chi'] == expected


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_local_als_failure_rolls_back_partial_mutation_before_retry(monkeypatch, backend):
    original = SweepOptimizer._optimize_axis_slice_with_current_env
    failures = []

    def fail_once(sweep, index, **kwargs):
        if not failures:
            failures.append(sweep.chi)
            tensor = sweep.state[0, 0]
            tensor.modify(data=tensor.data * 7)
            return {'boundary_failure': {'reason': 'invalid_norm', 'components': ('norm',)}}
        return original(sweep, index, **kwargs)

    monkeypatch.setattr(SweepOptimizer, '_optimize_axis_slice_with_current_env', fail_once)
    opt = driver(backend, boundary_convergence={'policy': 'on_failure', 'max_chi': (16, 32)})
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        output = opt.run(infidelity_tol=0.)
    assert opt.get_boundary_growth()['events'][0]['stage'] == 'local_als'
    reference = driver(backend, boundary_convergence=False)
    reference.boundary_chi = (16, 16)
    desired = reference.run(infidelity_tol=0., infidelity_kwargs={'norm_target': None})
    np.testing.assert_allclose(ar.to_numpy(output.to_dense()), ar.to_numpy(desired.to_dense()), atol=1e-10)


def test_ceiling_stops_without_accepting_or_mutating_failed_batch(monkeypatch):
    def invalid(*args, **kwargs):
        return {'infidelity': -.2, 'norm': 1., 'norm_target': 1., 'overlap': 1.1}

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', invalid)
    opt = driver(boundary_convergence={'policy': 'on_failure', 'max_chi': (16, 32)})
    before = opt.state.to_dense().copy()
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        with pytest.raises(BoundaryChiLimitError) as caught:
            opt.run(infidelity_tol=0., timing=True)
    np.testing.assert_array_equal(opt.state.to_dense(), before)
    assert opt.get_step_records() == []
    assert opt.last_result['reason'] == 'boundary_chi_limit'
    assert opt.get_timing()['status'] == 'failed'
    assert opt._active_boundary_growth is None
    assert caught.value.report['required_chi'] == (32, 64)
    assert opt.get_boundary_growth()['events'][-1]['status'] == 'limit_reached'


def test_fixed_fit_cap_is_rejected_before_any_state_change():
    opt = driver(boundary_convergence='on_failure', fit_max_bond=8)
    before = opt.state.to_dense().copy()
    with pytest.raises(ValueError, match='fit_max_bond'):
        opt.run()
    np.testing.assert_array_equal(opt.state.to_dense(), before)


def test_initial_normalization_failure_uses_same_controller_and_restores_input(monkeypatch):
    original = peps_mod.boundary_normalize
    first = True

    def fail_once(state, **kwargs):
        nonlocal first
        if first:
            first = False
            tensor = state[0, 0]
            tensor.modify(data=tensor.data * 7)
            return -1.
        return original(state, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_normalize', fail_once)
    opt = driver(normalize_initial=True, boundary_convergence='on_failure')
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        result = opt.run(infidelity_tol=0.)
    event, = opt.get_boundary_growth()['events']
    assert event['step'] == 0 and event['stage'] == 'normalization'
    assert event['components'] == ('norm',)
    reference = driver(normalize_initial=True, boundary_convergence=False)
    reference.boundary_chi = (16, 16)
    desired = reference.run(infidelity_tol=0., infidelity_kwargs={'norm_target': None})
    np.testing.assert_allclose(result.to_dense(), desired.to_dense(), atol=1e-10)


def test_postcheck_growth_refits_and_remeasures_both_candidates_at_same_caps(monkeypatch):
    original = peps_mod.boundary_infidelity
    calls = []

    def fail_second(*args, **kwargs):
        calls.append((kwargs['chi'], kwargs['norm_target']))
        if len(calls) == 2:
            return {'infidelity': .1, 'norm': -1., 'norm_target': 1., 'overlap': .5}
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', fail_second)
    opt = driver(boundary_convergence='on_failure')
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        result = opt.run(infidelity_tol=0.)
    assert [cap for cap, _ in calls] == [(8, 16), (8, 16), (16, 16), (16, 16)]
    assert calls[0][1] is calls[2][1] is None
    assert calls[1][1] is not None and calls[3][1] is not None
    reference = driver(boundary_convergence=False)
    reference.boundary_chi = (16, 16)
    desired = reference.run(infidelity_tol=0., infidelity_kwargs={'norm_target': None})
    np.testing.assert_allclose(result.to_dense(), desired.to_dense(), atol=1e-10)


def test_ceiling_preserves_gates_accepted_before_failed_batch(monkeypatch):
    opt = driver(boundary_convergence={'policy': 'on_failure', 'max_chi': (8, 16)})
    onsite = np.diag([1j, -1j])
    expected = opt.state.copy().gate_(onsite, (0, 0), contract=True).to_dense()
    opt.set_gates([(onsite, (0, 0))] + opt.gates)
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw: {
        'infidelity': -.2, 'norm': 1., 'norm_target': 1., 'overlap': 1.1})
    with pytest.raises(BoundaryChiLimitError):
        opt.run()
    np.testing.assert_allclose(opt.state.to_dense(), expected, atol=1e-12)
    assert opt.last_result['step'] == 2


def test_exact_readout_keeps_explicit_none_cap(monkeypatch):
    original = peps_mod.boundary_infidelity
    calls = []

    def capture(*args, **kwargs):
        calls.append((kwargs['method'], kwargs['chi']))
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', capture)
    opt = driver(boundary_convergence='on_failure', infidelity_kwargs={'method': 'exact', 'chi': None})
    opt.run(infidelity_tol=0.)
    assert calls and set(calls) == {('exact', None)}
    assert opt.get_boundary_growth()['events'] == []


def test_exact_normalization_preserves_uncapped_policy(monkeypatch):
    original = peps_mod.boundary_normalize
    calls = []

    def capture(*args, **kwargs):
        calls.append((kwargs['method'], kwargs['chi']))
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_normalize', capture)
    opt = driver(normalize_initial=True, boundary_convergence='on_failure',
                 normalize_kwargs={'method': 'exact', 'chi': None})
    opt.run(infidelity_tol=0.)
    assert calls and set(calls) == {('exact', None)}
    assert opt.get_boundary_growth()['events'] == []


def test_unmonitored_target_norm_failure_retries_without_overlap_probe(monkeypatch):
    original = peps_mod.boundary_norm
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(kwargs['chi'])
        return float('nan') if len(calls) == 1 else original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_norm', fail_once)
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw: pytest.fail(
        'Disabled fidelity monitoring must not add an overlap probe'))
    opt = driver(boundary_convergence='on_failure')
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        result = opt.run(measure_infidelity=False, measure_final_infidelity=False)
    assert calls == [8, 16]
    assert opt.get_boundary_growth()['events'][0]['stage'] == 'target_norm'
    reference = driver(boundary_convergence=False)
    reference.boundary_chi = (16, 16)
    desired = reference.run(measure_infidelity=False, measure_final_infidelity=False,
                            infidelity_kwargs={'norm_target': None})
    np.testing.assert_allclose(result.to_dense(), desired.to_dense(), atol=1e-10)


@pytest.mark.parametrize('location', ['normalize', 'infidelity', 'run_normalize', 'run_infidelity'])
def test_hidden_fixed_metric_fit_cap_is_rejected_before_mutation(location):
    option = {'fit_max_bond': 8}
    constructor = {} if location.startswith('run_') else {location + '_kwargs': option}
    run = {location.removeprefix('run_') + '_kwargs': option} if location.startswith('run_') else {}
    opt = driver(boundary_convergence='on_failure', **constructor)
    before = opt.state.to_dense().copy()
    with pytest.raises(ValueError, match='fit_max_bond'):
        opt.run(**run)
    np.testing.assert_array_equal(opt.state.to_dense(), before)


def test_onsite_final_normalization_retry_applies_gate_once(monkeypatch):
    original = peps_mod.boundary_normalize
    calls = []

    def fail_once(state, **kwargs):
        calls.append(kwargs['chi'])
        if len(calls) == 1:
            state[0, 0].modify(data=state[0, 0].data * 7)
            return -1.
        return original(state, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_normalize', fail_once)
    onsite = np.diag([1j, -1j])
    opt = driver(boundary_convergence='on_failure')
    opt.set_gates([(onsite, (0, 0))])
    desired = opt.state.copy().gate_(onsite, (0, 0), contract=True).to_dense()
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        output = opt.run()
    assert calls == [8, 16]
    np.testing.assert_allclose(output.to_dense(), desired, atol=1e-10)
    assert opt.get_boundary_growth()['events'][0]['step'] == 1


@pytest.mark.parametrize('infidelity', [0., .8, -1e-13])
def test_admissible_fidelity_and_roundoff_do_not_request_growth(monkeypatch, infidelity):
    def admissible(*args, **kwargs):
        return {'infidelity': infidelity, 'norm': 1., 'norm_target': 1.,
                'overlap': np.sqrt(max(0., 1. - infidelity))}

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', admissible)
    opt = driver(boundary_convergence='on_failure')
    opt.run(infidelity_tol=0.)
    assert opt.get_boundary_growth()['chi'] == (8, 16)
    assert opt.get_boundary_growth()['events'] == []


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_real_truncated_boundaries_recover_to_same_fixed_cap_fit(backend):
    state, gate = inputs(backend, (3, 3))
    gates = [(gate, ((0, 1), (1, 1)))]
    common = dict(chi=2, contraction_opt='greedy', normalize_initial=False,
                  sweep_kwargs={'local_contraction_opt': 'greedy'})
    adaptive = PepsOptimizer(
        state, gates, boundary_chi=(1, 2), normalize_chi=1, evaluation_chi=(1, 2),
        boundary_convergence={'policy': 'on_failure', 'max_chi': (64, 128)}, **common,
    )
    before = ar.to_numpy(state.to_dense()).copy()
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        output = adaptive.run(infidelity_tol=0.)
    report = adaptive.get_boundary_growth()
    assert report['events'] and report['chi'][0] > 1 and report['chi'][1] > 2
    fixed = PepsOptimizer(
        state, gates, boundary_chi=report['chi'], normalize_chi=report['chi'][0],
        evaluation_chi=report['chi'], boundary_convergence=False,
        infidelity_kwargs={'norm_target': None}, **common,
    )
    expected = fixed.run(infidelity_tol=0.)
    np.testing.assert_allclose(ar.to_numpy(output.to_dense()), ar.to_numpy(expected.to_dense()),
                               atol=1e-10)
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)
    record, = adaptive.get_step_records()
    assert record['reason'] == 'optimized'
    assert 0 <= record['final_infidelity'] <= 1


def test_requested_invalid_final_diagnostic_retries_even_with_finite_best(monkeypatch):
    original = SweepOptimizer._debug_loss
    first = True

    def fail_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            return float('nan')
        return original(*args, **kwargs)

    monkeypatch.setattr(SweepOptimizer, '_debug_loss', fail_once)
    opt = driver(boundary_convergence='on_failure',
                 sweep_optimize_kwargs={'compute_final_loss': True})
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        opt.run(infidelity_tol=0.)
    event, = opt.get_boundary_growth()['events']
    assert event['stage'] == 'sweep_diagnostic'
    assert event['diagnostic'] == 'raw_loss_after'
    assert opt.get_step_records()[0]['optimizer_result']['success']


@pytest.mark.parametrize('location', ['sweep', 'normalize', 'infidelity'])
def test_supplied_boundary_handles_rejected_before_state_change(location):
    opt = driver(boundary_convergence='on_failure')
    before = opt.state.to_dense().copy()
    with pytest.raises(ValueError, match='fresh boundary environments'):
        opt.run(**{location + '_kwargs': {'bdy': object()}})
    np.testing.assert_array_equal(opt.state.to_dense(), before)

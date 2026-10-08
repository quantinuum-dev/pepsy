"""Adaptive caps must test actual contractions, preserve state, and remain bounded."""

import json
import math

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps import optimizer as peps_mod
from pepsy.optimizers.peps._boundary_convergence import (
    compare_samples, convergence_options, sample_validity, select_boundary_chi,
)
from pepsy.tensors.contractions import build_optimizer


def sample(norm=1., target=1., overlap=.8):
    return {"norm": (norm, 0.), "norm_target": (target, 0.), "overlap": (overlap, 0.)}


def select(measure, minimum=(2, 2), retained=None, **opts):
    opts = {"schedule": "geometric", "patience": 1, "warm_start": False, **opts}
    return select_boundary_chi(measure, minimum=minimum, retained=retained,
                               options=convergence_options(opts), roundoff=1e-12)


def test_constant_fidelity_cannot_hide_unconverged_norms():
    calls = []

    def measure(cap, axis):
        calls.append((cap, axis))
        n = 1. + 1. / cap[0]
        return sample(n, n, .8 * n)  # Fidelity is exactly .64 at every cap.

    result = select(measure, max_chi=16, rtol=1e-5)
    assert not result["converged"]
    assert result["chi"] == (16, 16)
    assert [c for c, axis in calls if axis == "x"] == [(2, 2), (4, 4), (8, 8), (16, 16)]


def test_retained_cap_is_rechecked_without_automatic_growth():
    calls = []

    def measure(cap, axis):
        calls.append(cap)
        return sample(norm=1. + 1. / cap[0] if cap[0] < 8 else 1.)

    first = select(measure, max_chi=32)
    assert first["chi"] == (16, 16) and first["converged"]
    calls.clear()
    second = select(measure, retained=first["chi"], max_chi=32)
    assert second["chi"] == first["chi"] and second["converged"]
    assert calls == [(8, 8), (8, 8), (16, 16), (16, 16)]


def test_directional_disagreement_and_unphysical_ratios_do_not_converge():
    assert not select(lambda cap, axis: sample(overlap=.7 if axis == "x" else .8),
                      max_chi=8)["converged"]
    assert not select(lambda cap, axis: sample(overlap=1.01), max_chi=8)["converged"]


def test_small_imaginary_norm_error_uses_requested_accuracy_not_only_roundoff():
    options = convergence_options(True, bond_dim=4)
    values = sample(target=1. + 1.734e-9j)
    result = select_boundary_chi(lambda cap, axis: values, minimum=(16, 16),
                                 retained=None, options=options, roundoff=1e-12,
                                 confirm=lambda cap, axis: values)
    assert result['converged'] and result['chi'] == (48, 48)
    validity = result['attempts'][-1]['validity']['y']
    assert validity['norm_imaginary_threshold'] == 1e-5
    assert validity['norm_imaginary_relative']['norm_target'] == pytest.approx(1.734e-9)
    assert result['sample']['norm_target'][0].imag == 1.734e-9
    assert not sample_validity(values, roundoff=1e-12, norm_rtol=1e-10)['valid']


@pytest.mark.parametrize('values,reason', [
    (sample(target=1. + 1e-3j), 'norm_target_imaginary_residual'),
    (sample(norm=-1.), 'norm_nonpositive'),
    (sample(norm=float('nan')), 'norm_nonfinite'),
    (sample(overlap=1.000001), 'fidelity_above_one'),
])
def test_relaxed_norm_reality_preserves_other_validity_guards(values, reason):
    report = sample_validity(values, roundoff=1e-12, norm_rtol=1e-5)
    assert not report['valid'] and reason in report['reasons']
    json.dumps(report, allow_nan=False)


def test_norm_reality_tolerance_keeps_dtype_roundoff_floor():
    report = sample_validity(sample(norm=1. + 1e-7j), roundoff=1e-6, norm_rtol=1e-8)
    assert report['valid'] and report['norm_imaginary_threshold'] == 1e-6


def test_scaled_complex_overlap_and_zero_overlap():
    a = sample(overlap=.5j)
    b = {k: (v[0] * .1, v[1] + 1) for k, v in a.items()}
    assert compare_samples(a, b, rtol=1e-10, atol=1e-12)[:2] == (True, True)
    # State amplitudes can carry large exponents without overflowing a probe.
    huge = {k: (v[0], 800.) for k, v in a.items()}
    assert select(lambda cap, axis: huge, max_chi=4)["converged"]
    assert select(lambda cap, axis: sample(overlap=0.), max_chi=4)["converged"]
    changed_phase = sample(overlap=-.5j)
    assert not compare_samples(a, changed_phase, rtol=1e-5, atol=1e-8)[1]


@pytest.mark.parametrize("option", [{"growth": 1}, {"rtol": -1}, {"atol": float('nan')},
                                    {"max_chi": 0}, {"surprise": 1}])
def test_invalid_options(option):
    with pytest.raises(ValueError):
        convergence_options(option)


def optimizer(**kwargs):
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=7)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    if kwargs.get('boundary_convergence', True) is not False:
        kwargs['boundary_convergence'] = {"schedule": "geometric", "patience": 1,
                                         "warm_start": False, "max_chi": 32,
                                         **(kwargs.get('boundary_convergence')
                                            if isinstance(kwargs.get('boundary_convergence'), dict) else {})}
    return PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=1, boundary_chi=2,
        normalize_chi=2, evaluation_chi=2, contraction_opt="greedy", **kwargs,
    )


def test_default_check_runs_without_readout_and_reuses_cap_and_target_norm(monkeypatch):
    calls, refinements = [], []

    def metric(*args, **kwargs):
        assert kwargs['norm'] is kwargs['norm_target'] is None
        assert kwargs['bdy'] is kwargs['bdy_overlap'] is None
        calls.append((kwargs['chi'], kwargs['direction']))
        return sample_to_metric(sample(norm=1.2 if kwargs['chi'][0] < 4 else 1., target=2.))

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', metric)
    opt = optimizer()

    def refine(state, target, **kwargs):
        refinements.append(kwargs)
        return state, .1, {'success': True}

    monkeypatch.setattr(opt, '_optimize_state', refine)
    opt.run(measure_infidelity=False, measure_final_infidelity=False, normalize_final=False)
    assert refinements[0]['sweep_kwargs']['chi'] == (8, 8)
    assert refinements[0]['normalize_chi'] == 8
    assert refinements[0]['target_norm'] == (2., 0.)
    assert opt.get_step_records()[0]['boundary_convergence']['converged']
    assert opt.get_step_records()[0]['boundary_convergence']['selection_status'] == 'converged'
    json.dumps(opt.get_step_records(), allow_nan=False)
    calls.clear()
    opt.run(measure_infidelity=False, measure_final_infidelity=False, normalize_final=False)
    assert calls == [((4, 4), 'x'), ((4, 4), 'y'), ((8, 8), 'x'), ((8, 8), 'y')]
    assert opt._boundary_chi_floor == (8, 8)  # Survives reset_traces=True.


def sample_to_metric(s):
    return {**s, 'infidelity': .5}


def test_nonconvergence_warns_and_uses_limit_without_claiming_convergence(monkeypatch):
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw:
                        sample_to_metric(sample(norm=1. + 1. / kw['chi'][0])))
    opt = optimizer(boundary_convergence={'max_chi': 4})
    used = []
    monkeypatch.setattr(opt, '_optimize_state', lambda state, target, **kw:
                        (used.append(kw) or state, .1, {'success': True}))
    with pytest.warns(RuntimeWarning, match='convergence was not established'):
        opt.run(measure_infidelity=False, measure_final_infidelity=False, normalize_final=False)
    assert used[0]['sweep_kwargs']['chi'] == (4, 4)
    assert not opt.get_step_records()[0]['boundary_convergence']['converged']
    assert opt.get_step_records()[0]['boundary_convergence']['selection_status'] == 'limit_unconverged'
    assert opt.get_step_records()[0]['boundary_convergence']['at_ceiling']


def test_precheck_reuses_calibrated_contractions(monkeypatch):
    calls = []

    def metric(*args, **kwargs):
        calls.append(kwargs['chi'])
        return sample_to_metric(sample())

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', metric)
    opt = optimizer()
    monkeypatch.setattr(opt, '_optimize_state', lambda state, target, **kw:
                        (state, .1, {'success': True}))
    opt.run(measure_final_infidelity=False, normalize_final=False)
    assert calls == [(2, 2), (2, 2), (4, 4), (4, 4)]
    assert opt.get_step_records()[0]['pre_infidelity'] == pytest.approx(.36)
    assert opt.get_evaluation_records()[0]['source'] == 'boundary_convergence'
    assert opt.get_boundary_convergence()['chi_floor'] == (4, 4)


def test_unusable_norm_never_enters_fit_and_diagnostics_are_serializable(monkeypatch):
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw:
                        sample_to_metric(sample(norm=float('nan'))))
    opt = optimizer(boundary_convergence={'max_chi': 4})
    monkeypatch.setattr(opt, '_optimize_state', lambda *a, **kw: pytest.fail('invalid norm entered fit'))
    with pytest.warns(RuntimeWarning, match='convergence was not established'):
        with pytest.raises(ValueError, match='unusable norm'):
            opt.run(measure_infidelity=False)
    json.dumps(opt.get_boundary_convergence(), allow_nan=False)


def test_explicit_normalization_policy_is_preserved(monkeypatch):
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw: sample_to_metric(sample()))
    opt = optimizer(normalize_initial=False, normalize_kwargs={'method': 'exact'})
    calls = []
    normalize = opt._normalize_state

    def capture(state, **kwargs):
        calls.append(kwargs)
        return normalize(state, **kwargs)

    monkeypatch.setattr(opt, '_normalize_state', capture)
    monkeypatch.setattr(opt, '_optimize_state', lambda state, target, **kw:
                        (state, .1, {'success': True}))
    opt.run(measure_infidelity=False, measure_final_infidelity=False, normalize_final=False)
    assert len(calls) == 1 and calls[0]['normalize_chi'] == 4


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_real_torch_sweep_uses_adaptive_caps_and_normalizes_output(dtype):
    torch = pytest.importorskip('torch')
    policy = build_optimizer(max_repeats=2, parallel=False)
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype=dtype, seed=19)
    state.apply_to_arrays(lambda a: torch.tensor(a, dtype=getattr(torch, dtype)))
    gate = torch.diag(torch.exp(torch.tensor([-.2j, .2j, .2j, -.2j], dtype=getattr(torch, dtype))))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1,
                        boundary_chi=2, normalize_chi=2, evaluation_chi=2,
                        contraction_opt=policy, fit_mode='direct',
                        boundary_convergence={'max_chi': 16}, optimizer='scipy',
                        optimizer_options={'n_steps': 3},
                        sweep_optimize_kwargs={'n_round_trips': 0,
                                               'compute_initial_loss': False, 'compute_final_loss': False})
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False)
    record = opt.get_step_records()[0]
    assert record['boundary_convergence']['converged']
    assert record['optimizer_attempted']
    assert next(iter(out)).data.dtype == getattr(torch, dtype)
    vector = out.to_dense(optimize=policy).reshape(-1)
    assert float(torch.linalg.vector_norm(vector)) == pytest.approx(1., abs=2e-5)
    assert torch.isfinite(vector).all()


@pytest.mark.parametrize('mode,enabled', [('global', True), ('sweep', False)])
def test_legacy_and_global_paths_do_not_calibrate(monkeypatch, mode, enabled):
    opt = optimizer(mode=mode, boundary_convergence=enabled)
    monkeypatch.setattr(opt, '_calibrate_sweep_boundaries', lambda *a, **kw:
                        pytest.fail('unexpected calibration'))
    monkeypatch.setattr(opt, '_optimize_state', lambda state, target, **kw:
                        (state, .1, {'success': True}))
    opt.run(measure_infidelity=False, measure_final_infidelity=False, normalize_final=False)


@pytest.mark.parametrize('fit_mode,engine', [('direct', 'dmrg'), ('eff', 'dmrg'),
                                           ('direct', 'quimb-mps')])
def test_actual_norms_and_overlap_converge_to_exact_reference(fit_mode, engine):
    policy = build_optimizer(max_repeats=2, parallel=False)
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype='complex128', seed=317)
    target = state.copy(deep=True)
    target.gate_(np.diag(np.exp(-.21j * np.array([1., -1., -1., 1.]))),
                 ((1, 1), (1, 2)), contract='split', cutoff=0.)
    original = [t.data.copy() for t in state]
    opt = PepsOptimizer(state, chi=2, boundary_chi=2, normalize_initial=False,
                        contraction_opt=policy, fit_mode=fit_mode,
                        boundary_engine=engine,
                        boundary_convergence={'max_chi': 64, 'rtol': 1e-9, 'atol': 1e-10})
    result = opt._calibrate_sweep_boundaries(state, target, normalize_chi=2, evaluation_chi=2)
    assert result['converged']
    if fit_mode == 'eff':
        assert any(a['warm_started']['x'] for a in result['record']['attempts'])
        assert result['record']['attempts'][-1]['kind'] == 'confirmation'
        assert not result['record']['attempts'][-1]['warm_started']['x']
        assert result['record']['attempts'][-1]['actual_boundary_bonds']['x']
    v = state.to_dense(optimize=policy).reshape(-1)
    w = target.to_dense(optimize=policy).reshape(-1)
    expected = {'norm': np.vdot(v, v), 'norm_target': np.vdot(w, w), 'overlap': np.vdot(w, v)}
    for key in ('norm', 'norm_target', 'overlap'):
        value = result['sample'][key]
        actual = complex(value[0]) * 10. ** value[1]
        assert actual == pytest.approx(expected[key], rel=1e-8, abs=1e-8)
        assert math.isfinite(abs(actual))
    for t, before in zip(state, original):
        np.testing.assert_array_equal(t.data, before)


@pytest.mark.parametrize('D,expected', [(2, [4, 8, 12]), (4, [16, 32, 48])])
def test_d_squared_schedule_needs_two_stable_increases_and_rechecks_floor(D, expected):
    options = convergence_options(True, bond_dim=D)
    assert options['max_chi'] == (8 * D**2,) * 2
    calls = []

    def measure(cap, axis):
        if axis == 'x':
            calls.append(cap[0])
        return sample()

    for retained in (None, (expected[-1],) * 2):
        calls.clear()
        result = select_boundary_chi(measure, minimum=options['start_chi'],
                                     retained=retained, options=options, roundoff=1e-12)
        assert result['converged'] and result['chi'] == (expected[-1],) * 2
        assert calls == expected
        assert result['stop_reason'] == 'threshold_satisfied'


def test_warm_false_plateau_rejected_by_fresh_confirmation():
    options = convergence_options(True, bond_dim=2)
    result = select_boundary_chi(lambda cap, axis: sample(), minimum=(4, 4),
                                 retained=None, options=options, roundoff=1e-12,
                                 confirm=lambda cap, axis: sample(norm=2.))
    assert not result['converged'] and result['chi'] == (32, 32)
    confirmations = [a for a in result['attempts'] if a['kind'] == 'confirmation']
    assert confirmations and not confirmations[0]['norms_pass']
    assert result['stop_reason'] == 'chi_limit'
    assert result['sample']['norm'] == (2., 0.)  # Fresh value at the limit.


def test_short_plateau_does_not_stop_at_second_cap():
    options = convergence_options(True, bond_dim=2)
    result = select_boundary_chi(lambda cap, axis: sample(norm=1. if cap[0] <= 8 else 2.),
                                 minimum=(4, 4), retained=None, options=options, roundoff=1e-12)
    assert result['chi'] == (20, 20)
    assert result['converged']


def test_d2_defaults_override_legacy_caps_and_output_uses_selected_norm_chi(monkeypatch):
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype='complex128', seed=7)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1,
                        boundary_chi=128, normalize_chi=128, evaluation_chi=128,
                        contraction_opt='greedy', fit_mode='direct',
                        sweep_optimize_kwargs={'chi': (128, 128)})
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw: sample_to_metric(sample()))
    used = []
    monkeypatch.setattr(opt, '_optimize_state', lambda state, target, **kw:
                        (used.append(kw) or state, .1, {'success': True}))
    normalizations = []
    normalize = opt._normalize_state

    def capture(state, **kw):
        normalizations.append(kw['normalize_chi'])
        return normalize(state, **kw)

    monkeypatch.setattr(opt, '_normalize_state', capture)
    opt.run(measure_infidelity=False, measure_final_infidelity=False)
    assert used[0]['sweep_kwargs']['chi'] == (3, 3)
    assert used[0]['sweep_optimize_kwargs']['chi'] is None
    assert used[0]['normalize_chi'] == 3
    assert normalizations[-1] == 3
    assert opt.get_step_records()[0]['boundary_convergence']['bond_dim'] == 1

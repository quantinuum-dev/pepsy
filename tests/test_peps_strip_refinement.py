"""Strip ALS must improve a fixed exact target, preserve locality, and stay native."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._strip_update import refine_strip


def fixture(backend='torch', dtype='complex128', axis='x'):
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('CUDA unavailable')
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
        convert = cp.asarray
    else:
        convert = pytest.importorskip('torch').tensor
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype=dtype, seed=24)
    state.apply_to_arrays(convert)
    state /= state.norm()
    gate = convert(np.diag(np.exp(np.array([-.3j, .3j, .3j, -.3j]))).astype(dtype))
    where = ((1, 0), (1, 1)) if axis == 'x' else ((0, 1), (1, 1))
    return state, gate, where


def optimizer(state, gates, **kwargs):
    return PepsOptimizer(state, gates, chi=2, mode='full-update', contraction_opt='greedy',
                         fit_mode='direct', boundary_kwargs={'cutoff': 0.}, boundary_chi=32,
                         normalize_chi=32, boundary_convergence=False,
                         full_update_kwargs={'max_iterations': 4, **kwargs})


def run(opt, **kwargs):
    return opt.run(measure_infidelity=False, measure_final_infidelity=False,
                   accept_if_improved=False, **kwargs)


def fidelity(a, b):
    x, y = (ar.to_numpy(t.to_dense()).ravel() for t in (a, b))
    return abs(np.vdot(x, y))**2 / (np.vdot(x, x).real * np.vdot(y, y).real)


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('axis', ['x', 'y'])
def test_strip_refinement_improves_dense_reference_and_preserves_exterior(backend, dtype, axis, monkeypatch):
    state, gate, where = fixture(backend, dtype, axis)
    target = state.gate(gate, where, contract='split', cutoff=0.)
    # Different virtual names in the exact target must not break the cross metric.
    target.reindex_({ind: qtn.rand_uuid() for ind in target.inner_inds()})
    baseline = run(optimizer(state, [(gate, where)]))
    before = fidelity(baseline, target)
    with monkeypatch.context() as patch:
        original = ar.to_numpy
        def scalar_only(value):
            assert math_size(value) <= 1, 'bulk tensor transferred to NumPy during refinement'
            return original(value)
        patch.setattr(ar, 'to_numpy', scalar_only)
        out, report, _, _ = refine_strip(
            baseline, target, key=(axis, 1), chi=32, contraction_opt='greedy',
            boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, sweeps=2, rtol='auto',
        )
    after = fidelity(out, target)
    tolerance = 3e-5 if dtype == 'complex64' else 1e-9
    assert report['accepted']
    assert after > before + 1e-4
    assert report['infidelity_before'] == pytest.approx(1-before, abs=tolerance)
    assert report['infidelity_after'] == pytest.approx(1-after, abs=tolerance)
    assert report['strip_environment_reuse']['norm']['hits'] > 0
    assert report['strip_environment_reuse']['overlap']['hits'] > 0
    changed = []
    for site in state.gen_site_coos():
        assert out[site].inds == baseline[site].inds
        assert out[site].tags == baseline[site].tags
        assert ar.infer_backend(out[site].data) == backend
        assert out[site].data.dtype == baseline[site].data.dtype
        assert out[site].data.device == baseline[site].data.device
        if site[0 if axis == 'x' else 1] != 1:
            assert out[site].data is baseline[site].data
        elif out[site].data is not baseline[site].data:
            changed.append(site)
    assert len(changed) == 3 and set(changed) - set(where)
    assert out.max_bond() == baseline.max_bond()


def math_size(value):
    return np.prod(getattr(value, 'shape', ()))


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('gate_kind', ['numpy', 'native_complex64'])
def test_refinement_accepts_gate_backend_and_dtype_conversion(backend, gate_kind):
    state, gate, where = fixture(backend=backend)
    if gate_kind == 'numpy':
        supplied = ar.to_numpy(gate)
    else:
        supplied = ar.do('astype', gate, 'complex64')
    if backend == 'torch':
        import torch
        native = torch.as_tensor(supplied, dtype=gate.dtype, device=gate.device)
    else:
        import cupy as cp
        native = cp.asarray(supplied, dtype=gate.dtype)
    expected = run(optimizer(state, [(native, where)], refine_sweeps=1))
    opt = optimizer(state, [(supplied, where)], refine_sweeps=1)
    actual = run(opt)
    assert opt.get_step_records()[0]['strip_refinement']['accepted']
    np.testing.assert_allclose(ar.to_numpy(actual.to_dense()), ar.to_numpy(expected.to_dense()),
                               rtol=1e-10, atol=1e-11)
    for tensor in actual:
        assert ar.infer_backend(tensor.data) == backend
        assert tensor.data.dtype == gate.dtype
        assert tensor.data.device == gate.device


@pytest.mark.parametrize('shift', [0., 80.])
def test_cached_and_fresh_refinement_agree_with_large_network_exponents(shift):
    state, gate, where = fixture()
    target = state.gate(gate, where, contract='split', cutoff=0.)
    baseline = run(optimizer(state, [(gate, where)]))
    baseline.exponent += shift
    target.exponent += shift
    results = []
    for reuse in (False, True):
        out, report, _, _ = refine_strip(
            baseline, target, key=('x', 1), chi=32, contraction_opt='greedy',
            boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, sweeps=2, rtol=None, reuse=reuse,
        )
        assert report['accepted'] and report['sweeps'] == 2
        # For an exact metric, the least-squares projection satisfies
        # <candidate|candidate> = <candidate|target>, including physical scale.
        a, b = (ar.to_numpy(t.to_dense()).ravel() / 10.**shift for t in (out, target))
        np.testing.assert_allclose(np.vdot(a, a), np.vdot(a, b), rtol=1e-8, atol=1e-10)
        results.append(a)
    np.testing.assert_allclose(*results, rtol=1e-8, atol=1e-10)


def test_invalid_refinement_rolls_back_without_mutating_input(monkeypatch):
    state, gate, where = fixture()
    target = state.gate(gate, where, contract='split', cutoff=0.)
    arrays = [t.data for t in state]
    def invalid(*args, **kwargs):
        tensor = kwargs['tnAA']['__KET__', '__VAR0__']
        tensor.modify(data=tensor.data * float('nan'))
    monkeypatch.setattr(qtn, 'tensor_network_fit_als', invalid)
    out, report, _, _ = refine_strip(
        state, target, key=('x', 1), chi=32, contraction_opt='greedy',
        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, sweeps=1, rtol='auto',
    )
    assert out is state and not report['accepted']
    assert report['termination_reason'] == 'invalid_candidate'
    assert all(t.data is a for t, a in zip(state, arrays))


@pytest.mark.parametrize('bad_value,call_index', [(1. + .05j, 1), (1e200, 4), (complex(1., float('nan')), 3)])
def test_invalid_strip_metrics_are_rejected_independently_of_stopping_tolerance(
    monkeypatch, bad_value, call_index,
):
    from pepsy.optimizers.peps import _strip_update
    state, gate, where = fixture()
    target = state.gate(gate, where, contract='split', cutoff=0.)
    scalar, calls = _strip_update._scalar, 0
    def invalid_metric(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (bad_value, 0.) if calls == call_index else scalar(*args, **kwargs)
    monkeypatch.setattr(_strip_update, '_scalar', invalid_metric)
    out, report, _, _ = refine_strip(
        state, target, key=('x', 1), chi=32, contraction_opt='greedy',
        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, sweeps=1, rtol=.1,
    )
    assert out is state and not report['accepted']
    assert report['termination_reason'] == 'invalid_environment'
    assert report['sweeps'] == 0


def test_driver_retains_exact_block_target_and_closes_at_repeat_or_barrier(monkeypatch):
    state, gate, where = fixture()
    x = state[0, 0].data.new_tensor([[0., 1.], [1., 0.]])
    second = ((1, 1), (1, 2))
    gates = [(gate, where), (gate, second), (gate, where), (x, (0, 0)), (gate, second)]
    opt = optimizer(state, gates, refine_sweeps=1)
    exact = state.gate(gate, where, contract='split', cutoff=0.).gate(gate, second, contract='split', cutoff=0.)
    refine, targets = opt._full_update_refine, []
    def track(target, *args, **kwargs):
        targets.append(target.copy())
        return refine(target, *args, **kwargs)
    monkeypatch.setattr(opt, '_full_update_refine', track)
    out = run(opt)
    assert fidelity(targets[0], exact) == pytest.approx(1., abs=1e-12)
    records = opt.get_step_records()
    reports = [r['strip_refinement'] for r in records if r['strip_refinement'] is not None]
    assert [(r['start_step'], r['end_step']) for r in reports] == [(1, 2), (3, 3), (5, 5)]
    assert all(r['target_max_bond'] <= 4 * opt.chi for r in reports)
    assert float(out.norm().real) == pytest.approx(1., abs=1e-10)
    assert all(r['final_infidelity'] is None for r in records
               if r['strip_refinement'] and r['strip_refinement']['accepted'])
    assert len(records) == 4
    product = 1.
    for count, record in enumerate(records, 1):
        product *= record['local_fidelity']
        assert record['local_fidelity'] == record['optimizer_result']['fidelity']
        assert record['accumulated_local_infidelity'] == pytest.approx(1. - product, abs=1e-15)
        assert record['accumulated_local_gate_count'] == count


def test_local_gate_fidelity_survives_rollback_without_extra_metric_calls(monkeypatch):
    state, gate, where = fixture()
    opt = optimizer(state, [(gate, where)])
    calls = []
    def metric(*args, **kwargs):
        value = .01 if not calls else .02
        calls.append(value)
        opt._last_evaluation_raw_infidelity = value
        opt._last_evaluation_chi = 32
        return value
    monkeypatch.setattr(opt, 'estimate_infidelity', metric)
    opt.run(measure_infidelity=True, measure_final_infidelity=True, accept_if_improved=True)
    record, = opt.get_step_records()
    assert not record['optimized'] and len(calls) == 2
    assert record['local_fidelity'] == record['optimizer_result']['warmstart_fidelity']
    target = state.gate(gate, where, contract='split', cutoff=0.)
    assert record['local_fidelity'] == pytest.approx(fidelity(opt.state, target), abs=1e-9)


def test_local_gate_product_is_stable_optional_and_respects_trace_reset(monkeypatch):
    state, gate, where = fixture()
    opt = optimizer(state, [(gate, where)])
    def forbidden(*args, **kwargs):
        raise AssertionError('local gate recording must not contract an extra global metric')
    monkeypatch.setattr(opt, 'estimate_infidelity', forbidden)
    run(opt)
    first = opt.get_step_records()[0]['local_fidelity']
    run(opt, reset_traces=False)
    last = opt.get_step_records()[-1]
    assert last['accumulated_local_gate_count'] == 2
    assert last['accumulated_local_infidelity'] == pytest.approx(1. - first * last['local_fidelity'])
    run(opt, reset_traces=True)
    assert opt.get_step_records()[0]['accumulated_local_gate_count'] == 1
    opt._reset_traces()
    f = 1. - 1e-12
    for _ in range(10000):
        record = opt._record_local_gate_fidelity(f)
    assert record['accumulated_local_infidelity'] == pytest.approx(-np.expm1(10000 * np.log(f)), rel=1e-12)
    assert opt._record_local_gate_fidelity(0.)['accumulated_local_infidelity'] == 1.
    assert opt._record_local_gate_fidelity(1.)['accumulated_local_infidelity'] == 1.
    opt.full_update_kwargs['accumulate_local_infidelity'] = False
    run(opt)
    record, = opt.get_step_records()
    assert record['local_fidelity'] is not None and record['accumulated_local_infidelity'] is None
    assert record['accumulated_local_gate_count'] == 0


def test_nonunitary_strip_refinement_with_adaptive_boundaries_and_final_measurement():
    state, _, where = fixture()
    gate = state[0, 0].data.new_tensor(np.diag(np.exp([.15, -.15, -.15, .15])))
    opt = PepsOptimizer(
        state, [(gate, where)], chi=2, mode='full-update', contraction_opt='greedy',
        fit_mode='direct', boundary_kwargs={'cutoff': 0.}, boundary_chi=16,
        normalize_chi=16, evaluation_chi=16, boundary_convergence={'max_chi': 32},
        full_update_kwargs={'max_iterations': 4, 'refine_sweeps': 1},
    )
    out = opt.run(non_unitary=True, measure_infidelity=True, measure_final_infidelity=True,
                  accept_if_improved=False)
    record, = opt.get_step_records()
    target = state.gate(gate, where, contract='split', cutoff=0.)
    report = record['strip_refinement']
    assert report['accepted'] and report['boundary_convergence']['converged']
    assert float(out.norm().real) == pytest.approx(1., abs=1e-9)
    assert record['final_infidelity'] == pytest.approx(1. - fidelity(out, target), abs=1e-8)


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('adaptive', [False, True])
def test_refinement_honors_separate_caps_and_checked_handles(monkeypatch, backend, adaptive):
    from pepsy.optimizers.peps import _strip_update

    state, gate, where = fixture(backend=backend)
    opt = PepsOptimizer(
        state, [(gate, where)], chi=2, mode='full-update', contraction_opt='greedy',
        fit_mode='direct', boundary_kwargs={'cutoff': 0.}, normalize_initial=False,
        boundary_chi=(4, 32), normalize_chi=32,
        boundary_convergence=({'start_chi': (4, 32), 'max_chi': (32, 64),
                               'patience': 1} if adaptive else False),
        full_update_kwargs={'max_iterations': 4, 'refine_sweeps': 2},
    )
    calibrate, calibrations = opt._calibrate_sweep_boundaries, []
    def record_calibration(*args, **kwargs):
        result = calibrate(*args, **kwargs)
        calibrations.append(result)
        return result
    monkeypatch.setattr(opt, '_calibrate_sweep_boundaries', record_calibration)
    boundary_strip, caps = _strip_update.boundary_strip, []
    def record_cap(*args, **kwargs):
        caps.append(kwargs['chi'])
        return boundary_strip(*args, **kwargs)
    monkeypatch.setattr(_strip_update, 'boundary_strip', record_cap)
    refine = opt._full_update_refine
    def check_refinement(target, key, **kwargs):
        expected = calibrations[-1]['chi'] if adaptive else (4, 32)
        if adaptive:
            assert calibrations[-1]['converged']
            handles = calibrations[-1]['fit_boundaries']
            assert kwargs['boundary'] is handles['bdy']
            assert kwargs['overlap_boundary'] is handles['bdy_overlap']
            assert target is calibrations[-1]['fit_target']
        # An independent fresh fit must agree with the handed-off environments.
        fresh, fresh_report, _, _ = refine_strip(
            opt.state, target, key=key, chi=expected, contraction_opt='greedy',
            boundary_kwargs=opt.boundary_kwargs, sweeps=2, rtol='auto', reuse=False,
        )
        caps.clear()
        actual, report, handle = refine(target, key, **kwargs)
        assert caps == [expected[0], expected[1], expected[0]]
        assert (report['chi'], report['overlap_chi']) == expected
        assert report['accepted'] and fresh_report['accepted']
        np.testing.assert_allclose(ar.to_numpy(actual.to_dense()), ar.to_numpy(fresh.to_dense()),
                                   rtol=1e-9, atol=1e-10)
        assert report['infidelity_after'] == pytest.approx(fresh_report['infidelity_after'], abs=1e-10)
        assert report['infidelity_after'] == pytest.approx(1. - fidelity(actual, target), abs=1e-9)
        return actual, report, handle
    monkeypatch.setattr(opt, '_full_update_refine', check_refinement)
    out = run(opt)
    assert float(ar.to_numpy(out.norm()).real) == pytest.approx(1., abs=1e-9)

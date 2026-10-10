"""Column/row ALS fits against independent dense references and fixed exteriors."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.optimizers.sweep import _als


def fixture(backend='numpy', dtype='complex128', shape=(2, 3), engine='dmrg', bond=2):
    states = []
    for seed in (12, 14):
        state = qtn.PEPS.rand(*shape, bond_dim=bond, dtype=dtype, seed=seed)
        state /= state.norm()
        if backend in ('torch', 'torch-cuda'):
            torch = pytest.importorskip('torch')
            state.apply_to_arrays(torch.as_tensor)
            if backend == 'torch-cuda':
                if not torch.cuda.is_available():
                    pytest.skip('CUDA unavailable')
                state.apply_to_arrays(lambda x: x.to('cuda:0'))
        elif backend == 'cupy':
            cp = pytest.importorskip('cupy')
            try:
                if not cp.cuda.runtime.getDeviceCount():
                    pytest.skip('CUDA unavailable')
            except cp.cuda.runtime.CUDARuntimeError:
                pytest.skip('CUDA unavailable')
            state.apply_to_arrays(cp.asarray)
        states.append(state)
    states[1].mangle_inner_('_target')
    return SweepOptimizer(*states, chi=32, contraction_opt='greedy',
                          local_contraction_opt='greedy', fit_mode='direct',
                          cutoff=0., boundary_engine=engine)


def loss(a, b):
    x, y = (ar.to_numpy(state.to_dense()).ravel() for state in (a, b))
    return 1 - abs(np.vdot(x, y))**2 / (np.vdot(x, x).real * np.vdot(y, y).real)


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_local_als_dense_reference_schedule_and_exterior(axis, backend, dtype):
    sweep = fixture(backend, dtype)
    before = loss(sweep.state, sweep.state_target)
    old = {tag: (sweep.state[tag].data, sweep.state[tag].inds, set(sweep.state[tag].tags))
           for tag in sweep.state.site_tags}
    target = [t.data for t in sweep.state_target]
    sweep._refresh_right_boundaries_once(axis, env_n_iter=2)
    result = sweep._optimize_axis_slice_with_current_env(
        0, axis=axis, solver='als', solver_options={'n_round_trips': 2, 'rtol': 0.},
    )
    after = loss(sweep.state, sweep.state_target)
    tolerance = 3e-5 if dtype in ('float32', 'complex64') else 2e-10
    assert result['loss_initial'] == pytest.approx(before, abs=tolerance)
    assert result['loss_final'] == pytest.approx(after, abs=tolerance)
    assert after < before - 1e-4
    assert all(np.diff(result['history']) <= 0.)
    tags = sweep._site_tensor_tags(axis, 0)
    assert [s['site'] for s in result['local_solves']] == (tags + tags[::-1]) * 2
    assert result['n_round_trips'] == 2
    for tag, (data, inds, tensor_tags) in old.items():
        t = sweep.state[tag]
        assert t.inds == inds and set(t.tags) == tensor_tags and t.shape == data.shape
        assert ar.infer_backend(t.data) == backend and t.data.dtype == data.dtype
        if backend == 'torch':
            assert t.data.device == data.device
        if tag not in tags:
            assert t.data is data
    assert all(t.data is original for t, original in zip(sweep.state_target, target))


@pytest.mark.optional
@pytest.mark.parametrize('backend', ['torch-cuda', 'cupy'])
def test_als_keeps_gpu_arrays_native(backend, monkeypatch):
    sweep = fixture(backend)
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    device = sweep.state['I0,0'].data.device
    with monkeypatch.context() as patch:
        convert = ar.to_numpy

        def scalar_only(value):
            assert np.prod(getattr(value, 'shape', ())) <= 1
            return convert(value)

        patch.setattr(ar, 'to_numpy', scalar_only)
        result = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
    assert result['update_applied']
    assert sweep.state['I0,0'].data.device == device
    assert result['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


def test_rank_deficient_local_norm_has_finite_supported_solution():
    sweep = fixture(shape=(2, 2), bond=3)
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    result = sweep._optimize_axis_slice_with_current_env(
        0, axis='y', solver='als', solver_options={'linear_solver': 'pinv'})
    assert result['update_applied']
    assert any(s['retained_rank'] < s['matrix_size'] for s in result['local_solves'] if 'retained_rank' in s)
    assert result['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


def test_nonunit_target_norm():
    sweep = fixture()
    target = sweep.state.copy(deep=True)
    target['I0,0'].modify(data=target['I0,0'].data + .1)
    target *= 3.
    target.mangle_inner_('_new_target')
    sweep.set_target(target, target_norm=abs(target.norm())**2)
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    result = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
    assert result['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


def test_first_site_matches_independent_dense_least_squares(monkeypatch):
    sweep = fixture()
    state, target = sweep.state.copy(), sweep.state_target
    tag = 'I0,0'
    tensor = state[tag]
    columns = []
    for index in range(tensor.size):
        unit = np.zeros(tensor.shape, dtype=tensor.dtype)
        unit.flat[index] = 1.
        tensor.modify(data=unit)
        columns.append(np.asarray(state.to_dense()).ravel())
    design = np.stack(columns, axis=1)
    expected = np.linalg.lstsq(design, np.asarray(target.to_dense()).ravel(), rcond=1e-12)[0]
    original = _als._solve_site
    checked = []

    def check(*args, **kwargs):
        answer, report = original(*args, **kwargs)
        if not checked:
            # Fidelity fitting allows an arbitrary nonzero overall scale.
            actual = answer.ravel()
            scale = np.vdot(expected, actual) / np.vdot(expected, expected)
            np.testing.assert_allclose(actual, scale * expected, rtol=1e-9, atol=1e-10)
            checked.append(True)
        return answer, report

    monkeypatch.setattr(_als, '_solve_site', check)
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
    assert checked


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('engine', ['dmrg', 'quimb-mps'])
@pytest.mark.parametrize('linear_solver', ['dense-cg', 'dense', 'cg', 'lbfgs', 'dense-lbfgs'])
def test_interior_strip_updates_one_site_with_current_neighbors(axis, engine, linear_solver, monkeypatch):
    from pepsy.optimizers.peps._strip_cursor import StripSweepCursor

    sweep = fixture(shape=(3, 3), engine=engine)
    context = {}
    checked = []
    init_cursor = StripSweepCursor.__init__
    solve_site = _als._solve_site
    optimize_slice = _als._optimize_slice

    def capture_cursor(self, network, **kwargs):
        init_cursor(self, network, **kwargs)
        if '__ALS_KET__' in network.tags:
            context['norm'] = network
            context['boundary'] = [(t, t.data.copy()) for t in network
                                   if not {'__ALS_KET__', '__ALS_BRA__'} & set(t.tags)]

    def check_site(aa, ab, tag, **kwargs):
        current = {key: context['norm'][key, '__ALS_KET__'].data
                   for key in context['tags']}
        # Between consecutive solves only the preceding active site may change.
        for key, data in context.get('previous', {}).items():
            if key != context['previous_tag']:
                np.testing.assert_array_equal(current[key], data)
        context['previous'] = {key: data.copy() for key, data in current.items()}
        context['previous_tag'] = tag
        # Strip results are committed only after its inner passes; the exterior,
        # target, and attached boundary tensors remain fixed throughout.
        for tensor, data in context['frozen'] + context['boundary']:
            np.testing.assert_array_equal(tensor.data, data)
        answer, report = solve_site(aa, ab, tag, **kwargs)
        if tag == 'I1,1':
            state = sweep.state.copy()
            for key, data in current.items():
                state[key].modify(data=data)
            tensor = state[tag]
            columns = []
            for index in range(tensor.size):
                unit = np.zeros(tensor.shape, dtype=tensor.dtype)
                unit.flat[index] = 1.
                tensor.modify(data=unit)
                columns.append(np.asarray(state.to_dense()).ravel())
            design = np.stack(columns, axis=1)
            target = np.asarray(sweep.state_target.to_dense()).ravel()
            expected = np.linalg.lstsq(design, target, rcond=1e-12)[0]
            actual = answer.ravel()
            scale = np.vdot(expected, actual) / np.vdot(expected, expected)
            if 'lbfgs' in linear_solver:
                # A finite L-BFGS budget need not reproduce gauge-sensitive
                # tensor entries. Compare its physical fit with the optimum.
                def fidelity(vector):
                    statevector = design @ vector
                    return abs(np.vdot(statevector, target))**2 / (
                        np.vdot(statevector, statevector).real * np.vdot(target, target).real)
                assert fidelity(actual) == pytest.approx(fidelity(expected), abs=1e-9)
            else:
                np.testing.assert_allclose(actual, scale * expected, rtol=1e-8, atol=1e-9)
            checked.append(True)
        return answer, report

    def check_strip(owner, index, *, axis, controls):
        context.clear()
        tags = owner._site_tensor_tags(axis, index)
        context['tags'] = tags
        context['frozen'] = [(t, t.data.copy()) for tn in (owner.state, owner.state_target)
                             for t in tn]
        exterior = {tag: owner.state[tag].data for tag in owner.state.site_tags if tag not in tags}
        before = loss(owner.state, owner.state_target)
        report = optimize_slice(owner, index, axis=axis, controls=controls)
        assert report['loss_initial'] == pytest.approx(before, abs=1e-9)
        assert report['loss_final'] == pytest.approx(loss(owner.state, owner.state_target), abs=1e-9)
        assert [s['site'] for s in report['local_solves']] == (tags + tags[::-1]) * 2
        assert all(owner.state[tag].data is data for tag, data in exterior.items())
        return report

    monkeypatch.setattr(StripSweepCursor, '__init__', capture_cursor)
    monkeypatch.setattr(_als, '_solve_site', check_site)
    monkeypatch.setattr(_als, '_optimize_slice', check_strip)
    sweep.optimize_axis(axis, solver='als', solver_options={
                            'n_round_trips': 2, 'rtol': 0., 'linear_solver': linear_solver,
                            'cg_rtol': 1e-10, 'cg_shift': 0., 'cg_fallback': False,
                            'lbfgs_rtol': 1e-8, 'lbfgs_maxiter': 500},
                        n_round_trips=0, env_n_iter=2, renormalize=False)
    assert len(checked) == 4


@pytest.mark.parametrize('engine', ['dmrg', 'quimb-mps'])
def test_public_sweep_updates_both_axes_and_matches_exact_loss(engine):
    sweep = fixture(engine=engine)
    before = loss(sweep.state, sweep.state_target)
    sweep.set_optimize_kwargs(optimizer='als', optimizer_options={'n_round_trips': 1},
                             n_round_trips=1, compute_initial_loss=False, compute_final_loss=False)
    result = sweep.run(progress=False, renormalize=False)
    assert result['success']
    assert {r['axis'] for r in result['runs']} == {'x', 'y'}
    assert all(r['solver'] == 'als' for r in result['runs'])
    assert loss(sweep.state, sweep.state_target) < before - .1
    assert result['runs'][-1]['loss_final'] == pytest.approx(
        loss(sweep.state, sweep.state_target), abs=1e-9)


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('method', ['dense-cg', 'cg', 'dense-lbfgs', 'lbfgs'])
def test_compressed_complex_norm_is_scored_with_the_hermitian_metric(axis, method, monkeypatch):
    from pepsy.optimizers.peps._strip_cursor import StripSweepCursor

    sweep = fixture(shape=(3, 3))
    sweep._refresh_right_boundaries_once(axis, env_n_iter=2)
    original = StripSweepCursor.__init__
    injected = []

    def complex_boundary(self, network, **kwargs):
        if '__ALS_KET__' in network.tags and not injected:
            boundary = next(t for t in network
                            if not {'__ALS_KET__', '__ALS_BRA__'} & set(t.tags))
            # Its real quadratic form is unchanged, but an un-Hermitianized
            # scalar check would reject every site before reaching the solver.
            boundary.modify(data=(1 + .02j) * boundary.data)
            injected.append(True)
        original(self, network, **kwargs)

    monkeypatch.setattr(StripSweepCursor, '__init__', complex_boundary)
    report = sweep._optimize_axis_slice_with_current_env(
        0, axis=axis, solver='als', solver_options={
            'linear_solver': method, 'n_round_trips': 1})
    assert injected and report['update_applied'] and report['local_solves']
    assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


@pytest.mark.parametrize('value', [(complex(1., float('nan')), 0.), (-1. + .1j, 0.)])
def test_hermitian_norm_still_rejects_nonfinite_or_nonpositive_values(value):
    with pytest.raises(ValueError):
        _als._log_norm(_als._hermitian_norm(value))


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('method', ['dense-cg', 'cg', 'dense-lbfgs', 'lbfgs'])
def test_rejected_writeback_does_not_pollute_following_cached_solves(axis, method, monkeypatch):
    solve_site = _als._solve_site
    outputs = []
    for write_invalid in (False, True):
        sweep = fixture(shape=(3, 3))
        sweep._refresh_right_boundaries_once(axis, env_n_iter=2)
        calls = 0

        def reject_first(aa, ab, tag, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                if write_invalid:
                    # Exercise rollback after the candidate has been written
                    # into both full and cursor-reduced norm/overlap networks.
                    return np.zeros_like(aa[tag, '__ALS_KET__'].data), {}
                raise ValueError('reject before any candidate writeback')
            return solve_site(aa, ab, tag, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(_als, '_solve_site', reject_first)
            report = sweep._optimize_axis_slice_with_current_env(
                0, axis=axis, solver='als', solver_options={
                    'linear_solver': method, 'n_round_trips': 2, 'rtol': 0.})
        assert not report['local_solves'][0]['accepted']
        assert any(s['accepted'] for s in report['local_solves'][1:])
        assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)
        outputs.append((sweep.state.to_dense(), report['history']))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize('shift,target_shift', [(80., -70.), (-80., 70.)])
def test_network_exponents_do_not_change_normalized_fit(shift, target_shift):
    results = []
    for scaled in (False, True):
        sweep = fixture()
        if scaled:
            sweep.state.exponent += shift
            sweep.state_target.exponent += target_shift
            sweep.set_target_norm((1., 2 * target_shift))
        sweep._refresh_right_boundaries_once('y', env_n_iter=2)
        report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
        results.append(report['loss_final'])
    assert results[0] == pytest.approx(results[1], abs=1e-10)


def test_invalid_solve_preserves_all_arrays(monkeypatch):
    sweep = fixture()
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    original = [t.data for t in sweep.state]

    def invalid(*args, **kwargs):
        raise ValueError('nonfinite local equation')

    monkeypatch.setattr(_als, '_solve_site', invalid)
    result = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
    assert not result['update_applied']
    assert all(t.data is data for t, data in zip(sweep.state, original))


@pytest.mark.parametrize('options', [{'n_round_trips': 0}, {'n_round_trips': True},
    {'n_round_trips': 1.2}, {'rtol': -1.}, {'rcond': float('nan')}, {'rcond': 1.},
    {'max_matrix_size': 0}, {'maxiter': 10}])
def test_bad_als_controls_rejected(options):
    with pytest.raises(ValueError, match='ALS'):
        SweepOptimizer._resolve_user_solver('als', options)


def test_matrix_budget_fails_before_environment_or_state_changes():
    sweep = fixture()
    original = [t.data for t in sweep.state]
    with pytest.raises(ValueError, match='max_matrix_size'):
        sweep._optimize_axis_slice_with_current_env(
            0, axis='y', solver='als', solver_options={'linear_solver': 'dense', 'max_matrix_size': 1})
    assert all(t.data is data for t, data in zip(sweep.state, original))


def test_symmetry_guard_runs_before_normalization(monkeypatch):
    sweep = fixture()
    sweep._symmray_state = True

    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported ALS input must be rejected before normalization')

    monkeypatch.setattr(sweep, '_normalize_state', unexpected)
    with pytest.raises(TypeError, match='Symmray'):
        sweep.optimize_axis('y', solver='als')


@pytest.mark.parametrize('adaptive', [False, 'auto'])
@pytest.mark.parametrize('linear_solver', ['dense-cg', 'dense', 'cg', 'lbfgs', 'dense-lbfgs'])
def test_peps_driver_exposes_als_without_gradient_solver(monkeypatch, adaptive, linear_solver):
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype='complex128', seed=19)
    state /= state.norm()
    gate = np.diag(np.exp(-.3j * np.array([1., -1., -1., 1.])))
    where = ((0, 1), (1, 1))
    target = state.gate(gate, where, contract='split', cutoff=0.)

    def unexpected(*args, **kwargs):
        pytest.fail('ALS must not dispatch to a gradient solver')

    monkeypatch.setattr(SweepOptimizer, '_optimize_packed_params', unexpected)
    opt = PepsOptimizer(state, [(gate, where)], chi=2, mode='sweep', optimizer='als',
                        optimizer_options={'n_round_trips': 2, 'rtol': 0., 'linear_solver': linear_solver},
                        boundary_chi=32, boundary_convergence=adaptive,
                        contraction_opt='greedy', fit_mode='direct',
                        sweep_kwargs={'local_contraction_opt': 'greedy'},
                        sweep_optimize_kwargs={'n_round_trips': 0, 'renormalize': False})
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False, infidelity_tol=0.)
    assert loss(out, target) < .02
    assert opt.get_step_records()[0]['optimized']
    summary = opt.get_step_records()[0]['optimizer_result']
    assert summary['optimizer'] == 'als'
    assert summary['als']['site_solves'] > 0
    assert summary['als']['accepted_site_solves'] > 0
    counts = 'lbfgs_solves' if 'lbfgs' in linear_solver else ('dense_solves' if linear_solver == 'dense' else 'cg_solves')
    assert summary['als'][counts] > 0
    assert summary['als']['cg_fallbacks'] == 0

"""One-site L-BFGS uses analytic complex gradients and fixed environments."""

import autoray as ar
import numpy as np
import pytest

from pepsy.optimizers.sweep import _als
from pepsy.optimizers.sweep._als_lbfgs import solve_lbfgs

from test_peps_sweep_als import fixture, loss


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_quadratic_gradient_and_solution_match_independent_reference(dtype, monkeypatch):
    import scipy.optimize

    rng = np.random.default_rng(5)
    factor = rng.normal(size=(8, 6))
    rhs = rng.normal(size=(6, 2))
    initial = rng.normal(size=(6, 2))
    if dtype == 'complex128':
        factor = factor + 1j * rng.normal(size=factor.shape)
        rhs = rhs + 1j * rng.normal(size=rhs.shape)
        initial = initial + 1j * rng.normal(size=initial.shape)
    matrix = factor.conj().T @ factor + 2 * np.eye(6)
    minimize = scipy.optimize.minimize
    checked = []

    def check_gradient(fun, x0, **kwargs):
        assert kwargs['jac'] is True
        _, gradient = fun(x0)
        finite = np.empty_like(x0)
        for index in range(len(x0)):
            step = np.zeros_like(x0)
            step[index] = 1e-6
            finite[index] = (fun(x0 + step)[0] - fun(x0 - step)[0]) / 2e-6
        np.testing.assert_allclose(gradient, finite, rtol=2e-7, atol=2e-7)
        checked.append(True)
        return minimize(fun, x0, **kwargs)

    monkeypatch.setattr(scipy.optimize, 'minimize', check_gradient)
    answer, report = solve_lbfgs(lambda x: matrix @ x, rhs, initial,
                                 maxiter=300, rtol=1e-7, history=10, maxls=30)
    np.testing.assert_allclose(answer, np.linalg.solve(matrix, rhs), rtol=1e-5, atol=1e-6)
    assert checked and report['gradient'] == 'analytic'


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('method', ['lbfgs', 'dense-lbfgs'])
@pytest.mark.parametrize('backend', ['numpy', 'torch',
    pytest.param('torch-cuda', marks=pytest.mark.optional),
    pytest.param('cupy', marks=pytest.mark.optional)])
def test_one_site_lbfgs_preserves_exterior_device_and_exact_loss(axis, method, backend, monkeypatch):
    sweep = fixture(backend)
    before = loss(sweep.state, sweep.state_target)
    exterior = {tag: sweep.state[tag].data for tag in sweep.state.site_tags}
    target = [t.data for t in sweep.state_target]
    tags = sweep._site_tensor_tags(axis, 0)
    sweep._refresh_right_boundaries_once(axis, env_n_iter=2)
    to_numpy = ar.to_numpy
    max_site_size = max(sweep.state[tag].size for tag in tags)

    def vectors_only(value):
        assert np.prod(getattr(value, 'shape', ())) <= max_site_size
        return to_numpy(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, 'to_numpy', vectors_only)
        report = sweep._optimize_axis_slice_with_current_env(0, axis=axis, solver='als', solver_options={
            'linear_solver': method, 'n_round_trips': 2, 'rtol': 0.,
            'max_matrix_size': 1 if method == 'lbfgs' else 1024,
        })
    after = loss(sweep.state, sweep.state_target)
    assert after < before - .01
    assert report['loss_final'] == pytest.approx(after, abs=1e-9)
    assert [s['site'] for s in report['local_solves']] == (tags + tags[::-1]) * 2
    assert all(np.diff(report['history']) <= 0.)
    accepted = [s for s in report['local_solves'] if s['accepted']]
    assert accepted and all(s['solver'] == method and s['gradient'] == 'analytic' for s in accepted)
    for tag, previous in exterior.items():
        value = sweep.state[tag].data
        assert value.dtype == previous.dtype and ar.infer_backend(value) == ar.infer_backend(previous)
        if hasattr(previous, 'device'):
            assert value.device == previous.device
        if tag not in tags:
            assert value is previous
    assert all(t.data is original for t, original in zip(sweep.state_target, target))


@pytest.mark.parametrize('method', ['lbfgs', 'dense-lbfgs'])
@pytest.mark.parametrize('dtype', ['float32', 'complex64'])
def test_single_precision_and_scaled_network(method, dtype):
    results = []
    for shift in (0., 80.):
        sweep = fixture(dtype=dtype)
        sweep.state.exponent += shift
        sweep.state_target.exponent -= shift
        sweep.set_target_norm((1., -2 * shift))
        sweep._refresh_right_boundaries_once('y', env_n_iter=2)
        report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als', solver_options={
            'linear_solver': method, 'n_round_trips': 1,
        })
        assert report['update_applied']
        # Normalized fidelity is exponent-invariant; do not overflow a float32
        # reference statevector by explicitly multiplying it by 10**80.
        state, target = sweep.state.copy(), sweep.state_target.copy()
        state.exponent = target.exponent = 0.
        assert report['loss_final'] == pytest.approx(loss(state, target), abs=1e-5)
        assert all(ar.get_dtype_name(t.data) == dtype for t in sweep.state)
        results.append(report['loss_final'])
    # Float32 L-BFGS line searches can stop at slightly different iterates
    # after exponent cancellation. Use the same precision budget as the
    # independent fidelity check above (the residual target is 1e-4).
    assert results[0] == pytest.approx(results[1], abs=1e-5)


def test_finite_iteration_budget_is_reported_without_claiming_convergence():
    matrix = np.diag(np.arange(1., 11.))
    answer, report = solve_lbfgs(lambda x: matrix @ x, np.ones((10, 1)), np.zeros((10, 1)),
                                 maxiter=1, rtol=1e-14, history=3, maxls=10)
    assert np.isfinite(answer).all()
    assert report['lbfgs_iterations'] <= 1 and not report['converged']
    assert report['relative_residual'] > 1e-14


@pytest.mark.parametrize('warm_start', [False, True])
def test_singular_supported_quadratic_and_exact_warm_start(warm_start):
    matrix = np.diag([0., 1., 3.])
    expected = np.array([[2.j], [1.], [1. / 3.]])
    rhs = matrix @ expected
    initial = expected.copy() if warm_start else np.array([[2.j], [0.], [0.]])
    answer, report = solve_lbfgs(lambda x: matrix @ x, rhs, initial,
                                 maxiter=100, rtol=1e-8, history=5, maxls=20)
    np.testing.assert_allclose(answer, expected, atol=1e-8)
    residual = np.linalg.norm(matrix @ answer - rhs) / np.linalg.norm(rhs)
    assert report['relative_residual'] == pytest.approx(residual, abs=1e-14)
    assert report['converged']
    if warm_start:
        assert report['lbfgs_iterations'] == 0
        assert report['lbfgs_evaluations'] == 1


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('solver', ['lbfgs', 'torch-lbfgs'])
def test_whole_column_lbfgs_remains_a_separate_joint_fit(axis, solver, monkeypatch):
    sweep = fixture('torch')
    before = loss(sweep.state, sweep.state_target)
    tags = sweep._site_tensor_tags(axis, 0)
    exterior = {tag: sweep.state[tag].data for tag in sweep.state.site_tags if tag not in tags}
    sweep._refresh_right_boundaries_once(axis, env_n_iter=2)

    def no_site_fit(*args, **kwargs):
        pytest.fail('whole-column L-BFGS must not dispatch to one-site ALS')

    monkeypatch.setattr(_als, 'optimize_slice', no_site_fit)
    controls = {'n_steps': 5}
    if solver == 'torch-lbfgs':
        # The shared gradient defaults use Adam's small learning rate. Give
        # native L-BFGS its standard step scale and a safeguarded line search.
        controls.update(lr=1., line_search_fn='strong_wolfe')
    report = sweep._optimize_axis_slice_with_current_env(
        0, axis=axis, solver=solver, solver_options=controls)
    after = loss(sweep.state, sweep.state_target)
    assert report['update_applied'] and after < before - .01
    assert report['loss_final'] == pytest.approx(after, abs=1e-9)
    assert all(sweep.state[tag].data is data for tag, data in exterior.items())
    assert all(ar.infer_backend(t.data) == 'torch' for t in sweep.state)


@pytest.mark.parametrize('controls', [{'lbfgs_maxiter': 0}, {'lbfgs_maxiter': True},
    {'lbfgs_rtol': 0.}, {'lbfgs_rtol': -1.}, {'lbfgs_history': 0}, {'lbfgs_maxls': 0}])
def test_bad_lbfgs_controls(controls):
    with pytest.raises(ValueError, match='ALS'):
        _als.options(controls)

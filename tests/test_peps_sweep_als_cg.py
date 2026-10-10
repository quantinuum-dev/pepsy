"""Matrix-free ALS: operator references, backend preservation, and fallback."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import _strip_solve
from pepsy.optimizers.sweep import _als
from pepsy.optimizers.sweep._als_cg import CGFailure, prepare_operator, solve_cg

from test_peps_sweep_als import fixture, loss


BACKENDS = ['numpy', 'torch', pytest.param('torch-cuda', marks=pytest.mark.optional),
            pytest.param('cupy', marks=pytest.mark.optional)]


@pytest.mark.parametrize('backend', BACKENDS)
def test_hermitian_operator_matches_explicit_complex_ring(backend, monkeypatch):
    sweep = fixture(backend)
    sample = sweep.state['I0,0'].data
    rng = np.random.default_rng(21)
    tensors = []
    for inds in ('wxAa', 'xyBb', 'yzCc', 'zwDd'):
        value = rng.normal(size=(3, 3, 2, 2)) + 1j * rng.normal(size=(3, 3, 2, 2))
        if backend.startswith('torch'):
            import torch
            value = torch.as_tensor(value, device=sample.device)
        elif backend == 'cupy':
            import cupy
            value = cupy.asarray(value)
        tensors.append(qtn.Tensor(value, inds=inds))
    network = qtn.TensorNetwork(tensors)
    network.exponent = ar.do('sum', ar.do('abs', sample)) * 0 + 17.
    left, right = tuple('ABCD'), tuple('abcd')
    explicit, exponent = network.contract(all, output_inds=(*left, *right),
                                         preserve_tensor=True, strip_exponent=True)
    expected = ar.to_numpy(explicit.data).reshape(16, 16)
    expected = .5 * (expected + expected.conj().T)
    vector = explicit.data.reshape(16, 16)[:, :2]
    convert = ar.to_numpy

    def scalar_only(value):
        assert np.prod(getattr(value, 'shape', ())) <= 1
        return convert(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, 'to_numpy', scalar_only)
        action, diagonal, power = prepare_operator(network, left, right, vector, optimize='greedy')
        result = action(vector)
    assert float(network.exponent) == 17.
    factor = 10.**float(exponent - power)
    np.testing.assert_allclose(ar.to_numpy(result), factor * expected @ ar.to_numpy(vector), atol=1e-12)
    np.testing.assert_allclose(ar.to_numpy(diagonal), factor * expected.diagonal().real, atol=1e-12)
    assert result.dtype == sample.dtype
    if backend.startswith('torch') or backend == 'cupy':
        assert result.device == sample.device


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_cg_sweep_matches_dense_without_dense_solver(axis, backend, dtype, monkeypatch):
    dense = fixture(backend, dtype, shape=(3, 3))
    dense.optimize_axis(axis, solver='als', solver_options={'n_round_trips': 1, 'linear_solver': 'dense'},
                        n_round_trips=0, renormalize=False)
    expected = loss(dense.state, dense.state_target)
    sweep = fixture(backend, dtype, shape=(3, 3))

    def no_dense(*args, **kwargs):
        pytest.fail('matrix-free solve must not call the dense eigensolver')

    monkeypatch.setattr(_als, '_solve_dense', no_dense)
    monkeypatch.setattr(_strip_solve, 'solve_positive', no_dense)
    convert = ar.to_numpy

    def scalar_only(value):
        assert np.prod(getattr(value, 'shape', ())) <= 1
        return convert(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, 'to_numpy', scalar_only)
        reports = sweep.optimize_axis(axis, solver='als', solver_options={
            'n_round_trips': 1, 'linear_solver': 'cg', 'max_matrix_size': 1,
        }, n_round_trips=0, renormalize=False)
    tolerance = 2e-4 if dtype == 'complex64' else 1e-7
    actual = loss(sweep.state, sweep.state_target)
    assert actual == pytest.approx(expected, abs=tolerance)
    assert reports[-1]['loss_final'] == pytest.approx(actual, abs=tolerance)
    accepted = [s for report in reports for s in report['local_solves'] if s['accepted']]
    assert accepted and all(s['solver'] == 'cg' and s['converged'] for s in accepted)
    assert all(ar.get_dtype_name(t.data) == dtype for t in sweep.state)


@pytest.mark.parametrize('fallback', [True, False])
@pytest.mark.parametrize('method', ['cg', 'dense-cg'])
def test_cg_failure_falls_back_or_rolls_back(fallback, method):
    sweep = fixture()
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    before = [t.data for t in sweep.state]
    report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als', solver_options={
        'linear_solver': method, 'cg_maxiter': 1, 'cg_rtol': 1e-14,
        'cg_fallback': fallback or method == 'cg',
        'max_matrix_size': 1 if not fallback and method == 'cg' else 1024,
    })
    failures = [s for s in report['local_solves'] if 'cg_failure' in s]
    assert failures and all(s['cg_failure'] == 'iteration_limit' for s in failures)
    if fallback:
        assert report['update_applied']
        assert all(s['fallback_from'] == 'cg' for s in failures)
        assert all(s['relative_residual'] < 1e-8 for s in failures)
        assert all(s['cg_relative_residual'] > 1e-14 for s in failures)
        assert all(s.get('fallback_from') == 'cg' or s.get('converged') for s in report['local_solves'])
    else:
        assert not report['update_applied']
        assert all(t.data is data for t, data in zip(sweep.state, before))


def test_cg_does_not_fall_back_unless_requested(monkeypatch):
    sweep = fixture()
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)

    def unexpected(*args, **kwargs):
        pytest.fail('CG must not allocate a dense metric or diagonalize without permission')

    monkeypatch.setattr(_strip_solve, 'solve_positive', unexpected)
    monkeypatch.setattr(_als, '_solve_dense', unexpected)
    report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als', solver_options={
        'linear_solver': 'cg', 'cg_maxiter': 1, 'cg_rtol': 1e-14,
    })
    assert not report['update_applied']
    assert all(s['cg_failure'] == 'iteration_limit' for s in report['local_solves'])


def test_default_builds_explicit_metric_but_uses_no_factorization(monkeypatch):
    sweep = fixture()
    sweep._refresh_right_boundaries_once('y', env_n_iter=2)
    dispatch = ar.do

    def no_factorization(name, *args, **kwargs):
        assert name not in ('linalg.solve', 'linalg.eigh', 'linalg.svd', 'linalg.cholesky')
        return dispatch(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ar, 'do', no_factorization)
        report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als')
    assert report['update_applied']
    accepted = [s for s in report['local_solves'] if s['accepted']]
    assert all(s['solver'] == 'dense-cg' and s['converged'] for s in accepted)
    assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


def test_explicit_direct_solve_hermitianizes_and_rejects_singular_system():
    matrix = np.array([[3., 2. + 1.j], [1. - 2.j, 4.]])
    rhs = np.array([[1.j], [2.]])
    answer, report = _als._solve_dense(matrix, rhs)
    np.testing.assert_allclose(.5 * (matrix + matrix.conj().T) @ answer, rhs)
    assert report['solver'] == 'dense'
    singular = np.diag([1., 0.])
    with pytest.raises(ValueError, match='singular'):
        _als._solve_dense(singular, np.ones((2, 1)))
    answer, _ = _als._solve_dense(singular, np.ones((2, 1)), shift=1e-5)
    np.testing.assert_allclose((singular + 1e-5 * np.eye(2)) @ answer, 1.)


def test_cg_rejects_indefinite_curvature_and_checks_true_residual():
    matrix = np.array([[1., 2.], [2., 1.]])
    with pytest.raises(CGFailure, match='nonpositive_curvature'):
        solve_cg(lambda x: matrix @ x, matrix.diagonal(), np.array([[1.], [-1.]]),
                 np.zeros((2, 1)), rtol=1e-10, maxiter=10, shift=0.)
    matrix = np.diag([1e-8, 1., 2.])
    rhs = np.ones((3, 2))
    result, report = solve_cg(lambda x: matrix @ x, matrix.diagonal(), rhs,
                              np.zeros_like(rhs), rtol=1e-10, maxiter=10, shift=1e-6)
    shifted = matrix + 2e-6 * np.eye(3)
    np.testing.assert_allclose(shifted @ result, rhs, atol=1e-10)
    assert report['relative_residual'] <= 1e-10


def test_rank_deficient_cg_and_exponent_scaling():
    losses = []
    for shift in (0., 80., -80.):
        sweep = fixture(shape=(2, 2), bond=3)
        sweep.state.exponent += shift
        sweep.state_target.exponent -= shift
        sweep.set_target_norm((1., -2 * shift))
        sweep._refresh_right_boundaries_once('y', env_n_iter=2)
        report = sweep._optimize_axis_slice_with_current_env(0, axis='y', solver='als', solver_options={
            'linear_solver': 'cg', 'cg_fallback': False, 'cg_maxiter': 500,
        })
        assert report['update_applied']
        assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-8)
        losses.append(report['loss_final'])
    np.testing.assert_allclose(losses, losses[0], atol=1e-8)


@pytest.mark.parametrize('controls', [{'linear_solver': 'invalid'}, {'cg_maxiter': 0},
    {'cg_rtol': 0.}, {'cg_rtol': -1.}, {'cg_shift': -1.}, {'cg_fallback': 1}])
def test_invalid_cg_controls(controls):
    with pytest.raises(ValueError, match='ALS'):
        _als.options(controls)

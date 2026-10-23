"""Adaptive environment gauges preserve the metric and full-update result."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import _environment_gauge_r, options


@pytest.fixture
def adaptive_qr():
    if not callable(getattr(qtn.decomp, 'qr_auto', None)):
        pytest.skip('Quimb does not provide qr_auto')


@pytest.fixture(params=['torch-cpu', 'torch-cuda', 'cupy'])
def convert(request):
    if request.param == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip('CuPy needs a CUDA device')
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f'CUDA unavailable: {exc}')
        return cp.asarray
    torch = pytest.importorskip('torch')
    device = request.param.split('-')[1]
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Torch CUDA unavailable')
    return lambda x: torch.tensor(x, device=device)


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('case', ['tall', 'small', 'singular'])
def test_adaptive_gauge_factor_preserves_gram_and_input(
    adaptive_qr, convert, dtype, case, monkeypatch,
):
    rng = np.random.default_rng(71)
    shape = (32, 16) if case == 'small' else (2048, 32)
    data = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
    if case == 'singular':
        data[:, -1] = 0.
    matrix = convert(data)
    backend = ar.infer_backend(matrix)
    original_qr = ar.get_lib_fn(backend, 'linalg.qr')
    branches = []
    for name in ('qr_via_cholesky', 'qr_stabilized'):
        fn = getattr(qtn.decomp, name)
        def record(*args, _fn=fn, _name=name, **kwargs):
            branches.append(_name)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(qtn.decomp, name, record)
    factor = _environment_gauge_r(matrix, 'qr:auto')
    assert factor.dtype == matrix.dtype and factor.device == matrix.device
    assert ar.get_lib_fn(backend, 'linalg.qr') is original_qr
    actual = ar.to_numpy(factor)
    tol = 3e-6 if dtype == 'complex64' else 2e-13
    expected_gram = data.conj().T @ data
    error = np.linalg.norm(actual.conj().T @ actual - expected_gram)
    assert error / np.linalg.norm(expected_gram) < tol
    np.testing.assert_array_equal(ar.to_numpy(matrix), data)
    np.testing.assert_allclose(np.tril(actual, -1), 0., atol=tol)
    if case == 'tall':
        assert branches == ['qr_via_cholesky']
    elif case == 'small':
        assert branches == ['qr_stabilized']
    else:
        assert branches == ['qr_via_cholesky', 'qr_stabilized']


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_full_update_auto_gauge_matches_qr(adaptive_qr, convert, dtype):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype=dtype, seed=24)
    state /= state.norm()
    state.apply_to_arrays(convert)
    saved = {site: ar.to_numpy(state[site].data).copy() for site in state.sites}
    gate = convert(np.diag(np.exp(np.array([-.21j, .21j, .21j, -.21j]))).astype(dtype))
    where = ((0, 0), (0, 1))
    target = state.gate(gate, where, contract=False).to_dense()
    target = ar.to_numpy(target).ravel()
    outputs, reports = [], []
    for method in ('qr', 'qr:auto'):
        runner = PepsOptimizer(
            state, [(gate, where)], chi=1, mode='full-update',
            contraction_opt='greedy', fit_mode='direct', boundary_chi=16,
            normalize_initial=False, boundary_kwargs={'cutoff': 0.},
            full_update_kwargs={'gauge': True, 'gauge_qr': method,
                                'max_iterations': 8, 'solver': 'qr', 'rtol': 0.},
        )
        result = runner.run(normalize_final=False, measure_infidelity=False,
                            measure_final_infidelity=False, accept_if_improved=False)
        report = runner.get_step_records()[0]['optimizer_result']
        assert report['gauge_qr'] == method and report['environment_gauge_applied']
        for site in state.sites:
            assert result[site].data.dtype == state[site].data.dtype
            assert result[site].data.device == state[site].data.device
            np.testing.assert_array_equal(ar.to_numpy(state[site].data), saved[site])
            if site not in where:
                np.testing.assert_array_equal(ar.to_numpy(result[site].data), saved[site])
        outputs.append(ar.to_numpy(result.to_dense()).ravel())
        reports.append(report)
    tol = 5e-5 if dtype == 'complex64' else 2e-10
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=tol, atol=tol)
    for output, report in zip(outputs, reports):
        fidelity = abs(np.vdot(output, target))**2 / (np.vdot(output, output).real * np.vdot(target, target).real)
        assert report['fidelity'] == pytest.approx(fidelity, abs=tol)


@pytest.mark.parametrize('controls', [
    {'gauge': True, 'gauge_qr': 'cholesky'},
    {'gauge_qr': 'qr:auto'},
    {'gauge': True, 'gauge_qr': 'qr:auto', 'tensor_mode': 'full'},
])
def test_invalid_gauge_qr_options(controls):
    with pytest.raises(ValueError):
        options(controls)


def test_adaptive_qr_capability_is_optional(monkeypatch):
    monkeypatch.delattr(qtn.decomp, 'qr_auto', raising=False)
    assert options({'gauge': True})['gauge_qr'] == 'qr'
    with pytest.raises(NotImplementedError, match='requires Quimb'):
        options({'gauge': True, 'gauge_qr': 'qr:auto'})


def test_auto_gauge_keeps_differentiable_inputs_rejected(adaptive_qr):
    torch = pytest.importorskip('torch')
    from pepsy.optimizers.peps._full_update import ReducedPair
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=24)
    state.apply_to_arrays(lambda x: torch.tensor(x, requires_grad=True))
    with pytest.raises(TypeError, match='non-differentiable'):
        ReducedPair(state, torch.eye(4, dtype=torch.complex128), ((0, 0), (0, 1)), 1)

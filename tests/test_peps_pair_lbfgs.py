"""Joint pair fits: complex gradients, exact metrics and unfrozen site tensors."""

import numpy as np
import pytest

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import FullPair, ReducedPair, options
from pepsy.optimizers.peps._pair_lbfgs import pair_value_gradient
from test_peps_full_update import fixture


def test_pair_complex_gradients_match_autograd():
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)
    x = torch.randn(6, 6, dtype=torch.complex128)
    norm = (x.mH @ x).reshape(2, 3, 2, 3)
    target = torch.randn(2, 2, 3, 2, dtype=torch.complex128)
    left = torch.randn(2, 2, 2, dtype=torch.complex128, requires_grad=True)
    right = torch.randn(2, 3, 2, dtype=torch.complex128, requires_grad=True)
    value, ga, gb = pair_value_gradient(norm, target, left, right)
    expected = torch.autograd.grad(value, (left, right))
    torch.testing.assert_close(ga, expected[0])
    torch.testing.assert_close(gb, expected[1])


@pytest.mark.parametrize('where', [((1, 1), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (2, 1))])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_full_pair_lbfgs_matches_exact_metric_and_changes_external_subspace(where, dtype):
    torch = pytest.importorskip('torch')
    state, gate, policy = fixture(dtype=dtype, shape=(3, 3))
    rng = np.random.default_rng(18)
    gate = torch.tensor(np.linalg.qr(rng.normal(size=(4, 4)) +
                                   1j * rng.normal(size=(4, 4)))[0], dtype=gate.dtype)
    original = state.to_dense().clone()
    reduced = ReducedPair(state, gate, where, chi=2)
    pair = FullPair(state, gate, where, chi=2)
    assert np.prod(pair.shapes[0]) * np.prod(pair.shapes[1]) > reduced.na * reduced.nb
    guess, target = pair.initial_states()
    reference = state.gate(gate, where, contract='split', cutoff=0.)
    tol = 3e-5 if dtype == 'complex64' else 2e-10
    torch.testing.assert_close(target.to_dense(), reference.to_dense(), rtol=tol, atol=tol)
    norm, _ = pair.environment(guess, chi=64, contraction_opt=policy,
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    out, report = pair.optimize(norm, {'max_iterations': 80, 'rtol': 1e-10})
    x, y = out.to_dense().ravel(), target.to_dense().ravel()
    fidelity = float(abs(torch.vdot(x, y))**2 / (torch.vdot(x, x).real * torch.vdot(y, y).real))
    assert report['solver'] == 'lbfgs'
    assert report['tensor_mode'] == 'full'
    assert report['fidelity'] == pytest.approx(fidelity, abs=tol)
    assert report['loss_history'][-1] < report['warmstart_loss']
    assert fidelity >= report['warmstart_fidelity'] - tol
    assert not report['environment_gauge_applied']
    assert out.max_bond() <= 2
    torch.testing.assert_close(state.to_dense(), original, rtol=0, atol=0)
    for site in state.gen_site_coos():
        assert out[site].data.dtype == state[site].data.dtype
        assert out[site].data.device == state[site].data.device
        assert out[site].tags == state[site].tags
        assert out[site].inds == state[site].inds
        if site not in where:
            assert out[site].data is state[site].data
    # The center site's exterior space has dimension eight, versus four in
    # its frozen reduced basis. Prove the fit can leave that initial subspace.
    if where[0] == (1, 1):
        a = out[where[0]].transpose(*pair.external[0], pair.phys[0], pair.bond).data.reshape(int(np.prod(pair.shapes[0])), -1)
        outside = a - reduced.qa @ (reduced.qa.mH @ a)
    else:
        a = out[where[1]].transpose(pair.bond, pair.phys[1], *pair.external[1]).data.reshape(-1, int(np.prod(pair.shapes[1])))
        outside = a - (a @ reduced.qb.mH) @ reduced.qb
    assert float(torch.linalg.norm(outside)) > 1e-5


@pytest.mark.parametrize('tensor_mode', ['reduced', 'full'])
def test_lbfgs_driver_preserves_nonunitary_target_normalization(tensor_mode):
    state, gate, policy = fixture(shape=(3, 3))
    opt = PepsOptimizer(
        state, [(2 * gate, ((1, 1), (1, 2)))], chi=2, update_style='two-site',
        contraction_opt=policy, fit_mode='direct', boundary_chi=64,
        boundary_convergence=False, normalize_initial=False,
        full_update_kwargs={'tensor_mode': tensor_mode, 'solver': 'lbfgs', 'max_iterations': 20},
    )
    out = opt.run(non_unitary=True, normalize_target=True, normalize_final=True,
                  measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    assert float(out.norm().real) == pytest.approx(1., abs=2e-10)
    report = opt.get_step_records()[0]['optimizer_result']
    assert report['solver'] == 'lbfgs'
    assert report['tensor_mode'] == tensor_mode


def test_pair_modes_validate_before_fit_and_default_to_reduced_without_gauge():
    assert options()['tensor_mode'] == 'reduced'
    assert options()['gauge'] is False
    assert options({'tensor_mode': 'full'})['solver'] == 'lbfgs'
    for kwargs in ({'tensor_mode': 'unknown'}, {'tensor_mode': 'full', 'solver': 'qr'},
                   {'tensor_mode': 'full', 'gauge': True}, {'full_max_matrix_size': 0}):
        with pytest.raises(ValueError):
            options(kwargs)


def test_full_pair_normalization_preserves_exact_target_scale_and_phase():
    torch = pytest.importorskip('torch')
    state, gate, _ = fixture(shape=(3, 3))
    state.exponent = 1.5
    pair = FullPair(state, 2 * gate, ((1, 1), (1, 2)), chi=2)
    _, target = pair.initial_states()
    pair.normalize_target((4 * np.exp(.2j), 3.))
    _, normalized = pair.initial_states()
    torch.testing.assert_close(normalized.to_dense(), target.to_dense() / np.sqrt(4000 * np.exp(.2j)),
                               rtol=1e-11, atol=1e-11)


def test_full_pair_lbfgs_cupy_matches_torch():
    cp = pytest.importorskip('cupy')
    try:
        cp.zeros(1)
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    state, gate, policy = fixture(shape=(3, 3))
    where = ((1, 1), (1, 2))
    results = []
    for backend in ('torch', 'cupy'):
        native = state.copy()
        if backend == 'cupy':
            native.apply_to_arrays(lambda x: cp.asarray(x.numpy()))
        pair = FullPair(native, gate.numpy(), where, chi=2)
        guess, _ = pair.initial_states()
        norm, _ = pair.environment(guess, chi=64, contraction_opt=policy,
                                  boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
        out, report = pair.optimize(norm, {'max_iterations': 30, 'rtol': 1e-10})
        if backend == 'cupy':
            assert isinstance(out[where[0]].data, cp.ndarray)
            assert out[where[0]].data.device == native[where[0]].data.device
        results.append(report['fidelity'])
    np.testing.assert_allclose(results[0], results[1], atol=1e-8, rtol=0)

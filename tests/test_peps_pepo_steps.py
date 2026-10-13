"""Full-step PEPO targets, whole-state fitting, and disjoint direct layers."""

import importlib.util
from pathlib import Path

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import BoundaryChiLimitError, GlobalOptimizer, PepsOptimizer, SweepOptimizer
from pepsy.optimizers.peps import optimizer as peps_mod
from pepsy.optimizers import global_opt as global_mod
from pepsy.tensors import ps_to_peps

spec = importlib.util.spec_from_file_location(
    'tfim_pepo_example', Path(__file__).parents[1] / 'examples' / 'tfim_pepo.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def setup(backend='numpy'):
    operator = example.trotter_pepo(2, 3, .13, j=.7, h=1.1, backend=backend)
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype='complex128', seed=43)
    state /= state.norm()
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        state.apply_to_arrays(torch.as_tensor)
    return state, operator


def dense_loss(state, target):
    x, y = (ar.to_numpy(t.to_dense()).ravel() for t in (state, target))
    return 1 - abs(np.vdot(x, y))**2 / (np.vdot(x, x).real * np.vdot(y, y).real)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_bond_two_full_strang_pepo_matches_independent_dense_step(backend):
    state, operator = setup(backend)
    sites = list(state.gen_site_coos())
    n = len(sites)
    z = np.array([1., -1.])
    rx = np.cos(1.1 * .13 / 2) * np.eye(2) + 1j * np.sin(1.1 * .13 / 2) * np.array([[0., 1.], [1., 0.]])
    half = rx
    for _ in range(n - 1):
        half = np.kron(half, rx)
    diag = np.zeros(2**n)
    positions = {site: i for i, site in enumerate(sites)}
    for config in range(2**n):
        values = [z[(config >> (n - i - 1)) & 1] for i in range(n)]
        for x, y in sites:
            for dx, dy in ((1, 0), (0, 1)):
                other = (x + dx, y + dy)
                if other in positions:
                    diag[config] += values[positions[x, y]] * values[positions[other]]
    expected = half @ np.diag(np.exp(1j * .7 * .13 * diag)) @ half
    np.testing.assert_allclose(ar.to_numpy(operator.to_dense()), expected, atol=2e-14)
    assert operator.max_bond() == 2


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_full_pepo_als_uses_one_lazy_target_and_preserves_inputs(monkeypatch, backend):
    state, operator = setup(backend)
    before, op_before = ar.to_numpy(state.to_dense()).copy(), ar.to_numpy(operator.to_dense()).copy()
    seen = []
    solve = SweepOptimizer.optimize_axis

    def capture(sweeper, axis, **kwargs):
        assert sweeper.state_target.num_tensors == 2 * state.num_tensors
        assert len(sweeper.state_target.select('PEPO').tensors) == state.num_tensors
        assert sweeper.chi == (32, 128)
        seen.append(axis)
        return solve(sweeper, axis, **kwargs)

    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', capture)
    opt = PepsOptimizer(state, [], chi=2, boundary_chi=(32, 128), normalize_chi=32,
        evaluation_chi=(32, 128), contraction_opt='greedy', normalize_initial=False,
        sweep_kwargs={'local_contraction_opt': 'greedy'},
        sweep_optimize_kwargs={'n_round_trips': 0, 'n_cycles': 1})
    output = opt.run_pepo(operator, infidelity_tol=0.)
    assert seen == ['y', 'x']
    record, = opt.get_step_records()
    assert record['target_representation'] == 'pepo' and record['reason'] == 'optimized'
    assert record['final_infidelity'] <= record['pre_infidelity']
    assert record['final_infidelity'] == pytest.approx(dense_loss(output, operator.apply(state)), abs=1e-9)
    assert output.max_bond() <= 2
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)
    np.testing.assert_array_equal(ar.to_numpy(operator.to_dense()), op_before)


def test_global_direct_compresses_three_disjoint_overlap_layers(monkeypatch):
    state, operator = setup('torch')
    operator.add_tag('PEPO')
    target = operator.apply(state, contract=False)
    target.mangle_inner_('__target')
    original = qtn.TensorNetwork2D.contract_boundary
    seen = []

    def capture(network, **kwargs):
        layers = tuple(kwargs['layer_tags'])
        seen.append((layers, kwargs['max_bond']))
        for tensor in network:
            assert sum(tag in tensor.tags for tag in layers) == 1
        return original(network, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork2D, 'contract_boundary', capture)
    global_fit = GlobalOptimizer(state, target, chi=(32, 128))
    value = global_fit.loss(mode='mps', mode_='direct', canonize=False, cutoff=0.,
        layer_tags=('BRA', 'KET'), overlap_layer_tags=('BRA', 'PEPO', 'KET'),
        contraction_opt='greedy', target_norm=1.)
    assert float(value) == pytest.approx(dense_loss(state, target), abs=1e-10)
    assert seen == [(('BRA', 'KET'), 32), (('BRA', 'PEPO', 'KET'), 128)]


def test_global_full_pepo_step_improves_dense_fidelity():
    state, operator = setup('torch')
    opt = PepsOptimizer(state, [], chi=2, mode='global', optimizer='L-BFGS-B',
        boundary_chi=(32, 128), normalize_chi=32, evaluation_chi=(32, 128),
        contraction_opt='greedy', normalize_initial=False,
        global_optimize_kwargs={'n': 12, 'autodiff_backend': 'torch'})
    result = opt.run_pepo(operator, infidelity_tol=0.)
    record, = opt.get_step_records()
    assert record['reason'] == 'optimized'
    assert record['final_infidelity'] < record['pre_infidelity']
    assert record['final_infidelity'] == pytest.approx(dense_loss(result, operator.apply(state)), abs=1e-9)
    assert ar.infer_backend(next(iter(result)).data) == 'torch'


def test_four_by_four_pepo_action_matches_independent_dense_gate_replay():
    state = ps_to_peps(4, 4, theta=0.)
    operator = example.trotter_pepo(4, 4, .1)
    n = 16
    vector = np.zeros(2**n, complex)
    vector[0] = 1.

    def apply(gate, positions):
        nonlocal vector
        rest = [i for i in range(n) if i not in positions]
        order = list(positions) + rest
        data = vector.reshape((2,) * n).transpose(order).reshape(2**len(positions), -1)
        data = (gate @ data).reshape((2,) * n).transpose(np.argsort(order))
        vector = data.reshape(-1)

    rx = np.cos(.05) * np.eye(2) + 1j * np.sin(.05) * np.array([[0., 1.], [1., 0.]])
    zz = np.diag(np.exp(.1j * np.array([1., -1., -1., 1.])))
    for i in range(n):
        apply(rx, (i,))
    for x in range(4):
        for y in range(4):
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < 4 and y + dy < 4:
                    apply(zz, (4 * x + y, 4 * (x + dx) + y + dy))
    for i in range(n):
        apply(rx, (i,))
    target = operator.apply(state, contract=False)
    np.testing.assert_allclose(target.to_dense(optimize='greedy').ravel(), vector, atol=2e-14)
    assert operator.max_bond() == 2


def test_full_pepo_retry_restores_step_and_retains_larger_overlap_cap(monkeypatch):
    state, operator = setup()
    original = peps_mod.boundary_infidelity
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(kwargs['chi'])
        if len(calls) == 1:
            return {'norm': 1., 'norm_target': 1., 'overlap': complex(float('nan')), 'infidelity': .1}
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', fail_once)
    options = dict(chi=2, boundary_chi=(32, 64), normalize_chi=32,
                   evaluation_chi=(32, 64), contraction_opt='greedy', normalize_initial=False,
                   sweep_kwargs={'local_contraction_opt': 'greedy'},
                   sweep_optimize_kwargs={'n_round_trips': 0, 'n_cycles': 1})
    opt = PepsOptimizer(state, [(np.eye(2), (0, 0))],
        boundary_convergence={'policy': 'on_failure', 'max_chi': (64, 128)}, **options)
    with pytest.warns(RuntimeWarning, match='Increasing PEPS boundary chi'):
        output = opt.run_pepo(operator, infidelity_tol=0.)
    assert calls == [(32, 64), (32, 128), (32, 128)]
    assert len(opt.gates) == 1 and len(opt.get_step_records()) == 1
    reference = PepsOptimizer(state, [], boundary_convergence=False, **options)
    reference.set_boundary_chi(boundary_chi=(32, 128), evaluation_chi=(32, 128))
    expected = reference.run_pepo(operator, infidelity_tol=0.)
    np.testing.assert_allclose(output.to_dense(), expected.to_dense(), atol=1e-10)
    assert opt.get_boundary_growth()['chi'] == (32, 128)


def test_full_pepo_ceiling_preserves_input(monkeypatch):
    state, operator = setup()
    opt = PepsOptimizer(state, [], chi=2, boundary_chi=(32, 64), contraction_opt='greedy',
        normalize_initial=False, boundary_convergence={'policy': 'on_failure', 'max_chi': (32, 64)})
    before = opt.state.to_dense().copy()
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', lambda *a, **kw: {
        'infidelity': -.2, 'norm': 1., 'norm_target': 1., 'overlap': 1.1})
    with pytest.raises(BoundaryChiLimitError):
        opt.run_pepo(operator, timing=True)
    np.testing.assert_array_equal(opt.state.to_dense(), before)
    assert opt.get_step_records() == []
    assert opt.get_timing()['status'] == 'failed'


@pytest.mark.parametrize('mode', ['mps', 'direct'])
def test_direct_three_layer_gradient_matches_dense_directional_derivative(mode):
    torch = pytest.importorskip('torch')
    state, operator = setup('torch')
    operator.add_tag('PEPO')
    target = operator.apply(state, contract=False)
    target.mangle_inner_('__target')
    trainable = state.copy()
    for tensor in trainable:
        tensor.modify(data=tensor.data.clone().requires_grad_(True))
    fit = GlobalOptimizer(trainable, target, chi=(32, 128), target_norm=1., loss_kwargs={
        'mode': mode, 'mode_': 'direct', 'canonize': False, 'cutoff': 0.,
        'layer_tags': ('BRA', 'KET'), 'overlap_layer_tags': ('BRA', 'PEPO', 'KET'),
        'contraction_opt': 'greedy'})
    fit.make_tn_optimizer(autodiff_backend='torch', optimizer='L-BFGS-B', progbar=False)
    value = fit.loss()
    arrays = [t.data for t in trainable]
    gradients = torch.autograd.grad(value, arrays)
    rng = np.random.default_rng(829)
    directions = [torch.as_tensor(rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)) for t in arrays]
    scale = sum(torch.vdot(d.ravel(), d.ravel()).real for d in directions).sqrt()
    directions = [d / scale for d in directions]
    derivative = sum(torch.vdot(g.ravel(), d.ravel()).real for g, d in zip(gradients, directions))
    assert torch.isfinite(derivative)
    losses = []
    for sign in (1., -1.):
        shifted = trainable.copy()
        for tensor, direction in zip(shifted, directions):
            tensor.modify(data=(tensor.data + sign * 1e-5 * direction).detach())
        losses.append(float(fit.loss(state=shifted, mode='exact')))
    assert float(derivative) == pytest.approx((losses[0] - losses[1]) / 2e-5, abs=2e-7, rel=2e-5)


@pytest.mark.parametrize('layers', ['BRA', ('BRA', 'PEPO'), ('BRA', 'PEPO', 'PEPO', 'KET'),
                                   ('BRA', 'MISSING', 'KET')])
def test_invalid_overlap_layers_fail_before_contraction(layers):
    state, operator = setup()
    operator.add_tag('PEPO')
    with pytest.raises(ValueError, match='overlap_layer_tags'):
        GlobalOptimizer(state, operator.apply(state, contract=False)).loss(overlap_layer_tags=layers)


@pytest.mark.parametrize('method', ['als', 'column-lbfgs', 'global'])
def test_three_complete_pepo_steps_from_zero_match_dense_circuit(method):
    backend = 'numpy' if method == 'als' else 'torch'
    operator = example.trotter_pepo(2, 3, .1, backend=backend)
    state = ps_to_peps(2, 3, theta=0.)
    if backend == 'torch':
        state.apply_to_arrays(pytest.importorskip('torch').as_tensor)
    dense_operator = ar.to_numpy(operator.to_dense())
    reference = np.zeros(64, complex)
    reference[0] = 1.
    opt = PepsOptimizer(state, [], chi=2, mode='global' if method == 'global' else 'sweep',
        optimizer={'global': 'L-BFGS-B', 'als': 'als', 'column-lbfgs': 'scipy'}[method],
        boundary_chi=(32, 128), normalize_chi=32, evaluation_chi=(32, 128),
        contraction_opt='greedy', sweep_kwargs={'local_contraction_opt': 'greedy'},
        global_optimize_kwargs={'n': 12, 'autodiff_backend': 'torch'},
        sweep_optimize_kwargs={} if method != 'column-lbfgs' else {
            'optimizer_options': {'n_steps': 12, 'ftol': 1e-12, 'gtol': 1e-10}})
    for _ in range(3):
        reference = dense_operator @ reference
        output = opt.run_pepo(operator, unitary=True)
        vector = ar.to_numpy(output.to_dense()).ravel()
        fidelity = abs(np.vdot(reference, vector))**2 / np.vdot(vector, vector).real
        assert fidelity >= .99
        assert len(opt.get_step_records()) == 1 and output.max_bond() <= 2
        assert 'PEPO' not in output.tags


@pytest.mark.parametrize('cutoff', [None, 1e-9])
def test_full_pepo_boundary_cutoff_defaults_and_explicit_override(monkeypatch, cutoff):
    state, operator = setup()
    original = peps_mod.boundary_infidelity
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs['cutoff'])
        return original(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_infidelity', capture)
    opt = PepsOptimizer(state, [], chi=2, boundary_chi=(32, 128), contraction_opt='greedy',
        normalize_initial=False, boundary_kwargs={} if cutoff is None else {'cutoff': cutoff})
    opt.run_pepo(operator, infidelity_tol=1.)
    assert seen == [0. if cutoff is None else cutoff]


def test_pepsy_direct_loss_reuses_layered_compressor_and_preserves_backend(monkeypatch):
    state, operator = setup('torch')
    operator.add_tag('PEPO')
    target = operator.apply(state, contract=False)
    target.mangle_inner_('__target')
    original = global_mod.contract_layered
    seen = []

    def capture(network, **kwargs):
        layers = kwargs['layer_tags']
        assert kwargs['fit_mode'] == 'direct' and kwargs['fit_layer_mode'] == 'sequential'
        assert kwargs['preserve_backend']
        for tensor in network:
            assert sum(tag in tensor.tags for tag in layers) == 1
        seen.append((tuple(layers), kwargs['chi']))
        result = original(network, **kwargs)
        assert ar.infer_backend(result) == 'torch'
        return result

    monkeypatch.setattr(global_mod, 'contract_layered', capture)
    fit = GlobalOptimizer(state, target, chi=(32, 128), target_norm=1.)
    value = fit.loss(mode='direct', cutoff=0., contraction_opt='greedy',
        layer_tags=('BRA', 'KET'), overlap_layer_tags=('BRA', 'PEPO', 'KET'))
    assert float(value) == pytest.approx(dense_loss(state, target), abs=1e-10)
    assert seen == [(('BRA', 'KET'), 32), (('BRA', 'PEPO', 'KET'), 128)]


def test_column_lbfgs_fits_full_target_one_column_at_a_time(monkeypatch):
    state, operator = setup('torch')
    original = SweepOptimizer.optimize_axis
    seen = []

    def capture(owner, axis, **kwargs):
        seen.append(axis)
        assert kwargs['solver'] == 'scipy'
        assert owner.state_target.num_tensors == 2 * state.num_tensors
        return original(owner, axis, **kwargs)

    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', capture)
    fit = PepsOptimizer(state, [], chi=2, optimizer='scipy', update_style='column',
        boundary_chi=(32, 128), normalize_chi=32, evaluation_chi=(32, 128),
        contraction_opt='greedy', normalize_initial=False,
        sweep_kwargs={'local_contraction_opt': 'greedy'},
        sweep_optimize_kwargs={'n_cycles': 1, 'n_round_trips': 0,
            'optimizer_options': {'n_steps': 12, 'ftol': 1e-12, 'gtol': 1e-10}})
    result = fit.run_pepo(operator, infidelity_tol=0.)
    assert seen == ['y']
    record, = fit.get_step_records()
    assert record['optimizer'] == 'scipy' and record['reason'] == 'optimized'
    assert record['final_infidelity'] < record['pre_infidelity']
    assert record['final_infidelity'] == pytest.approx(dense_loss(result, operator.apply(state)), abs=1e-9)


@pytest.mark.parametrize('known_norm', [False, True])
def test_unitary_step_reuses_input_norm_and_never_contracts_target_norm(monkeypatch, known_norm):
    state, operator = setup()
    # Unit norm is a special case: ensure unitarity also preserves a supplied
    # or measured nonunit input norm, with initial normalization disabled.
    state *= np.sqrt(2.)
    original_norm = peps_mod.boundary_norm
    original_infidelity = peps_mod.boundary_infidelity
    norm_calls = []

    def norm_capture(network, **kwargs):
        assert network.num_tensors == state.num_tensors
        norm_calls.append(kwargs['chi'])
        return original_norm(network, **kwargs)

    def metric_capture(*args, **kwargs):
        assert kwargs['norm_target'] is not None
        return original_infidelity(*args, **kwargs)

    monkeypatch.setattr(peps_mod, 'boundary_norm', norm_capture)
    monkeypatch.setattr(peps_mod, 'boundary_infidelity', metric_capture)
    opt = PepsOptimizer(state, [], chi=2, boundary_chi=(32, 128), normalize_chi=32,
        evaluation_chi=(32, 128), contraction_opt='greedy', normalize_initial=False,
        sweep_kwargs={'local_contraction_opt': 'greedy'},
        sweep_optimize_kwargs={'n_cycles': 1, 'n_round_trips': 0})
    output = opt.run_pepo(operator, unitary=True, infidelity_tol=0.,
        infidelity_kwargs={'norm_target': 2.} if known_norm else None)
    assert norm_calls == ([] if known_norm else [32])
    record, = opt.get_step_records()
    assert record['unitary']
    assert record['final_infidelity'] == pytest.approx(dense_loss(output, operator.apply(state)), abs=1e-9)

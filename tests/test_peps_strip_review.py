"""Regression checks for strip boundary preparation and run controls."""

import autoray as ar
import numpy as np
import pytest

from pepsy.boundary import peps_infidelity
from pepsy.optimizers import PepsOptimizer, SweepOptimizer
from pepsy.optimizers.peps._strip_batch import layered_target
from test_peps_strip_batches import inputs


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('shape,axis,index', [
    ((3, 3), 'x', 1), ((3, 3), 'y', 1),
    ((1, 3), 'x', 0), ((3, 1), 'y', 0),
])
def test_quimb_layered_strip_visits_match_dense_reference(backend, shape, axis, index, monkeypatch):
    state, gate = inputs(backend, shape)
    sites = [(index, j) for j in range(shape[1])] if axis == 'x' else [
        (i, index) for i in range(shape[0])]
    gates = [(gate, (a, b)) for a, b in zip(sites, sites[1:])]
    original = ar.to_numpy(state.to_dense()).copy()
    solve = SweepOptimizer._optimize_axis_slice_with_current_env
    visits = []

    def check(self, position, **kwargs):
        target = ar.to_numpy(self.state_target.to_dense()).ravel()
        def loss():
            vector = ar.to_numpy(self.state.to_dense()).ravel()
            return 1 - abs(np.vdot(target, vector))**2 / (
                np.vdot(target, target).real * np.vdot(vector, vector).real)
        arrays = tuple(self.state_target.arrays)
        before = loss()
        result = solve(self, position, **kwargs)
        after = loss()
        assert result['loss_initial'] == pytest.approx(before, abs=2e-9)
        assert result['loss_final'] == pytest.approx(after, abs=2e-9)
        assert after <= before + 2e-9
        assert all(a is b for a, b in zip(arrays, self.state_target.arrays))
        visits.append(position)
        return result

    monkeypatch.setattr(SweepOptimizer, '_optimize_axis_slice_with_current_env', check)
    driver = PepsOptimizer(
        state, gates, chi=2, boundary_engine='quimb-mps',
        boundary_chi=(32, 64), normalize_chi=32, evaluation_chi=(32, 64),
        contraction_opt='greedy', sweep_kwargs={'local_contraction_opt': 'greedy'},
        sweep_optimize_kwargs={'n_round_trips': 1},
    )
    out = driver.run(measure_infidelity=False)
    assert visits == ([1, 2, 1, 0, 1, 2] if shape == (3, 3) else [0])
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), original)
    vector = ar.to_numpy(out.to_dense()).ravel()
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=2e-9)


def test_quimb_layered_metric_matches_dense_value_and_gradient():
    torch = pytest.importorskip('torch')
    state, _ = inputs('torch', (3, 3))
    rng = np.random.default_rng(4)
    gate = torch.as_tensor(np.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0])
    target = layered_target(state, [(gate, ((0, 1), (1, 1)), None),
                                    (gate, ((1, 1), (2, 1)), None)])
    original = tuple((t.data, tuple(t.tags), t.inds) for t in target)
    state.apply_to_arrays(lambda a: a.detach().clone().requires_grad_(True))
    observed = peps_infidelity(
        state, target, chi=(32, 64), method='mps', contraction_opt='greedy',
        layer_tags=('KET', 'BRA'), norm_target=1.,
    )
    x, y = (tn.to_dense().reshape(-1) for tn in (state, target))
    expected = 1 - torch.vdot(x, y).abs()**2 / (torch.vdot(x, x).real * torch.vdot(y, y).real)
    assert observed['infidelity'] == pytest.approx(float(expected.detach()), abs=2e-9)
    # The public diagnostic is a host scalar; its raw boundary contractions
    # must still preserve autograd through exact site-layer preparation.
    differentiable = 1 - observed['overlap_result'].cost.abs()**2 / observed['norm_result'].cost.real
    torch.testing.assert_close(differentiable, expected, atol=2e-9, rtol=2e-9)
    actual_grad = torch.autograd.grad(differentiable, state.arrays, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, state.arrays)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=2e-8, rtol=2e-8)
    assert all(t.data is a and tuple(t.tags) == tags and t.inds == inds
               for t, (a, tags, inds) in zip(target, original))


@pytest.mark.parametrize('name,value', [('boundary_engine', 'dmrg'), ('boundary_options', {})])
@pytest.mark.parametrize('entrypoint', ['constructor', 'setter', 'driver'])
def test_constructor_only_boundary_options_rejected_in_run_options(name, value, entrypoint):
    state, gate = inputs(shape=(2, 2))
    with pytest.raises(ValueError, match='constructor options'):
        if entrypoint == 'driver':
            PepsOptimizer(
                state, [(gate, ((0, 0), (0, 1)))], chi=2,
                fit_mode='direct', contraction_opt='greedy',
                sweep_optimize_kwargs={name: value},
            ).run(measure_infidelity=False)
        else:
            target = state.copy()
            target.mangle_inner_('_target')
            sweep = SweepOptimizer(
                state, target, chi=8, fit_mode='direct', contraction_opt='greedy',
                optimize_kwargs={name: value} if entrypoint == 'constructor' else None,
            )
            sweep.set_optimize_kwargs(**{name: value})


@pytest.mark.parametrize('options', [
    {'n_cycles': -1}, {'n_cycles': True}, {'n_cycles': 1.5},
    {'n_round_trips': -1}, {'n_round_trips': True},
    {'env_n_iter': 0}, {'env_n_iter': -1}, {'env_n_iter': False},
])
def test_invalid_sweep_budgets_fail_before_metrics_or_updates(options, monkeypatch):
    state, _ = inputs(shape=(2, 2))
    target = state.copy()
    target.mangle_inner_('_target')
    sweep = SweepOptimizer(state, target, chi=8, fit_mode='direct', contraction_opt='greedy')
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid controls must not start contractions or updates')
    monkeypatch.setattr(sweep, '_approx_infidelity_loss', unexpected)
    monkeypatch.setattr(sweep, 'optimize_axis', unexpected)
    with pytest.raises(ValueError, match=next(iter(options))):
        sweep._run_global_sweeps(**options)
    with pytest.raises(ValueError, match=next(iter(options))):
        sweep.set_optimize_kwargs(**options)


@pytest.mark.parametrize('entrypoint', ['optimize_axis', 'optimize_strip'])
def test_local_sweep_rejects_zero_boundary_budget(entrypoint):
    state, _ = inputs(shape=(2, 2))
    target = state.copy()
    target.mangle_inner_('_target')
    sweep = SweepOptimizer(state, target, chi=8, fit_mode='direct', contraction_opt='greedy')
    with pytest.raises(ValueError, match='env_n_iter'):
        getattr(sweep, entrypoint)('y' if entrypoint == 'optimize_axis' else 0, env_n_iter=0)

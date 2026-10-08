"""Reduced ALS updates against explicit small-state and positive-metric references."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import ReducedPair
from pepsy.tensors.contractions import build_optimizer


def fixture(dtype='complex128', shape=(2, 2), bond=2):
    torch = pytest.importorskip('torch')
    state = qtn.PEPS.rand(*shape, bond_dim=bond, dtype=dtype, seed=24)
    state.apply_to_arrays(lambda x: torch.tensor(x))
    state /= state.norm()
    gate = torch.diag(torch.exp(torch.tensor([-.21j, .21j, .21j, -.21j], dtype=getattr(torch, dtype))))
    policy = build_optimizer(max_repeats=1, parallel=False)
    return state, gate, policy


@pytest.mark.parametrize('where', [((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 1), (1, 1))])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_reduction_preserves_exact_gate_and_als_preserves_outside_pair(where, dtype):
    state, gate, policy = fixture(dtype)
    pair = ReducedPair(state, gate, where, chi=2)
    guess, target = pair.initial_states()
    reference = state.gate(gate, where, contract='split', cutoff=0.)
    tol = 2e-5 if dtype == 'complex64' else 2e-11
    np.testing.assert_allclose(target.to_dense().numpy(), reference.to_dense().numpy(), rtol=tol, atol=tol)
    norm, _ = pair.environment(guess, chi=32, contraction_opt=policy,
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0., 'n_iter': 2})
    final, result = pair.optimize(norm, {'max_iterations': 20, 'rtol': 1e-8})
    assert result['success'] and np.isfinite(result['fidelity'])
    assert all(b <= a for a, b in zip(result['loss_history'], result['loss_history'][1:]))
    assert final.max_bond() <= 2
    for site in state.gen_site_coos():
        if site not in where:
            assert final[site].data is state[site].data
    def fidelity(candidate):
        x, y = candidate.to_dense().numpy().ravel(), target.to_dense().numpy().ravel()
        return abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real)
    assert fidelity(final) >= fidelity(guess) - tol
    assert result['fidelity'] == pytest.approx(fidelity(final), abs=tol)


def test_positive_projection_handles_indefinite_environment_and_singular_gauge():
    import torch
    state, gate, _ = fixture()
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    n = pair.na * pair.nb
    norm = torch.eye(n, dtype=torch.complex128)
    norm[0, 0] = -.01
    norm[-1, -1] = 0
    norm[0, 1] = .01j
    out, result = pair.optimize(norm, {'max_iterations': 4})
    assert result['norm_negative_weight'] > 0
    assert result['norm_antihermitian_relative'] > 0
    assert bool(torch.isfinite(out.to_dense()).all())


def test_full_update_driver_checks_chi_and_normalizes_output():
    state, gate, policy = fixture(bond=1)
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1))), (gate, ((0, 1), (1, 1)))],
                        chi=2, mode='full-update', contraction_opt=policy,
                        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0., 'n_iter': 2},
                        boundary_convergence={'max_chi': 16},
                        full_update_kwargs={'max_iterations': 8})
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False, timing=True)
    assert out.max_bond() <= 2
    assert float(out.norm().real) == pytest.approx(1., abs=1e-10)
    records = opt.get_step_records()
    assert len(records) == 2
    assert all(r['boundary_convergence']['converged'] for r in records)
    assert all(r['optimizer_result']['backend'] == 'full-update' for r in records)
    assert all(r['k_2q_batch'] == 1 for r in records)


def test_full_update_rejects_nonadjacent_pairs():
    state, gate, _ = fixture()
    with pytest.raises(ValueError, match='nearest-neighbor'):
        ReducedPair(state, gate, ((0, 0), (1, 1)), chi=2)


@pytest.mark.parametrize('normalize_target', [None, True, False])
def test_single_site_full_update_honors_target_normalization(normalize_target):
    torch = pytest.importorskip('torch')
    state, _, _ = fixture(bond=1)
    identity = torch.eye(2, dtype=torch.complex128)
    opt = PepsOptimizer(
        state, [(2 * identity, (0, 0)), (3 * identity, (1, 1))],
        chi=2, mode='full-update', contraction_opt='greedy', fit_mode='direct',
        normalize_initial=False, normalize_chi=16, boundary_convergence=False,
    )
    out = opt.run(non_unitary=True, normalize_target=normalize_target, normalize_final=False,
                  measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    scale = 6. if normalize_target is False else 1.
    torch.testing.assert_close(out.to_dense(), scale * state.to_dense(), atol=1e-11, rtol=1e-11)


def test_custom_output_normalization_uses_selected_cap(monkeypatch):
    state, _, policy = fixture()
    opt = PepsOptimizer(state, chi=2, mode='full-update', contraction_opt=policy,
                        normalize_kwargs={'balance_bonds': False})
    calls = []
    monkeypatch.setattr(opt, '_normalize_state', lambda state, **kw: calls.append(kw))
    opt._normalize_cached_pair(state, None, chi=12, normalize_kwargs={'chi': 4},
                               phase_site=(0, 0))
    assert calls == [{'normalize_chi': 12,
                      'normalize_kwargs': {'balance_bonds': False, 'chi': 12}}]


def test_fixed_chi_full_update_reuses_norm_only_environment(monkeypatch):
    from pepsy.optimizers.peps import optimizer as module
    state, gate, policy = fixture(shape=(3, 3))
    opt = PepsOptimizer(state, [(gate, ((1, 0), (1, 1))), (gate, ((1, 1), (1, 2)))],
                        chi=2, update_style='two-site', contraction_opt=policy,
                        boundary_chi=16, normalize_chi=16, evaluation_chi=3,
                        boundary_convergence=False, fit_mode='direct',
                        full_update_kwargs={'max_iterations': 4})
    def forbidden(*args, **kwargs):
        pytest.fail('fixed-chi norm-only update contracted a global overlap')
    monkeypatch.setattr(module, 'boundary_infidelity', forbidden)
    handles = []
    environment = opt._full_update_environment
    def tracked(pair, guess, boundary, chi):
        assert chi == 16
        result = environment(pair, guess, boundary, chi)
        handles.append(result[1])
        return result
    monkeypatch.setattr(opt, '_full_update_environment', tracked)
    out = opt.run(k_2q_batch=1, measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    assert handles[0] is handles[1]
    assert handles[-1].mps_b.environment_cache.hits > 0
    assert float(out.norm().real) == pytest.approx(1., abs=1e-10)
    assert all(r['norm_environment_chi'] == 16 for r in opt.get_step_records())


@pytest.mark.parametrize('solver', ['qr', 'quimb'])
def test_shared_solver_routes_preserve_fidelity_and_report_solver(solver):
    state, gate, policy = fixture()
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    guess, target = pair.initial_states()
    norm, _ = pair.environment(guess, chi=32, contraction_opt=policy,
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    out, result = pair.optimize(norm, {'solver': solver, 'max_iterations': 10})
    assert result['solver'] == solver
    x, y = out.to_dense().numpy().ravel(), target.to_dense().numpy().ravel()
    exact = abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real)
    assert result['fidelity'] == pytest.approx(exact, abs=1e-10)


@pytest.mark.parametrize('solver', ['qr', 'quimb'])
@pytest.mark.parametrize('dtype,rtol', [('complex64', 1e-5), ('complex128', 1e-9)])
def test_full_update_native_torch_and_automatic_stopping(monkeypatch, solver, dtype, rtol):
    import torch
    state, gate, policy = fixture(dtype, bond=1)
    # Guard the whole update, including boundary construction and normalization.
    # Scalar diagnostics may synchronize; PEPS tensors and ALS matrices may not.
    for name in ('numpy', 'cpu'):
        original = getattr(torch.Tensor, name)
        def guarded(self, *args, _original=original, **kwargs):
            assert self.numel() == 1, 'bulk Torch tensor transferred to the host'
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, name, guarded)
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=2,
                        mode='full-update', contraction_opt=policy, fit_mode='direct',
                        boundary_convergence=False,
                        full_update_kwargs={'solver': solver, 'max_iterations': 8})
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    report = opt.get_step_records()[0]['optimizer_result']
    assert report['rtol'] == rtol
    assert report['converged']
    assert report['termination_reason'] == 'residual_tolerance'
    assert 1 <= report['iterations'] < 8
    assert all(isinstance(t.data, torch.Tensor) and t.data.dtype == getattr(torch, dtype)
               and t.data.device == state[(0, 0)].data.device for t in out)
    reference = state.gate(gate, ((0, 0), (0, 1)), contract='split', cutoff=0.)
    x, y = out.to_dense().ravel(), reference.to_dense().ravel()
    fidelity = float(abs(torch.vdot(x, y)) ** 2 / (torch.vdot(x, x).real * torch.vdot(y, y).real))
    assert fidelity == pytest.approx(1., abs=rtol)


@pytest.mark.parametrize('solver', ['qr', 'quimb'])
@pytest.mark.parametrize('rtol', [None, 0.])
def test_full_update_can_disable_automatic_stopping(solver, rtol):
    state, gate, policy = fixture(bond=1)
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    guess, _ = pair.initial_states()
    norm, _ = pair.environment(guess, chi=16, contraction_opt=policy,
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    _, result = pair.optimize(norm, {'solver': solver, 'max_iterations': 3, 'rtol': rtol})
    assert result['iterations'] == 3
    assert result['rtol'] == 0.
    assert result['solver_termination_reason'] == 'max_iterations'
    assert not result['solver_converged']
    assert not result['converged']


def test_two_site_style_alias_and_single_site_barrier_preserve_exact_evolution():
    import torch
    state, gate, policy = fixture(bond=1)
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
    gates = [(gate, ((0, 0), (1, 0))), (x, (0, 0)),
             (gate, ((1, 0), (1, 1)))]
    opt = PepsOptimizer(state, gates, chi=4, mode='sweep', update_style='two-site',
                        gate_order='column', contraction_opt=policy,
                        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.},
                        boundary_convergence={'max_chi': 48},
                        full_update_kwargs={'max_iterations': 4})
    out = opt.run(k_2q_batch=1, measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    reference = state
    for g, where in gates:
        reference = reference.gate(g, where, contract='split', cutoff=0.)
    a, b = out.to_dense().ravel(), reference.to_dense().ravel()
    assert float(abs(torch.vdot(a, b)) ** 2 / (torch.vdot(a, a).real * torch.vdot(b, b).real)) == pytest.approx(1.)
    assert float(out.norm().real) == pytest.approx(1.)
    assert opt.last_gate_order['original_steps'] == [1, 2, 3]
    assert all(r['optimizer_result']['solver'] == 'quimb' for r in opt.get_step_records())


@pytest.mark.parametrize('where', [((0, 0), (0, 1)), ((0, 1), (0, 0)), ((1, 1), (0, 1))])
def test_asymmetric_gate_matches_exact_reference_in_both_site_orders(where):
    import torch
    state, _, policy = fixture()
    rng = np.random.default_rng(514)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gate = torch.tensor(np.linalg.qr(matrix)[0], dtype=torch.complex128)
    pair = ReducedPair(state, gate, where, chi=2)
    guess, target = pair.initial_states()
    reference = state.gate(gate, where, contract='split', cutoff=0.)
    np.testing.assert_allclose(target.to_dense().numpy(), reference.to_dense().numpy(), atol=1e-11)
    norm, _ = pair.environment(guess, chi=32, contraction_opt=policy,
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    out, result = pair.optimize(norm, {'max_iterations': 12})
    x, y = out.to_dense().ravel(), target.to_dense().ravel()
    fidelity = float(abs(torch.vdot(x, y)) ** 2 / (torch.vdot(x, x).real * torch.vdot(y, y).real))
    assert result['fidelity'] == pytest.approx(fidelity, abs=1e-10)


@pytest.mark.parametrize('fit_mode', ['direct', 'eff'])
def test_cached_and_fresh_two_site_updates_agree_after_turning_a_corner(fit_mode):
    import torch
    state, gate, policy = fixture(shape=(3, 3))
    gates = [(gate, ((1, 0), (1, 1))), (gate, ((0, 1), (1, 1))),
             (gate, ((1, 1), (1, 2)))]
    outputs = []
    for reuse in (False, True):
        opt = PepsOptimizer(state, gates, chi=2, update_style='two-site',
                            contraction_opt=policy, fit_mode=fit_mode,
                            boundary_kwargs={'n_iter': 4},
                            boundary_convergence={'max_chi': 32, 'reuse_environments': reuse},
                            full_update_kwargs={'max_iterations': 8})
        out = opt.run(k_2q_batch=1, measure_infidelity=False,
                      measure_final_infidelity=False, accept_if_improved=False)
        assert all(r['boundary_convergence']['converged'] for r in opt.get_step_records())
        assert float(out.norm().real) == pytest.approx(1., abs=1e-9)
        outputs.append(out.to_dense().ravel())
    a, b = outputs
    fidelity = float(abs(torch.vdot(a, b)) ** 2 / (torch.vdot(a, a).real * torch.vdot(b, b).real))
    assert fidelity == pytest.approx(1., abs=1e-9)


@pytest.mark.parametrize('normalize_chi', [None, (12, 16)])
def test_fixed_cap_normalization_accepts_default_and_paired_caps(normalize_chi):
    state, gate, policy = fixture()
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=2,
                        mode='full-update', contraction_opt=policy, fit_mode='direct',
                        boundary_convergence=False, normalize_chi=normalize_chi,
                        full_update_kwargs={'max_iterations': 4})
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    assert float(out.norm().real) == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize('scores,accepted', [
    ((-5e-4, .01), True),   # A clipped precheck is not a perfect warm start.
    ((.01, 1.0005), True),  # A clipped postcheck cannot reject the ALS result.
    ((.01, .02), False),   # A valid worse candidate still gets rejected.
])
def test_full_update_acceptance_uses_raw_metric_validity(monkeypatch, scores, accepted):
    from pepsy.optimizers.peps import optimizer as module
    state, gate, policy = fixture()
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=2,
                        mode='full-update', contraction_opt=policy, fit_mode='direct',
                        boundary_convergence=False, full_update_kwargs={'max_iterations': 4})
    values = iter(scores)
    monkeypatch.setattr(module, 'boundary_infidelity', lambda *a, **k: {
        'infidelity': next(values), 'norm_target': 1.,
    })
    if accepted:
        with pytest.warns(RuntimeWarning, match='approximate PEPS infidelity'):
            opt.run()
    else:
        opt.run()
    record = opt.get_step_records()[0]
    assert record['optimized'] is accepted
    assert record['raw_pre_infidelity'] == scores[0]
    assert record['raw_post_infidelity'] == scores[1]
    assert len(record['evaluation_records']) == 2


def test_full_update_postcheck_keeps_successful_retry_cap(monkeypatch):
    from pepsy.optimizers.peps import optimizer as module
    state, gate, policy = fixture()
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=2,
                        mode='full-update', contraction_opt=policy, fit_mode='direct',
                        boundary_convergence=False, full_update_kwargs={'max_iterations': 4})
    caps, values = [], iter([-.1, .02, .01])
    def metric(*args, **kwargs):
        caps.append(kwargs['chi'])
        return {'infidelity': next(values), 'norm_target': 1.}
    monkeypatch.setattr(module, 'boundary_infidelity', metric)
    with pytest.warns(RuntimeWarning, match='retrying'):
        opt.run(infidelity_kwargs={'norm_target': None})
    assert caps == [(8, 10), 20, 20]
    record = opt.get_step_records()[0]
    assert record['optimized']
    assert record['evaluation_chi'] == (8, 10)
    assert record['effective_evaluation_chi'] == 20
    assert all(r['effective_chi'] == 20 for r in record['evaluation_records'])


@pytest.mark.parametrize('dtype,adaptive,normalize_target,exponent', [
    ('complex64', False, True, 0.),
    ('complex128', False, None, 80.),
    ('complex128', True, True, 0.),
    ('complex128', False, False, 0.),
])
def test_nonunitary_target_normalization_controls_output_scale(
    dtype, adaptive, normalize_target, exponent,
):
    import torch
    state, _, policy = fixture(dtype=dtype, bond=1)
    state.exponent = exponent
    gate = 2 * torch.eye(4, dtype=getattr(torch, dtype))
    gates = [(gate, ((0, 0), (0, 1))), (gate, ((0, 1), (1, 1)))]
    original = {site: state[site].data for site in state.gen_site_coos()}
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update',
                        contraction_opt=policy, fit_mode='direct',
                        normalize_initial=False, boundary_convergence=adaptive,
                        full_update_kwargs={'max_iterations': 4})
    out = opt.run(non_unitary=True, normalize_target=normalize_target,
                  normalize_final=False, measure_infidelity=False,
                  measure_final_infidelity=False, accept_if_improved=False)
    expected_norm = 4. if normalize_target is False else 1.
    tol = 2e-5 if dtype == 'complex64' else 1e-10
    assert float(out.norm().real) == pytest.approx(expected_norm, abs=tol)
    assert out[(1, 0)].data is original[(1, 0)]
    assert state.exponent == exponent
    assert all(state[site].data is data for site, data in original.items())
    expected = state.copy()
    expected.exponent = 0.
    expected /= expected.norm()
    np.testing.assert_allclose(out.to_dense().numpy(),
                               expected_norm * expected.to_dense().numpy(), atol=tol)


@pytest.mark.parametrize('gauge', [False, True])
def test_als_receives_hermitian_positive_environment(monkeypatch, gauge):
    import torch
    from pepsy.optimizers.peps import _full_update as module
    state, gate, _ = fixture()
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    n = pair.na * pair.nb
    norm = torch.eye(n, dtype=torch.complex128)
    norm[0, 0] = -.02
    norm[0, 1] = .1j
    received = []
    solve = module.solve_reduced_als
    def checked(problem, **kwargs):
        metric = problem.environment.data.permute(2, 3, 0, 1).reshape(n, n)
        torch.testing.assert_close(metric, metric.mH, atol=1e-12, rtol=1e-12)
        assert float(torch.linalg.eigvalsh(metric).min()) >= -1e-12
        if not gauge:
            values, vectors = torch.linalg.eigh((norm + norm.mH) / 2)
            expected = (vectors * values.clamp_min(0)) @ vectors.mH
            expected /= torch.linalg.vector_norm(expected)
            torch.testing.assert_close(metric, expected)
        received.append(metric)
        return solve(problem, **kwargs)
    monkeypatch.setattr(module, 'solve_reduced_als', checked)
    _, report = pair.optimize(norm, {'gauge': gauge, 'max_iterations': 4})
    assert len(received) == 1
    assert report['norm_negative_weight'] > 0
    assert report['norm_antihermitian_relative'] > 0


def test_independent_environment_gauges_whiten_a_separable_metric(monkeypatch):
    import torch
    from pepsy.optimizers.peps import _full_update as module
    state, gate, _ = fixture()
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    rng = np.random.default_rng(42)
    def positive(size):
        matrix = torch.tensor(rng.normal(size=(size, size)) +
                              1j * rng.normal(size=(size, size)))
        return matrix.mH @ matrix + .01 * torch.eye(size)
    norm = torch.kron(positive(pair.na), positive(pair.nb))
    solve = module.solve_reduced_als
    def checked(problem, **kwargs):
        n = pair.na * pair.nb
        metric = problem.environment.data.permute(2, 3, 0, 1).reshape(n, n)
        expected = torch.eye(n, dtype=norm.dtype) / n**.5
        torch.testing.assert_close(metric, expected, atol=1e-10, rtol=1e-10)
        return solve(problem, **kwargs)
    monkeypatch.setattr(module, 'solve_reduced_als', checked)
    _, report = pair.optimize(norm, {'max_iterations': 4})
    assert report['environment_gauge_applied']


@pytest.mark.parametrize('axis', ['row', 'column'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_strip_cache_matches_fresh_norms_and_invalidates_local_changes(axis, dtype):
    from pepsy.boundary._reuse import StripEnvironmentCache
    state, gate, policy = fixture(dtype=dtype, shape=(4, 4))
    tol = 2e-5 if dtype == 'complex64' else 1e-11
    cache, boundary = StripEnvironmentCache(), None
    def sites(i):
        return ((i, 1), (i + 1, 1)) if axis == 'column' else ((1, i), (1, i + 1))
    def environment(state, where, boundary=None, cache=None):
        pair = ReducedPair(state, gate, where, chi=2)
        guess, _ = pair.initial_states()
        norm, handle = pair.environment(guess, boundary=boundary, chi=16,
                                        contraction_opt=policy,
                                        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.},
                                        strip_cache=cache)
        return norm / norm.norm(), handle
    # Revisit an unchanged strip, move along it, then change a spectator in place.
    for i in (0, 0, 1, 2):
        actual, boundary = environment(state, sites(i), boundary, cache)
        expected, _ = environment(state, sites(i))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=tol)
    assert cache.hits > 0
    before = cache.rebuilds
    spectator = (0, 1) if axis == 'column' else (1, 0)
    state[spectator].data[..., 0].mul_(1.3 + .2j)
    actual, boundary = environment(state, sites(2), boundary, cache)
    expected, _ = environment(state, sites(2))
    assert cache.rebuilds > before
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=tol)
    # Changes outside the strip alter the boundary MPS feeding each slice.
    state[(0, 0)].data[..., 1].mul_(.7 + .3j)
    actual, _ = environment(state, sites(2), boundary, cache)
    expected, _ = environment(state, sites(2))
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=tol)


def test_column_full_updates_reuse_strip_contractions_without_changing_state(monkeypatch):
    from pepsy.boundary._reuse import StripEnvironmentCache
    state, gate, policy = fixture(shape=(4, 3))
    gates = [(gate, ((i, 1), (i + 1, 1))) for i in range(3)]
    outputs = []
    for reuse in (True, False):
        opt = PepsOptimizer(state, gates, chi=2, mode='full-update',
                            contraction_opt=policy, fit_mode='direct',
                            boundary_kwargs={'cutoff': 0.}, boundary_chi=16,
                            normalize_chi=16, boundary_convergence=False,
                            full_update_kwargs={'max_iterations': 4})
        cache = opt._full_update_strip_cache = StripEnvironmentCache()
        if not reuse:
            monkeypatch.setattr(cache, 'reduce', lambda strip, **kw: strip)
        out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                      accept_if_improved=False)
        if reuse:
            assert opt.get_step_records()[-1]['optimizer_result']['strip_environment_reuse']['hits'] > 0
        outputs.append(out.to_dense().numpy())
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-10)

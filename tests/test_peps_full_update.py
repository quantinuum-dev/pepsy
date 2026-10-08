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

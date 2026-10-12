"""Fixed-window targets, supported-subspace solves, and bounded dense work."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import options
from pepsy.optimizers.peps._strip_solve import solve_positive
from pepsy.optimizers.peps._strip_update import refine_strip
from pepsy.optimizers.peps._strip_cursor import StripSweepCursor


def convert(backend):
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip('CUDA unavailable')
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
        return cp.asarray
    return pytest.importorskip('torch').tensor


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
def test_pinv_hermitianizes_and_discards_negative_and_null_support(backend):
    native = convert(backend)
    matrix = native(np.diag([1.+.5j, 1e-14, -.1]).astype('complex128'))
    rhs = native(np.array([[1., .5j], [1e-8, 2e-8j], [1e-8j, 0.]], dtype='complex128'))
    answer, report = solve_positive(matrix, rhs, rcond=1e-12)
    np.testing.assert_allclose(ar.to_numpy(answer), [[1., .5j], [0., 0.], [0., 0.]], atol=1e-14)
    assert report['retained_rank'] == 1 and report['discarded_rhs_relative'] > 0
    assert report['negative_weight'] == pytest.approx(.1)
    rng = np.random.default_rng(12)
    factors = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(2)]
    factors[0][:, 1] *= .01
    root = np.kron(*factors)
    norm = root.conj().T @ root
    exact = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    answer, report = solve_positive(native(norm), native(norm @ exact), rcond=1e-12)
    np.testing.assert_allclose(ar.to_numpy(answer), exact, atol=1e-8, rtol=1e-8)
    assert report['condition'] > 1000
    assert ar.infer_backend(answer) == backend and answer.dtype == matrix.dtype


def fixture(backend='torch', dtype='complex128', seed=24):
    native = convert(backend)
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype=dtype, seed=seed)
    state.apply_to_arrays(native)
    state /= state.norm()
    gate = native(np.diag(np.exp([-.3j, .3j, .3j, -.3j])).astype(dtype))
    gates = [(gate, ((r, c), (r, c+1))) for r in range(3) for c in range(2)]
    gates += [(gate, ((r, c), (r+1, c))) for c in range(3) for r in range(2)]
    return state, gates


def infidelity(a, b):
    x, y = (ar.to_numpy(t.to_dense()).ravel() for t in (a, b))
    return float(1 - abs(np.vdot(x, y))**2 / (np.vdot(x, x).real * np.vdot(y, y).real))


def optimizer(state, gates, **controls):
    return PepsOptimizer(state, gates, chi=2, mode='full-update', normalize_initial=False,
                         contraction_opt='greedy', fit_mode='direct',
                         boundary_kwargs={'cutoff': 0.}, boundary_chi=32,
                         normalize_chi=32, evaluation_chi=32, boundary_convergence=False,
                         full_update_kwargs={'max_iterations': 4, **controls})


def run(opt, **kwargs):
    return opt.run(measure_infidelity=False, measure_final_infidelity=False,
                   accept_if_improved=False, **kwargs)


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_fixed_layer_refinement_improves_the_exact_layer(backend, dtype, monkeypatch):
    state, gates = fixture(backend, dtype)
    target = state.copy()
    for gate, where in gates:
        target = target.gate(gate, where, contract='split', cutoff=0., max_bond=4)
    baseline = run(optimizer(state, gates))
    opt = optimizer(state, gates, refine_scope='layer', refine_sweeps=2,
                    refine_solver='pinv', refine_balance=True)
    captured, original = [], opt._full_update_refine
    def capture(exact, *args, **kwargs):
        captured.append(exact.copy())
        return original(exact, *args, **kwargs)
    monkeypatch.setattr(opt, '_full_update_refine', capture)
    callbacks = []
    out = run(opt, step_callback=callbacks.append)
    tol = 5e-5 if dtype == 'complex64' else 1e-9
    assert len(captured) == 1 and infidelity(captured[0], target) < tol
    assert infidelity(out, target) < infidelity(baseline, target) - .01
    report = opt.get_step_records()[-1]['layer_refinement']
    assert report['accepted'] and report['scope'] == 'layer'
    assert report['start_step'] == 1 and report['end_step'] == 12
    assert report['infidelity_after'] == pytest.approx(infidelity(out, target), abs=tol)
    assert len(report['cycles'][0]['strips']) == 6
    assert len(callbacks) == 12 and callbacks[-1]['layer_refinement'] == report
    assert out.max_bond() <= 2
    assert float(ar.to_numpy(out.norm()).real) == pytest.approx(1., abs=tol)
    for site in state.gen_site_coos():
        assert out[site].inds == state[site].inds
        assert out[site].data.device == state[site].data.device
        assert out[site].data.dtype == state[site].data.dtype


def test_dense_limit_skips_before_environment_allocation(monkeypatch):
    from pepsy.optimizers.peps import _strip_update
    state, gates = fixture()
    gate, where = gates[0]
    target = state.gate(gate, where, contract='split', cutoff=0.)
    def forbidden(*args, **kwargs):
        pytest.fail('dense-size rejection must precede boundary or matrix construction')
    monkeypatch.setattr(_strip_update, 'boundary_strip', forbidden)
    out, report, _, _ = refine_strip(state, target, key=('x', 1), chi=32,
        contraction_opt='greedy', boundary_kwargs={}, sweeps=1, rtol=None, max_matrix_size=2)
    assert out is state and report['termination_reason'] == 'matrix_size_limit'


def test_cursor_has_linear_contraction_count_and_tracks_active_updates():
    network = qtn.MPS_rand_state(9, 2, seed=2).make_norm()
    network.retag_({f'I{i}': f'X{i}' for i in range(9)})
    cursor = StripSweepCursor(network, axis='X', length=9, optimize='greedy')
    for reverse in (False, True):
        cursor.start(reverse=reverse)
        for position in (range(8, -1, -1) if reverse else range(9)):
            reduced = cursor.local(position)
            expected = network.contract(all, optimize='greedy')
            assert reduced.contract(all, optimize='greedy') == pytest.approx(expected, rel=1e-10)
            for tensor in network.select(f'X{position}'):
                tensor.modify(data=tensor.data * 1.01)
    assert cursor.rebuilds == 2 * 2 * (9 - 1)
    assert cursor.hits > 0


@pytest.mark.parametrize('controls', [{'refine_scope': 'bad'}, {'refine_solver': 'bad'},
    {'refine_balance': 'yes'}, {'refine_max_matrix_size': False}, {'refine_max_matrix_size': 0}])
def test_refinement_options_reject_invalid_controls(controls):
    with pytest.raises((ValueError, TypeError)):
        options(controls)


def test_repeated_bonds_split_exact_windows_and_trailing_singles_stay_ordered(monkeypatch):
    state, original = fixture()
    # An interior/edge bond can grow beyond 2D; a corner bond saturates at 4.
    gate, where = original[2]
    x = state[0, 0].data.new_tensor([[0., 1.], [1., 0.]])
    gates = [(gate, where), (2*x, (0, 0)), (gate, where), (gate, where), (3*x, (2, 2))]
    opt = optimizer(state, gates, refine_scope='layer', refine_sweeps=1, refine_solver='pinv')
    collect, windows = opt._collect_auto_batch_target, []
    def checked(start, **kwargs):
        before = opt.state.copy()
        result = collect(start, **kwargs)
        entries, _, _, target, _, limit = result
        for g, sites, _ in entries:
            before = before.gate(g, sites, contract='split' if isinstance(sites[0], tuple) else True,
                                 cutoff=0.)
        np.testing.assert_allclose(ar.to_numpy(target.to_dense()), ar.to_numpy(before.to_dense()),
                                   atol=1e-10, rtol=1e-10)
        assert isinstance(entries[-1][1][0], tuple) and target.max_bond() <= limit
        windows.append((start+1, result[2]))
        return result
    monkeypatch.setattr(opt, '_collect_auto_batch_target', checked)
    out = run(opt, non_unitary=True)
    assert windows == [(1, 1), (3, 3), (4, 4)]
    assert len(opt.get_step_records()) == 3
    assert float(out.norm().real) == pytest.approx(1., abs=1e-9)
    assert [(r['layer_refinement']['start_step'], r['layer_refinement']['end_step'])
            for r in opt.get_step_records()] == windows


def test_whole_layer_rejects_a_worsening_cycle_without_mutating_input(monkeypatch):
    from pepsy.optimizers.peps import _layer_update
    state, _ = fixture()
    arrays = [t.data for t in state]
    def worsening(candidate, target, **kwargs):
        candidate = candidate.copy()
        tensor = candidate[0, 0]
        data = tensor.data.clone()
        data[..., 0] *= .1
        tensor.modify(data=data)
        return candidate, {'accepted': True}, None, None
    monkeypatch.setattr(_layer_update, 'refine_strip', worsening)
    out, report, _, _ = _layer_update.refine_layer(
        state, state.copy(), chi=32, contraction_opt='greedy',
        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, sweeps=2, rtol=None,
    )
    assert out is state and not report['accepted']
    assert report['termination_reason'] == 'cycle_rejected' and report['sweeps'] == 1
    assert all(t.data is a for t, a in zip(state, arrays))


def test_adaptive_layer_caps_and_global_metric_match_dense_reference():
    state, gates = fixture()
    gates = gates[:2]
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update', normalize_initial=False,
        contraction_opt='greedy', fit_mode='direct', boundary_kwargs={'cutoff': 0.},
        boundary_convergence={'start_chi': (4, 16), 'max_chi': (32, 48), 'patience': 1},
        full_update_kwargs={'max_iterations': 4, 'refine_scope': 'layer',
                            'refine_sweeps': 1, 'refine_solver': 'pinv'})
    out = opt.run(non_unitary=True, accept_if_improved=False)
    report = opt.get_step_records()[-1]['layer_refinement']
    assert report['boundary_convergence']['converged']
    assert (report['chi'], report['overlap_chi']) == report['boundary_convergence']['chi']
    target = state.copy()
    for g, where in gates:
        target = target.gate(g, where, contract='split', cutoff=0.)
    assert report['infidelity_after'] == pytest.approx(infidelity(out, target), abs=1e-9)


def test_layer_resource_limit_retains_fu_and_skips_global_checks(monkeypatch):
    from pepsy.optimizers.peps import _layer_update
    state, gates = fixture()
    gates = gates[:2]
    reference = run(optimizer(state, gates))
    def forbidden(*args, **kwargs):
        pytest.fail('oversized layer must skip global checks and dense solves')
    monkeypatch.setattr(_layer_update, 'peps_infidelity', forbidden)
    opt = optimizer(state, gates, refine_scope='layer', refine_sweeps=1, refine_max_matrix_size=2)
    monkeypatch.setattr(opt, '_collect_auto_batch_target', forbidden)
    out = run(opt)
    assert opt.get_step_records()[-1]['layer_refinement']['termination_reason'] == 'matrix_size_limit'
    np.testing.assert_allclose(ar.to_numpy(out.to_dense()), ar.to_numpy(reference.to_dense()),
                               rtol=1e-10, atol=1e-10)

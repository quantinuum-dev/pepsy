"""Guarded Torch ALS, compiled contractions and independent local fidelity."""

from dataclasses import replace
import math
from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TorchLinalgConfig
from pepsy.bp import (
    LoopClusterUpdateTracker, exact_reduced_update_problem,
    prepare_reduced_bond_pair, solve_reduced_als,
)
from pepsy.bp._reduced_torch import _guarded_solve
from pepsy.bp.reduced_update import _validated_local_fidelity, _with_local_fidelity
from pepsy.operators import gate_loop_cluster, renorm_gauge
from pepsy.tensors import id_to_pepo

torch = pytest.importorskip('torch')


def problem_on(device='cpu', *, exponent=.3):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=445)
    state.exponent = exponent
    state.apply_to_arrays(lambda a: torch.as_tensor(a, device=device))
    pair = prepare_reduced_bond_pair(state, where=((0, 0), (0, 1)), run_bp=False)
    rng = np.random.default_rng(56)
    gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))[0]
    gate = torch.as_tensor(gate*(.7+.3j), device=device)
    return exact_reduced_update_problem(pair, gate, psd_project=False, optimize='greedy')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('kind', ['positive', 'near_singular', 'indefinite'])
def test_guarded_solve_preserves_the_spectral_floor(device, dtype, kind):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    values = [1., .8, .4, {'positive': .2, 'near_singular': 1e-10, 'indefinite': -.1}[kind]]
    gram = torch.diag(torch.tensor(values, dtype=getattr(torch, dtype), device=device))
    rhs = torch.tensor([[1.+.2j, .3j]]*4, dtype=getattr(torch, dtype), device=device)
    counts = dict(cholesky=0, eigh=0)
    actual = _guarded_solve(gram, rhs, 1e-6, counts)
    eigenvalues, vectors = torch.linalg.eigh(gram)
    expected = vectors @ ((vectors.mH @ rhs) / eigenvalues.clamp_min(eigenvalues.max()*1e-6)[:, None])
    torch.testing.assert_close(actual, expected, rtol=1e-5 if dtype == 'complex64' else 1e-12, atol=1e-7)
    assert counts['cholesky'] == (kind == 'positive')
    assert counts['eigh'] == (kind != 'positive')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('rank', [1, 3])
def test_torch_als_matches_quimb_and_independent_full_state_fidelity(device, rank):
    problem = problem_on(device)
    options = dict(max_bond=rank, max_iterations=6, tol=0., monitor_convergence=True,
                   measure_fidelity=True)
    reference = solve_reduced_als(problem, solver='quimb', **options)
    actual = solve_reduced_als(problem, solver='torch', **options)
    torch.testing.assert_close(actual.theta(), reference.theta(), rtol=2e-7, atol=2e-9)
    assert actual.costs[-1] == pytest.approx(reference.costs[-1], rel=1e-8, abs=1e-10)
    assert all(b <= a + 1e-10*problem.target_norm for a, b in zip(actual.costs, actual.costs[1:]))
    expected_state = problem.pair.gate_target_tn(problem.gate).to_dense()
    actual_state = problem.pair.reconstruct_tn(actual.left, actual.right).to_dense()
    a, b = [np.asarray(ar.to_numpy(x)).reshape(-1) for x in (expected_state, actual_state)]
    fidelity = abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
    assert actual.local_fidelity == pytest.approx(fidelity, abs=2e-12)
    assert reference.local_fidelity == pytest.approx(fidelity, abs=2e-10)
    assert actual.solver_report['cholesky'] + actual.solver_report['eigh'] == 12
    # Fidelity must retain complex overlap information, independently of the
    # fitted operator's amplitude or global exponent.
    scaled = replace(actual, left=actual.left*(1.4+.7j), local_fidelity=None)
    measured = _with_local_fidelity(problem, scaled, True)
    assert measured.local_fidelity == pytest.approx(fidelity, abs=2e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('backend', ['aot_eager'])
def test_compiled_contractions_match_eager_als(device, backend):
    problem = problem_on(device)
    options = dict(solver='torch', max_bond=1, max_iterations=3,
                   tol=0., monitor_convergence=True, measure_fidelity=True)
    reference = solve_reduced_als(problem, **options)
    for _ in range(2):
        actual = solve_reduced_als(problem, torch_backend=backend, **options)
        torch.testing.assert_close(actual.theta(), reference.theta(), rtol=1e-9, atol=1e-10)
        assert actual.local_fidelity == pytest.approx(reference.local_fidelity, abs=2e-12)
    assert actual.solver_report['torch_backend'] == backend


def test_optional_fidelity_does_not_run_when_disabled(monkeypatch):
    import pepsy.bp.reduced_update as module

    def forbidden(*args, **kwargs):
        pytest.fail('fidelity was evaluated without being requested')
    monkeypatch.setattr(module, '_validated_local_fidelity', forbidden)
    for solver in ('quimb', 'torch'):
        result = solve_reduced_als(problem_on(), solver=solver, max_bond=1, max_iterations=1)
        assert result.local_fidelity is None and result.fidelity_status == 'not_requested'


def test_fidelity_rejects_bad_metrics_and_avoids_scale_overflow():
    assert _validated_local_fidelity(1e200, 1e200, 1e200, 1e-16) == (1., 'valid')
    for norm, target, overlap, status in ((0., 1., 0., 'nonpositive_norm'),
                                        (-1., 1., 1., 'nonpositive_norm'),
                                        (1., 1., 2., 'unphysical_metric'),
                                        (1., 1., math.inf, 'nonfinite_metric')):
        assert _validated_local_fidelity(norm, target, overlap, 1e-16) == (None, status)


def test_tracker_survives_stream_chunks_and_invalidates_missing_fidelities():
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=74)
    state.apply_to_arrays(torch.as_tensor)
    where = ((0, 0), (0, 1))
    gate = torch.diag(torch.exp(.2j*torch.tensor([1., -1., -1., 1.], dtype=torch.float64)))
    gauges = {}
    tracker = LoopClusterUpdateTracker()
    options = dict(gauges=gauges, max_bond=1, tracker=tracker, strip_exponent=True,
                   include_full_system=True, tree_reduction=False, optimize='greedy',
                   als_opts=dict(solver='torch', max_iterations=4, monitor_convergence=True))
    for _ in range(2):
        state = gate_loop_cluster(state, gate, where=where, **options)
    assert tracker.pair_updates == 2
    expected = math.prod(row['local_fidelity'] for row in tracker.records)
    assert tracker.accumulated_local_fidelity == pytest.approx(expected)
    assert tracker.accumulated_local_infidelity == pytest.approx(1-expected)
    for record in tracker.records:
        assert record['where'] == where and record['solver'] == 'torch'
        assert record['local_infidelity'] == pytest.approx(1-record['local_fidelity'])
    options['als_opts']['measure_fidelity'] = False
    gate_loop_cluster(state, gate, where=where, **options)
    assert tracker.summary()['missing_fidelities'] == 1
    assert tracker.accumulated_local_fidelity is None
    tracker.reset()
    assert tracker.records == [] and tracker.accumulated_local_fidelity == 1.


def test_batched_stream_scale_tracking_matches_manual_per_pair_bookkeeping():
    rng = np.random.default_rng(792)
    bonds = (((0, 0), (0, 1)), ((0, 1), (0, 2)), ((1, 0), (1, 1)),
             ((1, 1), (1, 2)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)))
    entries = []
    for i in range(100):
        gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))[0]
        entries.append((torch.as_tensor(gate), bonds[i % len(bonds)], 'upper' if i % 2 else 'lower'))
    state = id_to_pepo(2, 3)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    options = dict(max_bond=2, cluster_mode='loopunion', base_radius=4, fillin=True,
                   psd_project=False, optimize='greedy',
                   als_opts={'solver': 'torch', 'max_iterations': 2, 'monitor_convergence': True})
    manual = state.copy()
    weights = {}
    for entry in entries:
        manual = gate_loop_cluster(manual, [entry], gauges=weights, **options)
        renorm_gauge(manual, weights, entry[1], smudge=0.)
    batched_weights = {}
    batched = gate_loop_cluster(state, entries, gauges=batched_weights,
                               strip_exponent=True, inplace=False, **options)
    for actual, expected in ((batched, batched_weights), (manual, weights)):
        actual.gauge_simple_insert(expected)
    torch.testing.assert_close(batched.to_dense(), manual.to_dense(), rtol=1e-8, atol=1e-11)
    assert all(torch.isfinite(t.data).all() for t in batched.tensors)


@pytest.mark.parametrize('backend', [None, 'aot_eager'])
@pytest.mark.parametrize('stabilized', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_torch_als_retains_gate_derivatives(backend, stabilized, device):
    template = problem_on(device)
    control = torch.tensor(.17, dtype=torch.float64, device=device, requires_grad=True)

    def loss(value):
        phase = torch.exp(1j*value*torch.tensor([1., -1., -1., 1.], dtype=torch.float64, device=device))
        problem = exact_reduced_update_problem(template.pair, torch.diag(phase),
                                               psd_project=False, optimize='greedy')
        with TorchLinalgConfig(stabilized=stabilized).activated():
            result = solve_reduced_als(problem, max_bond=1, max_iterations=2, tol=0.,
                                       solver='torch', torch_backend=backend)
        return result.theta().abs().square().sum()

    actual = torch.autograd.grad(loss(control), control)[0]
    step = 1e-5
    expected = (loss(control.detach()+step)-loss(control.detach()-step))/(2*step)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('solver', ['quimb', 'qr', 'normal'])
def test_optional_fidelity_on_numpy_solvers_matches_dense_state(solver):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=114)
    pair = prepare_reduced_bond_pair(state, where=((0, 0), (0, 1)), run_bp=False)
    gate = np.diag(np.exp(.17j*np.array([1., -1., -1., 1.])))
    problem = exact_reduced_update_problem(pair, gate, optimize='greedy')
    solution = solve_reduced_als(problem, solver=solver, max_bond=1,
                                 max_iterations=3, measure_fidelity=True)
    a = np.asarray(pair.gate_target_tn(gate).to_dense()).ravel()
    b = np.asarray(pair.reconstruct_tn(solution.left, solution.right).to_dense()).ravel()
    expected = abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
    assert solution.local_fidelity == pytest.approx(expected, abs=2e-12)


def test_tracker_totals_handle_tiny_and_zero_fidelities_without_retaining_records():
    tracker = LoopClusterUpdateTracker(keep_records=False)
    solution = solve_reduced_als(problem_on(), solver='torch', max_bond=1, max_iterations=1)
    for fidelity in (1e-200, 0.):
        tracker.record(SimpleNamespace(solution=replace(solution, local_fidelity=fidelity,
                       local_infidelity=1-fidelity, fidelity_status='valid'),
                       pair=SimpleNamespace(where=((0, 0), (0, 1))),
                       problem=SimpleNamespace(which=None)))
        assert tracker.accumulated_local_fidelity == pytest.approx(fidelity, rel=1e-12, abs=0.)
    assert tracker.records == [] and tracker.pair_updates == 2
    assert tracker.accumulated_local_infidelity == 1.


@pytest.mark.parametrize('options', [dict(regularization=.1), dict(quimb_opts={'tol': .1}),
                                   dict(autodiff_opts={'steps': 2}), dict(torch_backend='inductor')])
def test_torch_solver_rejects_unsupported_options(options):
    with pytest.raises(ValueError):
        solve_reduced_als(problem_on(), solver='torch', **options)


def test_aot_capture_handles_many_patch_shapes():
    from pepsy.bp._reduced_torch import _kernels

    eager, *_ = _kernels(None)
    compiled, *_ = _kernels('aot_eager')
    for size in range(1, 11):
        target = torch.ones((size, 2, 2, 1), dtype=torch.complex128)
        environment = torch.eye(size, dtype=torch.complex128).reshape(size, 1, size, 1)
        torch.testing.assert_close(compiled(environment, target), eager(environment, target))
    for context in (torch.no_grad, torch.enable_grad, torch.inference_mode):
        with context():
            target = torch.ones((2, 2, 2, 1), dtype=torch.complex128)
            environment = torch.eye(2, dtype=torch.complex128).reshape(2, 1, 2, 1)
            torch.testing.assert_close(compiled(environment, target), eager(environment, target))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_aot_capture_preserves_backward_when_autocast_changes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from pepsy.bp._reduced_torch import _kernels

    eager, *_ = _kernels(None)
    compiled, *_ = _kernels('aot_eager')
    rng = torch.Generator().manual_seed(416)
    environment = torch.rand((2, 2, 2, 2), generator=rng).to(device).requires_grad_()
    target = torch.rand((2, 2, 2, 2), generator=rng).to(device).requires_grad_()
    low_precision = torch.bfloat16 if device == 'cpu' else torch.float16
    for enabled in (False, True, False):
        results = []
        for function in (eager, compiled):
            with torch.autocast(device, dtype=low_precision, enabled=enabled):
                output = function(environment, target)
                loss = output.square().sum()
            gradients = torch.autograd.grad(loss, (environment, target))
            results.append((output, *gradients))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected)


def test_torch_als_preserves_environment_scaling_and_warm_start():
    problem = problem_on()
    options = dict(solver='torch', max_bond=1, max_iterations=4,
                   monitor_convergence=True, measure_fidelity=True)
    initial = solve_reduced_als(problem, **{**options, 'max_iterations': 1})
    warm = (initial.left, initial.right)
    reference = solve_reduced_als(problem, initial_factors=warm, **options)
    for scale in (1e-40, 1e40):
        environment = problem.environment.copy()
        environment.modify(data=environment.data*scale)
        scaled_problem = replace(problem, environment=environment)
        actual = solve_reduced_als(scaled_problem, initial_factors=warm, **options)
        torch.testing.assert_close(actual.theta(), reference.theta(), rtol=1e-8, atol=1e-10)
        assert actual.iterations == reference.iterations
        assert actual.local_fidelity == pytest.approx(reference.local_fidelity, abs=2e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', ['float64', 'complex64', 'complex128'])
def test_rectangular_torch_als_matches_quimb_and_full_state(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = torch.Generator().manual_seed(614)
    kind = getattr(torch, dtype)
    # Both endpoints have a spectator leg; unequal physical and exterior
    # dimensions expose permutations hidden by the square qubit fixtures.
    state = qtn.PEPS([[
        torch.diag(torch.tensor([.7, 1.3], dtype=kind)),
        torch.randn((2, 2, 2), generator=rng, dtype=kind),
        torch.randn((3, 2, 3), generator=rng, dtype=kind),
        torch.diag(torch.tensor([.6, .8, 1.4], dtype=kind)),
    ]])
    state.apply_to_arrays(lambda a: a.to(device))
    pair = prepare_reduced_bond_pair(state, where=((0, 1), (0, 2)), run_bp=False)
    gate = torch.linalg.qr(torch.randn((6, 6), generator=rng, dtype=kind)).Q.to(device)
    problem = exact_reduced_update_problem(pair, gate, psd_project=False, optimize='greedy')
    assert problem.shape == (2, 2, 3, 3)
    options = dict(max_iterations=4, tol=0., monitor_convergence=True, measure_fidelity=True)
    tolerance = 3e-5 if dtype == 'complex64' else 1e-10
    for rank in (1, 3):
        reference = solve_reduced_als(problem, solver='quimb', max_bond=rank, **options)
        for backend in (None, 'aot_eager'):
            actual = solve_reduced_als(problem, solver='torch', torch_backend=backend,
                                       max_bond=rank, **options)
            torch.testing.assert_close(actual.theta(), reference.theta(),
                                       rtol=tolerance, atol=tolerance)
            target = pair.gate_target_tn(gate).to_dense().reshape(-1)
            fitted = pair.reconstruct_tn(actual.left, actual.right).to_dense().reshape(-1)
            expected = (torch.vdot(fitted, target).abs().square()
                        / (torch.vdot(fitted, fitted).real * torch.vdot(target, target).real))
            assert actual.local_fidelity == pytest.approx(float(expected), abs=tolerance)

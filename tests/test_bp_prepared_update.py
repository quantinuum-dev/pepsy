"""Prepared cluster replay preserves sequential gates, fresh environments and gradients."""

import math

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TorchLinalgConfig
from pepsy.bp import ContractionPlanCache, LoopClusterUpdateTracker, PreparedLoopClusterUpdate
from pepsy.operators import gate_loop_cluster
from pepsy.tensors import id_to_pepo

torch = pytest.importorskip('torch')


def initial(device='cpu', random=False):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = (qtn.PEPO.rand(2, 3, bond_dim=2, phys_dim=2, dtype='complex128', seed=177)
             if random else id_to_pepo(2, 3))
    state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128, device=device))
    return state


def gate(theta, width, device='cpu'):
    value = torch.as_tensor(theta, dtype=torch.float64, device=device)
    if width == 1:
        identity = torch.eye(2, dtype=torch.complex128, device=device)
        x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128, device=device)
        return value.cos()*identity-1j*value.sin()*x
    return torch.diag(torch.exp(-1j*value*torch.tensor(
        [1., -1., -1., 1.], dtype=torch.float64, device=device)))


def physical(state, gauges):
    state = state.copy()
    state.gauge_simple_insert(gauges)
    return state.to_dense()


def reference(state, gates, schedule, gauges, tracker=None, *, iterations=4, psd=False):
    return gate_loop_cluster(
        state, [(g, s[0] if len(s) == 1 else s, leg) for g, (s, leg) in zip(gates, schedule)],
        gauges=gauges, max_bond=2, cluster_mode='loopunion', base_radius=4, fillin=True,
        max_loop_size=0, psd_project=psd, optimize='greedy', plan_cache=ContractionPlanCache(),
        strip_exponent=True, tracker=tracker, return_results=True,
        als_opts=dict(solver='torch', max_iterations=iterations, tol=0., monitor_convergence=True))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('psd', [False, True])
def test_prepared_matches_sequential_across_source_barriers_and_rank_growth(device, psd):
    state = initial(device)
    a, b = ((0, 1), (1, 1)), ((0, 0), (0, 1))
    schedule = [(((0, 0),), 'upper'), (a, 'lower'), (b, 'upper'),
                (a, 'lower'), (((1, 2),), 'lower'), (b, 'lower'), (a, 'upper')]
    gates = [gate(.13 + .02*i, len(s), device) for i, (s, _) in enumerate(schedule)]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=4,
                                     tol=0., psd_project=psd, optimize='greedy')
    actual, expected = state.copy(), state.copy()
    weights, expected_weights = {}, {}
    tracker, expected_tracker = LoopClusterUpdateTracker(), LoopClusterUpdateTracker()
    seen, results = [], []
    for start, stop in ((0, 3), (3, len(schedule))):
        actual, rows = plan.apply(actual, gates, gauges=weights, start=start, stop=stop,
            tracker=tracker, return_results=True, step_callback=lambda i, r: seen.append((i, r is not None)))
        results.extend(rows)
        expected, refs = reference(expected, gates[start:stop], schedule[start:stop],
                                   expected_weights, expected_tracker, psd=psd)
        for row, ref in zip(rows, refs):
            torch.testing.assert_close(row.problem.environment.data, ref.problem.environment.data,
                                       rtol=2e-8, atol=2e-10)
            assert row.solution.costs[-1]/row.problem.target_norm == pytest.approx(
                ref.solution.costs[-1]/ref.problem.target_norm, abs=2e-9)
            assert row.solution.iterations == ref.solution.iterations == 4
        torch.testing.assert_close(physical(actual, weights), physical(expected, expected_weights),
                                   rtol=2e-8, atol=2e-9)
    assert seen == [(i, len(s) == 2) for i, (s, _) in enumerate(schedule)]
    assert tracker.summary()['pair_updates'] == 5
    assert tracker.accumulated_local_fidelity == pytest.approx(
        expected_tracker.accumulated_local_fidelity, abs=2e-9)
    assert plan.info['geometry_plans'] == 2
    assert plan.info['program_misses'] > plan.info['geometry_plans']
    assert plan.info['program_hits'] > 0
    assert plan.info['one_site_updates'] == 2
    assert all(t.data.device.type == device for t in actual.tensors)
    assert all(math.isfinite(float(torch.linalg.vector_norm(w))) for w in weights.values())


@pytest.mark.parametrize('backend', [None, 'aot_eager'])
def test_reused_program_reads_new_tensor_and_gauge_values(backend):
    state = initial(random=True)
    where = ((0, 1), (1, 1))
    schedule, gates = [(where, 'lower')], [gate(.17, 2)]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=4,
                                     tol=0., torch_backend=backend, optimize='greedy')
    results = []
    for scale in (1., 1.23):
        work = state.copy()
        work[0, 0].modify(data=work[0, 0].data*scale)
        weights, ref_weights = {}, {}
        expected, refs = reference(work.copy(), gates, schedule, ref_weights)
        actual, rows = plan.apply(work, gates, gauges=weights, return_results=True)
        results.append(rows[0])
        torch.testing.assert_close(rows[0].problem.environment.data, refs[0].problem.environment.data,
                                   rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(physical(actual, weights), physical(expected, ref_weights),
                                   rtol=1e-8, atol=1e-8)
    assert not torch.allclose(results[0].problem.environment.data, results[1].problem.environment.data)
    assert plan.info['program_hits'] == 1 and plan.info['program_misses'] == 1


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('backend', [None, 'aot_eager'])
@pytest.mark.parametrize('stabilized', [False, True])
def test_prepared_preserves_gate_gradients(device, backend, stabilized):
    state = initial(device, random=True)
    where = ((0, 1), (1, 1))
    schedule = [(where, 'upper'), (where, 'lower')]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2,
                                     max_iterations=2, tol=0., optimize='greedy', torch_backend=backend)
    control = torch.tensor(.17, dtype=torch.float64, device=device, requires_grad=True)

    def loss(value, prepared=True):
        work, weights = state.copy(), {}
        matrices = [gate(value, 2, device), gate(.21, 2, device)]
        with TorchLinalgConfig(stabilized=stabilized).activated():
            if prepared:
                output = plan.apply(work, matrices, gauges=weights)
            else:
                output, _ = reference(work, matrices, schedule, weights, iterations=2)
            return physical(output, weights).abs().square().sum()

    actual = torch.autograd.grad(loss(control), control)[0]
    expected = torch.autograd.grad(loss(control, False), control)[0]
    step = 1e-5
    finite = (loss(control.detach()+step)-loss(control.detach()-step))/(2*step)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, finite, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_prepared_preserves_complex64_replay(device):
    state = initial(device)
    state.apply_to_arrays(lambda a: a.to(torch.complex64))
    schedule = [(((0, 0),), 'upper'), (((0, 1), (1, 1)), 'lower')]
    gates = [gate(.1, len(s), device).to(torch.complex64) for s, _ in schedule]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=4,
                                     tol=0., optimize='greedy')
    weights, expected_weights = {}, {}
    actual = plan.apply(state.copy(), gates, gauges=weights)
    expected, _ = reference(state.copy(), gates, schedule, expected_weights)
    torch.testing.assert_close(physical(actual, weights), physical(expected, expected_weights),
                               rtol=4e-4, atol=4e-5)
    assert all(t.data.dtype == torch.complex64 for t in actual.tensors)


def test_prepared_cache_is_bounded_and_rebuilds_evicted_programs():
    state = initial()
    schedule = [(((0, 1), (1, 1)), 'upper'), (((0, 0), (0, 1)), 'lower')]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_programs=1, optimize='greedy')
    gates = [gate(.1, 2), gate(.2, 2)]
    for _ in range(2):
        plan.apply(state.copy(), gates, gauges={})
    assert plan.info['contraction_programs'] == len(plan.plan_cache.plans) == 1
    assert plan.info['program_evictions'] == 3


def test_prepared_validates_before_evolving_and_rejects_topology_changes():
    state = initial()
    schedule = [(((0, 0),), 'upper'), (((0, 1), (1, 1)), 'lower')]
    plan = PreparedLoopClusterUpdate(state, schedule, optimize='greedy')
    tensors = [t.data for t in state.tensors]
    weights = {}
    with pytest.raises(ValueError, match='gate arrays'):
        plan.apply(state, [gate(.1, 1), np.eye(4)], gauges=weights)
    assert not weights
    assert all(t.data is old for t, old in zip(state.tensors, tensors))
    index = next(iter(state.inner_inds()))
    with pytest.raises(ValueError, match='SU gauges'):
        plan.apply(state, [gate(.1, 1), gate(.2, 2)],
                   gauges={index: np.ones(state.ind_size(index))})
    broken = state.copy()
    broken.reindex_({next(iter(broken.inner_inds())): 'renamed-bond'})
    with pytest.raises(ValueError, match='topology'):
        plan.apply(broken, [gate(.1, 1), gate(.2, 2)], gauges={})


def test_prepared_copy_replay_preserves_the_input_core():
    state = initial()
    schedule, gates = [(((0, 1), (1, 1)), 'upper')], [gate(.17, 2)]
    before = [t.data for t in state.tensors]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, optimize='greedy')
    weights = {}
    result = plan.apply(state, gates, gauges=weights, inplace=False)
    assert result is not state and weights
    assert all(t.data is old for t, old in zip(state.tensors, before))
    assert state.exponent == 0.
    torch.testing.assert_close(state.to_dense(), torch.eye(64, dtype=torch.complex128))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('execution', ['prepared', 'reference'])
def test_omitting_physical_snapshots_preserves_stream_values_and_gradients(
        device, execution, monkeypatch):
    state = initial(device, random=True)
    schedule = [(((0, 1), (1, 1)), 'lower'), (((0, 0), (0, 1)), 'upper')]
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=2,
                                     tol=0., optimize='greedy')
    original_insert = qtn.TensorNetwork.gauge_simple_insert
    insertions = []

    def count_insertions(*args, **kwargs):
        insertions.append(1)
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, 'gauge_simple_insert', count_insertions)
    outputs = []
    for materialize in (True, False):
        value = torch.tensor(.17, dtype=torch.float64, device=device, requires_grad=True)
        matrices = [gate(value, 2, device), gate(.21, 2, device)]
        work, gauges, rows = state.copy(), {}, []
        tracker = LoopClusterUpdateTracker(keep_records=False)
        start = len(insertions)
        for index in range(2):
            if execution == 'prepared':
                work, results = plan.apply(work, matrices, gauges=gauges, start=index,
                    stop=index+1, tracker=tracker, return_results=True,
                    materialize_physical=materialize)
            else:
                sites, side = schedule[index]
                work, results = gate_loop_cluster(work, [(matrices[index], sites, side)],
                    gauges=gauges, max_bond=2, cluster_mode='loopunion', base_radius=4,
                    fillin=True, max_loop_size=0, psd_project=False, optimize='greedy',
                    strip_exponent=True, tracker=tracker, return_results=True,
                    materialize_physical=materialize,
                    als_opts=dict(solver='torch', max_iterations=2, tol=0., monitor_convergence=True))
            rows.extend(results)
        assert len(insertions)-start == (4 if materialize else 2)
        assert all((row.physical_tn is not None) is materialize for row in rows)
        dense = physical(work, gauges)
        gradient = torch.autograd.grad(dense.abs().square().sum(), value)[0]
        outputs.append((dense, gradient, work.exponent, tracker.summary(), rows))
    expected, actual = outputs
    for index in (0, 1, 2):
        torch.testing.assert_close(actual[index], expected[index], rtol=2e-8, atol=2e-9)
    assert actual[3] == expected[3]
    for row, ref in zip(actual[4], expected[4]):
        assert row.solution.costs == pytest.approx(ref.solution.costs)
        assert row.solution.iterations == ref.solution.iterations == 2
        assert row.solution.target_norm == pytest.approx(row.problem.target_norm, rel=1e-12)


def test_snapshot_option_is_validated_before_mutation():
    state = initial()
    schedule = [(((0, 1), (1, 1)), 'lower')]
    plan = PreparedLoopClusterUpdate(state, schedule, optimize='greedy')
    tensors, gauges = [t.data for t in state.tensors], {}
    with pytest.raises(TypeError, match='materialize_physical'):
        plan.apply(state, [gate(.1, 2)], gauges=gauges, materialize_physical=1)
    with pytest.raises(TypeError, match='materialize_physical'):
        gate_loop_cluster(state, [(gate(.1, 2), schedule[0][0], 'lower')],
                          gauges=gauges, materialize_physical=1)
    assert not gauges and all(t.data is old for t, old in zip(state.tensors, tensors))


def test_prepared_preserves_boundary_validation_and_cost_preflight():
    state = initial(random=True)
    schedule, gates = [(((0, 0), (0, 1)), 'upper')], [gate(.1, 2)]
    plan = PreparedLoopClusterUpdate(state, schedule, optimize='greedy')
    weights = {}
    _, rows = plan.apply(state.copy(), gates, gauges=weights, return_results=True)
    # A fresh invocation must validate new boundary values even on a cache hit.
    bad = {index: torch.ones(state.ind_size(index), dtype=torch.float64) for index in state.inner_inds()}
    index = rows[0].problem.boundary_inds[0]
    bad[index] = -bad[index]
    with pytest.raises(ValueError):
        plan.apply(state.copy(), gates, gauges=bad)
    from pepsy.bp import CompressionBudgetError
    limited = PreparedLoopClusterUpdate(state, schedule, max_peak_memory_log2=0, optimize='greedy')
    before = [t.data for t in state.tensors]
    with pytest.raises(CompressionBudgetError):
        limited.apply(state, gates, gauges={})
    assert all(t.data is old for t, old in zip(state.tensors, before))

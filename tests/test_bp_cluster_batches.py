"""Independent cluster batches retain physical updates, gradients and per-pair policy."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TorchLinalgConfig
from pepsy.bp import LoopClusterUpdateTracker, PreparedLoopClusterUpdate

torch = pytest.importorskip('torch')


def setup(device='cpu', dtype='complex128'):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = qtn.PEPO.rand(2, 6, bond_dim=2, dtype='complex128', seed=177)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, device=device, dtype=getattr(torch, dtype)))
    return state, [(((0, 1), (0, 2)), 'upper'), (((0, 3), (0, 4)), 'lower')]


def matrices(value, state):
    data = state.tensors[0].data
    z = torch.tensor([1., -1., -1., 1.], device=data.device, dtype=data.real.dtype)
    return [torch.diag(torch.exp(1j*v*z)).to(data.dtype) for v in (value, .23)]


def weights(state):
    data = state.tensors[0].data
    return {ix: torch.tensor([.6, 1.3], dtype=data.real.dtype, device=data.device)
            for ix in state.inner_inds()}


def pair_amplitude(state, gauges, sites):
    local = state.select(tuple(state.site_tag(s) for s in sites), which='any', virtual=False)
    local.gauge_simple_insert(gauges)
    return local.contract(output_inds=tuple(sorted(local.outer_inds())), optimize='greedy').data


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('psd', [False, True])
def test_batch_matches_sequential_updates_and_retains_each_active_gauge(device, dtype, psd):
    state, schedule = setup(device, dtype)
    outputs = []
    for size in (1, 2, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=4,
                                         tol=0., optimize='greedy', batch_size=size, psd_project=psd)
        gauges, tracker, seen = weights(state), LoopClusterUpdateTracker(), []
        out, results = plan.apply(state.copy(), matrices(.17, state), gauges=gauges,
            tracker=tracker, return_results=True, step_callback=lambda i, r: seen.append(i))
        assert seen == [0, 1]
        assert plan.info['batched_pair_updates'] == (0 if size == 1 else 2)
        assert tracker.pair_updates == 2
        outputs.append((out, gauges, results, tracker))
    tolerance = dict(rtol=8e-4, atol=3e-4) if dtype == 'complex64' else dict(rtol=2e-8, atol=2e-8)
    for candidate in outputs[1:]:
        for sites, _ in schedule:
            torch.testing.assert_close(pair_amplitude(*outputs[0][:2], sites),
                                       pair_amplitude(*candidate[:2], sites), **tolerance)
        for actual, expected in zip(candidate[2], outputs[0][2]):
            torch.testing.assert_close(actual.solution.theta(), expected.solution.theta(), **tolerance)
            assert actual.solution.iterations == expected.solution.iterations == 4
            assert actual.solution.target_norm == pytest.approx(
                actual.problem.target_norm, rel=8e-5 if dtype == 'complex64' else 2e-12)
            assert actual.solution.local_fidelity == pytest.approx(expected.solution.local_fidelity,
                                                                   abs=tolerance['atol'])
        assert candidate[3].accumulated_local_fidelity == pytest.approx(
            outputs[0][3].accumulated_local_fidelity, abs=tolerance['atol'])


def test_batch_preserves_independent_stopping_and_source_barriers():
    state, schedule = setup()
    plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=10,
                                     tol=1e-8, optimize='greedy', batch_size=2)
    _, results = plan.apply(state.copy(), matrices(0., state), gauges={}, return_results=True)
    assert results[0].solution.iterations == 1
    assert results[1].solution.iterations > 1
    for i in range(2):
        reference = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=10,
                                              tol=1e-8, optimize='greedy')
        _, rows = reference.apply(state.copy(), matrices(0., state), gauges={},
                                  start=i, stop=i+1, return_results=True)
        assert rows[0].solution.iterations == results[i].solution.iterations
    split = PreparedLoopClusterUpdate(state, schedule, batch_size=2, optimize='greedy')
    work, gauges = state.copy(), {}
    for i in range(2):
        split.apply(work, matrices(.17, state), gauges=gauges, start=i, stop=i+1)
    assert split.info['cluster_batches'] == 0


@pytest.mark.parametrize('size', [2, 'auto'])
def test_overlapping_environment_patches_are_not_batched(size):
    state, schedule = setup()
    # Endpoints are disjoint, but their 2x2 environment patches overlap.
    schedule[1] = (((1, 1), (1, 2)), 'lower')
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, optimize='greedy')
    plan.apply(state, matrices(.17, state), gauges={})
    assert plan.info['cluster_batches'] == 0


def test_rank_growth_and_one_site_barriers_preserve_physical_operator():
    from pepsy.tensors import id_to_pepo

    state, schedule = setup()
    state = id_to_pepo(2, 6)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    schedule = schedule*2
    gates = matrices(.17, state)*2
    outputs = []
    for size in (1, 2):
        plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=3,
                                         tol=0., optimize='greedy', batch_size=size)
        gauges = {}
        out = plan.apply(state.copy(), gates, gauges=gauges)
        physical = out.copy()
        physical.gauge_simple_insert(gauges)
        physical.isel_({ix: 0 for ix in physical.outer_inds()})
        outputs.append(physical.contract(optimize='greedy'))
        if size == 2:
            assert plan.info['cluster_batches'] == 2
        assert all(out.ind_size(out.bond(*sites)) == 2 for sites, _ in schedule)
    torch.testing.assert_close(*outputs, rtol=1e-9, atol=1e-10)
    expected = torch.exp(torch.tensor(2j*(.17+.23), dtype=torch.complex128))
    torch.testing.assert_close(outputs[1], expected, rtol=1e-9, atol=1e-10)
    schedule = [schedule[0], (((0, 0),), 'upper'), schedule[1]]
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size=2, optimize='greedy')
    plan.apply(state, [gates[0], torch.eye(2, dtype=torch.complex128), gates[1]], gauges={})
    assert plan.info['cluster_batches'] == 0


@pytest.mark.parametrize('size', [2, 'auto'])
def test_incompatible_patch_shapes_use_the_sequential_path(size):
    state, schedule = setup()
    schedule[0] = (((0, 0), (0, 1)), 'upper')
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, optimize='greedy')
    out = plan.apply(state, matrices(.17, state), gauges={})
    assert out is state
    assert plan.info['pair_updates'] == 2
    assert plan.info['cluster_batches'] == 0


def test_batch_workspace_budget_is_checked_before_any_update():
    import math
    from pepsy.bp import CompressionBudgetError, prepare_reduced_bond_pair

    state, schedule = setup()
    baseline = PreparedLoopClusterUpdate(state, schedule, batch_size=2, optimize='greedy')
    pair = prepare_reduced_bond_pair(
        state, weights(state), where=schedule[0][0], run_bp=False)
    program, _, _ = baseline._program_for(pair)
    cap = program.cost['peak_memory_log2'] + math.log2(1.5)
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size=2, optimize='greedy',
                                     max_peak_memory_log2=cap)
    before = [t.data for t in state.tensors]
    with pytest.raises(CompressionBudgetError, match='independent cluster batch'):
        plan.apply(state, matrices(.17, state), gauges=weights(state))
    assert all(t.data is old for t, old in zip(state.tensors, before))


@pytest.mark.parametrize('stabilized', [False, True])
def test_batch_gate_and_gauge_gradients_match_sequential_and_finite_differences(stabilized):
    state, schedule = setup()
    probe = torch.tensor(.17, dtype=torch.float64, requires_grad=True)

    def loss(value, size):
        gauges = weights(state)
        index = state.bond((0, 0), (0, 1))
        gauges[index] = torch.stack((value+.5, value*0+1.3))
        plan = PreparedLoopClusterUpdate(state, schedule, max_bond=2, max_iterations=2,
                                         tol=0., optimize='greedy', batch_size=size)
        with TorchLinalgConfig(stabilized=stabilized).activated():
            out = plan.apply(state.copy(), matrices(value, state), gauges=gauges)
        # A local core norm omits the detached RMS scale in the network
        # exponent. Differentiate a physical matrix element of the full PEPO.
        physical = out.copy()
        physical.gauge_simple_insert(gauges)
        physical.isel_({ix: i % 2 for i, ix in enumerate(physical.outer_inds())})
        element = physical.contract(optimize='greedy')
        return element.abs().square()

    actual = torch.autograd.grad(loss(probe, 2), probe)[0]
    reference = torch.autograd.grad(loss(probe, 1), probe)[0]
    step = 1e-5
    finite = (loss(probe.detach()+step, 2)-loss(probe.detach()-step, 2))/(2*step)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(actual, finite, rtol=1e-4, atol=2e-5)


def test_batch_mixed_cholesky_and_spectral_solves_preserve_gradients():
    from pepsy.bp._reduced_torch import _guarded_solve
    from pepsy.bp._reduced_torch_batch import _solve

    value = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    diagonals = torch.stack([torch.stack((value, value*.5, value*0+x)) for x in (.2, -.1)])
    gram = torch.diag_embed(diagonals).to(torch.complex128)
    rhs = torch.ones((2, 3, 1), dtype=torch.complex128)
    actual, decisions = _solve(gram, rhs, 1e-5)
    expected = torch.stack([_guarded_solve(g, r, 1e-5, dict(cholesky=0, eigh=0))
                            for g, r in zip(gram, rhs)])
    assert decisions == [True, False]
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.abs().square().sum(), value, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.abs().square().sum(), value)[0]
    assert torch.isfinite(grad)
    torch.testing.assert_close(grad, reference)


def test_batch_rejects_unsupported_compile_and_size_options():
    state, schedule = setup()
    for invalid in (0, -1, True, 1.5, 'AUTO', '2', None):
        with pytest.raises(ValueError, match='batch_size'):
            PreparedLoopClusterUpdate(state, schedule, batch_size=invalid)
    with pytest.raises(ValueError, match='eager'):
        PreparedLoopClusterUpdate(state, schedule, batch_size=2, torch_backend='aot_eager')
    with pytest.raises(ValueError, match='gate arrays'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=2)
        plan.apply(state, [matrices(.17, state)[0], np.eye(4)], gauges={})


def auto_fixture(device='cpu'):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = qtn.PEPO.rand(2, 8, bond_dim=2, dtype='complex128', seed=992)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, device=device))
    # Action 1 depends on 0; action 2 commutes with both. Auto may prepare
    # 0 and 2 together, but must still publish callbacks and mutations 0,1,2.
    schedule = [(((0, 1), (0, 2)), 'upper'), (((0, 2), (0, 3)), 'lower'),
                (((0, 5), (0, 6)), 'upper')]
    gates = [matrices(value, state)[0] for value in (.17, .31, .23)]
    return state, schedule, gates


def physical_samples(state, gauges):
    physical = state.copy()
    physical.gauge_simple_insert(gauges)
    outputs = []
    for pattern in (0, 1, 2):
        selected = {ix: ((i+pattern)//(pattern+1)) % 2
                    for i, ix in enumerate(physical.outer_inds())}
        outputs.append(physical.isel(selected).contract(optimize='greedy'))
    return torch.stack(outputs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_auto_finds_ready_clusters_beyond_dependent_intervening_gates(device):
    state, schedule, gates = auto_fixture(device)
    outputs, records = [], []
    for size in (1, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, optimize='greedy',
                                         max_bond=2, max_iterations=4, tol=0.)
        gauges, seen = weights(state), []
        out, rows = plan.apply(state.copy(), gates, gauges=gauges, return_results=True,
                               step_callback=lambda i, r: seen.append(i))
        outputs.append(physical_samples(out, gauges))
        records.append(rows)
        assert seen == [0, 1, 2]
        if size == 'auto':
            assert plan.info['largest_batch'] == 2
            assert plan.info['lookahead_pair_updates'] == 1
            assert plan.info['batch_histogram'] == {2: 1}
            assert plan.info['auto_workspace_bytes'] > 0
    torch.testing.assert_close(*outputs, rtol=2e-8, atol=2e-8)
    for actual, expected in zip(records[1], records[0]):
        assert actual.solution.iterations == expected.solution.iterations
        torch.testing.assert_close(actual.solution.theta(), expected.solution.theta(), rtol=2e-8, atol=2e-8)


def test_auto_respects_transitive_dependencies_and_source_barriers():
    state, schedule, gates = auto_fixture()
    schedule[2] = (((0, 3), (0, 4)), 'upper')
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size='auto', optimize='greedy')
    plan.apply(state.copy(), gates, gauges={})
    assert plan.info['cluster_batches'] == 0
    state, schedule, gates = auto_fixture()
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size='auto', optimize='greedy')
    work, gauges = state.copy(), {}
    plan.apply(work, gates, gauges=gauges, stop=2)
    plan.apply(work, gates, gauges=gauges, start=2)
    assert plan.info['cluster_batches'] == 0


@pytest.mark.parametrize('site,expected', [((1, 3), 2), ((0, 5), 0)])
def test_auto_only_crosses_independent_one_site_gates(site, expected):
    state, schedule, gates = auto_fixture()
    schedule[1] = ((site,), 'lower')
    gates[1] = torch.tensor([[1., .2j], [.1, .9]], dtype=torch.complex128)
    outputs = []
    for size in (1, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, max_bond=2,
                                         max_iterations=3, tol=0., optimize='greedy')
        gauges = weights(state)
        out = plan.apply(state.copy(), gates, gauges=gauges)
        outputs.append(physical_samples(out, gauges))
        if size == 'auto':
            assert plan.info['batched_pair_updates'] == expected
    torch.testing.assert_close(*outputs, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize('last,expected', [(5, 2), (7, 0)])
def test_auto_shape_skips_still_constrain_later_dependencies(last, expected):
    state = qtn.PEPO.rand(2, 10, bond_dim=2, dtype='complex128', seed=991)
    state.apply_to_arrays(torch.as_tensor)
    schedule = [(((0, 1), (0, 2)), 'upper'), (((0, 8), (0, 9)), 'lower'),
                (((0, last), (0, last+1)), 'upper')]
    gates = [matrices(value, state)[0] for value in (.17, .31, .23)]
    outputs = []
    for size in (1, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, max_bond=2,
                                         max_iterations=3, tol=0., optimize='greedy')
        gauges = weights(state)
        out = plan.apply(state.copy(), gates, gauges=gauges)
        outputs.append(physical_samples(out, gauges))
        if size == 'auto':
            assert plan.info['batched_pair_updates'] == expected
    torch.testing.assert_close(*outputs, rtol=2e-8, atol=2e-8)


def test_auto_handles_streams_with_only_one_site_actions():
    state, _ = setup()
    schedule = [(((0, 0),), 'upper'), (((1, 5),), 'lower')]
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size='auto', optimize='greedy')
    before = state.copy()
    gates = [torch.eye(2, dtype=torch.complex128)]*2
    plan.apply(state, gates, gauges={})
    assert plan.info['auto_batch_limit'] == 1
    assert plan.info['cluster_batches'] == 0
    assert plan.info['one_site_updates'] == 2
    for actual, expected in zip(state.tensors, before.tensors):
        torch.testing.assert_close(actual.data, expected.data)


def test_auto_batches_other_ready_gates_while_lookahead_results_are_pending():
    state = qtn.PEPO.rand(3, 14, bond_dim=2, dtype='complex128', seed=882)
    state.apply_to_arrays(torch.as_tensor)
    schedule = [(((1, 1), (1, 2)), 'upper'),
                (((0, 2), (1, 2)), 'lower'),
                (((0, 5), (1, 5)), 'upper'),
                (((0, 8), (1, 8)), 'lower'),
                (((1, 11), (1, 12)), 'upper')]
    gates = [matrices(.1+.02*i, state)[0] for i in range(len(schedule))]
    outputs = []
    for size in (1, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, max_bond=2,
                                         max_iterations=2, tol=0., optimize='greedy')
        gauges, seen = weights(state), []
        out = plan.apply(state.copy(), gates, gauges=gauges,
                         step_callback=lambda i, r: seen.append(i))
        outputs.append(physical_samples(out, gauges))
        assert seen == list(range(5))
        if size == 'auto':
            assert plan.info['batched_pair_updates'] == 5
            assert plan.info['batch_histogram'] == {2: 1, 3: 1}
    torch.testing.assert_close(*outputs, rtol=2e-8, atol=2e-8)


def test_auto_scales_width_with_lattice_and_stream():
    from pepsy.tensors import id_to_pepo

    state = id_to_pepo(3, 14)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    schedule = [(((1, j), (1, j+1)), 'upper') for j in (1, 3, 5, 7, 9, 11)]
    gates = [matrices(.1*(i+1), state)[0] for i in range(6)]
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size='auto', max_bond=2,
                                     max_iterations=2, optimize='greedy')
    gauges = {}
    out = plan.apply(state, gates, gauges=gauges)
    assert plan.info['auto_batch_limit'] == plan.info['largest_batch'] == 6
    out.gauge_simple_insert(gauges)
    value = out.isel({ix: 0 for ix in out.outer_inds()}).contract(optimize='greedy')
    torch.testing.assert_close(value, torch.exp(torch.tensor(2.1j, dtype=torch.complex128)))


@pytest.mark.parametrize('restriction', ['memory', 'peak', 'flops'])
def test_auto_reduces_width_instead_of_exceeding_workspace_or_cost_limits(monkeypatch, restriction):
    import math
    from pepsy.bp import prepare_reduced_bond_pair

    state, schedule = setup()
    gates = matrices(.17, state)
    baseline = PreparedLoopClusterUpdate(state, schedule, optimize='greedy')
    pair = prepare_reduced_bond_pair(state, weights(state), where=schedule[0][0], run_bp=False)
    program, _, _ = baseline._program_for(pair)
    options = {}
    if restriction == 'peak':
        options['max_peak_memory_log2'] = program.cost['peak_memory_log2']+math.log2(1.5)
    if restriction == 'flops':
        options['max_flops_log10'] = program.cost['flops_log10']+math.log10(1.5)
    plan = PreparedLoopClusterUpdate(state, schedule, batch_size='auto', optimize='greedy', **options)
    if restriction == 'memory':
        monkeypatch.setattr(plan, '_available_batch_memory', lambda: 1)
    outputs = []
    for candidate in (baseline, plan):
        gauges = weights(state)
        out = candidate.apply(state.copy(), gates, gauges=gauges)
        outputs.append(physical_samples(out, gauges))
    assert plan.info['cluster_batches'] == 0
    torch.testing.assert_close(*outputs, rtol=1e-8, atol=1e-8)


def test_auto_memory_policy_accounts_for_full_lattice_snapshots(monkeypatch):
    small, schedule = setup()
    small_plan = PreparedLoopClusterUpdate(small, schedule, batch_size='auto', max_bond=2,
                                           max_iterations=1, optimize='greedy')
    small_plan.apply(small, matrices(.17, small), gauges={})
    assert small_plan.info['largest_batch'] == 2
    allowance = 2*small_plan.info['auto_estimated_member_bytes']+1
    large = qtn.PEPO.rand(2, 20, bond_dim=2, dtype='complex128', seed=992)
    large.apply_to_arrays(torch.as_tensor)
    large_plan = PreparedLoopClusterUpdate(large, schedule, batch_size='auto', max_bond=2,
                                           max_iterations=1, optimize='greedy')
    monkeypatch.setattr(large_plan, '_available_batch_memory', lambda: allowance)
    large_plan.apply(large, matrices(.17, large), gauges={})
    assert large_plan.info['auto_estimated_member_bytes'] > small_plan.info['auto_estimated_member_bytes']
    assert large_plan.info['cluster_batches'] == 0


@pytest.mark.parametrize('stabilized', [False, True])
def test_auto_lookahead_preserves_gate_and_gauge_gradients(stabilized):
    state, schedule, gates = auto_fixture()
    control = torch.tensor(.17, dtype=torch.float64, requires_grad=True)

    def loss(value, size):
        bank = [matrices(value, state)[0], *gates[1:]]
        gauges = weights(state)
        index = state.bond((0, 4), (0, 5))
        gauges[index] = torch.stack((value+.5, value*0+1.3))
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, max_bond=2,
                                         max_iterations=2, tol=0., optimize='greedy')
        with TorchLinalgConfig(stabilized=stabilized).activated():
            out = plan.apply(state.copy(), bank, gauges=gauges)
        return physical_samples(out, gauges).abs().square().sum()

    actual = torch.autograd.grad(loss(control, 'auto'), control)[0]
    expected = torch.autograd.grad(loss(control, 1), control)[0]
    step = 1e-5
    finite = (loss(control.detach()+step, 'auto')-loss(control.detach()-step, 'auto'))/(2*step)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(actual, finite, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_auto_mixed_nonunitary_stream_and_cache_evictions_match_sequential(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(1501)
    state = qtn.PEPO.rand(3, 10, bond_dim=2, dtype='complex128', seed=901)
    state.apply_to_arrays(lambda a: torch.as_tensor(a, device=device))
    state.exponent = .17
    choices = [(((1, 1), (1, 2)), 'upper'), (((0, 2), (1, 2)), 'lower'),
               (((1, 6), (1, 7)), 'lower'), (((1, 2), (1, 3)), 'upper'),
               (((0, 6), (1, 6)), 'upper'), (((0, 0),), 'lower')]
    schedule, gates = [], []
    for _ in range(4):
        for index in rng.permutation(len(choices)):
            sites, side = choices[index]
            if len(sites) == 2 and rng.random() < .5:
                sites = sites[::-1]
            schedule.append((sites, side))
            width = 2**len(sites)
            raw = rng.normal(size=(width, width))+1j*rng.normal(size=(width, width))
            # Nonsymmetric complex actions expose lower-leg transpose and
            # operand-order errors that diagonal ZZ gates cannot distinguish.
            gates.append(torch.as_tensor(np.eye(width)+(.08+.02j)*raw, device=device))
    outputs, records = [], []
    for size in (1, 'auto'):
        plan = PreparedLoopClusterUpdate(state, schedule, batch_size=size, max_bond=2,
                                         max_iterations=3, tol=0., optimize='greedy', max_programs=2)
        gauges, seen = weights(state), []
        out, rows = plan.apply(state.copy(), gates, gauges=gauges, return_results=True,
                               step_callback=lambda i, r: seen.append(i))
        outputs.append(physical_samples(out, gauges))
        records.append(rows)
        assert seen == list(range(len(schedule)))
        assert plan.info['program_evictions'] > 0
        if size == 'auto':
            assert plan.info['lookahead_pair_updates'] > 0
    torch.testing.assert_close(*outputs, rtol=2e-8, atol=2e-8)
    for actual, expected in zip(records[1], records[0]):
        assert actual.solution.iterations == expected.solution.iterations
        assert actual.solution.costs[-1]/actual.problem.target_norm == pytest.approx(
            expected.solution.costs[-1]/expected.problem.target_norm, abs=1e-10)

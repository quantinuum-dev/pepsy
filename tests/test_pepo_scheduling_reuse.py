"""Leg-aware schedules and incremental boundary/SU work on native operators."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.optimizers.peps._gate_order import order_gates
from test_pepo_full_update import action, dense, fidelity, fixture, unitary

pytestmark = [pytest.mark.optional, pytest.mark.peps]


@pytest.mark.parametrize('policy', ['row', 'column', 'smart'])
@pytest.mark.parametrize('which', [None, 'upper', 'lower'])
def test_explicit_operator_legs_reorder_and_match_independent_dense_actions(policy, which):
    import torch
    state = fixture(shape=(2, 3))
    zz = np.diag(np.exp(np.array([-.2j, .2j, .2j, -.2j])))
    entries = [(zz, ((1, 1), (1, 2)), 'upper'),
               (zz, ((0, 0), (1, 0)), 'lower'),
               (zz, ((0, 1), (0, 2)), None),
               (unitary(2, 71), (0, 0), 'lower'),
               (unitary(2, 72), (0, 0), 'lower'),
               (unitary(2, 73), (0, 0), 'upper')]
    expected = dense(state)
    for gate, sites, side in entries:
        support = (sites,) if gate.shape == (2, 2) else sites
        expected = action(state, expected, gate, support, side or which or 'upper')
    opt = PepsOptimizer(state, [(torch.tensor(g), s, side) for g, s, side in entries],
                        chi=64, mode='full-update', which=which, gate_order=policy,
                        contraction_opt='greedy', boundary_convergence=False)
    assert [i for i, _ in order_gates(opt, policy)] != list(range(len(entries)))
    actual = opt.run()
    np.testing.assert_allclose(dense(actual), expected, rtol=2e-10, atol=2e-10)
    if policy == 'smart':
        assert opt.last_gate_order['single_qubit_gates_fused'] >= 1
        for group in opt.last_gate_order['original_step_groups']:
            assert len({entries[i-1][2] or which or 'upper' for i in group}) == 1


@pytest.mark.parametrize('policy', ['row', 'column', 'smart'])
def test_noncommuting_same_leg_dependencies_survive_mixed_leg_scheduling(policy):
    import torch
    state = fixture()
    sites = ((0, 0), (0, 1))
    entries = [(unitary(4, 13), sites, 'lower'),
               (unitary(4, 14), ((1, 0), (0, 0)), 'upper'),
               (unitary(4, 15), sites, 'lower')]
    opt = PepsOptimizer(state, [(torch.tensor(g), s, side) for g, s, side in entries],
                        chi=32, mode='full-update', gate_order=policy,
                        contraction_opt='greedy')
    ordered = [i for i, _ in order_gates(opt, policy)]
    assert ordered.index(0) < ordered.index(2)
    expected = dense(state)
    for gate, support, side in entries:
        expected = action(state, expected, gate, support, side)
    np.testing.assert_allclose(dense(opt.run()), expected, rtol=2e-10, atol=2e-10)


def test_smart_fuses_same_leg_rotations_across_commuting_opposite_leg():
    import torch
    state = fixture()
    entries = [(unitary(2, 21), 'upper'), (unitary(2, 22), 'lower'), (unitary(2, 23), 'upper')]
    expected = dense(state)
    for gate, side in entries:
        expected = action(state, expected, gate, ((0, 0),), side)
    opt = PepsOptimizer(state, [(torch.tensor(g), (0, 0), side) for g, side in entries],
                        chi=2, mode='full-update', gate_order='smart', contraction_opt='greedy')
    np.testing.assert_allclose(dense(opt.run()), expected, rtol=2e-11, atol=2e-11)
    assert opt.last_gate_order['original_step_groups'] == [[2], [1, 3]]


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_persistent_su_keeps_exterior_arrays_and_matches_fresh_boundaries(monkeypatch, fit_mode):
    import torch
    from pepsy.optimizers.peps._su_warmstart import SimpleUpdateWarmStart
    state = fixture(shape=(3, 3))
    entries = [(torch.tensor(unitary(4, 51+i)), ((1, j), (1, j+1)), side)
               for i, (j, side) in enumerate([(0, 'upper'), (1, 'lower'), (0, 'lower')])]
    finish = SimpleUpdateWarmStart.finish
    gauge_all = qtn.TensorNetwork.gauge_all_simple_
    global_passes = []
    def track_global(network, *args, **kwargs):
        global_passes.append(1)
        return gauge_all(network, *args, **kwargs)
    monkeypatch.setattr(qtn.TensorNetwork, 'gauge_all_simple_', track_global)
    def checked_finish(cache, source, where):
        expected = dense(source)
        arrays = {s: source[s].data for s in source.gen_site_coos()}
        result = finish(cache, source, where)
        np.testing.assert_allclose(dense(result), expected, rtol=2e-9, atol=2e-10)
        for s in source.gen_site_coos():
            if s not in where:
                assert result[s].data is arrays[s]
        return result
    monkeypatch.setattr(SimpleUpdateWarmStart, 'finish', checked_finish)
    outputs = []
    for reuse in (True, False):
        opt = PepsOptimizer(state, entries, chi=2, mode='full-update', fit_mode=fit_mode,
            boundary_chi=64, contraction_opt='greedy', boundary_convergence=False,
            boundary_kwargs={'cutoff': 0., 'n_iter': 2},
            full_update_kwargs={'solver': 'qr', 'max_iterations': 2, 'skip_exact': False})
        if not reuse:
            def fresh(pair, guess, boundary, chi):
                return pair.environment(guess, boundary=None, chi=chi,
                    contraction_opt='greedy', boundary_kwargs=opt.boundary_kwargs)
            monkeypatch.setattr(opt, '_full_update_environment', fresh)
        opt.run(warmstart_method='simple-update', cutoff=0.)
        output = opt.run(warmstart_method='simple-update', cutoff=0.)
        assert opt._full_update_su.report() == dict(initializations=1, reuses=5, local_refreshes=6)
        if reuse:
            assert opt._full_update_boundary.mps_b.environment_cache.hits > 0
            assert opt._full_update_strip_cache.hits > 0
        outputs.append(dense(output))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-8, atol=2e-10)
    assert len(global_passes) == 2  # One initialization for each independent optimizer.


def test_direct_boundaries_absorb_layers_stepwise_and_reuse_unchanged_cuts(monkeypatch):
    import torch
    from pepsy.boundary.sweeps import CompBdy
    state = fixture(shape=(3, 3))
    gate = torch.tensor(unitary(4))
    rows = [0, 0, 1, 2, 2, 1]
    entries = [(gate, ((row, 0), (row, 1)), 'upper' if i % 2 else 'lower')
               for i, row in enumerate(rows)]
    compress = CompBdy._compress_direct_target
    layers = []
    def tracked(comp, target, **kwargs):
        present = tuple(tag for tag in ('BRA', 'KET') if tag in target.tags)
        assert len(present) == 1
        layers.append(present[0])
        return compress(comp, target, **kwargs)
    def unexpected(*args, **kwargs):
        pytest.fail('direct boundary mode invoked the variational FIT solver')
    monkeypatch.setattr(CompBdy, '_compress_direct_target', tracked)
    monkeypatch.setattr(CompBdy, '_run_fit_solver', unexpected)
    opt = PepsOptimizer(state, entries, chi=2, mode='full-update', fit_mode='direct',
        boundary_chi=32, contraction_opt='greedy', boundary_convergence=False,
        full_update_kwargs={'solver': 'qr', 'max_iterations': 2, 'skip_exact': False})
    snapshots, traces = {}, []
    def observe(record):
        boundary = opt._full_update_boundary
        steps = record['optimizer_result']['boundary_mps_steps']
        for step in steps:
            assert isinstance(boundary.mps_b[step['cut']], qtn.MatrixProductState)
            if step['action'] == 'reuse' and step['cut'] in snapshots:
                assert boundary.mps_b[step['cut']] is snapshots[step['cut']]
        snapshots.update({step['cut']: boundary.mps_b[step['cut']] for step in steps})
        traces.extend(steps)
    opt.run(warmstart_method='simple-update', step_callback=observe)
    assert len(traces) == 2*len(rows)
    assert any(step['action'] == 'reuse' for step in traces)
    assert len(layers) == 2*sum(step['action'] == 'build' for step in traces)
    assert layers == ['BRA', 'KET']*(len(layers)//2)


def test_external_mutation_and_set_state_reinitialize_persistent_su():
    import torch
    state = fixture()
    gate = torch.tensor(unitary(4))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)), 'lower')], chi=2,
        mode='full-update', fit_mode='direct', contraction_opt='greedy',
        full_update_kwargs={'solver': 'qr', 'max_iterations': 2})
    opt.run(warmstart_method='simple-update')
    old = opt._full_update_su
    opt.state[1, 1].data.mul_(1.2)
    opt.run(warmstart_method='simple-update')
    assert opt._full_update_su is not old
    opt.set_state(state)
    assert opt._full_update_su is None


@pytest.mark.parametrize('backend', ['torch-cuda', 'cupy'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_accelerator_persistent_su_reuses_native_weights(monkeypatch, backend, dtype, fit_mode):
    import torch
    state = fixture(shape=(2, 3), dtype=dtype)
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip('CuPy needs CUDA')
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(str(exc))
        convert = lambda value: cp.asarray(value.detach().numpy() if hasattr(value, 'detach') else value)
        as_numpy = cp.asnumpy
    else:
        if not torch.cuda.is_available():
            pytest.skip('Torch CUDA unavailable')
        convert = lambda value: torch.as_tensor(value, dtype=getattr(torch, dtype), device='cuda')
        as_numpy = lambda value: value.detach().cpu().numpy()
    state.apply_to_arrays(convert)
    gate = convert(unitary(4).astype(dtype))
    where = ((0, 0), (0, 1))
    opt = PepsOptimizer(state, [(gate, where, 'lower'), (gate, where, 'upper')],
        chi=2, mode='full-update', fit_mode=fit_mode, boundary_chi=64,
        contraction_opt='greedy', boundary_kwargs={'n_iter': 2, 'cutoff': 0.},
        full_update_kwargs={'solver': 'qr', 'max_iterations': 3, 'skip_exact': False})
    prepare = opt._full_update_pair
    targets = []
    def tracked(payload, sites, which=None, **kwargs):
        targets.append(action(opt.state, as_numpy(opt.state.to_dense()),
                              as_numpy(payload), sites, which))
        return prepare(payload, sites, which, **kwargs)
    monkeypatch.setattr(opt, '_full_update_pair', tracked)
    def observe(record):
        assert record['local_fidelity'] == pytest.approx(
            fidelity(as_numpy(opt.state.to_dense()), targets[-1]),
            abs=5e-5 if dtype == 'complex64' else 2e-9)
    opt.run(warmstart_method='simple-update', step_callback=observe)
    first = as_numpy(opt.state.to_dense())
    opt.run(warmstart_method='simple-update', step_callback=observe)
    assert opt._full_update_su.report() == dict(initializations=1, reuses=3, local_refreshes=4)
    sample = next(iter(state)).data
    assert all(t.data.dtype == sample.dtype and t.data.device == sample.device for t in opt.state)
    assert all(value.device == sample.device for value in opt._full_update_su.gauges.values())
    assert np.isfinite(first).all() and np.isfinite(as_numpy(opt.state.to_dense())).all()

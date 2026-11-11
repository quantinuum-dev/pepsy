"""One ALS compression of an exact upper/lower action on a PEPO bond."""

import numpy as np
import pytest

from pepsy.optimizers import PepsOptimizer
from test_pepo_full_update import action, dense, fidelity, fixture, unitary

pytestmark = [pytest.mark.optional, pytest.mark.peps]


@pytest.mark.parametrize('solver', ['qr', 'quimb'])
@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('lower_first', [False, True])
def test_joint_actions_share_one_als_fit_and_match_dense_fidelity(solver, fit_mode, reverse, lower_first):
    import torch
    state = fixture(shape=(2, 3))
    state[1, 2].modify(data=(2+.3j)*state[1, 2].data)
    state.exponent = .4
    where = ((0, 0), (0, 1))
    other_where = where[::-1] if reverse else where
    entries = [(unitary(4, 11), where, 'upper'), (unitary(4, 12), other_where, 'lower')]
    if lower_first:
        entries.reverse()
    target = dense(state)
    for gate, sites, side in entries:
        target = action(state, target, gate, sites, side)
    opt = PepsOptimizer(state, [(torch.tensor(g), sites, side) for g, sites, side in entries],
        chi=2, mode='full-update', fit_mode=fit_mode, boundary_chi=64,
        contraction_opt='greedy', boundary_convergence=False,
        boundary_kwargs={'cutoff': 0., 'n_iter': 2},
        full_update_kwargs={'solver': solver, 'max_iterations': 6, 'skip_exact': False})
    output = opt.run(fuse_pepo_legs=True, measure_infidelity=True, measure_final_infidelity=True)
    records = opt.get_step_records()
    assert len(records) == 1
    record = records[0]
    assert record['which'] == 'both' and record['joint_pepo_legs']
    assert record['start_step'] == 1 and record['step'] == 2
    assert record['original_steps'] == [1, 2] and record['operator_gate_count'] == 2
    expected_fidelity = fidelity(dense(output), target)
    assert record['local_fidelity'] == pytest.approx(expected_fidelity, abs=2e-9)
    assert record['post_infidelity'] == pytest.approx(1-expected_fidelity, abs=2e-9)
    assert not opt.get_normalizations()
    assert np.linalg.norm(dense(output)) > 2.


@pytest.mark.parametrize('warmstart', ['compress', 'simple-update'])
def test_joint_sandwich_keeps_identity_without_intermediate_truncation(warmstart):
    import torch
    from pepsy.tensors import id_to_pepo
    state = id_to_pepo((2, 2), dtype='complex128')
    state.apply_to_arrays(torch.tensor)
    state[0, 0].modify(data=(2+.3j)*state[0, 0].data)
    expected = dense(state)
    gate = torch.tensor(unitary(4, 91))
    entries = [(gate, ((0, 0), (0, 1)), 'upper'),
               (gate.conj(), ((0, 0), (0, 1)), 'lower')]
    outputs = []
    for fused in (True, False):
        opt = PepsOptimizer(state, entries, chi=1, mode='full-update', fit_mode='direct',
                            contraction_opt='greedy', boundary_convergence=False,
                            full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False})
        outputs.append(dense(opt.run(fuse_pepo_legs=fused, warmstart_method=warmstart)))
        assert len(opt.get_step_records()) == (1 if fused else 2)
    np.testing.assert_allclose(outputs[0], expected, rtol=2e-10, atol=2e-10)
    # Normalized fidelity can hide amplitude loss in sequential truncations.
    assert np.linalg.norm(outputs[1]-expected) > .05*np.linalg.norm(expected)


@pytest.mark.parametrize('same_side', [False, True])
def test_joint_update_does_not_cross_a_different_bond_or_same_leg(same_side):
    import torch
    state = fixture()
    gate = torch.tensor(unitary(4))
    entries = [(gate, ((0, 0), (0, 1)), 'upper'),
               (gate, ((0, 0), (0, 1)) if same_side else ((0, 1), (1, 1)),
                'upper' if same_side else 'lower')]
    opt = PepsOptimizer(state, entries, chi=2, mode='full-update', fit_mode='direct',
                        contraction_opt='greedy',
                        full_update_kwargs={'solver': 'qr', 'max_iterations': 2})
    opt.run(fuse_pepo_legs=True)
    assert len(opt.get_step_records()) == 2
    assert all(not record['joint_pepo_legs'] for record in opt.get_step_records())


@pytest.mark.parametrize('backend', ['torch-cuda', 'cupy'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_joint_quimb_als_preserves_accelerator_arrays(backend, dtype, fit_mode):
    import torch
    state = fixture(shape=(2, 3), dtype=dtype)
    upper, lower = unitary(4, 83).astype(dtype), unitary(4, 84).astype(dtype)
    where = ((0, 0), (0, 1))
    expected = action(state, action(state, dense(state), upper, where, 'upper'), lower, where, 'lower')
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
    opt = PepsOptimizer(state, [(convert(upper), where, 'upper'), (convert(lower), where, 'lower')],
        chi=2, mode='full-update', fit_mode=fit_mode, boundary_chi=64,
        contraction_opt='greedy', boundary_kwargs={'cutoff': 0., 'n_iter': 2},
        full_update_kwargs={'solver': 'quimb', 'max_iterations': 4, 'skip_exact': False})
    output = opt.run(fuse_pepo_legs=True)
    sample = next(iter(state)).data
    assert all(t.data.dtype == sample.dtype and t.data.device == sample.device for t in output)
    assert len(opt.get_step_records()) == 1
    assert opt.get_step_records()[0]['local_fidelity'] == pytest.approx(
        fidelity(as_numpy(output.to_dense()), expected), abs=5e-5 if dtype == 'complex64' else 2e-9)


@pytest.mark.parametrize('solver,tensor_mode', [('qr', 'reduced'), ('quimb', 'reduced'),
                                            ('lbfgs', 'reduced'), ('lbfgs', 'full')])
@pytest.mark.parametrize('normalize_target', [False, True])
def test_joint_su_initialization_preserves_exact_scaled_target(solver, tensor_mode, normalize_target):
    import torch
    state = fixture(shape=(2, 3))
    state.exponent = .3
    state[0, 0].modify(data=(2+.3j)*state[0, 0].data)
    upper, lower = (1.2+.1j)*unitary(4, 17), (.7-.3j)*unitary(4, 18)
    where = ((0, 0), (0, 1))
    expected = action(state, action(state, dense(state), upper, where, 'upper'), lower, where, 'lower')
    if normalize_target:
        expected /= np.linalg.norm(expected)
    opt = PepsOptimizer(state, [(torch.tensor(upper), where, 'upper'), (torch.tensor(lower), where, 'lower')],
        chi=2, mode='full-update', fit_mode='direct', boundary_chi=64,
        contraction_opt='greedy', boundary_convergence=False,
        full_update_kwargs={'solver': solver, 'tensor_mode': tensor_mode,
                            'skip_exact': False, 'max_iterations': 6})
    output = opt.run(fuse_pepo_legs=True, warmstart_method='simple-update', non_unitary=True,
                     normalize_target=normalize_target, measure_infidelity=True, measure_final_infidelity=True)
    record = opt.get_step_records()[0]
    assert record['operator_gate_count'] == 2 and record['which'] == 'both'
    assert record['local_fidelity'] == pytest.approx(fidelity(dense(output), expected), abs=3e-9)
    assert record['post_infidelity'] == pytest.approx(1-fidelity(dense(output), expected), abs=3e-9)
    assert opt._full_update_su.report() == dict(initializations=1, reuses=0, local_refreshes=1)


@pytest.mark.parametrize('scope', ['strip', 'layer'])
@pytest.mark.parametrize('warmstart', ['compress', 'simple-update'])
def test_joint_refinement_targets_include_both_gates(scope, warmstart):
    import torch
    state = fixture()
    entries = [(torch.tensor(unitary(4, 32)), ((0, 0), (0, 1)), 'lower'),
               (torch.tensor(unitary(4, 33)), ((0, 1), (0, 0)), 'upper')]
    expected = dense(state)
    for gate, sites, side in entries:
        expected = action(state, expected, gate.numpy(), sites, side)
    opt = PepsOptimizer(state, entries, chi=2, mode='full-update', fit_mode='direct',
        boundary_chi=64, contraction_opt='greedy', boundary_convergence=False,
        full_update_kwargs={'refine_scope': scope, 'refine_sweeps': 1, 'skip_exact': False,
                            'solver': 'qr', 'max_iterations': 3})
    output = opt.run(fuse_pepo_legs=True, warmstart_method=warmstart,
                     measure_infidelity=True, measure_final_infidelity=True)
    record = opt.get_step_records()[0]
    assert record[f'{scope}_refinement']['start_step'] == 1
    assert record[f'{scope}_refinement']['end_step'] == 2
    assert record[f'{scope}_refinement']['original_steps'] == [1, 2]
    assert record['post_infidelity'] == pytest.approx(1-fidelity(dense(output), expected), abs=3e-9)


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_joint_updates_reuse_su_weights_and_boundaries_across_runs(fit_mode):
    import torch
    state = fixture(shape=(2, 3))
    where = ((0, 0), (0, 1))
    entries = [(torch.tensor(unitary(4, 91+i)), where, side)
               for i, side in enumerate(['upper', 'lower', 'upper', 'lower'])]
    opt = PepsOptimizer(state, entries, chi=2, mode='full-update', fit_mode=fit_mode,
        boundary_chi=64, contraction_opt='greedy', boundary_kwargs={'n_iter': 2, 'cutoff': 0.},
        full_update_kwargs={'solver': 'qr', 'skip_exact': False, 'max_iterations': 3})
    opt.run(fuse_pepo_legs=True, warmstart_method='simple-update')
    opt.run(fuse_pepo_legs=True, warmstart_method='simple-update')
    assert opt._full_update_su.report() == dict(initializations=1, reuses=3, local_refreshes=4)
    records = opt.get_step_records()
    assert [r['original_steps'] for r in records] == [[1, 2], [3, 4]]
    assert all(r['operator_gate_count'] == 2 for r in records)
    assert opt._full_update_boundary.mps_b.environment_cache.hits > 0


def test_smart_joint_updates_bring_ready_partners_together_without_crossing_dependencies():
    import torch
    state = fixture()
    a, b = ((0, 1), (1, 1)), ((0, 0), (0, 1))
    entries = [(torch.tensor(unitary(4, 15+i)), sites, side) for i, (sites, side) in
               enumerate([(a, 'upper'), (b, 'upper'), (a, 'lower'), (b, 'lower')])]
    expected = dense(state)
    for gate, sites, side in entries:
        expected = action(state, expected, gate.numpy(), sites, side)
    opt = PepsOptimizer(state, entries, chi=32, mode='full-update', gate_order='smart',
                        contraction_opt='greedy')
    output = opt.run(fuse_pepo_legs=True)
    assert [r['original_steps'] for r in opt.get_step_records()] == [[1, 3], [2, 4]]
    np.testing.assert_allclose(dense(output), expected, rtol=2e-10, atol=2e-10)

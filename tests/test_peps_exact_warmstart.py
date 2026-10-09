"""Untruncated local SVD updates bypass full-update contractions and fitting."""

import pytest

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import ReducedPair
from test_peps_full_update import fixture


@pytest.mark.parametrize('tensor_mode', ['reduced', 'full'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_product_state_exact_gates_skip_fu_and_record_unit_fidelity(monkeypatch, tensor_mode, dtype):
    torch = pytest.importorskip('torch')
    state, gate, policy = fixture(dtype=dtype, shape=(2, 3), bond=1)
    gates = [(gate, ((0, 0), (0, 1))), (gate, ((0, 1), (0, 2))),
             (gate.conj(), ((0, 0), (0, 1)))]
    reference = state.copy()
    for g, where in gates:
        reference.gate_(g, where, contract=False)
    opt = PepsOptimizer(state, gates, chi=4, mode='full-update', contraction_opt=policy,
                        normalize_initial=False, full_update_kwargs={'tensor_mode': tensor_mode})

    def forbidden(*args, **kwargs):
        pytest.fail('an untruncated warm start requested a full-update contraction or solve')

    for name in ('_full_update_environment', '_full_update_optimize',
                 '_calibrate_sweep_boundaries', 'estimate_infidelity'):
        monkeypatch.setattr(opt, name, forbidden)
    out = opt.run(normalize_final=False)
    tol = 2e-5 if dtype == 'complex64' else 2e-11
    torch.testing.assert_close(out.to_dense(), reference.to_dense(), rtol=tol, atol=tol)
    assert out.max_bond() <= 4
    for record in opt.get_step_records():
        assert record['reason'] == 'exact_svd'
        assert not record['optimizer_attempted']
        assert not record['optimized']
        assert record['local_fidelity'] == 1.
        assert record['accumulated_local_infidelity'] == 0.
        assert record['norm_environment_chi'] is None
        assert record['optimizer_result']['iterations'] == 0


def test_svd_support_can_equal_cap_and_does_not_pad_zero_singular_values():
    torch = pytest.importorskip('torch')
    state, gate, _ = fixture(bond=1)
    pair = ReducedPair(state, gate, ((0, 0), (0, 1)), chi=2)
    guess, _ = pair.initial_states()
    assert pair.warmstart_exact and pair.target_rank == 2
    assert guess.bond_size((0, 0), (0, 1)) == 2
    identity = ReducedPair(state, torch.eye(4, dtype=gate.dtype), ((0, 0), (0, 1)), chi=4)
    guess, _ = identity.initial_states()
    assert identity.warmstart_exact and identity.target_rank == 1
    assert guess.bond_size((0, 0), (0, 1)) == 1


def test_small_actual_truncation_still_runs_full_update(monkeypatch):
    torch = pytest.importorskip('torch')
    state, _, policy = fixture(bond=1)
    gate = torch.diag(torch.exp(torch.tensor([-1e-4j, 1e-4j, 1e-4j, -1e-4j],
                                            dtype=torch.complex128)))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1, mode='full-update',
                        contraction_opt=policy, full_update_kwargs={'rtol': 1e-3})
    calls = []
    solve = opt._full_update_optimize
    def tracked(*args, **kwargs):
        calls.append(True)
        return solve(*args, **kwargs)
    monkeypatch.setattr(opt, '_full_update_optimize', tracked)
    opt.run(measure_infidelity=False, measure_final_infidelity=False, accept_if_improved=False)
    assert calls == [True]
    assert opt.get_step_records()[0]['optimizer_attempted']

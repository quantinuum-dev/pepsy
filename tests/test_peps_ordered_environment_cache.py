"""Gate compilation and strip traversal share validated FU environments."""

import pytest

from pepsy.optimizers.peps import PepsOptimizer
from test_peps_full_update import fixture


@pytest.mark.parametrize('policy', ['input', 'row', 'column', 'smart'])
@pytest.mark.parametrize('tensor_mode', ['reduced', 'full'])
def test_ordered_two_site_caches_match_fresh_environments(monkeypatch, policy, tensor_mode):
    import torch

    state, gate, _ = fixture(shape=(4, 4))
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=gate.dtype)
    y = torch.tensor([[0., -1j], [1j, 0.]], dtype=gate.dtype)
    # Alternate orientations in the input. Reordering can finish each strip;
    # smart compilation must also handle fused, noncommuting local rotations.
    gates = [(x, (1, 1)), (y, (1, 1))]
    for i in range(3):
        gates.extend([(gate, ((1, i), (1, i + 1))),
                      (gate, ((i, 1), (i + 1, 1)))])
    gates.append((gate.conj(), ((1, 0), (1, 1))))
    outputs, orders = [], []
    for reuse in (True, False):
        # An ample cap makes fresh and reused boundary fits exact here. Keep
        # the normal DMRG fitter; the default-cap policy is tested separately.
        opt = PepsOptimizer(state, gates, chi=2, mode='full-update', gate_order=policy,
                            boundary_chi=32, contraction_opt='greedy', normalize_initial=False,
                            full_update_kwargs={'tensor_mode': tensor_mode, 'max_iterations': 2})
        original_queue = opt.gates
        environment = opt._full_update_environment
        handles, scopes = [], []
        hits = {'row': 0, 'column': 0}

        def tracked(pair, guess, boundary, chi):
            if not reuse:
                return pair.environment(guess, boundary=None, chi=chi, contraction_opt='greedy',
                                        boundary_kwargs=opt.boundary_kwargs, strip_cache=None)
            cache = getattr(opt, '_full_update_strip_cache', None)
            before = 0 if cache is None else cache.hits
            result = environment(pair, guess, boundary, chi)
            cache = opt._full_update_strip_cache
            a, b = pair.where
            direction = 'column' if a[1] == b[1] else 'row'
            expected = ('X', a[1]) if direction == 'column' else ('Y', a[0])
            assert cache.scope[:2] == expected
            handles.append(result[1])
            scopes.append(cache.scope)
            hits[direction] += cache.hits - before
            return result

        monkeypatch.setattr(opt, '_full_update_environment', tracked)
        out = opt.run(normalize_final=False, measure_infidelity=False,
                      measure_final_infidelity=False, accept_if_improved=False)
        assert opt.gates is original_queue
        assert opt.boundary_kwargs['fit_mode'] == 'dmrg'
        records = opt.get_step_records()
        assert len(records) == 7 and all(r['optimizer_attempted'] for r in records)
        orders.append([r['original_steps'] for r in records])
        assert opt.last_gate_order['single_qubit_gates_fused'] == (1 if policy == 'smart' else 0)
        if reuse:
            assert len({id(handle) for handle in handles}) == 1
            assert handles[-1].mps_b.environment_cache.hits > 0
            assert {scope[0] for scope in scopes} == {'X', 'Y'}
            if policy == 'input':
                assert hits == {'row': 0, 'column': 0}  # every pair switches strips
            else:
                assert hits['row'] > 0 and hits['column'] > 0
        outputs.append(out.to_dense(optimize='greedy').ravel())
    assert orders[0] == orders[1]
    a, b = outputs
    fidelity = abs(torch.vdot(a, b))**2 / (torch.vdot(a, a).real * torch.vdot(b, b).real)
    assert float(fidelity) == pytest.approx(1., abs=1e-9)

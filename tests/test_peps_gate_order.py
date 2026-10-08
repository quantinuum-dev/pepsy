"""Traversal may reorder commuting layers but never change circuit ordering barriers."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._gate_order import order_gates


@pytest.mark.parametrize('policy', ['row', 'column'])
def test_commuting_traversal_matches_exact_circuit_and_preserves_barriers(policy):
    state = qtn.PEPS.rand(3, 3, bond_dim=1, dtype='complex128', seed=7)
    zz = np.diag(np.exp(np.array([-.2j, .2j, .2j, -.2j])))
    x = np.array([[0., 1.], [1., 0.]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    layer = [(zz, ((2, 1), (2, 2))), (zz, ((0, 0), (1, 0))),
             (zz, ((0, 1), (0, 2)))]
    gates = [*layer, (x, (0, 0)), *layer, (cnot, ((0, 0), (1, 0))), *layer]
    opt = PepsOptimizer(state, gates, chi=4, normalize_initial=False,
                        boundary_convergence=False, contraction_opt='greedy')
    ordered = order_gates(opt, policy)
    assert [i for i, _ in ordered] != list(range(len(gates)))
    assert ordered[3][0] == 3 and ordered[7][0] == 7
    def exact(entries):
        out = state.copy()
        for g, where, _ in entries:
            out.gate_(g, where, contract='split', cutoff=0.)
        return out.to_dense()
    np.testing.assert_allclose(exact(opt.gates), exact([entry for _, entry in ordered]), atol=1e-10)


def test_failed_run_restores_original_queue():
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype='complex128', seed=8)
    gate = np.diag([1., 1j, 1j, 1.])
    opt = PepsOptimizer(state, [(gate, ((0, 1), (1, 1))), (gate, ((0, 0), (1, 0)))],
                        chi=2, gate_order='column', update_style='two-site',
                        normalize_initial=False, contraction_opt='greedy')
    original = opt.gates
    with pytest.raises(ValueError, match='gate-by-gate'):
        opt.run(k_2q_batch=3)
    assert opt.gates is original
    assert opt.last_gate_order['original_steps'] == [2, 1]


@pytest.mark.parametrize('style,axes', [('row', ('x',)), ('column', ('y',))])
def test_local_scope_routes_to_the_requested_sweep_axis(monkeypatch, style, axes):
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype='complex128', seed=13)
    gate = np.diag(np.exp(np.array([-.3j, .3j, .3j, -.3j])))
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1,
                        fit_mode='direct', contraction_opt='greedy',
                        boundary_convergence=False, normalize_initial=False)
    received = []
    def fit(state, target, **kwargs):
        received.append(kwargs['sweep_optimize_kwargs']['axes'])
        return state, .01, {'success': True}
    monkeypatch.setattr(opt, '_optimize_state', fit)
    opt.run(update_style=style, k_2q_batch=1, measure_infidelity=False,
            measure_final_infidelity=False, normalize_final=False)
    assert received == [axes]
    assert opt.update_style == 'row-column'
    assert opt.get_step_records()[0]['update_style'] == style

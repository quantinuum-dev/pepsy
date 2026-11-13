"""Traversal may reorder commuting layers but never change circuit ordering barriers."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._gate_order import (
    _cuda_diagonal_gates, _cuda_pauli_supports, _pauli_patterns, _pauli_support, order_gates,
)


@pytest.mark.parametrize('arity', [1, 2])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_cuda_batch_gate_classification_matches_scalar_reference(arity, dtype):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    _, first, second, _, _ = _pauli_patterns(arity)
    convert = lambda x: torch.as_tensor(x, dtype=getattr(torch, dtype), device='cuda')
    gates = [convert(a) * (.6+.2j) + convert(b) * (.3-.1j)
             for a, b in zip(first, second)]
    dim = 2**arity
    gates.extend((torch.zeros((dim, dim), dtype=getattr(torch, dtype), device='cuda'),
                  torch.diag(convert(range(1, dim+1))),
                  convert(np.arange(dim**2).reshape(dim, dim) + .13j)))
    if arity == 2:
        gates[-1] = gates[-1].reshape(2, 2, 2, 2)
    expected = [_pauli_support(gate, arity, {}) for gate in gates]
    assert _cuda_pauli_supports(gates, arity, {}) == expected


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_cuda_batch_diagonal_check_matches_scalar_reference(dtype):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    diagonal = torch.diag(torch.tensor([1, 1j, 2, -1], dtype=getattr(torch, dtype), device='cuda'))
    arbitrary = torch.ones((4, 4), dtype=getattr(torch, dtype), device='cuda')
    nonfinite = diagonal.clone()
    nonfinite[0, 0] = float('inf')
    gates = [diagonal, arbitrary.reshape(2, 2, 2, 2), nonfinite]
    expected = [PepsOptimizer._gate_bond_factor(gate) == 2 for gate in gates]
    assert _cuda_diagonal_gates(gates) == expected


def test_cuda_smart_gate_order_matches_scalar_classification(monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from pepsy.optimizers.peps import _gate_order

    state = qtn.PEPO.rand(3, 3, bond_dim=1, dtype='complex128', seed=13)
    state.apply_to_arrays(lambda x: torch.tensor(x, device='cuda'))
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device='cuda')
    rx = torch.eye(2, dtype=torch.complex128, device='cuda') * .8 + .6j*x
    zz = torch.diag(torch.tensor([1, 1j, 1j, 1], dtype=torch.complex128, device='cuda'))
    entries = [(rx, (0, 0), 'upper'), (zz, ((0, 0), (1, 0)), 'upper'),
               (zz.clone(), ((0, 1), (1, 1)), 'lower'), (rx.clone(), (1, 0), 'lower'),
               (zz.clone(), ((1, 0), (2, 0)), 'upper'),
               (zz.clone(), ((1, 0), (1, 1)), 'lower')]
    opt = PepsOptimizer(state, entries, chi=2, normalize_initial=False)
    actual = [index for index, _ in order_gates(opt, 'smart', pair_opposite_legs=True)]
    monkeypatch.setattr(_gate_order, '_cuda_pauli_supports',
                        lambda gates, n, constants: [_pauli_support(g, n, constants) for g in gates])
    reference = [index for index, _ in order_gates(opt, 'smart', pair_opposite_legs=True)]
    assert actual == reference


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
    assert ordered[3][0] == 3  # Single-site gates remain barriers.
    positions = {original: position for position, (original, _) in enumerate(ordered)}
    # A CNOT can now cross disjoint gates, but never overlapping ZZ gates.
    assert positions[5] < positions[7] < positions[9]
    def exact(entries):
        out = state.copy()
        for g, where, _ in entries:
            out.gate_(g, where, contract='split', cutoff=0.)
        return out.to_dense()
    np.testing.assert_allclose(exact(opt.gates), exact([entry for _, entry in ordered]), atol=1e-10)


@pytest.mark.parametrize('policy', ['row', 'column'])
def test_disjoint_nondiagonal_layer_is_sorted_without_changing_circuit(policy):
    state = qtn.PEPS.rand(3, 3, bond_dim=1, dtype='complex128', seed=15)
    rng = np.random.default_rng(91)
    gates = [(np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0], where)
             for where in [((2, 1), (2, 2)), ((0, 2), (1, 2)), ((0, 0), (1, 0))]]
    opt = PepsOptimizer(state, gates, chi=2, normalize_initial=False, contraction_opt='greedy')
    ordered = order_gates(opt, policy)
    assert [i for i, _ in ordered] != [0, 1, 2]
    outputs = []
    for entries in (opt.gates, [entry for _, entry in ordered]):
        out = state.copy()
        for gate, where, _ in entries:
            out.gate_(gate, where, contract='split', cutoff=0.)
        outputs.append(out.to_dense())
    np.testing.assert_allclose(*outputs, atol=1e-12)


@pytest.mark.parametrize('policy', ['row', 'column'])
def test_diagonal_traversal_finishes_strips_before_turning(policy):
    state = qtn.PEPS.ones(4, 4, bond_dim=1)
    gate = np.diag([1., 1j, 1j, 1.])
    bonds = [((i, j), (i + 1, j)) for i in range(3) for j in range(4)]
    bonds += [((i, j), (i, j + 1)) for j in range(3) for i in range(4)]
    opt = PepsOptimizer(state, [(gate, b) for b in bonds], chi=2, normalize_initial=False)
    strips = []
    for _, (_, (a, b), _) in order_gates(opt, policy):
        strips.append(('column', a[1]) if a[1] == b[1] else ('row', a[0]))
    assert strips[0][0] == policy
    assert sum(a != b for a, b in zip(strips, strips[1:])) == 7


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

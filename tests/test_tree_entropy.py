"""Physical Schmidt-spectrum regressions for tree entropy readout."""

import numpy as np
import pytest
import autoray as ar

import pepsy
from pepsy.sampling import TreeSampler


def _dense_state(state):
    tensor = state.contract(all).transpose(
        *(state.site_ind(q) for q in range(state.plan.n))
    )
    data = tensor.data
    if hasattr(data, 'to_dense'):
        data = data.to_dense()
    return ar.to_numpy(data).reshape(-1)


def _dense_edge_entropies(state):
    """Trace each physical subtree from the full small reference state."""
    plan = state.plan
    psi = _dense_state(state).reshape((2,) * plan.n)
    values = []
    for _, child in state.tree_edges():
        nodes = set()
        stack = [child]
        while stack:
            node = stack.pop()
            nodes.add(node)
            stack.extend(plan.children[node])
        left = [q for q in range(plan.n) if plan.node_of_qubit[q] in nodes]
        right = [q for q in range(plan.n) if q not in left]
        matrix = psi.transpose(left + right).reshape(2 ** len(left), -1)
        weights = np.linalg.svd(matrix, compute_uv=False) ** 2
        weights /= weights.sum()
        positive = weights[weights > 0]
        values.append(-np.sum(positive * np.log2(positive)))
    return np.asarray(values)


@pytest.mark.parametrize('root_qubit', [None, 2])
@pytest.mark.parametrize('backend', ['numpy', 'torch', 'cupy'])
def test_tree_entropy_matches_unequal_dense_schmidt_spectra(backend, root_qubit):
    order = [q for q in range(5) if q != root_qubit]
    plan = pepsy.TreePlan.from_order(
        order, structure='balanced', root_qubit=root_qubit,
    )
    state = pepsy.TreeTensorNetwork.rand(plan, D=3, seed=23, dtype='complex128')
    tolerance = 1e-10
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    elif backend == 'cupy':
        cupy = pytest.importorskip('cupy')
        try:
            if cupy.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('CUDA unavailable')
        except cupy.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
        state.apply_to_arrays(lambda a: cupy.asarray(a, dtype=cupy.complex64))
        tolerance = 5e-6
    state.shift_orthogonality_center(plan.node_of_qubit[0])
    expected = _dense_edge_entropies(state)
    before = _dense_state(state).copy()
    region = state.canonical_region
    sampler = TreeSampler(state)
    configurations = np.zeros((1, plan.n), dtype=int)
    probabilities = sampler.probabilities(configurations).copy()
    for method in ('svd', 'eig'):
        actual, edges = state.tree_edge_entropies(method=method, return_edges=True)
        assert edges == state.tree_edges()
        np.testing.assert_allclose(actual, expected, atol=tolerance)
        np.testing.assert_allclose(
            sampler.tree_edge_entropies(method=method), expected, atol=tolerance,
        )
    for edge, value in zip(edges, expected):
        assert state.entropy(edge) == pytest.approx(value, abs=tolerance)
        assert state.entropy(edge[::-1]) == pytest.approx(value, abs=tolerance)
    np.testing.assert_allclose(_dense_state(state), before, atol=tolerance)
    assert state.canonical_region == region
    np.testing.assert_array_equal(sampler.probabilities(configurations), probabilities)
    # A sampler is a snapshot until refresh, even if the source later changes.
    tensor = state.node_tensor(state.node_of_qubit(0))
    gate = np.diag([1.0, 0.2]).astype(complex)
    if backend == 'torch':
        gate = torch.as_tensor(gate, dtype=torch.complex128)
    elif backend == 'cupy':
        gate = cupy.asarray(gate, dtype=cupy.complex64)
    tensor.gate_(gate, state.site_ind(0))
    state.invalidate_canonical_form()
    np.testing.assert_allclose(sampler.tree_edge_entropies(), expected, atol=tolerance)
    sampler.refresh()
    np.testing.assert_allclose(
        sampler.tree_edge_entropies(), _dense_edge_entropies(state), atol=tolerance,
    )


@pytest.mark.parametrize('fermionic', [False, True])
def test_native_tree_entropy_matches_nonmaximal_charge_sector_state(fermionic):
    pytest.importorskip('symmray')
    plan = pepsy.TreePlan.from_order(range(4), structure='balanced')
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan, symmetry='U1', physical_sectors={0: 1, 1: 1},
        leaf_charges={0: 0, 1: 1, 2: 0, 3: 1}, bond_dim=3,
        seed=7, dtype='complex128', fermionic=fermionic,
    )
    expected = _dense_edge_entropies(state)
    before = _dense_state(state).copy()
    for method in ('svd', 'eig'):
        np.testing.assert_allclose(
            state.tree_edge_entropies(method=method), expected, atol=1e-10,
        )
        np.testing.assert_allclose(
            TreeSampler(state, backend='symmray').tree_edge_entropies(method=method),
            expected, atol=1e-10,
        )
    np.testing.assert_allclose(_dense_state(state), before, atol=1e-12)

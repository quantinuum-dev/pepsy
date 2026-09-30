"""Frontier preparation shares completed histories without listing collections."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import MPOClusterProductExpansion, MPOParameter, MPOProductTerm, prepare_cluster_channels
from test_cluster_channel_structure import builder, contract
from test_cluster_channels import source, arguments

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def disjoint_crossings(blocks, **options):
    n = 4*blocks
    edges = [(4*i+j, 4*i+j+2) for i in range(blocks) for j in (0, 1)]
    factors = [[MPOProductTerm.from_pauli(edge, 'ZZ', coefficient=MPOParameter('a')) for edge in edges],
               [MPOProductTerm.from_pauli((i,), 'X', coefficient=MPOParameter('b')) for i in range(n)]]
    return MPOClusterProductExpansion(n, factors, graph=(range(n), edges), cluster_size=2,
        factorization='fixed', cutoff=0., graph_assembly='exact', **options)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('crossing', [False, True])
def test_frontier_exact_for_independent_residuals_and_adjoint_derivatives(native, crossing):
    torch = pytest.importorskip('torch')
    owner = builder(native, crossing)
    ordinary = prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, structural_reuse=False)
    frontier = prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='frontier')
    rng = np.random.default_rng(208)
    residuals = tuple(torch.tensor(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape),
                                   dtype=torch.complex128, requires_grad=True)
                      for x in frontier.pack_residuals(-.1j, {'a': .3, 'b': .2}))
    actual = contract(frontier.bind_assembler(residuals[0])(residuals))
    expected = contract(ordinary.bind_assembler(residuals[0])(residuals))
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    weight = torch.tensor(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32)))
    gradients = torch.autograd.grad((actual*weight).real.sum(), residuals)
    reference = torch.autograd.grad((expected*weight).real.sum(), residuals)
    for a, b in zip(gradients, reference):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=3e-11)
    assert frontier.report['full_collections_enumerated'] == 0


def test_many_completed_collections_never_enumerated_and_trace_is_actual_mpo():
    torch = pytest.importorskip('torch')
    # Twenty pair clusters have 2**20-1 nonempty collections, but at most
    # two clusters cross any one chain cut. Their frontier has four states.
    owner = disjoint_crossings(10, collection_budget=1)
    reference = {'a': .3, 'b': .2}
    with pytest.raises(ValueError, match='collection_budget|budget'):
        prepare_cluster_channels(owner, -.1j, reference)
    with ExitStack() as stack:
        for name in ('_bounded_graph_cluster_collections', '_graph_collection_plan', '_assemble',
                     '_assemble_graph', '_assemble_graph_collections', '_assemble_graph_recursive', 'exp',
                     'trace_exp'):
            stack.enter_context(patch.object(MPOClusterProductExpansion, name,
                                            side_effect=AssertionError('full collection/source assembly')))
        plan = prepare_cluster_channels(owner, -.1j, reference, preparation='frontier')
        for values in ([.31, -.19], [0., 0.]):
            theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            value = plan.trace_exp(-.1j, dict(zip('ab', theta)), normalized=True)
            expected = (torch.cos(.1*theta[0])*torch.cos(.1*theta[1])**2)**20
            # Torch's existing local exponential path differs from the analytic
            # trace by 1.6e-11 here (the old assembler has the same bias).
            torch.testing.assert_close(value.real, expected, atol=3e-11, rtol=3e-13)
            actual_grad, = torch.autograd.grad(value.real, theta)
            expected_grad, = torch.autograd.grad(expected, theta)
            torch.testing.assert_close(actual_grad, expected_grad, atol=3e-12, rtol=3e-12)
        numpy_value = plan.trace_exp(-.1j, {'a': .31, 'b': -.19}, normalized=True)
        np.testing.assert_allclose(numpy_value, (np.cos(.031)*np.cos(.019)**2)**20, atol=3e-13)
    assert max(plan.report['frontier_state_counts']) == 4
    assert max(plan.report['frontier_bond_dimensions']) == 25
    assert plan.report['full_collections_enumerated'] == plan.report['replay_history_blocks'] == 0
    with patch.object(plan, 'exp', side_effect=RuntimeError('actual materialization')):
        with pytest.raises(RuntimeError, match='actual materialization'):
            plan.trace_exp(-.1j, reference)


@pytest.mark.parametrize('kind', ['mpo', 'native'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_frontier_capped_values_gradients_and_full_assembly_compilation(kind, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]),
                                     preparation='frontier', max_bond=3)
    for values in ([.1, .4, -.3, .8], [0., 0., 0., .7]):
        theta = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
        residuals = plan.pack_residuals(**arguments(theta))
        packed = plan.pack(**arguments(theta))
        expected = plan.bind_projector(packed[0])(packed)
        kernel = plan.bind_assembler(residuals[0])
        with patch('scipy.linalg.qr', side_effect=AssertionError('QR in replay')), \
                patch('torch.linalg.svd', side_effect=AssertionError('SVD in replay')):
            eager = kernel(residuals)
        # Dynamo inspects the Torch namespace itself; a mocked torch.linalg
        # entry prevents that inspection, unrelated to the kernel's graph.
        actual = torch.compile(kernel, backend='aot_eager', fullgraph=True, dynamic=False)(residuals) \
            if device == 'cpu' else eager
        reference_grad, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in expected), theta)
        gradient, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in actual), theta)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(gradient, reference_grad)
        eps = 1e-5
        def objective(v):
            return sum((x.real+.23*x.imag).sum() for x in plan.arrays(**arguments(v)))
        fd = [(objective(values+eps*e)-objective(values-eps*e))/(2*eps) for e in np.eye(4)]
        np.testing.assert_allclose(gradient.detach().cpu(), fd, atol=3e-9)


@pytest.mark.parametrize('kind', ['mpo', 'native'])
def test_frontier_jax_trace_jit(kind):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]),
                                     preparation='frontier', max_bond=3)
    with jax.enable_x64(True):
        function = jax.jit(jax.value_and_grad(lambda t: plan.trace_exp(**arguments(t)).real))
        values = np.array([.1, .4, -.3, .8])
        value, gradient = function(jnp.asarray(values))
        objective = lambda t: plan.trace_exp(**arguments(t)).real
        eps = 1e-5
        expected = [(objective(values+eps*e)-objective(values-eps*e))/(2*eps) for e in np.eye(4)]
        np.testing.assert_allclose(value, objective(values), atol=3e-13)
        np.testing.assert_allclose(gradient, expected, atol=3e-9)


def test_frontier_policy_guards_and_recursive_budget_are_explicit():
    owner = disjoint_crossings(2, collection_budget=1, assembly='recursive')
    prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='frontier', max_bond=3)
    assert owner.assembly == 'recursive' and owner.collection_budget == 1
    owner.assembly_state_budget = 1
    with pytest.raises(ValueError, match='frontier_state_budget'):
        prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='frontier')
    prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='frontier', frontier_state_budget=4)
    assert owner.assembly_state_budget == 1
    for limit in (0, True, 1.5):
        with pytest.raises(ValueError, match='frontier_state_budget must be'):
            prepare_cluster_channels(owner, preparation='frontier', frontier_state_budget=limit)
    with pytest.raises(ValueError, match='frontier_state_budget requires'):
        prepare_cluster_channels(owner, frontier_state_budget=4)
    for mode in ('auto', 'bounded'):
        owner = disjoint_crossings(1)
        owner.graph_assembly = mode
        with pytest.raises(ValueError, match="graph_assembly='exact'"):
            prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='frontier')
    with pytest.raises(ValueError, match='MPO builders only'):
        prepare_cluster_channels(source('graph'), **arguments([.2, -.3, .1, .7]), preparation='frontier')
    with pytest.raises(ValueError, match='preparation must be'):
        prepare_cluster_channels(source('mpo'), preparation='automatic')


def test_gapped_three_body_cluster_and_crossing_pair_preserve_joint_product():
    torch = pytest.importorskip('torch')
    factors = [[MPOProductTerm.from_pauli((0, 2, 4), 'XZY', coefficient=MPOParameter('a'))],
               [MPOProductTerm.from_pauli((1, 3), 'YX', coefficient=MPOParameter('b'))],
               [MPOProductTerm.from_pauli((i,), 'Z', coefficient=MPOParameter('c')) for i in range(5)]]
    owner = MPOClusterProductExpansion(5, factors, graph=(range(5), ((0, 2), (2, 4), (1, 3))),
        cluster_size=3, factorization='fixed', cutoff=0., graph_assembly='exact')
    plan = prepare_cluster_channels(owner, -.13j, dict(a=.2, b=-.3, c=.1), preparation='frontier')
    for values in ([.3, -.2, .4], [0., 0., 0.]):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        bindings = dict(zip('abc', theta))
        expected = owner.exp(-.13j, bindings).to_mpo().to_dense()
        actual = plan.exp(-.13j, bindings).to_dense()
        torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
        reference, = torch.autograd.grad((expected.real+.23*expected.imag).sum(), theta)
        gradient, = torch.autograd.grad((actual.real+.23*actual.imag).sum(), theta)
        torch.testing.assert_close(gradient, reference, atol=3e-12, rtol=3e-12)

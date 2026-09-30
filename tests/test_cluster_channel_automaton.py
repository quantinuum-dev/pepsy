"""Pauli weighted-automaton reduction preserves the declared cluster family."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import MPOClusterProductExpansion, MPOParameter, MPOProductTerm, prepare_cluster_channels
from test_cluster_channel_frontier import disjoint_crossings
from test_cluster_channel_structure import builder, contract
from test_cluster_channels import arguments, dense, fd, loss, reference, source

pytestmark = [pytest.mark.integration, pytest.mark.operators]


@pytest.mark.parametrize('word', ['XX', 'YY', 'ZZ'])
def test_pauli_automaton_removes_states_beyond_identical_paths(word):
    terms = [MPOProductTerm.from_pauli((i, i+1), word, coefficient=MPOParameter('a')) for i in range(3)]
    owner = MPOClusterProductExpansion(4, [terms], cluster_size=4, factorization='fixed', cutoff=0.)
    frontier = prepare_cluster_channels(owner, -.1j, {'a': 0.}, preparation='frontier')
    automaton = prepare_cluster_channels(owner, -.1j, {'a': 0.}, preparation='automaton')
    assert max(automaton.report['bond_dimensions']) < max(frontier.report['bond_dimensions'])
    assert automaton.report['structural_reuse']['reduction'] == 'rational-delinearisation'
    pauli = {'X': np.array([[0., 1.], [1., 0.]]),
             'Y': np.array([[0., -1j], [1j, 0.]]), 'Z': np.diag([1., -1.])}[word[0]]
    hamiltonian = sum(np.kron(np.eye(2**i), np.kron(np.kron(pauli, pauli), np.eye(2**(2-i))))
                      for i in range(3))
    for coupling in (0., .37, -.2):
        expected = expm(-.1j*coupling*hamiltonian)
        np.testing.assert_allclose(automaton.exp(-.1j, {'a': coupling}).to_dense(), expected, atol=8e-14)


@pytest.mark.parametrize('crossing', [False, True])
def test_independent_projected_residuals_and_adjoint_derivatives(crossing):
    torch = pytest.importorskip('torch')
    owner = builder(crossing=crossing)
    args = dict(step=-.1j, parameters={'a': 0., 'b': 0.})
    raw = prepare_cluster_channels(owner, **args, preparation='frontier', structural_reuse=False)
    plan = prepare_cluster_channels(owner, **args, preparation='automaton')
    rng = np.random.default_rng(299)
    residuals = plan.pack_residuals(**args)
    inputs = tuple(torch.tensor(rng.normal(size=v.shape)+1j*rng.normal(size=v.shape),
                                dtype=torch.complex128, requires_grad=True) for v in residuals)
    projected = tuple(space.bind(v)[1](v) for space, v in zip(plan._binding.algebra.spaces, inputs))
    actual = contract(plan.bind_assembler(inputs[0])(inputs))
    expected = contract(raw.bind_assembler(inputs[0])(projected))
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=3e-12)
    weight = torch.tensor(rng.normal(size=actual.shape)+1j*rng.normal(size=actual.shape))
    a = torch.autograd.grad((actual*weight).real.sum(), inputs)
    b = torch.autograd.grad((expected*weight).real.sum(), inputs)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=3e-11, rtol=3e-12)


def test_no_decomposition_no_collection_expansion_and_zero_parameter_gradients():
    torch = pytest.importorskip('torch')
    owner = source('mpo')
    with ExitStack() as stack:
        for target in ('numpy.linalg.svd', 'scipy.linalg.svd', 'scipy.linalg.qr', 'numpy.linalg.qr',
                       'torch.linalg.svd', 'torch.linalg.qr'):
            stack.enter_context(patch(target, side_effect=AssertionError('numerical decomposition')))
        for name in ('exp', '_graph_collection_plan', '_assemble', '_assemble_graph',
                     '_assemble_graph_collections', '_assemble_graph_recursive'):
            stack.enter_context(patch.object(MPOClusterProductExpansion, name,
                                            side_effect=AssertionError('expanded construction')))
        plan = prepare_cluster_channels(owner, **arguments([0., 0., 0., .7]), preparation='automaton')
        for values in (np.array([0., 0., 0., .7]), np.array([.23, -.31, .17, .83])):
            theta = torch.tensor(values, requires_grad=True)
            actual = dense(plan, theta)
            np.testing.assert_allclose(actual.detach(), reference(values), atol=3e-13)
            gradient, = torch.autograd.grad(loss(actual), theta)
            np.testing.assert_allclose(gradient, fd(lambda v: loss(reference(v)), values), atol=3e-9)
    assert plan.report['preparation'] == 'automaton'
    assert plan.report['method'] == 'fixed-pauli-automaton'
    assert plan.report['full_collections_enumerated'] == plan.report['replay_history_blocks'] == 0
    assert plan.report['replay_factorizations'] == 0


def test_structural_maps_do_not_depend_on_reference_time_or_couplings():
    owner = builder(crossing=True)
    plans = [prepare_cluster_channels(owner, t, dict(zip('ab', values)), preparation='automaton')
             for t, values in ((0., (0., 0.)), (-.1j, (.2, .2)), (.13, (-.3, .4)))]
    for plan in plans[1:]:
        assert plan.report['structural_reuse'] == plans[0].report['structural_reuse']
        for a, b in zip(plan.bases+plan._dual_bases, plans[0].bases+plans[0]._dual_bases):
            np.testing.assert_array_equal(a, b)


@pytest.fixture
def isolated_torch_compile():
    # Each case compiles a different static plan. Do not consume the shared
    # eight-entry Dynamo frame cache of unrelated compilation regressions.
    torch = pytest.importorskip('torch')
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize('cap', [None, 3])
def test_torch_fullgraph_replay_and_jax_trace_gradients(cap, isolated_torch_compile):
    torch = pytest.importorskip('torch')
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    plan = prepare_cluster_channels(source('mpo'), **arguments([.2, -.3, .1, .7]),
                                     preparation='automaton', max_bond=cap)
    theta = torch.tensor([.1, .4, -.3, .8], dtype=torch.float64, requires_grad=True)
    residuals = plan.pack_residuals(**arguments(theta))
    kernel = plan.bind_assembler(residuals[0])
    actual = torch.compile(kernel, backend='aot_eager', fullgraph=True, dynamic=False)(residuals)
    packed = plan.pack(**arguments(theta))
    expected = plan.bind_projector(packed[0])(packed)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    grad = torch.autograd.grad(sum(loss(v) for v in actual), theta)
    ref_grad = torch.autograd.grad(sum(loss(v) for v in expected), theta)
    torch.testing.assert_close(grad, ref_grad)
    values = np.array([.1, .4, -.3, .8])
    with jax.enable_x64(True):
        function = jax.jit(jax.value_and_grad(lambda t: plan.trace_exp(**arguments(t)).real))
        value, gradient = function(jnp.asarray(values))
        objective = lambda v: plan.trace_exp(**arguments(v)).real
        np.testing.assert_allclose(value, objective(values), atol=3e-13)
        np.testing.assert_allclose(gradient, fd(objective, values), atol=3e-9)


def test_independent_coefficients_single_precision():
    torch = pytest.importorskip('torch')
    owner = source('mpo', dtype=np.complex64)
    vectors = (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32))
    plan = prepare_cluster_channels(owner, -.1j, coefficients=vectors, preparation='automaton')
    theta = torch.tensor([0., .21, -.1, .13, .2, .3, -.2], dtype=torch.float32, requires_grad=True)
    coefficients = (theta[:2], theta[2:4], theta[4:])
    actual = plan.exp(-.1j, coefficients=coefficients).to_dense()
    expected = owner.exp(-.1j, coefficients=coefficients).to_mpo().to_dense()
    assert actual.dtype == expected.dtype == torch.complex64
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), theta),
                               torch.autograd.grad(loss(expected), theta))


def test_guards_budget_operator_inventory_and_opt_out():
    with pytest.raises(ValueError, match='dense qubit Pauli'):
        prepare_cluster_channels(source('native'), preparation='automaton')
    with pytest.raises(ValueError, match='MPO builders only'):
        prepare_cluster_channels(source('graph'), preparation='automaton')
    term = MPOProductTerm((0,), (np.diag([1., 2.]),))
    owner = MPOClusterProductExpansion(2, [[term]], cluster_size=1, factorization='fixed', cutoff=0.)
    with pytest.raises(ValueError, match='proportional'):
        prepare_cluster_channels(owner, preparation='automaton')
    owner = builder()
    args = dict(step=-.1j, parameters={'a': .2, 'b': -.3})
    with pytest.raises(MemoryError, match='memory_budget'):
        prepare_cluster_channels(owner, **args, preparation='automaton', memory_budget=100)
    plan = prepare_cluster_channels(owner, **args, preparation='automaton', structural_reuse=False)
    np.testing.assert_allclose(plan.exp(**args).to_dense(), owner.exp(**args).to_mpo().to_dense(), atol=5e-13)
    operator = owner.factors[0].terms[0].operators[0]
    original = operator.copy()
    try:
        operator[:] = np.array([[0., 1.], [1., 0.]])
        with pytest.raises(ValueError, match='inventory changed'):
            plan.arrays(**args)
    finally:
        operator[:] = original
    owner = disjoint_crossings(2, collection_budget=1)
    with pytest.raises(ValueError, match='frontier_state_budget'):
        prepare_cluster_channels(owner, **args, preparation='automaton', frontier_state_budget=1)
    plan = prepare_cluster_channels(owner, **args, preparation='automaton', frontier_state_budget=4)
    assert plan.report['full_collections_enumerated'] == 0


def test_forty_site_automaton_never_enumerates_complete_collections():
    owner = disjoint_crossings(10, collection_budget=1)
    with ExitStack() as stack:
        for name in ('exp', '_graph_collection_plan', '_bounded_graph_cluster_collections',
                     '_assemble_graph', '_assemble_graph_collections', '_assemble_graph_recursive'):
            stack.enter_context(patch.object(MPOClusterProductExpansion, name,
                                            side_effect=AssertionError('expanded collection construction')))
        plan = prepare_cluster_channels(owner, -.1j, {'a': .3, 'b': .2}, preparation='automaton')
        for a, b in ((0., 0.), (.31, -.19)):
            expected = (np.cos(.1*a)*np.cos(.1*b)**2)**20
            np.testing.assert_allclose(plan.trace_exp(-.1j, {'a': a, 'b': b}, normalized=True),
                                       expected, atol=3e-13)
    assert max(plan.report['frontier_state_counts']) == 4
    assert plan.report['full_collections_enumerated'] == plan.report['replay_history_blocks'] == 0

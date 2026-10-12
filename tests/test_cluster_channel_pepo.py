"""Exact edge quotients preserve PEPO connectivity and live parameter directions."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    PauliPEPOBasis, prepare_cluster_channels,
)
from test_cluster_channels import arguments, dense, fd, loss, reference, source

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def geometry(kind, case):
    if case == 'branch':
        supports, words, order = ((0, 1), (0, 2), (0, 3)), ('XZ',)*3, 4
    elif case == 'loop':
        supports, words, order = ((0, 1), (1, 3), (2, 3), (0, 2)), ('XZ',)*4, 3
    elif case == 'crossing':
        supports, words, order = ((0, 3), (1, 2)), ('XZ',)*2, 2
    else:
        supports, words, order = ((0, 2, 3),), ('XYZ',), 3
    factors = [[MPOProductTerm.from_pauli(s, word, coefficient=MPOParameter('a'))
                for s, word in zip(supports, words)],
               [MPOProductTerm.from_pauli((i,), 'X', coefficient=MPOParameter('b')) for i in range(4)]]
    plan = ClusterPlan.from_terms([t for f in factors for t in f], sites=4, shape=(2, 2),
                                  cyclic=case == 'crossing', cluster_size=order)
    return PEPOClusterProductExpansion.from_plan(plan, factors, layout=kind, factorization='fixed')


def contract(plan, arrays):
    import quimb.tensor as qtn

    layout = plan._layout
    indices = [[None]*len(keys[0]) for keys in layout.keys]
    for edge, ends in enumerate(layout.ends):
        for site, axis in ends:
            indices[site][axis] = ('v', edge)
    return qtn.TensorNetwork([qtn.Tensor(a, inds=(*inds, ('y', i), ('x', i)))
        for i, (a, inds) in enumerate(zip(arrays, indices))]).to_dense(
            [('y', i) for i in range(len(arrays))], [('x', i) for i in range(len(arrays))], optimize='greedy')


@pytest.mark.parametrize('kind', ['graph', 'square'])
@pytest.mark.parametrize('case', ['branch', 'loop', 'crossing', 'higher_body'])
def test_symbolic_pepo_preserves_arbitrary_residuals_and_adjoints(kind, case):
    torch = pytest.importorskip('torch')
    builder = geometry(kind, case)
    kw = dict(step=-.1j, parameters={'a': .2, 'b': -.3})
    expanded = prepare_cluster_channels(builder, **kw)
    shared = prepare_cluster_channels(builder, **kw, preparation='symbolic')
    rng = np.random.default_rng(814)
    inputs = tuple(torch.tensor(rng.normal(size=v.shape)+1j*rng.normal(size=v.shape),
                                dtype=torch.complex128, requires_grad=True)
                   for v in shared.pack_residuals(**kw))
    weight = torch.tensor(rng.normal(size=(16, 16))+1j*rng.normal(size=(16, 16)))
    expected = contract(expanded, expanded.bind_assembler(inputs[0])(inputs))
    actual = contract(shared, shared.bind_assembler(inputs[0])(inputs))
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-12)
    a = torch.autograd.grad((actual*weight).real.sum(), inputs)
    b = torch.autograd.grad((expected*weight).real.sum(), inputs)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=3e-11, rtol=3e-12)
    assert shared.report['expanded_reference_builds'] == 0
    assert shared.report['output_dense_entries'] <= expanded.report['output_dense_entries']


@pytest.mark.parametrize('kind', ['graph', 'square'])
def test_symbolic_preparation_reuses_templates_and_preserves_zero_gradients(kind):
    builder = source(kind)
    values = np.array([.2, -.3, .1, .7])
    with patch('pepsy.operators.graph_pepo_product.GraphPEPOClusterProductExpansion.exp',
               side_effect=AssertionError('expanded reference builder')):
        plan = prepare_cluster_channels(builder, **arguments(values), preparation='symbolic',
            samples=[arguments(-values)], tangent_pairs=[(arguments(values+.001), arguments(values-.001), .002)])
    assert plan.report['structural_reuse']['removed_channels'] > 0
    assert plan.report['bond_dimensions'] == (9, 5)
    torch = pytest.importorskip('torch')
    for v in (values, np.array([0., 0., 0., .7])):
        theta = torch.tensor(v, requires_grad=True)
        actual = dense(plan, theta)
        np.testing.assert_allclose(actual.detach(), reference(v), atol=3e-13)
        gradient, = torch.autograd.grad(loss(actual), theta)
        np.testing.assert_allclose(gradient, fd(lambda x: loss(reference(x)), v), atol=3e-9)
        np.testing.assert_allclose(plan.trace_exp(**arguments(v)), np.trace(reference(v)), atol=3e-13)
    zero_plan = prepare_cluster_channels(builder, **arguments([0., 0., 0., .7]), preparation='symbolic')
    assert zero_plan.report['structural_reuse'] == plan.report['structural_reuse']
    for a, b in zip(plan._binding.pepo.graph_maps, zero_plan._binding.pepo.graph_maps):
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('kind', ['graph', 'square'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('preparation', ['symbolic', 'algebraic'])
def test_capped_symbolic_replay_and_diagnostic_projection_gradients(kind, device, preparation):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]),
                                    preparation=preparation, max_bond=3)
    theta = torch.tensor([0., 0., 0., .7], device=device, dtype=torch.float64, requires_grad=True)
    packed = plan.pack(**arguments(theta))
    expected = plan.bind_projector(packed[0])(packed)
    expected_grad, = torch.autograd.grad(sum(loss(v) for v in expected), theta)
    with ExitStack() as stack:
        for target in ('scipy.linalg.qr', 'torch.linalg.svd', 'torch.linalg.qr',
                       'pepsy.operators.pepo_routing.route_graph_blocks',
                       'pepsy.operators.graph_pepo_product.GraphPEPOClusterProductExpansion.exp'):
            stack.enter_context(patch(target, side_effect=AssertionError('factorization/expanded replay')))
        actual = plan.arrays(**arguments(theta))
        actual_grad, = torch.autograd.grad(sum(loss(v) for v in actual), theta)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=3e-13, rtol=3e-13)
        assert a.device == theta.device and a.dtype == torch.complex128
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('kind', ['graph', 'square'])
@pytest.mark.parametrize('preparation', ['symbolic', 'algebraic'])
def test_symbolic_jax_trace_and_torch_assembly_compile(kind, preparation):
    torch = pytest.importorskip('torch')
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    values = np.array([.21, -.29, .13, .71])
    plan = prepare_cluster_channels(source(kind), **arguments(values), preparation=preparation, max_bond=3)
    with jax.enable_x64(True):
        objective = lambda theta: plan.trace_exp(**arguments(theta)).real
        value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(values))
        np.testing.assert_allclose(value, objective(values), atol=3e-13)
        np.testing.assert_allclose(gradient, fd(objective, values), atol=3e-9)
    residuals = tuple(torch.tensor(v, requires_grad=True) for v in plan.pack_residuals(**arguments(values)))
    kernel = plan.bind_assembler(residuals[0])
    compiled = torch.compile(kernel, backend='aot_eager', fullgraph=True, dynamic=False)
    actual, expected = compiled(residuals), kernel(residuals)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    a = torch.autograd.grad(sum(loss(v) for v in actual), residuals)
    b = torch.autograd.grad(sum(loss(v) for v in expected), residuals)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_symbolic_guards_and_opt_out():
    with pytest.raises(ValueError, match='PEPO builders only'):
        prepare_cluster_channels(source('mpo'), preparation='symbolic')
    with pytest.raises(MemoryError, match='memory_budget'):
        prepare_cluster_channels(source('square'), **arguments([.2, -.3, .1, .7]),
                                 preparation='symbolic', memory_budget=100)
    builder = source('square')
    reference_plan = prepare_cluster_channels(builder, **arguments([.2, -.3, .1, .7]))
    plan = prepare_cluster_channels(builder, **arguments([.2, -.3, .1, .7]),
                                    preparation='symbolic', structural_reuse=False)
    assert plan.report['structural_reuse']['removed_channels'] == 0
    assert plan.report['bond_dimensions'] == reference_plan.report['bond_dimensions']
    np.testing.assert_allclose(dense(plan, [.3, -.2, .1, .8]), dense(reference_plan, [.3, -.2, .1, .8]), atol=3e-13)


@pytest.mark.parametrize('kind', ['graph', 'square'])
def test_identity_zeros_are_removed_without_decompositions(kind):
    terms = [MPOProductTerm.from_pauli((i, i+1), 'ZZ', coefficient=MPOParameter('a')) for i in range(3)]
    geometry = ClusterPlan.from_terms(terms, sites=4, shape=(1, 4), cluster_size=4)
    builder = PEPOClusterProductExpansion.from_plan(geometry, [terms], layout=kind, factorization='fixed')
    with patch('scipy.linalg.qr', side_effect=AssertionError('QR forbidden')), \
            patch('numpy.linalg.svd', side_effect=AssertionError('SVD forbidden')):
        plan = prepare_cluster_channels(builder, -.1j, {'a': 0.}, preparation='symbolic')
    assert plan.report['symbolic_zero_blocks_removed'] == 48
    assert plan.report['symbolic_routed_blocks'] < plan.report['unreduced_sparse_blocks']
    theta = {'a': .23}
    expanded = prepare_cluster_channels(builder, -.1j, theta)
    np.testing.assert_allclose(contract(plan, plan.arrays(-.1j, theta)),
                               contract(expanded, expanded.arrays(-.1j, theta)), atol=3e-13)


@pytest.mark.parametrize('kind', ['graph', 'square'])
@pytest.mark.parametrize('preparation', ['symbolic', 'algebraic'])
def test_symbolic_single_precision_and_independent_vector_bindings(kind, preparation):
    torch = pytest.importorskip('torch')
    builder = source(kind, dtype=np.complex64)
    vectors = ([.2, .2], [-.3, -.3], [.1, .1, .1])
    plan = prepare_cluster_channels(builder, -.1j, coefficients=vectors, preparation=preparation, max_bond=3)
    # Independent slots may agree in the reference, then separate on replay.
    theta = torch.tensor([0., .21, -.1, .13, .2, .3, -.2], dtype=torch.float32, requires_grad=True)
    coefficients = (theta[:2], theta[2:4], theta[4:])
    packed = plan.pack(-.1j, coefficients=coefficients)
    expected = plan.bind_projector(packed[0])(packed)
    actual = plan.arrays(-.1j, coefficients=coefficients)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype == torch.complex64
        torch.testing.assert_close(a, b)
    a = torch.autograd.grad(sum(loss(v) for v in actual), theta)
    b = torch.autograd.grad(sum(loss(v) for v in expected), theta)
    torch.testing.assert_close(a, b)


def test_reduction_precedes_routed_allocation_budget():
    builder = geometry('square', 'branch')
    options = dict(step=-.1j, parameters={'a': .2, 'b': -.3}, memory_budget=1024**2)
    with pytest.raises(MemoryError, match='memory_budget'):
        prepare_cluster_channels(builder, **options)
    plan = prepare_cluster_channels(builder, **options, preparation='symbolic')
    assert plan.report['unreduced_bond_dimensions'] == (289, 17, 1, 17)
    assert plan.report['bond_dimensions'] == (25, 5, 1, 5)
    assert plan.report['symbolic_routed_blocks'] == 160
    # The budget is an allocation guard, not an implicit approximation.
    target = prepare_cluster_channels(builder, -.1j, {'a': .2, 'b': -.3})
    np.testing.assert_allclose(plan.exp(-.1j, {'a': .31, 'b': -.17}).to_dense(),
                               target.exp(-.1j, {'a': .31, 'b': -.17}).to_dense(), atol=3e-13)


@pytest.mark.parametrize('shape', [(1, 1), (1, 3)])
@pytest.mark.parametrize('preparation', ['symbolic', 'algebraic'])
def test_single_pauli_basis_and_isolated_sites(shape, preparation):
    builder = PauliPEPOBasis(*shape, [('onsite', 'X', .2)], cluster_size=2, factorization='fixed')
    plan = prepare_cluster_channels(builder.compile_exp(), -.1j, preparation=preparation)
    np.testing.assert_allclose(plan.exp(-.1j).to_dense(), builder.exp(-.1j).to_dense(), atol=3e-13)

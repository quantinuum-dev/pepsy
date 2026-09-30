"""Fused construction agrees with explicit projection without history replay."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOClusterProductExpansion, MPOProductTerm,
    PEPOClusterProductExpansion, PauliPEPOBasis, prepare_cluster_channels,
)
from test_cluster_channels import arguments, source

pytestmark = [pytest.mark.integration, pytest.mark.operators]


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
@pytest.mark.parametrize('cap', [None, 3])
def test_fused_replay_matches_explicit_projection_without_expanded_builders(kind, cap):
    torch = pytest.importorskip('torch')
    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]), max_bond=cap)
    for values in ([.1, .4, -.3, .8], [0., 0., 0., .7]):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        packed = plan.pack(**arguments(theta))
        expected = plan.bind_projector(packed[0])(packed)
        expected_grad, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in expected), theta)
        with ExitStack() as stack:
            for target in ('pepsy.operators.cluster_channels._capture',
                           'pepsy.operators.mpo_product.MPOClusterProductExpansion.exp',
                           'pepsy.operators.mpo_product.MPOClusterProductExpansion._multiply_mpo_cores',
                           'pepsy.operators.graph_pepo_product.GraphPEPOClusterProductExpansion.exp',
                           'pepsy.operators.pepo_routing.route_graph_blocks',
                           'scipy.linalg.qr', 'torch.linalg.svd', 'torch.linalg.qr'):
                stack.enter_context(patch(target, side_effect=AssertionError('expanded construction or factorization')))
            actual = plan.arrays(**arguments(theta))
            gradient, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in actual), theta)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-13)
        torch.testing.assert_close(gradient, expected_grad, atol=3e-12, rtol=3e-12)
    assert plan.report['replay_history_blocks'] == 0
    assert plan.report['assembly'] == 'fused-local-contractions'


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
def test_crossing_gaps_collections_and_coefficient_vectors(kind):
    torch = pytest.importorskip('torch')
    native = kind == 'native'
    words = ('ZZ',) if native else ('XZ',)
    factors = [[MPOProductTerm.from_pauli(edge, word, coefficient=.2) for word in words]
               for edge in ((0, 3), (1, 2))]
    geometry = ClusterPlan.from_terms([t for f in factors for t in f], sites=4,
                                      shape=(2, 2), cyclic=True, cluster_size=2)
    if kind in ('mpo', 'native'):
        builder = MPOClusterProductExpansion.from_plan(geometry, factors, factorization='fixed', cutoff=0.,
                    graph_assembly='exact', **(dict(symmetry='U1', physical_charges=(0, 1)) if native else {}))
    else:
        builder = PEPOClusterProductExpansion.from_plan(geometry, factors, layout=kind, factorization='fixed')
    for cap in (None, 3):
        plan = prepare_cluster_channels(builder, -.2j, max_bond=cap)
        theta = torch.tensor([.3, -.4], dtype=torch.float64, requires_grad=True)
        coefficients = tuple(theta[i].expand(len(words)) for i in range(2))
        packed = plan.pack(-.2j, coefficients=coefficients)
        expected = plan.bind_projector(packed[0])(packed)
        with patch('pepsy.operators.mpo_product.MPOClusterProductExpansion._multiply_mpo_cores',
                   side_effect=AssertionError('unprojected product')):
            actual = plan.arrays(-.2j, coefficients=coefficients)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-13)
        gradient, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in actual), theta)
        expected_grad, = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in expected), theta)
        torch.testing.assert_close(gradient, expected_grad, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('shape,order', [((1, 4), 3), ((2, 2), 3), ((1, 5), 4), ((2, 2), 4)])
def test_uniform_square_graph_residual_route_preserves_uncapped_operator(shape, order):
    builder = PEPOClusterProductExpansion.from_bases([
        PauliPEPOBasis(*shape, [('onsite', 'X', .2), ('edge', 'ZZ', .3)],
                       cluster_size=order, factorization='fixed')])
    # Compare the underlying exact channel construction without allocating an
    # identity projection for a large square degree-four site tensor.
    from pepsy.operators._cluster_channel_assembly import ChannelSource

    converted = ChannelSource(builder).evaluate(dict(step=-.11j, parameters=None, coefficients=None))
    np.testing.assert_allclose(converted.to_dense(), builder.exp(-.11j).to_dense(), atol=3e-13)
    plan = prepare_cluster_channels(builder, -.11j, max_bond=3)
    packed = plan.pack(-.11j)
    for x, y in zip(plan.arrays(-.11j), plan.bind_projector(packed[0])(packed)):
        np.testing.assert_allclose(x, y, atol=3e-13)


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
def test_full_assembly_kernel_torch_compile(kind):
    torch = pytest.importorskip('torch')
    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]), max_bond=3)
    residuals = plan.pack_residuals(**arguments(torch.tensor([.2, -.3, .1, .7], dtype=torch.float64)))
    kernel = plan.bind_assembler(residuals[0])
    compiled = torch.compile(kernel, backend='aot_eager', fullgraph=True, dynamic=False)
    values = tuple(v.detach().requires_grad_(True) for v in residuals)
    expected, actual = kernel(values), compiled(values)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y)
    gradients = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in actual), values)
    reference = torch.autograd.grad(sum((x.real+.23*x.imag).sum() for x in expected), values)
    for x, y in zip(gradients, reference):
        torch.testing.assert_close(x, y)


def test_fused_native_replay_validates_new_independent_coefficient_bindings():
    builder = source('native')
    plan = prepare_cluster_channels(builder, **arguments([.2, -.3, .1, .7]), max_bond=3)
    # Independent XX and YY slots do not conserve U1 as a parameter family,
    # even when their values happen to agree at this evaluation.
    with pytest.raises(ValueError, match='every independent coefficient binding'):
        plan.arrays(-.1j, coefficients=([.2, .2], [.3, .3], [.1, .1, .1]))


@pytest.mark.parametrize('native', [False, True])
def test_fixed_recursive_input_keeps_exact_collections_and_caller_policy(native):
    factors = [[MPOProductTerm.from_pauli(edge, 'ZZ', coefficient=.2)] for edge in ((0, 3), (1, 2))]
    geometry = ClusterPlan.from_terms([t for f in factors for t in f], sites=4, cluster_size=2)
    builder = MPOClusterProductExpansion.from_plan(geometry, factors, factorization='fixed', cutoff=0.,
                assembly='recursive', **(dict(symmetry='U1', physical_charges=(0, 1)) if native else {}))
    plan = prepare_cluster_channels(builder, -.2j)
    np.testing.assert_allclose(plan.exp(-.2j).to_dense(), builder.exp(-.2j).to_mpo().to_dense(), atol=3e-13)
    assert builder.assembly == 'recursive'
    assert plan.report['replay_history_blocks'] == 0
    # Preparation must refuse a too-small enumeration budget, never adopt
    # the direct auto strategy's one-cluster approximation silently.
    builder.collection_budget = 1
    with pytest.raises(ValueError, match='collection_budget|budget'):
        prepare_cluster_channels(builder, -.2j, max_bond=3)


@pytest.mark.parametrize('symmetry,charges', [
    ('U1', (0, 0, 1)), ('Z2', (0, 0, 1)),
    ('U1U1', ((0, 0), (0, 1), (1, 0), (1, 1))),
    ('Z2Z2', ((0, 0), (0, 1), (1, 0), (1, 1))),
])
def test_bound_native_split_preserves_sector_factors_and_derivatives(symmetry, charges):
    torch = pytest.importorskip('torch')
    from pepsy.operators._cluster_native import bind_fixed_sector_operator, fixed_sector_operator
    from pepsy.operators.mpo_semantic import MPOPhysicalSpace

    d, n = len(charges), 3
    space = MPOPhysicalSpace(d, symmetry=symmetry, physical_charges=charges)
    # Diagonal entries conserve every tested charge group; off-diagonal
    # directions are exercised by the hopping tests in test_cluster_channels.
    values = torch.arange(1., d**n+1, dtype=torch.float64, requires_grad=True)
    operator = torch.diag(values).to(torch.complex128)
    reference = fixed_sector_operator(operator, n, space)
    actual = bind_fixed_sector_operator(n, space, operator)(operator)
    for x, y in zip(actual, reference):
        torch.testing.assert_close(x, y)
    gradient, = torch.autograd.grad(sum(x.real.sum() for x in actual), values, retain_graph=True)
    expected, = torch.autograd.grad(sum(x.real.sum() for x in reference), values)
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('compiled', [False, True])
def test_single_pauli_basis_and_compiled_basis(order, compiled):
    basis = PauliPEPOBasis(1, 2, [('onsite', 'X', .2), ('edge', 'ZZ', .3)],
                           cluster_size=order, factorization='fixed')
    plan = prepare_cluster_channels(basis.compile_exp() if compiled else basis, -.11j)
    np.testing.assert_allclose(plan.exp(-.11j).to_dense(), basis.exp(-.11j).to_dense(), atol=3e-13)

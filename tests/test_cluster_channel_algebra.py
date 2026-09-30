"""Family-valid Pauli edge identities preserve PEPO values and derivatives."""

from contextlib import ExitStack
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    prepare_cluster_channels,
)
from pepsy.operators._cluster_channel_linear import delinearize_slices, transfer_matrix
from pepsy.operators._cluster_channel_pauli import PauliSpace
from pepsy.operators._cluster_channel_structure import _Linear
from test_cluster_channel_pepo import contract, geometry
from test_cluster_channels import arguments, dense, fd, loss, reference, source

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def test_exact_delinearisation_proves_nonidentical_columns_and_protects_rail():
    # The last active column is 1/3 of the first plus twice the second;
    # the protected rail deliberately duplicates an active column.
    atom = lambda i, c=1: _Linear(((i, Fraction(c)),))
    columns = [[atom(1), atom(2)], [atom(1), atom(2)], [atom(2), atom(1)],
               [atom(1, Fraction(1, 3))+atom(2, 2), atom(2, Fraction(1, 3))+atom(1, 2)]]
    blocks = {(i,): np.array(c, dtype=object).reshape(2, 1) for i, c in enumerate(columns)}
    select, transfer, width = delinearize_slices(blocks, 0, 4, 1024**2)
    assert width == 3
    assert select[0] == transfer[0] == {0: Fraction(1)}
    assert transfer[3] == {1: Fraction(1, 3), 2: Fraction(2)}
    values = {1: 2+.7j, 2: -.2+1.3j}
    evaluated = np.array([[sum(c*values[i] for i, c in entry.terms) for entry in col]
                          for col in columns], dtype=complex).T
    np.testing.assert_allclose(evaluated @ transfer_matrix(select, width, 1024**2)
                               @ transfer_matrix(transfer, width, 1024**2).T, evaluated)
    with pytest.raises(MemoryError, match='memory_budget'):
        delinearize_slices(blocks, 0, 4, 100)


@pytest.mark.parametrize('words', [(0, 3, 4, 7), tuple(range(16))])
def test_weyl_coordinates_and_projection_match_independent_matrix_basis(words):
    space = PauliSpace((0, 1), words)
    rng = np.random.default_rng(337)
    matrix = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    coordinates, project = space.bind(matrix)
    if space.full:
        np.testing.assert_array_equal(project(matrix), matrix)
        return
    basis = []
    for word in words:
        x, z = word & 3, word >> 2
        basis.append(np.array([[int(row == (col ^ x))*(-1)**((z & col).bit_count())
                                for col in range(4)] for row in range(4)]))
    coefficients = np.einsum('kij,ij->k', basis, matrix)/4
    np.testing.assert_allclose(coordinates(matrix), coefficients)
    np.testing.assert_allclose(project(matrix), np.einsum('k,kij->ij', coefficients, basis))
    np.testing.assert_allclose(project(project(matrix)), project(matrix))


@pytest.mark.parametrize('kind', ['graph', 'square'])
@pytest.mark.parametrize('case', ['branch', 'loop', 'crossing', 'higher_body'])
def test_algebraic_family_reduction_preserves_projected_residuals_and_adjoints(kind, case):
    torch = pytest.importorskip('torch')
    builder = geometry(kind, case)
    arguments = dict(step=-.1j, parameters={'a': 0., 'b': 0.})
    expanded = prepare_cluster_channels(builder, **arguments)
    reduced = prepare_cluster_channels(builder, **arguments, preparation='algebraic')
    rng = np.random.default_rng(419)
    residuals = reduced.pack_residuals(**arguments)
    inputs = tuple(torch.tensor(rng.normal(size=v.shape)+1j*rng.normal(size=v.shape),
                                dtype=torch.complex128, requires_grad=True) for v in residuals)
    spaces = reduced._binding.pepo.algebra.spaces
    projected = tuple(space.bind(v)[1](v) for space, v in zip(spaces, inputs))
    expected = contract(expanded, expanded.bind_assembler(inputs[0])(projected))
    actual = contract(reduced, reduced.bind_assembler(inputs[0])(inputs))
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-12)
    weight = torch.tensor(rng.normal(size=(16, 16))+1j*rng.normal(size=(16, 16)))
    a = torch.autograd.grad((actual*weight).real.sum(), inputs)
    b = torch.autograd.grad((expected*weight).real.sum(), inputs)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=3e-11, rtol=3e-12)
    params = {'a': .31, 'b': -.17}
    np.testing.assert_allclose(contract(reduced, reduced.arrays(-.1j, params)),
                               contract(expanded, expanded.arrays(-.1j, params)), atol=3e-13)
    if kind == 'square' and case == 'loop':
        assert reduced.report['bond_dimensions'] == (7, 7, 7, 7)
    if kind == 'square' and case == 'higher_body':
        assert reduced.report['bond_dimensions'] == (13, 1, 1, 5)


@pytest.mark.parametrize('kind', ['graph', 'square'])
def test_joint_ordered_exponentials_and_zero_derivatives_without_decompositions(kind):
    torch = pytest.importorskip('torch')
    builder = source(kind)
    zeros = np.array([0., 0., 0., .7])
    with ExitStack() as stack:
        for target in ('scipy.linalg.qr', 'scipy.linalg.svd', 'numpy.linalg.qr',
                       'numpy.linalg.svd', 'torch.linalg.svd', 'torch.linalg.qr',
                       'pepsy.operators.graph_pepo_product.GraphPEPOClusterProductExpansion.exp'):
            stack.enter_context(patch(target, side_effect=AssertionError('decomposition/expanded builder forbidden')))
        plan = prepare_cluster_channels(builder, **arguments(zeros), preparation='algebraic')
        for values in (zeros, np.array([.23, -.31, .17, .83])):
            theta = torch.tensor(values, requires_grad=True)
            actual = dense(plan, theta)
            np.testing.assert_allclose(actual.detach(), reference(values), atol=3e-13)
            gradient, = torch.autograd.grad(loss(actual), theta)
            np.testing.assert_allclose(gradient, fd(lambda v: loss(reference(v)), values), atol=3e-9)
            np.testing.assert_allclose(plan.trace_exp(**arguments(values)), np.trace(reference(values)), atol=3e-13)
    assert plan.report['operator_basis'] == 'pauli-closure'
    assert plan.report['expanded_reference_builds'] == 0
    assert plan.report['structural_reuse']['graph']['reduction'] == 'rational-delinearisation'


def test_algebraic_rejects_nonpauli_inventory_and_unsupported_sources():
    with pytest.raises(ValueError, match='PEPO builders only'):
        prepare_cluster_channels(source('mpo'), preparation='algebraic')
    for operator in (np.diag([1., 2.]), np.eye(3)):
        term = MPOProductTerm((0,), (operator,), coefficient=MPOParameter('a'))
        geometry = ClusterPlan.from_terms([term], sites=1, cluster_size=1)
        builder = PEPOClusterProductExpansion.from_plan(geometry, [[term]], layout='graph', factorization='fixed')
        with pytest.raises(ValueError, match='Pauli|proportional'):
            prepare_cluster_channels(builder, -.1j, {'a': .2}, preparation='algebraic')


def test_algebraic_rejects_operator_family_changes_and_allows_opt_out():
    builder = source('square')
    options = arguments([.2, -.3, .1, .7])
    plan = prepare_cluster_channels(builder, **options, preparation='algebraic', structural_reuse=False)
    assert plan.report['structural_reuse']['removed_channels'] == 0
    np.testing.assert_allclose(dense(plan, [.1, .2, .3, .8]), reference([.1, .2, .3, .8]), atol=3e-13)
    resolved, _ = plan._binding.resolve(options)
    graph = getattr(resolved, '_graph', resolved)
    operator = graph.factors[0].terms[0].operators[0]
    original = operator.copy()
    try:
        operator[:] = np.diag([1., -1.])
        with pytest.raises(ValueError, match='inventory changed'):
            plan.arrays(**options)
    finally:
        operator[:] = original

"""Symbolic transition sharing preserves independent residual families."""

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOClusterProductExpansion, MPOParameter, MPOProductTerm,
    prepare_cluster_channels,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def builder(native=False, crossing=False):
    edges = ((0, 4), (1, 3)) if crossing else tuple((i, i+1) for i in range(4))
    factors = [[MPOProductTerm.from_pauli(edge, 'ZZ', coefficient=MPOParameter('a')) for edge in edges],
               [MPOProductTerm.from_pauli((i,), 'Z' if native else 'X', coefficient=MPOParameter('b'))
                for i in range(5)]]
    options = dict(factorization='fixed', cutoff=0.)
    if native:
        options.update(symmetry='U1', physical_charges=(0, 1))
    if crossing:
        geometry = ClusterPlan.from_terms([t for f in factors for t in f], sites=5, cluster_size=3)
        return MPOClusterProductExpansion.from_plan(geometry, factors, graph_assembly='exact', **options)
    return MPOClusterProductExpansion(5, factors, cluster_size=3, **options)


def contract(arrays):
    import torch

    result = arrays[0][0]
    for array in arrays[1:]:
        size = result.shape[-1]*array.shape[-1]
        result = torch.einsum('aij,abkl->bikjl', result, array).reshape(array.shape[1], size, size)
    return result[0]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('crossing', [False, True])
def test_symbolic_quotient_preserves_arbitrary_residual_values_and_adjoints(native, crossing):
    torch = pytest.importorskip('torch')
    source = builder(native, crossing)
    shared = prepare_cluster_channels(source, -.1j, {'a': .3, 'b': .3})
    expanded = prepare_cluster_channels(source, -.1j, {'a': .3, 'b': .3}, structural_reuse=False)
    assert shared.report['structural_reuse']['removed_channels'] > 0
    assert sum(shared.report['bond_dimensions']) < sum(expanded.report['bond_dimensions'])
    # Exercise every residual entry independently, rather than only values
    # reachable by this small Hamiltonian's two parameters. Native splits
    # restrict both paths to the same declared charge sectors.
    rng = np.random.default_rng(1234)
    shapes = shared.pack_residuals(-.1j, {'a': .2, 'b': -.1})
    inputs = tuple(torch.tensor(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape),
                                dtype=torch.complex128, requires_grad=True) for x in shapes)
    weight = torch.tensor(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32)))
    expected = contract(expanded.bind_assembler(inputs[0])(inputs))
    actual = contract(shared.bind_assembler(inputs[0])(inputs))
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    gradients = torch.autograd.grad((actual*weight).real.sum(), inputs)
    reference = torch.autograd.grad((expected*weight).real.sum(), inputs)
    for a, b in zip(gradients, reference):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)


def test_structural_maps_are_independent_of_equal_or_zero_reference_values():
    source = builder()
    plans = [prepare_cluster_channels(source, -.1j, parameters) for parameters in
             ({'a': .3, 'b': .3}, {'a': 0., 'b': 0.}, {'a': -.4, 'b': .2})]
    for plan in plans[1:]:
        assert plan.report['structural_reuse'] == plans[0].report['structural_reuse']
        for a, b in zip(plan.bases+plan._dual_bases, plans[0].bases+plans[0]._dual_bases):
            np.testing.assert_array_equal(a, b)
    # Sum/select maps are distinct: using an orthogonal projector here would
    # alter the multiplicity of histories merged at only one endpoint.
    assert any(not np.array_equal(a, b) for a, b in zip(plans[0].bases, plans[0]._dual_bases))


@pytest.mark.parametrize('values', [[0., 0.], [.31, -.19]])
def test_chi_is_independent_of_order_and_uncapped_limit_preserves_live_gradients(values):
    torch = pytest.importorskip('torch')
    source = builder()
    reference = {'a': .3, 'b': .2}
    exact = prepare_cluster_channels(source, -.1j, reference, structural_reuse=False)
    original_order = source.cluster_size
    target_parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    target = exact.exp(-.1j, dict(zip('ab', target_parameters))).to_dense()
    target_gradient, = torch.autograd.grad((target.real+.23*target.imag).sum(), target_parameters)
    errors = []
    for cap in (1, 3, 6, 9, None):
        plan = prepare_cluster_channels(source, -.1j, reference, max_bond=cap)
        assert source.cluster_size == original_order
        if cap is not None:
            assert max(plan.report['bond_dimensions']) <= cap
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        matrix = plan.exp(-.1j, dict(zip('ab', theta))).to_dense()
        gradient, = torch.autograd.grad((matrix.real+.23*matrix.imag).sum(), theta)
        errors.append(float(torch.linalg.norm(matrix.detach()-target.detach())))
        if cap is None or cap == 9:
            torch.testing.assert_close(matrix, target, atol=3e-13, rtol=3e-13)
            torch.testing.assert_close(gradient, target_gradient, atol=3e-12, rtol=3e-12)
    if any(values):
        assert errors[0] > 1e-3 and errors[-1] < 1e-12


def test_structural_reuse_option_rejects_non_boolean_values():
    with pytest.raises(TypeError, match='structural_reuse must be a bool'):
        prepare_cluster_channels(builder(), structural_reuse=1)


@pytest.mark.parametrize('symmetry,charges', [
    ('U1', (0, 0, 1)), ('Z2', (0, 0, 1)),
    ('U1U1', ((0, 0), (0, 1), (1, 0), (1, 1))),
    ('Z2Z2', ((0, 0), (0, 1), (1, 0), (1, 1))),
])
@pytest.mark.parametrize('preparation', ['reference', 'frontier'])
def test_symbolic_sharing_preserves_product_and_repeated_native_charge_groups(symmetry, charges, preparation):
    torch = pytest.importorskip('torch')
    from test_native_cluster_mpo import hopping, reference

    d = len(charges)
    up = np.zeros((d, d))
    up[0, -1] = 1.
    other = np.zeros((d, d))
    other[0, 1] = 1.

    def factors(a, b):
        return [hopping(0, 1, up, a), hopping(1, 2, other, b)]

    source = MPOClusterProductExpansion(3, factors(MPOParameter('a'), MPOParameter('b')),
        cluster_size=3, factorization='fixed', cutoff=0., symmetry=symmetry, physical_charges=charges)
    plan = prepare_cluster_channels(source, -.17j, {'a': .3, 'b': -.2}, preparation=preparation)
    assert plan.report['structural_reuse']['removed_channels'] > 0
    for values in ([.31, -.19], [0., 0.]):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        native = plan.exp(-.17j, dict(zip('ab', theta)))
        assert all(hasattr(t.data, 'blocks') for t in native)
        matrix = native.to_dense()
        np.testing.assert_allclose(matrix.detach(), reference(factors(*values), 3, -.17j), atol=3e-13)
        gradient, = torch.autograd.grad((matrix.real+.23*matrix.imag).sum(), theta)
        eps = 1e-5
        def loss(v):
            m = reference(factors(*v), 3, -.17j)
            return (m.real+.23*m.imag).sum()
        expected = [(loss(values+eps*e)-loss(values-eps*e))/(2*eps) for e in np.eye(2)]
        np.testing.assert_allclose(gradient, expected, atol=3e-9)

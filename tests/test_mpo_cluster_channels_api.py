"""Integrated channel construction and numerical QR preserve cluster targets."""

from contextlib import ExitStack
from functools import reduce
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    MPOClusterProductExpansion, MPOParameter, MPOProductTerm,
    exp_mpo_cluster, exp_mpo_cluster_product, prepare_cluster_channels,
)
from test_cluster_channels import source

pytestmark = [pytest.mark.integration, pytest.mark.operators, pytest.mark.mpo]

X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])


def embed(n, operators):
    return reduce(np.kron, (operators.get(i, np.eye(2)) for i in range(n)))


def forbid_expanded_assembly_and_svd():
    stack = ExitStack()
    for target in ('numpy.linalg.svd', 'scipy.linalg.svd', 'numpy.linalg.lstsq'):
        stack.enter_context(patch(target, side_effect=AssertionError('SVD forbidden')))
    stack.enter_context(patch.object(MPOClusterProductExpansion, 'exp',
                                    side_effect=AssertionError('expanded assembly forbidden')))
    return stack


@pytest.mark.parametrize('preparation', ['frontier', 'automaton'])
@pytest.mark.parametrize('edges', [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))])
def test_facade_retains_disjoint_crossing_and_nested_cluster_products(preparation, edges):
    # Disconnected two-site components make p=2 equal the full exponential.
    terms = [(('ZZ', .7), edge) for edge in edges]
    terms += [(('X', .3), i) for i in range(4)]
    h = sum(.7*embed(4, dict.fromkeys(edge, Z)) for edge in edges)
    h += sum(.3*embed(4, {i: X}) for i in range(4))
    with forbid_expanded_assembly_and_svd():
        result, report = exp_mpo_cluster(
            terms, -.08j, shape=4, graph=(range(4), edges), cluster_size=2,
            factorization='fixed', cutoff=0., graph_assembly='exact', collection_budget=1,
            preparation=preparation, delinearize=True, return_report=True,
        )
    np.testing.assert_allclose(result.to_dense(), expm(-.08j*h), atol=2e-13)
    assert report['full_collections_enumerated'] == 0
    assert report['rank_selection'] == 'evaluation-qr'
    assert report['bond_dimensions'] == tuple(result.bond_sizes())
    assert report['delinearization'].initial_bond_dimensions == report['channel_bond_dimensions']
    assert report['output_dense_entries'] == sum(a.size for a in result.arrays)
    assert result.pepsy_cluster_report is report
    assert result.pepsy_channel_report is report
    assert result.pepsy_first_degree is None


@pytest.mark.parametrize('periodic', [False, True])
def test_product_facade_preserves_order_and_runtime_coefficients(periodic):
    n = 4
    edges = [(i, i+1) for i in range(n-1)] + ([(0, n-1)] if periodic else [])
    factors = [[(('ZZ', 99.), edge) for edge in edges], [(('X', 99.), i) for i in range(n)]]
    coefficients = [np.linspace(.2, .5, len(edges)), np.linspace(-.3, .2, n)]
    hz = sum(c*embed(n, dict.fromkeys(edge, Z)) for c, edge in zip(coefficients[0], edges))
    hx = sum(c*embed(n, {i: X}) for i, c in enumerate(coefficients[1]))
    with forbid_expanded_assembly_and_svd():
        result = exp_mpo_cluster_product(
            factors, dt=-.12j, coefficients=coefficients, shape=n, graph='chain',
            cyclic=periodic, cluster_size=n, factorization='fixed', cutoff=0.,
            graph_assembly='exact', preparation='automaton', delinearize=True,
            delinearize_opts={'rtol': 2e-12, 'preserve_zeros': False, 'max_sweeps': 3},
        )
    expected = expm(-.12j*hz) @ expm(-.12j*hx)
    np.testing.assert_allclose(result.to_dense(), expected, atol=3e-13)
    assert np.linalg.norm(expected-expm(-.12j*hx) @ expm(-.12j*hz)) > 1e-4
    assert result.pepsy_delinearization_report.rtol == 2e-12
    assert not result.pepsy_delinearization_report.preserve_zeros


def test_plan_reduces_fresh_evaluations_without_mutating_static_maps():
    terms = [MPOProductTerm.from_pauli((i, i+1), 'ZZ', coefficient=MPOParameter('a'))
             for i in range(3)]
    terms += [MPOProductTerm.from_pauli((i,), 'X', coefficient=.3) for i in range(4)]
    builder = MPOClusterProductExpansion(4, [terms], cluster_size=4,
                                         factorization='fixed', cutoff=0.)
    plan = prepare_cluster_channels(builder, -.1j, {'a': .7}, preparation='automaton')
    static = plan.report
    assert static['method'] == 'fixed-pauli-automaton'
    assert not static['projection_applied']
    original = tuple(a.copy() for a in plan.bases)
    for coupling in (.7, 0., -.4):
        with forbid_expanded_assembly_and_svd():
            result, report = plan.exp(-.1j, {'a': coupling}, delinearize=True, return_report=True)
        h = sum(coupling*embed(4, {i: Z, i+1: Z}) for i in range(3))
        h += sum(.3*embed(4, {i: X}) for i in range(4))
        np.testing.assert_allclose(result.to_dense(), expm(-.1j*h), atol=3e-13)
        np.testing.assert_allclose(plan.trace_exp(-.1j, {'a': coupling}, delinearize=True,
                                                 normalized=True),
                                   np.trace(result.to_dense())/16, atol=3e-13)
        assert report['channel_bond_dimensions'] == static['bond_dimensions']
        assert report['method'] == 'fixed-pauli-automaton+delinearisation-qr'
        assert all(a <= b for a, b in zip(report['bond_dimensions'], static['bond_dimensions']))
        assert plan.report == static
    for a, b in zip(original, plan.bases):
        np.testing.assert_array_equal(a, b)


def test_partial_cluster_cutoff_matches_independent_partition_sum():
    fields = [.2, -.3, .4]
    singles = [expm(-.1j*h*X) for h in fields]
    pair01 = expm(-.1j*(.7*np.kron(Z, Z) + fields[0]*np.kron(X, np.eye(2))
                       + fields[1]*np.kron(np.eye(2), X)))
    pair12 = expm(-.1j*(.7*np.kron(Z, Z) + fields[1]*np.kron(X, np.eye(2))
                       + fields[2]*np.kron(np.eye(2), X)))
    # Three partitions: singles, B01 tensor B2, B0 tensor B12.
    reference = (np.kron(pair01, singles[2]) + np.kron(singles[0], pair12)
                 - reduce(np.kron, singles))
    terms = [(('ZZ', .7), (0, 1)), (('ZZ', .7), (1, 2))]
    terms += [(('X', h), i) for i, h in enumerate(fields)]
    result = exp_mpo_cluster(terms, -.1j, shape=3, cluster_size=2,
                             factorization='fixed', cutoff=0.,
                             preparation='automaton', delinearize=True)
    np.testing.assert_allclose(result.to_dense(), reference, atol=3e-13)
    h = .7*(embed(3, {0: Z, 1: Z}) + embed(3, {1: Z, 2: Z}))
    h += sum(field*embed(3, {i: X}) for i, field in enumerate(fields))
    assert np.linalg.norm(reference-expm(-.1j*h)) > 1e-5


def test_single_facade_coordinate_mapping_and_parameter_bindings():
    terms = [(('ZZ', MPOParameter('g')), ((0, 0), (1, 0))),
             (('X', .3), (0, 1))]
    options = dict(shape=(2, 2), graph='square', cluster_size=2,
                   factorization='fixed', cutoff=0., graph_assembly='exact')
    expected = exp_mpo_cluster(terms, -.1j, parameters={'g': .7}, **options).to_dense()
    with forbid_expanded_assembly_and_svd():
        actual = exp_mpo_cluster(terms, -.1j, parameters={'g': .7},
                                 preparation='automaton', delinearize=True, **options)
    np.testing.assert_allclose(actual.to_dense(), expected, atol=3e-13)
    # Single-factor coefficient vectors override the parsed term coefficients.
    override = exp_mpo_cluster(terms, -.1j, coefficients=[.7, .3],
                              preparation='automaton', delinearize=True, **options)
    np.testing.assert_allclose(override.to_dense(), expected, atol=3e-13)


@pytest.mark.parametrize('options, error, match', [
    ({'preparation': 'unknown'}, ValueError, 'preparation'),
    ({'preparation': None, 'delinearize': True}, ValueError, 'requires preparation'),
    ({'factorization': 'auto'}, ValueError, 'factorization'),
    ({'return_semantic': True}, ValueError, 'Quimb MPO'),
    ({'graph_assembly': 'bounded'}, ValueError, 'exact'),
    ({'graph_assembly': 'auto'}, ValueError, 'exact'),
    ({'delinearize': 1}, TypeError, 'boolean'),
    ({'delinearize_opts': {}}, ValueError, 'requires delinearize'),
    ({'delinearize': True, 'delinearize_opts': []}, TypeError, 'mapping'),
    ({'delinearize': True, 'delinearize_opts': {'return_report': False}}, ValueError, 'accepts only'),
])
def test_facade_rejects_incompatible_or_ignored_options(options, error, match):
    base = dict(shape=2, graph='chain', graph_assembly='exact', factorization='fixed',
                cutoff=0., preparation='automaton')
    with pytest.raises(error, match=match):
        exp_mpo_cluster([(('ZZ', .2), (0, 1))], -.1j, **(base | options))


@pytest.mark.parametrize('kind', ['native', 'graph', 'square'])
def test_plan_rejects_non_dense_mpo_delinearisation(kind):
    if kind == 'native':
        pytest.importorskip('symmray')
    plan = prepare_cluster_channels(source(kind), -.1j, dict(a=.2, b=.3, c=.4))
    with pytest.raises(NotImplementedError, match='dense NumPy MPO'):
        plan.exp(-.1j, dict(a=.2, b=.3, c=.4), delinearize=True)


def test_torch_qr_rejects_without_detaching_and_exact_channel_path_keeps_gradients():
    torch = pytest.importorskip('torch')
    plan = prepare_cluster_channels(source('mpo'), -.1j, dict(a=.2, b=.3, c=.4),
                                    preparation='automaton')
    parameter = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    args = dict(step=-.1j, parameters=dict(a=parameter, b=.3, c=.4))
    with pytest.raises(TypeError, match='autodiff'):
        plan.exp(**args, delinearize=True)
    result = plan.exp(**args)
    gradient, = torch.autograd.grad(result.to_dense().real.sum(), parameter)
    assert torch.isfinite(gradient)


def test_frontier_facade_keeps_native_sectors_without_qr():
    pytest.importorskip('symmray')
    terms = [(('ZZ', .7), (0, 1)), (('Z', .3), 0)]
    options = dict(shape=2, cluster_size=2, factorization='fixed', cutoff=0.,
                   symmetry='U1', physical_charges=(0, 1))
    expected = exp_mpo_cluster(terms, -.1j, **options).to_dense()
    actual = exp_mpo_cluster(terms, -.1j, preparation='frontier', **options)
    np.testing.assert_allclose(actual.to_dense(), expected, atol=3e-13)
    assert all(hasattr(a, 'blocks') for a in actual.arrays)
    with pytest.raises(NotImplementedError, match='dense NumPy MPO'):
        exp_mpo_cluster(terms, -.1j, preparation='frontier', delinearize=True, **options)
    with pytest.raises(ValueError, match='dense qubit Pauli'):
        exp_mpo_cluster(terms, -.1j, preparation='automaton', **options)

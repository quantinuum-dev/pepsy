"""Compact channel replay preserves the chosen projected operator and gradient."""

from functools import reduce
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterChannelPlan, ClusterPlan, MPOClusterProductExpansion, MPOParameter,
    MPOProductTerm, PEPOClusterProductExpansion, prepare_cluster_channels,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])


def source(kind, dtype=None):
    factors = [[MPOProductTerm.from_pauli(edge, word, coefficient=MPOParameter(name))
                for word in ('XX', 'YY')] for edge, name in (((0, 1), 'a'), ((1, 2), 'b'))]
    factors.append([MPOProductTerm.from_pauli((i,), 'Z', coefficient=MPOParameter('c')) for i in range(3)])
    if dtype is not None:
        factors = [[MPOProductTerm(t.sites, tuple(o.astype(dtype) for o in t.operators), t.coefficient)
                    for t in factor] for factor in factors]
    if kind in ('mpo', 'native'):
        return MPOClusterProductExpansion(3, factors, cluster_size=3, factorization='fixed', cutoff=0.,
            **(dict(symmetry='U1', physical_charges=(0, 1)) if kind == 'native' else {}))
    plan = ClusterPlan.from_terms([t for f in factors for t in f], sites=3, shape=(1, 3), cluster_size=3)
    return PEPOClusterProductExpansion.from_plan(plan, factors, layout=kind, factorization='fixed')


def arguments(values):
    return dict(step=-.2j*values[3], parameters=dict(zip('abc', values)))


def dense(plan, values):
    result = plan.exp(**arguments(values))
    if plan.report['representation'] == 'graph':
        return result.to_dense([('graph-bra', s) for s in range(3)], [('graph-ket', s) for s in range(3)])
    return result.to_dense()


def reference(values):
    a, b, c, t = values
    def embed(pair):
        return reduce(np.kron, (pair.get(i, np.eye(2)) for i in range(3)))
    h = [a*(embed({0: X, 1: X})+embed({0: Y, 1: Y})),
         b*(embed({1: X, 2: X})+embed({1: Y, 2: Y})), c*sum(embed({i: Z}) for i in range(3))]
    return reduce(np.matmul, (expm(-.2j*t*matrix) for matrix in h))


def loss(matrix):
    return (matrix.real+.37*matrix.imag).sum()


def fd(function, values):
    eps = 1e-5
    return np.array([(function(values+eps*e)-function(values-eps*e))/(2*eps) for e in np.eye(4)])


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
def test_exact_channels_preserve_joint_operator_and_all_derivatives_without_svd(kind):
    torch = pytest.importorskip('torch')
    if kind == 'native':
        pytest.importorskip('symmray')
    builder = source(kind)
    with patch('numpy.linalg.svd', side_effect=AssertionError('SVD forbidden')), \
            patch('scipy.linalg.svd', side_effect=AssertionError('SVD forbidden')):
        plan = prepare_cluster_channels(builder, **arguments([.2, -.3, .1, .7]))
        assert isinstance(plan, ClusterChannelPlan)
        for values in (np.array([.2, -.3, .1, .7]), np.array([0., 0., 0., .7])):
            theta = torch.tensor(values, requires_grad=True)
            actual = dense(plan, theta)
            np.testing.assert_allclose(actual.detach(), reference(values), atol=3e-13)
            gradient, = torch.autograd.grad(loss(actual), theta)
            np.testing.assert_allclose(gradient, fd(lambda v: loss(reference(v)), values), atol=3e-9)
            np.testing.assert_allclose(plan.trace_exp(**arguments(values)), np.trace(reference(values)), atol=3e-13)


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_capped_channels_match_their_actual_objective_gradient_and_bound_storage(kind, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    builder = source(kind)
    values = np.array([.2, -.3, .1, .7])
    plan = prepare_cluster_channels(builder, **arguments(values), max_bond=3,
                                    samples=[arguments([-.1, .4, -.2, .6])])
    assert max(plan.report['bond_dimensions']) <= 3
    assert plan.report['output_dense_entries'] < plan.report['input_dense_entries']
    assert all(not matrix.flags.writeable for matrix in plan.bases)
    # A cap is a measured approximation, not a claim of exact cancellation.
    assert np.linalg.norm(dense(plan, values)-reference(values)) > 1e-8
    with patch('scipy.linalg.qr', side_effect=AssertionError('QR in replay')), \
            patch('numpy.linalg.svd', side_effect=AssertionError('SVD in replay')):
        for v in (values, np.array([0., 0., 0., .7])):
            theta = torch.tensor(v, device=device, requires_grad=True)
            result = dense(plan, theta)
            assert result.device == theta.device and result.dtype == torch.complex128
            np.testing.assert_allclose(result.detach().cpu(), dense(plan, v), atol=3e-13)
            gradient, = torch.autograd.grad(loss(result), theta)
            np.testing.assert_allclose(gradient.detach().cpu(), fd(lambda x: loss(dense(plan, x)), v), atol=3e-9)
            np.testing.assert_allclose(plan.trace_exp(**arguments(v)), np.trace(dense(plan, v)), atol=3e-13)


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
def test_jax_compact_arrays_and_measurement_compile(kind):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    plan = prepare_cluster_channels(source(kind), **arguments([.2, -.3, .1, .7]), max_bond=3)
    with jax.enable_x64(True):
        evaluate = jax.jit(jax.value_and_grad(lambda theta: plan.trace_exp(**arguments(theta)).real))
        for values in (np.array([.2, -.3, .1, .7]), np.array([0., 0., 0., .7])):
            value, gradient = evaluate(jnp.asarray(values))
            expected = lambda v: plan.trace_exp(**arguments(v)).real
            np.testing.assert_allclose(value, expected(values), atol=3e-13)
            np.testing.assert_allclose(gradient, fd(expected, values), atol=3e-9)


def test_array_only_projection_fullgraph_torch_compile_and_sparse_mpo_capture():
    torch = pytest.importorskip('torch')

    plan = prepare_cluster_channels(source('mpo'), **arguments([.2, -.3, .1, .7]), max_bond=3)
    with patch('pepsy.operators.mpo_semantic._sparse_virtual_to_dense',
               side_effect=AssertionError('expanded dense allocation')):
        packed = plan.pack(**arguments(torch.tensor([.2, -.3, .1, .7], dtype=torch.float64)))
    project = plan.bind_projector(packed[0])
    compiled = torch.compile(project, backend='aot_eager', fullgraph=True)
    inputs = tuple(v.detach().requires_grad_(True) for v in packed)
    expected = project(inputs)
    actual = compiled(inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    grad = torch.autograd.grad(sum(loss(v) for v in actual), inputs)
    reference_grad = torch.autograd.grad(sum(loss(v) for v in expected), inputs)
    for a, b in zip(grad, reference_grad):
        torch.testing.assert_close(a, b)


def test_projection_validation_and_tangent_snapshots():
    builder = source('mpo')
    values = np.array([.2, -.3, .1, .7])
    eps = 1e-4
    pairs = [(arguments(values+eps*e), arguments(values-eps*e), 2*eps) for e in np.eye(4)]
    plan = prepare_cluster_channels(builder, **arguments(values), max_bond=5, tangent_pairs=pairs)
    assert plan.report['replay_factorizations'] == 0
    assert np.isfinite(plan.reference_local_defects).all()
    for chi in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='max_bond'):
            prepare_cluster_channels(builder, **arguments(values), max_bond=chi)
    with pytest.raises(MemoryError, match='memory_budget'):
        prepare_cluster_channels(builder, **arguments(values), max_bond=3, memory_budget=1)
    with pytest.raises(ValueError, match='tangent denominator'):
        prepare_cluster_channels(builder, **arguments(values), tangent_pairs=[({}, {}, 0.)])
    other = MPOClusterProductExpansion(2, [[MPOProductTerm.from_pauli((0, 1), 'XX')]])
    with pytest.raises(ValueError, match='fixed direct'):
        prepare_cluster_channels(other)


def test_graph_materialization_relabels_each_edge_without_numerical_truncation():
    graph = source('graph').exp(**arguments([.2, -.3, .1, .7]))
    network = graph.to_tensor_network(remove_orphans=False)
    old = graph.to_tensor_network(remove_orphans=False, compact_bonds=False)
    assert graph.dense_nbytes == sum(t.data.nbytes for t in network)
    assert graph.dense_nbytes < sum(t.data.nbytes for t in old)
    assert max(graph.bond_dimensions.values()) < graph.bond_dim
    bra, ket = [('graph-bra', s) for s in range(3)], [('graph-ket', s) for s in range(3)]
    np.testing.assert_allclose(network.to_dense(bra, ket), old.to_dense(bra, ket), atol=3e-13)


@pytest.mark.parametrize('kind', ['mpo', 'native', 'graph', 'square'])
def test_compact_single_precision_and_parameter_graph_is_not_retained(kind):
    torch = pytest.importorskip('torch')
    initial = torch.tensor([.2, -.3, .1, .7], dtype=torch.float32, requires_grad=True)
    plan = prepare_cluster_channels(source(kind, dtype='complex64'), **arguments(initial), max_bond=3)
    theta = initial.detach().clone().requires_grad_(True)
    actual = dense(plan, theta)
    assert actual.dtype == torch.complex64
    gradient, previous = torch.autograd.grad(loss(actual), (theta, initial), allow_unused=True)
    assert previous is None and torch.isfinite(gradient).all()
    packed = plan.pack(**arguments(theta))
    numpy_inputs = tuple(v.detach().numpy() for v in packed)
    numpy_arrays = plan.bind_projector(numpy_inputs[0])(numpy_inputs)
    assert all(value.dtype == np.complex64 for value in numpy_arrays)


@pytest.mark.parametrize('layout', ['graph', 'square'])
def test_compact_crossing_collections_and_short_periodic_axes(layout):
    factors = [[MPOProductTerm.from_pauli((0, 3), 'XY', coefficient=MPOParameter('a'))],
               [MPOProductTerm.from_pauli((1, 2), 'ZZ', coefficient=MPOParameter('b'))]]
    graph = ClusterPlan.from_terms([t for f in factors for t in f], sites=4, shape=(2, 2),
                                  cyclic=True, cluster_size=2)
    builder = PEPOClusterProductExpansion.from_plan(graph, factors, layout=layout, factorization='fixed')
    plan = prepare_cluster_channels(builder, -.1j, {'a': .2, 'b': .3})
    for params in ({'a': 0., 'b': 0.}, {'a': -.4, 'b': .1}):
        expected = builder.exp(-.1j, params, materialize=True)
        actual = plan.exp(-.1j, params)
        if layout == 'graph':
            axes = ([('graph-bra', i) for i in range(4)], [('graph-ket', i) for i in range(4)])
            actual, expected = actual.to_dense(*axes), expected.to_dense(*axes)
        else:
            actual, expected = actual.to_dense(), expected.to_dense()
        np.testing.assert_allclose(actual, expected, atol=3e-13)


@pytest.mark.parametrize("preparation, method", [
    ("symbolic", "exact-symbolic-channels"),
    ("algebraic", "fixed-pauli-algebra"),
])
@pytest.mark.parametrize("max_bond", [None, 100])
def test_exact_symbolic_preparation_skips_unused_numerical_references(
    preparation, method, max_bond,
):
    from pepsy.operators._cluster_channel_assembly import ChannelSource

    values = np.array([.2, -.3, .1, .7])
    samples = [arguments(-values)]
    tangents = [(arguments(values+.01), arguments(values-.01), .02)]
    with patch.object(ChannelSource, "residuals",
                      side_effect=AssertionError("unused reference evaluation")):
        plan = prepare_cluster_channels(
            source("square"), **arguments(values), preparation=preparation,
            max_bond=max_bond, samples=samples, tangent_pairs=tangents,
        )
    report = plan.report
    assert report["method"] == method
    assert report["reference_evaluations"] == 0
    assert report["projection_applied"] is False
    assert report["rank_selection"] == "none"
    np.testing.assert_allclose(dense(plan, values), reference(values), atol=3e-13)


@pytest.mark.parametrize(("preparation", "method"), [
    ("reference", "exact-reference-channels"),
    ("frontier", "exact-frontier-channels"),
    ("automaton", "fixed-pauli-automaton"),
])
@pytest.mark.parametrize("max_bond", [None, 100])
def test_exact_mpo_preparation_skips_unused_optional_references(
    preparation, method, max_bond,
):
    from pepsy.operators._cluster_channel_assembly import ChannelSource

    values = np.array([.2, -.3, .1, .7])
    samples = [arguments(-values)]
    tangents = [(arguments(values+.01), arguments(values-.01), .02)]
    original = ChannelSource.evaluate

    def tracked(channel_source, reference_arguments):
        return original(channel_source, reference_arguments)

    with patch.object(ChannelSource, "evaluate", autospec=True, side_effect=tracked) as evaluate:
        plan = prepare_cluster_channels(
            source("mpo"), **arguments(values), preparation=preparation,
            max_bond=max_bond, samples=samples, tangent_pairs=tangents,
        )
    assert evaluate.call_count == 1
    assert plan.report["method"] == method
    assert plan.report["reference_evaluations"] == 1
    assert plan.report["projection_applied"] is False
    np.testing.assert_allclose(dense(plan, values), reference(values), atol=3e-13)


def test_channel_report_distinguishes_exact_and_projected_preparation():
    values = np.array([.2, -.3, .1, .7])
    exact = prepare_cluster_channels(source("square"), **arguments(values))
    projected = prepare_cluster_channels(
        source("square"), **arguments(values), preparation="algebraic", max_bond=1)
    assert exact.report["method"] == "exact-reference-channels"
    assert exact.report["reference_evaluations"] == 1
    assert exact.report["projection_applied"] is False
    assert projected.report["method"] == "frozen-channel-qr"
    assert projected.report["reference_evaluations"] == 1
    assert projected.report["projection_applied"] is True
    assert projected.report["rank_selection"] == "reference-qr"


def test_exact_symbolic_preparation_still_validates_reference_arguments():
    values = np.array([.2, -.3, .1, .7])
    with pytest.raises(ValueError, match="samples accept only"):
        prepare_cluster_channels(
            source("square"), **arguments(values), preparation="symbolic",
            samples=[{"unknown": values}],
        )
    with pytest.raises(ValueError, match="tangent denominator"):
        prepare_cluster_channels(
            source("square"), **arguments(values), preparation="algebraic",
            tangent_pairs=[({}, {}, 0.)],
        )


def test_channel_trace_options_are_explicit():
    values = np.array([.2, -.3, .1, .7])
    square = prepare_cluster_channels(
        source("square"), **arguments(values), preparation="algebraic")
    np.testing.assert_allclose(
        square.trace_exp(**arguments(values), contract_opts={"optimize": "greedy"}),
        np.trace(reference(values)), atol=3e-13,
    )
    with pytest.raises(ValueError, match="state_budget.*cannot be combined"):
        square.trace_exp(**arguments(values), state_budget=1, contract_opts={})
    with pytest.raises(TypeError, match="contract_opts must be a mapping"):
        square.trace_exp(**arguments(values), contract_opts=[])
    mpo = prepare_cluster_channels(source("mpo"), **arguments(values))
    with patch.object(mpo, "exp", side_effect=AssertionError("unexpected construction")):
        with pytest.raises(ValueError, match="contract_opts requires"):
            mpo.trace_exp(**arguments(values), contract_opts={})

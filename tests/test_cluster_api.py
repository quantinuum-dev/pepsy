"""Shared cluster call conventions, checked against independent dense targets."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterExpansionPlan,
    ClusterLattice,
    GraphClusterExpansionPlan,
    MPOBasis,
    MPOClusterProductExpansion,
    MPOParameter,
    PEPOClusterProductExpansion,
    PauliPEPOBasis,
    build_cluster_expansion_pepo,
    build_graph_cluster_expansion_pepo,
    build_itf_cluster_expansion_pepo,
    build_model_cluster_expansion_pepo,
    build_real_time_cluster_expansion_pepo,
    exp_mpo_cluster_product,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)


def _dense(result):
    if hasattr(result, "to_tensor_network"):
        return result.to_dense()
    if hasattr(result, "to_mpo"):
        result = result.to_mpo()
    elif hasattr(result, "to_pepo"):
        result = result.to_pepo()
    return result.to_dense()


def _pepo_factories(**kwargs):
    return (
        PauliPEPOBasis(1, 2, [("onsite", "X")], **kwargs),
        ClusterExpansionPlan(1, 2, np.kron(Z, Z), X, **kwargs),
        GraphClusterExpansionPlan(ClusterLattice((0, 1), ((0, 1),)),
                                  np.kron(Z, Z), X, **kwargs),
    )


def test_spatial_cutoff_aliases_defaults_and_legacy_positional_plan():
    assert [p.cluster_size for p in _pepo_factories()] == [4, 3, 3]
    for kwargs in ({"cluster_size": 2}, {"order": 2}, {"order": 2, "cluster_size": 2}):
        assert [p.order for p in _pepo_factories(**kwargs)] == [2, 2, 2]
        assert [p.compile_exp().cluster_size for p in _pepo_factories(**kwargs)] == [2, 2, 2]
    legacy = ClusterExpansionPlan(1, 2, np.kron(Z, Z), X, 2, False)
    assert legacy.cluster_size == legacy.order == 2


@pytest.mark.parametrize("kind", ["pauli", "square", "graph"])
@pytest.mark.parametrize("kwargs, error", [
    ({"order": 2, "cluster_size": 3}, ValueError),
    ({"cluster_size": 2.5}, TypeError),
    ({"cluster_size": True}, TypeError),
])
def test_cutoff_alias_validation(kind, kwargs, error):
    with pytest.raises(error):
        if kind == "pauli":
            PauliPEPOBasis(1, 2, [("onsite", "X")], **kwargs)
        elif kind == "square":
            ClusterExpansionPlan(1, 2, np.kron(Z, Z), X, **kwargs)
        else:
            GraphClusterExpansionPlan(ClusterLattice((0, 1), ((0, 1),)),
                                      np.kron(Z, Z), X, **kwargs)


@pytest.mark.parametrize("graph", [False, True])
def test_dense_plan_exp_sign_and_old_build(graph):
    step = .023 - .11j
    edge, onsite = .3 * np.kron(Z, Z), .2 * X
    if graph:
        plan = GraphClusterExpansionPlan(ClusterLattice((0, 1), ((0, 1),)),
                                         edge, onsite, cluster_size=2)
    else:
        plan = ClusterExpansionPlan(1, 2, edge, onsite, cluster_size=2)
    expected = expm(step * (edge + np.kron(onsite, I) + np.kron(I, onsite)))
    assert plan.compile_exp() is plan
    np.testing.assert_allclose(_dense(plan.compile_exp()(step)), expected, atol=2e-12)
    np.testing.assert_allclose(_dense(plan.build(-step, materialize=False)), expected, atol=2e-12)
    operator, report = plan.exp(step, return_report=True)
    assert report.order == 2
    np.testing.assert_allclose(_dense(operator), expected, atol=2e-12)


def test_dense_convenience_builders_accept_cluster_size():
    step = -.03j
    edge = np.kron(Z, Z)
    expected = expm(step * (edge + np.kron(X, I) + np.kron(I, X)))
    results = (
        build_cluster_expansion_pepo(1, 2, -step, edge, X, cluster_size=2),
        build_graph_cluster_expansion_pepo(ClusterLattice((0, 1), ((0, 1),)),
                                           -step, edge, X, cluster_size=2, materialize=False),
        build_itf_cluster_expansion_pepo(1, 2, -step, cluster_size=2, symmetry=None),
        build_model_cluster_expansion_pepo(1, 2, -step,
                                          {"twosite_op": edge, "onesite_op": X},
                                          cluster_size=2),
        build_real_time_cluster_expansion_pepo(1, 2, .03, edge, X,
                                              cluster_size=2, fit_method=None),
    )
    for result in results:
        np.testing.assert_allclose(_dense(result), expected, atol=2e-12)


def test_common_product_factory_and_runtime_bindings():
    scales = (.7, -.4, 1.2)
    vectors = ([.3], [.5], [-.2])
    parameters = {"X": .3, "Y": .5, "Z": -.2}
    step = .03 - .17j
    mpo_bases = [MPOBasis.from_terms([((0,), (op,), MPOParameter(label))])
                 for label, op in zip("XYZ", (X, Y, Z))]
    pepo_bases = [PauliPEPOBasis(1, 1, [("onsite", label, MPOParameter(label))],
                                cluster_size=1) for label in "XYZ"]
    mpo = MPOClusterProductExpansion.from_bases(
        mpo_bases, coefficients=scales, cluster_size=1, cutoff=0., factorization="fixed",
    )
    pepo = PEPOClusterProductExpansion.from_bases(pepo_bases, coefficients=scales)
    expected = np.eye(2, dtype=complex)
    for op, scale, values in zip((X, Y, Z), scales, vectors):
        expected = expected @ expm(step * scale * values[0] * op)
    for expansion in (mpo, pepo):
        assert expansion.cluster_size == expansion.compile_exp().cluster_size == 1
        for evaluate in (expansion, expansion.compile_exp()):
            np.testing.assert_allclose(_dense(evaluate.exp(step, parameters)), expected, atol=2e-12)
            np.testing.assert_allclose(_dense(evaluate.exp(step, coefficients=vectors, materialize=False)), expected, atol=2e-12)
            for normalized in (False, True):
                expected_trace = np.trace(expected) / (2 if normalized else 1)
                np.testing.assert_allclose(evaluate.trace_exp(step, parameters, normalized=normalized),
                                           expected_trace, atol=2e-12)
                np.testing.assert_allclose(evaluate.trace_exp(step, coefficients=vectors, normalized=normalized),
                                           expected_trace, atol=2e-12)
    result, report = mpo.compile_exp().exp(step, coefficients=vectors, return_report=True)
    materialized, materialized_report = mpo.compile_exp().exp(
        step, coefficients=vectors, materialize=True, return_report=True,
    )
    np.testing.assert_allclose(materialized.to_dense(), expected, atol=2e-12)
    assert materialized_report is mpo.last_report
    assert report.factor_count == materialized_report.factor_count
    assert report.factor_count == 3
    assert mpo.cache_info["coefficient_topology_compiled"]
    oneshot = exp_mpo_cluster_product(
        [{"terms": basis.terms, "coefficient": scale}
         for basis, scale in zip(mpo_bases, scales)],
        step, coefficients=vectors, cluster_size=1, cutoff=0., factorization="fixed",
    )
    np.testing.assert_allclose(_dense(oneshot), expected, atol=2e-12)


@pytest.mark.parametrize("method", ["exp", "trace_exp"])
def test_mpo_coefficient_validation_and_scalar_vector(method):
    expansion = MPOClusterProductExpansion.from_local_terms(1, [((0,), (X,))], cluster_size=1)
    evaluate = getattr(expansion.compile_exp(), method)
    with pytest.raises(ValueError, match="either parameters or coefficients"):
        evaluate(.1, {}, coefficients=[.2])
    with pytest.raises(ValueError, match="length"):
        evaluate(.1, coefficients=[.2, .3])
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate(.1, coefficients=np.ones((1, 1)))
    with pytest.raises(TypeError, match="one-dimensional"):
        evaluate(.1, coefficients=.2)
    result = evaluate(.1, coefficients=np.array(.2))
    if method == "exp":
        np.testing.assert_allclose(_dense(result), expm(.02 * X), atol=2e-12)
    else:
        np.testing.assert_allclose(result, np.trace(expm(.02 * X)), atol=2e-12)


@pytest.mark.parametrize("graph", [False, True])
def test_mpo_independent_coefficient_gradients_and_fresh_calls(graph):
    torch = pytest.importorskip("torch")
    terms = [((0,), (X,)), ((1,), (X,)), ((0, 1), (Z, Z))]
    options = {"graph": ClusterLattice((0, 1), ((0, 1),)), "assembly": "recursive"} if graph else {}
    expansion = MPOClusterProductExpansion.from_local_terms(
        2, terms, cluster_size=2, factorization="fixed", cutoff=0., **options,
    )
    ops = [torch.tensor(op, dtype=torch.float64) for op in
           (np.kron(X, I), np.kron(I, X), np.kron(Z, Z))]
    # Equal initial onsite slots may share topology but must not share values.
    for values in ([.2, .2, .3], [.1, -.4, .6]):
        coefficients = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        step = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
        actual = expansion.compile_exp().trace_exp(step, coefficients=coefficients)
        reference = torch.trace(torch.matrix_exp(step * sum(c * op for c, op in zip(coefficients, ops))))
        gradients = torch.autograd.grad(actual, (coefficients, step), retain_graph=True)
        expected_gradients = torch.autograd.grad(reference, (coefficients, step))
        torch.testing.assert_close(actual, reference)
        for gradient, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(gradient, expected)
        operator = expansion.exp(step, coefficients=coefficients).to_mpo().to_dense()
        target = torch.matrix_exp(step * sum(c * op for c, op in zip(coefficients, ops)))
        torch.testing.assert_close(operator, target)
        loss = operator[0, 1] + 2 * operator[0, 2]
        expected_loss = target[0, 1] + 2 * target[0, 2]
        torch.testing.assert_close(torch.autograd.grad(loss, coefficients)[0],
                                   torch.autograd.grad(expected_loss, coefficients)[0])
    assert expansion.cache_info["coefficient_topology_compiled"]


def test_mpo_coefficient_bindings_under_jax_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jl = pytest.importorskip("jax.scipy.linalg")
    expansion = MPOClusterProductExpansion.from_local_terms(
        1, [((0,), (X,)), ((0,), (Z,))], cluster_size=1,
        factorization="fixed", cutoff=0.,
    ).compile_exp()
    with jax.enable_x64(True):
        actual = jax.jit(jax.value_and_grad(lambda c: expansion.trace_exp(.13, coefficients=c)))
        expected = jax.value_and_grad(lambda c: jnp.trace(jl.expm(.13 * (c[0] * X + c[1] * Z))))
        for values in ([.2, .2], [.3, -.5]):
            a, da = actual(jnp.array(values))
            b, db = expected(jnp.array(values))
            np.testing.assert_allclose(a, b, atol=2e-12)
            np.testing.assert_allclose(da, db, atol=2e-12)


def test_mpo_runtime_term_vectors_preserve_factor_scale_defaults():
    basis = MPOBasis.from_terms([((0,), (X,), .7)])
    expansion = MPOClusterProductExpansion.from_bases(
        [basis], coefficients=[MPOParameter("scale", default=.3)], cluster_size=1,
    )
    np.testing.assert_allclose(_dense(expansion.exp(.1, {"scale": .4})),
                               expm(.1 * .4 * .7 * X), atol=2e-12)
    np.testing.assert_allclose(_dense(expansion.exp(.1, coefficients=[.8])),
                               expm(.1 * .3 * .8 * X), atol=2e-12)
    required = MPOClusterProductExpansion.from_bases(
        [basis], coefficients=[MPOParameter("scale")], cluster_size=1,
    )
    with pytest.raises(ValueError, match="parameters"):
        required.exp(.1, coefficients=[.8])
    callback = MPOClusterProductExpansion.from_bases(
        [basis], coefficients=[lambda parameters: parameters["scale"]], cluster_size=1,
    )
    with pytest.raises(KeyError, match="require parameters"):
        callback.trace_exp(.1, coefficients=[.8])
    np.testing.assert_allclose(callback.trace_exp(.1, {"scale": .4}),
                               np.trace(expm(.1 * .4 * .7 * X)), atol=2e-12)


def test_mpo_product_coefficient_vectors_require_one_batch_per_factor():
    basis = MPOBasis.from_terms([((0,), (X,))])
    expansion = MPOClusterProductExpansion.from_bases([basis, basis], cluster_size=1)
    with pytest.raises(ValueError, match="one vector per factor"):
        expansion.exp(.1, coefficients=[[.2]])
    with pytest.raises(TypeError, match="one vector per factor"):
        expansion.trace_exp(.1, coefficients=np.array(.2))

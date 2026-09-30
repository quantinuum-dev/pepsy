"""General graph PEPO construction keeps backend tensors and zero derivatives."""

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOClusterFactor, MPOParameter, MPOProductTerm,
    PEPOClusterProductExpansion,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)


def graph_builder(**kwargs):
    factors = [
        MPOClusterFactor([
            MPOProductTerm((0, 2), (X, Y), MPOParameter("a")),
            MPOProductTerm((3,), (Z,), .13),
        ]),
        MPOClusterFactor([
            MPOProductTerm((0, 1, 2), (Y, Z, X), MPOParameter("b")),
        ], MPOParameter("scale", default=.8)),
    ]
    plan = ClusterPlan.from_terms([term for factor in factors for term in factor.terms],
                                 sites=4, cluster_size=3)
    return PEPOClusterProductExpansion.from_plan(plan, factors, **kwargs)


def generators():
    return (np.kron(np.kron(np.kron(X, I), Y), I),
            np.kron(np.kron(np.kron(I, I), I), Z),
            np.kron(np.kron(np.kron(Y, Z), X), I))


@pytest.mark.parametrize("dtype,policy", [("float32", "fixed"), ("float64", "fixed"),
                                         ("float64", "auto")])
@pytest.mark.parametrize("phase", [-1j, 1.])
def test_graph_torch_materialization_values_gradients_and_zeros(dtype, policy, phase):
    torch = pytest.importorskip("torch")
    builder = graph_builder(factorization=policy)
    tolerance = 3e-6 if dtype == "float32" else 3e-12
    # The declared NumPy operator matrices are float64/complex128, so the
    # existing graph evaluator promotes even float32 coefficient inputs.
    cdtype = torch.complex128
    a, z, b = (torch.as_tensor(g, dtype=cdtype) for g in generators())
    weights = torch.linspace(-.4, .8, 256, dtype=getattr(torch, dtype)).reshape(16, 16)
    for values in ([.2, -.3, .8, .07], [0., 0., .8, .07], [.2, -.3, .8, 0.]):
        theta = torch.tensor(values, dtype=getattr(torch, dtype), requires_grad=True)
        params = dict(zip(("a", "b", "scale"), theta))
        active, report = builder.exp(phase * theta[-1], params, return_report=True)
        actual = active.to_dense()
        assert actual.dtype == cdtype
        assert actual.device == theta.device
        assert report["factorization"] == policy
        expected = (torch.matrix_exp(phase * theta[-1] * (theta[0] * a + .13 * z))
                    @ torch.matrix_exp(phase * theta[-1] * theta[2] * theta[1] * b))
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        ga, = torch.autograd.grad(((actual.real + .37 * actual.imag) * weights).sum(), theta)
        ge, = torch.autograd.grad(((expected.real + .37 * expected.imag) * weights).sum(), theta)
        torch.testing.assert_close(ga, ge, atol=10*tolerance, rtol=10*tolerance)


def test_fixed_graph_jax_jit_complete_materialization_and_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jsl = pytest.importorskip("jax.scipy.linalg")
    with jax.enable_x64(True):
        builder = graph_builder(factorization="fixed")
        a, z, b = map(jnp.asarray, generators())
        weights = jnp.linspace(-.4, .8, 256).reshape(16, 16)

        def objective(theta):
            network = builder.exp(-1j * theta[-1], dict(zip(("a", "b", "scale"), theta)),
                                  materialize=True)
            matrix = network.to_dense([("graph-bra", i) for i in range(4)],
                                      [("graph-ket", i) for i in range(4)])
            return jnp.sum((matrix.real + .37 * matrix.imag) * weights)

        def reference(theta):
            matrix = (jsl.expm(-1j * theta[-1] * (theta[0] * a + .13 * z))
                      @ jsl.expm(-1j * theta[-1] * theta[2] * theta[1] * b))
            return jnp.sum((matrix.real + .37 * matrix.imag) * weights)

        actual = jax.jit(jax.value_and_grad(objective))
        expected = jax.value_and_grad(reference)
        for values in ([.2, -.3, .8, .07], [0., 0., .8, .07], [.2, -.3, .8, 0.]):
            for result, correct in zip(actual(jnp.asarray(values)), expected(jnp.asarray(values))):
                np.testing.assert_allclose(result, correct, atol=3e-11, rtol=3e-11)


def test_fixed_graph_rank_cap_is_rejected():
    with pytest.raises(ValueError, match="max_tree_rank=None"):
        graph_builder(factorization="fixed", max_tree_rank=1)


@pytest.mark.parametrize("rank", [None, 1])
def test_auto_graph_torch_matches_numpy_materialization(rank):
    torch = pytest.importorskip("torch")
    builder = graph_builder(max_tree_rank=rank)
    params = {"a": .2, "b": -.3, "scale": .8}
    expected = builder.exp(-.07j, params).to_dense()
    actual = builder.exp(-.07j, {name: torch.tensor(value, dtype=torch.float64)
                               for name, value in params.items()}).to_dense()
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-12)


def test_partial_cutoff_graph_materialization_matches_mpo_values_and_gradients():
    torch = pytest.importorskip("torch")
    from pepsy.operators import MPOClusterProductExpansion

    factors = [MPOClusterFactor([
        MPOProductTerm((0, 2), (X, Y), MPOParameter("a")),
        MPOProductTerm((1,), (Z,), .13),
    ]), MPOClusterFactor([
        MPOProductTerm((0, 1), (Y, Z), MPOParameter("b")),
        MPOProductTerm((1, 2), (Z, X), -.2),
    ])]
    plan = ClusterPlan.from_terms([term for factor in factors for term in factor.terms],
                                 sites=3, cluster_size=2)
    pepo = PEPOClusterProductExpansion.from_plan(plan, factors, factorization="fixed")
    mpo = MPOClusterProductExpansion.from_plan(
        plan, factors, factorization="fixed", graph_assembly="exact", cutoff=0,
    )
    for values in ([.2, -.3], [0., 0.]):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        params = dict(zip(("a", "b"), theta))
        actual = pepo.exp(-.07j, params).to_dense()
        expected = mpo.exp(-.07j, parameters=params, materialize=True).to_dense()
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        torch.testing.assert_close(torch.trace(actual), pepo.trace_exp(-.07j, params),
                                   atol=3e-12, rtol=3e-12)
        ga, = torch.autograd.grad((actual.real + .37 * actual.imag).sum(), theta)
        ge, = torch.autograd.grad((expected.real + .37 * expected.imag).sum(), theta)
        torch.testing.assert_close(ga, ge, atol=3e-11, rtol=3e-11)

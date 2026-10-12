"""Individual ordered factors reuse only proven bindings, with fresh gradients."""

from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, GraphPEPOClusterProductExpansion, MPOClusterFactor,
    MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    PauliPEPOBasis, PauliPEPOTerm,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
PAULI = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
         "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}


def embedded(word, sites, nsites):
    labels = dict(zip(sites, word))
    matrix = np.ones((1, 1), dtype=complex)
    for site in range(nsites):
        matrix = np.kron(matrix, PAULI[labels.get(site, "I")])
    return matrix


def builder_and_terms(kind, reuse):
    nsites = 4 if kind == "square" else 3
    edges = ((0, 1), (0, 2), (1, 3), (2, 3)) if kind == "square" else ((0, 1), (1, 2))
    terms = []
    for index, (onsite, bond) in enumerate((("X", "ZZ"), ("Z", "XY"), ("Y", "XX"))):
        terms.append([(onsite, (i,), f"h{index}:{i}" if index == 1 else f"h{index}")
                      for i in range(nsites)]
                     + [(bond, edge, f"J{index}:{i}" if index == 1 else f"J{index}")
                        for i, edge in enumerate(edges)])
    scales = (.7, -.4, MPOParameter("scale", default=1.2))
    if kind == "square":
        bases = [PauliPEPOBasis(
            2, 2, [PauliPEPOTerm(
                "onsite" if len(sites) == 1 else "edge", word, MPOParameter(name),
                where=divmod(sites[0], 2) if len(sites) == 1
                else tuple(divmod(site, 2) for site in sites),
            ) for word, sites, name in factor], order=4, factorization="fixed",
            spatial_reuse=reuse,
        ) for factor in terms]
        builder = PEPOClusterProductExpansion.from_bases(bases, coefficients=scales)
    else:
        plan = ClusterPlan.from_supports(nsites, edges, cluster_size=nsites)
        factors = [MPOClusterFactor([
            MPOProductTerm.from_pauli(sites, word, coefficient=MPOParameter(name))
            for word, sites, name in factor
        ], scale) for factor, scale in zip(terms, scales)]
        builder = GraphPEPOClusterProductExpansion.from_plan(
            plan, factors, factorization="fixed", spatial_reuse=reuse,
        )
    return builder, terms, nsites


def evaluate_counted(builder, kind, step, *, trace=False, **bindings):
    from pepsy.operators import mpo_product, pepo_basis

    module, name = ((pepo_basis, "_backend_expm") if kind == "square"
                    else (mpo_product, "_matrix_exponential"))
    original = getattr(module, name)
    counts = []

    def count(matrix):
        counts.append(matrix.shape[0] if matrix.ndim == 3 else 1)
        return original(matrix)

    with patch.object(module, name, count):
        result = builder.trace_exp(step, **bindings) if trace else builder.exp(step, **bindings).to_dense()
    return result, sum(counts)


@pytest.mark.parametrize("kind,expected_counts", [("square", (21, 39)), ("graph", (12, 18))])
def test_partial_joint_symmetry_reuses_individual_factors_and_live_gradients(kind, expected_counts):
    torch = pytest.importorskip("torch")
    reused, terms, nsites = builder_and_terms(kind, True)
    direct, _, _ = builder_and_terms(kind, False)
    names = tuple(dict.fromkeys(name for factor in terms for _, _, name in factor)) + ("scale", "step")
    weights = torch.linspace(-.4, .8, 4**nsites, dtype=torch.float64).reshape(2**nsites, -1)
    for offset in (.1, -.2):
        theta = torch.linspace(-.3 + offset, .5 + offset, len(names), dtype=torch.float64,
                               requires_grad=True)
        params = dict(zip(names, theta))
        actual, count = evaluate_counted(reused, kind, -.07j * params["step"], parameters=params)
        baseline, direct_count = evaluate_counted(direct, kind, -.07j * params["step"], parameters=params)
        assert (count, direct_count) == expected_counts
        trace, trace_count = evaluate_counted(reused, kind, -.07j * params["step"],
                                              parameters=params, trace=True)
        assert trace_count == count
        torch.testing.assert_close(trace, torch.trace(actual), atol=3e-12, rtol=3e-12)
        expected = torch.eye(2**nsites, dtype=torch.complex128)
        for factor, scale in zip(terms, (.7, -.4, params["scale"])):
            generator = sum(params[name] * torch.as_tensor(embedded(word, sites, nsites))
                            for word, sites, name in factor)
            expected = expected @ torch.matrix_exp(-.07j * params["step"] * scale * generator)
        for value in (actual, baseline):
            torch.testing.assert_close(value, expected, atol=3e-12, rtol=3e-12)
        gradients = [torch.autograd.grad(((value.real + .37 * value.imag) * weights).sum(),
                                        theta, retain_graph=True)[0]
                     for value in (actual, baseline, expected)]
        for gradient in gradients[:2]:
            torch.testing.assert_close(gradient, gradients[2], atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("kind", ["square", "graph"])
def test_equal_independent_vectors_do_not_gain_factor_symmetry(kind):
    torch = pytest.importorskip("torch")
    reused, terms, nsites = builder_and_terms(kind, True)
    direct, _, _ = builder_and_terms(kind, False)
    count = sum(map(len, terms))
    weights = torch.linspace(-.4, .8, 4**nsites, dtype=torch.float64).reshape(2**nsites, -1)
    for value in (0., .2):
        theta = torch.full((count,), value, dtype=torch.float64, requires_grad=True)
        vectors = tuple(theta[i:i + len(terms[0])] for i in range(0, count, len(terms[0])))
        # Explicit vectors override term bindings; the scale retains its default.
        actual, count_a = evaluate_counted(reused, kind, -.07j, coefficients=vectors)
        expected, count_b = evaluate_counted(direct, kind, -.07j, coefficients=vectors)
        assert count_a == count_b
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        ga, = torch.autograd.grad(((actual.real + .37 * actual.imag) * weights).sum(), theta)
        ge, = torch.autograd.grad(((expected.real + .37 * expected.imag) * weights).sum(), theta)
        torch.testing.assert_close(ga, ge, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("kind", ["square", "graph"])
def test_factor_reuse_under_jax_jit(kind):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jsl = pytest.importorskip("jax.scipy.linalg")
    with jax.enable_x64(True):
        builder, terms, nsites = builder_and_terms(kind, True)
        names = tuple(dict.fromkeys(name for factor in terms for _, _, name in factor)) + ("scale", "step")
        weights = jnp.linspace(-.4, .8, 4**nsites).reshape(2**nsites, -1)

        def objective(theta):
            params = dict(zip(names, theta))
            matrix = builder.exp(-.07j * params["step"], parameters=params).to_dense()
            return jnp.sum((matrix.real + .37 * matrix.imag) * weights)

        def reference(theta):
            params = dict(zip(names, theta))
            result = jnp.eye(2**nsites, dtype=complex)
            for factor, scale in zip(terms, (.7, -.4, params["scale"])):
                generator = sum(params[name] * jnp.asarray(embedded(word, sites, nsites))
                                for word, sites, name in factor)
                result = result @ jsl.expm(-.07j * params["step"] * scale * generator)
            return jnp.sum((result.real + .37 * result.imag) * weights)

        actual = jax.jit(jax.value_and_grad(objective))
        expected = jax.value_and_grad(reference)
        for offset in (.1, -.2):
            theta = jnp.linspace(-.3 + offset, .5 + offset, len(names))
            for value, correct in zip(actual(theta), expected(theta)):
                np.testing.assert_allclose(value, correct, atol=3e-11, rtol=3e-11)


def test_graph_opaque_callback_evaluation_counts_are_preserved():
    counts = []
    outputs = []
    for reuse in (True, False):
        calls = []

        def callback(params):
            calls.append(1)
            return .1 + .001 * len(calls)

        plan = ClusterPlan.from_supports(3, [(0, 1), (1, 2)], cluster_size=3)
        factors = [MPOClusterFactor([
            MPOProductTerm.from_pauli((i,), "X", coefficient=callback) for i in range(3)
        ], callback), MPOClusterFactor([
            MPOProductTerm.from_pauli((i,), "Z", coefficient=.2) for i in range(3)
        ])]
        builder = GraphPEPOClusterProductExpansion.from_plan(plan, factors, spatial_reuse=reuse)
        outputs.append(builder.exp(-.07j).to_dense())
        counts.append(len(calls))
    assert counts[0] == counts[1]
    np.testing.assert_allclose(*outputs, atol=3e-12)


def test_pooled_onsite_exponentials_keep_double_precision_and_gradients():
    torch = pytest.importorskip("torch")
    from pepsy.operators import pepo_basis

    labels = ("X", "Y", "Z") * 3
    bases = [PauliPEPOBasis(1, 1, [PauliPEPOTerm("onsite", word, MPOParameter(str(i)),
                                              where=(0, 0))], order=1, factorization="fixed")
             for i, word in enumerate(labels)]
    builder = PEPOClusterProductExpansion.from_bases(bases)
    theta = torch.linspace(.022, .038, len(labels), dtype=torch.float64, requires_grad=True)
    original = pepo_basis._backend_expm
    sizes = []

    def count(matrix):
        sizes.append(matrix.shape[0])
        return original(matrix)

    with patch.object(pepo_basis, "_backend_expm", count):
        actual = builder.exp(-1j, parameters=dict(zip(map(str, range(len(labels))), theta))).to_dense()
    assert sizes == [7, 2]  # No redundant exponentials or singleton tail.
    identity = torch.eye(2, dtype=torch.complex128)
    expected = identity
    for angle, word in zip(theta, labels):
        expected = expected @ (torch.cos(angle) * identity
                               - 1j * torch.sin(angle) * torch.as_tensor(PAULI[word]))
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    ga, = torch.autograd.grad((actual.real + .37 * actual.imag).sum(), theta)
    ge, = torch.autograd.grad((expected.real + .37 * expected.imag).sum(), theta)
    torch.testing.assert_close(ga, ge, atol=2e-14, rtol=2e-14)


def test_uniform_factor_reuse_aligns_constant_and_live_backends():
    torch = pytest.importorskip("torch")
    matrices = []
    for reuse in (True, False):
        bases = [PauliPEPOBasis(1, 2, [("onsite", word, value), ("edge", "ZZ", .4)],
                               order=2, factorization="fixed", spatial_reuse=reuse)
                 for word, value in (("X", .2), ("Y", MPOParameter("h")))]
        builder = PEPOClusterProductExpansion.from_bases(bases)
        h = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        matrix = builder.exp(-.07j, parameters={"h": h}).to_dense()
        gradient, = torch.autograd.grad((matrix.real + .37 * matrix.imag).sum(), h)
        matrices.append((matrix, gradient))
    for actual, expected in zip(*matrices):
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)

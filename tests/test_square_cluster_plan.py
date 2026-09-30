"""Shared square plans preserve graph expansions, physical order and bindings."""

from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ActivePEPOBlocks,
    ClusterLattice,
    ClusterPlan,
    GraphPEPOClusterProductExpansion,
    MPOClusterFactor,
    MPOParameter,
    MPOProductTerm,
    PEPOClusterProductExpansion,
    PauliPEPOBasis,
    SquarePEPOClusterProductExpansion,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
I = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.diag([1.0, -1.0])


def plan_for(shape=(2, 2), cutoff=3, cyclic=False):
    return ClusterPlan(
        ClusterLattice.square(*shape, cyclic=cyclic), cluster_size=cutoff, cyclic=cyclic
    )


def embed(term, nsites):
    local = dict(zip(term.sites, term.operators))
    result = np.ones((1, 1))
    for site in range(nsites):
        result = np.kron(result, local.get(site, I))
    return result


def ordered_factors(plan):
    terms = [MPOProductTerm(edge, (X, Y), 0.17) for edge in plan.index_edges]
    # Preserve repeated and reversed asymmetric terms on small periodic tori.
    edge = plan.index_edges[0]
    terms += [MPOProductTerm(edge[::-1], (X, Y), -0.08), MPOProductTerm(edge, (X, Y), 0.06)]
    return (
        MPOClusterFactor(terms, 0.7),
        MPOClusterFactor([MPOProductTerm((i,), (Z,), 0.11) for i in range(4)], -0.4),
        MPOClusterFactor([MPOProductTerm((i,), (X,), 0.13) for i in range(4)], 1.2),
    )


@pytest.mark.parametrize("cyclic", [False, (False, True), True])
@pytest.mark.parametrize("cutoff", [2, 3, 4])
def test_square_dispatch_matches_graph_and_dense_ordered_product(cyclic, cutoff):
    import quimb.tensor as qtn

    plan = plan_for(cutoff=cutoff, cyclic=cyclic)
    factors = ordered_factors(plan)
    square = PEPOClusterProductExpansion.from_plan(plan, (factor for factor in factors))
    graph = PEPOClusterProductExpansion.from_plan(plan, factors, layout="graph")
    assert isinstance(square, SquarePEPOClusterProductExpansion)
    assert isinstance(graph, GraphPEPOClusterProductExpansion)
    assert square.cluster_plan is plan
    assert all(factor.basis.cluster_plan is plan for factor in square._square.factors)
    assert square.compile_exp() is square
    for step in (0.09, -0.08j):
        active, report = square.exp(step, return_report=True)
        assert isinstance(active, ActivePEPOBlocks)
        pepo = active.to_pepo()
        assert isinstance(pepo, qtn.PEPO)
        assert (pepo.Lx, pepo.Ly) == plan.shape
        assert active.cyclic == plan.cyclic
        assert report["cluster_counts"] == plan.counts
        actual = active.to_dense()
        np.testing.assert_allclose(actual, graph.exp(step).to_dense(), atol=3e-12)
        np.testing.assert_allclose(square.trace_exp(step), np.trace(actual), atol=3e-12)
        if cutoff == 4:
            target = np.eye(16, dtype=complex)
            for factor in factors:
                target = target @ expm(
                    step
                    * factor.coefficient
                    * sum(term.coefficient * embed(term, 4) for term in factor.terms)
                )
            np.testing.assert_allclose(actual, target, atol=3e-12)


def test_pauli_factory_shared_bindings_and_runtime_vectors():
    plan = plan_for()
    h = MPOParameter("h")
    terms = [MPOProductTerm((i,), (X,), h) for i in range(4)]
    terms += [MPOProductTerm(edge, (Z, Z), 0.2) for edge in plan.index_edges]
    basis = PauliPEPOBasis.from_plan(plan, terms)
    assert isinstance(basis, SquarePEPOClusterProductExpansion)
    assert basis.factors[0].terms[0].coefficient is h
    assert basis._square.factors[0].basis.terms[0].coefficient is h
    for value in (0.3, -0.2):
        bound = {"h": value}
        a = basis.exp(0.1, bound, materialize=True).to_dense()
        reference = PEPOClusterProductExpansion.from_plan(plan, [terms], layout="graph")
        np.testing.assert_allclose(a, reference.exp(0.1, bound).to_dense(), atol=1e-12)
    cache = basis.cache_info["factor_cache_info"][0]
    assert cache["last_local_targets_evaluated"] < sum(plan.counts.values())
    assert not basis.cache_info["local_residuals_compiled"]
    values = [0.3, 0.3, -0.2, 0.4] + [0.2] * len(plan.index_edges)
    a = basis.exp(0.1, coefficients=values).to_dense()
    np.testing.assert_allclose(a, reference.exp(0.1, coefficients=values).to_dense(), atol=1e-12)
    for cluster, residual in basis.residuals(0.1, coefficients=values).items():
        np.testing.assert_allclose(
            residual, reference.residuals(0.1, coefficients=values)[cluster], atol=1e-12
        )
    assert basis.cache_info["local_residuals_compiled"]


@pytest.mark.parametrize(
    "reason", ["diagonal", "missing_edge", "one_dimensional", "matrix", "permuted"]
)
def test_unsupported_square_plans_keep_graph_path(reason):
    plan = plan_for(cutoff=2)
    terms = [MPOProductTerm((0, 1), (X, Z), 0.2)]
    if reason in ("diagonal", "missing_edge"):
        edges = list(plan.index_edges)
        if reason == "diagonal":
            edges.append((0, 3))
            terms.append(MPOProductTerm((0, 3), (X, Z), 0.1))
        else:
            edges.pop()
        plan = ClusterPlan.from_supports(4, edges, shape=(2, 2), cluster_size=2)
    elif reason == "one_dimensional":
        plan = ClusterPlan.from_supports(4, [(0, 1), (1, 2), (2, 3)], cluster_size=2)
    elif reason == "matrix":
        terms = [MPOProductTerm((0,), (0.7 * X + 0.2 * Z,), 0.2)]
    else:
        sites = tuple(reversed(plan.sites))
        plan = ClusterPlan(ClusterLattice(sites, plan.lattice.edges), cluster_size=2)
    assert isinstance(
        PEPOClusterProductExpansion.from_plan(plan, [terms]), GraphPEPOClusterProductExpansion
    )
    if reason in {"one_dimensional", "permuted"}:
        with pytest.raises(ValueError, match="square"):
            PEPOClusterProductExpansion.from_plan(plan, [terms], layout="square")
    else:
        routed = PEPOClusterProductExpansion.from_plan(plan, [terms], layout="square")
        graph = PEPOClusterProductExpansion.from_plan(plan, [terms], layout="graph")
        assert routed.cluster_plan is plan
        np.testing.assert_allclose(routed.exp(-.03j, materialize=True).to_dense(),
                                   graph.exp(-.03j).to_dense(), atol=2e-13)


def test_arbitrary_labels_explicit_shape_and_plan_validation():
    plan = ClusterPlan.from_supports(
        "abcd", [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")], shape=(2, 2), cluster_size=2
    )
    basis = PauliPEPOBasis.from_plan(plan, [((0, 1), "XY", 0.2)], layout="square")
    graph = PauliPEPOBasis.from_plan(plan, [((0, 1), "XY", 0.2)], layout="graph")
    np.testing.assert_allclose(basis.exp(0.1).to_dense(), graph.exp(0.1).to_dense(), atol=1e-12)
    with pytest.raises(ValueError, match="match cluster_plan"):
        PauliPEPOBasis(2, 2, [("onsite", "X")], cluster_size=3, cluster_plan=plan)
    with pytest.raises(ValueError, match="layout"):
        PEPOClusterProductExpansion.from_plan(plan, [[((0,), "X")]], layout="invalid")


@pytest.mark.parametrize("cyclic", [False, (False, True)])
def test_square_fixed_torch_materialization_and_independent_gradients(cyclic):
    torch = pytest.importorskip("torch")
    plan = plan_for((1, 2), 2, cyclic)
    terms = [MPOProductTerm((i,), (X,), MPOParameter("h")) for i in range(2)]
    terms += [MPOProductTerm((0, 1), (Y, Z), 0.2)]
    basis = PauliPEPOBasis.from_plan(plan, terms, factorization="fixed").compile_exp()
    for values in ((0.2, 0.2, 0.3), (0.1, -0.2, 0.4)):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        step = torch.tensor(0.07, dtype=torch.float64, requires_grad=True)
        actual = basis.exp(step, coefficients=theta, materialize=True).to_dense()
        hamiltonian = sum(
            theta[i] * torch.tensor(embed(term, 2), dtype=torch.complex128)
            for i, term in enumerate(terms)
        )
        expected = torch.matrix_exp(step * hamiltonian)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for a, b in zip(
            torch.autograd.grad(actual.real.sum(), (theta, step)),
            torch.autograd.grad(expected.real.sum(), (theta, step)),
        ):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_square_fixed_jax_jit_shared_parameter_and_step_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64(True):
        plan = plan_for((1, 2), 2)
        factors = [
            MPOClusterFactor([MPOProductTerm((i,), (X,), MPOParameter("h")) for i in range(2)]),
            MPOClusterFactor([MPOProductTerm((0, 1), (Z, Z), 0.3)], -0.7),
        ]
        basis = PEPOClusterProductExpansion.from_plan(
            plan, factors, factorization="fixed"
        ).compile_exp()

        def objective(h, step):
            matrix = basis.exp(step, {"h": h}, materialize=True).to_dense()
            return jnp.real(matrix[0, 0] + matrix[0, 1])

        def reference(h, step):
            matrix = jax.scipy.linalg.expm(step * h * jnp.asarray(np.kron(X, I) + np.kron(I, X)))
            matrix = matrix @ jax.scipy.linalg.expm(-0.7 * step * 0.3 * jnp.asarray(np.kron(Z, Z)))
            return jnp.real(matrix[0, 0] + matrix[0, 1])

        compiled = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
        direct = jax.value_and_grad(reference, argnums=(0, 1))
        for h, step in ((0.2, 0.1), (0.0, 0.1), (0.3, 0.0)):
            a, ga = compiled(jnp.asarray(h), jnp.asarray(step))
            b, gb = direct(jnp.asarray(h), jnp.asarray(step))
            np.testing.assert_allclose(a, b, atol=2e-12)
            np.testing.assert_allclose(ga, gb, atol=2e-12)


def test_layout_does_not_infer_wraparound_bonds():
    # Marking an open 3x2 graph periodic does not add the missing seam edges.
    coordinates = tuple(product(range(3), range(2)))
    obc = ClusterLattice.square(3, 2)
    plan = ClusterPlan(ClusterLattice(coordinates, obc.edges), cluster_size=2, cyclic=(True, False))
    factors = [[MPOProductTerm((0,), (X,), 0.2)]]
    assert isinstance(
        PEPOClusterProductExpansion.from_plan(plan, factors), GraphPEPOClusterProductExpansion
    )
    routed = PEPOClusterProductExpansion.from_plan(plan, factors, layout="square")
    assert routed.cluster_plan is plan
    assert routed.cluster_plan.index_edges == plan.index_edges
    graph = PEPOClusterProductExpansion.from_plan(plan, factors, layout="graph")
    np.testing.assert_allclose(routed.exp(-.03j, materialize=True).to_dense(),
                               graph.exp(-.03j).to_dense(), atol=2e-13)


def test_three_site_periodic_axis_routes_the_seam_with_correct_orientation():
    plan = plan_for((3, 2), 2, (True, False))
    terms = [
        MPOProductTerm((0, 4), (X, Y), 0.2),
        MPOProductTerm((4, 0), (X, Z), -0.13),
        MPOProductTerm((4,), (Z,), 0.1),
    ]
    square = PauliPEPOBasis.from_plan(plan, terms)
    graph = PauliPEPOBasis.from_plan(plan, terms, layout="graph")
    hamiltonian = sum(term.coefficient * embed(term, 6) for term in terms)
    for step in (0.1, -0.07j):
        expected = expm(step * hamiltonian)
        actual = square.exp(step).to_dense()
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        np.testing.assert_allclose(actual, graph.exp(step).to_dense(), atol=2e-12)


def test_partial_factor_vectors_and_binding_conflicts_keep_square_contract():
    plan = plan_for((1, 2), 2)
    calls = []

    def callback(params):
        calls.append(params)
        return params["h"]

    factors = [
        MPOClusterFactor([MPOProductTerm((0,), (X,), callback)]),
        MPOClusterFactor([MPOProductTerm((0, 1), (Z, Y), 0.2)]),
    ]
    basis = PEPOClusterProductExpansion.from_plan(plan, factors)
    for method in (basis.exp, basis.trace_exp):
        with pytest.raises(ValueError, match="mutually exclusive"):
            method(0.1, {"h": 0.3}, coefficients=[None, [0.4]])
    assert not calls
    factors = [MPOClusterFactor([MPOProductTerm((0,), (X,), 0.3)], 0.7), factors[1]]
    basis = PEPOClusterProductExpansion.from_plan(plan, factors)
    for coefficient in (0.4, -0.1):
        actual = basis.exp(0.1, coefficients=[None, [coefficient]]).to_dense()
        expected = expm(0.1 * 0.7 * 0.3 * np.kron(X, I)) @ expm(0.1 * coefficient * np.kron(Z, Y))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

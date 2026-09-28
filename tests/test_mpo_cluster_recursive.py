"""Complete collection assembly through shared MPO subproblems."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice,
    MPOBasis,
    MPOClusterFactor,
    MPOGraphClusterBasisExpansion,
    MPOParameter,
    exp_mpo_cluster,
    exp_mpo_cluster_product,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators, pytest.mark.mpo]

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
I = np.eye(2)


def kron(values):
    result = values[0]
    for value in values[1:]:
        result = np.kron(result, value)
    return result


@pytest.mark.parametrize("edges", [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))])
@pytest.mark.parametrize("assembly", ["recursive", "streaming"])
def test_disjoint_products_match_independent_exponentials(edges, assembly):
    """Separated, crossing and nested spans retain the quadratic residual term."""
    graph = ClusterLattice.from_edges(range(4), edges)
    terms = [(edge, (X, X), 0.7) for edge in edges]
    terms += [((site,), (Z,), 0.2) for site in range(4)]
    h = sum(0.7 * kron([X if i in edge else I for i in range(4)]) for edge in edges)
    h += sum(0.2 * kron([Z if i == site else I for i in range(4)]) for site in range(4))
    result, report = exp_mpo_cluster(
        terms,
        0.3,
        shape=4,
        graph=graph,
        cluster_size=2,
        cutoff=0.0,
        graph_assembly="exact",
        assembly=assembly,
        assembly_chi=64,
        assembly_batch_size=1,
        return_report=True,
    )
    np.testing.assert_allclose(result.to_dense(), expm(0.3 * h), atol=3e-12)
    assert not report.graph_collection_truncated
    assert report.assembly_compression_count > 0


@pytest.mark.parametrize("p,collection_count", [(2, 6), (4, 11)])
@pytest.mark.parametrize(
    "chi,cutoff,form", [(None, None, "left"), (64, None, "left"), (64, 0.0, "right")]
)
def test_recursive_matches_complete_enumeration_without_collection_budget(
    monkeypatch, chi, cutoff, form, p, collection_count
):
    """A cyclic interacting model retains every collection without enumeration."""
    graph = ClusterLattice.from_edges(range(4), ((0, 1), (1, 2), (2, 3), (3, 0)))
    terms = [(edge, (X, X), 0.4) for edge in graph.edges]
    terms += [((i,), (Z,), 0.2 + i / 20) for i in range(4)]
    expected, direct_report = exp_mpo_cluster(
        terms,
        0.12,
        shape=4,
        graph=graph,
        cluster_size=p,
        cutoff=0.0,
        graph_assembly="exact",
        collection_budget=100,
        return_report=True,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("recursive assembly must not enumerate compatible collections")

    from pepsy.operators.mpo_product import MPOClusterProductExpansion

    monkeypatch.setattr(MPOClusterProductExpansion, "_bounded_graph_cluster_collections", forbidden)
    result, report = exp_mpo_cluster(
        terms,
        0.12,
        shape=4,
        graph=graph,
        cluster_size=p,
        cutoff=0.0,
        assembly="recursive",
        assembly_chi=chi,
        assembly_cutoff=cutoff,
        assembly_form=form,
        collection_budget=1,
        return_report=True,
    )
    np.testing.assert_allclose(result.to_dense(), expected.to_dense(), atol=3e-12)
    assert report.graph_collection_count == direct_report.graph_collection_count == collection_count
    assert report.graph_collection_budget is None
    assert report.graph_planner == "subset_dp"
    assert not report.graph_collection_truncated
    assert report.assembly_peak_cached_states > 0
    if chi is None:
        assert report.assembly_compression_count == 0
        assert not report.assembly_truncated
    else:
        assert report.assembly_compression_count > 0
    if cutoff is not None:
        assert report.assembly_discarded_weights


def test_full_cluster_ordered_product():
    """Ordered local products and inclusion-exclusion survive the new assembly."""
    a = [((0, 1), (X, Z), 0.7), ((1, 2), (Z, X), -0.4)]
    b = [((0,), (Z,), 0.2), ((2,), (X,), 0.3)]
    result = exp_mpo_cluster_product(
        [MPOClusterFactor(a), MPOClusterFactor(b)],
        0.2,
        shape=3,
        graph="chain",
        cluster_size=3,
        cutoff=0.0,
        assembly="recursive",
    )
    ah = 0.7 * kron([X, Z, I]) - 0.4 * kron([I, Z, X])
    bh = 0.2 * kron([Z, I, I]) + 0.3 * kron([I, I, X])
    np.testing.assert_allclose(result.to_dense(), expm(0.2 * ah) @ expm(0.2 * bh), atol=2e-12)


def test_recursive_compression_converges_and_reports_rank_reduction():
    terms = [((0, 1), (X, Z), 0.7), ((1, 2), (X, X), -0.4), ((0, 2), (Z, X), 0.3)]
    graph = ClusterLattice.from_edges(range(3), ((0, 1), (1, 2), (0, 2)))
    options = dict(shape=3, graph=graph, cluster_size=3, cutoff=0.0, assembly="recursive")
    exact = exp_mpo_cluster(terms, 0.3, **options).to_dense()
    low, report = exp_mpo_cluster(terms, 0.3, assembly_chi=1, return_report=True, **options)
    high = exp_mpo_cluster(terms, 0.3, assembly_chi=16, **options)
    assert np.linalg.norm(low.to_dense() - exact) > 1e-4
    np.testing.assert_allclose(high.to_dense(), exact, atol=3e-12)
    assert report.assembly_truncated
    assert report.api_info.truncated
    assert max(report.initial_bond_dimensions) <= 1


def test_planner_reuse_square_snake_and_state_budget():
    terms = [(("X", 0.2), (i, j)) for i in range(2) for j in range(3)]
    terms += [(("ZZ", 0.4), ((0, j), (1, j))) for j in range(3)]
    basis = MPOBasis.from_terms(terms, shape=(2, 3))
    options = dict(graph="square", cluster_size=2, assembly="recursive", assembly_chi=64)
    compiled = basis.compile_graph_cluster_expansion(**options)
    assert compiled is basis.compile_graph_cluster_expansion(**options)
    assert compiled.cache_info["graph_planner"] == "subset_dp"
    assert compiled.cache_info["assembly_subproblem_count"] > 0
    changed = basis.compile_graph_cluster_expansion(**options, assembly_state_budget=128)
    assert changed is not compiled
    result, report = compiled(0.03, return_report=True)
    expected = basis.graph_cluster_expansion(
        0.03, graph="square", cluster_size=2, graph_assembly="exact"
    )
    np.testing.assert_allclose(result.to_mpo().to_dense(), expected.to_mpo().to_dense(), atol=2e-12)
    assert report.graph_collection_count == 21  # all nonempty matchings of a 2x3 grid
    assert compiled.cache_info["builds"] == 1
    with pytest.raises(ValueError, match="assembly_state_budget=1"):
        basis.compile_graph_cluster_expansion(**options, assembly_state_budget=1)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_state_budget_rejected(budget):
    with pytest.raises(ValueError, match="assembly_state_budget"):
        exp_mpo_cluster(
            [((0,), (Z,))],
            0.1,
            shape=1,
            graph="chain",
            assembly="recursive",
            assembly_state_budget=budget,
        )


def test_singleton_and_isolated_sites():
    for length in (1, 3):
        result, report = exp_mpo_cluster(
            [((0,), (Z,), 0.2)],
            0.1,
            shape=length,
            graph=ClusterLattice.from_edges(range(length), ()),
            assembly="recursive",
            cluster_size=1,
            return_report=True,
        )
        np.testing.assert_allclose(result.to_dense(), kron([expm(0.02 * Z)] + [I] * (length - 1)))
        assert report.graph_collection_count == 0
        assert report.assembly_compression_count == 0


def test_recursive_rejects_incompatible_controls():
    base = dict(shape=2, graph="chain", assembly="recursive")
    terms = [((0, 1), (Z, Z))]
    for options, message in [
        ({"graph_assembly": "bounded"}, "retains complete"),
        ({"max_collection_order": 1}, "retains complete"),
        ({"assembly_cutoff": 1e-8}, "requires assembly_chi"),
        ({"symmetry": "U1", "physical_charges": (0, 1)}, "native symmetry"),
        ({"graph": None}, "requires graph"),
    ]:
        with pytest.raises(ValueError, match=message):
            exp_mpo_cluster(terms, 0.1, **(base | options))


@pytest.mark.parametrize("chi", [None, 4])
def test_compiled_recursive_torch_repeated_value_and_gradient(chi):
    torch = pytest.importorskip("torch")
    # Full-rank, distinct local singular values avoid a degenerate SVD derivative.
    local = np.array(
        [[0.2, 0.3, 0.1, 0.0], [0.3, -0.4, 0.2, 0.1], [0.1, 0.2, 0.7, -0.2], [0.0, 0.1, -0.2, -0.1]]
    )
    from pepsy.operators import MPOLocalOperatorTerm

    factor = MPOClusterFactor([MPOLocalOperatorTerm((0, 1), local, MPOParameter("j"))])
    expansion = MPOGraphClusterBasisExpansion.from_factors(
        2,
        [factor],
        graph=ClusterLattice.from_edges(range(2), ((0, 1),)),
        cluster_size=2,
        cutoff=0.0,
        assembly="recursive",
        assembly_chi=chi,
        to_backend=lambda value: torch.as_tensor(value, dtype=torch.float64),
    ).compile_exp()
    for value in (0.4, -0.2):
        j = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        step = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
        result = expansion(step, parameters={"j": j}).to_mpo().to_dense()
        reference = torch.matrix_exp(step * j * torch.as_tensor(local))
        weights = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        actual_grad = torch.autograd.grad((result * weights).sum(), (j, step), retain_graph=True)
        expect_grad = torch.autograd.grad((reference * weights).sum(), (j, step))
        torch.testing.assert_close(result, reference)
        for actual, expected in zip(actual_grad, expect_grad):
            torch.testing.assert_close(actual, expected)
    assert expansion.cache_info["builds"] == 2


@pytest.mark.parametrize("assembly", ["recursive", "streaming"])
@pytest.mark.parametrize("form", ["left", "right"])
def test_compression_uses_an_orthonormal_environment(assembly, form):
    terms = [(("X", 0.2), (i, j)) for i in range(2) for j in range(3)]
    terms += [(("ZZ", 0.4), ((0, j), (1, j))) for j in range(3)]
    options = dict(shape=(2, 3), graph="square", cluster_size=2, graph_assembly="exact")
    exact = exp_mpo_cluster(terms, 0.03, **options).to_dense()
    compressed = exp_mpo_cluster(
        terms,
        0.03,
        assembly=assembly,
        assembly_chi=8,
        assembly_form=form,
        assembly_batch_size=1,
        **options,
    )
    # A raw-core SVD lost O(1) weight on this near-identity operator.
    assert np.linalg.norm(compressed.to_dense() - exact) < 1e-5


def test_numpy_svd_nonconvergence_falls_back_to_finite_gesvd(monkeypatch):
    import scipy.linalg
    from pepsy.operators import mpo_semantic

    original_do = mpo_semantic.ar.do
    original_svd = scipy.linalg.svd
    drivers = []

    def fail_default(name, *args, **kwargs):
        if name == "linalg.svd":
            raise np.linalg.LinAlgError("simulated divide-and-conquer failure")
        return original_do(name, *args, **kwargs)

    def record_driver(*args, **kwargs):
        drivers.append(kwargs.get("lapack_driver"))
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(mpo_semantic.ar, "do", fail_default)
    monkeypatch.setattr(scipy.linalg, "svd", record_driver)
    matrix = np.arange(30.0).reshape(6, 5) + 1j * np.eye(6, 5)
    left, singular_values, right = mpo_semantic._fixed_rank_svd(matrix)
    np.testing.assert_allclose(
        (left * singular_values) @ right, matrix, atol=1e-13
    )
    assert drivers == ["gesvd"]
    with pytest.raises(np.linalg.LinAlgError, match="simulated"):
        mpo_semantic._fixed_rank_svd(np.array([[np.nan]]))
    assert drivers == ["gesvd"]
